"""PyTorch 模型的公共工具：批处理、掩码、训练循环。

统一约定：每条序列是 (kc_seq, resp_seq)，长度 T。
位置 t 的输入含 kc_t 与 resp_{t-1}（resp_{-1} 视为 0），输出预测 resp_t。
不足长度的序列用 pad_kc 填充，用 mask 屏蔽，不参与损失。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ml.config import get_profile


@dataclass
class Batch:
    kc: torch.Tensor       # (B, L) long
    resp: torch.Tensor     # (B, L) long, 0/1，填充位为 0
    mask: torch.Tensor     # (B, L) bool，True 表示是真实交互


def pad_collate(sequences: list[dict], pad_kc: int, max_len: int | None = None) -> Batch:
    lengths = [min(len(s["resp_seq"]), max_len) if max_len else len(s["resp_seq"]) for s in sequences]
    L = max(lengths) if lengths else 1
    B = len(sequences)
    kc = np.full((B, L), pad_kc, dtype=np.int64)
    resp = np.zeros((B, L), dtype=np.int64)
    mask = np.zeros((B, L), dtype=bool)
    for i, s in enumerate(sequences):
        n = lengths[i]
        kc[i, :n] = s["kc_seq"][:n]
        resp[i, :n] = s["resp_seq"][:n]
        mask[i, :n] = True
    return Batch(kc=torch.from_numpy(kc), resp=torch.from_numpy(resp), mask=torch.from_numpy(mask))


def iterate_batches(
    sequences: list[dict],
    batch_size: int,
    pad_kc: int,
    max_len: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
):
    idx = np.arange(len(sequences))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, len(idx), batch_size):
        chunk = [sequences[i] for i in idx[start : start + batch_size]]
        yield pad_collate(chunk, pad_kc=pad_kc, max_len=max_len)


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """只对真实位置计算二元交叉熵。"""
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    loss = loss * mask.float()
    denom = mask.float().sum().clamp(min=1.0)
    return loss.sum() / denom


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    sequences: list[dict],
    pad_kc: int,
    device: str,
    batch_size: int = 256,
    max_len: int | None = None,
) -> list[np.ndarray]:
    """逐序列返回长度 T 的答对概率。"""
    model.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        batch = pad_collate(chunk, pad_kc=pad_kc, max_len=max_len)
        logits = model(
            batch.kc.to(device),
            batch.resp.to(device),
        )
        probs = torch.sigmoid(logits).cpu().numpy()
        for i, s in enumerate(chunk):
            n = min(len(s["resp_seq"]), max_len) if max_len else len(s["resp_seq"])
            out.append(probs[i, :n])
    return out


def train_loop(
    model: nn.Module,
    train_seqs: list[dict],
    valid_seqs: list[dict],
    pad_kc: int,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_len: int | None = None,
    weight_decay: float = 1e-5,
    seed: int = 42,
    verbose: bool = True,
    patience: int = 20,
) -> dict:
    """通用训练循环，早停按验证集损失。返回训练历史。

    **模型选择用验证集 AUC，不用验证集 loss**。这一条是踩过坑才改的：
    答题数据本身噪声很大（失误/猜测），验证 loss 往往在第 2 个 epoch 就平台化，
    而模型其实还在持续改善排序能力。若按 loss 选，会回滚到一个几乎没训起来的 checkpoint，
    线上表现就是 AUC ≈ 0.5（等于没学到）。AUC 只看排序，不受整体概率标定影响，
    是知识追踪里更可靠的选型依据。
    """
    from sklearn.metrics import roc_auc_score
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best_auc = -1.0
    best_valid = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0
    history = {"train_loss": [], "valid_loss": [], "valid_auc": []}

    for ep in range(1, epochs + 1):
        model.train()
        total, n_batch = 0.0, 0
        for batch in iterate_batches(
            train_seqs, batch_size, pad_kc, max_len=max_len, shuffle=True, seed=seed + ep
        ):
            kc = batch.kc.to(device)
            resp = batch.resp.to(device)
            mask = batch.mask.to(device)
            logits = model(kc, resp)
            loss = masked_bce(logits, resp, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item())
            n_batch += 1
        sched.step()

        # 验证：同时算 loss 与 AUC，选型看 AUC
        model.eval()
        vtotal, vn = 0.0, 0
        vy, vp = [], []
        with torch.no_grad():
            for batch in iterate_batches(
                valid_seqs, batch_size, pad_kc, max_len=max_len, shuffle=False
            ):
                kc_v = batch.kc.to(device)
                resp_v = batch.resp.to(device)
                mask_v = batch.mask.to(device)
                logits = model(kc_v, resp_v)
                vtotal += float(masked_bce(logits, resp_v, mask_v).item())
                vn += 1
                probs = torch.sigmoid(logits)[mask_v]
                vy.append(resp_v[mask_v].cpu().numpy())
                vp.append(probs.cpu().numpy())
        train_loss = total / max(n_batch, 1)
        valid_loss = vtotal / max(vn, 1)
        if vy:
            y_true = np.concatenate(vy)
            y_pred = np.concatenate(vp)
            valid_auc = (
                float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
            )
        else:
            valid_auc = float("nan")

        history["train_loss"].append(round(train_loss, 6))
        history["valid_loss"].append(round(valid_loss, 6))
        history["valid_auc"].append(round(valid_auc, 6) if not np.isnan(valid_auc) else None)

        if verbose:
            auc_str = "  --  " if np.isnan(valid_auc) else f"{valid_auc:.4f}"
            print(f"  epoch {ep:3d}  train {train_loss:.4f}  valid {valid_loss:.4f}  AUC {auc_str}")

        improved = (not np.isnan(valid_auc)) and valid_auc > best_auc + 1e-5
        if improved:
            best_auc = valid_auc
            best_valid = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"  早停于 epoch {ep}（验证集 AUC 连续 {patience} 轮未改善）")
                break

    model.load_state_dict(best_state)
    model.to(device)
    history["best_valid_loss"] = best_valid
    history["best_valid_auc"] = best_auc
    return history


def default_profile_kwargs(profile_name: str | None = None) -> dict:
    p = get_profile(profile_name)
    return {
        "d_model": p.d_model,
        "d_ff": p.d_ff,
        "n_heads": p.n_heads,
        "n_blocks": p.n_blocks,
        "dropout": p.dropout,
        "epochs": p.epochs,
        "batch_size": p.batch_size,
        "lr": p.lr,
        "max_len": p.max_seq_len,
        "seed": p.seed,
    }
