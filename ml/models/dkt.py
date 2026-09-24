"""DKT（深度知识追踪，Piech et al., 2015）——本项目的主力模型。

结构：知识点嵌入 + 上一题作答嵌入 → LSTM → 单输出 sigmoid。
输入位置 t 用 (kc_t, resp_{t-1})，输出预测 resp_t。
不需要人工标注知识点映射到状态（可解释性弱，这是它的代价）。

说明：原始论文输出层是"对每个知识点各出一个概率"，评测时取当前知识点对应的那个。
第一版简化为单输出头（在单知识点被问到时只关心该题的正确率），
参数量更小、CPU 上也能秒级跑完一个 epoch，且 AUC 口径与其它模型完全一致。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ml.models.base import TorchKnowledgeTracer
from ml.models.torch_utils import default_profile_kwargs, predict_logits, train_loop


class DKTModule(nn.Module):
    def __init__(
        self,
        n_kc: int,
        d_model: int = 32,
        dropout: float = 0.2,
        n_layers: int = 1,
    ):
        super().__init__()
        # 多留一行给 padding
        self.pad_kc = n_kc
        self.kc_emb = nn.Embedding(n_kc + 1, d_model, padding_idx=self.pad_kc)
        self.resp_emb = nn.Embedding(2, d_model)
        self.proj = nn.Linear(d_model * 2, d_model)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            d_model, d_model, num_layers=n_layers, batch_first=True, dropout=0.0
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, kc: torch.Tensor, resp: torch.Tensor) -> torch.Tensor:
        # prev_resp[:, 0] = 0 表示"序列开始"
        prev = torch.zeros_like(resp)
        prev[:, 1:] = resp[:, :-1]
        x = torch.cat([self.kc_emb(kc), self.resp_emb(prev)], dim=-1)
        x = self.drop(torch.relu(self.proj(x)))
        h, _ = self.lstm(x)
        return self.head(self.drop(h)).squeeze(-1)


class DKT(TorchKnowledgeTracer):
    name = "DKT"

    def __init__(self, n_kc: int, device: str | None = None, **kwargs):
        super().__init__(device)
        cfg = default_profile_kwargs()
        cfg.update({k: v for k, v in kwargs.items() if v is not None})
        self.n_kc = n_kc
        self.pad_kc = n_kc
        self.config = cfg
        self.model = DKTModule(
            n_kc=n_kc,
            d_model=cfg["d_model"],
            dropout=cfg["dropout"],
        )
        self.history: dict = {}

    def fit(self, sequences: list[dict], valid: list[dict] | None = None) -> "DKT":
        if valid is None:
            valid = sequences
        torch.manual_seed(self.config["seed"])
        self.history = train_loop(
            self.model,
            train_seqs=sequences,
            valid_seqs=valid,
            pad_kc=self.pad_kc,
            device=self.device,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            lr=self.config["lr"],
            max_len=self.config["max_len"],
            seed=self.config["seed"],
        )
        return self

    def predict(self, sequences: list[dict]) -> list[np.ndarray]:
        return predict_logits(
            self.model,
            sequences,
            pad_kc=self.pad_kc,
            device=self.device,
            batch_size=max(256, self.config["batch_size"] * 4),
            max_len=self.config["max_len"],
        )

    def state_dict(self) -> dict:
        return {"name": self.name, "n_kc": self.n_kc, "config": self.config}

    @classmethod
    def from_state_dict(cls, payload: dict) -> "DKT":
        obj = cls(n_kc=payload["n_kc"], **payload.get("config", {}))
        return obj
