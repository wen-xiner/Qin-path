"""评测指标。

知识追踪领域通用口径（设计文档第五节要求）：
- AUC：核心指标，衡量"把答对排在答错前面"的能力，不受阈值影响
- ACC：准确率，阈值 0.5
- RMSE / NLL 作为补充

同时给出两套口径，避免答辩时被追问口径不清：
- *_all      ：包含每条序列的第一个位置（该位置只有先验，没有历史）
- *_skip1st  ：排除每条序列的第一个位置，只看"有历史可依"的预测
两者都要报，主指标用 AUC_all（与多数论文一致），AUC_skip1st 作为佐证。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def flatten_predictions(
    sequences: list[dict],
    probs: list[np.ndarray],
    skip_first: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    for s, p in zip(sequences, probs):
        resp = np.asarray(s["resp_seq"], dtype=int)
        p = np.asarray(p, dtype=float)
        n = min(len(resp), len(p))
        resp, p = resp[:n], p[:n]
        start = 1 if skip_first else 0
        y_true.append(resp[start:])
        y_pred.append(p[start:])
    if not y_true:
        return np.array([]), np.array([])
    return np.concatenate(y_true), np.concatenate(y_pred)


def _safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def evaluate(sequences: list[dict], probs: list[np.ndarray]) -> dict:
    """返回完整指标字典。"""
    out: dict[str, float] = {}

    for tag, skip in (("all", False), ("skip1st", True)):
        y, p = flatten_predictions(sequences, probs, skip_first=skip)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        out[f"auc_{tag}"] = _safe_auc(y, p)
        out[f"acc_{tag}"] = float(((p >= 0.5).astype(int) == y).mean()) if len(y) else float("nan")
        out[f"rmse_{tag}"] = float(np.sqrt(((p - y) ** 2).mean())) if len(y) else float("nan")
        out[f"nll_{tag}"] = (
            float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()) if len(y) else float("nan")
        )
        out[f"n_{tag}"] = int(len(y))
    return out


def per_student_auc(sequences: list[dict], probs: list[np.ndarray]) -> np.ndarray:
    """逐学生的 AUC，用于画分布（有的学生模型预测得好，有的差，只看均值会掩盖）。"""
    vals = []
    for s, p in zip(sequences, probs):
        resp = np.asarray(s["resp_seq"], dtype=int)
        p = np.asarray(p, dtype=float)
        n = min(len(resp), len(p))
        vals.append(_safe_auc(resp[:n], p[:n]))
    return np.asarray(vals, dtype=float)


def bootstrap_ci(
    sequences: list[dict],
    probs: list[np.ndarray],
    metric: str = "auc_all",
    n_boot: int = 200,
    seed: int = 42,
) -> tuple[float, float, float]:
    """按学生做 bootstrap，给出指标的 95% 置信区间。

    答辩时"两个模型差距 0.01 到底算不算差"这个问题，靠这个区间回答：
    如果区间大幅重叠，就不能声称谁更好——这正是设计文档第三节引用的那篇实证研究的结论。
    """
    rng = np.random.default_rng(seed)
    n = len(sequences)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    point = evaluate(sequences, probs)[metric]
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub_seq = [sequences[i] for i in idx]
        sub_prob = [probs[i] for i in idx]
        v = evaluate(sub_seq, sub_prob)[metric]
        if not np.isnan(v):
            stats.append(v)
    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def summarize_table(results: dict[str, dict]) -> str:
    """把 {模型名: 指标} 渲染成对齐的文本表格。"""
    cols = [
        ("auc_all", "AUC"),
        ("acc_all", "ACC"),
        ("rmse_all", "RMSE"),
        ("nll_all", "NLL"),
        ("auc_skip1st", "AUC*"),
    ]
    header = f"{'模型':<10}" + "".join(f"{label:>9}" for _, label in cols)
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        row = f"{name:<10}"
        for key, _ in cols:
            v = m.get(key, float("nan"))
            row += f"{v:>9.4f}" if not np.isnan(v) else f"{'--':>9}"
        lines.append(row)
    lines.append("")
    lines.append("AUC* = 排除每条序列第一个位置的口径")
    return "\n".join(lines)
