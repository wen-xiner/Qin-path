"""非学习基线。

设计文档第五、七节明确要求：必须保留一个非学习基线作为下限参照，只有超过基线模型的价值才成立。
这里给两个：
1. GlobalMeanBaseline  —— 全数据集答对率常数预测，最弱的基线
2. KCMeanBaseline      —— 每个知识点各自的答对率常数预测，是更严格、更有说服力的下限

两者都不使用序列信息，也不做任何学习，只统计频率。
"""

from __future__ import annotations

import numpy as np

from ml.models.base import KnowledgeTracer


class GlobalMeanBaseline(KnowledgeTracer):
    name = "Mean"
    is_learned = False

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.p: float = 0.5

    def fit(self, sequences: list[dict], valid: list[dict] | None = None) -> "GlobalMeanBaseline":
        n_pos = sum(sum(s["resp_seq"]) for s in sequences)
        n_tot = sum(len(s["resp_seq"]) for s in sequences)
        if n_tot == 0:
            raise ValueError("训练集为空")
        # 拉普拉斯平滑，避免极端值
        self.p = (n_pos + self.alpha) / (n_tot + 2 * self.alpha)
        return self

    def predict(self, sequences: list[dict]) -> list[np.ndarray]:
        return [np.full(len(s["resp_seq"]), self.p) for s in sequences]

    def predict_next(self, kc_seq, resp_seq, next_kc: int) -> float:
        return float(self.p)

    def mastery(self, kc_seq, resp_seq, n_kc: int) -> np.ndarray:
        return np.full(n_kc, self.p)

    def state_dict(self) -> dict:
        return {"name": self.name, "p": self.p}

    @classmethod
    def from_state_dict(cls, payload: dict) -> "GlobalMeanBaseline":
        obj = cls()
        obj.p = payload["p"]
        return obj


class KCMeanBaseline(KnowledgeTracer):
    """按知识点的答对率做常数预测，带贝叶斯平滑（向全局均值收缩）。"""

    name = "KC-Mean"
    is_learned = False

    def __init__(self, n_kc: int, alpha: float = 5.0):
        self.n_kc = n_kc
        self.alpha = alpha
        self.prior: float = 0.5
        self.p_per_kc: np.ndarray = np.full(n_kc, 0.5)

    def fit(self, sequences: list[dict], valid: list[dict] | None = None) -> "KCMeanBaseline":
        pos = np.zeros(self.n_kc)
        tot = np.zeros(self.n_kc)
        for s in sequences:
            for kc, r in zip(s["kc_seq"], s["resp_seq"]):
                tot[kc] += 1
                pos[kc] += r
        self.prior = (pos.sum() + 1.0) / (tot.sum() + 2.0)
        # 平滑：p_kc = (pos_kc + alpha * prior) / (tot_kc + alpha)
        self.p_per_kc = (pos + self.alpha * self.prior) / (tot + self.alpha)
        return self

    def predict(self, sequences: list[dict]) -> list[np.ndarray]:
        out = []
        for s in sequences:
            kc_arr = np.asarray(s["kc_seq"], dtype=int)
            out.append(self.p_per_kc[kc_arr])
        return out

    def predict_next(self, kc_seq, resp_seq, next_kc: int) -> float:
        return float(self.p_per_kc[next_kc])

    def mastery(self, kc_seq, resp_seq, n_kc: int) -> np.ndarray:
        return self.p_per_kc[:n_kc].copy()

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "n_kc": self.n_kc,
            "alpha": self.alpha,
            "prior": self.prior,
            "p_per_kc": self.p_per_kc.tolist(),
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> "KCMeanBaseline":
        obj = cls(n_kc=payload["n_kc"], alpha=payload.get("alpha", 5.0))
        obj.prior = payload["prior"]
        obj.p_per_kc = np.asarray(payload["p_per_kc"], dtype=float)
        return obj
