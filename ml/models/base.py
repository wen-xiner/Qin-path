"""知识追踪模型的统一接口。

所有模型（含基线）都实现 KnowledgeTracer，这样对比实验才能用同一套代码跑，
保证"同一数据集、同一套评测"成立。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class KnowledgeTracer(ABC):
    """知识追踪模型接口。

    序列对齐约定：predict() 返回的数组第 t 个元素，是"用前 t 个交互预测第 t 个交互答对"的概率。
    也就是说第 0 个位置只能用先验预测，评测时同样计入（这是 KT 领域通行做法）。
    """

    name: str = "kt"
    # 是否为"学习型"模型。False 表示非学习基线。
    is_learned: bool = True

    @abstractmethod
    def fit(self, sequences: list[dict], valid: list[dict] | None = None) -> "KnowledgeTracer":
        ...

    @abstractmethod
    def predict(self, sequences: list[dict]) -> list[np.ndarray]:
        """返回每条序列逐位置的答对概率。"""
        ...

    # ------------------------------------------------------------ 在线推断
    def predict_next(self, kc_seq: list[int], resp_seq: list[int], next_kc: int) -> float:
        """给定历史交互，预测下一个知识点 next_kc 上的答对概率。

        供后端实时推荐使用。默认实现走整段 predict 后取末位，
        子类（如 BKT）会覆写成更高效的增量版本。
        """
        if not kc_seq:
            raise ValueError("需要至少一个历史交互")
        seq = {"kc_seq": list(kc_seq), "q_seq": list(kc_seq), "resp_seq": list(resp_seq)}
        probs = self.predict([seq])[0]
        return float(probs[-1])

    def mastery(self, kc_seq: list[int], resp_seq: list[int], n_kc: int) -> np.ndarray:
        """估计每个知识点的掌握概率。默认用"最近一次在该知识点上的预测正确率"近似。"""
        probs = self.predict(
            [{"kc_seq": list(kc_seq), "q_seq": list(kc_seq), "resp_seq": list(resp_seq)}]
        )[0]
        out = np.full(n_kc, np.nan)
        for t, kc in enumerate(kc_seq):
            if t < len(probs):
                out[kc] = probs[t]
        return out

    # ------------------------------------------------------------ 存取
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        # 没有后缀时补 .json：否则产物会叫 BKT_synthetic（无后缀），
        # 加载端按 "BKT_synthetic.*" 通配找不到，就会静默退化成"每次启动重训一遍"。
        if not path.suffix:
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def state_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeTracer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_state_dict(payload)

    @classmethod
    def from_state_dict(cls, payload: dict) -> "KnowledgeTracer":
        raise NotImplementedError


class TorchKnowledgeTracer(KnowledgeTracer):
    """基于 PyTorch 的模型基类，负责设备放置与权重存取。"""

    def __init__(self, device: str | None = None):
        from ml.config import resolve_device

        self.device = resolve_device(device)
        self.model = None
        self.config: dict = {}

    def _to_device(self):
        if self.model is not None:
            self.model.to(self.device)
        return self

    def save(self, path: str | Path) -> Path:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".pt":
            path = path.with_suffix(".pt")
        torch.save(
            {
                "name": self.name,
                "config": self.config,
                "state_dict": self.model.state_dict() if self.model is not None else None,
            },
            path,
        )
        return path

    @classmethod
    def load_weights(cls, model, path: str | Path):
        import torch

        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        return ckpt.get("config", {})
