"""模型注册表。

对比实验按这里的清单跑，增删模型只改一处。
"""

from __future__ import annotations

from ml.models.base import KnowledgeTracer
from ml.models.baseline import GlobalMeanBaseline, KCMeanBaseline
from ml.models.bkt import BKT
from ml.models.dkt import DKT
from ml.models.sakt import SAKT

# order 决定报告里的排列顺序
REGISTRY: dict[str, dict] = {
    "Mean": {"cls": GlobalMeanBaseline, "kind": "baseline", "desc": "全局均值预测（最弱基线）"},
    "KC-Mean": {"cls": KCMeanBaseline, "kind": "baseline", "desc": "逐知识点频率预测（严格下限）"},
    "BKT": {"cls": BKT, "kind": "classic", "desc": "贝叶斯知识追踪，四参数、可解释"},
    "DKT": {"cls": DKT, "kind": "deep", "desc": "深度知识追踪，LSTM 序列建模（主力）"},
    "SAKT": {"cls": SAKT, "kind": "deep", "desc": "自注意力知识追踪（对照）"},
}


def build_model(name: str, n_kc: int, device: str | None = None, **kwargs) -> KnowledgeTracer:
    if name not in REGISTRY:
        raise KeyError(f"未注册的模型 {name}，可选：{list(REGISTRY)}")
    cls = REGISTRY[name]["cls"]
    if cls is GlobalMeanBaseline:
        return GlobalMeanBaseline(**kwargs)
    if cls is KCMeanBaseline:
        return KCMeanBaseline(n_kc=n_kc, **kwargs)
    if cls is BKT:
        return BKT(n_kc=n_kc, **kwargs)
    # 深度模型额外接受 device
    return cls(n_kc=n_kc, device=device, **kwargs)


def default_model_list() -> list[str]:
    return ["Mean", "KC-Mean", "BKT", "DKT", "SAKT"]
