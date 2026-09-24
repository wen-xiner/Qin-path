"""知途（ZhiTu）全局配置。

设计原则：一份代码，CPU 与 GPU 都能跑。
- 设备：自动探测，有 CUDA 就用 GPU，否则回退 CPU（可用 ZT_DEVICE 强制指定）。
- 规模：用 PROFILE 档位控制，迁移到有显卡的机器时把 configs/profile 切到 large/gpu 即可，
  不需要改任何模型代码。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # Qin-path/
ML_DIR = ROOT / "ml"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = ML_DIR / "artifacts"

for _d in (RAW_DIR, PROCESSED_DIR, ARTIFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# 设备
# --------------------------------------------------------------------------
def resolve_device(prefer: str | None = None) -> str:
    """返回 'cuda' / 'mps' / 'cpu'。

    优先读环境变量 ZT_DEVICE，其次调参，最后自动探测。
    GPU 机器上装上 CUDA 版 torch 后无需改代码，这里会自动返回 'cuda'。
    """
    want = (prefer or os.environ.get("ZT_DEVICE") or "auto").lower()
    if want != "auto":
        return want
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = resolve_device()


# --------------------------------------------------------------------------
# 实验档位：CPU 机器用小规模，GPU 机器切大规模
# --------------------------------------------------------------------------
@dataclass
class Profile:
    name: str
    n_students: int            # 参与合成/采样的学生数
    max_seq_len: int           # 截断的答题序列长度
    batch_size: int
    epochs: int
    # DKT / SAKT 超参（刻意压小，保证 CPU 分钟级可跑完）
    d_model: int = 32
    d_ff: int = 64
    n_heads: int = 2
    n_blocks: int = 2
    dropout: float = 0.2
    lr: float = 2e-3
    seed: int = 42


PROFILES: dict[str, Profile] = {
    # 无显卡：小规模，但轮次要够——深度模型步数不足会直接退化成"预测均值"
    "cpu": Profile(
        name="cpu",
        n_students=800,
        max_seq_len=100,
        batch_size=32,
        epochs=60,
        d_model=48,
        d_ff=96,
    ),
    # 有显卡：放大数据与轮次，指标更稳
    "gpu": Profile(
        name="gpu",
        n_students=4000,
        max_seq_len=200,
        batch_size=128,
        epochs=120,
        d_model=128,
        d_ff=256,
        n_heads=4,
        n_blocks=3,
    ),
}
ALTERNATE_NAMES = {"large": "gpu", "small": "cpu", "cpu": "cpu", "gpu": "gpu"}

# 重要：max_seq_len 同时决定两件事——
#   1) 合成数据生成长度 / 真实数据的截断长度（build_dataset --max-seq-len 默认取它）
#   2) DKT / SAKT 在训练与评测时能看到多少步（torch_utils.default_profile_kwargs）
# 两者必须一致。否则 BKT（会在全长序列上评测）和神经网络模型其实不在同一批交互上比，
# 汇总表看着整齐，实际不可比。真实数据集构建时请显式传 --max-seq-len 对齐本值。


def get_profile(name: str | None = None) -> Profile:
    key = (name or os.environ.get("ZT_PROFILE") or "auto").lower()
    if key == "auto":
        key = "gpu" if DEVICE == "cuda" else "cpu"
    key = ALTERNATE_NAMES.get(key, "cpu")
    return PROFILES[key]


# --------------------------------------------------------------------------
# 知识点 / 数据集
# --------------------------------------------------------------------------
# 合成数据（用于打通常规流水线 + 演示）与真实公开数据集（用于答辩指标）分开标记，
# 避免把合成数据的指标当成真实结论。
SYNTH_DATASET = "synth_ds_v1"
DEFAULT_COURSE = "数据结构"

# 掌握度阈值：低于该值认为该知识点未掌握
MASTERY_THRESHOLD = 0.70
# 路径规划中判断"前置知识点是否拖后腿"的阈值
PREREQ_THRESHOLD = 0.70
