"""数据集切分与持久化。

约定（全项目统一）：
- 切分按**学生**做，80% / 10% / 10%。不按交互切，避免同一学生的序列泄漏到测试集，
  这是知识追踪评测里最常见的坑。
- 落盘格式为 JSON，字段见 SequenceDataset。
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

from ml.config import PROCESSED_DIR
from ml.data.synth import SequenceDataset


def split_dataset(
    dataset: SequenceDataset,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, SequenceDataset]:
    """按学生切分，返回 {'train':..., 'valid':..., 'test':...}。"""
    assert abs(sum(ratios) - 1.0) < 1e-6, "切分比例之和必须为 1"
    idx = list(range(len(dataset.sequences)))
    random.Random(seed).shuffle(idx)

    n = len(idx)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    parts = {
        "train": idx[:n_train],
        "valid": idx[n_train : n_train + n_valid],
        "test": idx[n_train + n_valid :],
    }

    out: dict[str, SequenceDataset] = {}
    for name, sel in parts.items():
        out[name] = SequenceDataset(
            kc_ids=dataset.kc_ids,
            question_ids=dataset.question_ids,
            question_to_kc=dataset.question_to_kc,
            question_difficulty=dataset.question_difficulty,
            sequences=[dataset.sequences[i] for i in sel],
            data_source=dataset.data_source,
            meta={**dataset.meta, "split": name, "split_seed": seed},
        )
    return out


def save_dataset(dataset: SequenceDataset, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kc_ids": dataset.kc_ids,
        "question_ids": dataset.question_ids,
        "question_to_kc": dataset.question_to_kc,
        "question_difficulty": dataset.question_difficulty,
        "sequences": dataset.sequences,
        "data_source": dataset.data_source,
        "meta": dataset.meta,
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path


def load_dataset(path: str | Path) -> SequenceDataset:
    path = Path(path)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    return SequenceDataset(**payload)


def save_splits(splits: dict[str, SequenceDataset], prefix: str = "dataset") -> dict[str, Path]:
    paths = {}
    for name, ds in splits.items():
        p = PROCESSED_DIR / f"{prefix}_{name}.json.gz"
        paths[name] = save_dataset(ds, p)
    # 索引文件，方便后端与训练脚本按名字找到数据
    index = {
        "prefix": prefix,
        "data_source": next(iter(splits.values())).data_source,
        "files": {k: str(v.name) for k, v in paths.items()},
        "stats": {k: v.stats() for k, v in splits.items()},
    }
    (PROCESSED_DIR / f"{prefix}_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths


def load_splits(prefix: str = "dataset") -> dict[str, SequenceDataset]:
    index_path = PROCESSED_DIR / f"{prefix}_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"未找到数据集索引 {index_path}，请先运行 python -m ml.data.build_dataset")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return {k: load_dataset(PROCESSED_DIR / v) for k, v in index["files"].items()}


def merge_sequences(*datasets: SequenceDataset) -> SequenceDataset:
    """把多个切分合并回一个（后端初始化时用）。"""
    base = datasets[0]
    seqs: list[dict] = []
    for ds in datasets:
        seqs.extend(ds.sequences)
    return SequenceDataset(
        kc_ids=base.kc_ids,
        question_ids=base.question_ids,
        question_to_kc=base.question_to_kc,
        question_difficulty=base.question_difficulty,
        sequences=seqs,
        data_source=base.data_source,
        meta={**base.meta, "merged": True},
    )
