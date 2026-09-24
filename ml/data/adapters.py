"""公开学术数据集适配器。

设计文档第六节的第一选择：使用本领域的标准公开数据集，答辩时可以说"我们采用了领域标准数据集"。
已支持的格式：

1. **ASSISTments 原始 CSV**（2009 / 2012 / 2015 等版本）
   列名常见为：user_id, skill_id（或 skill_name）, problem_id, correct, order_id
   适配后 → SequenceDataset

2. **pyKT 预处理后的 pickle**
   pyKT（pykt-toolkit）把数据整理成 dict：
       {split: {0: Q, 1: ...}} 形式各异，主流为
       d[split]['q_seqs'] / ['r_seqs'] / ['qshft_seqs'] / ['rshft_seqs']
   本适配器读取 'q_seqs' 与 'r_seqs'。

3. **KDD Cup 2010 / EdNet / XES3G5M**
   这三个的原始格式差别较大，建议先用 EduData 或 pyKT 统一成上面的 pickle 格式，
   再用 load_pykt_pickle() 接入。这是最省事的路径。

使用方式：
    ds = load_assistments_csv("data/raw/assistments2009.csv", name="assistments2009")
    ds = load_pykt_pickle("data/raw/assist2009_pid.pkl", name="assist2009")

下载渠道：
    方式一（本机已跑通，推荐）：直接拉 USTC 镜像的原始 zip，不依赖 EduData
        curl -L -o data/raw/assist2009_skill_builder.zip \
            http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip
        解压后得到 skill_builder_data_corrected.csv → load_assistments_csv()
    方式二：pip install EduData
        EduData 内置数据集：assist2009 / assist2012 / assist2015 / algebra2005 /
                            bridge2006 / statics2011 / junyi / xes3g5m / ednet_kt1
        USTC 镜像会间歇性被网络拦截（连接被 RST，或响应不带 Content-Length
        导致下载器抛 TypeError），被拦时改用方式一。
"""

from __future__ import annotations

import codecs
import contextlib
import csv
import json
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path

from ml.data.synth import SequenceDataset

CSV_ALIASES = {
    "student": ["user_id", "student_id", "user", "anon_student_id", "student"],
    "kc": ["skill_id", "skill", "kc_id", "skill_name", "knowledge_component", "kc"],
    "item": ["problem_id", "item_id", "question_id", "problem", "item"],
    "correct": ["correct", "is_correct", "score", "response", "label"],
    "order": ["order_id", "timestamp", "start_time", "order", "opportunity"],
}


def _pick_column(fieldnames: list[str], key: str) -> str | None:
    lower = {f.lower(): f for f in fieldnames}
    for alias in CSV_ALIASES[key]:
        if alias in lower:
            return lower[alias]
    return None


# ASSISTments 早期的 skill_builder 导出文件不是 UTF-8（实测 2009 版是 cp1252/latin-1，
# 题目文本里有 0x80 这类字节，直接按 utf-8 读会在第 2634 字节就炸）。
# 这里按 "先严格后宽松" 的顺序嗅探：latin-1 对任意字节都不报错，是保底选项。
TEXT_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def _detect_encoding(path: Path, chunk_size: int = 1 << 20) -> str:
    """流式嗅探文件编码。

    刻意不把整份文件读进内存：ASSISTments 2009 的 corrected 版有 40 万行 × 30 列，
    一次性物化成 dict 列表要吃掉近 1GB。这里只按块增量解码，失败就换下一种编码。
    """
    for enc in TEXT_ENCODINGS:
        try:
            with open(path, "rb") as f:
                decoder = codecs.getincrementaldecoder(enc)()
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        decoder.decode(b"", final=True)
                        break
                    decoder.decode(chunk)
            return enc
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path.name} 无法用 {TEXT_ENCODINGS} 任一编码解码")


@contextlib.contextmanager
def _open_csv(path: Path, encoding: str | None = None):
    """按指定（或嗅探到的）编码惰性打开 CSV，产出 (fieldnames, 行迭代器)。"""
    with open(path, encoding=encoding or _detect_encoding(path), newline="") as f:
        reader = csv.DictReader(f)
        yield (reader.fieldnames or []), reader


def _read_fieldnames(path: Path, encoding: str) -> list[str]:
    """只读表头第一行，用来在正式解析前校验列名。"""
    with open(path, encoding=encoding, newline="") as f:
        return next(csv.reader(f), [])


def _parse_assistments_rows(
    path: Path, cols: dict[str, str | None], encoding: str
) -> tuple[dict, dict[str, int], int]:
    """把 CSV 逐行归集成 student -> kc -> [(order, item, correct)]。

    单独抽出来是为了让文件句柄有明确的 with 作用域（40 万行的文件不能常驻打开）。
    """
    raw: dict[str, dict[str, list[tuple[float, str, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    dropped = {"kc_empty": 0, "item_empty": 0, "value_error": 0}
    n_rows = 0
    with _open_csv(path, encoding) as (_, rows):
        for row in rows:
            n_rows += 1
            sid = (row.get(cols["student"]) or "").strip()
            kc = (row.get(cols["kc"]) or "").strip()
            item = (row.get(cols["item"]) or "").strip()
            # ASSISTments 2009 有约 16% 的行 skill_id / skill_name 为空，无法归到知识点，直接丢弃
            if not kc or kc.lower() in {"nan", "none", "null"}:
                dropped["kc_empty"] += 1
                continue
            if not item or item.lower() in {"nan", "none", "null"}:
                dropped["item_empty"] += 1
                continue
            try:
                correct = int(float(row[cols["correct"]]) > 0)
            except (TypeError, ValueError):
                dropped["value_error"] += 1
                continue
            if cols["order"]:
                try:
                    order = float(row[cols["order"]])
                except (TypeError, ValueError):
                    order = float(len(raw[sid][kc]))
            else:
                order = float(len(raw[sid][kc]))
            raw[sid][kc].append((order, item, correct))
    return raw, dropped, n_rows


def load_assistments_csv(
    path: str | Path,
    name: str = "assistments",
    min_kc_count: int = 30,
    min_seq_len: int = 5,
    max_seq_len: int = 300,
) -> SequenceDataset:
    """读入 ASSISTments 风格的 CSV。

    min_kc_count：出现次数少于该值的学习点会被丢弃（长尾知识点样本太少，会污染指标）。
    max_seq_len：每条学生序列的截断长度（须与训练时模型的 max_len 一致，见 build_dataset 注释）。
    """
    path = Path(path)
    if zipfile.is_zipfile(path):
        raise ValueError("请先解压 CSV 再调用本函数")

    encoding = _detect_encoding(path)
    fieldnames = _read_fieldnames(path, encoding)
    cols = {k: _pick_column(fieldnames, k) for k in CSV_ALIASES}
    missing = [k for k, v in cols.items() if v is None and k != "order"]
    if missing:
        raise ValueError(
            f"CSV 缺少必要列 {missing}；实际列为 {fieldnames}。"
            f"可用列名别名见 CSV_ALIASES。"
        )

    raw, dropped, n_rows = _parse_assistments_rows(path, cols, encoding)

    # 知识点频次过滤：出现次数少于 min_kc_count 的视为长尾，丢弃（样本太少会污染指标）
    kc_counter: dict[str, int] = defaultdict(int)
    for kc_map in raw.values():
        for kc, recs in kc_map.items():
            kc_counter[kc] += len(recs)
    kept_kc = sorted(k for k, c in kc_counter.items() if c >= min_kc_count)
    if not kept_kc:
        raise ValueError("过滤后没有剩余知识点，请调小 min_kc_count")
    kc_index = {k: i for i, k in enumerate(kept_kc)}

    # 题目 → 知识点。注意必须只在保留下来的知识点里构建：
    # 否则题目表会带上属于已过滤 KC 的题目，question_to_kc 映射时直接 KeyError。
    item_to_kc: dict[str, str] = {}
    for kc_map in raw.values():
        for kc, recs in kc_map.items():
            if kc not in kc_index:
                continue
            for _, item, _ in recs:
                item_to_kc.setdefault(item, kc)

    item_ids = sorted(item_to_kc)
    item_index = {it: i for i, it in enumerate(item_ids)}

    sequences: list[dict] = []
    for sid, kc_map in raw.items():
        # 把该学生的所有交互按时间（order）串成一条序列
        events: list[tuple[float, str, str, int]] = []
        for kc, recs in kc_map.items():
            for order, item, correct in recs:
                events.append((order, kc, item, correct))
        events.sort(key=lambda e: e[0])
        if len(events) < min_seq_len:
            continue

        events = events[:max_seq_len]
        kc_seq, q_seq, resp_seq = [], [], []
        for _, kc, item, correct in events:
            if kc not in kc_index:
                continue
            kc_seq.append(kc_index[kc])
            q_seq.append(item_index[item])
            resp_seq.append(int(correct))
        if len(resp_seq) < min_seq_len:
            continue
        sequences.append(
            {"student_id": sid, "kc_seq": kc_seq, "q_seq": q_seq, "resp_seq": resp_seq}
        )

    if not sequences:
        raise ValueError("没有构造出任何有效序列，请检查 min_seq_len / min_kc_count")

    return SequenceDataset(
        kc_ids=kept_kc,
        question_ids=item_ids,
        question_to_kc=[kc_index[item_to_kc[it]] for it in item_ids],
        question_difficulty=[2] * len(item_ids),  # 公开数据集没有难度标注，统一置 2
        sequences=sequences,
        data_source=name,
        meta={
            "format": "assistments_csv",
            "path": str(path),
            "encoding": encoding,
            "raw_rows": n_rows,
            "dropped_rows": dropped,
            "kc_total_raw": len(kc_counter),
            "kc_kept": len(kept_kc),
            "min_kc_count": min_kc_count,
            "min_seq_len": min_seq_len,
            "max_seq_len": max_seq_len,
        },
    )


def load_pykt_pickle(path: str | Path, name: str = "pykt", max_seq_len: int = 300) -> SequenceDataset:
    """读入 pyKT 预处理后的 pickle。

    pyKT 的数据结构为
        d[split] = {'q_seqs': LongTensor[n, L], 'r_seqs': LongTensor[n, L], ...}
    其中 q_seqs 已是"题目 → 知识点/题目 id"的整数编码，q 与 kc 在 pyKT 里同义。
    这里把它当作题目 id 与知识点 id 相同的编码（pyKT 官方示例就是这么评测的）。
    """
    path = Path(path)
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError("pickle 内容不是 dict，请确认是 pyKT 预处理产物")

    sequences: list[dict] = []
    max_item = 0
    for split, block in data.items():
        if not isinstance(block, dict) or "q_seqs" not in block:
            continue
        q_seqs = block["q_seqs"]
        r_seqs = block["r_seqs"]
        for i in range(len(q_seqs)):
            q_row = q_seqs[i]
            r_row = r_seqs[i]
            q_list = [int(x) for x in (q_row.tolist() if hasattr(q_row, "tolist") else q_row)]
            r_list = [int(x) for x in (r_row.tolist() if hasattr(r_row, "tolist") else r_row)]
            # pyKT 用 0 做 padding，末尾对齐；这里裁掉后端的 padding
            while q_list and q_list[-1] == 0:
                q_list.pop()
                r_list.pop()
            if len(q_list) < 5:
                continue
            q_list, r_list = q_list[:max_seq_len], r_list[:max_seq_len]
            max_item = max(max_item, max(q_list))
            sequences.append(
                {
                    "student_id": f"{split}_{i}",
                    "kc_seq": q_list,
                    "q_seq": q_list,
                    "resp_seq": r_list,
                }
            )

    if not sequences:
        raise ValueError("未从 pickle 中解析出序列，请检查字段名（q_seqs / r_seqs）")

    n_kc = max_item + 1
    return SequenceDataset(
        kc_ids=[f"kc_{i}" for i in range(n_kc)],
        question_ids=[f"q_{i}" for i in range(n_kc)],
        question_to_kc=list(range(n_kc)),
        question_difficulty=[2] * n_kc,
        sequences=sequences,
        data_source=name,
        meta={"format": "pykt_pickle", "path": str(path)},
    )


def dataset_available_report() -> dict:
    """看看 data/raw 下有没有可用的真实数据集。"""
    from ml.config import RAW_DIR

    found = []
    for p in sorted(RAW_DIR.glob("*")):
        if p.is_file():
            found.append({"file": p.name, "size_mb": round(p.stat().st_size / 1e6, 2)})
    return {
        "raw_dir": str(RAW_DIR),
        "files": found,
        "hint": "把公开数据集放到 data/raw/ 下，再用 load_assistments_csv 或 load_pykt_pickle 接入",
    }


if __name__ == "__main__":
    print(json.dumps(dataset_available_report(), ensure_ascii=False, indent=2))
