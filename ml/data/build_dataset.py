"""数据集构建入口（第一步）。

用法：
    # 默认：合成数据集（打通常规流水线 + 演示），按当前设备档位决定规模
    python -m ml.data.build_dataset

    # 指定规模档位
    python -m ml.data.build_dataset --profile gpu

    # 接入真实公开数据集（答辩指标必须走这条）
    python -m ml.data.build_dataset --source assistments \
        --csv data/raw/skill_builder_data_corrected.csv \
        --name assistments2009 --prefix assistments2009 --max-seq-len 100
    python -m ml.data.build_dataset --source pykt --csv data/raw/assist2009_pid.pkl --name assist2009

关键：`--max-seq-len` 必须与训练时模型的 max_len 一致（CPU 档默认见 ml/config.py）。
否则 BKT 会在全长序列上评测、而 DKT/SAKT 只评测前 max_len 步，
五个模型其实不在同一批交互上比，指标不可比。

产物：
    data/processed/knowledge_graph.json     知识点依赖图
    data/processed/questions.csv            题库
    data/processed/<prefix>_{train,valid,test}.json.gz   三个切分
    data/processed/<prefix>_index.json      索引与统计
"""

from __future__ import annotations

import argparse
import json

from ml.config import DEVICE, PROCESSED_DIR, get_profile
from ml.data.adapters import load_assistments_csv, load_pykt_pickle
from ml.data.dataset import save_splits, split_dataset
from ml.data.knowledge_graph import get_knowledge_graph
from ml.data.question_bank import build_question_bank, export_bank_csv
from ml.data.synth import generate_synthetic_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="知途数据集构建")
    ap.add_argument("--source", choices=["synth", "assistments", "pykt"], default="synth")
    ap.add_argument("--csv", help="真实数据集文件路径（assistments 用 csv、pykt 用 pkl）")
    ap.add_argument("--name", default=None, help="数据集名称，用于标记指标来源")
    ap.add_argument("--profile", default=None, help="规模档位 cpu / gpu")
    ap.add_argument("--students", type=int, default=None, help="覆盖合成数据的学生数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="dataset", help="产物文件名前缀")
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="每条学生序列的最大长度（真实数据集建议显式指定，且与训练时 max_len 一致）",
    )
    ap.add_argument(
        "--min-kc-count",
        type=int,
        default=30,
        help="真实数据集里出现次数少于该值的学习点会被丢弃",
    )
    args = ap.parse_args()

    profile = get_profile(args.profile)
    print(f"[device] {DEVICE}   [profile] {profile.name}")

    # 1) 知识点图与题库：无论用哪份数据都要导出，前端和后端都依赖它
    graph = get_knowledge_graph()
    graph.to_json(PROCESSED_DIR / "knowledge_graph.json")
    questions = build_question_bank()
    export_bank_csv(questions, PROCESSED_DIR / "questions.csv")
    print(f"[graph] {len(graph)} 个知识点，{graph.summary()['n_edges']} 条前置依赖")
    print(f"[bank ] {len(questions)} 道题，已导出 {PROCESSED_DIR / 'questions.csv'}")

    # 2) 数据集
    if args.source == "synth":
        ds = generate_synthetic_dataset(
            n_students=args.students or profile.n_students,
            max_seq_len=profile.max_seq_len,
            seed=args.seed,
            graph=graph,
            bank=questions,
        )
    elif args.source == "assistments":
        if not args.csv:
            raise SystemExit("--source assistments 需要 --csv 指定文件")
        ds = load_assistments_csv(
            args.csv,
            name=args.name or "assistments",
            max_seq_len=args.max_seq_len or profile.max_seq_len,
            min_kc_count=args.min_kc_count,
        )
    else:  # pykt
        if not args.csv:
            raise SystemExit("--source pykt 需要 --csv 指定 .pkl 文件")
        ds = load_pykt_pickle(
            args.csv,
            name=args.name or "pykt",
            max_seq_len=args.max_seq_len or profile.max_seq_len,
        )

    print(f"[data ] {json.dumps(ds.stats(), ensure_ascii=False)}")

    # 3) 切分与落盘
    splits = split_dataset(ds, seed=args.seed)
    paths = save_splits(splits, prefix=args.prefix)
    for k, v in paths.items():
        print(f"[split] {k:5s} -> {v.name}  {splits[k].stats()['n_students']} 名学生")

    if ds.data_source == "synthetic":
        print(
            "\n[提醒] 当前是合成数据。答辩指标请改用公开数据集：\n"
            "        python -m ml.data.build_dataset --source assistments --csv data/raw/<文件>.csv\n"
        )


if __name__ == "__main__":
    main()
