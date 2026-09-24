"""单模型训练入口（第二步，对应设计文档"三个台阶"的第一步）。

跑法：
    python -m ml.train --model DKT
    python -m ml.train --model BKT
    python -m ml.train --model SAKT --profile gpu

用途：
- 台阶 1：先跑通一个 DKT，拿到第一个 AUC 数字，建立信心；
- 调试：只想看某个模型的训练曲线时用这个，不用跑全套对比实验。
"""

from __future__ import annotations

import argparse
import json
import time

from ml.config import ARTIFACT_DIR, DEVICE, get_profile
from ml.data.dataset import load_splits
from ml.metrics import evaluate, summarize_table
from ml.models.registry import REGISTRY, build_model


def main() -> None:
    ap = argparse.ArgumentParser(description="知途单模型训练")
    ap.add_argument("--model", default="DKT", choices=list(REGISTRY))
    ap.add_argument("--prefix", default="dataset")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    profile = get_profile(args.profile)
    splits = load_splits(args.prefix)
    n_kc = splits["train"].n_kc

    print(f"[device] {DEVICE}   [profile] {profile.name}   [model] {args.model}")
    print(f"[data  ] {json.dumps(splits['train'].stats(), ensure_ascii=False)}")

    model = build_model(args.model, n_kc=n_kc, device=DEVICE)
    t0 = time.time()
    model.fit(splits["train"], splits["valid"])
    elapsed = time.time() - t0

    metrics = evaluate(splits["test"], model.predict(splits["test"]))
    print(summarize_table({args.model: metrics}))
    print(f"训练 + 评测用时 {elapsed:.1f} 秒")

    saved = model.save(ARTIFACT_DIR / "models" / f"{args.model}_single")
    print(f"模型已保存 -> {saved}")

    if args.model == "BKT":
        from ml.data.knowledge_graph import get_knowledge_graph

        graph = get_knowledge_graph()
        print("\nBKT 学到的参数（前 10 个知识点）：")
        print(f"{'知识点':<24}{'L0':>8}{'T':>8}{'S':>8}{'G':>8}{'样本数':>8}")
        for row in model.param_table()[:10]:
            i = row["kc"]
            name = graph[graph.kc_ids[i]].name if i < len(graph) else str(i)
            print(
                f"{name:<24}{row['L0']:>8.4f}{row['T']:>8.4f}"
                f"{row['S']:>8.4f}{row['G']:>8.4f}{row['n_obs']:>8}"
            )


if __name__ == "__main__":
    main()
