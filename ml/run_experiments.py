"""对比实验入口（第三步，本项目的核心产出）。

跑法：
    python -m ml.run_experiments                     # 用当前设备档位，跑全部模型
    python -m ml.run_experiments --models BKT DKT SAKT
    python -m ml.run_experiments --skip-bootstrap    # 快速模式，不算置信区间

产出：
    ml/artifacts/results_<数据集名>.json     指标原始数据
    ml/artifacts/report_<数据集名>.md        可读的实验报告
    ml/artifacts/models/<模型>_<数据集名>.pt|.json   训练好的模型（后端要加载）
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ml.config import ARTIFACT_DIR, DEVICE, PROCESSED_DIR, get_profile
from ml.data.dataset import load_splits
from ml.metrics import bootstrap_ci, evaluate, per_student_auc, summarize_table
from ml.models.registry import REGISTRY, build_model, default_model_list


def run_one(name: str, splits: dict, device: str, n_kc: int, do_bootstrap: bool, seed: int) -> dict:
    print(f"\n{'=' * 70}\n[模型] {name}  ({REGISTRY[name]['desc']})\n{'=' * 70}")
    model = build_model(name, n_kc=n_kc, device=device)
    t0 = time.time()
    model.fit(splits["train"], splits["valid"])
    train_sec = time.time() - t0

    probs_test = model.predict(splits["test"])
    metrics = evaluate(splits["test"], probs_test)
    metrics["train_seconds"] = round(train_sec, 2)
    metrics["n_params"] = _count_params(model)

    if do_bootstrap:
        point, lo, hi = bootstrap_ci(splits["test"], probs_test, metric="auc_all")
        metrics["auc_all_ci95"] = [round(lo, 4), round(hi, 4)]

    psa = per_student_auc(splits["test"], probs_test)
    psa_valid = psa[~np.isnan(psa)]
    metrics["per_student_auc_mean"] = round(float(psa_valid.mean()), 4) if len(psa_valid) else None
    metrics["per_student_auc_std"] = round(float(psa_valid.std()), 4) if len(psa_valid) else None

    print(
        f"  AUC {metrics['auc_all']:.4f}  ACC {metrics['acc_all']:.4f}  "
        f"RMSE {metrics['rmse_all']:.4f}  用时 {train_sec:.1f}s"
    )
    if "auc_all_ci95" in metrics:
        print(f"  AUC 95% 置信区间 {metrics['auc_all_ci95']}")

    _save_model(model, name, splits["train"].data_source)
    return metrics


def _count_params(model) -> int | None:
    if getattr(model, "model", None) is None:
        return None
    return int(sum(p.numel() for p in model.model.parameters()))


def _save_model(model, name: str, data_source: str) -> None:
    out_dir = ARTIFACT_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = data_source.replace("/", "_").replace("\\", "_")
    path = out_dir / f"{name}_{safe}"
    saved = model.save(path)
    print(f"  模型已保存 -> {saved.relative_to(ARTIFACT_DIR.parent.parent)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="知途对比实验")
    ap.add_argument("--models", nargs="*", default=None, help="要跑的模型名，默认全部")
    ap.add_argument("--prefix", default="dataset", help="数据集前缀，对应 data/processed/dataset_index.json")
    ap.add_argument("--profile", default=None, help="规模档位 cpu / gpu")
    ap.add_argument("--skip-bootstrap", action="store_true", help="跳过 bootstrap 置信区间（更快）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    profile = get_profile(args.profile)
    models = args.models or default_model_list()

    splits = load_splits(args.prefix)
    n_kc = splits["train"].n_kc
    data_source = splits["train"].data_source

    print("=" * 70)
    print("知途 · 知识追踪模型对比实验")
    print("=" * 70)
    print(f"设备         : {DEVICE}")
    print(f"规模档位     : {profile.name}")
    print(f"数据集       : {data_source}")
    print(f"数据规模     : {json.dumps(splits['train'].stats(), ensure_ascii=False)}")
    print(f"验证集       : {splits['valid'].stats()['n_students']} 名学生 / "
          f"{splits['valid'].stats()['n_interactions']} 条交互")
    print(f"测试集       : {splits['test'].stats()['n_students']} 名学生 / "
          f"{splits['test'].stats()['n_interactions']} 条交互")
    if data_source == "synthetic":
        print("\n!! 当前为合成数据，指标仅用于验证实现是否正确，不能作为答辩结论 !!")
        print("!! 答辩指标请执行：python -m ml.data.build_dataset --source assistments --csv data/raw/<文件>.csv")

    results: dict[str, dict] = {}
    for name in models:
        results[name] = run_one(
            name, splits, DEVICE, n_kc, do_bootstrap=not args.skip_bootstrap, seed=args.seed
        )

    # -------------------------------------------------------------- 输出
    table = summarize_table(results)
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    print(table)

    payload = {
        "meta": {
            "device": DEVICE,
            "profile": profile.name,
            "data_source": data_source,
            "n_kc": n_kc,
            "dataset_meta": splits["train"].meta,
            "train": splits["train"].stats(),
            "valid": splits["valid"].stats(),
            "test": splits["test"].stats(),
            "seed": args.seed,
            "bootstrap": not args.skip_bootstrap,
            "note": (
                "合成数据仅用于验证实现与演示" if data_source == "synthetic" else "公开数据集"
            ),
        },
        "results": results,
        "models": {k: {"kind": v["kind"], "desc": v["desc"]} for k, v in REGISTRY.items() if k in results},
    }
    safe = data_source.replace("/", "_").replace("\\", "_")
    out_json = ARTIFACT_DIR / f"results_{safe}.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = render_report(payload, table)
    out_md = ARTIFACT_DIR / f"report_{safe}.md"
    out_md.write_text(report, encoding="utf-8")

    print(f"\n指标已保存 -> {out_json.name}")
    print(f"报告已保存 -> {out_md.name}")


def render_report(payload: dict, table: str) -> str:
    meta = payload["meta"]
    res = payload["results"]
    dm = meta.get("dataset_meta", {}) or {}
    lines = [
        "# 知途 · 知识追踪模型对比实验报告",
        "",
        f"- 数据集：`{meta['data_source']}`",
        f"- 知识点数：{meta['n_kc']}",
        f"- 训练 / 验证 / 测试：{meta['train']['n_students']} / {meta['valid']['n_students']} / {meta['test']['n_students']} 名学生",
        f"- 交互条数：{meta['train']['n_interactions']} / {meta['valid']['n_interactions']} / {meta['test']['n_interactions']}",
        f"- 运行设备：`{meta['device']}`（档位 `{meta['profile']}`）",
        f"- 随机种子：{meta['seed']}",
        "",
    ]
    if dm.get("path"):
        dropped = dm.get("dropped_rows") or {}
        dropped_txt = "、".join(f"{k}={v}" for k, v in dropped.items()) or "无"
        lines += [
            "### 数据来源与预处理（可追溯）",
            "",
            f"- 源文件：`{dm.get('path')}`（编码 `{dm.get('encoding')}`）",
            f"- 原始行数：{dm.get('raw_rows')}；丢弃：{dropped_txt}（`skill_id` 为空的行无法归到知识点）",
            f"- 知识点：保留 {dm.get('kc_kept')} / 原始 {dm.get('kc_total_raw')}"
            f"（`min_kc_count={dm.get('min_kc_count')}`，长尾知识点样本太少会污染指标）",
            f"- 每条序列截断长度：{dm.get('max_seq_len')}，最短保留长度：{dm.get('min_seq_len')}",
            "",
            "> 截断长度与训练时模型的 `max_len` 一致（见 `ml/config.py` 的档位定义），",
            "> 否则 BKT 会在全长序列上评测、神经网络模型只评测前 `max_len` 步，五个模型不可比。",
            "",
        ]
    if meta["data_source"] == "synthetic":
        lines += [
            "> **注意**：本报告使用合成数据，目的是验证模型实现与评测流水线是否正确，",
            "> **不能作为答辩结论**。答辩指标必须来自公开学术数据集。",
            "",
        ]

    lines += ["## 一、指标汇总", "", "```", table, "```", ""]

    lines += ["## 二、逐模型明细", ""]
    for name, m in res.items():
        lines.append(f"### {name}（{payload['models'][name]['desc']}）")
        lines.append("")
        lines.append(f"- AUC（全部位置）：{m['auc_all']:.4f}")
        if m.get("auc_all_ci95"):
            lo, hi = m["auc_all_ci95"]
            lines.append(f"- AUC 95% 置信区间（按学生 bootstrap）：[{lo:.4f}, {hi:.4f}]")
        lines.append(f"- AUC（排除首题）：{m['auc_skip1st']:.4f}")
        lines.append(f"- 准确率：{m['acc_all']:.4f}")
        lines.append(f"- RMSE：{m['rmse_all']:.4f}　NLL：{m['nll_all']:.4f}")
        lines.append(
            f"- 逐学生 AUC 均值 ± 标准差："
            f"{m['per_student_auc_mean']} ± {m['per_student_auc_std']}"
        )
        lines.append(f"- 训练耗时：{m['train_seconds']} 秒　参数量：{m['n_params'] or '不适用'}")
        lines.append("")

    # 自动生成结论
    lines += ["## 三、结论", ""]
    ranked = sorted(
        [(n, m) for n, m in res.items() if not np.isnan(m["auc_all"])],
        key=lambda kv: kv[1]["auc_all"],
        reverse=True,
    )
    if ranked:
        best_name, best = ranked[0]
        lines.append(f"1. 表现最好的模型是 **{best_name}**（AUC {best['auc_all']:.4f}）。")
        baseline_names = [n for n in res if REGISTRY[n]["kind"] == "baseline"]
        strongest = [
            (n, res[n])
            for n in baseline_names
            if not np.isnan(res[n]["auc_all"])
        ]
        strongest.sort(key=lambda kv: kv[1]["auc_all"], reverse=True)
        if strongest:
            base_name, base_m = strongest[0]
            gain = best["auc_all"] - base_m["auc_all"]
            base_desc = "、".join(
                f"{n}（AUC {res[n]['auc_all']:.4f}）" for n, _ in strongest
            )
            lines.append(f"2. 非学习基线：{base_desc}。")
            lines.append(
                f"   其中最强的是 **{base_name}**（AUC {base_m['auc_all']:.4f}）；"
                f"学习型模型相对它的提升为 **{gain:+.4f}**。"
            )
            lines.append(
                "   "
                + (
                    "提升为正，说明引入知识状态建模确实带来了增益，模型的价值成立。"
                    if gain > 0
                    else "**提升不为正，说明在这个数据规模下模型还没有超过简单频率基线**，"
                    "需要检查数据量、训练轮次与知识点粒度——这个结论本身也是有价值的结果，"
                    "不要掩盖。"
                )
            )
        # 显著性判断
        ci_note = []
        for n, m in ranked:
            if m.get("auc_all_ci95"):
                ci_note.append((n, m["auc_all_ci95"]))
        if len(ci_note) >= 2:
            lines.append("3. 置信区间对比（判断「差距是否真实存在」）：")
            for n, (lo, hi) in ci_note:
                lines.append(f"   - {n}：[{lo:.4f}, {hi:.4f}]")
            lines.append(
                "   若两个模型的区间大幅重叠，则**不能声称二者有实质差异**——"
                "这与设计文档第三节引用的实证研究结论一致：模型间相对差异很细微，且随数据集变化。"
            )
    lines.append("")
    lines.append(
        "> 口径说明：AUC 为主指标；ACC 用 0.5 阈值；`AUC*` 为排除每条序列第一个位置的结果。"
        "两个口径都报，避免答辩时被追问口径不清。"
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
