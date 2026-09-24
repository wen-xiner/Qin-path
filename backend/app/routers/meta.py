"""元信息接口：知识点依赖图、题库、模型信息、实验结果。

前端初始化时读这些，实验结论也通过这里暴露给界面，避免"实验是实验、系统是系统"两张皮。
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Query

from backend.app.config import DATASET_PREFIX
from backend.app.deps import get_current_user, get_kt
from ml.config import ARTIFACT_DIR
from ml.data.dataset import load_splits
from ml.models.registry import REGISTRY

router = APIRouter(prefix="/api/meta", tags=["元信息"])


@router.get("", summary="系统与模型信息")
def meta(user=Depends(get_current_user)) -> dict:
    kt = get_kt()
    base = kt.meta()
    try:
        splits = load_splits(DATASET_PREFIX)
        base["data_source"] = splits["train"].data_source
        base["data_stats"] = splits["train"].stats()
    except FileNotFoundError:
        base["data_source"] = "unknown"
    base["experiment"] = load_experiment_summary()
    return base


@router.get("/knowledge-graph", summary="知识点依赖图")
def knowledge_graph(user=Depends(get_current_user)) -> dict:
    kt = get_kt()
    g = kt.graph
    return {
        "course": "数据结构",
        "nodes": [
            {
                "kc_id": kc.kc_id,
                "name": kc.name,
                "chapter": kc.chapter,
                "description": kc.description,
                "prerequisites": list(kc.prerequisites),
                "depth": g.depth_of(kc.kc_id),
            }
            for kc in g
        ],
        "layers": g.learning_layers(),
        "topological_order": g.topological_order(),
    }


@router.get("/questions", summary="题库")
def questions(
    kc_id: str | None = Query(default=None),
    user=Depends(get_current_user),
) -> list[dict]:
    kt = get_kt()
    out = []
    for q in kt.questions:
        if kc_id and q.kc_id != kc_id:
            continue
        out.append(
            {
                "question_id": q.question_id,
                "kc_id": q.kc_id,
                "kc_name": kt.graph[q.kc_id].name,
                "difficulty": q.difficulty,
                "stem": q.stem,
                "options": list(q.options),
                # 题库接口会返回答案，仅用于演示环境；真实产品要按角色裁剪
                "answer_index": q.answer_index,
            }
        )
    return out


@router.get("/models", summary="模型清单与实验指标")
def models(user=Depends(get_current_user)) -> dict:
    return {
        "registry": {
            k: {"kind": v["kind"], "desc": v["desc"]} for k, v in REGISTRY.items()
        },
        "experiments": load_experiment_summary(),
    }


def load_experiment_summary() -> dict | None:
    """读取最近一次对比实验的指标（ml/artifacts/results_*.json）。

    取 mtime 最新的那个文件，所以在真实数据集上跑完实验后，
    界面会自动从「合成数据」切到公开数据集，不用改任何前端代码。
    """
    files = sorted(ARTIFACT_DIR.glob("results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        meta = payload.get("meta", {})
        res = payload.get("results", {})
        split_stats = {k: meta.get(k, {}) or {} for k in ("train", "valid", "test")}
        return {
            "source_file": f.name,
            "data_source": meta.get("data_source"),
            "is_synthetic": meta.get("data_source") == "synthetic",
            "device": meta.get("device"),
            "profile": meta.get("profile"),
            "n_kc": meta.get("n_kc"),
            "n_students": sum(s.get("n_students", 0) for s in split_stats.values()),
            "n_interactions": sum(s.get("n_interactions", 0) for s in split_stats.values()),
            "splits": {
                k: {
                    "n_students": s.get("n_students"),
                    "n_interactions": s.get("n_interactions"),
                }
                for k, s in split_stats.items()
            },
            "table": [
                {
                    "model": name,
                    "auc": m.get("auc_all"),
                    "acc": m.get("acc_all"),
                    "rmse": m.get("rmse_all"),
                    "nll": m.get("nll_all"),
                    "auc_ci95": m.get("auc_all_ci95"),
                    "auc_skip1st": m.get("auc_skip1st"),
                    "seconds": m.get("train_seconds"),
                    "params": m.get("n_params"),
                }
                for name, m in res.items()
            ],
        }
    return None
