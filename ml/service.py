"""在线推理服务层：把训练好的模型 + 知识点图 + 题库 + 路径规划串成一条链路。

后端（FastAPI）只调用这里，不直接碰模型细节。
可解释推荐的所有理由模板也集中在这里，保证"为什么推荐"这一段是可核查、可复现的。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.config import ARTIFACT_DIR, DEVICE, MASTERY_THRESHOLD, get_profile
from ml.data.dataset import merge_sequences
from ml.data.dataset import load_splits
from ml.data.knowledge_graph import KnowledgeGraph, get_knowledge_graph
from ml.data.question_bank import Question, build_question_bank
from ml.models.base import KnowledgeTracer
from ml.models.bkt import BKT
from ml.models.dkt import DKT
from ml.models.sakt import SAKT
from ml.planner import plan_path


class KTService:
    """知识追踪在线服务。

    默认用 BKT 产出"掌握度"（逐知识点后验，可解释、可增量更新），
    用 DKT/SAKT 产出"下一题答对概率"（序列建模能力更强）。
    两个数字分工不同，接口里都返回，避免把两件事混成一个数。
    """

    def __init__(
        self,
        model: KnowledgeTracer,
        graph: KnowledgeGraph | None = None,
        questions: list[Question] | None = None,
        model_name: str = "BKT",
    ):
        self.model = model
        self.model_name = model_name
        self.graph = graph or get_knowledge_graph()
        self.questions = questions or build_question_bank()
        self.kc_index = {k: i for i, k in enumerate(self.graph.kc_ids)}
        self.by_kc: dict[str, list[Question]] = {}
        for q in self.questions:
            self.by_kc.setdefault(q.kc_id, []).append(q)

    # ------------------------------------------------------------ 构建
    @classmethod
    def create(
        cls,
        model_name: str = "BKT",
        dataset_prefix: str = "dataset",
        artifact: str | Path | None = None,
    ) -> "KTService":
        """加载模型产物。找不到产物时，就用训练集现跑一个（首次启动的自举路径）。"""
        graph = get_knowledge_graph()
        questions = build_question_bank()
        splits = load_splits(dataset_prefix)
        n_kc = splits["train"].n_kc

        path = Path(artifact) if artifact else ARTIFACT_DIR / "models" / f"{model_name}_synthetic"
        # 产物可能是 BKT_x.json / DKT_x.pt，也可能是历史遗留的 BKT_x（无后缀），三种都要认。
        params_path: Path | None = None
        if path.is_file():
            params_path = path
        else:
            for suffix in (".json", ".pt", ""):
                candidate = path.with_suffix(suffix) if suffix else path
                if candidate.is_file():
                    params_path = candidate
                    break
            if params_path is None:
                for p in sorted(path.parent.glob(f"{path.name}.*")):
                    if p.is_file():
                        params_path = p
                        break

        model: KnowledgeTracer
        if params_path is not None and params_path.exists():
            print(f"[KTService] 加载模型产物：{params_path.name}")
            if model_name == "BKT":
                model = BKT.load(params_path)
            elif model_name == "DKT":
                model = DKT(n_kc=n_kc, device=DEVICE)
                DKT.load_weights(model.model, params_path)
            elif model_name == "SAKT":
                model = SAKT(n_kc=n_kc, device=DEVICE)
                SAKT.load_weights(model.model, params_path)
            else:
                raise ValueError(f"不支持在服务层加载的模型：{model_name}")
        else:
            # 自举：直接训练一个（首次部署、还没跑过实验时会走到这里）
            print(f"[KTService] 未找到 {model_name} 的产物，现场训练一个（会慢几秒到几十秒）")
            from ml.models.registry import build_model

            model = build_model(model_name, n_kc=n_kc, device=DEVICE)
            model.fit(splits["train"], splits["valid"])

        return cls(model=model, graph=graph, questions=questions, model_name=model_name)

    # ------------------------------------------------------------ 掌握度
    def mastery_with_evidence(self, kc_seq: list[int], resp_seq: list[int]) -> dict:
        """返回逐知识点掌握度 + 作答证据。未观测的知识点返回 NaN。"""
        n_kc = len(self.graph)
        if not kc_seq:
            return {
                "mastery": np.full(n_kc, np.nan),
                "attempts": np.zeros(n_kc, dtype=int),
                "correct": np.zeros(n_kc, dtype=int),
            }
        if isinstance(self.model, BKT):
            return self.model.mastery_with_evidence(kc_seq, resp_seq, n_kc)

        # 深度模型：用逐位置预测概率近似掌握度（按知识点取最后一次预测）
        mastery = self.model.mastery(kc_seq, resp_seq, n_kc)
        attempts = np.zeros(n_kc, dtype=int)
        correct = np.zeros(n_kc, dtype=int)
        for kc, r in zip(kc_seq, resp_seq):
            if 0 <= kc < n_kc:
                attempts[kc] += 1
                correct[kc] += int(r)
        return {"mastery": mastery, "attempts": attempts, "correct": correct}

    def plan(self, kc_seq: list[int], resp_seq: list[int], horizon: int = 5) -> dict:
        ev = self.mastery_with_evidence(kc_seq, resp_seq)
        plan = plan_path(
            ev["mastery"], ev["attempts"], ev["correct"], graph=self.graph, horizon=horizon
        )
        plan["mastery_vector"] = [
            None if np.isnan(v) else round(float(v), 4) for v in ev["mastery"]
        ]
        return plan

    # ------------------------------------------------------------ 推荐
    def recommend_question(
        self,
        kc_seq: list[int],
        resp_seq: list[int],
        answered_question_ids: set[str] | None = None,
        prefer_difficulty: int | None = None,
    ) -> dict | None:
        """按规划给出的第一个目标知识点挑一道题，并给出可解释理由。"""
        plan = self.plan(kc_seq, resp_seq, horizon=1)
        target = plan.get("next")
        if target is None:
            return None

        kc_id = target["kc_id"]
        pool = self.by_kc.get(kc_id, [])
        if not pool:
            return None

        answered = answered_question_ids or set()
        fresh = [q for q in pool if q.question_id not in answered]
        candidates = fresh or pool
        if prefer_difficulty is not None:
            candidates = sorted(candidates, key=lambda q: abs(q.difficulty - prefer_difficulty)) 
        else:
            candidates = sorted(candidates, key=lambda q: q.difficulty)
        q = candidates[0]

        # 关键：这里的概率来自"学生自己的作答历史"，不是模型的猜测值
        p_next = None
        if kc_seq:
            try:
                p_next = float(self.model.predict_next(kc_seq, resp_seq, self.kc_index[kc_id]))
            except Exception:
                p_next = None

        return {
            "question": {
                "question_id": q.question_id,
                "kc_id": q.kc_id,
                "kc_name": self.graph[q.kc_id].name,
                "difficulty": q.difficulty,
                "stem": q.stem,
                "options": list(q.options),
            },
            "action": target["action"],
            "reason": self.explain(target, p_next, answered_count=len(answered)),
            "evidence": {
                "mastery": target["mastery"],
                "attempts": target["attempts"],
                "correct": target["correct"],
                "threshold": MASTERY_THRESHOLD,
                "predicted_correct_prob": None if p_next is None else round(p_next, 4),
                "blocked_by": target["blocked_by"],
            },
            "plan": plan,
        }

    @staticmethod
    def explain(target: dict, p_next: float | None, answered_count: int = 0) -> str:
        """可解释推荐的核心：把"为什么"拼成一句可核查的话。

        模板刻意保持"结论 + 依据 + 阈值"三段式，答辩时可以直接指着这句话念，
        每个数字都能在系统里查到出处。

        注意：分支必须按 **action** 走，不能按 mastery 是否为空走。
        「未观测 + 推进」和「未观测 + 先补前置」的 mastery 都是空，但说的是两件不同的事
        （前者是"可以学了"，后者是"它是别人的前置，得先补上"），混在一起会出现
        「动作说补前置、理由说推进新知识点」的自相矛盾。
        """
        name = target["name"]
        m = target["mastery"]
        thr = MASTERY_THRESHOLD
        act = target["action"]
        acts = {
            "remedy": "补强",
            "remedy_prerequisite": "先补前置",
            "advance": "推进新知识点",
        }
        action_cn = acts.get(act, act)
        blocked = target.get("blocked_by") or []

        if act == "remedy":
            head = (
                f"推荐这道题，因为它是「{name}」这个知识点的练习，"
                f"而你在这个知识点上的掌握概率为 {(m or 0):.2f}，低于 {thr:.2f} 的阈值"
            )
        elif act == "remedy_prerequisite":
            if m is None:
                head = (
                    f"推荐这道题，它是「{name}」的练习。你还没有这个知识点的作答记录，"
                    f"但它被更靠后的知识点依赖，前置不达标就不能往下学，所以需要先把它补上"
                )
            else:
                head = (
                    f"推荐这道题，它是「{name}」的练习。它是后续知识点的前置，"
                    f"而你在这个知识点上的掌握概率只有 {m:.2f}，低于 {thr:.2f} 的阈值，"
                    f"补上它才能解锁后面的内容"
                )
        else:  # advance
            head = (
                f"推荐这道题，它是「{name}」的练习。你在这个知识点上还没有作答记录，"
                f"而它的前置知识点已经达标，属于当前可以开始学习的内容"
            )

        tail = [f"作答依据：已作答 {target['attempts']} 次、答对 {target['correct']} 次"]
        if p_next is not None:
            tail.append(f"模型预测你答对这道题的概率约为 {p_next:.2f}")
        if answered_count:
            tail.append(f"已避开你做过的 {answered_count} 道题")

        return f"{head}（{action_cn}）。" + "；".join(tail) + "。"

    # ------------------------------------------------------------ 教师端
    def class_mastery(
        self, students: list[dict], graph: KnowledgeGraph | None = None
    ) -> dict:
        """班级掌握度分布。

        students: [{'student_id':.., 'kc_seq':[...], 'resp_seq':[...]}, ...]
        """
        graph = graph or self.graph
        n_kc = len(graph)
        rows = []
        for s in students:
            ev = self.mastery_with_evidence(s["kc_seq"], s["resp_seq"])
            rows.append(ev["mastery"])
        if not rows:
            return {"kc": [], "per_student": []}
        mat = np.vstack(rows)

        kc_stats = []
        for i, kc_id in enumerate(graph.kc_ids):
            col = mat[:, i]
            observed = col[~np.isnan(col)]
            kc_stats.append(
                {
                    "kc_id": kc_id,
                    "name": graph[kc_id].name,
                    "chapter": graph[kc_id].chapter,
                    "n_observed": int(len(observed)),
                    "n_students": int(mat.shape[0]),
                    "mean_mastery": round(float(observed.mean()), 4) if len(observed) else None,
                    "n_below": int((observed < MASTERY_THRESHOLD).sum()),
                    "weak_ratio": round(float((observed < MASTERY_THRESHOLD).mean()), 4)
                    if len(observed)
                    else None,
                }
            )

        # 结构性问题定位：班级整体卡住的知识点（观测人数够、均值低）
        weak = [
            k
            for k in kc_stats
            if k["mean_mastery"] is not None
            and k["n_observed"] >= max(3, 0.3 * mat.shape[0])
            and k["mean_mastery"] < MASTERY_THRESHOLD
        ]
        weak.sort(key=lambda k: k["mean_mastery"])

        return {
            "n_students": int(mat.shape[0]),
            "threshold": MASTERY_THRESHOLD,
            "kc": kc_stats,
            "class_weak_points": weak,   # 对应设计文档"这个班在第 3 章整体卡住了"
            "per_student": [
                {
                    "student_id": students[i].get("student_id", i),
                    "mastery": [None if np.isnan(v) else round(float(v), 4) for v in mat[i]],
                }
                for i in range(mat.shape[0])
            ],
        }

    def learning_curve(self, kc_seq: list[int], resp_seq: list[int], kc_id: str | None = None) -> dict:
        """掌握度随时间的变化曲线（前端折线图用）。

        指定 kc_id 时只跟踪该知识点；否则返回"平均掌握度"随作答次数的变化。
        """
        target_idx = self.kc_index[kc_id] if kc_id else None
        points = []
        for t in range(1, len(kc_seq) + 1):
            ev = self.mastery_with_evidence(kc_seq[:t], resp_seq[:t])
            m = ev["mastery"]
            if target_idx is not None:
                val = m[target_idx]
                value = None if np.isnan(val) else round(float(val), 4)
            else:
                obs = m[~np.isnan(m)]
                value = round(float(obs.mean()), 4) if len(obs) else None
            points.append(
                {
                    "step": t,
                    "kc_id": kc_id if target_idx is not None else self.graph.kc_ids[kc_seq[t - 1]],
                    "correct": int(resp_seq[t - 1]),
                    "mastery": value,
                }
            )
        return {"kc_id": kc_id, "points": points}

    # ------------------------------------------------------------ 元信息
    def meta(self) -> dict:
        return {
            "model": self.model_name,
            "device": DEVICE,
            "profile": get_profile().name,
            "n_kc": len(self.graph),
            "n_questions": len(self.questions),
            "mastery_threshold": MASTERY_THRESHOLD,
        }


if __name__ == "__main__":
    svc = KTService.create("BKT")
    print(json.dumps(svc.meta(), ensure_ascii=False, indent=2))
