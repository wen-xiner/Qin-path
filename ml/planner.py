"""掌握度 → 学习路径的显式规划。

设计文档第五节的原文要求："规则本身是显式的：若某知识点的前置知识点掌握度低于阈值，
则优先补前置。这条规则的可解释性远强于让大模型自由推荐，也更经得起追问。"

所以这里**没有大模型参与**，规则全部写死、可复现、可逐条追问：
1. 若某知识点自己没掌握，且它的某些前置也没掌握 → 先补前置（remedy_prerequisite）
2. 若某知识点自己没掌握，但前置都达标 → 直接补它（remedy）
3. 若某知识点从未作答，且前置都达标 → 推进新知识点（advance）
4. 从未作答的知识点**不视为薄弱**，只视为"未知"——这是容易出错的地方，
   把没做过当成不会，会把大批没学到的知识点当成漏洞，推荐就乱了

排序打分（可解释的线性组合）：
    priority = 缺口(1 - mastery) * 0.6
             + 阻塞影响力(它卡住了多少个下游知识点) * 0.3
             + 学习顺序加成(越靠前越优先) * 0.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ml.config import MASTERY_THRESHOLD, PREREQ_THRESHOLD
from ml.data.knowledge_graph import KnowledgeGraph, get_knowledge_graph


@dataclass
class PlanItem:
    kc_id: str
    name: str
    chapter: str
    action: str                  # remedy / remedy_prerequisite / advance
    mastery: float | None        # None 表示未观测
    attempts: int
    correct: int
    priority: float
    reason: str
    blocked_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kc_id": self.kc_id,
            "name": self.name,
            "chapter": self.chapter,
            "action": self.action,
            "mastery": None if self.mastery is None else round(float(self.mastery), 4),
            "attempts": self.attempts,
            "correct": self.correct,
            "priority": round(float(self.priority), 4),
            "reason": self.reason,
            "blocked_by": self.blocked_by,
        }


def _downstream_count(graph: KnowledgeGraph, kc_id: str) -> int:
    """该知识点「卡住」了多少个下游知识点（传递闭包大小），代表修复它的杠杆率。"""
    seen: set[str] = set()
    stack = list(graph.children_of(kc_id))
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(graph.children_of(cur))
    return len(seen)


def plan_path(
    mastery: np.ndarray,
    attempts: np.ndarray,
    correct: np.ndarray | None = None,
    graph: KnowledgeGraph | None = None,
    threshold: float = MASTERY_THRESHOLD,
    prereq_threshold: float = PREREQ_THRESHOLD,
    horizon: int = 5,
) -> dict:
    """生成学习路径。

    mastery / attempts 的长度都必须等于知识点个数，下标与 graph.kc_ids 对齐。
    mastery 中的 NaN 表示该知识点没有被观测过。
    """
    graph = graph or get_knowledge_graph()
    mastery = np.asarray(mastery, dtype=float)
    attempts = np.asarray(attempts, dtype=int)
    if correct is None:
        correct = np.zeros_like(attempts)
    correct = np.asarray(correct, dtype=int)

    kc_ids = graph.kc_ids
    idx = {k: i for i, k in enumerate(kc_ids)}
    n = len(kc_ids)
    order = {k: i for i, k in enumerate(graph.topological_order())}

    def m_of(k: str) -> float | None:
        v = mastery[idx[k]]
        return None if np.isnan(v) else float(v)

    def is_ok(k: str) -> bool:
        """前置是否达标。未观测的前置不算达标（不能假设学生会）。"""
        v = m_of(k)
        return v is not None and v >= prereq_threshold

    candidate_scores: list[tuple[float, PlanItem]] = []
    blocked_items: list[dict] = []

    for k in kc_ids:
        i = idx[k]
        m = m_of(k)
        att = int(attempts[i])
        cor = int(correct[i])
        weak_prereqs = [p for p in graph[k].prerequisites if not is_ok(p)]
        downs = _downstream_count(graph, k)
        order_bonus = 1.0 - (order[k] / max(n - 1, 1))

        if m is None:
            # 从未作答：只有前置都达标才作为"推进"候选，否则算被阻塞
            if weak_prereqs:
                blocked_items.append(
                    {
                        "kc_id": k,
                        "name": graph[k].name,
                        "blocked_by": weak_prereqs,
                        "reason": (
                            f"尚未接触，且前置知识点"
                            f"{'、'.join(graph[p].name for p in weak_prereqs)}尚未达标，暂不推进"
                        ),
                    }
                )
                continue
            # 未观测的不会去补，走"推进"
            priority = 0.35 * 0.6 + downs / max(n, 1) * 0.3 + order_bonus * 0.1
            reason = (
                f"该知识点你还没有练习记录，而它的前置（"
                f"{'、'.join(graph[p].name for p in graph[k].prerequisites) or '无'}）都已达标，"
                f"适合作为下一步新学的知识点"
            )
            candidate_scores.append(
                (
                    priority,
                    PlanItem(
                        kc_id=k,
                        name=graph[k].name,
                        chapter=graph[k].chapter,
                        action="advance",
                        mastery=None,
                        attempts=att,
                        correct=cor,
                        priority=priority,
                        reason=reason,
                    ),
                )
            )
            continue

        if m < threshold:
            if weak_prereqs:
                # 自己弱、前置也弱 → 先补前置（把前置本身作为候选，这里只记录阻塞关系）
                blocked_items.append(
                    {
                        "kc_id": k,
                        "name": graph[k].name,
                        "blocked_by": weak_prereqs,
                        "mastery": round(m, 4),
                        "reason": (
                            f"掌握概率 {m:.2f} 低于阈值 {threshold:.2f}，但它的前置"
                            f"{'、'.join(graph[p].name for p in weak_prereqs)}同样没达标，"
                            f"修复顺序应该反过来"
                        ),
                    }
                )
                # 仍然把它的薄弱前置推为候选
                for p in weak_prereqs:
                    pm = m_of(p)
                    pdowns = _downstream_count(graph, p)
                    # 前置的杠杆率更高：补它能同时解锁多个下游
                    priority = (
                        (1.0 - (pm if pm is not None else 0.3)) * 0.6
                        + pdowns / max(n, 1) * 0.3
                        + (1.0 - order[p] / max(n - 1, 1)) * 0.1
                    )
                    reason = (
                        f"它是「{graph[k].name}」的前置知识点"
                        + (f"，当前掌握概率 {pm:.2f}" if pm is not None else "，尚无练习记录")
                        + f"，补上它可以解锁后续 {pdowns} 个知识点"
                    )
                    candidate_scores.append(
                        (
                            priority,
                            PlanItem(
                                kc_id=p,
                                name=graph[p].name,
                                chapter=graph[p].chapter,
                                action="remedy_prerequisite",
                                mastery=pm,
                                attempts=int(attempts[idx[p]]),
                                correct=int(correct[idx[p]]),
                                priority=priority,
                                reason=reason,
                                blocked_by=[],
                            ),
                        )
                    )
                continue
            # 直接补这个知识点
            priority = (1.0 - m) * 0.6 + downs / max(n, 1) * 0.3 + order_bonus * 0.1
            reason = (
                f"当前掌握概率 {m:.2f}，低于阈值 {threshold:.2f}"
                f"（已作答 {att} 次，答对 {cor} 次），属于需要补强的最短路径"
            )
            candidate_scores.append(
                (
                    priority,
                    PlanItem(
                        kc_id=k,
                        name=graph[k].name,
                        chapter=graph[k].chapter,
                        action="remedy",
                        mastery=m,
                        attempts=att,
                        correct=cor,
                        priority=priority,
                        reason=reason,
                    ),
                )
            )

    # 去重（同一个前置可能被多个下游重复推上来），保留优先级最高的一条
    best: dict[str, tuple[float, PlanItem]] = {}
    for score, item in candidate_scores:
        if item.kc_id not in best or score > best[item.kc_id][0]:
            best[item.kc_id] = (score, item)
    ranked = sorted(best.values(), key=lambda t: t[0], reverse=True)

    queue = [item.to_dict() for _, item in ranked[:horizon]]
    observed = int(np.sum(~np.isnan(mastery)) if mastery.size else 0)

    return {
        "threshold": threshold,
        "prereq_threshold": prereq_threshold,
        "n_kc": n,
        "n_observed": observed,
        "next": queue[0] if queue else None,
        "queue": queue,
        "blocked": blocked_items,
        "diagnosis": _diagnose(mastery, attempts, graph, threshold),
    }


def _diagnose(mastery: np.ndarray, attempts: np.ndarray, graph: KnowledgeGraph, threshold: float) -> dict:
    weakest, strongest = [], []
    for i, k in enumerate(graph.kc_ids):
        m = mastery[i]
        if np.isnan(m) or attempts[i] == 0:
            continue
        (weakest if m < threshold else strongest).append((graph[k].name, round(float(m), 4)))
    weakest.sort(key=lambda t: t[1])
    strongest.sort(key=lambda t: -t[1])
    return {
        "weakest": weakest[:5],
        "strongest": strongest[:5],
        "n_weak": len(weakest),
        "n_strong": len(strongest),
    }
