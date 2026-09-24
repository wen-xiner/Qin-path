"""学生端接口：做题 → 掌握度更新 → 推荐 → 看理由，形成完整闭环。

这条闭环对应设计文档第七节的"学生端实现完整闭环：登录、答题、掌握度更新、接收推荐、查看推荐理由"。
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.deps import get_current_user, get_kt
from backend.app.models import AnswerRecord, RecommendationLog, User
from backend.app.schemas import (
    AnswerIn,
    AnswerOut,
    CurveOut,
    CurvePoint,
    Evidence,
    HistoryItem,
    MasteryItem,
    MasteryOut,
    QuestionOut,
    RecommendationOut,
)
from backend.app.services.history import answered_question_ids, load_sequence, next_seq_no
from ml.config import MASTERY_THRESHOLD

router = APIRouter(prefix="/api/student", tags=["学生端"])


def _question_lookup():
    kt = get_kt()
    return {q.question_id: q for q in kt.questions}, kt


def _to_mastery_out(kc_seq: list[int], resp_seq: list[int]) -> MasteryOut:
    kt = get_kt()
    graph = kt.graph
    ev = kt.mastery_with_evidence(kc_seq, resp_seq)
    plan = kt.plan(kc_seq, resp_seq, horizon=5)

    items = []
    for i, kc_id in enumerate(graph.kc_ids):
        m = ev["mastery"][i]
        att = int(ev["attempts"][i])
        mval = None if np.isnan(m) else round(float(m), 4)
        if att == 0:
            state = "unobserved"
        elif mval is not None and mval >= MASTERY_THRESHOLD:
            state = "mastered"
        else:
            state = "weak"
        items.append(
            MasteryItem(
                kc_id=kc_id,
                name=graph[kc_id].name,
                chapter=graph[kc_id].chapter,
                mastery=mval,
                attempts=att,
                correct=int(ev["correct"][i]),
                state=state,
            )
        )

    return MasteryOut(
        threshold=MASTERY_THRESHOLD,
        n_kc=len(graph),
        n_observed=int(sum(1 for it in items if it.state != "unobserved")),
        items=items,
        plan=plan,
    )


def _build_recommendation(user_id: int, db: Session) -> RecommendationOut | None:
    kt = get_kt()
    kc_seq, resp_seq, _ = load_sequence(db, user_id)
    rec = kt.recommend_question(
        kc_seq,
        resp_seq,
        answered_question_ids=answered_question_ids(db, user_id),
    )
    if rec is None:
        return None

    q = rec["question"]
    db.add(
        RecommendationLog(
            user_id=user_id,
            kc_id=q["kc_id"],
            question_id=q["question_id"],
            action=rec["action"],
            reason=rec["reason"][:500],
            mastery=rec["evidence"]["mastery"],
        )
    )
    db.commit()

    ev = rec["evidence"]
    return RecommendationOut(
        question=QuestionOut(**q),
        action=rec["action"],
        reason=rec["reason"],
        evidence=Evidence(
            mastery=ev["mastery"],
            attempts=ev["attempts"],
            correct=ev["correct"],
            threshold=ev["threshold"],
            predicted_correct_prob=ev["predicted_correct_prob"],
            blocked_by=ev["blocked_by"],
        ),
    )


@router.get("/mastery", response_model=MasteryOut, summary="我的掌握度与学习路径")
def my_mastery(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> MasteryOut:
    kc_seq, resp_seq, _ = load_sequence(db, user.id)
    return _to_mastery_out(kc_seq, resp_seq)


@router.get("/next-question", response_model=RecommendationOut | None, summary="获取下一道推荐题")
def next_question(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> RecommendationOut | None:
    return _build_recommendation(user.id, db)


@router.post("/answer", response_model=AnswerOut, summary="提交答案")
def submit_answer(
    payload: AnswerIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> AnswerOut:
    questions, kt = _question_lookup()
    q = questions.get(payload.question_id)
    if q is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"题目 {payload.question_id} 不存在")

    # 作答前的掌握度
    kc_seq, resp_seq, _ = load_sequence(db, user.id)
    graph = kt.graph
    kc_i = graph.kc_ids.index(q.kc_id)
    ev_before = kt.mastery_with_evidence(kc_seq, resp_seq)
    m_before = ev_before["mastery"][kc_i]
    m_before = None if np.isnan(m_before) else float(m_before)

    is_correct = payload.chosen_index == q.answer_index

    # 作答后重算（把本次记录也算进去）
    kc_seq_after = kc_seq + [kc_i]
    resp_seq_after = resp_seq + [int(is_correct)]
    ev_after = kt.mastery_with_evidence(kc_seq_after, resp_seq_after)
    m_after = ev_after["mastery"][kc_i]
    m_after = None if np.isnan(m_after) else float(m_after)

    db.add(
        AnswerRecord(
            user_id=user.id,
            question_id=q.question_id,
            kc_id=q.kc_id,
            chosen_index=payload.chosen_index,
            is_correct=is_correct,
            mastery_before=m_before,
            mastery_after=m_after,
            seq_no=next_seq_no(db, user.id),
        )
    )
    db.commit()

    delta = None
    if m_before is not None and m_after is not None:
        delta = round(m_after - m_before, 4)

    if is_correct:
        feedback = f"回答正确。「{graph[q.kc_id].name}」的掌握概率"
        feedback += f"从 {m_before:.2f} 变为 {m_after:.2f}。" if m_before is not None else f"更新为 {m_after:.2f}。"
    else:
        feedback = f"回答错误。正确答案是「{q.answer_text}」。「{graph[q.kc_id].name}」的掌握概率"
        feedback += f"从 {m_before:.2f} 变为 {m_after:.2f}。" if m_before is not None else f"更新为 {m_after:.2f}。"

    return AnswerOut(
        is_correct=is_correct,
        correct_index=q.answer_index,
        kc_id=q.kc_id,
        kc_name=graph[q.kc_id].name,
        mastery_before=None if m_before is None else round(m_before, 4),
        mastery_after=None if m_after is None else round(m_after, 4),
        mastery_delta=delta,
        next=_build_recommendation(user.id, db),
        feedback=feedback,
    )


@router.get("/curve", response_model=CurveOut, summary="掌握度变化曲线")
def curve(
    kc_id: str | None = Query(default=None, description="指定知识点则只看该知识点"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CurveOut:
    kt = get_kt()
    kc_seq, resp_seq, _ = load_sequence(db, user.id)
    if not kc_seq:
        return CurveOut(kc_id=kc_id, points=[])
    data = kt.learning_curve(kc_seq, resp_seq, kc_id=kc_id)
    return CurveOut(
        kc_id=kc_id, points=[CurvePoint(**p) for p in data["points"]]
    )


@router.get("/history", response_model=list[HistoryItem], summary="我的作答记录")
def history(
    limit: int = Query(default=50, ge=1, le=500),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[HistoryItem]:
    kc_seq, resp_seq, records = load_sequence(db, user.id)
    graph = get_kt().graph
    out = []
    for r in records[-limit:][::-1]:
        out.append(
            HistoryItem(
                seq_no=r.seq_no,
                question_id=r.question_id,
                kc_id=r.kc_id,
                kc_name=graph[r.kc_id].name if r.kc_id in graph else r.kc_id,
                is_correct=bool(r.is_correct),
                mastery_after=r.mastery_after,
                created_at=r.created_at,
            )
        )
    return out
