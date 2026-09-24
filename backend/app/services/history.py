"""把数据库里的作答记录还原成模型的输入序列。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import AnswerRecord
from ml.data.knowledge_graph import get_knowledge_graph


def load_sequence(db: Session, user_id: int) -> tuple[list[int], list[int], list[AnswerRecord]]:
    """返回 (kc_seq, resp_seq, records)，按作答顺序排列。

    模型内部用知识点下标（0..n_kc-1），所以这里把 kc_id 转成下标。
    """
    graph = get_knowledge_graph()
    kc_index = {k: i for i, k in enumerate(graph.kc_ids)}
    records = list(
        db.scalars(
            select(AnswerRecord)
            .where(AnswerRecord.user_id == user_id)
            .order_by(AnswerRecord.seq_no.asc(), AnswerRecord.id.asc())
        )
    )
    kc_seq, resp_seq = [], []
    for r in records:
        if r.kc_id in kc_index:
            kc_seq.append(kc_index[r.kc_id])
            resp_seq.append(int(r.is_correct))
    return kc_seq, resp_seq, records


def answered_question_ids(db: Session, user_id: int) -> set[str]:
    return set(
        db.scalars(
            select(AnswerRecord.question_id).where(AnswerRecord.user_id == user_id)
        )
    )


def next_seq_no(db: Session, user_id: int) -> int:
    last = db.scalar(
        select(AnswerRecord.seq_no)
        .where(AnswerRecord.user_id == user_id)
        .order_by(AnswerRecord.seq_no.desc())
        .limit(1)
    )
    return int(last or 0) + 1
