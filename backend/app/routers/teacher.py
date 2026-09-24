"""教师端接口：班级掌握度分布 + 共性薄弱点定位。

对应设计文档第二节的痛点之一："老师看到的是全班平均分，看不到『这个班在第 3 章整体卡住了』
这类结构性问题"。所以这里除了均值，还专门给一个 class_weak_points。
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.deps import get_kt, require_teacher
from backend.app.models import Classroom, User
from backend.app.schemas import ClassMasteryOut, KCStat
from backend.app.services.history import load_sequence

router = APIRouter(prefix="/api/teacher", tags=["教师端"])


def _students_of(db: Session, teacher: User) -> list[User]:
    if teacher.class_id is None:
        return []
    return list(
        db.scalars(select(User).where(User.class_id == teacher.class_id, User.role == "student"))
    )


def _class_mastery(db: Session, teacher: User) -> ClassMasteryOut:
    if teacher.class_id is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "当前教师账号尚未绑定班级")
    cls = db.get(Classroom, teacher.class_id)
    kt = get_kt()

    payload = []
    for s in _students_of(db, teacher):
        kc_seq, resp_seq, _ = load_sequence(db, s.id)
        payload.append({"student_id": s.id, "kc_seq": kc_seq, "resp_seq": resp_seq})

    data = kt.class_mastery(payload)
    return ClassMasteryOut(
        class_id=cls.id,
        class_name=cls.name,
        n_students=data.get("n_students", 0),
        threshold=data.get("threshold", 0.7),
        kc=[KCStat(**k) for k in data.get("kc", [])],
        class_weak_points=[KCStat(**k) for k in data.get("class_weak_points", [])],
        per_student=data.get("per_student", []),
    )


@router.get("/class-mastery", response_model=ClassMasteryOut, summary="班级掌握度分布")
def class_mastery(teacher: User = Depends(require_teacher), db: Session = Depends(get_db)):
    return _class_mastery(db, teacher)


@router.get("/students", summary="班级学生列表")
def students(teacher: User = Depends(require_teacher), db: Session = Depends(get_db)) -> list[dict]:
    kt = get_kt()
    threshold = kt.meta()["mastery_threshold"]
    out = []
    for s in _students_of(db, teacher):
        kc_seq, resp_seq, _ = load_sequence(db, s.id)
        ev = kt.mastery_with_evidence(kc_seq, resp_seq)
        obs = ev["mastery"][~np.isnan(ev["mastery"])]
        out.append(
            {
                "student_id": s.id,
                "username": s.username,
                "display_name": s.display_name,
                "n_answered": len(resp_seq),
                "accuracy": round(sum(resp_seq) / len(resp_seq), 4) if resp_seq else None,
                "mean_mastery": round(float(obs.mean()), 4) if len(obs) else None,
                "n_weak": int((obs < threshold).sum()) if len(obs) else 0,
            }
        )
    return out


@router.get("/weak-points", summary="班级共性薄弱点")
def weak_points(
    teacher: User = Depends(require_teacher), db: Session = Depends(get_db)
) -> list[dict]:
    data = _class_mastery(db, teacher)
    return [k.model_dump() for k in data.class_weak_points]
