"""演示数据初始化。

用法：
    cd Qin-path
    python -m backend.app.seed              # 幂等，已存在则跳过
    python -m backend.app.seed --reset      # 清空重建

创建内容：
- 1 个班级「数据结构 2026 级 1 班」
- 1 个教师账号 teacher / 123456
- 12 个学生账号 stu01..stu12 / 123456
- 每个学生预填一段作答历史，让教师端一进去就有数据可看

预填的作答是按"模拟掌握度"生成的，不是真实学生数据，界面上会标注为演示数据。
"""

from __future__ import annotations

import argparse

import numpy as np
from sqlalchemy import delete, select

from backend.app.database import SessionLocal, init_db
from backend.app.models import AnswerRecord, Classroom, RecommendationLog, User
from backend.app.security import hash_password
from backend.app.services.history import load_sequence
from ml.data.knowledge_graph import get_knowledge_graph
from ml.data.question_bank import build_question_bank

DEFAULT_PASSWORD = "123456"
CLASS_NAME = "数据结构 2026 级 1 班"


def _simulate_history(student_idx: int, n_items: int, rng: np.random.Generator) -> list[dict]:
    """模拟一名学生的作答过程，返回 [{question_id, chosen_index, is_correct}, ...]。"""
    graph = get_knowledge_graph()
    bank = build_question_bank()
    by_kc: dict[str, list] = {}
    for q in bank:
        by_kc.setdefault(q.kc_id, []).append(q)

    order = graph.topological_order()
    # 每个学生的进度与基础不同
    progress = float(rng.uniform(0.3, 0.85))
    reachable = order[: max(3, int(np.ceil(progress * len(order))))]
    theta = {kc: float(rng.uniform(0.10, 0.55)) for kc in reachable}

    out: list[dict] = []
    current = reachable[int(rng.integers(0, len(reachable)))]
    for _ in range(n_items):
        cands = by_kc[current]
        q = cands[int(rng.integers(0, len(cands)))]
        p = theta[current] * (1 - 0.12) + (1 - theta[current]) * 0.18
        correct = bool(rng.random() < p)
        chosen = q.answer_index if correct else (q.answer_index + 1 + int(rng.integers(0, 3))) % 4
        out.append({"question_id": q.question_id, "chosen_index": chosen, "is_correct": correct})

        # 学习
        theta[current] = min(0.99, theta[current] + (0.5 if correct else 0.15) * (1 - theta[current]) * 0.25)

        if rng.random() < 0.35:
            current = reachable[int(rng.integers(0, len(reachable)))]
    return out


def seed(reset: bool = False, n_students: int = 12, n_items: int = 45, seed_value: int = 7) -> dict:
    init_db()
    db = SessionLocal()
    try:
        if reset:
            db.execute(delete(RecommendationLog))
            db.execute(delete(AnswerRecord))
            db.execute(delete(User))
            db.execute(delete(Classroom))
            db.commit()

        cls = db.scalar(select(Classroom).where(Classroom.name == CLASS_NAME))
        if cls is None:
            cls = Classroom(name=CLASS_NAME, course="数据结构")
            db.add(cls)
            db.commit()
            db.refresh(cls)

        teacher = db.scalar(select(User).where(User.username == "teacher"))
        if teacher is None:
            teacher = User(
                username="teacher",
                password_hash=hash_password(DEFAULT_PASSWORD),
                display_name="陈老师",
                role="teacher",
                class_id=cls.id,
            )
            db.add(teacher)
            db.commit()

        from ml.service import KTService

        kt = KTService.create("BKT")
        graph = kt.graph
        created = 0
        rng = np.random.default_rng(seed_value)

        for i in range(1, n_students + 1):
            username = f"stu{i:02d}"
            if db.scalar(select(User).where(User.username == username)) is not None:
                continue
            student = User(
                username=username,
                password_hash=hash_password(DEFAULT_PASSWORD),
                display_name=f"学生{i:02d}",
                role="student",
                class_id=cls.id,
            )
            db.add(student)
            db.commit()
            db.refresh(student)

            history = _simulate_history(i, n_items, rng)
            kc_seq: list[int] = []
            resp_seq: list[int] = []
            for seq_no, h in enumerate(history, start=1):
                q = next(x for x in kt.questions if x.question_id == h["question_id"])
                kc_i = graph.kc_ids.index(q.kc_id)
                m_before = kt.mastery_with_evidence(kc_seq, resp_seq)["mastery"][kc_i]
                kc_seq.append(kc_i)
                resp_seq.append(int(h["is_correct"]))
                m_after = kt.mastery_with_evidence(kc_seq, resp_seq)["mastery"][kc_i]
                db.add(
                    AnswerRecord(
                        user_id=student.id,
                        question_id=q.question_id,
                        kc_id=q.kc_id,
                        chosen_index=h["chosen_index"],
                        is_correct=h["is_correct"],
                        mastery_before=None if np.isnan(m_before) else float(m_before),
                        mastery_after=None if np.isnan(m_after) else float(m_after),
                        seq_no=seq_no,
                    )
                )
            db.commit()
            created += 1

        return {
            "class": cls.name,
            "class_id": cls.id,
            "teacher": "teacher / " + DEFAULT_PASSWORD,
            "students_created": created,
            "student_accounts": f"stu01..stu{n_students:02d} / {DEFAULT_PASSWORD}",
        }
    finally:
        db.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="知途演示数据初始化")
    ap.add_argument("--reset", action="store_true", help="清空已有数据后重建")
    ap.add_argument("--students", type=int, default=12)
    ap.add_argument("--items", type=int, default=45, help="每个学生预填的作答条数")
    args = ap.parse_args()
    info = seed(reset=args.reset, n_students=args.students, n_items=args.items)
    print("演示数据已就绪：")
    for k, v in info.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
