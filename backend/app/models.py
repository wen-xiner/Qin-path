"""ORM 模型。

设计取舍：知识点、题目、知识点依赖图这类**静态内容**不放进数据库，
而是放在 data/processed/ 下的 JSON/CSV 里由服务层加载。
数据库只存**会变的东西**：用户、班级、作答记录。
这样换学科时（设计文档第八节「第二版换学科」）只需换数据文件，不用迁移数据库。
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    display_name: Mapped[str] = mapped_column(String(64), default="")
    role: Mapped[str] = mapped_column(String(16), default="student")  # student | teacher
    class_id: Mapped[int | None] = mapped_column(ForeignKey("classes.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    classroom: Mapped["Classroom | None"] = relationship(back_populates="members")

    @property
    def is_teacher(self) -> bool:
        return self.role == "teacher"


class Classroom(Base):
    __tablename__ = "classes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    course: Mapped[str] = mapped_column(String(64), default="数据结构")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    members: Mapped[list[User]] = relationship(back_populates="classroom")


class AnswerRecord(Base):
    """一条作答记录。学生端每次提交答案写一条。"""

    __tablename__ = "answer_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    question_id: Mapped[str] = mapped_column(String(32), index=True)
    kc_id: Mapped[str] = mapped_column(String(64), index=True)
    chosen_index: Mapped[int] = mapped_column(Integer)
    is_correct: Mapped[bool] = mapped_column(Boolean)
    # 作答时该知识点的掌握度快照，用于画曲线与事后复盘（不必回放全部历史）
    mastery_before: Mapped[float | None] = mapped_column(Float, nullable=True)
    mastery_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    seq_no: Mapped[int] = mapped_column(Integer, default=0)  # 该学生的第几次作答
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)

    __table_args__ = (UniqueConstraint("user_id", "seq_no", name="uq_user_seq"),)


class RecommendationLog(Base):
    """推荐日志：记录系统推荐了什么、理由是什么。

    存在的意义是**可追溯**——答辩现场被问"你这个推荐是怎么来的"，
    可以把这个表调出来，逐条对。
    """

    __tablename__ = "recommendation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    kc_id: Mapped[str] = mapped_column(String(64))
    question_id: Mapped[str] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(String(512))
    mastery: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)
    followed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # 学生是否采纳了推荐
