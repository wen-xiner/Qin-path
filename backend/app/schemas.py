"""Pydantic 请求/响应模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ 认证
class RegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=4, max_length=64)
    display_name: str = ""
    role: str = "student"
    class_name: str | None = None


class LoginIn(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    display_name: str
    role: str
    class_id: int | None = None

    model_config = {"from_attributes": True}


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ------------------------------------------------------------------ 答题
class QuestionOut(BaseModel):
    question_id: str
    kc_id: str
    kc_name: str
    difficulty: int
    stem: str
    options: list[str]


class Evidence(BaseModel):
    mastery: float | None = None
    attempts: int = 0
    correct: int = 0
    threshold: float
    predicted_correct_prob: float | None = None
    blocked_by: list[str] = []


class RecommendationOut(BaseModel):
    question: QuestionOut
    action: str
    reason: str
    evidence: Evidence


class AnswerIn(BaseModel):
    question_id: str
    chosen_index: int = Field(ge=0, le=10)


class AnswerOut(BaseModel):
    is_correct: bool
    correct_index: int
    kc_id: str
    kc_name: str
    mastery_before: float | None = None
    mastery_after: float | None = None
    mastery_delta: float | None = None
    # 作答后立刻给出下一步推荐，形成闭环
    next: RecommendationOut | None = None
    feedback: str = ""


# ------------------------------------------------------------------ 掌握度
class MasteryItem(BaseModel):
    kc_id: str
    name: str
    chapter: str
    mastery: float | None = None
    attempts: int = 0
    correct: int = 0
    state: str  # mastered | weak | unobserved


class PlanItemOut(BaseModel):
    kc_id: str
    name: str
    chapter: str
    action: str
    mastery: float | None
    attempts: int
    correct: int
    priority: float
    reason: str
    blocked_by: list[str] = []


class MasteryOut(BaseModel):
    threshold: float
    n_kc: int
    n_observed: int
    items: list[MasteryItem]
    plan: dict


class CurvePoint(BaseModel):
    step: int
    kc_id: str | None = None
    correct: int
    mastery: float | None = None


class CurveOut(BaseModel):
    kc_id: str | None = None
    points: list[CurvePoint]


# ------------------------------------------------------------------ 教师端
class KCStat(BaseModel):
    kc_id: str
    name: str
    chapter: str
    n_observed: int
    n_students: int
    mean_mastery: float | None
    n_below: int
    weak_ratio: float | None


class ClassMasteryOut(BaseModel):
    class_id: int
    class_name: str
    n_students: int
    threshold: float
    kc: list[KCStat]
    class_weak_points: list[KCStat]
    per_student: list[dict]


# ------------------------------------------------------------------ 元信息
class MetaOut(BaseModel):
    model: str
    device: str
    profile: str
    n_kc: int
    n_questions: int
    mastery_threshold: float
    data_source: str
    experiment: dict | None = None


class HistoryItem(BaseModel):
    seq_no: int
    question_id: str
    kc_id: str
    kc_name: str
    is_correct: bool
    mastery_after: float | None
    created_at: datetime
