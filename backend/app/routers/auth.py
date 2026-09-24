"""认证接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.deps import get_current_user
from backend.app.models import Classroom, User
from backend.app.schemas import LoginIn, RegisterIn, TokenOut, UserOut
from backend.app.security import create_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["认证"])


@router.post("/register", response_model=TokenOut, summary="注册")
def register(payload: RegisterIn, db: Session = Depends(get_db)) -> TokenOut:
    exists = db.scalar(select(User).where(User.username == payload.username))
    if exists:
        raise HTTPException(status.HTTP_409_CONFLICT, "用户名已存在")

    class_id = None
    if payload.class_name:
        cls = db.scalar(select(Classroom).where(Classroom.name == payload.class_name))
        if cls is None:
            cls = Classroom(name=payload.class_name)
            db.add(cls)
            db.flush()
        class_id = cls.id

    user = User(
        username=payload.username,
        password_hash=hash_password(payload.password),
        display_name=payload.display_name or payload.username,
        role=payload.role if payload.role in ("student", "teacher") else "student",
        class_id=class_id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return TokenOut(access_token=create_token(user.id, user.role), user=UserOut.model_validate(user))


@router.post("/login", response_model=TokenOut, summary="登录")
def login(payload: LoginIn, db: Session = Depends(get_db)) -> TokenOut:
    user = db.scalar(select(User).where(User.username == payload.username))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")
    return TokenOut(access_token=create_token(user.id, user.role), user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut, summary="当前用户")
def me(user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(user)
