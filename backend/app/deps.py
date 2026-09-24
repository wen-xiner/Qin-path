"""FastAPI 依赖：当前用户、角色校验、KT 服务单例。"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import User
from backend.app.security import decode_token

bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "缺少访问令牌")
    payload = decode_token(creds.credentials)
    if payload is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "令牌无效或已过期")
    user = db.get(User, int(payload["uid"]))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户不存在")
    return user


def require_teacher(user: User = Depends(get_current_user)) -> User:
    if not user.is_teacher:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "该接口仅教师可用")
    return user


_kt_service = None


def get_kt():
    """加载知识追踪服务（进程内单例）。

    首次调用会尝试加载训练产物；找不到时会现场训练一个，所以即使还没跑实验也能起服务。
    """
    global _kt_service
    if _kt_service is None:
        from backend.app.config import DATASET_PREFIX, KT_MODEL
        from ml.service import KTService

        _kt_service = KTService.create(KT_MODEL, dataset_prefix=DATASET_PREFIX)
    return _kt_service


def reset_kt() -> None:
    """测试用：清掉单例。"""
    global _kt_service
    _kt_service = None
