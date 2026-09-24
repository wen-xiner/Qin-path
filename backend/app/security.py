"""认证与签名。

刻意不引入 JWT 库：用一个 HMAC-SHA256 签名的紧凑 token（payload.signature），
依赖为零，行为完全透明、可调试。教学项目里"看得懂"比"用主流库"更重要。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import bcrypt

from backend.app.config import SECRET_KEY, TOKEN_TTL_HOURS


# ------------------------------------------------------------------ 口令
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=10)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ------------------------------------------------------------------ token
def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def _sign(payload_b64: str) -> str:
    mac = hmac.new(SECRET_KEY.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256)
    return _b64e(mac.digest())


def create_token(user_id: int, role: str, ttl_hours: int = TOKEN_TTL_HOURS) -> str:
    payload = {"uid": user_id, "role": role, "exp": int(time.time()) + ttl_hours * 3600}
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    return f"{body}.{_sign(body)}"


def decode_token(token: str) -> dict | None:
    """校验签名与过期时间，失败返回 None。"""
    if not token or "." not in token:
        return None
    body, sig = token.rsplit(".", 1)
    if not hmac.compare_digest(sig, _sign(body)):
        return None
    try:
        payload = json.loads(_b64d(body))
    except (ValueError, json.JSONDecodeError):
        return None
    if int(payload.get("exp", 0)) < time.time():
        return None
    return payload
