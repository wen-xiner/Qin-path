"""后端配置。"""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT = BACKEND_DIR.parent

# SQLite 起步。需要并发时把 ZT_DATABASE_URL 换成 PostgreSQL 即可，
# 例如 postgresql+psycopg://user:pass@localhost:5432/zhitu
DATABASE_URL = os.environ.get("ZT_DATABASE_URL", f"sqlite:///{BACKEND_DIR / 'zhitu.db'}")

# 演示用签名密钥。生产环境务必用环境变量覆盖。
SECRET_KEY = os.environ.get("ZT_SECRET_KEY", "zhitu-dev-secret-change-me")
TOKEN_TTL_HOURS = int(os.environ.get("ZT_TOKEN_TTL_HOURS", "72"))

# 用哪个模型产出掌握度（BKT 可解释性最强，默认它）
KT_MODEL = os.environ.get("ZT_KT_MODEL", "BKT")
# 数据集前缀，对应 data/processed/dataset_index.json
DATASET_PREFIX = os.environ.get("ZT_DATASET_PREFIX", "dataset")

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]
