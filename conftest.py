"""pytest 配置：用独立的测试数据库，避免污染开发库。"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 必须在导入应用之前设置
_tmp = Path(tempfile.gettempdir()) / "zhitu_test.db"
if _tmp.exists():
    _tmp.unlink()
os.environ["ZT_DATABASE_URL"] = f"sqlite:///{_tmp}"
os.environ["ZT_SECRET_KEY"] = "test-secret"
