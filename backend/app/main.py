"""知途 · 后端入口。

启动：
    cd Qin-path
    python -m uvicorn backend.app.main:app --reload --port 8000
接口文档：http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import CORS_ORIGINS
from backend.app.database import init_db
from backend.app.routers import auth, meta, student, teacher


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # 预热知识追踪服务：把模型产物、数据集、知识点图一次性加载好。
    # 不预热的话第一个请求要等加载完成（产物缺失时还会现场训练），前端会以为服务挂了。
    try:
        from backend.app.deps import get_kt

        get_kt()
    except Exception as exc:  # noqa: BLE001  启动阶段只告警，不阻塞服务
        print(f"[启动警告] 知识追踪服务预热失败：{exc}")
    yield


app = FastAPI(
    title="知途 · 知识追踪与自适应学习路径系统",
    description=(
        "用知识追踪模型估计每个知识点的掌握程度，据此规划下一步学习路径，"
        "并让每一次推荐都附带可解释的依据。"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(student.router)
app.include_router(teacher.router)
app.include_router(meta.router)


@app.get("/api/health", tags=["系统"], summary="健康检查")
def health() -> dict:
    return {"status": "ok", "service": "zhitu"}


@app.get("/", include_in_schema=False)
def index() -> dict:
    return {
        "service": "知途",
        "docs": "/docs",
        "health": "/api/health",
    }
