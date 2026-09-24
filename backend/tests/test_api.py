"""后端接口测试：把「答题 → 掌握度更新 → 推荐 → 可解释理由」这条闭环走一遍。

跑法：
    cd Qin-path
    python -m pytest backend/tests -q
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.database import init_db
from backend.app.deps import reset_kt
from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    reset_kt()
    init_db()
    with TestClient(app) as c:
        yield c


def _register(client, username: str, role: str = "student", class_name: str | None = None):
    resp = client.post(
        "/api/auth/register",
        json={
            "username": username,
            "password": "test1234",
            "display_name": username,
            "role": role,
            "class_name": class_name,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_meta_endpoints(client):
    token = _register(client, "t_meta")
    meta = client.get("/api/meta", headers=_auth(token)).json()
    assert meta["n_kc"] > 0
    assert meta["n_questions"] >= 200
    assert meta["model"] in ("BKT", "DKT", "SAKT")

    graph = client.get("/api/meta/knowledge-graph", headers=_auth(token)).json()
    assert len(graph["nodes"]) == meta["n_kc"]
    # 依赖图必须是有向无环的，拓扑序长度应等于知识点数
    assert len(graph["topological_order"]) == meta["n_kc"]

    questions = client.get("/api/meta/questions", headers=_auth(token)).json()
    assert len(questions) == meta["n_questions"]


def test_login_and_me(client):
    _register(client, "t_login")
    resp = client.post("/api/auth/login", json={"username": "t_login", "password": "test1234"})
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    me = client.get("/api/auth/me", headers=_auth(token)).json()
    assert me["username"] == "t_login"


def test_login_wrong_password(client):
    _register(client, "t_wrong")
    resp = client.post("/api/auth/login", json={"username": "t_wrong", "password": "bad"})
    assert resp.status_code == 401


def test_unauthorized_access(client):
    assert client.get("/api/student/mastery").status_code == 401


def test_student_closed_loop(client):
    token = _register(client, "t_loop")
    h = _auth(token)

    # 1) 初始状态：无任何记录，掌握度全为「未观测」
    m0 = client.get("/api/student/mastery", headers=h).json()
    assert m0["n_observed"] == 0
    assert all(item["state"] == "unobserved" for item in m0["items"])

    # 2) 推荐第一题：应该是「推进新知识点」而不是「补强」
    rec = client.get("/api/student/next-question", headers=h).json()
    assert rec is not None
    assert rec["action"] == "advance"
    assert rec["question"]["question_id"]
    assert "掌握" in rec["reason"] or "前置" in rec["reason"]
    assert len(rec["question"]["options"]) == 4

    # 3) 连答 8 题，每题都应拿到下一题的推荐（闭环）
    answered = 0
    for _ in range(8):
        rec = client.get("/api/student/next-question", headers=h).json()
        if rec is None:
            break
        qid = rec["question"]["question_id"]
        # 故意选一个肯定成立的答案下标（0），只验证链路，不验证对错
        ans = client.post(
            "/api/student/answer", json={"question_id": qid, "chosen_index": 0}, headers=h
        )
        assert ans.status_code == 200, ans.text
        body = ans.json()
        assert isinstance(body["is_correct"], bool)
        assert 0 <= body["correct_index"] <= 3
        assert body["mastery_after"] is not None
        assert body["feedback"]
        answered += 1

    assert answered >= 5

    # 4) 掌握度有观测值了，且有知识点进入「已掌握」或「薄弱」
    m1 = client.get("/api/student/mastery", headers=h).json()
    assert m1["n_observed"] > 0
    assert any(item["state"] in ("mastered", "weak") for item in m1["items"])

    # 5) 学习曲线随作答增长
    curve = client.get("/api/student/curve", headers=h).json()
    assert len(curve["points"]) == answered

    # 6) 历史记录
    hist = client.get("/api/student/history", headers=h).json()
    assert len(hist) == answered


def test_answer_unknown_question(client):
    token = _register(client, "t_badq")
    resp = client.post(
        "/api/student/answer", json={"question_id": "NOT_EXIST", "chosen_index": 0},
        headers=_auth(token),
    )
    assert resp.status_code == 404


def test_teacher_class_mastery(client):
    teacher_token = _register(client, "t_teacher", role="teacher", class_name="测试班")
    _register(client, "t_s1", class_name="测试班")
    _register(client, "t_s2", class_name="测试班")

    h = _auth(teacher_token)
    data = client.get("/api/teacher/class-mastery", headers=h).json()
    assert data["class_name"] == "测试班"
    assert data["n_students"] == 2
    assert len(data["kc"]) > 0

    students = client.get("/api/teacher/students", headers=h).json()
    assert len(students) == 2


def test_teacher_endpoint_forbidden_for_student(client):
    token = _register(client, "t_notteacher")
    assert client.get("/api/teacher/class-mastery", headers=_auth(token)).status_code == 403


def test_token_is_signed(client):
    """篡改 token 应该被拒。"""
    token = _register(client, "t_sig")
    bad = token[:-2] + ("aa" if token[-2:] != "aa" else "bb")
    assert client.get("/api/auth/me", headers=_auth(bad)).status_code == 401
