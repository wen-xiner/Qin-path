"""公开数据集适配器的单元测试。

这两个用例都是为了拦住实际踩过的坑：
1. ASSISTments 2009 的导出文件不是 UTF-8（cp1252），按 utf-8 读会在第 2634 字节抛异常；
2. 知识点按频次过滤后，题目表里仍带着属于已过滤知识点的题目，
   拼 question_to_kc 时直接 KeyError（真机数据一定会触发，合成数据碰不到）。

跑法：
    cd Qin-path
    python -m pytest ml/tests/test_adapters.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.data.adapters import _detect_encoding, load_assistments_csv

HEADER = "order_id,user_id,assistment_id,problem_id,correct,skill_id,skill_name\n"


def _rows_for(student: str, skill: str, skill_name: str, n: int, start_order: int):
    out = []
    for i in range(n):
        pid = 50000 + start_order + i
        out.append(
            f"{start_order + i},{student},900,{pid},1,{skill},{skill_name}\n"
        )
    return out


def _write_csv(path: Path, encoding: str, skill_name_for_a: str = "Linear Equations") -> Path:
    lines = [HEADER]
    order = 1
    # 知识点 A：5 名学生各 8 次 → 共 40 次（保留）
    for s in range(1, 6):
        lines += _rows_for(f"u{s}", "A", skill_name_for_a, 8, order)
        order += 8
    # 知识点 B：只有一名学生 3 次（min_kc_count=10 时会被丢掉）
    lines += _rows_for("u1", "B", "Box and Whisker", 3, order)
    path.write_bytes("".join(lines).encode(encoding))
    return path


def test_detect_encoding_prefers_utf8(tmp_path: Path):
    p = _write_csv(tmp_path / "plain.csv", "utf-8")
    assert _detect_encoding(p) == "utf-8-sig"


def test_assistments_csv_falls_back_to_cp1252(tmp_path: Path):
    """带 0x80 字节的文件必须能被读出来，而不是直接抛 UnicodeDecodeError。"""
    p = _write_csv(tmp_path / "latin.csv", "utf-8")
    # 在 skill_name 列里塞一个 utf-8 非法、cp1252 合法的字节
    raw = p.read_bytes().replace(b"Linear Equations", b"Ecuaci\xf3n Lineal")
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    p.write_bytes(raw)

    assert _detect_encoding(p) == "cp1252"
    ds = load_assistments_csv(p, name="t", min_kc_count=10, min_seq_len=2, max_seq_len=100)
    assert ds.n_kc == 1 and ds.kc_ids == ["A"]
    assert ds.stats()["n_students"] == 5


def test_filtered_kc_does_not_leak_into_question_table(tmp_path: Path):
    """被过滤知识点的题目不能留在题目表里，否则 question_to_kc 映射会 KeyError。"""
    p = _write_csv(tmp_path / "filter.csv", "utf-8")
    ds = load_assistments_csv(p, name="t", min_kc_count=10, min_seq_len=2, max_seq_len=100)

    assert ds.kc_ids == ["A"], "出现 3 次的知识点 B 应当被过滤"
    # 题目表长度必须与映射长度一致，且每个映射都指向保留下来的知识点下标
    assert len(ds.question_ids) == len(ds.question_to_kc)
    assert set(ds.question_to_kc) == {0}
    # B 的题目（order 41..43 对应的 pid）不应该出现在题目表里
    b_pids = {str(50000 + o) for o in (41, 42, 43)}
    assert not (b_pids & set(ds.question_ids))


def test_keeps_all_kc_when_threshold_is_low(tmp_path: Path):
    p = _write_csv(tmp_path / "all.csv", "utf-8")
    ds = load_assistments_csv(p, name="t", min_kc_count=1, min_seq_len=2, max_seq_len=100)
    assert ds.kc_ids == ["A", "B"]
    assert len(ds.question_ids) == len(ds.question_to_kc)
    # 序列里出现过的知识点下标都在合法范围内
    for seq in ds.sequences:
        assert all(0 <= k < ds.n_kc for k in seq["kc_seq"])


def test_missing_required_column_raises(tmp_path: Path):
    p = tmp_path / "bad.csv"
    p.write_text("order_id,user_id,problem_id\n1,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少必要列"):
        load_assistments_csv(p, name="t")
