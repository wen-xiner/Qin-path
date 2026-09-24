"""模型层与规划层的单元测试。

跑法：
    cd Qin-path
    python -m pytest ml/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.data.knowledge_graph import KnowledgeGraph, get_knowledge_graph
from ml.data.question_bank import build_question_bank
from ml.data.synth import SequenceDataset, generate_synthetic_dataset
from ml.metrics import evaluate
from ml.models.baseline import GlobalMeanBaseline, KCMeanBaseline
from ml.models.bkt import BKT
from ml.planner import plan_path


# ------------------------------------------------------------------ 知识点图
def test_knowledge_graph_structure():
    g = get_knowledge_graph()
    assert 20 <= len(g) <= 30, "设计文档要求知识点控制在 20-30 个"
    # 拓扑序长度必须等于知识点数，说明是无环图
    assert len(g.topological_order()) == len(g)
    # 第一层应该是没有前置的那些
    layers = g.learning_layers()
    assert all(not g[k].prerequisites for k in layers[0])
    assert len(layers) >= 3


def test_knowledge_graph_detects_cycle():
    from ml.data.knowledge_graph import KnowledgeComponent

    with pytest.raises(ValueError):
        KnowledgeGraph(
            [
                KnowledgeComponent("a", "A", "c1", ("b",)),
                KnowledgeComponent("b", "B", "c1", ("a",)),
            ]
        )


def test_transitive_prerequisites():
    g = get_knowledge_graph()
    # 二叉搜索树 → 二叉树遍历 → 树的遍历依赖栈与队列、树基本概念
    pre = g.prerequisites_of("bst", transitive=True)
    assert "binary_tree_traversal" in pre
    assert "stack" in pre


# ------------------------------------------------------------------ 题库
def test_question_bank_coverage():
    bank = build_question_bank()
    assert 200 <= len(bank) <= 300, "设计文档要求题目规模 200-300 道"
    graph = get_knowledge_graph()
    kcs = {q.kc_id for q in bank}
    assert kcs == set(graph.kc_ids), "每个知识点都要有题"
    for q in bank:
        assert len(q.options) == 4
        assert 0 <= q.answer_index < 4
        assert q.difficulty in (1, 2, 3)
        assert q.stem and q.stem.strip()


# ------------------------------------------------------------------ 基线
def test_mean_baseline_predicts_constant():
    ds = generate_synthetic_dataset(n_students=40, max_seq_len=20, seed=1)
    m = GlobalMeanBaseline().fit(ds.sequences)
    probs = m.predict(ds.sequences[:3])
    assert all(len(p) == len(s["resp_seq"]) for p, s in zip(probs, ds.sequences[:3]))
    assert all(np.allclose(p, m.p) for p in probs)
    # 均值基线在 AUC 上应当接近 0.5
    assert abs(evaluate(ds.sequences, m.predict(ds.sequences))["auc_all"] - 0.5) < 0.05


def test_kc_mean_baseline_varies_by_kc():
    ds = generate_synthetic_dataset(n_students=60, max_seq_len=25, seed=2)
    m = KCMeanBaseline(n_kc=ds.n_kc).fit(ds.sequences)
    assert m.p_per_kc.shape == (ds.n_kc,)
    assert np.all((m.p_per_kc > 0) & (m.p_per_kc < 1))
    # 逐知识点均值应当是全局均值的更精细版本，AUC 不低于全局均值太多
    assert evaluate(ds.sequences, m.predict(ds.sequences))["auc_all"] > 0.45


# ------------------------------------------------------------------ BKT
def test_bkt_learns_plausible_parameters():
    """用已知真值的生成过程反推：BKT 应当能学到接近真值的参数。"""
    ds = generate_synthetic_dataset(n_students=300, max_seq_len=40, seed=3)
    bkt = BKT(n_kc=ds.n_kc, n_restarts=3).fit(ds.sequences)

    # 生成过程的真值是 slip=0.10、guess=0.20（难度会做正负 0.05 的修正）
    slips = [p["S"] for p in bkt.params.values()]
    guesses = [p["G"] for p in bkt.params.values()]
    learns = [p["T"] for p in bkt.params.values()]
    assert 0.02 < np.mean(slips) < 0.30, f"失误概率估计不合理：{np.mean(slips)}"
    assert 0.05 < np.mean(guesses) < 0.40, f"猜测概率估计不合理：{np.mean(guesses)}"
    assert 0.01 < np.mean(learns) < 0.60, f"学习转移概率估计不合理：{np.mean(learns)}"


def test_bkt_probabilities_in_range_and_better_than_chance():
    ds = generate_synthetic_dataset(n_students=200, max_seq_len=30, seed=4)
    bkt = BKT(n_kc=ds.n_kc, n_restarts=2).fit(ds.sequences)
    probs = bkt.predict(ds.sequences)
    flat = np.concatenate(probs)
    assert np.all((flat >= 0) & (flat <= 1))
    assert evaluate(ds.sequences, probs)["auc_all"] > 0.52, "BKT 应当明显优于随机"


def test_bkt_predict_next_matches_full_predict():
    """增量接口 predict_next 必须和整段重算的结果一致，否则后端线上行为会和离线评测不符。

    正确的等价关系：用前 t 个交互预测第 t 个交互，应当等于整段预测在第 t 个位置的值。
    注意不能拿 predict_next 去比 predict 的末位——末位是"预测最后一个已发生的题"，
    而 predict_next 是"预测下一个还没发生的题"，两者本来就差一步。
    """
    ds = generate_synthetic_dataset(n_students=60, max_seq_len=25, seed=5)
    bkt = BKT(n_kc=ds.n_kc, n_restarts=2).fit(ds.sequences)
    s = ds.sequences[0]
    for t in range(1, len(s["resp_seq"])):
        target = s["kc_seq"][t]
        inc = bkt.predict_next(s["kc_seq"][:t], s["resp_seq"][:t], target)
        full = bkt.predict(
            [
                {
                    "kc_seq": s["kc_seq"][: t + 1],
                    "q_seq": s["kc_seq"][: t + 1],
                    "resp_seq": s["resp_seq"][: t + 1],
                }
            ]
        )[0][t]
        assert abs(inc - full) < 1e-9


def test_bkt_mastery_marks_unobserved_as_nan():
    """未观测的知识点必须是 NaN 而不是 0——把没做过当成不会会让推荐彻底乱掉。"""
    ds = generate_synthetic_dataset(n_students=50, max_seq_len=20, seed=6)
    bkt = BKT(n_kc=ds.n_kc, n_restarts=2).fit(ds.sequences)
    s = ds.sequences[0]
    mastery = bkt.mastery(s["kc_seq"], s["resp_seq"], ds.n_kc)
    observed = set(s["kc_seq"])
    for i in range(ds.n_kc):
        if i in observed:
            assert not np.isnan(mastery[i])
        else:
            assert np.isnan(mastery[i])


# ------------------------------------------------------------------ 路径规划
def test_plan_prefers_prerequisite_when_both_weak():
    """前置和它自己都没掌握时，必须先补前置。"""
    g = get_knowledge_graph()
    n = len(g)
    mastery = np.full(n, 0.95)
    attempts = np.ones(n, dtype=int)
    # 让「二叉树遍历」和它的前置「树的遍历」(tree_basic) 都很弱
    mastery[g.kc_ids.index("binary_tree_traversal")] = 0.20
    mastery[g.kc_ids.index("tree_basic")] = 0.15

    plan = plan_path(mastery, attempts, graph=g, horizon=5)
    top_ids = [item["kc_id"] for item in plan["queue"]]
    assert "tree_basic" in top_ids, "前置薄弱时应当优先补前置"
    # 被阻塞的知识点要出现在 blocked 里
    assert any(b["kc_id"] == "binary_tree_traversal" for b in plan["blocked"])


def test_plan_advances_when_prerequisites_ok():
    """前置都达标、自己没学过 → 推进新知识点。"""
    g = get_knowledge_graph()
    n = len(g)
    mastery = np.full(n, np.nan)
    attempts = np.zeros(n, dtype=int)
    # 只有第一层达标
    for kc in g.learning_layers()[0]:
        mastery[g.kc_ids.index(kc)] = 0.9
        attempts[g.kc_ids.index(kc)] = 3

    plan = plan_path(mastery, attempts, graph=g, horizon=5)
    assert plan["next"] is not None
    assert plan["next"]["action"] == "advance"
    # 不能推荐一个前置还没达标的知识点
    for item in plan["queue"]:
        if item["action"] == "advance":
            for p in g[item["kc_id"]].prerequisites:
                assert mastery[g.kc_ids.index(p)] >= 0.7


def test_plan_empty_history_recommends_first_layer():
    g = get_knowledge_graph()
    n = len(g)
    plan = plan_path(np.full(n, np.nan), np.zeros(n, dtype=int), graph=g, horizon=5)
    assert plan["n_observed"] == 0
    assert plan["next"] is not None
    assert plan["next"]["kc_id"] in g.learning_layers()[0]


def test_plan_weak_but_prereq_ok_uses_remedy():
    g = get_knowledge_graph()
    n = len(g)
    mastery = np.full(n, 0.9)
    attempts = np.ones(n, dtype=int) * 3
    target = "sort_basic"
    mastery[g.kc_ids.index(target)] = 0.30

    plan = plan_path(mastery, attempts, graph=g, horizon=3)
    ids = [item["kc_id"] for item in plan["queue"]]
    assert target in ids
    item = next(i for i in plan["queue"] if i["kc_id"] == target)
    assert item["action"] == "remedy"
    assert "0.30" in item["reason"]


# ------------------------------------------------------------------ 数据集
def test_dataset_split_by_student():
    from ml.data.dataset import split_dataset

    ds = generate_synthetic_dataset(n_students=100, max_seq_len=20, seed=7)
    splits = split_dataset(ds, seed=7)
    total = sum(len(s) for s in splits.values())
    assert total == 100, "切分不能丢学生"
    # 三份之间不能有重叠的学生
    ids = [set(s["student_id"] for s in splits[k]) for k in ("train", "valid", "test")]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_dataset_iterable_like_list():
    ds = generate_synthetic_dataset(n_students=10, max_seq_len=15, seed=8)
    assert len(ds) == 10
    assert isinstance(ds[0], dict)
    assert len(list(ds)) == 10


def test_synthetic_repeat_structure_exists():
    """真实刷题有成串练同一知识点的结构；生成器没有这个结构会不公平地压制序列模型。"""
    ds = generate_synthetic_dataset(n_students=50, max_seq_len=40, seed=9)
    repeats = 0
    total = 0
    for s in ds.sequences:
        for a, b in zip(s["kc_seq"], s["kc_seq"][1:]):
            total += 1
            repeats += int(a == b)
    ratio = repeats / max(total, 1)
    assert ratio > 0.3, f"连续同知识点比例过低（{ratio:.2f}），序列缺乏可学结构"


# ------------------------------------------------------------------ 指标
def test_evaluate_shapes_and_ranges():
    y = SequenceDataset(
        kc_ids=["a"], question_ids=["q"], question_to_kc=[0], question_difficulty=[1],
        sequences=[{"student_id": 0, "kc_seq": [0, 0], "q_seq": [0, 0], "resp_seq": [1, 0]}],
        data_source="t", meta={},
    )
    m = evaluate(y.sequences, [np.array([0.9, 0.1])])
    assert 0 <= m["acc_all"] <= 1
    assert m["n_all"] == 2
    assert m["n_skip1st"] == 1
    assert m["rmse_all"] < 0.2
