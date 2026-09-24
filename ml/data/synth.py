"""合成作答序列生成器。

用途（对应设计文档第六节的"第三选择"）：
- 打通常规流水线、验证模型实现是否正确（生成过程的真值参数已知，可以反过来检验模型能否学到）；
- 支撑前端演示与接口调试。

**重要口径**：合成数据产出的指标不能当作答辩结论。答辩用的指标必须来自公开学术数据集
（ASSISTments / EdNet / XES3G5M 等，见 adapters.py），合成数据的角色是"证明实现没写错"。
本模块在产物里写死 data_source='synthetic'，下游会据此打标，避免混淆。

生成模型（与 BKT 同族，但加入知识点依赖与题目难度）：
    每个学生 s 对知识点 k 有一个潜在掌握度 θ_{s,k} ∈ (0,1)
    观察到答对概率： p = θ_eff * (1 - slip) + (1 - θ_eff) * guess
    其中 θ_eff = θ_{s,k} * (0.5 + 0.5 * min_{p ∈ prereq(k)} θ_{s,p})   ← 前置知识点没学好会拉低表现
    每答一题后按学习转移更新： θ ← θ + (1 - θ) * learn_rate
    难度修正：题型越难，slip 越大、guess 越小
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.data.knowledge_graph import KnowledgeGraph, get_knowledge_graph
from ml.data.question_bank import Question, build_question_bank


@dataclass
class SyntheticConfig:
    n_students: int = 600
    min_seq_len: int = 15
    max_seq_len: int = 60
    slip: float = 0.10            # 失误：会了但答错
    guess: float = 0.20           # 猜测：不会但答对
    learn_rate: float = 0.12      # 每答一题的学习增益
    forget_rate: float = 0.01     # 遗忘（BKT 原生假设无遗忘，这里留一个很小的量做扰动）
    prereq_penalty: float = 0.50  # 前置知识点对表现的影响强度 0~1
    theta_alpha: float = 3.0      # 学生初始掌握度的 Beta 先验
    theta_beta: float = 2.5
    # 连续练习同一知识点的概率。真实数据里学生是「成串刷同一个知识点」的，
    # 而不是每道题随机换知识点。这个参数直接决定序列里是否存在可被序列模型利用的结构。
    repeat_prob: float = 0.55


def _difficulty_adjusted(item_difficulty: int, cfg: SyntheticConfig) -> tuple[float, float]:
    """难度 1/2/3 → 实际 slip / guess。"""
    delta = (item_difficulty - 2) * 0.05
    slip = float(np.clip(cfg.slip + delta, 0.01, 0.45))
    guess = float(np.clip(cfg.guess - delta, 0.02, 0.45))
    return slip, guess


@dataclass
class SequenceDataset:
    """统一的序列数据集结构。"""

    kc_ids: list[str]                 # 下标即知识点编号
    question_ids: list[str]
    question_to_kc: list[int]
    question_difficulty: list[int]
    sequences: list[dict]             # [{student_id, kc_seq, q_seq, resp_seq}]
    data_source: str                  # 'synthetic' | 数据集名
    meta: dict

    @property
    def n_kc(self) -> int:
        return len(self.kc_ids)

    @property
    def n_questions(self) -> int:
        return len(self.question_ids)

    # 实现序列协议，这样 SequenceDataset 可以像 list[dict] 一样直接喂给模型，
    # 避免调用处反复写 .sequences 出错。
    def __iter__(self):
        return iter(self.sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, i):
        return self.sequences[i]

    def stats(self) -> dict:
        lengths = [len(s["resp_seq"]) for s in self.sequences]
        correct = sum(sum(s["resp_seq"]) for s in self.sequences)
        total = sum(lengths) or 1
        return {
            "data_source": self.data_source,
            "n_students": len(self.sequences),
            "n_kc": self.n_kc,
            "n_questions": self.n_questions,
            "n_interactions": total,
            "avg_seq_len": round(float(np.mean(lengths)), 2),
            "accuracy": round(correct / total, 4),
        }


def generate_synthetic_dataset(
    n_students: int = 600,
    min_seq_len: int = 15,
    max_seq_len: int = 60,
    seed: int = 42,
    graph: KnowledgeGraph | None = None,
    bank: list[Question] | None = None,
) -> SequenceDataset:
    """按上述生成过程合成一批学生答题序列。"""
    rng = np.random.default_rng(seed)
    graph = graph or get_knowledge_graph()
    bank = bank or build_question_bank()

    kc_ids = graph.kc_ids
    kc_index = {k: i for i, k in enumerate(kc_ids)}

    # 题目按知识点分组，便于抽取"某知识点的题"
    by_kc: dict[str, list[Question]] = {}
    for q in bank:
        by_kc.setdefault(q.kc_id, []).append(q)

    # 学习顺序：拓扑序。学生按顺序推进，符合真实学习节奏。
    order = graph.topological_order()
    # 每个知识点的（直接）前置在 kc 下标空间中的位置
    prereq_idx = {kc_index[k]: [kc_index[p] for p in graph[k].prerequisites] for k in kc_ids}

    question_ids = [q.question_id for q in bank]
    question_to_kc = [kc_index[q.kc_id] for q in bank]
    question_difficulty = [q.difficulty for q in bank]
    q_index = {q.question_id: i for i, q in enumerate(bank)}

    cfg = SyntheticConfig(n_students=n_students)
    n_kc = len(kc_ids)
    sequences: list[dict] = []

    for sid in range(n_students):
        # 初始掌握度：Beta 先验，学生之间存在"基础好/基础弱"的个体差异
        theta = rng.beta(cfg.theta_alpha, cfg.theta_beta, size=n_kc)
        # 每个学生有个"推进进度"，越靠后的知识点练得越少
        progress = float(rng.uniform(0.35, 1.0))

        max_reachable = max(1, int(np.ceil(progress * len(order))))
        reachable = set(order[:max_reachable])

        # 允许作答的知识点：已推进到的知识点，再加少量越界（乱序刷题的真实感）
        pool = [k for k in kc_ids if k in reachable]
        if len(pool) < 3:
            pool = kc_ids[:3]

        seq_len = int(rng.integers(min_seq_len, max_seq_len + 1))
        kc_seq: list[int] = []
        q_seq: list[int] = []
        resp_seq: list[int] = []

        # 第一个知识点在池中随机选；之后以 repeat_prob 继续刷同一个知识点，
        # 否则在池中按「越薄弱越可能被练」的权重换一个 —— 这是真实刷题行为的简化。
        kc = pool[int(rng.integers(0, len(pool)))]

        for _ in range(seq_len):
            k_idx = kc_index[kc]

            # 前置影响
            pre = prereq_idx[k_idx]
            if pre:
                weakest = float(min(theta[p] for p in pre))
                theta_eff = theta[k_idx] * (1 - cfg.prereq_penalty + cfg.prereq_penalty * weakest)
            else:
                theta_eff = theta[k_idx]
            theta_eff = float(np.clip(theta_eff, 0.005, 0.995))

            cands = by_kc[kc]
            q = cands[int(rng.integers(0, len(cands)))]
            slip, guess = _difficulty_adjusted(q.difficulty, cfg)

            p_correct = theta_eff * (1 - slip) + (1 - theta_eff) * guess
            resp = int(rng.random() < p_correct)

            kc_seq.append(k_idx)
            q_seq.append(q_index[q.question_id])
            resp_seq.append(resp)

            # 学习与遗忘
            if resp:
                theta[k_idx] = theta[k_idx] + (1 - theta[k_idx]) * cfg.learn_rate
            else:
                # 作答错误也会带来一点学习（订正效应），但幅度更小
                theta[k_idx] = theta[k_idx] + (1 - theta[k_idx]) * cfg.learn_rate * 0.35
            theta[k_idx] -= theta[k_idx] * cfg.forget_rate
            theta[k_idx] = float(np.clip(theta[k_idx], 0.001, 0.999))

            # 选择下一题的知识点
            if rng.random() < cfg.repeat_prob:
                pass  # 继续同一知识点
            else:
                weights = np.array(
                    [max(0.02, 1.0 - theta[kc_index[k]]) ** 1.5 for k in pool], dtype=float
                )
                weights /= weights.sum()
                kc = pool[int(rng.choice(len(pool), p=weights))]

        sequences.append(
            {
                "student_id": sid,
                "kc_seq": kc_seq,
                "q_seq": q_seq,
                "resp_seq": resp_seq,
            }
        )

    return SequenceDataset(
        kc_ids=kc_ids,
        question_ids=question_ids,
        question_to_kc=question_to_kc,
        question_difficulty=question_difficulty,
        sequences=sequences,
        data_source="synthetic",
        meta={
            "generator": "bkt_with_prerequisites",
            "seed": seed,
            "slip": cfg.slip,
            "guess": cfg.guess,
            "learn_rate": cfg.learn_rate,
            "prereq_penalty": cfg.prereq_penalty,
            "note": "合成数据仅用于验证实现与演示，不作为答辩指标来源",
        },
    )


if __name__ == "__main__":
    ds = generate_synthetic_dataset(n_students=200, seed=42)
    import json

    print(json.dumps(ds.stats(), ensure_ascii=False, indent=2))
