"""BKT（贝叶斯知识追踪，Corbett & Anderson, 1994）。

四个参数，含义清晰、结果可解释，也是本项目的基线模型：
    L0  P(L0)  初始掌握概率
    T   P(T)   学习转移概率（本次不会、下次学会）
    S   P(S)   失误概率（会了却答错）
    G   P(G)   猜测概率（不会却蒙对）

参数用 EM 算法逐知识点估计（forward-backward，带 scaling，避免下溢）。
每个知识点独立一套参数，用多个随机初值重启，取似然最高的解。

本实现不引入遗忘，与 BKT 原始假设一致（论文里也常说 BKT 的局限之一就是"学会后不会遗忘"）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.models.base import KnowledgeTracer

# 数值下界，防止除零与 log(0)
EPS = 1e-9

DEFAULT_PARAMS = {"L0": 0.3, "T": 0.15, "S": 0.1, "G": 0.2}


def _emission(obs: np.ndarray, S: float, G: float) -> np.ndarray:
    """返回 (T, 2)：emis[t, k] = P(obs_t | X_t = k)，0=未掌握 1=已掌握。"""
    e = np.empty((len(obs), 2))
    correct = obs == 1
    e[:, 1] = np.where(correct, 1.0 - S, S)      # 已掌握：答对概率 1-S
    e[:, 0] = np.where(correct, G, 1.0 - G)      # 未掌握：答对概率 G
    return e


def _forward_backward(obs: np.ndarray, L0: float, T: float, S: float, G: float):
    """缩放版前向-后向。返回 (gamma, xi, loglik)。"""
    n = len(obs)
    emis = _emission(obs, S, G)
    # 状态转移矩阵 A[i, j] = P(X_{t+1}=j | X_t=i)；已掌握不会退回未掌握
    A = np.array([[1.0 - T, T], [0.0, 1.0]])

    alpha = np.zeros((n, 2))
    scale = np.zeros(n)
    prior = np.array([1.0 - L0, L0])
    for t in range(n):
        v = prior * emis[t]
        s = v.sum()
        if s < EPS:
            v, s = emis[t], max(emis[t].sum(), EPS)
        alpha[t] = v / s
        scale[t] = s
        prior = alpha[t] @ A

    beta = np.zeros((n, 2))
    beta[n - 1] = 1.0
    for t in range(n - 2, -1, -1):
        v = A @ (emis[t + 1] * beta[t + 1])
        s = v.sum()
        if s > EPS:
            v = v / s
        beta[t] = v

    gamma = alpha * beta
    denom = np.clip(gamma.sum(axis=1, keepdims=True), EPS, None)
    gamma = gamma / denom

    xi = np.zeros((max(n - 1, 0), 2, 2))
    for t in range(n - 1):
        m = np.outer(alpha[t], emis[t + 1] * beta[t + 1]) * A
        s = m.sum()
        if s > EPS:
            m = m / s
        xi[t] = m

    return gamma, xi, float(np.log(np.clip(scale, EPS, None)).sum())


def fit_bkt_em(
    obs_list: list[np.ndarray],
    n_iter: int = 60,
    n_restarts: int = 4,
    tol: float = 1e-5,
    seed: int = 42,
) -> tuple[dict, float]:
    """对一个知识点的多段观测序列做 EM，返回 (参数, 对数似然)。"""
    rng = np.random.default_rng(seed)
    obs_list = [np.asarray(o, dtype=int) for o in obs_list if len(o) > 0]
    if not obs_list:
        return dict(DEFAULT_PARAMS), float("-inf")

    best_params, best_ll = dict(DEFAULT_PARAMS), float("-inf")

    inits = [dict(DEFAULT_PARAMS)]
    for _ in range(max(0, n_restarts - 1)):
        # 参数范围约束为合法概率，避免落到鞍点
        inits.append(
            {
                "L0": float(rng.uniform(0.05, 0.8)),
                "T": float(rng.uniform(0.02, 0.6)),
                "S": float(rng.uniform(0.01, 0.35)),
                "G": float(rng.uniform(0.05, 0.4)),
            }
        )

    for init in inits:
        p = dict(init)
        prev_ll = float("-inf")
        for _ in range(n_iter):
            # ---------------- E 步 ----------------
            gammas, xis, ll = [], [], 0.0
            for obs in obs_list:
                g, x, l = _forward_backward(obs, p["L0"], p["T"], p["S"], p["G"])
                gammas.append(g)
                xis.append(x)
                ll += l

            # ---------------- M 步 ----------------
            num_l0 = 0.0
            num_t = den_t = 0.0
            num_s = den_s = 0.0
            num_g = den_g = 0.0
            for obs, g, x in zip(obs_list, gammas, xis):
                num_l0 += g[0, 1]
                if len(x):
                    num_t += x[:, 0, 1].sum()
                    den_t += x[:, 0, 0].sum() + x[:, 0, 1].sum()
                num_s += (g[:, 1] * (obs == 0)).sum()
                den_s += g[:, 1].sum()
                num_g += (g[:, 0] * (obs == 1)).sum()
                den_g += g[:, 0].sum()

            n_seq = len(obs_list)
            p["L0"] = float(np.clip(num_l0 / n_seq, 0.01, 0.99))
            p["T"] = float(np.clip(num_t / max(den_t, EPS), 1e-4, 0.99))
            p["S"] = float(np.clip(num_s / max(den_s, EPS), 1e-4, 0.6))
            p["G"] = float(np.clip(num_g / max(den_g, EPS), 1e-4, 0.6))

            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        if ll > best_ll:
            best_ll, best_params = ll, dict(p)

    return best_params, best_ll


class BKT(KnowledgeTracer):
    """逐知识点的 BKT。"""

    name = "BKT"

    def __init__(self, n_kc: int, n_iter: int = 60, n_restarts: int = 4, seed: int = 42):
        self.n_kc = n_kc
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self.seed = seed
        # 逐知识点参数
        self.params: dict[int, dict] = {k: dict(DEFAULT_PARAMS) for k in range(n_kc)}
        self.loglik: dict[int, float] = {}
        self.n_obs: dict[int, int] = {}

    # ------------------------------------------------------------ 训练
    def fit(self, sequences: list[dict], valid: list[dict] | None = None) -> "BKT":
        # 按知识点把交互拆成一段段序列
        per_kc: dict[int, list[np.ndarray]] = {k: [] for k in range(self.n_kc)}
        for s in sequences:
            buckets: dict[int, list[int]] = {}
            for kc, r in zip(s["kc_seq"], s["resp_seq"]):
                buckets.setdefault(kc, []).append(int(r))
            for kc, obs in buckets.items():
                if kc < self.n_kc:
                    per_kc[kc].append(np.asarray(obs, dtype=int))

        for kc, obs_list in per_kc.items():
            self.n_obs[kc] = int(sum(len(o) for o in obs_list))
            if self.n_obs[kc] < 5:
                # 样本太少，退回默认参数，不硬拟合
                self.params[kc] = dict(DEFAULT_PARAMS)
                self.loglik[kc] = float("nan")
                continue
            params, ll = fit_bkt_em(
                obs_list,
                n_iter=self.n_iter,
                n_restarts=self.n_restarts,
                seed=self.seed + kc,
            )
            self.params[kc] = params
            self.loglik[kc] = ll
        return self

    # ------------------------------------------------------------ 推断
    def _prior_sequence(self, kc: int, obs: np.ndarray) -> np.ndarray:
        """返回 (T, 2)：prior[t] 为"用 obs_{1..t-1} 预测时"的状态分布（含 t=0 的先验）。"""
        p = self.params[kc]
        T = p["T"]
        A = np.array([[1.0 - T, T], [0.0, 1.0]])
        emis = _emission(obs, p["S"], p["G"])
        n = len(obs)
        priors = np.empty((n, 2))
        prior = np.array([1.0 - p["L0"], p["L0"]])
        for t in range(n):
            priors[t] = prior
            v = prior * emis[t]
            s = v.sum()
            if s < EPS:
                v, s = emis[t], max(emis[t].sum(), EPS)
            prior = (v / s) @ A
        return priors

    def predict(self, sequences: list[dict]) -> list[np.ndarray]:
        out = []
        for s in sequences:
            kc_seq = np.asarray(s["kc_seq"], dtype=int)
            resp = np.asarray(s["resp_seq"], dtype=int)
            probs = np.empty(len(resp), dtype=float)
            for kc in np.unique(kc_seq):
                idx = np.where(kc_seq == kc)[0]
                sub = resp[idx]
                priors = self._prior_sequence(int(kc), sub)
                p = self.params[int(kc)]
                probs[idx] = priors[:, 1] * (1.0 - p["S"]) + priors[:, 0] * p["G"]
            out.append(probs)
        return out

    def predict_next(self, kc_seq: list[int], resp_seq: list[int], next_kc: int) -> float:
        """只对 next_kc 的知识点历史做增量计算，比整段重跑快得多。"""
        idx = [i for i, kc in enumerate(kc_seq) if kc == next_kc]
        p = self.params.get(next_kc, dict(DEFAULT_PARAMS))
        T = p["T"]
        prior = np.array([1.0 - p["L0"], p["L0"]])
        for i in idx:
            obs = int(resp_seq[i])
            emis = np.array([p["G"] if obs else 1.0 - p["G"], (1.0 - p["S"]) if obs else p["S"]])
            v = prior * emis
            s = v.sum()
            if s < EPS:
                v, s = emis, max(emis.sum(), EPS)
            post = v / s
            prior = np.array([post[0] * (1.0 - T), post[0] * T + post[1]])
        return float(prior[1] * (1.0 - p["S"]) + prior[0] * p["G"])

    def mastery(self, kc_seq: list[int], resp_seq: list[int], n_kc: int) -> np.ndarray:
        """每个知识点的掌握概率 = 该知识点上最后一次作答后的滤波后验 post_t(1)。

        没做过的知识点返回 NaN（而不是 0），交给上层按"未观测"处理——
        这类知识点不该被当成"薄弱"，只能当成"未知"。
        """
        out = np.full(n_kc, np.nan)
        kc_arr = np.asarray(kc_seq, dtype=int)
        resp = np.asarray(resp_seq, dtype=int)
        for kc in np.unique(kc_arr):
            kc = int(kc)
            if kc >= n_kc:
                continue
            sub = resp[kc_arr == kc]
            if len(sub) == 0:
                continue
            p = self.params.get(kc, dict(DEFAULT_PARAMS))
            T, S, G = p["T"], p["S"], p["G"]
            state = np.array([1.0 - p["L0"], p["L0"]])   # 作答前的状态分布
            post = state
            for obs in sub:
                emis = np.array([G if obs else 1.0 - G, (1.0 - S) if obs else S])
                v = state * emis
                s = v.sum()
                if s < EPS:
                    v, s = emis, max(emis.sum(), EPS)
                post = v / s
                state = np.array([post[0] * (1.0 - T), post[0] * T + post[1]])
            out[kc] = float(post[1])
        return out

    def mastery_with_evidence(self, kc_seq: list[int], resp_seq: list[int], n_kc: int) -> dict:
        """带证据的掌握度：给出每个知识点的作答次数与答对次数，供可解释推荐引用。"""
        kc_arr = np.asarray(kc_seq, dtype=int)
        resp = np.asarray(resp_seq, dtype=int)
        mastery = self.mastery(kc_seq, resp_seq, n_kc)
        attempts = np.zeros(n_kc, dtype=int)
        correct = np.zeros(n_kc, dtype=int)
        for kc, r in zip(kc_arr, resp):
            if kc < n_kc:
                attempts[kc] += 1
                correct[kc] += int(r)
        return {"mastery": mastery, "attempts": attempts, "correct": correct}

    # ------------------------------------------------------------ 存取
    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "n_kc": self.n_kc,
            "params": {str(k): v for k, v in self.params.items()},
            "loglik": {str(k): (None if np.isnan(v) else v) for k, v in self.loglik.items()},
            "n_obs": {str(k): v for k, v in self.n_obs.items()},
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> "BKT":
        obj = cls(n_kc=payload["n_kc"])
        obj.params = {int(k): v for k, v in payload["params"].items()}
        obj.loglik = {
            int(k): (float("nan") if v is None else v) for k, v in payload.get("loglik", {}).items()
        }
        obj.n_obs = {int(k): v for k, v in payload.get("n_obs", {}).items()}
        return obj

    def param_table(self) -> list[dict]:
        return [
            {"kc": k, **{kk: round(vv, 4) for kk, vv in v.items()}, "n_obs": self.n_obs.get(k, 0)}
            for k, v in sorted(self.params.items())
        ]


if __name__ == "__main__":
    print(json.dumps({"module": "bkt", "params": DEFAULT_PARAMS}, ensure_ascii=False))
