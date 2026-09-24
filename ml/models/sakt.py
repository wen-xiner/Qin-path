"""SAKT（自注意力知识追踪，Pandey & Karypis, 2019）——对照模型。

用自注意力机制代替 RNN，直接建立"当前题目"与"历史相关交互"之间的关联，
在数据稀疏、序列较长时通常表现更稳。这里用带因果掩码的 Transformer Encoder 实现，
位置 t 的输入是 (kc_t, resp_{t-1})，注意力只能看到位置 ≤ t（不会泄漏 resp_t）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ml.models.base import TorchKnowledgeTracer
from ml.models.torch_utils import default_profile_kwargs, predict_logits, train_loop


class SAKTModule(nn.Module):
    def __init__(
        self,
        n_kc: int,
        max_len: int = 50,
        d_model: int = 32,
        d_ff: int = 64,
        n_heads: int = 2,
        n_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pad_kc = n_kc
        self.max_len = max_len
        self.item_emb = nn.Embedding(n_kc + 1, d_model, padding_idx=self.pad_kc)
        # 0=序列开始（占位）、1=答错、2=答对
        self.resp_emb = nn.Embedding(3, d_model)
        self.pos_emb = nn.Embedding(max_len + 1, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.head = nn.Linear(d_model, 1)

    def forward(self, kc: torch.Tensor, resp: torch.Tensor) -> torch.Tensor:
        B, L = kc.shape
        prev = torch.zeros_like(resp)
        prev[:, 1:] = resp[:, :-1]
        pos = torch.arange(L, device=kc.device).unsqueeze(0).expand(B, L)

        x = self.item_emb(kc) + self.resp_emb(prev + 1) + self.pos_emb(pos)
        x = self.drop(x)

        # 因果掩码：位置 t 只能看到 ≤ t
        causal = torch.triu(torch.ones(L, L, device=kc.device, dtype=torch.bool), diagonal=1)
        pad_mask = kc == self.pad_kc            # True 表示需要屏蔽
        h = self.encoder(x, mask=causal, src_key_padding_mask=pad_mask)
        return self.head(self.drop(h)).squeeze(-1)


class SAKT(TorchKnowledgeTracer):
    name = "SAKT"

    def __init__(self, n_kc: int, device: str | None = None, **kwargs):
        super().__init__(device)
        cfg = default_profile_kwargs()
        cfg.update({k: v for k, v in kwargs.items() if v is not None})
        self.n_kc = n_kc
        self.pad_kc = n_kc
        self.config = cfg
        self.model = SAKTModule(
            n_kc=n_kc,
            max_len=cfg["max_len"],
            d_model=cfg["d_model"],
            d_ff=cfg["d_ff"],
            n_heads=cfg["n_heads"],
            n_blocks=cfg["n_blocks"],
            dropout=cfg["dropout"],
        )
        self.history: dict = {}

    def fit(self, sequences: list[dict], valid: list[dict] | None = None) -> "SAKT":
        if valid is None:
            valid = sequences
        torch.manual_seed(self.config["seed"])
        self.history = train_loop(
            self.model,
            train_seqs=sequences,
            valid_seqs=valid,
            pad_kc=self.pad_kc,
            device=self.device,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            lr=self.config["lr"],
            max_len=self.config["max_len"],
            seed=self.config["seed"],
        )
        return self

    def predict(self, sequences: list[dict]) -> list[np.ndarray]:
        return predict_logits(
            self.model,
            sequences,
            pad_kc=self.pad_kc,
            device=self.device,
            batch_size=max(256, self.config["batch_size"] * 4),
            max_len=self.config["max_len"],
        )

    def state_dict(self) -> dict:
        return {"name": self.name, "n_kc": self.n_kc, "config": self.config}

    @classmethod
    def from_state_dict(cls, payload: dict) -> "SAKT":
        obj = cls(n_kc=payload["n_kc"], **payload.get("config", {}))
        return obj
