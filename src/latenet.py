"""LateNet: an embedding per categorical input, joined with the numeric inputs,
then a small MLP that outputs one raw score (logit) per trip.

No sigmoid at the end: training uses BCEWithLogitsLoss, which applies it inside
the loss. Apply torch.sigmoid yourself only when you need a probability.
"""
from __future__ import annotations

import torch
from torch import nn

from features import NUMERIC


class LateNet(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], embedding_dims: dict[str, int],
                 hidden: list[int], dropout: float):
        """vocab_sizes: known values per column (len of its vocabulary, without 0).
        embedding_dims, hidden, dropout: from the config."""
        super().__init__()
        self.columns = list(vocab_sizes)        # the categorical inputs this model uses, in order
        # One table per column: row 0 = unknown, kept at zero (padding_idx=0).
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_sizes[col] + 1, embedding_dims[col], padding_idx=0)
            for col in self.columns
        })
        width = sum(embedding_dims[col] for col in self.columns) + len(NUMERIC)
        layers = []
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 1))      # one logit per trip, no sigmoid
        self.mlp = nn.Sequential(*layers)

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        """categorical: (batch, n_columns) int64 in self.columns order; numeric: (batch, 4)
        float32 in NUMERIC order. Returns (batch,) logits."""
        parts = [self.embeddings[col](categorical[:, i]) for i, col in enumerate(self.columns)]
        x = torch.cat(parts + [numeric], dim=1)
        return self.mlp(x).squeeze(1)
