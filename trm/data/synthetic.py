"""Synthetic dataset that benefits from recursive / adaptive‑depth processing.

The task: given a sequence of tokens, compute a *nested reduction*.
  - depth‑1 sequences: direct sum‑mod‑C
  - depth‑2 sequences: sum of sub‑group sums, mod C
  - depth‑N sequences: recursively nested groups

Harder sequences naturally require more recursion steps, which lets us
verify that the TRM learns to allocate depth adaptively.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
from torch.utils.data import Dataset


class SyntheticRecursionDataset(Dataset):
    """Generate sequences that need recursive reduction to solve.

    Parameters
    ----------
    num_samples   : number of samples
    seq_len       : fixed sequence length
    vocab_size    : token vocabulary size
    num_classes   : output classes (result is mod num_classes)
    max_nest      : maximum nesting depth for the reduction
    seed          : reproducibility
    """

    def __init__(
        self,
        num_samples: int = 10_000,
        seq_len: int = 32,
        vocab_size: int = 256,
        num_classes: int = 10,
        max_nest: int = 4,
        seed: Optional[int] = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.max_nest = max_nest

        rng = random.Random(seed)
        self.data: list[tuple[list[int], int, int]] = []

        for _ in range(num_samples):
            tokens = [rng.randint(0, vocab_size - 1) for _ in range(seq_len)]
            nest_depth = rng.randint(1, max_nest)
            label = self._nested_reduce(tokens, nest_depth, num_classes)
            self.data.append((tokens, label, nest_depth))

    @staticmethod
    def _nested_reduce(tokens: list[int], depth: int, mod: int) -> int:
        """Recursively reduce *tokens* by splitting into halves and summing."""
        if depth <= 1 or len(tokens) <= 1:
            return sum(tokens) % mod
        mid = len(tokens) // 2
        left = SyntheticRecursionDataset._nested_reduce(
            tokens[:mid], depth - 1, mod
        )
        right = SyntheticRecursionDataset._nested_reduce(
            tokens[mid:], depth - 1, mod
        )
        return (left + right) % mod

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        tokens, label, nest_depth = self.data[idx]
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "nest_depth": torch.tensor(nest_depth, dtype=torch.long),
        }
