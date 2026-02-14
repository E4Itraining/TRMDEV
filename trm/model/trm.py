"""Tiny Recursive Model (TRM).

The TRM wraps one or several RecursiveCells and applies them in a
depth‑adaptive loop: at each step the recursion gate decides whether to
continue recursing or to stop early.  This gives the model a form of
*adaptive compute* — easy inputs use fewer steps, hard inputs use more.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .recursive_cell import RecursiveCell


class TinyRecursiveModel(nn.Module):
    """Tiny Recursive Model with adaptive‑depth recursion.

    Parameters
    ----------
    vocab_size   : int   — size of the input vocabulary (0 for raw vectors)
    embed_dim    : int   — embedding dimension
    hidden_dim   : int   — hidden‑state dimension of the recursive cell
    output_dim   : int   — number of output classes / regression targets
    max_depth    : int   — hard cap on recursion depth
    gate_threshold : float — if gate < threshold the recursion stops early
    num_cells    : int   — number of stacked recursive cells
    dropout      : float — dropout probability
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 10,
        max_depth: int = 8,
        gate_threshold: float = 0.5,
        num_cells: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.gate_threshold = gate_threshold
        self.hidden_dim = hidden_dim

        # Embedding
        if vocab_size > 0:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        else:
            self.embedding = None

        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Recursive cells (stacked)
        self.cells = nn.ModuleList(
            [RecursiveCell(hidden_dim, dropout) for _ in range(num_cells)]
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # For logging / graph visualisation
        self._last_depths: list[int] = []
        self._last_gates: list[list[float]] = []

    # ------------------------------------------------------------------
    def _init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, return_meta: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        x : (batch, seq_len) of token ids  **or**  (batch, seq_len, embed_dim)
        return_meta : if True, also return recursion metadata

        Returns
        -------
        logits : (batch, output_dim)
        meta   : dict with recursion stats (optional)
        """
        if self.embedding is not None and x.dtype in (torch.long, torch.int):
            x = self.embedding(x)  # (B, S, E)

        # Pool over sequence dimension → (B, E)
        x = x.mean(dim=1) if x.dim() == 3 else x

        x = self.input_proj(x)  # (B, H)

        batch_size = x.size(0)
        device = x.device

        h = self._init_hidden(batch_size, device)

        all_gates: list[torch.Tensor] = []
        depths = torch.zeros(batch_size, device=device)
        active = torch.ones(batch_size, dtype=torch.bool, device=device)

        for depth in range(self.max_depth):
            for cell in self.cells:
                h_new, gate = cell(x, h)

            gate_squeezed = gate.squeeze(-1)  # (B,)
            all_gates.append(gate_squeezed.detach())

            # Update only active samples
            h = torch.where(active.unsqueeze(-1), h_new, h)
            depths = depths + active.float()

            # Decide who stops
            stop = gate_squeezed < self.gate_threshold
            active = active & ~stop

            if not active.any():
                break

        logits = self.output_head(h)

        # Store metadata for visualisation
        self._last_depths = depths.detach().cpu().tolist()
        self._last_gates = [g.cpu().tolist() for g in all_gates]

        if return_meta:
            meta = {
                "depths": depths.detach(),
                "gates": all_gates,
                "mean_depth": depths.mean().item(),
            }
            return logits, meta

        return logits
