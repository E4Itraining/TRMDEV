"""Recursive cell — the atomic building block of TRM.

Each cell receives an input embedding and its own previous hidden state,
then produces a new hidden state **and** a recursion gate that decides
whether to recurse deeper or emit the output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecursiveCell(nn.Module):
    """A single recursive processing cell.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the hidden state.
    dropout : float
        Dropout probability applied inside the cell.
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input‑to‑hidden and hidden‑to‑hidden projections (GRU‑style)
        self.W_z = nn.Linear(hidden_dim * 2, hidden_dim)  # update gate
        self.W_r = nn.Linear(hidden_dim * 2, hidden_dim)  # reset gate
        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim)  # candidate

        # Recursion gate — scalar that controls depth
        self.W_gate = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, hidden_dim) — input embedding
        h : (batch, hidden_dim) — previous hidden state

        Returns
        -------
        h_new  : (batch, hidden_dim) — updated hidden state
        gate   : (batch, 1)          — recursion gate (0→emit, 1→recurse)
        """
        combined = torch.cat([x, h], dim=-1)

        z = torch.sigmoid(self.W_z(combined))       # update gate
        r = torch.sigmoid(self.W_r(combined))       # reset gate

        combined_r = torch.cat([x, r * h], dim=-1)
        h_candidate = torch.tanh(self.W_h(combined_r))

        h_new = (1 - z) * h + z * h_candidate
        h_new = self.norm(self.dropout(h_new))

        gate = torch.sigmoid(self.W_gate(h_new))     # recursion gate
        return h_new, gate
