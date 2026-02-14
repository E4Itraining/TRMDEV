import torch
import pytest
from trm.model.recursive_cell import RecursiveCell


@pytest.fixture
def cell():
    return RecursiveCell(hidden_dim=32, dropout=0.0)


def test_output_shapes(cell):
    batch = 4
    x = torch.randn(batch, 32)
    h = torch.zeros(batch, 32)
    h_new, gate = cell(x, h)
    assert h_new.shape == (batch, 32)
    assert gate.shape == (batch, 1)


def test_gate_range(cell):
    x = torch.randn(8, 32)
    h = torch.randn(8, 32)
    _, gate = cell(x, h)
    assert (gate >= 0).all() and (gate <= 1).all(), "gate must be in [0, 1]"


def test_hidden_state_changes(cell):
    x = torch.randn(2, 32)
    h = torch.zeros(2, 32)
    h_new, _ = cell(x, h)
    assert not torch.allclose(h_new, h), "hidden state should change after forward"


def test_gradient_flow(cell):
    x = torch.randn(2, 32, requires_grad=True)
    h = torch.zeros(2, 32)
    h_new, gate = cell(x, h)
    loss = h_new.sum() + gate.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
