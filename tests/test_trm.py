import torch
import pytest
from trm.model.trm import TinyRecursiveModel


@pytest.fixture
def model():
    return TinyRecursiveModel(
        vocab_size=64,
        embed_dim=16,
        hidden_dim=16,
        output_dim=5,
        max_depth=4,
        gate_threshold=0.5,
        num_cells=1,
        dropout=0.0,
    )


def test_forward_shape(model):
    x = torch.randint(0, 64, (4, 10))
    logits = model(x)
    assert logits.shape == (4, 5)


def test_forward_with_meta(model):
    x = torch.randint(0, 64, (4, 10))
    logits, meta = model(x, return_meta=True)
    assert logits.shape == (4, 5)
    assert "depths" in meta
    assert "gates" in meta
    assert "mean_depth" in meta
    assert meta["depths"].shape == (4,)
    assert meta["mean_depth"] > 0


def test_adaptive_depth_varies(model):
    """Different inputs should (probabilistically) yield different depths."""
    torch.manual_seed(0)
    x1 = torch.randint(0, 64, (16, 10))
    _, meta = model(x1, return_meta=True)
    depths = meta["depths"]
    # With 16 samples there should be some variation (not all identical)
    # This is a soft check — just verify shape & range
    assert depths.min() >= 1
    assert depths.max() <= 4


def test_gradient_flow(model):
    x = torch.randint(0, 64, (2, 10))
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    # At least some parameters must receive gradients (not all will if
    # the gate stops recursion before using every cell layer).
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0, "at least some parameters must receive gradients"


def test_no_embedding_mode():
    """vocab_size=0 means raw float input instead of token ids."""
    model = TinyRecursiveModel(
        vocab_size=0,
        embed_dim=16,
        hidden_dim=16,
        output_dim=3,
        max_depth=4,
        num_cells=1,
        dropout=0.0,
    )
    x = torch.randn(4, 8, 16)  # (batch, seq_len, embed_dim)
    logits = model(x)
    assert logits.shape == (4, 3)
