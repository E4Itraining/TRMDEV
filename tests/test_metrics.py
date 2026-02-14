import json
import tempfile
from pathlib import Path

import pytest
from trm.utils.metrics import MetricsStore


@pytest.fixture
def store():
    return MetricsStore()


def test_log_step(store):
    store.log_step(step=1, loss=2.3, accuracy=0.1, mean_depth=3.0)
    snap = store.snapshot()
    assert snap["steps"] == [1]
    assert snap["losses"] == [2.3]
    assert snap["accuracies"] == [0.1]
    assert snap["mean_depths"] == [3.0]


def test_log_step_with_gates_and_lr(store):
    store.log_step(step=1, loss=1.0, accuracy=0.5, mean_depth=2.0, gates=[0.3, 0.7], lr=1e-3)
    snap = store.snapshot()
    assert snap["gate_histograms"] == [[0.3, 0.7]]
    assert snap["learning_rates"] == [1e-3]


def test_log_validation(store):
    store.log_validation(step=10, val_loss=1.5, val_accuracy=0.4)
    snap = store.snapshot()
    assert snap["val_steps"] == [10]
    assert snap["val_losses"] == [1.5]
    assert snap["val_accuracies"] == [0.4]


def test_log_epoch_time(store):
    store.log_epoch_time(12.5)
    store.log_epoch_time(11.3)
    snap = store.snapshot()
    assert snap["epoch_times"] == [12.5, 11.3]


def test_save_and_load(store):
    store.log_step(step=1, loss=1.0, accuracy=0.5, mean_depth=2.0)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sub" / "metrics.json"
        store.save(path)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["steps"] == [1]


def test_snapshot_is_copy(store):
    store.log_step(step=1, loss=1.0, accuracy=0.5, mean_depth=2.0)
    snap = store.snapshot()
    snap["steps"].append(999)
    assert store.snapshot()["steps"] == [1], "snapshot must return a copy"


def test_multiple_steps(store):
    for i in range(5):
        store.log_step(step=i, loss=float(i), accuracy=0.1 * i, mean_depth=float(i))
    snap = store.snapshot()
    assert len(snap["steps"]) == 5
    assert snap["steps"] == [0, 1, 2, 3, 4]
