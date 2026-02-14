"""Thread‑safe metrics store used by the trainer and the graph server."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MetricsStore:
    """Accumulates training metrics and exposes them as JSON‑serialisable dicts.

    The store is protected by a lock so the Dash callback thread can read
    while the training loop writes.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    mean_depths: list[float] = field(default_factory=list)
    gate_histograms: list[list[float]] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    val_steps: list[int] = field(default_factory=list)

    def log_step(
        self,
        step: int,
        loss: float,
        accuracy: float,
        mean_depth: float,
        gates: list[float] | None = None,
        lr: float | None = None,
    ) -> None:
        with self._lock:
            self.steps.append(step)
            self.losses.append(loss)
            self.accuracies.append(accuracy)
            self.mean_depths.append(mean_depth)
            if gates is not None:
                self.gate_histograms.append(gates)
            if lr is not None:
                self.learning_rates.append(lr)

    def log_validation(
        self, step: int, val_loss: float, val_accuracy: float
    ) -> None:
        with self._lock:
            self.val_steps.append(step)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

    def log_epoch_time(self, t: float) -> None:
        with self._lock:
            self.epoch_times.append(t)

    def snapshot(self) -> dict:
        """Return a JSON‑serialisable snapshot of all metrics."""
        with self._lock:
            return {
                "steps": list(self.steps),
                "losses": list(self.losses),
                "accuracies": list(self.accuracies),
                "mean_depths": list(self.mean_depths),
                "gate_histograms": [list(g) for g in self.gate_histograms],
                "learning_rates": list(self.learning_rates),
                "epoch_times": list(self.epoch_times),
                "val_steps": list(self.val_steps),
                "val_losses": list(self.val_losses),
                "val_accuracies": list(self.val_accuracies),
            }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.snapshot(), f, indent=2)
