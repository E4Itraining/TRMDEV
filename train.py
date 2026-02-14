#!/usr/bin/env python3
"""Training script for the Tiny Recursive Model.

Launches training in a background thread and (optionally) starts a Dash
graph server so that metrics can be visualised in real time.
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from trm.data import SyntheticRecursionDataset
from trm.model import TinyRecursiveModel
from trm.utils import MetricsStore


# ── helpers ───────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_lr_lambda(warmup: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return max(step, 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, 1000 - warmup)))
    return lr_lambda


# ── training loop ─────────────────────────────────────────────────────
def train(cfg: dict, metrics: MetricsStore) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRM] device = {device}")

    # Data
    dcfg = cfg["data"]
    train_ds = SyntheticRecursionDataset(
        num_samples=dcfg["num_train"],
        seq_len=dcfg["seq_len"],
        vocab_size=cfg["model"]["vocab_size"],
        num_classes=dcfg["num_classes"],
        max_nest=dcfg["max_nest"],
        seed=dcfg["seed"],
    )
    val_ds = SyntheticRecursionDataset(
        num_samples=dcfg["num_val"],
        seq_len=dcfg["seq_len"],
        vocab_size=cfg["model"]["vocab_size"],
        num_classes=dcfg["num_classes"],
        max_nest=dcfg["max_nest"],
        seed=dcfg["seed"] + 1,
    )

    tcfg = cfg["training"]
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False
    )

    # Model
    mcfg = cfg["model"]
    model = TinyRecursiveModel(
        vocab_size=mcfg["vocab_size"],
        embed_dim=mcfg["embed_dim"],
        hidden_dim=mcfg["hidden_dim"],
        output_dim=mcfg["output_dim"],
        max_depth=mcfg["max_depth"],
        gate_threshold=mcfg["gate_threshold"],
        num_cells=mcfg["num_cells"],
        dropout=mcfg["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[TRM] total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, get_lr_lambda(tcfg["warmup_steps"])
    )

    global_step = 0
    save_dir = Path(tcfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        epoch_start = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits, meta = model(input_ids, return_meta=True)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            acc = (logits.argmax(-1) == labels).float().mean().item()

            if global_step % tcfg["log_every"] == 0:
                # Flatten all gate values for histogram
                flat_gates = []
                for g in meta["gates"]:
                    flat_gates.extend(g.cpu().tolist())

                metrics.log_step(
                    step=global_step,
                    loss=loss.item(),
                    accuracy=acc,
                    mean_depth=meta["mean_depth"],
                    gates=flat_gates,
                    lr=scheduler.get_last_lr()[0],
                )
                print(
                    f"  [step {global_step:>5d}] loss={loss.item():.4f}  "
                    f"acc={acc:.3f}  depth={meta['mean_depth']:.2f}"
                )

            # Validation
            if global_step % tcfg["val_every"] == 0:
                model.eval()
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for vb in val_loader:
                        vi = vb["input_ids"].to(device)
                        vl = vb["label"].to(device)
                        vlogits = model(vi)
                        val_loss_sum += criterion(vlogits, vl).item() * vi.size(0)
                        val_correct += (vlogits.argmax(-1) == vl).sum().item()
                        val_total += vi.size(0)
                v_loss = val_loss_sum / max(val_total, 1)
                v_acc = val_correct / max(val_total, 1)
                metrics.log_validation(global_step, v_loss, v_acc)
                print(
                    f"  [val   {global_step:>5d}] loss={v_loss:.4f}  acc={v_acc:.3f}"
                )
                model.train()

        elapsed = time.time() - epoch_start
        metrics.log_epoch_time(elapsed)
        print(f"[epoch {epoch}/{tcfg['epochs']}] done  ({elapsed:.1f}s)")

        # Save checkpoint
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "step": global_step},
            save_dir / f"trm_epoch{epoch:03d}.pt",
        )

    metrics.save(save_dir / "metrics.json")
    print("[TRM] training complete — metrics saved.")


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRM")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="YAML config"
    )
    parser.add_argument(
        "--no-graph", action="store_true", help="Disable live graph server"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics = MetricsStore()

    # Start graph server in a background thread (unless disabled)
    if cfg.get("graph_server", {}).get("enabled") and not args.no_graph:
        from graphs.server import create_app

        gs = cfg["graph_server"]
        app = create_app(metrics)

        def run_server():
            app.run(host=gs["host"], port=gs["port"], debug=False)

        t = threading.Thread(target=run_server, daemon=True)
        t.start()
        print(f"[TRM] graph server running on http://{gs['host']}:{gs['port']}")

    train(cfg, metrics)


if __name__ == "__main__":
    main()
