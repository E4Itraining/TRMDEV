#!/usr/bin/env python3
"""
Lab 02 — Entraînement TRM instrumenté avec Phoenix
====================================================

Objectif : Instrumenter la boucle d'entraînement du TRM avec des traces
OpenTelemetry envoyées à Phoenix. Chaque epoch, step et validation
devient un span explorable dans l'UI.

Ce script :
  1. Démarre Phoenix
  2. Configure le tracing OpenTelemetry
  3. Lance l'entraînement TRM avec instrumentation complète
  4. Chaque span porte des attributs riches (loss, accuracy, depth, etc.)

Usage:
    cd /home/user/TRMDEV
    python labs/observability-phoenix/02_traced_training.py
    python labs/observability-phoenix/02_traced_training.py --config labs/observability-phoenix/lab_config.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Ajouter le répertoire racine au path pour importer trm
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from trm.data import SyntheticRecursionDataset
from trm.model import TinyRecursiveModel

# ── Phoenix + OpenTelemetry ──────────────────────────────────────────
import phoenix as px
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def setup_phoenix() -> str:
    """Démarre Phoenix et configure le tracing OTel."""
    session = px.launch_app()
    phoenix_url = session.url
    print(f"[Phoenix] UI : {phoenix_url}")

    resource = Resource.create({
        "service.name": "trm-training",
        "service.version": "1.0.0",
        "deployment.environment": "lab",
    })
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=f"{phoenix_url}v1/traces")
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    return phoenix_url


# ── Helpers ──────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_lr_lambda(warmup: int):
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return max(step, 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, 1000 - warmup)))
    return lr_lambda


# ── Boucle d'entraînement instrumentée ──────────────────────────────
def traced_train(cfg: dict, tracer: trace.Tracer) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRM] device = {device}")

    # ── Data ──
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
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    # ── Model ──
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

    # ══════════════════════════════════════════════════════════════
    #  SPAN RACINE : training_run — couvre tout l'entraînement
    # ══════════════════════════════════════════════════════════════
    with tracer.start_as_current_span("training_run") as run_span:
        run_span.set_attribute("model.name", "TinyRecursiveModel")
        run_span.set_attribute("model.params", total_params)
        run_span.set_attribute("model.max_depth", mcfg["max_depth"])
        run_span.set_attribute("model.num_cells", mcfg["num_cells"])
        run_span.set_attribute("training.epochs", tcfg["epochs"])
        run_span.set_attribute("training.batch_size", tcfg["batch_size"])
        run_span.set_attribute("training.lr", tcfg["lr"])
        run_span.set_attribute("data.num_train", dcfg["num_train"])
        run_span.set_attribute("data.num_val", dcfg["num_val"])
        run_span.set_attribute("device", str(device))

        for epoch in range(1, tcfg["epochs"] + 1):

            # ── SPAN : epoch ──
            with tracer.start_as_current_span(f"epoch") as epoch_span:
                epoch_span.set_attribute("epoch.number", epoch)
                model.train()
                epoch_start = time.time()
                epoch_loss_sum = 0.0
                epoch_acc_sum = 0.0
                epoch_steps = 0

                for batch in train_loader:
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["label"].to(device)

                    # ── SPAN : training_step ──
                    with tracer.start_as_current_span("training_step") as step_span:
                        step_span.set_attribute("step.global", global_step)

                        # Forward
                        logits, meta = model(input_ids, return_meta=True)
                        loss = criterion(logits, labels)

                        # Backward
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()

                        global_step += 1
                        acc = (logits.argmax(-1) == labels).float().mean().item()

                        # Attributs du span
                        step_span.set_attribute("step.loss", round(loss.item(), 6))
                        step_span.set_attribute("step.accuracy", round(acc, 4))
                        step_span.set_attribute("step.mean_depth", round(meta["mean_depth"], 4))
                        step_span.set_attribute("step.lr", scheduler.get_last_lr()[0])
                        step_span.set_attribute("step.batch_size", input_ids.size(0))

                        # Distribution des profondeurs de récursion
                        depths = meta["depths"].cpu().tolist()
                        step_span.set_attribute("recursion.min_depth", min(depths))
                        step_span.set_attribute("recursion.max_depth", max(depths))

                        epoch_loss_sum += loss.item()
                        epoch_acc_sum += acc
                        epoch_steps += 1

                    if global_step % tcfg["log_every"] == 0:
                        print(
                            f"  [step {global_step:>5d}] loss={loss.item():.4f}  "
                            f"acc={acc:.3f}  depth={meta['mean_depth']:.2f}"
                        )

                    # ── Validation périodique ──
                    if global_step % tcfg["val_every"] == 0:
                        with tracer.start_as_current_span("validation") as val_span:
                            val_span.set_attribute("val.at_step", global_step)
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

                            val_span.set_attribute("val.loss", round(v_loss, 6))
                            val_span.set_attribute("val.accuracy", round(v_acc, 4))
                            val_span.set_attribute("val.num_samples", val_total)

                            print(
                                f"  [val   {global_step:>5d}] loss={v_loss:.4f}  acc={v_acc:.3f}"
                            )
                            model.train()

                elapsed = time.time() - epoch_start
                avg_epoch_loss = epoch_loss_sum / max(epoch_steps, 1)
                avg_epoch_acc = epoch_acc_sum / max(epoch_steps, 1)

                epoch_span.set_attribute("epoch.duration_s", round(elapsed, 2))
                epoch_span.set_attribute("epoch.avg_loss", round(avg_epoch_loss, 6))
                epoch_span.set_attribute("epoch.avg_accuracy", round(avg_epoch_acc, 4))
                epoch_span.set_attribute("epoch.steps", epoch_steps)

                print(f"[epoch {epoch}/{tcfg['epochs']}] done  ({elapsed:.1f}s)")

                # Checkpoint
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(), "step": global_step},
                    save_dir / f"trm_epoch{epoch:03d}.pt",
                )

        run_span.set_attribute("training.final_step", global_step)
        run_span.set_attribute("training.status", "completed")

    print("[TRM] Entraînement terminé — traces envoyées à Phoenix.")


# ── main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Lab 02 — TRM + Phoenix tracing")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "labs" / "observability-phoenix" / "lab_config.yaml"),
        help="YAML config",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Lab 02 — Entraînement TRM avec Phoenix tracing")
    print("=" * 60)
    print()

    phoenix_url = setup_phoenix()
    cfg = load_config(args.config)
    tracer = trace.get_tracer("trm.training")

    traced_train(cfg, tracer)

    print(f"\n{'=' * 60}")
    print(f"  Explorez les traces dans Phoenix :")
    print(f"  → {phoenix_url}")
    print(f"{'=' * 60}")
    print()
    print("  Dans l'UI, vous verrez :")
    print("  • training_run  (span racine)")
    print("  •   └── epoch  (× epochs)")
    print("  •       ├── training_step  (× steps)")
    print("  •       └── validation     (périodique)")
    print()
    print("  Cliquez sur chaque span pour voir :")
    print("  • loss, accuracy, mean_depth à chaque step")
    print("  • durée de chaque epoch")
    print("  • métriques de validation")
    print()
    print("  Appuyez sur Ctrl+C pour arrêter Phoenix.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Phoenix] Arrêt.")


if __name__ == "__main__":
    main()
