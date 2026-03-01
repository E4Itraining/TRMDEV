#!/usr/bin/env python3
"""
Lab 03 — Comparaison d'expériences avec Phoenix
=================================================

Objectif : Lancer plusieurs entraînements avec des hyperparamètres
différents et comparer les résultats dans Phoenix.

Ce script :
  1. Démarre Phoenix
  2. Lance 3 entraînements courts avec des gate_threshold différents
  3. Chaque run est un projet séparé dans les traces
  4. Permet de comparer les profondeurs de récursion dans Phoenix

Variantes testées :
  A) gate_threshold = 0.3  → récursion profonde (le modèle recurse plus)
  B) gate_threshold = 0.5  → comportement par défaut
  C) gate_threshold = 0.7  → récursion courte (le modèle s'arrête vite)

Usage:
    cd /home/user/TRMDEV
    python labs/observability-phoenix/03_experiments.py
"""

from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from trm.data import SyntheticRecursionDataset
from trm.model import TinyRecursiveModel

import phoenix as px
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def setup_phoenix() -> str:
    session = px.launch_app()
    phoenix_url = session.url
    print(f"[Phoenix] UI : {phoenix_url}")
    return phoenix_url


def create_tracer(phoenix_url: str, experiment_name: str) -> trace.Tracer:
    """Crée un tracer avec un service.name unique par expérience."""
    resource = Resource.create({
        "service.name": f"trm-experiment-{experiment_name}",
        "service.version": "1.0.0",
    })
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=f"{phoenix_url}v1/traces")
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Note : on ne fait PAS set_tracer_provider global ici,
    # on utilise le provider directement via get_tracer
    return provider.get_tracer("trm.experiment")


def load_config() -> dict:
    config_path = ROOT / "labs" / "observability-phoenix" / "lab_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_lr_lambda(warmup: int):
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return max(step, 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, 1000 - warmup)))
    return lr_lambda


def run_experiment(
    cfg: dict,
    tracer: trace.Tracer,
    experiment_name: str,
    gate_threshold: float,
) -> dict:
    """Lance un entraînement court et retourne les métriques finales."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Adapter la config pour cette expérience
    cfg = copy.deepcopy(cfg)
    cfg["model"]["gate_threshold"] = gate_threshold
    # Entraînement très court pour la démo
    cfg["training"]["epochs"] = 3
    cfg["data"]["num_train"] = 1000
    cfg["data"]["num_val"] = 250

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

    mcfg = cfg["model"]
    model = TinyRecursiveModel(
        vocab_size=mcfg["vocab_size"],
        embed_dim=mcfg["embed_dim"],
        hidden_dim=mcfg["hidden_dim"],
        output_dim=mcfg["output_dim"],
        max_depth=mcfg["max_depth"],
        gate_threshold=gate_threshold,
        num_cells=mcfg["num_cells"],
        dropout=mcfg["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, get_lr_lambda(tcfg["warmup_steps"])
    )

    global_step = 0
    final_metrics = {}

    with tracer.start_as_current_span("experiment") as exp_span:
        exp_span.set_attribute("experiment.name", experiment_name)
        exp_span.set_attribute("experiment.gate_threshold", gate_threshold)
        exp_span.set_attribute("model.max_depth", mcfg["max_depth"])
        exp_span.set_attribute("model.num_cells", mcfg["num_cells"])

        for epoch in range(1, tcfg["epochs"] + 1):
            with tracer.start_as_current_span("epoch") as epoch_span:
                epoch_span.set_attribute("epoch.number", epoch)
                model.train()
                epoch_start = time.time()

                for batch in train_loader:
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["label"].to(device)

                    with tracer.start_as_current_span("step") as step_span:
                        logits, meta = model(input_ids, return_meta=True)
                        loss = criterion(logits, labels)

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()

                        global_step += 1
                        acc = (logits.argmax(-1) == labels).float().mean().item()

                        step_span.set_attribute("step.loss", round(loss.item(), 6))
                        step_span.set_attribute("step.accuracy", round(acc, 4))
                        step_span.set_attribute("step.mean_depth", round(meta["mean_depth"], 4))

                        depths = meta["depths"].cpu().tolist()
                        step_span.set_attribute("recursion.min_depth", min(depths))
                        step_span.set_attribute("recursion.max_depth", max(depths))

                elapsed = time.time() - epoch_start
                epoch_span.set_attribute("epoch.duration_s", round(elapsed, 2))

        # Validation finale
        with tracer.start_as_current_span("final_validation") as val_span:
            model.eval()
            val_loss_sum, val_correct, val_total = 0.0, 0, 0
            depth_sum = 0.0
            depth_count = 0

            with torch.no_grad():
                for vb in val_loader:
                    vi = vb["input_ids"].to(device)
                    vl = vb["label"].to(device)
                    vlogits, vmeta = model(vi, return_meta=True)
                    val_loss_sum += criterion(vlogits, vl).item() * vi.size(0)
                    val_correct += (vlogits.argmax(-1) == vl).sum().item()
                    val_total += vi.size(0)
                    depth_sum += vmeta["mean_depth"] * vi.size(0)
                    depth_count += vi.size(0)

            v_loss = val_loss_sum / max(val_total, 1)
            v_acc = val_correct / max(val_total, 1)
            v_depth = depth_sum / max(depth_count, 1)

            val_span.set_attribute("val.loss", round(v_loss, 6))
            val_span.set_attribute("val.accuracy", round(v_acc, 4))
            val_span.set_attribute("val.mean_depth", round(v_depth, 4))
            val_span.set_attribute("val.num_samples", val_total)

            final_metrics = {
                "loss": round(v_loss, 4),
                "accuracy": round(v_acc, 4),
                "mean_depth": round(v_depth, 2),
            }

        exp_span.set_attribute("result.val_loss", final_metrics["loss"])
        exp_span.set_attribute("result.val_accuracy", final_metrics["accuracy"])
        exp_span.set_attribute("result.val_mean_depth", final_metrics["mean_depth"])

    return final_metrics


def main():
    print("=" * 60)
    print("  Lab 03 — Comparaison d'expériences avec Phoenix")
    print("=" * 60)
    print()

    phoenix_url = setup_phoenix()
    cfg = load_config()

    # ── Définir les expériences ──
    experiments = [
        ("deep",    0.3, "Récursion profonde (gate_threshold=0.3)"),
        ("default", 0.5, "Comportement par défaut (gate_threshold=0.5)"),
        ("shallow", 0.7, "Récursion courte (gate_threshold=0.7)"),
    ]

    results = {}

    for name, threshold, description in experiments:
        print(f"\n{'─' * 60}")
        print(f"  Expérience : {description}")
        print(f"{'─' * 60}\n")

        tracer = create_tracer(phoenix_url, name)
        metrics = run_experiment(cfg, tracer, name, threshold)
        results[name] = metrics

        print(f"  → val_loss={metrics['loss']:.4f}  "
              f"val_acc={metrics['accuracy']:.4f}  "
              f"mean_depth={metrics['mean_depth']:.2f}")

    # ── Résumé ──
    print(f"\n{'=' * 60}")
    print("  Résumé des expériences")
    print(f"{'=' * 60}")
    print(f"  {'Expérience':<12} {'Threshold':>10} {'Val Loss':>10} {'Val Acc':>10} {'Depth':>8}")
    print(f"  {'─' * 52}")
    for (name, threshold, _) in experiments:
        m = results[name]
        print(f"  {name:<12} {threshold:>10.1f} {m['loss']:>10.4f} {m['accuracy']:>10.4f} {m['mean_depth']:>8.2f}")

    print(f"\n{'=' * 60}")
    print(f"  Explorez les résultats dans Phoenix :")
    print(f"  → {phoenix_url}")
    print(f"{'=' * 60}")
    print()
    print("  Dans l'UI Phoenix, comparez les projets :")
    print("  • trm-experiment-deep     → profondeurs élevées")
    print("  • trm-experiment-default  → comportement standard")
    print("  • trm-experiment-shallow  → profondeurs faibles")
    print()
    print("  Points d'observation :")
    print("  • Comment gate_threshold affecte la profondeur moyenne ?")
    print("  • Quel impact sur la loss et l'accuracy ?")
    print("  • Quel compromis profondeur/performance ?")
    print()
    print("  Appuyez sur Ctrl+C pour arrêter.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Phoenix] Arrêt.")


if __name__ == "__main__":
    main()
