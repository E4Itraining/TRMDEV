#!/usr/bin/env python3
"""
Lab 01 — Phoenix Quickstart
============================

Objectif : Lancer Phoenix en local et envoyer des traces manuelles
pour se familiariser avec l'outil.

Ce script :
  1. Démarre Phoenix en arrière-plan
  2. Configure un TracerProvider OpenTelemetry pointant vers Phoenix
  3. Crée quelques spans de démonstration (simulant un pipeline ML)
  4. Affiche l'URL de l'interface Phoenix

Usage:
    python 01_phoenix_quickstart.py
"""

from __future__ import annotations

import time

import phoenix as px
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def setup_phoenix() -> str:
    """Lance Phoenix et configure OpenTelemetry. Retourne l'URL de l'UI."""
    # Lancer Phoenix en arrière-plan
    session = px.launch_app()
    phoenix_url = session.url
    print(f"[Phoenix] UI disponible sur : {phoenix_url}")

    # Configurer OpenTelemetry pour envoyer les traces à Phoenix
    resource = Resource.create({"service.name": "trm-lab-quickstart"})
    provider = TracerProvider(resource=resource)

    # Phoenix écoute par défaut sur http://localhost:6006/v1/traces
    exporter = OTLPSpanExporter(endpoint=f"{phoenix_url}v1/traces")
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    trace.set_tracer_provider(provider)

    return phoenix_url


def demo_traces():
    """Crée des spans de démonstration pour illustrer le tracing."""
    tracer = trace.get_tracer("trm.lab.quickstart")

    # ── Simuler un pipeline ML complet ──────────────────────────
    with tracer.start_as_current_span("pipeline.ml") as pipeline_span:
        pipeline_span.set_attribute("pipeline.name", "demo-quickstart")
        pipeline_span.set_attribute("pipeline.version", "1.0")

        # Étape 1 : Chargement des données
        with tracer.start_as_current_span("data.load") as span:
            span.set_attribute("data.num_samples", 1000)
            span.set_attribute("data.source", "synthetic")
            time.sleep(0.1)  # simuler du travail
            span.set_attribute("data.status", "ok")
            print("  [1/4] Données chargées")

        # Étape 2 : Prétraitement
        with tracer.start_as_current_span("data.preprocess") as span:
            span.set_attribute("preprocess.tokenizer", "char-level")
            span.set_attribute("preprocess.seq_len", 32)
            time.sleep(0.05)
            span.set_attribute("preprocess.status", "ok")
            print("  [2/4] Prétraitement terminé")

        # Étape 3 : Inférence du modèle (3 appels)
        with tracer.start_as_current_span("model.inference_batch") as batch_span:
            batch_span.set_attribute("model.name", "TinyRecursiveModel")
            batch_span.set_attribute("model.max_depth", 8)

            for i in range(3):
                with tracer.start_as_current_span(f"model.forward_pass") as span:
                    span.set_attribute("batch.index", i)
                    span.set_attribute("batch.size", 64)
                    # Simuler des profondeurs de récursion différentes
                    depth = [2.1, 4.7, 3.3][i]
                    span.set_attribute("recursion.mean_depth", depth)
                    span.set_attribute("recursion.max_depth", 8)
                    span.set_attribute("model.accuracy", [0.72, 0.85, 0.91][i])
                    time.sleep(0.08)

            batch_span.set_attribute("inference.total_batches", 3)
            print("  [3/4] Inférence terminée (3 batches)")

        # Étape 4 : Évaluation
        with tracer.start_as_current_span("evaluation") as span:
            span.set_attribute("eval.metric", "accuracy")
            span.set_attribute("eval.value", 0.827)
            span.set_attribute("eval.num_samples", 500)
            time.sleep(0.05)
            print("  [4/4] Évaluation terminée")

        pipeline_span.set_attribute("pipeline.status", "completed")

    print("\n[OK] Traces envoyées à Phoenix !")


def main():
    print("=" * 60)
    print("  Lab 01 — Phoenix Quickstart")
    print("=" * 60)
    print()

    phoenix_url = setup_phoenix()

    print("\n── Envoi de traces de démonstration ──\n")
    demo_traces()

    print(f"\n{'=' * 60}")
    print(f"  Ouvrez Phoenix dans votre navigateur :")
    print(f"  → {phoenix_url}")
    print(f"{'=' * 60}")
    print()
    print("  Dans l'UI Phoenix, explorez :")
    print("  • L'onglet 'Traces' pour voir le pipeline complet")
    print("  • Cliquez sur un span pour voir ses attributs")
    print("  • Observez la hiérarchie parent-enfant des spans")
    print()
    print("  Appuyez sur Ctrl+C pour arrêter Phoenix.")

    # Garder le processus vivant pour que Phoenix reste accessible
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Phoenix] Arrêt.")


if __name__ == "__main__":
    main()
