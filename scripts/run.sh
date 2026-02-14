#!/usr/bin/env bash
# ──────────────────────────────────────────────
#  Quick‑start script for TRM
# ──────────────────────────────────────────────
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
MODE="${2:-cpu}"

echo "╔══════════════════════════════════════════╗"
echo "║  TRM — Tiny Recursive Model              ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Config : $CONFIG"
echo "║  Mode   : $MODE"
echo "╚══════════════════════════════════════════╝"

if [ "$MODE" = "gpu" ]; then
    echo "[*] Launching with GPU support..."
    docker compose --profile gpu up --build trm-gpu
else
    echo "[*] Launching CPU mode..."
    docker compose up --build trm
fi
