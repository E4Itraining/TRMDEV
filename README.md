# TRM — Tiny Recursive Model

A minimal recursive neural network with **adaptive-depth computation** packaged in Docker, with a **live Dash dashboard** for real-time training graphs.

## Architecture

```
Input tokens → Embedding → [RecursiveCell × N] → Output head
                               ↑         │
                               └─── gate ─┘  (recurse if gate > threshold)
```

**Key ideas:**
- **RecursiveCell**: GRU-style cell with a learned recursion gate
- **Adaptive depth**: easy samples exit early, hard samples use more recursion steps
- **Live graphs**: 6-panel Dash dashboard updates every 2s during training

## Quick Start

### Docker (recommended)

```bash
# CPU
docker compose up --build

# GPU (NVIDIA)
docker compose --profile gpu up --build trm-gpu
```

Open **http://localhost:8050** to see live training graphs.

### Local

```bash
pip install -r requirements.txt
python train.py --config configs/default.yaml
```

### Helper script

```bash
./scripts/run.sh                          # CPU, default config
./scripts/run.sh configs/custom.yaml gpu  # GPU, custom config
```

## Dashboard Panels

| Panel | Description |
|-------|-------------|
| Loss | Train + validation loss curves |
| Accuracy | Train + validation accuracy |
| Recursion Depth | Mean adaptive depth over training |
| Gate Histogram | Distribution of gate values (latest step) |
| Learning Rate | Warmup + cosine decay schedule |
| Epoch Time | Wall-clock time per epoch |

## Configuration

Edit `configs/default.yaml`:

```yaml
model:
  hidden_dim: 64      # increase for more capacity
  max_depth: 8        # hard cap on recursion
  gate_threshold: 0.5 # lower → deeper recursion
  num_cells: 2        # stacked recursive cells

training:
  epochs: 20
  batch_size: 128
  lr: 3.0e-4

graph_server:
  enabled: true
  port: 8050
  refresh_interval_ms: 2000
```

## Project Structure

```
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── train.py                 # Training entry point
├── configs/
│   └── default.yaml         # Default hyperparameters
├── trm/
│   ├── model/
│   │   ├── recursive_cell.py  # GRU + recursion gate
│   │   └── trm.py             # Full model with adaptive depth
│   ├── data/
│   │   └── synthetic.py       # Nested-reduction dataset
│   └── utils/
│       └── metrics.py         # Thread-safe metrics store
├── graphs/
│   └── server.py            # Dash live dashboard
└── scripts/
    └── run.sh               # Helper launch script
```
