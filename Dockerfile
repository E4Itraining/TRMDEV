FROM python:3.11-slim

LABEL maintainer="E4Itraining"
LABEL description="TRM — Tiny Recursive Model with live graph dashboard"

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Dash dashboard port
EXPOSE 8050

# Health check — ping the Dash server
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8050/ || exit 1

ENTRYPOINT ["python", "-u", "train.py"]
CMD ["--config", "configs/default.yaml"]
