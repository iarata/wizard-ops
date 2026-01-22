FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

# Build arguments for model configuration
ARG BUCKET_NAME=dtu-kfc-bucket
ARG CHECKPOINT_BLOB=checkpoints/nutrition_resnet18_0115_1951/best-nutrition-epoch=04-val-loss=0.00.ckpt

# Set environment variables
ENV BUCKET_NAME=${BUCKET_NAME}
ENV CHECKPOINT_BLOB=${CHECKPOINT_BLOB}

RUN apt-get update && apt-get install -y \
  libxcb1 \
  libxcb-shm0 \
  libxcb-render0 \
  libgl1-mesa-glx \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml pyproject.toml
COPY README.md README.md

# Generate lock file and sync dependencies (without project itself)
RUN uv lock && uv sync --no-install-project --no-cache

COPY src src/
COPY configs configs/

RUN uv sync --all-packages --no-cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8080}/ || exit 1

ENTRYPOINT ["sh", "-c", "uv run uvicorn wizard_ops.backend.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
