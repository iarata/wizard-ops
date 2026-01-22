FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y \
    build-essential gcc git \
    libglib2.0-0 libgl1 \
    libxext6 libxrender1 libxcb1 && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install DVC with GCS support
RUN pip install dvc "dvc[gs]"

# Copy project files first (for dependency resolution)
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# Generate lock file (since uv.lock is not tracked in git)
RUN uv lock

# Copy source code and configs
COPY src/ src/
COPY configs/ configs/

# Copy DVC tracked metafiles (needed for dvc pull/push at runtime)
COPY data.nosync.dvc data.nosync.dvc
COPY checkpoints.dvc checkpoints.dvc

# Sync Python dependencies
RUN uv sync --no-cache

# Copy DVC configuration directory
COPY .dvc/ .dvc/

# Configure DVC for no-scm mode (don't reinit since .dvc already exists)
RUN uv run dvc config core.no_scm true

# Default environment variables (can be overridden at runtime)
# Note: Sensitive values like WANDB_API_KEY should be passed at runtime, not baked into image
ENV EXPERIMENT_NAME=default_experiment
ENV BACKBONE=resnet18
ENV MAX_EPOCHS=10
ENV BATCH_SIZE=32
ENV FAST_DEV_RUN=false
# WANDB_API_KEY should be passed at runtime via -e WANDB_API_KEY=xxx
# GOOGLE_APPLICATION_CREDENTIALS should be mounted at runtime

# Training entrypoint script
COPY dockerfiles/train_entrypoint.sh /app/train_entrypoint.sh
RUN chmod +x /app/train_entrypoint.sh

ENTRYPOINT ["/app/train_entrypoint.sh"]
