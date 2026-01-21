FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

RUN apt-get update && apt-get install -y \
  libxcb1 \
  libxcb-shm0 \
  libxcb-render0 \
  libgl1-mesa-glx \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml pyproject.toml

RUN uv sync --no-install-project

COPY src src/
COPY README.md README.md

RUN uv sync --all-packages

ENTRYPOINT ["sh", "-c", "uv run uvicorn wizard_ops.backend.api:app --host 0.0.0.0 --port $PORT"]
