FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

COPY uv.lock uv.lock
COPY ../src/wizard_ops/backend/pyproject.toml pyproject.toml

COPY src/wizard_ops/backend/api.py api.py


RUN uv sync --frozen

ENTRYPOINT ["sh", "-c", "uv run uvicorn api:app --host 0.0.0.0 --port $PORT"]
