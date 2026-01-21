FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

COPY ../src/wizard_ops/backend/pyproject.toml pyproject.toml

RUN uv sync --no-install-project
#RUN uv sync --frozen --no-dev #Expected a Python module at: /src/wizard_ops/__init__.py

COPY src src/
COPY README.md README.md


RUN uv sync --frozen

ENTRYPOINT ["sh", "-c", "uv run uvicorn api:app --host 0.0.0.0 --port $PORT"]
