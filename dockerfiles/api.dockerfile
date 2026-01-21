FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project
#RUN uv sync --frozen --no-dev #Expected a Python module at: /src/wizard_ops/__init__.py

COPY src src/
COPY README.md README.md


RUN uv sync --frozen

ENTRYPOINT ["sh", "-c", "uv run uvicorn src.wizard_ops.api:app --host 0.0.0.0 --port $PORT"]
