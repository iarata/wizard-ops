FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

COPY ../src/wizard_ops/frontend/pyproject.toml pyproject.toml

COPY src/wizard_ops/frontend/frontend.py frontend.py

RUN uv sync

ENTRYPOINT ["sh", "-c", "echo 'PORT is:' ${PORT} && exec uv run streamlit run frontend.py --server.address=0.0.0.0 --server.fileWatcherType=none --server.port ${PORT}"]
