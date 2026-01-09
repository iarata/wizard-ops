FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/
COPY README.md README.md

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/wizard_ops/train.py"]
