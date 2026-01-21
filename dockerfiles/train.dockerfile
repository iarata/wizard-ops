FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install dvc dvc[gs] 

WORKDIR /app

RUN uv sync --locked --no-cache --no-install-project

COPY uv.lock uv.lock
COPY pyproject.toml README.md ./
COPY dtumlops-484413-083ba11aaab8.json default.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/default.json
COPY src/ src/
COPY .dvc/config .dvc/config
COPY *.dvc ./ 

RUN uv run dvc config core.no_scm true

RUN uv run dvc pull

ENTRYPOINT ["uv", "run", "src/wizard_ops", "train"]