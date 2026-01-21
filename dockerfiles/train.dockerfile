FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install dvc dvc[gs] 

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY data/ data/

COPY dtumlops-484413-083ba11aaab8.json default.json
COPY .dvc/config .dvc/config
COPY *.dvc .dvc/

WORKDIR /
RUN uv sync --locked --no-cache --no-install-project

RUN uv run dvc init --no-scm
RUN uv run dvc config core.no_scm true
RUN uv run dvc pull

RUN dvc remote modify dtu_kfc_gs --local gdrive_service_account_json_file_path default.json

ENTRYPOINT ["uv", "run", "wizard_ops", "train"]
