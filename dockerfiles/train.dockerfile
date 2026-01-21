FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install dvc dvc[gs] 

WORKDIR /app

COPY uv.lock pyproject.toml README.md ./
RUN uv sync --locked --no-cache --no-install-project

COPY src/ src/
COPY .dvc/config .dvc/config
COPY *.dvc ./ 

RUN uv run dvc config core.no_scm true

# RUN uv run dvc remote modify dtu_kfc_gs --local \
#     gdrive_service_account_json_file_path /default.json

RUN uv run dvc pull

ENTRYPOINT ["uv", "run", "wizard_ops", "train"]