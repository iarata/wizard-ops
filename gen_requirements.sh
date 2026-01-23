uv pip freeze > requirements.txt

uv pip compile pyproject.toml \
  --group dev \
  -o requirements_dev.txt