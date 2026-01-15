## run locally
### backend
```bash
uv run uvicorn wizard_ops.api:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000
```

### frontend
```bash
uv run streamlit run src/wizard_ops/frontend/frontend.py
```

## docker
### backend
```bash
docker build -t backend -f dockerfiles/api.dockerfile .
```

### frontend
```bash
docker build -t acherrydev/wizard_ops_fe -f dockerfiles/frontend.dockerfile . 
```

#### build and run
```bash
docker build -t acherrydev/wizard_ops_fe -f dockerfiles/frontend.dockerfile . \ 
&& docker run --rm -p 8001:8001 -e "PORT=8001" acherrydev/wizard_ops_fe
```

#### publish
```bash
docker push acherrydev/wizard_ops_fe
```
