FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

# Install only the dependencies needed for frontend (not the full wizard_ops package)
RUN uv pip install --system \
    streamlit>=1.30.0 \
    requests>=2.32.5 \
    google-cloud-run>=0.14.0

# Copy frontend code
COPY src/wizard_ops/frontend/frontend.py frontend.py

# Environment variable for backend URL (can be overridden at runtime)
ENV WIZARD_BACKEND=""

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

ENTRYPOINT ["sh", "-c", "exec streamlit run frontend.py --server.address=0.0.0.0 --server.fileWatcherType=none --server.port=${PORT:-8501}"]
