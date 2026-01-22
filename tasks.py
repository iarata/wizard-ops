import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "wizard_ops"
PYTHON_VERSION = "3.12"

# GCP Configuration (matches configs/config.yaml)
GCP_PROJECT = os.environ.get("GCP_PROJECT", "dtumlops-484413")
GCP_REGION = os.environ.get("GCP_REGION", "europe-west4")
GCP_BUCKET = os.environ.get("GCP_BUCKET", "dtu-kfc-bucket")
ARTIFACT_REGISTRY = f"{GCP_REGION}-docker.pkg.dev/{GCP_PROJECT}/container-registry"

# Project commands
@task
def generate_hdf5(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run {PROJECT_NAME} data generate-hdf5 -d data.nosync -c configs/metadata/data_stats.csv -o data.nosync/images_hdf5.h5", echo=True, pty=not WINDOWS)

@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run {PROJECT_NAME} train", echo=True, pty=not WINDOWS)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)

# Docker commands
@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build all docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t frontend:latest . -f dockerfiles/frontend.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

@task
def docker_build_train(ctx: Context, progress: str = "plain") -> None:
    """Build training docker image."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

@task
def docker_build_api(ctx: Context, progress: str = "plain") -> None:
    """Build API docker image."""
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

@task
def docker_build_frontend(ctx: Context, progress: str = "plain") -> None:
    """Build frontend docker image."""
    ctx.run(
        f"docker build -t frontend:latest . -f dockerfiles/frontend.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

@task
def docker_run_api(ctx: Context, port: int = 8080, credentials: str = "") -> None:
    """Run API container locally.
    
    Args:
        port: Port to expose the API on (default: 8080)
        credentials: Path to GCP service account JSON file. If not provided,
                     will try to use Application Default Credentials from ~/.config/gcloud
    """
    if credentials:
        # Use explicit service account credentials (convert to absolute path for Docker)
        creds_path = os.path.abspath(credentials)
        ctx.run(
            f'docker run -p {port}:8080 -e PORT=8080 '
            f'-v "{creds_path}:/app/credentials.json:ro" '
            f'-e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json '
            f'api:latest',
            echo=True,
            pty=not WINDOWS
        )
    else:
        # Use Application Default Credentials from gcloud
        home = os.path.expanduser("~")
        gcloud_config = f"{home}/.config/gcloud"
        ctx.run(
            f'docker run -p {port}:8080 -e PORT=8080 '
            f'-v "{gcloud_config}:/root/.config/gcloud:ro" '
            f'api:latest',
            echo=True,
            pty=not WINDOWS
        )

@task
def docker_run_frontend(ctx: Context, port: int = 8501, backend: str = "http://host.docker.internal:8080") -> None:
    """Run frontend container locally.
    
    Args:
        port: Port to expose the frontend on (default: 8501)
        backend: URL of the backend API. Defaults to host.docker.internal:8080 
                 to connect to API running on the host machine.
    """
    ctx.run(
        f"docker run -p {port}:8501 -e PORT=8501 -e WIZARD_BACKEND={backend} frontend:latest",
        echo=True,
        pty=not WINDOWS
    )

# GCP Deployment commands
@task
def gcp_build_train(ctx: Context) -> None:
    """Build and push training image to Artifact Registry."""
    ctx.run(
        f"docker build --platform linux/amd64 -t {ARTIFACT_REGISTRY}/wizard-ops-train -f dockerfiles/train.dockerfile .",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(f"docker push {ARTIFACT_REGISTRY}/wizard-ops-train", echo=True, pty=not WINDOWS)

@task
def gcp_build_api(ctx: Context) -> None:
    """Build and push API image to Artifact Registry."""
    ctx.run(
        f"docker build --platform linux/amd64 -t {ARTIFACT_REGISTRY}/wizard-ops-api -f dockerfiles/api.dockerfile .",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(f"docker push {ARTIFACT_REGISTRY}/wizard-ops-api", echo=True, pty=not WINDOWS)

@task
def gcp_build_frontend(ctx: Context) -> None:
    """Build and push frontend image to Artifact Registry."""
    ctx.run(
        f"docker build --platform linux/amd64 -t {ARTIFACT_REGISTRY}/streamlit-app -f dockerfiles/frontend.dockerfile .",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(f"docker push {ARTIFACT_REGISTRY}/streamlit-app", echo=True, pty=not WINDOWS)

@task
def gcp_deploy_api(ctx: Context, memory: str = "2Gi") -> None:
    """Deploy API to Cloud Run."""
    ctx.run(
        f"gcloud run deploy wizard-ops-api "
        f"--image {ARTIFACT_REGISTRY}/wizard-ops-api "
        f"--region {GCP_REGION} "
        f"--memory {memory} "
        f"--allow-unauthenticated",
        echo=True,
        pty=not WINDOWS
    )

@task
def gcp_deploy_frontend(ctx: Context, backend_url: str = "") -> None:
    """Deploy frontend to Cloud Run."""
    env_vars = f"--set-env-vars WIZARD_BACKEND={backend_url}" if backend_url else ""
    ctx.run(
        f"gcloud run deploy streamlit-app "
        f"--image {ARTIFACT_REGISTRY}/streamlit-app "
        f"--region {GCP_REGION} "
        f"--allow-unauthenticated {env_vars}",
        echo=True,
        pty=not WINDOWS
    )

@task
def gcp_deploy_all(ctx: Context) -> None:
    """Build and deploy all services to GCP."""
    gcp_build_api(ctx)
    gcp_build_frontend(ctx)
    gcp_deploy_api(ctx)
    # Get backend URL and deploy frontend
    result = ctx.run(
        f"gcloud run services describe wizard-ops-api --region {GCP_REGION} --format 'value(status.url)'",
        echo=True,
        pty=not WINDOWS,
        hide=True
    )
    backend_url = result.stdout.strip() if result else ""
    gcp_deploy_frontend(ctx, backend_url)

@task
def gcp_trigger_train(ctx: Context, experiment: str = "default", backbone: str = "resnet18", epochs: int = 10) -> None:
    """Trigger Cloud Build training pipeline."""
    ctx.run(
        f"gcloud builds submit --config=cloudbuild-train.yaml "
        f"--substitutions=_EXPERIMENT_NAME={experiment},_BACKBONE={backbone},_MAX_EPOCHS={epochs}",
        echo=True,
        pty=not WINDOWS
    )

@task
def gcp_trigger_full(ctx: Context, experiment: str = "default", backbone: str = "resnet18") -> None:
    """Trigger full Cloud Build pipeline (train + deploy)."""
    ctx.run(
        f"gcloud builds submit --config=cloudbuild.yaml "
        f"--substitutions=_EXPERIMENT_NAME={experiment},_BACKBONE={backbone}",
        echo=True,
        pty=not WINDOWS
    )

@task
def gcp_trigger_serve(ctx: Context) -> None:
    """Trigger serving-only deployment (no training)."""
    ctx.run(
        "gcloud builds submit --config=cloudbuild-serve.yaml",
        echo=True,
        pty=not WINDOWS
    )

@task
def gcp_setup(ctx: Context) -> None:
    """Set up GCP triggers and permissions."""
    ctx.run("chmod +x setup_triggers.sh && ./setup_triggers.sh", echo=True, pty=not WINDOWS)

# DVC commands
@task
def dvc_pull(ctx: Context) -> None:
    """Pull data and checkpoints from DVC."""
    ctx.run("uv run dvc pull", echo=True, pty=not WINDOWS)

@task
def dvc_push(ctx: Context) -> None:
    """Push data and checkpoints to DVC."""
    ctx.run("uv run dvc push", echo=True, pty=not WINDOWS)

@task
def dvc_push_checkpoints(ctx: Context) -> None:
    """Add and push checkpoints to DVC."""
    ctx.run("uv run dvc add checkpoints", echo=True, pty=not WINDOWS)
    ctx.run("uv run dvc push checkpoints.dvc", echo=True, pty=not WINDOWS)

# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
