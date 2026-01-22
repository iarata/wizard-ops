# GCP Automated Deployment Guide

This document explains how to set up and use the automated GCP deployment pipeline for the Wizard Ops project.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Cloud Build Pipelines                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │   Training   │───▶│   Backend    │───▶│   Frontend   │           │
│  │   Pipeline   │    │   Deploy     │    │   Deploy     │           │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘           │
│         │                   │                   │                    │
│         ▼                   ▼                   ▼                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │  GCS Bucket  │    │  Cloud Run   │    │  Cloud Run   │           │
│  │ (DVC Remote) │    │  (FastAPI)   │    │ (Streamlit)  │           │
│  └──────────────┘    └──────────────┘    └──────────────┘           │
│                                                                      │
│  Images stored in: Artifact Registry (europe-west4)                  │
│  DVC files stored in: gs://dtu-kfc-bucket/files/                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Configuration

All deployment settings are centralized in `configs/config.yaml`:

```yaml
gcp:
  project_id: "dtumlops-484413"
  region: "europe-west4"
  bucket: "dtu-kfc-bucket"
  artifact_registry: "europe-west4-docker.pkg.dev/dtumlops-484413/container-registry"

checkpoint:
  serving_checkpoint: "checkpoints/nutrition_resnet18_0115_1951/best-nutrition-*.ckpt"
```

## Prerequisites

1. **GCP Project** with billing enabled
2. **APIs enabled**:
   - Cloud Build API
   - Cloud Run API
   - Artifact Registry API
   - Secret Manager API
   - Vertex AI API (for GPU training)
3. **gcloud CLI** installed and authenticated
4. **Docker** installed locally
5. **DVC** configured with GCS remote (`gs://dtu-kfc-bucket/files/`)

## Quick Setup

```bash
# 1. Clone and setup
git clone <repo-url>
cd wizard_ops

# 2. Configure GCP project
gcloud config set project dtumlops-484413

# 3. Authenticate Docker with Artifact Registry
gcloud auth configure-docker europe-west4-docker.pkg.dev

# 3.1 Setup Wandb API key
echo -n "your-wandb-api-key" | gcloud secrets create wandb-api-key --data-file=-

# 4. Run the setup script
chmod +x setup_triggers.sh
./setup_triggers.sh dtumlops-484413 iarata wizard-ops
```

## Pipeline Options

### 1. Full Pipeline (Training + Deployment)

Runs training, pushes checkpoints to DVC/GCS, then deploys backend and frontend.

```bash
# Via Cloud Build
gcloud builds submit --config=cloudbuild.yaml \
  --substitutions=_EXPERIMENT_NAME=my_experiment,_BACKBONE=resnet18,_MAX_EPOCHS=20

# Via invoke task
invoke gcp-trigger-full --experiment=my_experiment --backbone=resnet18
```

**Triggered automatically on**: Version tags (`v*.*.*`)

### 2. Training Only

Run training and push new checkpoints without deploying.

```bash
# Via Cloud Build
gcloud builds submit --config=cloudbuild-train.yaml \
  --substitutions=_EXPERIMENT_NAME=experiment_v2,_BACKBONE=resnet50

# Via invoke task
invoke gcp-trigger-train --experiment=experiment_v2 --backbone=resnet50 --epochs=50
```

**Triggered automatically on**: Changes to `src/wizard_ops/train.py`, `src/wizard_ops/model.py`, `configs/**`

### 3. Training with GPU (Vertex AI)

For larger models or longer training, use Vertex AI with GPU acceleration.

```bash
gcloud builds submit --config=cloudbuild-train-vertex.yaml \
  --substitutions=_EXPERIMENT_NAME=large_model,_BACKBONE=efficientnet_b3,_ACCELERATOR_TYPE=NVIDIA_TESLA_V100
```

### 4. Serving Only (No Training)

Deploy backend and frontend without running training.

```bash
# Via Cloud Build
gcloud builds submit --config=cloudbuild-serve.yaml

# Via invoke task
invoke gcp-trigger-serve
```

**Triggered automatically on**: `deploy/*` branches

### 5. Backend Only

Deploy just the API backend.

```bash
# Via Cloud Build
gcloud builds submit --config=cloudbuild-backend.yaml

# Via invoke task
invoke gcp-build-api
invoke gcp-deploy-api
```

**Triggered automatically on**: Changes to `src/wizard_ops/backend/**`

### 6. Frontend Only

Deploy just the Streamlit frontend.

```bash
# Via Cloud Build
gcloud builds submit --config=cloudbuild-frontend.yaml

# Via invoke task
invoke gcp-build-frontend
invoke gcp-deploy-frontend
```

**Triggered automatically on**: Changes to `src/wizard_ops/frontend/**`

## Configuration

### Environment Variables

Set these in `.env.gcp` or as substitutions:

| Variable           | Description                | Default              |
| ------------------ | -------------------------- | -------------------- |
| `_EXPERIMENT_NAME` | Training experiment name   | `default_experiment` |
| `_BACKBONE`        | Model backbone             | `resnet18`           |
| `_MAX_EPOCHS`      | Training epochs            | `10`                 |
| `_BATCH_SIZE`      | Batch size                 | `32`                 |
| `_CHECKPOINT_BLOB` | Checkpoint path in GCS     | Latest trained       |
| `_API_MEMORY`      | Backend memory allocation  | `2Gi`                |
| `_FRONTEND_MEMORY` | Frontend memory allocation | `512Mi`              |

### Using a New Model Checkpoint

After training, update the checkpoint path:

```bash
# 1. Find the new checkpoint in GCS
gsutil ls gs://dtu-kfc-bucket/checkpoints/

# 2. Deploy with new checkpoint
gcloud builds submit --config=cloudbuild-backend.yaml \
  --substitutions=_CHECKPOINT_BLOB=checkpoints/nutrition_resnet50_0122_1430/best-nutrition-epoch=15-val-loss=0.00.ckpt
```

Or update the default in `cloudbuild-backend.yaml`:

```yaml
substitutions:
  _CHECKPOINT_BLOB: "checkpoints/your_new_checkpoint/best-nutrition-*.ckpt"
```

## Invoke Tasks Reference

```bash
# Local development
invoke train                    # Run training locally
invoke docker-build             # Build all Docker images
invoke docker-run-api           # Run API locally
invoke docker-run-api --credentials=dtumlops-484413-083ba11aaab8.json
invoke docker-run-frontend      # Run frontend locally

# GCP deployment
invoke gcp-build-api            # Build & push API image
invoke gcp-build-frontend       # Build & push frontend image
invoke gcp-deploy-api           # Deploy API to Cloud Run
invoke gcp-deploy-frontend      # Deploy frontend to Cloud Run
invoke gcp-deploy-all           # Build and deploy everything
invoke gcp-trigger-train        # Trigger training pipeline
invoke gcp-trigger-full         # Trigger full pipeline
invoke gcp-trigger-serve        # Trigger serving pipeline
invoke gcp-setup                # Set up GCP triggers

# DVC operations
invoke dvc-pull                 # Pull data/checkpoints
invoke dvc-push                 # Push data/checkpoints
invoke dvc-push-checkpoints     # Push only checkpoints
```

## Workflow Examples

### Starting a New Experiment

```bash
# 1. Modify config or code
# 2. Push to trigger training
git add -A
git commit -m "feat: New model architecture"
git push origin main

# Or trigger manually with custom params
invoke gcp-trigger-train --experiment=new_arch_v1 --backbone=efficientnet_b0 --epochs=50
```

### Deploying to Production

```bash
# 1. Create a version tag
git tag v1.2.0
git push origin v1.2.0

# This triggers the full pipeline automatically
```

### Quick Frontend Fix

```bash
# 1. Make changes to frontend
# 2. Push to main (auto-triggers frontend build)
git add src/wizard_ops/frontend/
git commit -m "fix: UI improvement"
git push origin main

# Or deploy manually
invoke gcp-build-frontend
invoke gcp-deploy-frontend
```

## Monitoring

### View Build Logs

```bash
gcloud builds list --limit=10
gcloud builds log <BUILD_ID>
```

### View Service Status

```bash
gcloud run services list --region=europe-west1
gcloud run services describe wizard-ops-api --region=europe-west1
gcloud run services describe streamlit-app --region=europe-west1
```

### View Service Logs

```bash
gcloud run services logs read wizard-ops-api --region=europe-west1
gcloud run services logs read streamlit-app --region=europe-west1
```

## Troubleshooting

### Build Failures

1. Check Cloud Build logs: `gcloud builds log <BUILD_ID>`
2. Verify secrets are set up: `gcloud secrets list`
3. Check service account permissions

### Training Failures

1. Ensure DVC is configured correctly
2. Verify GCS bucket access
3. Check CUDA availability for GPU training

### Deployment Failures

1. Verify Cloud Run quotas
2. Check memory/CPU limits
3. Ensure container health checks pass
