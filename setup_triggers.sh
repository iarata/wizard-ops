#!/bin/bash
# GCP Cloud Build Triggers Setup Script
# Run this script to set up all the Cloud Build triggers for automated deployment
# 
# Prerequisites:
# 1. gcloud CLI installed and authenticated
# 2. Cloud Build API enabled
# 3. Repository connected to Cloud Build
#
# Usage: ./setup_triggers.sh <PROJECT_ID> <REPO_OWNER> <REPO_NAME>

set -e

PROJECT_ID=${1:-"dtumlops-484413"}
REPO_OWNER=${2:-"iarata"}
REPO_NAME=${3:-"wizard-ops"}
REGION="europe-west4"

echo "Setting up Cloud Build triggers for project: $PROJECT_ID"
echo "Repository: $REPO_OWNER/$REPO_NAME"
echo "Region: $REGION"
echo ""

# Enable required APIs
echo "Enabling required GCP APIs..."
gcloud services enable cloudbuild.googleapis.com --project=$PROJECT_ID
gcloud services enable run.googleapis.com --project=$PROJECT_ID
gcloud services enable artifactregistry.googleapis.com --project=$PROJECT_ID
gcloud services enable secretmanager.googleapis.com --project=$PROJECT_ID
gcloud services enable aiplatform.googleapis.com --project=$PROJECT_ID

# Create GCS credentials secret (if not exists)
echo "Setting up secrets..."
if ! gcloud secrets describe gcs-credentials --project=$PROJECT_ID &>/dev/null; then
    echo "Creating gcs-credentials secret..."
    echo "Please provide your GCS service account JSON key file path:"
    read -r KEY_FILE
    gcloud secrets create gcs-credentials --project=$PROJECT_ID
    gcloud secrets versions add gcs-credentials --data-file="$KEY_FILE" --project=$PROJECT_ID
else
    echo "gcs-credentials secret already exists"
fi

# Grant Cloud Build access to secrets
echo "Granting Cloud Build service account access to secrets..."
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding gcs-credentials \
    --project=$PROJECT_ID \
    --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

# Grant Cloud Build permission to deploy to Cloud Run
echo "Granting Cloud Build service account Cloud Run permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
    --role="roles/run.admin"

gcloud iam service-accounts add-iam-policy-binding \
    $PROJECT_NUMBER-compute@developer.gserviceaccount.com \
    --project=$PROJECT_ID \
    --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"

# Create triggers
echo ""
echo "Creating Cloud Build triggers..."

# 1. Training trigger - manual or on specific files
echo "Creating training trigger..."
gcloud builds triggers create github \
    --project=$PROJECT_ID \
    --name="wizard-ops-train" \
    --description="Train model and push checkpoints to DVC/GCS" \
    --repo-owner=$REPO_OWNER \
    --repo-name=$REPO_NAME \
    --branch-pattern="^main$" \
    --included-files="src/wizard_ops/train.py,src/wizard_ops/model.py,configs/**" \
    --build-config="cloudbuild-train.yaml" \
    --substitutions="_EXPERIMENT_NAME=auto_$(date +%m%d),_BACKBONE=resnet18,_MAX_EPOCHS=10" \
    2>/dev/null || echo "Trigger may already exist, updating..."

# 2. Backend deployment trigger
echo "Creating backend deployment trigger..."
gcloud builds triggers create github \
    --project=$PROJECT_ID \
    --name="wizard-ops-backend" \
    --description="Deploy backend API to Cloud Run" \
    --repo-owner=$REPO_OWNER \
    --repo-name=$REPO_NAME \
    --branch-pattern="^main$" \
    --included-files="src/wizard_ops/backend/**,dockerfiles/api.dockerfile" \
    --build-config="cloudbuild-backend.yaml" \
    2>/dev/null || echo "Trigger may already exist, updating..."

# 3. Frontend deployment trigger
echo "Creating frontend deployment trigger..."
gcloud builds triggers create github \
    --project=$PROJECT_ID \
    --name="wizard-ops-frontend" \
    --description="Deploy frontend to Cloud Run" \
    --repo-owner=$REPO_OWNER \
    --repo-name=$REPO_NAME \
    --branch-pattern="^main$" \
    --included-files="src/wizard_ops/frontend/**,dockerfiles/frontend.dockerfile" \
    --build-config="cloudbuild-frontend.yaml" \
    2>/dev/null || echo "Trigger may already exist, updating..."

# 4. Full pipeline trigger - on release tags
echo "Creating full pipeline trigger..."
gcloud builds triggers create github \
    --project=$PROJECT_ID \
    --name="wizard-ops-full-pipeline" \
    --description="Full pipeline: Train + Deploy Backend + Deploy Frontend" \
    --repo-owner=$REPO_OWNER \
    --repo-name=$REPO_NAME \
    --tag-pattern="^v[0-9]+\.[0-9]+\.[0-9]+$" \
    --build-config="cloudbuild.yaml" \
    2>/dev/null || echo "Trigger may already exist, updating..."

# 5. Serving-only trigger - on specific branch
echo "Creating serving-only trigger..."
gcloud builds triggers create github \
    --project=$PROJECT_ID \
    --name="wizard-ops-serve-only" \
    --description="Deploy backend and frontend without training" \
    --repo-owner=$REPO_OWNER \
    --repo-name=$REPO_NAME \
    --branch-pattern="^deploy/.*$" \
    --build-config="cloudbuild-serve.yaml" \
    2>/dev/null || echo "Trigger may already exist, updating..."

echo ""
echo "============================================"
echo "           SETUP COMPLETE                  "
echo "============================================"
echo ""
echo "Created triggers:"
echo "  - wizard-ops-train: Triggers on changes to training code"
echo "  - wizard-ops-backend: Triggers on backend code changes"
echo "  - wizard-ops-frontend: Triggers on frontend code changes"
echo "  - wizard-ops-full-pipeline: Triggers on version tags (v*.*.*))"
echo "  - wizard-ops-serve-only: Triggers on deploy/* branches"
echo ""
echo "Manual trigger commands:"
echo "  Train:    gcloud builds triggers run wizard-ops-train --branch=main"
echo "  Backend:  gcloud builds triggers run wizard-ops-backend --branch=main"
echo "  Frontend: gcloud builds triggers run wizard-ops-frontend --branch=main"
echo "  Full:     gcloud builds submit --config=cloudbuild.yaml"
echo ""
