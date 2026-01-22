# Makefile for Wizard Ops
# Convenience wrapper around invoke tasks

.PHONY: help train test serve evaluate docker-build docker-run-api docker-run-frontend \
        gcp-build gcp-deploy gcp-full dvc-pull dvc-push docs

# Default target
help:
	@echo "Wizard Ops - Available Commands"
	@echo "================================"
	@echo ""
	@echo "Local Development:"
	@echo "  make train              - Run training locally (uses configs/config.yaml)"
	@echo "  make evaluate           - Run evaluation locally (uses configs/config.yaml)"
	@echo "  make serve              - Start API server locally (uses configs/config.yaml)"
	@echo "  make test               - Run tests with coverage"
	@echo "  make docker-build       - Build all Docker images"
	@echo "  make docker-run-api     - Run API container locally"
	@echo "  make docker-run-frontend - Run frontend container locally"
	@echo ""
	@echo "GCP Deployment:"
	@echo "  make gcp-setup          - Set up GCP triggers and permissions"
	@echo "  make gcp-build          - Build and push all images to GCR"
	@echo "  make gcp-deploy         - Deploy all services to Cloud Run"
	@echo "  make gcp-train          - Trigger training pipeline"
	@echo "  make gcp-full           - Trigger full pipeline (train + deploy)"
	@echo "  make gcp-serve          - Trigger serving-only deployment"
	@echo ""
	@echo "DVC:"
	@echo "  make dvc-pull           - Pull data and checkpoints"
	@echo "  make dvc-push           - Push data and checkpoints"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs               - Build documentation"
	@echo "  make docs-serve         - Serve documentation locally"

# Local Development
train:
	uv run wizard_ops train

test:
	uv run invoke test

serve:
	uv run wizard_ops serve

evaluate:
	uv run wizard_ops evaluate

docker-build:
	uv run invoke docker-build

docker-run-api:
	uv run invoke docker-run-api

docker-run-frontend:
	uv run invoke docker-run-frontend --backend=http://localhost:8080

# GCP Deployment
gcp-setup:
	chmod +x setup_triggers.sh && ./setup_triggers.sh

gcp-build:
	uv run invoke gcp-build-api
	uv run invoke gcp-build-frontend

gcp-deploy:
	uv run invoke gcp-deploy-all

gcp-train:
	uv run invoke gcp-trigger-train

gcp-full:
	uv run invoke gcp-trigger-full

gcp-serve:
	uv run invoke gcp-trigger-serve

# DVC
dvc-pull:
	uv run invoke dvc-pull

dvc-push:
	uv run invoke dvc-push

# Documentation
docs:
	uv run invoke build-docs

docs-serve:
	uv run invoke serve-docs
