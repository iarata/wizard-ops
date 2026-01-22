#!/bin/bash
set -e

echo "=== Wizard Ops Training Pipeline ==="

# Read defaults from config.yaml if environment variables are not set
CONFIG_FILE="/app/configs/config.yaml"

# Function to read yaml value (simple grep-based, works for flat values)
get_config() {
    local key=$1
    grep -E "^\s*${key}:" "$CONFIG_FILE" | head -1 | sed 's/.*:\s*"\?\([^"]*\)"\?.*/\1/' | tr -d ' '
}

# Use environment variables if set, otherwise fall back to config.yaml
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(get_config 'experiment_name')}"
BACKBONE="${BACKBONE:-$(get_config 'backbone')}"
MAX_EPOCHS="${MAX_EPOCHS:-$(get_config 'max_epochs')}"
BATCH_SIZE="${BATCH_SIZE:-$(get_config 'batch_size')}"
FAST_DEV_RUN="${FAST_DEV_RUN:-$(get_config 'fast_dev_run')}"

# Set final defaults if still empty
EXPERIMENT_NAME="${EXPERIMENT_NAME:-default_experiment}"
BACKBONE="${BACKBONE:-resnet18}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
FAST_DEV_RUN="${FAST_DEV_RUN:-false}"

echo "Experiment: ${EXPERIMENT_NAME}"
echo "Backbone: ${BACKBONE}"
echo "Max Epochs: ${MAX_EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Fast Dev Run: ${FAST_DEV_RUN}"

# Pull data + current checkpoints from DVC
echo ""
echo "Pulling data from DVC (data.nosync.dvc)..."
uv run dvc pull data.nosync.dvc

echo ""
echo "Pulling current checkpoints from DVC (checkpoints.dvc)..."
uv run dvc pull checkpoints.dvc

# Run training with Hydra config overrides
echo ""
echo "Starting training..."
uv run wizard_ops train \
    logging.experiment_name=${EXPERIMENT_NAME} \
    model.backbone=${BACKBONE} \
    train.max_epochs=${MAX_EPOCHS} \
    data.batch_size=${BATCH_SIZE} \
    train.fast_dev_run=${FAST_DEV_RUN} \
    ${EXTRA_HYDRA_ARGS:-}

# Push checkpoints back to DVC
echo ""
echo "Pushing checkpoints to DVC..."
uv run dvc add checkpoints
uv run dvc push checkpoints.dvc

echo ""
echo "=== Training Complete ==="
echo "Checkpoints have been pushed to GCS bucket"
