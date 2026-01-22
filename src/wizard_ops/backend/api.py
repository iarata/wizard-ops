import io
import logging
import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from PIL import Image

from wizard_ops.evaluate import load_model_for_inference, predict_nutrition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("/tmp/best-nutrition-v0.ckpt")

# Check for local checkpoint first (set by CLI)
CHECKPOINT_LOCAL = os.environ.get("CHECKPOINT_LOCAL", "")

# Read from environment variables with fallback defaults (can be set from config.yaml via CLI)
BUCKET_NAME = os.environ.get("BUCKET_NAME", "dtu-kfc-bucket")
BLOB_NAME = os.environ.get(
    "CHECKPOINT_BLOB", 
    "checkpoints/nutrition_resnet18_0115_1951/best-nutrition-epoch=04-val-loss=0.00.ckpt"
)

model_dev_t: tuple[torch.nn.Module, torch.device] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_dev_t

    # Determine which checkpoint to use
    if CHECKPOINT_LOCAL and Path(CHECKPOINT_LOCAL).exists():
        # Use local checkpoint (set via CLI from config.yaml)
        checkpoint_path = Path(CHECKPOINT_LOCAL)
        logger.info(f"Using local checkpoint: {checkpoint_path}")
    else:
        # Download from GCS
        checkpoint_path = MODEL_PATH
        logger.info(f"Starting API with bucket={BUCKET_NAME}, blob={BLOB_NAME}")

        if not checkpoint_path.exists():
            logger.info(f"Downloading model from gs://{BUCKET_NAME}/{BLOB_NAME}")
            # Import here to avoid dependency issues when running locally
            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob(BLOB_NAME)
            blob.download_to_filename(checkpoint_path)
            logger.info("Model downloaded successfully")

    # Checkpoint PyTorch-Lightning
    model_dev_t = load_model_for_inference(
        checkpoint_path, device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze")
async def analyze_food(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))  # noqa: F841

    # load model
    # https://storage.cloud.google.com/dtu-kfc-bucket/best-nutrition-v0.ckpt
    if model_dev_t is None:
        return {
            "message": "Model not loaded yet. Please try again later.",
            "status": HTTPStatus.INTERNAL_SERVER_ERROR,
        }

    result = {
        "calories": 0.0,
        "fat_g": 0.0,
        "protein_g": 0.0,
        "carbs_g": 0.0,
    }

    try:
        prediction = predict_nutrition(model_dev_t, images=[image])

        result = {
            "calories": prediction["normalized"]["total_calories"],
            "fat_g": prediction["normalized"]["total_fat"],
            "protein_g": prediction["normalized"]["total_protein"],
            "carbs_g": prediction["normalized"]["total_carb"],
        }

    except Exception as e:
        logger.exception("Error during inference")
        logger.error(f"Exception: {e}")
        pass

    return result


@app.get("/")
def read_root():
    return {"status": "ready", "model_loaded": MODEL_PATH.exists()}
