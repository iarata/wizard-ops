import io
import logging
import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from google.cloud import storage

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
    # Use locally existant model or download from GCS using config.yaml configuration
    global model_dev_t

    if CHECKPOINT_LOCAL and Path(CHECKPOINT_LOCAL).exists():
        checkpoint_path = Path(CHECKPOINT_LOCAL)
        logger.info(f"Using local checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = MODEL_PATH
        logger.info(f"Starting API with bucket={BUCKET_NAME}, blob={BLOB_NAME}")

        if not checkpoint_path.exists():
            logger.info(f"Downloading model from gs://{BUCKET_NAME}/{BLOB_NAME}")
            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob(BLOB_NAME)
            blob.download_to_filename(checkpoint_path)
            logger.info(f"Model downloaded to {checkpoint_path} successfully")

    model_dev_t = load_model_for_inference(
        checkpoint_path, device=torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    yield


app = FastAPI(lifespan=lifespan)

# Not the proper way, but in this project we are not worried about external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze")
async def analyze_food(file: UploadFile = File(...)):
    """Feed image to model, read normalized result data and include in response result field"""
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    if model_dev_t is None:
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"message": "Model not loaded yet. Please try again later."}
        )

    try:
        prediction = predict_nutrition(model_dev_t, images=[image])

        result = {
            "calories": prediction["normalized"]["total_calories"],
            "fat_g": prediction["normalized"]["total_fat"],
            "protein_g": prediction["normalized"]["total_protein"],
            "carbs_g": prediction["normalized"]["total_carb"],
        }

    except Exception as e:
        logger.exception(f"Error during inference: {e}")
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"message": f"Error during inference {e.__cause__}"}
        )

    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={"result": result}
    )


@app.get("/")
def read_root():
    """status -- able to handle HTTP requests, model_loaded -- is model is available to process those requests"""
    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={"model_loaded": MODEL_PATH.exists()}
    )
