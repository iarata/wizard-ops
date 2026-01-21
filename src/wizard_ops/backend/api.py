import io
import logging
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from PIL import Image

from src.wizard_ops import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("/tmp/best-nutrition-v0.ckpt")
BUCKET_NAME = "dtu-kfc-bucket"
BLOB_NAME = "checkpoints/nutrition_resnet18_0115_1951/best-nutrition-epoch=04-val-loss=0.00.ckpt"

model_dev_t: tuple[torch.nn.Module, torch.device] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_dev_t

    if not MODEL_PATH.exists():
        print(f"Downloading model from gs://{BUCKET_NAME}/{BLOB_NAME}")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(BLOB_NAME)
        blob.download_to_filename(MODEL_PATH)
        print("Model downloaded successfully")

    # Checkpoint PyTorch-Lightning
    model_dev_t = evaluate.load_model_for_inference(
        MODEL_PATH, device=torch.device("cpu")
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
        prediction = evaluate.predict_nutrition(model_dev_t, images=[image])

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
