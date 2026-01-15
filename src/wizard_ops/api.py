from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from http import HTTPStatus
from contextlib import asynccontextmanager
import torch

from google.cloud import storage

from .model import DishMultiViewRegressor
from .data import get_default_transforms

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("/tmp/best-nutrition-v0.ckpt")
BUCKET_NAME = "dtu-kfc-bucket"
BLOB_NAME = "best-nutrition-v0.ckpt"

model = None
transform = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, transform

    if not MODEL_PATH.exists():
        print(f"Downloading model from gs://{BUCKET_NAME}/{BLOB_NAME}")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(BLOB_NAME)
        blob.download_to_filename(MODEL_PATH)
        print("Model downloaded successfully")

    # Checkpoint PyTorch-Lightning
    checkpoint = torch.load(MODEL_PATH.resolve(), map_location=torch.device("cpu"))

    # Extract only weights
    model = DishMultiViewRegressor()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    transform = get_default_transforms(224)
    yield
    # Shutdown


app = FastAPI(lifespan=lifespan)

# Allow Streamlit to call FastAPI
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
    if model is None or transform is None:
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
        imgarr = np.array(image.convert("RGB"))
        out = transform(image=imgarr)

        # albumentations typically returns a dict {'image': ...}
        image_obj = None
        if isinstance(out, dict):
            image_obj = out.get("image")
        else:
            image_obj = out

        # normalize to a torch tensor with shape (C, H, W)
        if isinstance(image_obj, np.ndarray):
            image_tensor = torch.from_numpy(image_obj).permute(2, 0, 1).float() / 255.0
        elif isinstance(image_obj, torch.Tensor):
            image_tensor = image_obj
        else:
            raise TypeError(f"Unsupported transform output type: {type(image_obj)}")

        image_batch = (
            torch.stack([image_tensor], dim=0).unsqueeze(0).to(torch.device("cpu"))
        )

        with torch.no_grad():
            predictions, _ = model(image_batch)

        pred_values = predictions.squeeze().cpu().numpy()

        print(f"pred_values: {pred_values}")

        result = {
            "calories": float(pred_values[0]),
            "fat_g": float(pred_values[1]),
            "protein_g": float(pred_values[2]),
            "carbs_g": float(pred_values[3]),
        }

    except Exception as e:
        logger.exception("Error during inference")
        logger.error(f"Exception: {e}")
        pass

    return result


# @app.get("/")
# def root():
#     """Health check."""
#     response = {
#         "message": HTTPStatus.OK.phrase,
#         "status": HTTPStatus.OK,
#     }
#     return response

@app.get("/")
def read_root():
    return {"status": "ready", "model_loaded": MODEL_PATH.exists()}