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

from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig

from src.wizard_ops import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("/tmp/best-nutrition-v0.ckpt")
BUCKET_NAME = "dtu-kfc-bucket"
BLOB_NAME = "checkpoints/nutrition_resnet18_0115_1951/best-nutrition-epoch=04-val-loss=0.00.ckpt"

CONFIG_LOCATION = (
    "../../../../configs"  # config path must be relative according to Hydra
)

model_dev_t: tuple[torch.nn.Module, torch.device] | None = None
cfg: DictConfig | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_dev_t

    assert Path(
        CONFIG_LOCATION
    ).is_dir(), f"Config location {CONFIG_LOCATION} is not a directory."

    experiment_name = None
    checkpoints_dir = None
    run_name = None

    with initialize(version_base=None, config_path=str(CONFIG_LOCATION)):
        global cfg
        cfg = compose(config_name="config")

        experiment_name = cfg.get("logging", {}).get("experiment_name", "N/A")
        checkpoints_dir = cfg.get("model", {}).get("checkpoint_dir", "N/A")
        backbone = cfg.get("model", {}).get("backbone", "N/A")
        run_name = f"nutrition_{backbone}_{experiment_name}"

        logger.info(
            "Configuration loaded:"
            + f"\n  - experiment_name: {experiment_name}"
            + f"\n  - backbone: {backbone}"
            + f"\n  - checkpoint_dir: {checkpoints_dir}"
            + f"\n  - run_name: {run_name}"
        )

    if not MODEL_PATH.exists():
        bucket_subpath = str(Path(checkpoints_dir) / run_name)
        print(f"Downloading model from gs://{BUCKET_NAME}/{bucket_subpath}")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        # TODO: must derive filename somehow or change how it is generated
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
