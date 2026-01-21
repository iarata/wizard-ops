from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from wizard_ops.model import DishMultiViewRegressor
from wizard_ops.utils import get_default_transforms, load_normalization_stats

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

NUM_CAMERAS = 4
NUM_FRAMES = 5

TARGET_COLUMNS = [
    "total_calories",
    "total_mass",
    "total_fat",
    "total_carb",
    "total_protein",
]


def _as_rgb_numpy(x: Any) -> np.ndarray:
    """
    Convert input to RGB uint8 numpy array in HWC layout.
    Accepts: path/Path, PIL.Image, numpy array.
    """
    if isinstance(x, (str, Path)):
        img = Image.open(x).convert("RGB")
        return np.array(img, dtype=np.uint8)

    if isinstance(x, Image.Image):
        return np.array(x.convert("RGB"), dtype=np.uint8)

    if isinstance(x, np.ndarray):
        arr = x
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        # Accept HWC or CHW; convert to HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        return arr

    raise TypeError(f"Unsupported image type: {type(x)}")


def _tile_or_trim_views(
    views: list[Any],
    *,
    num_cameras: int = NUM_CAMERAS,
    num_frames: int = NUM_FRAMES,
) -> list[Any]:
    """
    Normalize view list length to exactly num_cameras*num_frames using common heuristics.
    """
    want = num_cameras * num_frames
    n = len(views)

    if n == want:
        return views

    if n == 1:
        return views * want

    if n == num_cameras:
        # 4 camera snapshots -> repeat frames
        out = []
        for cam_img in views:
            out.extend([cam_img] * num_frames)
        return out  # cam0 f0..f4, cam1 f0..f4, ...

    if n == num_frames:
        # 5 frames from one camera -> repeat cameras
        return views * num_cameras

    # Fallback: tile then trim
    if n < want:
        reps = (want + n - 1) // n
        tiled = (views * reps)[:want]
        logger.warning(
            "Got %d images; tiling to %d views for model input.",
            n,
            want,
        )
        return tiled

    # n > want
    logger.warning(
        "Got %d images; trimming to first %d views for model input.",
        n,
        want,
    )
    return views[:want]


def _build_model_input_tensor(
    images: Any,
    *,
    image_size: int = 224,
    transform=None,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a tensor shaped (1, NUM_CAMERAS, NUM_FRAMES, 3, H, W) float32,
    normalized exactly like validation (Resize + ImageNet Normalize).
    """
    if torch.is_tensor(images):
        x = images
        # Accept (4,5,3,H,W) or (B,4,5,3,H,W)
        if x.ndim == 5:
            x = x.unsqueeze(0)
        if x.ndim != 6:
            raise ValueError(f"Unexpected tensor shape: {tuple(x.shape)}")
        return x.to(device)

    # Normalize input to a list of "views"
    if isinstance(images, (str, Path, Image.Image, np.ndarray)):
        views = [images]
    elif isinstance(images, (list, tuple)):
        # allow nested lists; flatten one level
        flat: list[Any] = []
        for item in images:
            if isinstance(item, (list, tuple)):
                flat.extend(list(item))
            else:
                flat.append(item)
        views = flat
    else:
        raise TypeError(f"Unsupported images container type: {type(images)}")

    views = _tile_or_trim_views(views, num_cameras=NUM_CAMERAS, num_frames=NUM_FRAMES)

    tf = transform or get_default_transforms(image_size=image_size)

    per_view: list[torch.Tensor] = []
    for v in views:
        rgb = _as_rgb_numpy(v)  # HWC uint8
        t = tf(image=rgb)["image"]  # (3,H,W) torch
        if not torch.is_tensor(t):
            t = torch.from_numpy(t)
        per_view.append(t)

    stacked = torch.stack(per_view, dim=0)  # (20,3,H,W)
    stacked = stacked.view(NUM_CAMERAS, NUM_FRAMES, *stacked.shape[1:])  # (4,5,3,H,W)
    return stacked.unsqueeze(0).to(device)  # (1,4,5,3,H,W)


def denormalize_predictions(
    preds: dict[str, float],
    *,
    stats: dict,
    method: str = "zscore",
) -> dict[str, float]:
    """
    Convert normalized predictions back to real units using normalisation_stats.json.
    """
    target_stats = (stats or {}).get("targets", {})
    out: dict[str, float] = {}

    for k, v in preds.items():
        val = float(v)
        stat = target_stats.get(k)
        if not stat:
            out[k] = val
            continue

        if method == "zscore":
            out[k] = val * float(stat.get("std", 0.0)) + float(stat.get("mean", 0.0))
        elif method == "minmax":
            mn = float(stat.get("min", 0.0))
            mx = float(stat.get("max", 0.0))
            out[k] = val * (mx - mn) + mn
        elif method == "max":
            mx = float(stat.get("max", 0.0))
            out[k] = val * mx
        else:
            out[k] = val

    return out


def _extract_pred_tensor(model_out: Any) -> torch.Tensor:
    """
    Try to robustly get a (B,5) tensor from whatever the model returns.
    """
    if torch.is_tensor(model_out):
        return model_out

    if isinstance(model_out, (list, tuple)) and len(model_out) > 0:
        if torch.is_tensor(model_out[0]):
            return model_out[0]

    if isinstance(model_out, dict):
        # common keys
        for k in ["preds", "y_hat", "yhat", "output", "outputs", "logits"]:
            if k in model_out and torch.is_tensor(model_out[k]):
                return model_out[k]

        # or dict already keyed by target names
        if all(name in model_out for name in TARGET_COLUMNS):
            vals = [model_out[name] for name in TARGET_COLUMNS]
            vals = [v if torch.is_tensor(v) else torch.tensor(v) for v in vals]
            t = torch.stack(vals, dim=-1)  # (..., 5)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            return t

    raise TypeError(
        "Could not extract prediction tensor from model output of type "
        f"{type(model_out)}"
    )


def load_model_for_inference(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, torch.device]:
    """
    Supports:
      - Lightning .ckpt via DishMultiViewRegressor.load_from_checkpoint
      - torch-saved full model object (.pth) via torch.load
    """
    p = Path(checkpoint_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if p.suffix == ".ckpt":
        model = DishMultiViewRegressor.load_from_checkpoint(str(p), map_location=device)
    else:
        model = torch.load(str(p), map_location=device)

    model.eval()
    model.to(device)
    return model, device


@torch.inference_mode()
def predict_nutrition(
    checkpoint_path: str | Path,
    images: Any,
    *,
    image_size: int = 224,
    normalisation_method: str = "zscore",
    stats_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, dict[str, float]]:
    """
    Predict the 5 nutrition metrics for one dish (one or many images).

    Returns:
      {
        "normalized": {...},
        "denormalized": {...}
      }
    """
    model, dev = load_model_for_inference(checkpoint_path, device=device)

    x = _build_model_input_tensor(
        images,
        image_size=image_size,
        device=dev,
    )

    # call model (support both positional and keyword styles)
    try:
        out = model(x)
    except TypeError:
        out = model(images=x)

    pred_t = _extract_pred_tensor(out)  # (B,5) expected
    pred_t = pred_t.float().detach().cpu()

    if pred_t.ndim != 2 or pred_t.shape[1] != len(TARGET_COLUMNS):
        raise ValueError(f"Expected (B,5) predictions, got {tuple(pred_t.shape)}")

    # single dish -> first row
    pred = pred_t[0].tolist()
    normalized = {k: float(v) for k, v in zip(TARGET_COLUMNS, pred)}

    stats = load_normalization_stats(stats_path) if stats_path else load_normalization_stats()
    denorm = denormalize_predictions(
        normalized,
        stats=stats,
        method=normalisation_method,
    )

    return {"normalized": normalized, "denormalized": denorm}