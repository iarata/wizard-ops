from typing import Annotated

import lightning as L
import numpy as np
import torch
import typer
from data import get_default_transforms
from model import DishMultiViewRegressor
from PIL import Image

app = typer.Typer()


@app.command()
@torch.inference_mode()
def predict_from_pil_images(img_path: Annotated[str, typer.Option(help="Path to the image file")],
                            device: Annotated[str, typer.Option(help="Device to run on (cpu/mps/cuda)")] =
                            "mps" if torch.backends.mps.is_available() else "cpu"):
    """Predict from a single image path. The augmentation/transform is constructed
    inside the function to avoid Typer trying to serialize it to the CLI.
    """

    # build transform at runtime (Typer doesn't try to serialize this)
    transform = get_default_transforms(224)

    model = DishMultiViewRegressor.load_from_checkpoint("checkpoints/best-nutrition-0.ckpt")
    print(model.hparams)

    model.eval().to(device)

    tensors = []
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)

    out = transform(image=arr)
    # albumentations typically returns a dict {'image': ...}
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

    tensors.append(image_tensor)

    images = torch.stack(tensors, dim=0).unsqueeze(0).to(device)
    # images: (1, V, C, H, W)

    preds, attn = model(images)
    preds = preds.squeeze(0).cpu()

    result = {
        "total_calories": float(preds[0]),
        "total_mass": float(preds[1]),
        "total_fat": float(preds[2]),
        "total_carb": float(preds[3]),
        "total_protein": float(preds[4]),
        "num_images_used": len(tensors),
    }
    typer.echo(result)
    return result


if __name__ == "__main__":
    app()