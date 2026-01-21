from __future__ import annotations

from pathlib import Path

import albumentations as A
import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from wizard_ops.train import train

_CONFIG_DIR = (Path(__file__).resolve().parents[2] / "configs").as_posix()


@hydra.main(
    version_base=None,
    config_path=_CONFIG_DIR,
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    if cfg.mode == "train":
        logger.info("Starting training...")
        train_transforms = A.Compose([
            A.Resize(cfg.data.image_size, cfg.data.image_size),
            A.HorizontalFlip(p=0.5),
            A.GridDropout(p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            A.ToTensorV2(),
        ])
        train(cfg, train_transform=train_transforms)
    elif cfg.mode == "evaluate":
        logger.info("Starting evaluation...")
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
