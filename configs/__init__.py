"""Configuration utilities for wizard_ops.

Provides functions to load and access the config.yaml configuration file.
"""

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).parent
CONFIG_PATH = CONFIG_DIR / "config.yaml"


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load the configuration from a YAML file.

    Args:
        config_path: Path to the config file. Defaults to configs/config.yaml.

    Returns:
        Dictionary containing the configuration.
    """
    if config_path is None:
        config_path = CONFIG_PATH
    elif isinstance(config_path, str):
        config_path = Path(config_path)

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_config() -> dict[str, Any]:
    """Get the default configuration.

    Returns:
        Dictionary containing the configuration from configs/config.yaml.
    """
    return load_config()


__all__ = ["load_config", "get_config", "CONFIG_PATH", "CONFIG_DIR"]
