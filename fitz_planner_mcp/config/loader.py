# fitz_planner_mcp/config/loader.py
"""
Configuration loading with auto-creation of defaults.

Uses platformdirs for cross-platform config directory management.
"""

import logging
from pathlib import Path

import yaml
from platformdirs import user_config_path

from .schema import FitzPlannerConfig

logger = logging.getLogger(__name__)


def get_config_path() -> Path:
    """Get path to config file, ensuring config directory exists."""
    config_dir = user_config_path("fitz-planner", ensure_exists=True)
    return config_dir / "config.yaml"


def load_config() -> FitzPlannerConfig:
    """
    Load configuration from YAML file.

    If config file doesn't exist, creates it with defaults.
    Returns validated Pydantic model.
    """
    config_path = get_config_path()

    if not config_path.exists():
        # Create default config
        default_config = FitzPlannerConfig()
        config_dict = default_config.model_dump(mode="json")

        # Write to YAML
        with config_path.open("w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Created default config at {config_path}")
        return default_config

    # Load existing config
    with config_path.open("r") as f:
        config_data = yaml.safe_load(f)

    # Parse and validate with Pydantic
    config = FitzPlannerConfig(**config_data)
    logger.info(f"Loaded config from {config_path}")
    return config
