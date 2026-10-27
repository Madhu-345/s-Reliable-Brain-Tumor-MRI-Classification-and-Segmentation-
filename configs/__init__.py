"""
Configuration package for the uncertainty-aware multi-task
brain tumor MRI classification and segmentation framework.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main project directories
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Default configuration
DEFAULT_CONFIG = CONFIG_DIR / "multitask.yaml"

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "CONFIG_DIR",
    "OUTPUT_DIR",
    "CHECKPOINT_DIR",
    "DEFAULT_CONFIG",
]