"""
Checkpoint utilities for the uncertainty-aware multi-task
brain tumor MRI classification and segmentation framework.

Provides:
    - Save model checkpoints
    - Load model checkpoints
    - Save best checkpoints
    - Restore optimizer state
    - Restore scheduler state
    - Restore AMP GradScaler state
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


# ---------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
    is_best: bool = False,
) -> Path:
    """
    Save a training checkpoint.

    Parameters
    ----------
    path:
        Destination checkpoint path.

    model:
        Model whose parameters should be saved.

    optimizer:
        Optional optimizer.

    scheduler:
        Optional learning-rate scheduler.

    scaler:
        Optional AMP GradScaler.

    epoch:
        Current epoch number.

    metrics:
        Optional dictionary containing training/validation metrics.

    config:
        Optional experiment configuration.

    is_best:
        Whether this checkpoint represents the best model.

    Returns
    -------
    Path
        Path to the saved checkpoint.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
    }

    if optimizer is not None:
        checkpoint[
            "optimizer_state_dict"
        ] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint[
            "scheduler_state_dict"
        ] = scheduler.state_dict()

    if scaler is not None:
        checkpoint[
            "scaler_state_dict"
        ] = scaler.state_dict()

    if config is not None:
        checkpoint[
            "config"
        ] = config

    checkpoint[
        "is_best"
    ] = is_best

    torch.save(
        checkpoint,
        path,
    )

    return path


# ---------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load a checkpoint and optionally restore optimizer, scheduler,
    and AMP scaler states.

    Parameters
    ----------
    path:
        Checkpoint file.

    model:
        Model into which parameters are loaded.

    optimizer:
        Optional optimizer to restore.

    scheduler:
        Optional scheduler to restore.

    scaler:
        Optional AMP scaler to restore.

    device:
        Device used for checkpoint loading.

    strict:
        Whether model state-dict loading should be strict.

    Returns
    -------
    dict
        Complete checkpoint dictionary.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if (
        not isinstance(
            checkpoint,
            dict,
        )
        or "model_state_dict"
        not in checkpoint
    ):
        raise ValueError(
            f"Invalid checkpoint format: {path}"
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=strict,
    )

    if (
        optimizer is not None
        and "optimizer_state_dict"
        in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    if (
        scheduler is not None
        and "scheduler_state_dict"
        in checkpoint
    ):
        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

    if (
        scaler is not None
        and "scaler_state_dict"
        in checkpoint
    ):
        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )

    return checkpoint


# ---------------------------------------------------------------------
# Save best checkpoint
# ---------------------------------------------------------------------


def save_best_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """
    Save a checkpoint explicitly marked as the best model.
    """

    return save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        metrics=metrics,
        config=config,
        is_best=True,
    )


# ---------------------------------------------------------------------
# Checkpoint existence
# ---------------------------------------------------------------------


def checkpoint_exists(
    path: str | Path,
) -> bool:
    """
    Check whether a checkpoint exists.
    """

    return Path(path).is_file()


# ---------------------------------------------------------------------
# Extract checkpoint metadata
# ---------------------------------------------------------------------


def checkpoint_info(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Return lightweight checkpoint metadata.

    The model parameters are loaded only to inspect the checkpoint
    structure; the returned dictionary contains epoch, metrics,
    configuration, and best-model status.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    return {
        "path": str(path),
        "epoch": checkpoint.get(
            "epoch"
        ),
        "metrics": checkpoint.get(
            "metrics",
            {},
        ),
        "config": checkpoint.get(
            "config"
        ),
        "is_best": checkpoint.get(
            "is_best",
            False,
        ),
    }


# ---------------------------------------------------------------------
# Find latest checkpoint
# ---------------------------------------------------------------------


def find_latest_checkpoint(
    directory: str | Path,
    pattern: str = "*.pt",
) -> Path | None:
    """
    Find the most recently modified checkpoint in a directory.
    """

    directory = Path(directory)

    if not directory.exists():
        return None

    checkpoints = list(
        directory.glob(pattern)
    )

    if not checkpoints:
        return None

    return max(
        checkpoints,
        key=lambda path: path.stat().st_mtime,
    )


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "save_best_checkpoint",
    "checkpoint_exists",
    "checkpoint_info",
    "find_latest_checkpoint",
]