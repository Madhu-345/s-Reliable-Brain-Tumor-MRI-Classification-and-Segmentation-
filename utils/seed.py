"""
Reproducibility utilities for the uncertainty-aware multi-task
brain tumor MRI classification and segmentation framework.

Provides:
    - Global random seed configuration
    - Python random seed
    - NumPy random seed
    - PyTorch CPU/GPU random seeds
    - Deterministic CUDA configuration
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(
    seed: int = 42,
    deterministic: bool = True,
) -> None:
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed:
        Random seed used across Python, NumPy, and PyTorch.

    deterministic:
        If True, configure PyTorch/CUDA for deterministic
        execution where supported.

    Notes
    -----
    Deterministic execution can reduce performance and may
    restrict some CUDA operations.
    """

    if not isinstance(seed, int):
        raise TypeError(
            f"seed must be an integer, got {type(seed).__name__}"
        )

    if seed < 0:
        raise ValueError(
            "seed must be non-negative."
        )

    # -------------------------------------------------------------
    # Python
    # -------------------------------------------------------------

    random.seed(seed)

    # -------------------------------------------------------------
    # Hash-based reproducibility
    # -------------------------------------------------------------

    os.environ["PYTHONHASHSEED"] = str(seed)

    # -------------------------------------------------------------
    # NumPy
    # -------------------------------------------------------------

    np.random.seed(seed)

    # -------------------------------------------------------------
    # PyTorch CPU
    # -------------------------------------------------------------

    torch.manual_seed(seed)

    # -------------------------------------------------------------
    # PyTorch CUDA
    # -------------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # -------------------------------------------------------------
    # Deterministic behavior
    # -------------------------------------------------------------

    if deterministic:
        # cuDNN deterministic execution
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Use deterministic algorithms where available
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )

    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(
    worker_id: int,
) -> None:
    """
    Initialize a PyTorch DataLoader worker.

    This should be passed to DataLoader as:

        worker_init_fn=seed_worker

    The worker seed is derived from PyTorch's initial seed,
    allowing independent but reproducible workers.
    """

    worker_seed = torch.initial_seed() % (
        2**32
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_generator(
    seed: int = 42,
) -> torch.Generator:
    """
    Create a seeded PyTorch random generator.

    Useful for reproducible DataLoader shuffling.

    Parameters
    ----------
    seed:
        Random seed.

    Returns
    -------
    torch.Generator
        Seeded PyTorch generator.
    """

    if not isinstance(seed, int):
        raise TypeError(
            f"seed must be an integer, got {type(seed).__name__}"
        )

    if seed < 0:
        raise ValueError(
            "seed must be non-negative."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


def get_seed_state() -> dict:
    """
    Capture the current random-number-generator states.

    Returns
    -------
    dict
        Python, NumPy, PyTorch, and CUDA RNG states.
    """

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def set_seed_state(
    state: dict,
) -> None:
    """
    Restore previously captured random-number-generator states.
    """

    if "python" in state:
        random.setstate(
            state["python"]
        )

    if "numpy" in state:
        np.random.set_state(
            state["numpy"]
        )

    if "torch" in state:
        torch.set_rng_state(
            state["torch"]
        )

    if (
        "torch_cuda" in state
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            state["torch_cuda"]
        )


__all__ = [
    "set_seed",
    "seed_worker",
    "create_generator",
    "get_seed_state",
    "set_seed_state",
]