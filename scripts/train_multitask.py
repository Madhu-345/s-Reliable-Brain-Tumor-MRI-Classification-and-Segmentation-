"""
Joint multi-task training for the uncertainty-aware brain tumor
MRI classification and segmentation framework.

The model contains:
    - ResNet-50 shared feature extractor
    - Uncertainty-Calibrated Spatial Attention (UCSA)
    - Evidential classification head
    - Multi-scale segmentation decoder
    - Evidential segmentation head

The total objective is:

    L_total =
        lambda_c  * L_ECL
      + lambda_s  * L_ESL
      + lambda_k  * L_DKL
      + lambda_cal * L_Cal
      + lambda_a  * L_AU

The classification and segmentation datasets are evaluated
independently because the current datasets have different input
structures:

    Classification:
        3-channel MRI image

    BraTS segmentation:
        4-channel multi-modal MRI
        (T1, T1ce, T2, FLAIR)

Training alternates between classification and segmentation
batches so that both tasks update the same shared model.
"""

from __future__ import annotations

import argparse
import csv
import random
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.classification_dataset import (
    BrainTumorClassificationDataset,
)

from data.brats_dataset import (
    BraTSDataset,
)

from losses.evidential import (
    evidential_classification_loss,
    evidential_segmentation_loss,
)

from losses.dice import (
    dice_loss,
)

from losses.kl_regularization import (
    dirichlet_kl_loss,
)

from losses.calibration import (
    calibration_loss,
)

from losses.attention_uncertainty import (
    attention_uncertainty_alignment_loss,
)

from models.multitask_model import (
    UncertaintyAwareMultiTaskModel,
)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def set_seed(
    seed: int,
) -> None:
    """Set random seeds."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------


def get_device(
    requested: str,
) -> torch.device:
    """Select training device."""

    if requested == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ---------------------------------------------------------------------
# Output extraction
# ---------------------------------------------------------------------


def get_classification_output(
    outputs: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Extract classification outputs."""

    if "classification" in outputs:

        value = outputs["classification"]

        if isinstance(value, dict):
            return value

    if "classification_alpha" in outputs:

        result = {}

        for key, value in outputs.items():

            if key.startswith(
                "classification_"
            ):

                result[
                    key.replace(
                        "classification_",
                        "",
                    )
                ] = value

        return result

    raise KeyError(
        "Classification output was not found."
    )


def get_segmentation_output(
    outputs: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Extract segmentation outputs."""

    if "segmentation" in outputs:

        value = outputs["segmentation"]

        if isinstance(value, dict):
            return value

    if "segmentation_alpha" in outputs:

        result = {}

        for key, value in outputs.items():

            if key.startswith(
                "segmentation_"
            ):

                result[
                    key.replace(
                        "segmentation_",
                        "",
                    )
                ] = value

        return result

    raise KeyError(
        "Segmentation output was not found."
    )


# ---------------------------------------------------------------------
# Evidential losses
# ---------------------------------------------------------------------


def classification_evidential_loss(
    classification: dict,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute evidential classification loss."""

    if "alpha" in classification:

        alpha = classification["alpha"]

    elif "evidence" in classification:

        alpha = (
            classification["evidence"]
            + 1.0
        )

    else:
        raise KeyError(
            "Classification output must contain "
            "'alpha' or 'evidence'."
        )

    return evidential_classification_loss(
        alpha=alpha,
        target=targets,
    )


def segmentation_evidential_loss(
    segmentation: dict,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute evidential segmentation loss."""

    if "alpha" in segmentation:

        alpha = segmentation["alpha"]

    elif "evidence" in segmentation:

        alpha = (
            segmentation["evidence"]
            + 1.0
        )

    else:
        raise KeyError(
            "Segmentation output must contain "
            "'alpha' or 'evidence'."
        )

    return evidential_segmentation_loss(
        alpha=alpha,
        target=targets,
    )


# ---------------------------------------------------------------------
# KL loss
# ---------------------------------------------------------------------


def classification_kl_loss(
    classification: dict,
) -> torch.Tensor:
    """Compute classification Dirichlet KL loss."""

    if "alpha" not in classification:
        raise KeyError(
            "Classification output does not contain alpha."
        )

    return dirichlet_kl_loss(
        classification["alpha"]
    )


def segmentation_kl_loss(
    segmentation: dict,
) -> torch.Tensor:
    """Compute segmentation Dirichlet KL loss."""

    if "alpha" not in segmentation:
        raise KeyError(
            "Segmentation output does not contain alpha."
        )

    return dirichlet_kl_loss(
        segmentation["alpha"]
    )


# ---------------------------------------------------------------------
# Attention / uncertainty loss
# ---------------------------------------------------------------------


def compute_attention_uncertainty_loss(
    outputs: dict[str, Any],
) -> torch.Tensor:
    """
    Compute attention–uncertainty alignment loss.

    The model returns attention and uncertainty maps for the
    UCSA blocks. If they are unavailable, a zero loss is returned.
    """

    attention_maps = outputs.get(
        "attention_maps"
    )

    uncertainty_maps = outputs.get(
        "uncertainty_maps"
    )

    if (
        attention_maps is None
        or uncertainty_maps is None
    ):
        return torch.tensor(
            0.0,
            device=next(
                iter(
                    outputs.values()
                )
            ).device
            if outputs
            else "cpu",
        )

    if len(attention_maps) == 0:
        return torch.tensor(
            0.0,
            device=attention_maps[0].device
            if attention_maps
            else "cpu",
        )

    losses = []

    for attention, uncertainty in zip(
        attention_maps,
        uncertainty_maps,
    ):

        losses.append(
            attention_uncertainty_alignment_loss(
                attention=attention,
                uncertainty=uncertainty,
            )
        )

    if not losses:
        return torch.tensor(
            0.0,
            device=attention_maps[0].device,
        )

    return torch.stack(
        losses
    ).mean()


# ---------------------------------------------------------------------
# Calibration loss
# ---------------------------------------------------------------------


def compute_calibration_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute calibration loss.

    The exact calibration implementation is delegated to
    losses.calibration.calibration_loss.
    """

    return calibration_loss(
        probabilities,
        targets,
    )


# ---------------------------------------------------------------------
# Classification task loss
# ---------------------------------------------------------------------


def compute_classification_losses(
    outputs: dict[str, Any],
    targets: torch.Tensor,
    lambda_kl: float,
    lambda_cal: float,
    lambda_attention: float,
) -> dict[str, torch.Tensor]:
    """Compute all classification-side losses."""

    classification = (
        get_classification_output(
            outputs
        )
    )

    loss_ecl = (
        classification_evidential_loss(
            classification,
            targets,
        )
    )

    loss_kl = (
        classification_kl_loss(
            classification
        )
    )

    probabilities = classification[
        "probabilities"
    ]

    loss_cal = (
        compute_calibration_loss(
            probabilities,
            targets,
        )
    )

    loss_attention = (
        compute_attention_uncertainty_loss(
            outputs
        )
    )

    total = (
        loss_ecl
        + lambda_kl * loss_kl
        + lambda_cal * loss_cal
        + lambda_attention * loss_attention
    )

    return {
        "evidential": loss_ecl,
        "kl": loss_kl,
        "calibration": loss_cal,
        "attention": loss_attention,
        "total": total,
    }


# ---------------------------------------------------------------------
# Segmentation task loss
# ---------------------------------------------------------------------


def compute_segmentation_losses(
    outputs: dict[str, Any],
    targets: torch.Tensor,
    lambda_kl: float,
    lambda_cal: float,
    lambda_attention: float,
    lambda_dice: float,
) -> dict[str, torch.Tensor]:
    """Compute all segmentation-side losses."""

    segmentation = (
        get_segmentation_output(
            outputs
        )
    )

    loss_esl = (
        segmentation_evidential_loss(
            segmentation,
            targets,
        )
    )

    loss_dice = dice_loss(
        segmentation["probabilities"],
        targets,
    )

    loss_kl = (
        segmentation_kl_loss(
            segmentation
        )
    )

    probabilities = segmentation[
        "probabilities"
    ]

    loss_cal = (
        compute_calibration_loss(
            probabilities,
            targets,
        )
    )

    loss_attention = (
        compute_attention_uncertainty_loss(
            outputs
        )
    )

    total = (
        loss_esl
        + lambda_dice * loss_dice
        + lambda_kl * loss_kl
        + lambda_cal * loss_cal
        + lambda_attention * loss_attention
    )

    return {
        "evidential": loss_esl,
        "dice": loss_dice,
        "kl": loss_kl,
        "calibration": loss_cal,
        "attention": loss_attention,
        "total": total,
    }


# ---------------------------------------------------------------------
# Classification epoch
# ---------------------------------------------------------------------


def train_classification_batch(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    use_amp: bool,
    lambda_kl: float,
    lambda_cal: float,
    lambda_attention: float,
) -> dict[str, float]:
    """Perform one classification update."""

    images = batch["image"].to(
        device,
        non_blocking=True,
    )

    targets = batch["label"].to(
        device,
        non_blocking=True,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    if use_amp:

        with autocast():

            outputs = model(
                images,
                return_features=False,
            )

            losses = (
                compute_classification_losses(
                    outputs=outputs,
                    targets=targets,
                    lambda_kl=lambda_kl,
                    lambda_cal=lambda_cal,
                    lambda_attention=lambda_attention,
                )
            )

            total_loss = losses["total"]

    else:

        outputs = model(
            images,
            return_features=False,
        )

        losses = (
            compute_classification_losses(
                outputs=outputs,
                targets=targets,
                lambda_kl=lambda_kl,
                lambda_cal=lambda_cal,
                lambda_attention=lambda_attention,
            )
        )

        total_loss = losses["total"]

    if not torch.isfinite(total_loss):
        raise RuntimeError(
            "Non-finite classification loss."
        )

    if scaler is not None:

        scaler.scale(
            total_loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

    else:

        total_loss.backward()

        optimizer.step()

    probabilities = (
        get_classification_output(
            outputs
        )["probabilities"]
    )

    predictions = probabilities.argmax(
        dim=-1
    )

    accuracy = (
        predictions == targets
    ).float().mean().item()

    return {
        "total": float(
            total_loss.detach().item()
        ),
        "evidential": float(
            losses["evidential"]
            .detach()
            .item()
        ),
        "kl": float(
            losses["kl"]
            .detach()
            .item()
        ),
        "calibration": float(
            losses["calibration"]
            .detach()
            .item()
        ),
        "attention": float(
            losses["attention"]
            .detach()
            .item()
        ),
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------
# Segmentation epoch
# ---------------------------------------------------------------------


def train_segmentation_batch(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    use_amp: bool,
    lambda_kl: float,
    lambda_cal: float,
    lambda_attention: float,
    lambda_dice: float,
) -> dict[str, float]:
    """Perform one segmentation update."""

    images = batch["image"].to(
        device,
        non_blocking=True,
    )

    targets = batch["mask"].to(
        device,
        non_blocking=True,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    if use_amp:

        with autocast():

            outputs = model(
                images,
                return_features=False,
            )

            losses = (
                compute_segmentation_losses(
                    outputs=outputs,
                    targets=targets,
                    lambda_kl=lambda_kl,
                    lambda_cal=lambda_cal,
                    lambda_attention=lambda_attention,
                    lambda_dice=lambda_dice,
                )
            )

            total_loss = losses["total"]

    else:

        outputs = model(
            images,
            return_features=False,
        )

        losses = (
            compute_segmentation_losses(
                outputs=outputs,
                targets=targets,
                lambda_kl=lambda_kl,
                lambda_cal=lambda_cal,
                lambda_attention=lambda_attention,
                lambda_dice=lambda_dice,
            )
        )

        total_loss = losses["total"]

    if not torch.isfinite(total_loss):
        raise RuntimeError(
            "Non-finite segmentation loss."
        )

    if scaler is not None:

        scaler.scale(
            total_loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

    else:

        total_loss.backward()

        optimizer.step()

    return {
        "total": float(
            total_loss.detach().item()
        ),
        "evidential": float(
            losses["evidential"]
            .detach()
            .item()
        ),
        "dice": float(
            losses["dice"]
            .detach()
            .item()
        ),
        "kl": float(
            losses["kl"]
            .detach()
            .item()
        ),
        "calibration": float(
            losses["calibration"]
            .detach()
            .item()
        ),
        "attention": float(
            losses["attention"]
            .detach()
            .item()
        ),
    }


# ---------------------------------------------------------------------
# Validation classification
# ---------------------------------------------------------------------


@torch.no_grad()
def validate_classification(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    lambda_kl: float,
    lambda_cal: float,
    lambda_attention: float,
) -> dict[str, float]:
    """Validate classification branch."""

    model.eval()

    total_loss = 0.0
    correct = 0
    samples = 0

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        targets = batch["label"].to(
            device,
            non_blocking=True,
        )

        if use_amp:

            with autocast():

                outputs = model(
                    images,
                    return_features=False,
                )

                losses = (
                    compute_classification_losses(
                        outputs=outputs,
                        targets=targets,
                        lambda_kl=lambda_kl,
                        lambda_cal=lambda_cal,
                        lambda_attention=lambda_attention,
                    )
                )

        else:

            outputs = model(
                images,
                return_features=False,
            )

            losses = (
                compute_classification_losses(
                    outputs=outputs,
                    targets=targets,
                    lambda_kl=lambda_kl,
                    lambda_cal=lambda_cal,
                    lambda_attention=lambda_attention,
                )
            )

        probabilities = (
            get_classification_output(
                outputs
            )["probabilities"]
        )

        predictions = probabilities.argmax(
            dim=-1
        )

        batch_size = (
            targets.shape[0]
        )

        total_loss += (
            losses["total"]
            .detach()
            .item()
            * batch_size
        )

        correct += int(
            (
                predictions == targets
            ).sum().item()
        )

        samples += batch_size

    return {
        "loss": total_loss
        / max(samples, 1),
        "accuracy": (
            100.0
            * correct
            / max(samples, 1)
        ),
    }


# ---------------------------------------------------------------------
# Validation segmentation
# ---------------------------------------------------------------------


@torch.no_grad()
def validate_segmentation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    lambda_kl: float,
    lambda_cal: float,
    lambda_attention: float,
    lambda_dice: float,
) -> dict[str, float]:
    """Validate segmentation branch."""

    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    batches = 0

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        targets = batch["mask"].to(
            device,
            non_blocking=True,
        )

        if use_amp:

            with autocast():

                outputs = model(
                    images,
                    return_features=False,
                )

                losses = (
                    compute_segmentation_losses(
                        outputs=outputs,
                        targets=targets,
                        lambda_kl=lambda_kl,
                        lambda_cal=lambda_cal,
                        lambda_attention=lambda_attention,
                        lambda_dice=lambda_dice,
                    )
                )

        else:

            outputs = model(
                images,
                return_features=False,
            )

            losses = (
                compute_segmentation_losses(
                    outputs=outputs,
                    targets=targets,
                    lambda_kl=lambda_kl,
                    lambda_cal=lambda_cal,
                    lambda_attention=lambda_attention,
                    lambda_dice=lambda_dice,
                )
            )

        probabilities = (
            get_segmentation_output(
                outputs
            )["probabilities"]
        )

        predictions = (
            probabilities >= 0.5
        ).float()

        intersection = (
            predictions * targets
        ).sum(
            dim=(0, 2, 3)
        )

        denominator = (
            predictions.sum(
                dim=(0, 2, 3)
            )
            + targets.sum(
                dim=(0, 2, 3)
            )
        )

        dice_per_class = (
            (2.0 * intersection + 1e-6)
            / (denominator + 1e-6)
        )

        mean_dice = (
            dice_per_class.mean()
        )

        total_loss += (
            losses["total"]
            .detach()
            .item()
        )

        total_dice += (
            mean_dice.detach().item()
        )

        batches += 1

    return {
        "loss": (
            total_loss
            / max(batches, 1)
        ),
        "dice": (
            total_dice
            / max(batches, 1)
        ),
    }


# ---------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    """Save a multitask checkpoint."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        },
        path,
    )


# ---------------------------------------------------------------------
# History
# ---------------------------------------------------------------------


def save_history(
    history: list[dict[str, Any]],
    path: Path,
) -> None:
    """Save training history."""

    if not history:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        history[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train the uncertainty-aware "
            "multi-task evidential model."
        )
    )

    # --------------------------------------------------------------
    # Dataset paths
    # --------------------------------------------------------------

    parser.add_argument(
        "--classification-root",
        type=str,
        required=True,
        help=(
            "Classification dataset root."
        ),
    )

    parser.add_argument(
        "--segmentation-root",
        type=str,
        required=True,
        help=(
            "BraTS segmentation dataset root."
        ),
    )

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for both tasks.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Adam weight decay.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help=(
            "Early stopping patience."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    # --------------------------------------------------------------
    # Loss weights
    # --------------------------------------------------------------

    parser.add_argument(
        "--lambda-classification",
        type=float,
        default=1.0,
        help="Classification loss weight.",
    )

    parser.add_argument(
        "--lambda-segmentation",
        type=float,
        default=1.0,
        help="Segmentation loss weight.",
    )

    parser.add_argument(
        "--lambda-kl",
        type=float,
        default=0.1,
        help="Dirichlet KL loss weight.",
    )

    parser.add_argument(
        "--lambda-calibration",
        type=float,
        default=0.1,
        help="Calibration loss weight.",
    )

    parser.add_argument(
        "--lambda-attention",
        type=float,
        default=0.1,
        help=(
            "Attention–uncertainty alignment "
            "loss weight."
        ),
    )

    parser.add_argument(
        "--lambda-dice",
        type=float,
        default=1.0,
        help="Dice loss weight.",
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help=(
            "Use ImageNet-pretrained "
            "ResNet-50 weights."
        ),
    )

    # --------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        help="Training device.",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help=(
            "Disable automatic mixed precision."
        ),
    )

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Seed/device
    # --------------------------------------------------------------

    set_seed(
        args.seed
    )

    device = get_device(
        args.device
    )

    print(
        f"Training device: {device}"
    )

    use_amp = (
        not args.no_amp
        and device.type == "cuda"
    )

    scaler = (
        GradScaler()
        if use_amp
        else None
    )

    print(
        f"Mixed precision: {use_amp}"
    )

    # --------------------------------------------------------------
    # Datasets
    # --------------------------------------------------------------

    print(
        "\nLoading classification dataset..."
    )

    classification_train = (
        BrainTumorClassificationDataset(
            root=args.classification_root,
            split="train",
            image_size=224,
            augment=True,
        )
    )

    classification_val = (
        BrainTumorClassificationDataset(
            root=args.classification_root,
            split="val",
            image_size=224,
            augment=False,
        )
    )

    print(
        f"Classification train: "
        f"{len(classification_train)}"
    )

    print(
        f"Classification validation: "
        f"{len(classification_val)}"
    )

    print(
        "\nLoading BraTS dataset..."
    )

    segmentation_train = (
        BraTSDataset(
            root=args.segmentation_root,
            split="train",
            augment=True,
        )
    )

    segmentation_val = (
        BraTSDataset(
            root=args.segmentation_root,
            split="val",
            augment=False,
        )
    )

    print(
        f"Segmentation train: "
        f"{len(segmentation_train)}"
    )

    print(
        f"Segmentation validation: "
        f"{len(segmentation_val)}"
    )

    # --------------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------------

    pin_memory = (
        device.type == "cuda"
    )

    classification_train_loader = (
        DataLoader(
            classification_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    )

    classification_val_loader = (
        DataLoader(
            classification_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    )

    segmentation_train_loader = (
        DataLoader(
            segmentation_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    )

    segmentation_val_loader = (
        DataLoader(
            segmentation_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model = (
        UncertaintyAwareMultiTaskModel(
            num_classes=4,
            segmentation_classes=3,
            input_channels=4,
            classification_channels=3,
            pretrained=args.pretrained,
        )
    )

    model.to(device)

    print(
        "\nMulti-task model initialized."
    )

    # --------------------------------------------------------------
    # Optimizer and scheduler
    # --------------------------------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------

    checkpoint_dir = Path(
        args.checkpoint_dir
    )

    output_dir = Path(
        args.output_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint = (
        checkpoint_dir
        / "multitask_best.pt"
    )

    last_checkpoint = (
        checkpoint_dir
        / "multitask_last.pt"
    )

    history_path = (
        output_dir
        / "multitask_training_history.csv"
    )

    # --------------------------------------------------------------
    # Training state
    # --------------------------------------------------------------

    best_score = float(
        "inf"
    )

    epochs_without_improvement = 0

    history = []

    # The larger loader determines the number of task updates.
    steps_per_epoch = max(
        len(
            classification_train_loader
        ),
        len(
            segmentation_train_loader
        ),
    )

    print(
        f"\nSteps per epoch: "
        f"{steps_per_epoch}"
    )

    print(
        "\nStarting multi-task training..."
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        classification_iterator = cycle(
            classification_train_loader
        )

        segmentation_iterator = cycle(
            segmentation_train_loader
        )

        classification_loss_sum = 0.0
        segmentation_loss_sum = 0.0

        classification_correct = 0
        classification_samples = 0

        # ----------------------------------------------------------
        # Alternate task updates
        # ----------------------------------------------------------

        for step in range(
            steps_per_epoch
        ):

            # ------------------------------------------------------
            # Classification update
            # ------------------------------------------------------

            classification_batch = next(
                classification_iterator
            )

            cls_result = (
                train_classification_batch(
                    model=model,
                    batch=classification_batch,
                    optimizer=optimizer,
                    device=device,
                    scaler=scaler,
                    use_amp=use_amp,
                    lambda_kl=args.lambda_kl,
                    lambda_cal=args.lambda_calibration,
                    lambda_attention=args.lambda_attention,
                )
            )

            classification_loss_sum += (
                cls_result["total"]
            )

            batch_size = (
                classification_batch[
                    "label"
                ].shape[0]
            )

            classification_correct += int(
                cls_result["accuracy"]
                * batch_size
            )

            classification_samples += (
                batch_size
            )

            # ------------------------------------------------------
            # Segmentation update
            # ------------------------------------------------------

            segmentation_batch = next(
                segmentation_iterator
            )

            seg_result = (
                train_segmentation_batch(
                    model=model,
                    batch=segmentation_batch,
                    optimizer=optimizer,
                    device=device,
                    scaler=scaler,
                    use_amp=use_amp,
                    lambda_kl=args.lambda_kl,
                    lambda_cal=args.lambda_calibration,
                    lambda_attention=args.lambda_attention,
                    lambda_dice=args.lambda_dice,
                )
            )

            segmentation_loss_sum += (
                seg_result["total"]
            )

        # ----------------------------------------------------------
        # Training summaries
        # ----------------------------------------------------------

        mean_cls_loss = (
            classification_loss_sum
            / steps_per_epoch
        )

        mean_seg_loss = (
            segmentation_loss_sum
            / steps_per_epoch
        )

        train_cls_accuracy = (
            100.0
            * classification_correct
            / max(
                classification_samples,
                1,
            )
        )

        # ----------------------------------------------------------
        # Validation
        # ----------------------------------------------------------

        cls_val = (
            validate_classification(
                model=model,
                loader=classification_val_loader,
                device=device,
                use_amp=use_amp,
                lambda_kl=args.lambda_kl,
                lambda_cal=args.lambda_calibration,
                lambda_attention=args.lambda_attention,
            )
        )

        seg_val = (
            validate_segmentation(
                model=model,
                loader=segmentation_val_loader,
                device=device,
                use_amp=use_amp,
                lambda_kl=args.lambda_kl,
                lambda_cal=args.lambda_calibration,
                lambda_attention=args.lambda_attention,
                lambda_dice=args.lambda_dice,
            )
        )

        scheduler.step()

        learning_rate = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        # ----------------------------------------------------------
        # Joint validation score
        # ----------------------------------------------------------

        joint_score = (
            args.lambda_classification
            * cls_val["loss"]
            + args.lambda_segmentation
            * seg_val["loss"]
        )

        improved = (
            joint_score
            < best_score
        )

        if improved:

            best_score = joint_score
            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------------
        # Save best
        # ----------------------------------------------------------

        metrics = {
            "classification_train_loss":
                mean_cls_loss,

            "classification_train_accuracy":
                train_cls_accuracy,

            "segmentation_train_loss":
                mean_seg_loss,

            "classification_val_loss":
                cls_val["loss"],

            "classification_val_accuracy":
                cls_val["accuracy"],

            "segmentation_val_loss":
                seg_val["loss"],

            "segmentation_val_dice":
                seg_val["dice"],

            "joint_validation_score":
                joint_score,
        }

        if improved:

            save_checkpoint(
                path=best_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
            )

        # ----------------------------------------------------------
        # Always save latest
        # ----------------------------------------------------------

        save_checkpoint(
            path=last_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
        )

        # ----------------------------------------------------------
        # History
        # ----------------------------------------------------------

        history.append(
            {
                "epoch": epoch,
                "classification_train_loss":
                    mean_cls_loss,
                "classification_train_accuracy":
                    train_cls_accuracy,
                "segmentation_train_loss":
                    mean_seg_loss,
                "classification_val_loss":
                    cls_val["loss"],
                "classification_val_accuracy":
                    cls_val["accuracy"],
                "segmentation_val_loss":
                    seg_val["loss"],
                "segmentation_val_dice":
                    seg_val["dice"],
                "joint_validation_score":
                    joint_score,
                "learning_rate":
                    learning_rate,
            }
        )

        save_history(
            history,
            history_path,
        )

        marker = (
            " *"
            if improved
            else ""
        )

        print(
            f"Epoch "
            f"{epoch:03d}/{args.epochs:03d} | "
            f"Cls Train Loss: "
            f"{mean_cls_loss:.5f} | "
            f"Cls Val Acc: "
            f"{cls_val['accuracy']:.2f}% | "
            f"Seg Train Loss: "
            f"{mean_seg_loss:.5f} | "
            f"Seg Val Dice: "
            f"{seg_val['dice']:.4f} | "
            f"Joint: "
            f"{joint_score:.5f} | "
            f"LR: "
            f"{learning_rate:.7f}"
            f"{marker}"
        )

        # ----------------------------------------------------------
        # Early stopping
        # ----------------------------------------------------------

        if (
            epochs_without_improvement
            >= args.patience
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "Multi-task training completed."
    )

    print(
        f"Best joint validation score: "
        f"{best_score:.5f}"
    )

    print(
        f"Best checkpoint: "
        f"{best_checkpoint.resolve()}"
    )

    print(
        f"Last checkpoint: "
        f"{last_checkpoint.resolve()}"
    )

    print(
        f"Training history: "
        f"{history_path.resolve()}"
    )


if __name__ == "__main__":
    main()