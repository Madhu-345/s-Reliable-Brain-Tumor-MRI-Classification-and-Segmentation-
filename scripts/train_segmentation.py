"""
Segmentation training for the uncertainty-aware brain tumor MRI
classification and segmentation framework.

The segmentation branch uses:

    ResNet-50 backbone
        ↓
    Multi-scale feature extraction
        ↓
    Uncertainty-Calibrated Spatial Attention
        ↓
    Multi-scale segmentation decoder
        ↓
    Evidential segmentation head

The segmentation objective is:

    L_seg =
        L_ESL
        + lambda_dice * L_Dice
        + lambda_kl * L_DKL
        + lambda_cal * L_Cal
        + lambda_a * L_AU

BraTS input:
    T1, T1ce, T2, FLAIR

BraTS target:
    WT, TC, ET

The three segmentation regions are treated as independent binary
channels rather than mutually exclusive multiclass labels.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.brats_dataset import BraTSDataset

from losses.evidential import (
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


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""

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


def get_device(requested: str) -> torch.device:
    """Return the requested or automatically selected device."""

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


def get_segmentation_output(
    outputs: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Extract the segmentation output dictionary."""

    if "segmentation" in outputs:

        value = outputs["segmentation"]

        if isinstance(value, dict):
            return value

    if "segmentation_alpha" in outputs:

        result = {}

        for key, value in outputs.items():

            if key.startswith("segmentation_"):

                result[
                    key.replace(
                        "segmentation_",
                        "",
                    )
                ] = value

        return result

    raise KeyError(
        "Segmentation output was not found in model output."
    )


# ---------------------------------------------------------------------
# Attention–uncertainty alignment
# ---------------------------------------------------------------------


def compute_attention_uncertainty_loss(
    outputs: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute attention–uncertainty alignment loss.

    The UCSA module should return attention and uncertainty maps.
    If those maps are not present, a zero loss is returned.
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
            device=device,
        )

    if len(attention_maps) == 0:
        return torch.tensor(
            0.0,
            device=device,
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
            device=device,
        )

    return torch.stack(
        losses
    ).mean()


# ---------------------------------------------------------------------
# Segmentation losses
# ---------------------------------------------------------------------


def compute_segmentation_losses(
    outputs: dict[str, Any],
    targets: torch.Tensor,
    device: torch.device,
    lambda_kl: float,
    lambda_calibration: float,
    lambda_attention: float,
    lambda_dice: float,
) -> dict[str, torch.Tensor]:
    """
    Compute all losses used by the segmentation branch.
    """

    segmentation = get_segmentation_output(
        outputs
    )

    # --------------------------------------------------------------
    # Evidential segmentation loss
    # --------------------------------------------------------------

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

    loss_esl = (
        evidential_segmentation_loss(
            alpha=alpha,
            target=targets,
        )
    )

    # --------------------------------------------------------------
    # Dice loss
    # --------------------------------------------------------------

    probabilities = segmentation[
        "probabilities"
    ]

    loss_dice = dice_loss(
        probabilities,
        targets,
    )

    # --------------------------------------------------------------
    # Dirichlet KL regularization
    # --------------------------------------------------------------

    loss_kl = dirichlet_kl_loss(
        alpha
    )

    # --------------------------------------------------------------
    # Calibration loss
    # --------------------------------------------------------------

    loss_calibration = calibration_loss(
        probabilities,
        targets,
    )

    # --------------------------------------------------------------
    # Attention–uncertainty alignment
    # --------------------------------------------------------------

    loss_attention = (
        compute_attention_uncertainty_loss(
            outputs,
            device,
        )
    )

    # --------------------------------------------------------------
    # Total segmentation objective
    # --------------------------------------------------------------

    total_loss = (
        loss_esl
        + lambda_dice * loss_dice
        + lambda_kl * loss_kl
        + lambda_calibration * loss_calibration
        + lambda_attention * loss_attention
    )

    return {
        "evidential": loss_esl,
        "dice": loss_dice,
        "kl": loss_kl,
        "calibration": loss_calibration,
        "attention": loss_attention,
        "total": total_loss,
    }


# ---------------------------------------------------------------------
# Dice calculation
# ---------------------------------------------------------------------


def batch_dice(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Compute mean channel-wise Dice score.

    Expected tensors:

        probabilities: [B, C, H, W]
        targets:      [B, C, H, W]
    """

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = predictions.flatten(
        start_dim=2
    )

    targets = targets.float().flatten(
        start_dim=2
    )

    intersection = (
        predictions * targets
    ).sum(dim=2)

    denominator = (
        predictions.sum(dim=2)
        + targets.sum(dim=2)
    )

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        denominator + 1e-6
    )

    return dice.mean()


# ---------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    use_amp: bool,
    lambda_kl: float,
    lambda_calibration: float,
    lambda_attention: float,
    lambda_dice: float,
) -> dict[str, float]:
    """Train the segmentation branch for one epoch."""

    model.train()

    totals = {
        "total": 0.0,
        "evidential": 0.0,
        "dice": 0.0,
        "kl": 0.0,
        "calibration": 0.0,
        "attention": 0.0,
        "metric_dice": 0.0,
    }

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
                        device=device,
                        lambda_kl=lambda_kl,
                        lambda_calibration=(
                            lambda_calibration
                        ),
                        lambda_attention=(
                            lambda_attention
                        ),
                        lambda_dice=lambda_dice,
                    )
                )

                total_loss = losses[
                    "total"
                ]

        else:

            outputs = model(
                images,
                return_features=False,
            )

            losses = (
                compute_segmentation_losses(
                    outputs=outputs,
                    targets=targets,
                    device=device,
                    lambda_kl=lambda_kl,
                    lambda_calibration=(
                        lambda_calibration
                    ),
                    lambda_attention=(
                        lambda_attention
                    ),
                    lambda_dice=lambda_dice,
                )
            )

            total_loss = losses[
                "total"
            ]

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                "Non-finite segmentation loss "
                "encountered during training."
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

        metric_dice = batch_dice(
            get_segmentation_output(
                outputs
            )["probabilities"],
            targets,
        )

        totals["total"] += (
            total_loss.detach().item()
        )

        totals["evidential"] += (
            losses["evidential"]
            .detach()
            .item()
        )

        totals["dice"] += (
            losses["dice"]
            .detach()
            .item()
        )

        totals["kl"] += (
            losses["kl"]
            .detach()
            .item()
        )

        totals["calibration"] += (
            losses["calibration"]
            .detach()
            .item()
        )

        totals["attention"] += (
            losses["attention"]
            .detach()
            .item()
        )

        totals["metric_dice"] += (
            metric_dice.detach().item()
        )

        batches += 1

    return {
        key: value / max(batches, 1)
        for key, value in totals.items()
    }


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    lambda_kl: float,
    lambda_calibration: float,
    lambda_attention: float,
    lambda_dice: float,
) -> dict[str, float]:
    """Validate the segmentation model."""

    model.eval()

    totals = {
        "total": 0.0,
        "evidential": 0.0,
        "dice": 0.0,
        "kl": 0.0,
        "calibration": 0.0,
        "attention": 0.0,
        "metric_dice": 0.0,
    }

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
                        device=device,
                        lambda_kl=lambda_kl,
                        lambda_calibration=(
                            lambda_calibration
                        ),
                        lambda_attention=(
                            lambda_attention
                        ),
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
                    device=device,
                    lambda_kl=lambda_kl,
                    lambda_calibration=(
                        lambda_calibration
                    ),
                    lambda_attention=(
                        lambda_attention
                    ),
                    lambda_dice=lambda_dice,
                )
            )

        probabilities = (
            get_segmentation_output(
                outputs
            )["probabilities"]
        )

        metric_dice = batch_dice(
            probabilities,
            targets,
        )

        totals["total"] += (
            losses["total"]
            .detach()
            .item()
        )

        totals["evidential"] += (
            losses["evidential"]
            .detach()
            .item()
        )

        totals["dice"] += (
            losses["dice"]
            .detach()
            .item()
        )

        totals["kl"] += (
            losses["kl"]
            .detach()
            .item()
        )

        totals["calibration"] += (
            losses["calibration"]
            .detach()
            .item()
        )

        totals["attention"] += (
            losses["attention"]
            .detach()
            .item()
        )

        totals["metric_dice"] += (
            metric_dice.detach().item()
        )

        batches += 1

    return {
        key: value / max(batches, 1)
        for key, value in totals.items()
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
    """Save model checkpoint."""

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
    """Write training history to CSV."""

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
            "evidential brain tumor "
            "segmentation model."
        )
    )

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="BraTS dataset root.",
    )

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size.",
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
        help="Early stopping patience.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
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
        "--lambda-dice",
        type=float,
        default=1.0,
        help="Dice loss weight.",
    )

    parser.add_argument(
        "--lambda-kl",
        type=float,
        default=0.1,
        help="Dirichlet KL weight.",
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
    # Device
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
        help="Disable mixed precision.",
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
    # Seed and device
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
    # Dataset
    # --------------------------------------------------------------

    print(
        "\nLoading BraTS training dataset..."
    )

    train_dataset = BraTSDataset(
        root=args.data_root,
        split="train",
        augment=True,
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        "\nLoading BraTS validation dataset..."
    )

    val_dataset = BraTSDataset(
        root=args.data_root,
        split="val",
        augment=False,
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    # --------------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------------

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
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
        "\nSegmentation model initialized."
    )

    # --------------------------------------------------------------
    # Optimizer
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
    # Output directories
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
        / "segmentation_best.pt"
    )

    last_checkpoint = (
        checkpoint_dir
        / "segmentation_last.pt"
    )

    history_path = (
        output_dir
        / "segmentation_training_history.csv"
    )

    # --------------------------------------------------------------
    # Training state
    # --------------------------------------------------------------

    best_val_dice = -float(
        "inf"
    )

    epochs_without_improvement = 0

    history = []

    # --------------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------------

    print(
        "\nStarting segmentation training..."
    )

    print(
        "=" * 80
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            lambda_kl=args.lambda_kl,
            lambda_calibration=(
                args.lambda_calibration
            ),
            lambda_attention=(
                args.lambda_attention
            ),
            lambda_dice=args.lambda_dice,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=use_amp,
            lambda_kl=args.lambda_kl,
            lambda_calibration=(
                args.lambda_calibration
            ),
            lambda_attention=(
                args.lambda_attention
            ),
            lambda_dice=args.lambda_dice,
        )

        scheduler.step()

        learning_rate = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        # ----------------------------------------------------------
        # Improvement
        # ----------------------------------------------------------

        improved = (
            val_metrics["metric_dice"]
            > best_val_dice
        )

        if improved:

            best_val_dice = (
                val_metrics["metric_dice"]
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------------
        # Metrics
        # ----------------------------------------------------------

        metrics = {
            "train_loss":
                train_metrics["total"],

            "train_evidential_loss":
                train_metrics["evidential"],

            "train_dice_loss":
                train_metrics["dice"],

            "train_kl_loss":
                train_metrics["kl"],

            "train_calibration_loss":
                train_metrics["calibration"],

            "train_attention_loss":
                train_metrics["attention"],

            "train_dice":
                train_metrics["metric_dice"],

            "val_loss":
                val_metrics["total"],

            "val_evidential_loss":
                val_metrics["evidential"],

            "val_dice_loss":
                val_metrics["dice"],

            "val_kl_loss":
                val_metrics["kl"],

            "val_calibration_loss":
                val_metrics["calibration"],

            "val_attention_loss":
                val_metrics["attention"],

            "val_dice":
                val_metrics["metric_dice"],
        }

        # ----------------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------------

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
        # Save latest checkpoint
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
                **metrics,
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
            f"Train Loss: "
            f"{train_metrics['total']:.5f} | "
            f"Train Dice: "
            f"{train_metrics['metric_dice']:.4f} | "
            f"Val Loss: "
            f"{val_metrics['total']:.5f} | "
            f"Val Dice: "
            f"{val_metrics['metric_dice']:.4f} | "
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
        "Segmentation training completed."
    )

    print(
        f"Best validation Dice: "
        f"{best_val_dice:.4f}"
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