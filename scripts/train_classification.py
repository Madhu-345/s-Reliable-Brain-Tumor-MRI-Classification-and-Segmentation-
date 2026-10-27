"""
Train the evidential brain tumor MRI classification branch.

Training configuration
----------------------
Model:
    ResNet-50 backbone
    Uncertainty-Calibrated Spatial Attention
    Evidential classification head

Optimizer:
    Adam
    Learning rate: 1e-4
    Weight decay: 1e-5

Training:
    Batch size: 16
    Epochs: 100
    Cosine annealing
    Validation-based early stopping

Outputs
-------
checkpoints/
    classification_best.pt
    classification_last.pt

outputs/
    classification_training_history.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.classification_dataset import (
    BrainTumorClassificationDataset,
)

from losses.evidential import (
    evidential_classification_loss,
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
    """
    Set random seeds for reproducibility.
    """

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
    """
    Select training device.
    """

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
# Checkpoint utilities
# ---------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    best_val_loss: float,
) -> None:
    """
    Save a training checkpoint.
    """

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
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "best_val_loss": best_val_loss,
        },
        path,
    )


# ---------------------------------------------------------------------
# Classification output
# ---------------------------------------------------------------------


def get_classification_output(
    outputs: dict,
) -> dict:
    """
    Extract classification outputs from the multi-task model.
    """

    if "classification" in outputs:

        classification = outputs[
            "classification"
        ]

        if isinstance(
            classification,
            dict,
        ):
            return classification

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
        "Classification output was not found "
        "in model output."
    )


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------


def compute_classification_loss(
    classification: dict,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the evidential classification loss.

    The evidential loss operates on the model's Dirichlet
    parameters and class targets.
    """

    if "alpha" in classification:

        alpha = classification[
            "alpha"
        ]

    elif "evidence" in classification:

        evidence = classification[
            "evidence"
        ]

        alpha = evidence + 1.0

    else:
        raise KeyError(
            "Classification output must contain "
            "'alpha' or 'evidence'."
        )

    return evidential_classification_loss(
        alpha=alpha,
        target=targets,
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    use_amp: bool,
) -> tuple[float, float]:
    """
    Train for one epoch.

    Returns
    -------
    loss, accuracy
    """

    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch_index, batch in enumerate(
        loader
    ):

        images = batch[
            "image"
        ].to(
            device,
            non_blocking=True,
        )

        targets = batch[
            "label"
        ].to(
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

                classification = (
                    get_classification_output(
                        outputs
                    )
                )

                loss = (
                    compute_classification_loss(
                        classification,
                        targets,
                    )
                )

        else:

            outputs = model(
                images,
                return_features=False,
            )

            classification = (
                get_classification_output(
                    outputs
                )
            )

            loss = (
                compute_classification_loss(
                    classification,
                    targets,
                )
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss "
                f"at batch {batch_index}: "
                f"{loss.item()}"
            )

        if scaler is not None:

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        probabilities = classification[
            "probabilities"
        ]

        predictions = probabilities.argmax(
            dim=-1
        )

        batch_size = (
            targets.shape[0]
        )

        running_loss += (
            loss.detach().item()
            * batch_size
        )

        running_correct += int(
            (
                predictions == targets
            ).sum().item()
        )

        total_samples += batch_size

    epoch_loss = (
        running_loss
        / max(total_samples, 1)
    )

    epoch_accuracy = (
        100.0
        * running_correct
        / max(total_samples, 1)
    )

    return epoch_loss, epoch_accuracy


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """
    Evaluate validation loss and accuracy.
    """

    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch in loader:

        images = batch[
            "image"
        ].to(
            device,
            non_blocking=True,
        )

        targets = batch[
            "label"
        ].to(
            device,
            non_blocking=True,
        )

        if use_amp:

            with autocast():

                outputs = model(
                    images,
                    return_features=False,
                )

                classification = (
                    get_classification_output(
                        outputs
                    )
                )

                loss = (
                    compute_classification_loss(
                        classification,
                        targets,
                    )
                )

        else:

            outputs = model(
                images,
                return_features=False,
            )

            classification = (
                get_classification_output(
                    outputs
                )
            )

            loss = (
                compute_classification_loss(
                    classification,
                    targets,
                )
            )

        probabilities = classification[
            "probabilities"
        ]

        predictions = probabilities.argmax(
            dim=-1
        )

        batch_size = (
            targets.shape[0]
        )

        running_loss += (
            loss.detach().item()
            * batch_size
        )

        running_correct += int(
            (
                predictions == targets
            ).sum().item()
        )

        total_samples += batch_size

    epoch_loss = (
        running_loss
        / max(total_samples, 1)
    )

    epoch_accuracy = (
        100.0
        * running_correct
        / max(total_samples, 1)
    )

    return epoch_loss, epoch_accuracy


# ---------------------------------------------------------------------
# History
# ---------------------------------------------------------------------


def save_history(
    history: list[dict],
    path: Path,
) -> None:
    """
    Save training history as CSV.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not history:
        return

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

        writer.writerows(
            history
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train the evidential brain tumor "
            "MRI classification model."
        )
    )

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help=(
            "Root directory of the "
            "classification dataset."
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
        help=(
            "Validation-based early stopping "
            "patience."
        ),
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
    # Model
    # --------------------------------------------------------------

    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="Number of classification classes.",
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help=(
            "Use ImageNet-pretrained ResNet-50 "
            "weights."
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
        help=(
            "Directory for model checkpoints."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help=(
            "Directory for training outputs."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------

    set_seed(
        args.seed
    )

    # --------------------------------------------------------------
    # Device
    # --------------------------------------------------------------

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
        "\nLoading datasets..."
    )

    train_dataset = (
        BrainTumorClassificationDataset(
            root=args.data_root,
            split="train",
            image_size=224,
            augment=True,
        )
    )

    validation_dataset = (
        BrainTumorClassificationDataset(
            root=args.data_root,
            split="val",
            image_size=224,
            augment=False,
        )
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model = (
        UncertaintyAwareMultiTaskModel(
            num_classes=args.num_classes,
            segmentation_classes=3,
            input_channels=3,
            classification_channels=3,
            pretrained=args.pretrained,
        )
    )

    model.to(device)

    print(
        "\nModel initialized."
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
    # Output paths
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
        / "classification_best.pt"
    )

    last_checkpoint = (
        checkpoint_dir
        / "classification_last.pt"
    )

    history_path = (
        output_dir
        / "classification_training_history.csv"
    )

    # --------------------------------------------------------------
    # Training state
    # --------------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    best_val_accuracy = 0.0

    epochs_without_improvement = 0

    history = []

    # --------------------------------------------------------------
    # Training loop
    # --------------------------------------------------------------

    print(
        "\nStarting training..."
    )

    print(
        "=" * 78
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
            )
        )

        val_loss, val_accuracy = (
            validate(
                model=model,
                loader=validation_loader,
                device=device,
                use_amp=use_amp,
            )
        )

        scheduler.step()

        current_lr = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        # ----------------------------------------------------------
        # Improvement
        # ----------------------------------------------------------

        improved = (
            val_loss
            < best_val_loss
        )

        if improved:

            best_val_loss = (
                val_loss
            )

            best_val_accuracy = (
                val_accuracy
            )

            epochs_without_improvement = 0

            save_checkpoint(
                path=best_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                best_val_loss=best_val_loss,
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------------
        # Save latest checkpoint
        # ----------------------------------------------------------

        save_checkpoint(
            path=last_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            best_val_loss=best_val_loss,
        )

        # ----------------------------------------------------------
        # History
        # ----------------------------------------------------------

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": current_lr,
            }
        )

        save_history(
            history,
            history_path,
        )

        # ----------------------------------------------------------
        # Progress
        # ----------------------------------------------------------

        marker = (
            " *"
            if improved
            else ""
        )

        print(
            f"Epoch "
            f"{epoch:03d}/{args.epochs:03d} | "
            f"Train Loss: "
            f"{train_loss:.5f} | "
            f"Train Acc: "
            f"{train_accuracy:.2f}% | "
            f"Val Loss: "
            f"{val_loss:.5f} | "
            f"Val Acc: "
            f"{val_accuracy:.2f}% | "
            f"LR: "
            f"{current_lr:.7f}"
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
        "\n" + "=" * 78
    )

    print(
        "Training completed."
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.5f}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.2f}%"
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