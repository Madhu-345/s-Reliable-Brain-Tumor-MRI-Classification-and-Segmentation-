"""
Selective evaluation for the uncertainty-aware brain tumor MRI
classification framework.

The script evaluates reliability-based selective prediction using
the definitions reported in the manuscript:

    Coverage(tau)
        = N_accepted / N_total * 100

    Selective Accuracy(tau)
        = N_correct,accepted / N_accepted * 100

    Selective Risk(tau)
        = N_incorrect,accepted / N_accepted * 100

Accepted predictions satisfy:

    Reliability >= tau

The script can either:

1. Evaluate a CSV file containing prediction correctness and
   reliability, or

2. Run a trained model on the classification test set and compute
   the selective results directly.

Expected CSV columns
--------------------
correct
    Whether the prediction is correct (1/0 or True/False).

reliability
    Evidential reliability score in [0, 1].

Default thresholds
------------------
0.50, 0.60, 0.70, 0.80, 0.90, 0.95
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.classification_dataset import (
    BrainTumorClassificationDataset,
)

from models.multitask_model import (
    UncertaintyAwareMultiTaskModel,
)


DEFAULT_THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
]


# ---------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    """
    Load a trained model checkpoint.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ValueError(
            "Unsupported checkpoint format."
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint[
            "state_dict"
        ]

    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module.") :]

        cleaned_state_dict[key] = value

    missing, unexpected = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )
    )

    if missing:
        print(
            "\nWarning: missing checkpoint "
            "parameters:"
        )

        for key in missing:
            print(f"  - {key}")

    if unexpected:
        print(
            "\nWarning: unexpected checkpoint "
            "parameters:"
        )

        for key in unexpected:
            print(f"  - {key}")


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------


def get_device(
    requested: str,
) -> torch.device:
    """
    Select evaluation device.
    """

    if requested == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is "
                "not available."
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
# Selective metrics
# ---------------------------------------------------------------------


def compute_selective_metrics(
    correct: np.ndarray,
    reliability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """
    Compute coverage, selective accuracy,
    and selective risk.

    Parameters
    ----------
    correct:
        Boolean array indicating whether each
        prediction is correct.

    reliability:
        Reliability score for each prediction.

    threshold:
        Reliability acceptance threshold.

    Returns
    -------
    dict
        Selective prediction metrics.
    """

    correct = np.asarray(
        correct,
        dtype=bool,
    )

    reliability = np.asarray(
        reliability,
        dtype=float,
    )

    if correct.ndim != 1:
        raise ValueError(
            "'correct' must be one-dimensional."
        )

    if reliability.ndim != 1:
        raise ValueError(
            "'reliability' must be one-dimensional."
        )

    if len(correct) != len(reliability):
        raise ValueError(
            "'correct' and 'reliability' "
            "must have the same length."
        )

    if len(correct) == 0:
        raise ValueError(
            "No samples were provided."
        )

    if not np.all(
        np.isfinite(reliability)
    ):
        raise ValueError(
            "Reliability contains NaN "
            "or infinite values."
        )

    if np.any(
        (reliability < 0.0)
        | (reliability > 1.0)
    ):
        raise ValueError(
            "Reliability values must lie "
            "within [0, 1]."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must lie within [0, 1]."
        )

    accepted = (
        reliability >= threshold
    )

    n_total = len(correct)
    n_accepted = int(
        accepted.sum()
    )

    coverage_value = (
        100.0
        * n_accepted
        / n_total
    )

    if n_accepted == 0:

        selective_accuracy = float(
            "nan"
        )

        selective_risk = float(
            "nan"
        )

    else:

        correct_accepted = (
            correct & accepted
        )

        n_correct_accepted = int(
            correct_accepted.sum()
        )

        selective_accuracy = (
            100.0
            * n_correct_accepted
            / n_accepted
        )

        selective_risk = (
            100.0
            - selective_accuracy
        )

    return {
        "threshold": float(
            threshold
        ),
        "coverage": float(
            coverage_value
        ),
        "selective_accuracy": float(
            selective_accuracy
        ),
        "selective_risk": float(
            selective_risk
        ),
    }


def compute_selective_table(
    correct: np.ndarray,
    reliability: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    """
    Compute selective prediction results
    for all thresholds.
    """

    rows = []

    for threshold in thresholds:

        row = compute_selective_metrics(
            correct=correct,
            reliability=reliability,
            threshold=threshold,
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# CSV input
# ---------------------------------------------------------------------


def load_prediction_csv(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load correctness and reliability from CSV.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction CSV not found: "
            f"{path}"
        )

    dataframe = pd.read_csv(
        path
    )

    required = {
        "correct",
        "reliability",
    }

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    correct_values = (
        dataframe["correct"]
    )

    if correct_values.dtype.kind in {
        "O",
        "U",
        "S",
    }:

        normalized = (
            correct_values
            .astype(str)
            .str.strip()
            .str.lower()
        )

        mapping = {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        }

        correct_values = normalized.map(
            mapping
        )

        if correct_values.isna().any():
            raise ValueError(
                "Column 'correct' contains "
                "unsupported values."
            )

    correct = (
        np.asarray(
            correct_values
        ).astype(bool)
    )

    reliability = np.asarray(
        dataframe["reliability"],
        dtype=float,
    )

    return correct, reliability


# ---------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------


def extract_classification_output(
    outputs: dict,
) -> dict:
    """
    Extract the classification output dictionary.
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

        extracted = {}

        for key, value in outputs.items():

            if key.startswith(
                "classification_"
            ):

                new_key = key.replace(
                    "classification_",
                    "",
                )

                extracted[
                    new_key
                ] = value

        return extracted

    raise KeyError(
        "Classification output not found "
        "in model output."
    )


@torch.no_grad()
def collect_model_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model inference and collect:

        correct
        reliability
    """

    model.eval()

    all_correct = []
    all_reliability = []

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

        outputs = model(
            images,
            return_features=False,
        )

        classification = (
            extract_classification_output(
                outputs
            )
        )

        probabilities = (
            classification[
                "probabilities"
            ]
        )

        predictions = (
            probabilities.argmax(
                dim=-1
            )
        )

        if "reliability" in classification:

            reliability = (
                classification[
                    "reliability"
                ]
            )

        elif "uncertainty" in classification:

            reliability = (
                1.0
                - classification[
                    "uncertainty"
                ]
            )

        else:

            alpha = classification[
                "alpha"
            ]

            strength = alpha.sum(
                dim=-1
            )

            uncertainty = (
                alpha.shape[-1]
                / strength
            )

            reliability = (
                1.0
                - uncertainty
            )

        correct = (
            predictions
            == targets
        )

        all_correct.append(
            correct.detach()
            .cpu()
            .numpy()
        )

        all_reliability.append(
            reliability.detach()
            .cpu()
            .numpy()
        )

        print(
            f"\rProcessed batch "
            f"{batch_index + 1}",
            end="",
        )

    print()

    correct = np.concatenate(
        all_correct,
        axis=0,
    )

    reliability = np.concatenate(
        all_reliability,
        axis=0,
    )

    return correct, reliability


def build_classification_loader(
    root: str | Path,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """
    Build the classification test loader.
    """

    dataset = (
        BrainTumorClassificationDataset(
            root=root,
            split="test",
            image_size=224,
            augment=False,
        )
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------


def save_table(
    table: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save selective prediction table.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        output_path,
        index=False,
        float_format="%.4f",
    )


def print_table(
    table: pd.DataFrame,
) -> None:
    """
    Print a manuscript-friendly selective
    prediction table.
    """

    display_table = table.copy()

    display_table[
        "threshold"
    ] = display_table[
        "threshold"
    ].map(
        lambda value: f"{value:.2f}"
    )

    display_table[
        "coverage"
    ] = display_table[
        "coverage"
    ].map(
        lambda value: (
            f"{value:.2f}"
            if np.isfinite(value)
            else "N/A"
        )
    )

    display_table[
        "selective_accuracy"
    ] = display_table[
        "selective_accuracy"
    ].map(
        lambda value: (
            f"{value:.2f}"
            if np.isfinite(value)
            else "N/A"
        )
    )

    display_table[
        "selective_risk"
    ] = display_table[
        "selective_risk"
    ].map(
        lambda value: (
            f"{value:.2f}"
            if np.isfinite(value)
            else "N/A"
        )
    )

    print(
        "\nSelective Prediction Results"
    )

    print(
        "=" * 72
    )

    print(
        display_table.to_string(
            index=False
        )
    )

    print(
        "=" * 72
    )


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------


def parse_thresholds(
    value: str,
) -> list[float]:
    """
    Parse comma-separated thresholds.
    """

    thresholds = []

    for item in value.split(","):

        item = item.strip()

        if not item:
            continue

        threshold = float(
            item
        )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Invalid threshold "
                f"{threshold}. "
                "Expected a value in [0, 1]."
            )

        thresholds.append(
            threshold
        )

    if not thresholds:
        raise ValueError(
            "No thresholds were provided."
        )

    return thresholds


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate reliability-based "
            "selective prediction."
        )
    )

    # --------------------------------------------------------------
    # Input mode
    # --------------------------------------------------------------

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "CSV containing 'correct' and "
            "'reliability' columns."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Trained model checkpoint. "
            "Required when --input is not used."
        ),
    )

    parser.add_argument(
        "--classification-root",
        type=str,
        default=None,
        help=(
            "Classification dataset root. "
            "Required when evaluating directly "
            "from a checkpoint."
        ),
    )

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )

    parser.add_argument(
        "--thresholds",
        type=str,
        default=(
            "0.50,0.60,0.70,"
            "0.80,0.90,0.95"
        ),
        help=(
            "Comma-separated reliability "
            "thresholds."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        help="Evaluation device.",
    )

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "outputs/"
            "selective_evaluation.csv"
        ),
        help=(
            "Output CSV path."
        ),
    )

    args = parser.parse_args()

    thresholds = parse_thresholds(
        args.thresholds
    )

    # --------------------------------------------------------------
    # Validate input mode
    # --------------------------------------------------------------

    if (
        args.input is None
        and args.checkpoint is None
    ):

        raise ValueError(
            "Provide either --input or "
            "--checkpoint."
        )

    if (
        args.input is not None
        and args.checkpoint is not None
    ):

        raise ValueError(
            "Use either --input or "
            "--checkpoint, not both."
        )

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------

    if args.input is not None:

        print(
            f"Loading prediction file: "
            f"{args.input}"
        )

        correct, reliability = (
            load_prediction_csv(
                args.input
            )
        )

    else:

        if args.classification_root is None:
            raise ValueError(
                "--classification-root is "
                "required when using "
                "--checkpoint."
            )

        device = get_device(
            args.device
        )

        print(
            f"Evaluation device: {device}"
        )

        model = (
            UncertaintyAwareMultiTaskModel(
                num_classes=4,
                segmentation_classes=3,
                input_channels=4,
                classification_channels=3,
                pretrained=False,
            )
        )

        load_checkpoint(
            model=model,
            checkpoint_path=args.checkpoint,
            device=device,
        )

        model.to(device)

        loader = (
            build_classification_loader(
                root=args.classification_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        )

        print(
            "Running classification inference..."
        )

        correct, reliability = (
            collect_model_predictions(
                model=model,
                loader=loader,
                device=device,
            )
        )

    # --------------------------------------------------------------
    # Compute selective results
    # --------------------------------------------------------------

    print(
        f"\nNumber of evaluated samples: "
        f"{len(correct)}"
    )

    overall_accuracy = (
        100.0
        * np.mean(correct)
    )

    print(
        f"Overall accuracy: "
        f"{overall_accuracy:.2f}%"
    )

    table = compute_selective_table(
        correct=correct,
        reliability=reliability,
        thresholds=thresholds,
    )

    # --------------------------------------------------------------
    # Display
    # --------------------------------------------------------------

    print_table(
        table
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    save_table(
        table=table,
        output_path=args.output,
    )

    print(
        f"\nSaved selective evaluation to:"
        f"\n{Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()