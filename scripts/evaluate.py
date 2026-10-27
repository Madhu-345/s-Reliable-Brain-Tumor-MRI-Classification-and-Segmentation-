"""
Evaluation script for the uncertainty-aware multi-task
brain tumor MRI classification and segmentation framework.

Supported evaluation outputs
----------------------------
Classification:
    - Accuracy
    - Precision
    - Recall
    - Specificity
    - F1-score
    - AUROC

Segmentation:
    - Dice
    - IoU
    - Precision
    - Recall
    - HD95

Uncertainty:
    - Predictive uncertainty
    - Expected calibration error
    - Negative log-likelihood

Reliability:
    - Reliability score
    - Coverage
    - Selective accuracy
    - Selective risk

The script is designed for the unified
Uncertainty-Aware Multi-Task Evidential Learning model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs import PROJECT_ROOT
from data.brats_dataset import BraTSDataset
from data.classification_dataset import BrainTumorClassificationDataset

from models.multitask_model import (
    UncertaintyAwareMultiTaskModel,
)

from metrics.classification import classification_metrics
from metrics.segmentation import segmentation_metrics
from metrics.uncertainty import (
    expected_calibration_error,
    negative_log_likelihood,
)
from metrics.reliability import (
    classification_reliability,
    segmentation_reliability,
    coverage,
    selective_accuracy,
    selective_risk,
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """
    Load a trained model checkpoint.

    Supports checkpoints containing either:
        - state_dict
        - model_state_dict

    Returns
    -------
    dict
        Full checkpoint dictionary when available.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            # Allow a raw state dictionary.
            state_dict = checkpoint

    else:
        raise ValueError(
            "Unsupported checkpoint format. "
            "Expected a state dictionary or checkpoint dictionary."
        )

    # Remove a possible DistributedDataParallel prefix.
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    if missing:
        print(
            "\nWarning: missing checkpoint parameters:"
        )
        for key in missing:
            print(f"  - {key}")

    if unexpected:
        print(
            "\nWarning: unexpected checkpoint parameters:"
        )
        for key in unexpected:
            print(f"  - {key}")

    return checkpoint if isinstance(checkpoint, dict) else {}


def move_to_device(
    value: Any,
    device: torch.device,
) -> Any:
    """
    Recursively move tensors to the selected device.
    """

    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)

    if isinstance(value, dict):
        return {
            key: move_to_device(item, device)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            move_to_device(item, device)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            move_to_device(item, device)
            for item in value
        )

    return value


def detach_to_cpu(
    value: torch.Tensor,
) -> torch.Tensor:
    """
    Detach a tensor and move it to CPU.
    """

    return value.detach().cpu()


def extract_classification_output(
    outputs: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """
    Extract classification evidential outputs.

    The model is expected to return a dictionary containing
    either:

        outputs["classification"]

    or directly:

        outputs["classification_logits"]
        outputs["classification_alpha"]
        outputs["classification_probability"]

    Returns
    -------
    dict
        alpha, probability, uncertainty, reliability.
    """

    if "classification" in outputs:
        cls = outputs["classification"]

        if isinstance(cls, dict):
            return cls

    required = [
        "classification_alpha",
        "classification_probability",
    ]

    missing = [
        key for key in required
        if key not in outputs
    ]

    if missing:
        raise KeyError(
            "Classification output is missing required keys: "
            + ", ".join(missing)
        )

    return {
        key.replace("classification_", ""): value
        for key, value in outputs.items()
        if key.startswith("classification_")
    }


def extract_segmentation_output(
    outputs: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """
    Extract segmentation evidential outputs.
    """

    if "segmentation" in outputs:
        seg = outputs["segmentation"]

        if isinstance(seg, dict):
            return seg

    required = [
        "segmentation_alpha",
        "segmentation_probability",
    ]

    missing = [
        key for key in required
        if key not in outputs
    ]

    if missing:
        raise KeyError(
            "Segmentation output is missing required keys: "
            + ", ".join(missing)
        )

    return {
        key.replace("segmentation_", ""): value
        for key, value in outputs.items()
        if key.startswith("segmentation_")
    }


# ---------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------


@torch.no_grad()
def evaluate_classification(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """
    Evaluate the classification branch.
    """

    model.eval()

    all_targets = []
    all_predictions = []
    all_probabilities = []
    all_uncertainties = []
    all_reliabilities = []

    for batch in loader:
        batch = move_to_device(
            batch,
            device,
        )

        images = batch["image"]
        targets = batch["label"]

        outputs = model(
            images,
            return_features=False,
        )

        cls = extract_classification_output(
            outputs
        )

        probabilities = cls["probabilities"]

        if "uncertainty" in cls:
            uncertainty = cls["uncertainty"]
        else:
            strength = cls["alpha"].sum(
                dim=-1
            )
            uncertainty = (
                probabilities.shape[-1]
                / strength
            )

        if "reliability" in cls:
            reliability = cls["reliability"]
        else:
            reliability = 1.0 - uncertainty

        predictions = probabilities.argmax(
            dim=-1
        )

        all_targets.append(
            detach_to_cpu(targets).view(-1)
        )

        all_predictions.append(
            detach_to_cpu(predictions).view(-1)
        )

        all_probabilities.append(
            detach_to_cpu(probabilities)
        )

        all_uncertainties.append(
            detach_to_cpu(uncertainty).view(-1)
        )

        all_reliabilities.append(
            detach_to_cpu(reliability).view(-1)
        )

    targets = torch.cat(
        all_targets
    )

    predictions = torch.cat(
        all_predictions
    )

    probabilities = torch.cat(
        all_probabilities
    )

    uncertainties = torch.cat(
        all_uncertainties
    )

    reliabilities = torch.cat(
        all_reliabilities
    )

    metrics = classification_metrics(
        targets,
        predictions,
        probabilities,
    )

    # Calibration and NLL.
    ece = expected_calibration_error(
        probabilities,
        targets,
    )

    nll = negative_log_likelihood(
        probabilities,
        targets,
    )

    correct = (
        predictions == targets
    ).numpy()

    reliability_np = (
        reliabilities.numpy()
    )

    uncertainty_np = (
        uncertainties.numpy()
    )

    return {
        "metrics": metrics,
        "ece": float(ece),
        "nll": float(nll),
        "targets": targets.numpy(),
        "predictions": predictions.numpy(),
        "probabilities": probabilities.numpy(),
        "uncertainty": uncertainty_np,
        "reliability": reliability_np,
        "correct": correct,
    }


# ---------------------------------------------------------------------
# Segmentation evaluation
# ---------------------------------------------------------------------


@torch.no_grad()
def evaluate_segmentation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """
    Evaluate the segmentation branch.

    The BraTS dataset loader represents the tumor regions as
    three channels:

        WT - Whole Tumor
        TC - Tumor Core
        ET - Enhancing Tumor

    Therefore, evaluation is performed channel-wise rather than
    using argmax over the three channels.
    """

    model.eval()

    all_predictions = []
    all_targets = []
    all_uncertainties = []
    all_reliabilities = []

    for batch in loader:
        batch = move_to_device(
            batch,
            device,
        )

        images = batch["image"]
        masks = batch["mask"]

        outputs = model(
            images,
            return_features=False,
        )

        seg = extract_segmentation_output(
            outputs
        )

        probabilities = seg["probabilities"]

        if "uncertainty" in seg:
            uncertainty = seg["uncertainty"]
        else:
            alpha = seg["alpha"]

            strength = alpha.sum(
                dim=1,
                keepdim=True,
            )

            uncertainty = (
                alpha.shape[1]
                / strength
            )

        predictions = (
            probabilities >= 0.5
        ).float()

        if "reliability" in seg:
            reliability = seg["reliability"]
        else:
            reliability = 1.0 - uncertainty

        all_predictions.append(
            detach_to_cpu(predictions)
        )

        all_targets.append(
            detach_to_cpu(masks)
        )

        all_uncertainties.append(
            detach_to_cpu(uncertainty)
        )

        all_reliabilities.append(
            detach_to_cpu(reliability)
        )

    predictions = torch.cat(
        all_predictions,
        dim=0,
    )

    targets = torch.cat(
        all_targets,
        dim=0,
    )

    uncertainties = torch.cat(
        all_uncertainties,
        dim=0,
    )

    reliabilities = torch.cat(
        all_reliabilities,
        dim=0,
    )

    # BraTS region names.
    region_names = [
        "WT",
        "TC",
        "ET",
    ]

    region_results = {}

    for index, region_name in enumerate(
        region_names
    ):
        pred_region = predictions[
            :, index
        ]

        target_region = targets[
            :, index
        ]

        result = segmentation_metrics(
            pred_region,
            target_region,
            num_classes=2,
        )

        region_results[region_name] = result

    return {
        "regions": region_results,
        "predictions": predictions.numpy(),
        "targets": targets.numpy(),
        "uncertainty": uncertainties.numpy(),
        "reliability": reliabilities.numpy(),
    }


# ---------------------------------------------------------------------
# Selective prediction
# ---------------------------------------------------------------------


def compute_classification_selective_results(
    evaluation: dict[str, Any],
    thresholds: list[float],
) -> list[dict[str, float]]:
    """
    Compute selective classification results.
    """

    correct = evaluation["correct"]
    reliability = evaluation["reliability"]

    results = []

    for threshold in thresholds:
        accepted = (
            reliability >= threshold
        )

        cov = coverage(
            reliability,
            threshold,
        )

        if accepted.sum() == 0:
            selected_accuracy = float("nan")
            selected_risk = float("nan")
        else:
            selected_accuracy = (
                selective_accuracy(
                    correct,
                    reliability,
                    threshold,
                )
            )

            selected_risk = (
                selective_risk(
                    correct,
                    reliability,
                    threshold,
                )
            )

        results.append(
            {
                "threshold": float(threshold),
                "coverage": float(cov),
                "selective_accuracy": float(
                    selected_accuracy
                ),
                "selective_risk": float(
                    selected_risk
                ),
            }
        )

    return results


def compute_segmentation_reliability_summary(
    evaluation: dict[str, Any],
) -> dict[str, float]:
    """
    Compute the mean segmentation uncertainty and reliability.
    """

    uncertainty = evaluation[
        "uncertainty"
    ]

    reliability = evaluation[
        "reliability"
    ]

    return {
        "mean_uncertainty": float(
            np.mean(uncertainty)
        ),
        "mean_reliability": float(
            np.mean(reliability)
        ),
    }


# ---------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------


def convert_numpy(
    value: Any,
) -> Any:
    """
    Recursively convert NumPy values to JSON-compatible types.
    """

    if isinstance(value, dict):
        return {
            key: convert_numpy(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            convert_numpy(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            convert_numpy(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def save_results(
    results: dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    Save evaluation results as JSON.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable = convert_numpy(
        results
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable,
            file,
            indent=4,
        )


# ---------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------


def build_classification_loader(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """
    Build the classification evaluation loader.

    The dataset implementation is expected to provide
    a test/evaluation split.
    """

    dataset = BrainTumorClassificationDataset(
        root=data_root,
        split="test",
        image_size=224,
        augment=False,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_segmentation_loader(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """
    Build the BraTS evaluation loader.
    """

    dataset = BraTSDataset(
        root=data_root,
        split="test",
        augment=False,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the uncertainty-aware multi-task "
            "brain tumor MRI model."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint.",
    )

    parser.add_argument(
        "--classification-root",
        type=str,
        default=None,
        help="Root directory of the classification dataset.",
    )

    parser.add_argument(
        "--segmentation-root",
        type=str,
        default=None,
        help="Root directory of the BraTS dataset.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation_results.json",
        help="Output JSON file.",
    )

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
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--classification",
        action="store_true",
        help="Evaluate classification branch.",
    )

    parser.add_argument(
        "--segmentation",
        action="store_true",
        help="Evaluate segmentation branch.",
    )

    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.50,0.60,0.70,0.80,0.90,0.95",
        help=(
            "Reliability thresholds used for selective "
            "prediction."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Evaluation device.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Device
    # --------------------------------------------------------------

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        device = torch.device("cuda")

    elif args.device == "cpu":
        device = torch.device("cpu")

    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print(
        f"Evaluation device: {device}"
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model = UncertaintyAwareMultiTaskModel(
        num_classes=4,
        segmentation_classes=3,
        input_channels=4,
        classification_channels=3,
        pretrained=False,
    )

    checkpoint = load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    model.to(device)
    model.eval()

    print(
        f"Loaded checkpoint: {args.checkpoint}"
    )

    results = {
        "checkpoint": str(
            Path(args.checkpoint)
        ),
        "device": str(device),
    }

    # --------------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------------

    thresholds = [
        float(value.strip())
        for value in args.thresholds.split(",")
        if value.strip()
    ]

    # --------------------------------------------------------------
    # Classification
    # --------------------------------------------------------------

    if args.classification:

        if args.classification_root is None:
            raise ValueError(
                "--classification-root is required "
                "when --classification is specified."
            )

        print(
            "\nEvaluating classification..."
        )

        classification_loader = (
            build_classification_loader(
                data_root=args.classification_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        )

        classification_results = (
            evaluate_classification(
                model=model,
                loader=classification_loader,
                device=device,
            )
        )

        selective_results = (
            compute_classification_selective_results(
                evaluation=classification_results,
                thresholds=thresholds,
            )
        )

        results["classification"] = {
            "metrics": classification_results[
                "metrics"
            ],
            "ece": classification_results[
                "ece"
            ],
            "nll": classification_results[
                "nll"
            ],
            "mean_uncertainty": float(
                np.mean(
                    classification_results[
                        "uncertainty"
                    ]
                )
            ),
            "mean_reliability": float(
                np.mean(
                    classification_results[
                        "reliability"
                    ]
                )
            ),
            "selective_prediction": (
                selective_results
            ),
        }

        print(
            "\nClassification metrics:"
        )

        for key, value in (
            classification_results[
                "metrics"
            ].items()
        ):
            print(
                f"  {key}: {value:.4f}"
            )

        print(
            f"  ECE: "
            f"{classification_results['ece']:.4f}"
        )

        print(
            f"  NLL: "
            f"{classification_results['nll']:.4f}"
        )

    # --------------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------------

    if args.segmentation:

        if args.segmentation_root is None:
            raise ValueError(
                "--segmentation-root is required "
                "when --segmentation is specified."
            )

        print(
            "\nEvaluating segmentation..."
        )

        segmentation_loader = (
            build_segmentation_loader(
                data_root=args.segmentation_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        )

        segmentation_results = (
            evaluate_segmentation(
                model=model,
                loader=segmentation_loader,
                device=device,
            )
        )

        reliability_summary = (
            compute_segmentation_reliability_summary(
                segmentation_results
            )
        )

        results["segmentation"] = {
            "regions": segmentation_results[
                "regions"
            ],
            "reliability": reliability_summary,
        }

        print(
            "\nSegmentation metrics:"
        )

        for region_name, region_metrics in (
            segmentation_results[
                "regions"
            ].items()
        ):
            print(
                f"\n  {region_name}:"
            )

            for key, value in (
                region_metrics.items()
            ):
                print(
                    f"    {key}: {value:.4f}"
                )

        print(
            "\nSegmentation reliability:"
        )

        print(
            "  Mean uncertainty: "
            f"{reliability_summary['mean_uncertainty']:.4f}"
        )

        print(
            "  Mean reliability: "
            f"{reliability_summary['mean_reliability']:.4f}"
        )

    # --------------------------------------------------------------
    # Check that at least one task was selected
    # --------------------------------------------------------------

    if not args.classification and not args.segmentation:
        raise ValueError(
            "Specify at least one evaluation task: "
            "--classification or --segmentation."
        )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    save_results(
        results=results,
        output_path=args.output,
    )

    print(
        f"\nEvaluation results saved to: "
        f"{Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()