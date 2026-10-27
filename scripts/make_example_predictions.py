"""
Generate example classification and segmentation predictions
for qualitative visualization.

The script:
    1. Loads a trained multi-task checkpoint.
    2. Selects example samples from the classification and/or
       BraTS segmentation dataset.
    3. Runs evidential inference.
    4. Saves predictions, uncertainty maps, reliability maps,
       and visualization figures.

Output structure
----------------
outputs/example_predictions/
    classification/
        sample_000/
            input.png
            prediction.json
        ...
    segmentation/
        sample_000/
            input.png
            ground_truth.png
            prediction.png
            uncertainty.png
            reliability.png
        ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.brats_dataset import BraTSDataset
from data.classification_dataset import (
    BrainTumorClassificationDataset,
)

from models.multitask_model import (
    UncertaintyAwareMultiTaskModel,
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    """Load model weights from a checkpoint."""

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
            state_dict = checkpoint
    else:
        raise ValueError(
            "Unsupported checkpoint format."
        )

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
            "Warning: missing parameters:"
        )

        for key in missing:
            print(f"  {key}")

    if unexpected:
        print(
            "Warning: unexpected parameters:"
        )

        for key in unexpected:
            print(f"  {key}")


def get_device(
    requested: str,
) -> torch.device:
    """Return the requested evaluation device."""

    if requested == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def tensor_to_numpy(
    tensor: torch.Tensor,
) -> np.ndarray:
    """Convert tensor to NumPy."""

    return (
        tensor.detach()
        .cpu()
        .numpy()
    )


def normalize_image(
    image: np.ndarray,
) -> np.ndarray:
    """
    Normalize an image to [0, 1] for visualization.
    """

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum - minimum < 1e-8:
        return np.zeros_like(image)

    return (
        (image - minimum)
        / (maximum - minimum)
    )


def save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Save a JSON dictionary."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )


# ---------------------------------------------------------------------
# Model-output extraction
# ---------------------------------------------------------------------


def extract_classification_output(
    outputs: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Extract classification evidential outputs."""

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
        "Classification output not found."
    )


def extract_segmentation_output(
    outputs: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Extract segmentation evidential outputs."""

    if "segmentation" in outputs:

        segmentation = outputs[
            "segmentation"
        ]

        if isinstance(
            segmentation,
            dict,
        ):
            return segmentation

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
        "Segmentation output not found."
    )


# ---------------------------------------------------------------------
# Classification examples
# ---------------------------------------------------------------------


CLASS_NAMES = [
    "glioma",
    "meningioma",
    "pituitary",
    "no-tumor",
]


@torch.no_grad()
def make_classification_examples(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    output_dir: Path,
    device: torch.device,
    num_examples: int,
) -> None:
    """
    Generate classification examples.

    Each example contains:
        - input image
        - predicted class
        - ground-truth class
        - class probabilities
        - evidential uncertainty
        - reliability
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.eval()

    number_available = len(dataset)

    number_examples = min(
        num_examples,
        number_available,
    )

    print(
        f"Generating {number_examples} "
        "classification examples..."
    )

    for index in range(
        number_examples
    ):

        sample = dataset[index]

        image = sample["image"]
        target = sample["label"]

        image_batch = image.unsqueeze(
            0
        ).to(device)

        outputs = model(
            image_batch,
            return_features=False,
        )

        classification = (
            extract_classification_output(
                outputs
            )
        )

        probabilities = (
            classification["probabilities"]
        )

        if "uncertainty" in classification:
            uncertainty = (
                classification["uncertainty"]
            )
        else:
            alpha = classification["alpha"]

            strength = alpha.sum(
                dim=-1
            )

            uncertainty = (
                alpha.shape[-1]
                / strength
            )

        if "reliability" in classification:
            reliability = (
                classification["reliability"]
            )
        else:
            reliability = (
                1.0 - uncertainty
            )

        prediction = (
            probabilities.argmax(
                dim=-1
            )
        )

        probabilities_np = (
            tensor_to_numpy(
                probabilities[0]
            )
        )

        predicted_class = int(
            prediction.item()
        )

        target_class = int(
            target.item()
            if torch.is_tensor(target)
            else target
        )

        uncertainty_value = float(
            uncertainty.item()
        )

        reliability_value = float(
            reliability.item()
        )

        sample_dir = (
            output_dir
            / f"sample_{index:03d}"
        )

        sample_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------------
        # Input image
        # ----------------------------------------------------------

        image_np = tensor_to_numpy(
            image
        )

        if image_np.ndim == 3:

            # CHW -> HWC
            image_display = (
                np.transpose(
                    image_np,
                    (1, 2, 0),
                )
            )

            if image_display.shape[-1] == 1:
                image_display = (
                    image_display[..., 0]
                )

        else:
            image_display = image_np

        image_display = normalize_image(
            image_display
        )

        plt.figure(
            figsize=(6, 6)
        )

        if image_display.ndim == 2:
            plt.imshow(
                image_display,
                cmap="gray",
            )
        else:
            plt.imshow(
                image_display
            )

        plt.axis("off")

        plt.title(
            f"GT: {CLASS_NAMES[target_class]}\n"
            f"Pred: {CLASS_NAMES[predicted_class]} | "
            f"R={reliability_value:.3f}"
        )

        plt.tight_layout()

        plt.savefig(
            sample_dir / "input.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        # ----------------------------------------------------------
        # Prediction information
        # ----------------------------------------------------------

        prediction_data = {
            "sample_index": index,
            "ground_truth": (
                CLASS_NAMES[target_class]
                if target_class
                < len(CLASS_NAMES)
                else target_class
            ),
            "prediction": (
                CLASS_NAMES[predicted_class]
                if predicted_class
                < len(CLASS_NAMES)
                else predicted_class
            ),
            "correct": (
                predicted_class
                == target_class
            ),
            "uncertainty": uncertainty_value,
            "reliability": reliability_value,
            "probabilities": {
                CLASS_NAMES[class_index]
                if class_index
                < len(CLASS_NAMES)
                else str(class_index):
                    float(
                        probability
                    )
                for class_index, probability
                in enumerate(
                    probabilities_np
                )
            },
        }

        save_json(
            prediction_data,
            sample_dir
            / "prediction.json",
        )


# ---------------------------------------------------------------------
# Segmentation examples
# ---------------------------------------------------------------------


SEGMENTATION_REGIONS = [
    "WT",
    "TC",
    "ET",
]


@torch.no_grad()
def make_segmentation_examples(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    output_dir: Path,
    device: torch.device,
    num_examples: int,
) -> None:
    """
    Generate qualitative segmentation examples.

    The BraTS segmentation target is represented by three
    region channels:

        WT - Whole Tumor
        TC - Tumor Core
        ET - Enhancing Tumor
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.eval()

    number_available = len(dataset)

    number_examples = min(
        num_examples,
        number_available,
    )

    print(
        f"Generating {number_examples} "
        "segmentation examples..."
    )

    for index in range(
        number_examples
    ):

        sample = dataset[index]

        image = sample["image"]
        target = sample["mask"]

        image_batch = image.unsqueeze(
            0
        ).to(device)

        outputs = model(
            image_batch,
            return_features=False,
        )

        segmentation = (
            extract_segmentation_output(
                outputs
            )
        )

        probabilities = (
            segmentation["probabilities"]
        )

        prediction = (
            probabilities >= 0.5
        ).float()

        if "uncertainty" in segmentation:
            uncertainty = (
                segmentation["uncertainty"]
            )
        else:
            alpha = segmentation["alpha"]

            strength = alpha.sum(
                dim=1,
                keepdim=True,
            )

            uncertainty = (
                alpha.shape[1]
                / strength
            )

        reliability = (
            1.0 - uncertainty
        )

        prediction_np = (
            tensor_to_numpy(
                prediction[0]
            )
        )

        target_np = (
            tensor_to_numpy(
                target
            )
        )

        uncertainty_np = (
            tensor_to_numpy(
                uncertainty[0]
            )
        )

        reliability_np = (
            tensor_to_numpy(
                reliability[0]
            )
        )

        image_np = tensor_to_numpy(
            image
        )

        # ----------------------------------------------------------
        # Select a representative image channel.
        #
        # For multi-modal BraTS input, use the first channel for
        # visualization while retaining all modalities for inference.
        # ----------------------------------------------------------

        if image_np.ndim == 3:
            image_display = image_np[0]
        else:
            image_display = image_np

        image_display = normalize_image(
            image_display
        )

        sample_dir = (
            output_dir
            / f"sample_{index:03d}"
        )

        sample_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------------
        # Save raw arrays
        # ----------------------------------------------------------

        np.save(
            sample_dir
            / "prediction.npy",
            prediction_np,
        )

        np.save(
            sample_dir
            / "ground_truth.npy",
            target_np,
        )

        np.save(
            sample_dir
            / "uncertainty.npy",
            uncertainty_np,
        )

        np.save(
            sample_dir
            / "reliability.npy",
            reliability_np,
        )

        # ----------------------------------------------------------
        # Input image
        # ----------------------------------------------------------

        plt.figure(
            figsize=(6, 6)
        )

        plt.imshow(
            image_display,
            cmap="gray",
        )

        plt.axis("off")
        plt.title("MRI Input")

        plt.tight_layout()

        plt.savefig(
            sample_dir / "input.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        # ----------------------------------------------------------
        # Ground-truth regions
        # ----------------------------------------------------------

        for channel, region in enumerate(
            SEGMENTATION_REGIONS
        ):

            if channel >= target_np.shape[0]:
                continue

            plt.figure(
                figsize=(6, 6)
            )

            plt.imshow(
                image_display,
                cmap="gray",
            )

            plt.imshow(
                target_np[channel],
                alpha=0.45,
                cmap="jet",
            )

            plt.axis("off")

            plt.title(
                f"Ground Truth - {region}"
            )

            plt.tight_layout()

            plt.savefig(
                sample_dir
                / f"ground_truth_{region}.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close()

        # ----------------------------------------------------------
        # Predicted regions
        # ----------------------------------------------------------

        for channel, region in enumerate(
            SEGMENTATION_REGIONS
        ):

            if channel >= prediction_np.shape[0]:
                continue

            plt.figure(
                figsize=(6, 6)
            )

            plt.imshow(
                image_display,
                cmap="gray",
            )

            plt.imshow(
                prediction_np[channel],
                alpha=0.45,
                cmap="jet",
            )

            plt.axis("off")

            plt.title(
                f"Prediction - {region}"
            )

            plt.tight_layout()

            plt.savefig(
                sample_dir
                / f"prediction_{region}.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close()

        # ----------------------------------------------------------
        # Uncertainty map
        # ----------------------------------------------------------

        uncertainty_display = (
            np.squeeze(
                uncertainty_np
            )
        )

        plt.figure(
            figsize=(6, 6)
        )

        plt.imshow(
            image_display,
            cmap="gray",
        )

        plt.imshow(
            uncertainty_display,
            alpha=0.55,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )

        plt.axis("off")
        plt.title(
            "Pixel-wise Predictive Uncertainty"
        )

        plt.tight_layout()

        plt.savefig(
            sample_dir
            / "uncertainty.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        # ----------------------------------------------------------
        # Reliability map
        # ----------------------------------------------------------

        reliability_display = (
            np.squeeze(
                reliability_np
            )
        )

        plt.figure(
            figsize=(6, 6)
        )

        plt.imshow(
            image_display,
            cmap="gray",
        )

        plt.imshow(
            reliability_display,
            alpha=0.55,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )

        plt.axis("off")
        plt.title(
            "Pixel-wise Reliability"
        )

        plt.tight_layout()

        plt.savefig(
            sample_dir
            / "reliability.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        # ----------------------------------------------------------
        # Metadata
        # ----------------------------------------------------------

        metadata = {
            "sample_index": index,
            "regions": SEGMENTATION_REGIONS,
            "mean_uncertainty": float(
                np.mean(
                    uncertainty_np
                )
            ),
            "mean_reliability": float(
                np.mean(
                    reliability_np
                )
            ),
        }

        save_json(
            metadata,
            sample_dir
            / "prediction.json",
        )


# ---------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------


def build_classification_dataset(
    root: str | Path,
):
    """
    Build the classification test dataset.
    """

    return BrainTumorClassificationDataset(
        root=root,
        split="test",
        image_size=224,
        augment=False,
    )


def build_segmentation_dataset(
    root: str | Path,
):
    """
    Build the BraTS test dataset.
    """

    return BraTSDataset(
        root=root,
        split="test",
        augment=False,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate example predictions and "
            "uncertainty visualizations."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint.",
    )

    parser.add_argument(
        "--classification-root",
        type=str,
        default=None,
        help=(
            "Classification dataset root."
        ),
    )

    parser.add_argument(
        "--segmentation-root",
        type=str,
        default=None,
        help=(
            "BraTS dataset root."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "outputs/example_predictions"
        ),
        help=(
            "Output directory."
        ),
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help=(
            "Number of examples to generate "
            "for each task."
        ),
    )

    parser.add_argument(
        "--classification",
        action="store_true",
        help=(
            "Generate classification examples."
        ),
    )

    parser.add_argument(
        "--segmentation",
        action="store_true",
        help=(
            "Generate segmentation examples."
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
        help="Inference device.",
    )

    args = parser.parse_args()

    if not args.classification and not args.segmentation:
        raise ValueError(
            "Specify at least one task: "
            "--classification or --segmentation."
        )

    device = get_device(
        args.device
    )

    print(
        f"Using device: {device}"
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

    load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    model.to(device)
    model.eval()

    print(
        f"Loaded checkpoint: "
        f"{args.checkpoint}"
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Classification
    # --------------------------------------------------------------

    if args.classification:

        if args.classification_root is None:
            raise ValueError(
                "--classification-root is required "
                "for classification examples."
            )

        classification_dataset = (
            build_classification_dataset(
                args.classification_root
            )
        )

        make_classification_examples(
            model=model,
            dataset=classification_dataset,
            output_dir=(
                output_dir
                / "classification"
            ),
            device=device,
            num_examples=args.num_examples,
        )

    # --------------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------------

    if args.segmentation:

        if args.segmentation_root is None:
            raise ValueError(
                "--segmentation-root is required "
                "for segmentation examples."
            )

        segmentation_dataset = (
            build_segmentation_dataset(
                args.segmentation_root
            )
        )

        make_segmentation_examples(
            model=model,
            dataset=segmentation_dataset,
            output_dir=(
                output_dir
                / "segmentation"
            ),
            device=device,
            num_examples=args.num_examples,
        )

    print(
        "\nExample prediction generation completed."
    )

    print(
        f"Results saved to: "
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()