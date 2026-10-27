"""
Segmentation metrics for the uncertainty-aware multi-task
brain tumor MRI classification and segmentation framework.

Supported metrics:
    - Dice score
    - IoU
    - Precision
    - Recall
    - HD95

The implementation supports multi-class segmentation and computes
metrics independently for each class/region before averaging.
"""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(array):
    """Convert a PyTorch tensor or array-like object to NumPy."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _prepare_masks(
    prediction,
    target,
):
    """
    Convert segmentation inputs to NumPy arrays.

    Supported formats:
        [B, H, W]
        [B, K, H, W]

    For one-hot/multi-channel inputs, the class dimension is
    converted to class indices using argmax.
    """
    prediction = _to_numpy(prediction)
    target = _to_numpy(target)

    if prediction.ndim == 4:
        prediction = np.argmax(
            prediction,
            axis=1,
        )

    if target.ndim == 4:
        target = np.argmax(
            target,
            axis=1,
        )

    if prediction.ndim != 3:
        raise ValueError(
            "Prediction must have shape [B,H,W] or [B,K,H,W]."
        )

    if target.ndim != 3:
        raise ValueError(
            "Target must have shape [B,H,W] or [B,K,H,W]."
        )

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes must match."
        )

    return prediction, target


def _binary_dice(
    prediction,
    target,
    smooth: float = 1e-6,
):
    """Compute binary Dice score."""
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()

    denominator = (
        prediction.sum()
        + target.sum()
    )

    if denominator == 0:
        return 1.0

    return float(
        (2.0 * intersection + smooth)
        / (denominator + smooth)
    )


def _binary_iou(
    prediction,
    target,
    smooth: float = 1e-6,
):
    """Compute binary intersection-over-union."""
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()

    union = np.logical_or(
        prediction,
        target,
    ).sum()

    if union == 0:
        return 1.0

    return float(
        (intersection + smooth)
        / (union + smooth)
    )


def _binary_precision(
    prediction,
    target,
):
    """Compute binary precision."""
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    true_positive = np.logical_and(
        prediction,
        target,
    ).sum()

    false_positive = np.logical_and(
        prediction,
        ~target,
    ).sum()

    denominator = (
        true_positive
        + false_positive
    )

    if denominator == 0:
        return 1.0

    return float(
        true_positive / denominator
    )


def _binary_recall(
    prediction,
    target,
):
    """Compute binary recall."""
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    true_positive = np.logical_and(
        prediction,
        target,
    ).sum()

    false_negative = np.logical_and(
        ~prediction,
        target,
    ).sum()

    denominator = (
        true_positive
        + false_negative
    )

    if denominator == 0:
        return 1.0

    return float(
        true_positive / denominator
    )


def dice_score(
    prediction,
    target,
    num_classes: int | None = None,
    include_background: bool = False,
) -> float:
    """
    Compute mean Dice score.

    Parameters
    ----------
    prediction : array-like
        Predicted segmentation mask.

    target : array-like
        Ground-truth segmentation mask.

    num_classes : int, optional
        Number of segmentation classes.

    include_background : bool
        Whether class 0 should be included.

    Returns
    -------
    float
        Mean Dice score.
    """
    prediction, target = _prepare_masks(
        prediction,
        target,
    )

    if num_classes is None:
        num_classes = int(
            max(
                prediction.max(initial=0),
                target.max(initial=0),
            )
            + 1
        )

    start_class = 0 if include_background else 1

    scores = []

    for class_id in range(
        start_class,
        num_classes,
    ):
        pred_class = prediction == class_id
        target_class = target == class_id

        scores.append(
            _binary_dice(
                pred_class,
                target_class,
            )
        )

    if not scores:
        return 0.0

    return float(np.mean(scores))


def iou_score(
    prediction,
    target,
    num_classes: int | None = None,
    include_background: bool = False,
) -> float:
    """
    Compute mean intersection-over-union.

    Parameters
    ----------
    prediction : array-like
        Predicted segmentation mask.

    target : array-like
        Ground-truth segmentation mask.

    num_classes : int, optional
        Number of segmentation classes.

    include_background : bool
        Whether class 0 should be included.

    Returns
    -------
    float
        Mean IoU.
    """
    prediction, target = _prepare_masks(
        prediction,
        target,
    )

    if num_classes is None:
        num_classes = int(
            max(
                prediction.max(initial=0),
                target.max(initial=0),
            )
            + 1
        )

    start_class = 0 if include_background else 1

    scores = []

    for class_id in range(
        start_class,
        num_classes,
    ):
        pred_class = prediction == class_id
        target_class = target == class_id

        scores.append(
            _binary_iou(
                pred_class,
                target_class,
            )
        )

    if not scores:
        return 0.0

    return float(np.mean(scores))


def segmentation_precision(
    prediction,
    target,
    num_classes: int | None = None,
    include_background: bool = False,
) -> float:
    """
    Compute mean segmentation precision.

    Parameters
    ----------
    prediction : array-like
        Predicted segmentation mask.

    target : array-like
        Ground-truth segmentation mask.

    num_classes : int, optional
        Number of segmentation classes.

    include_background : bool
        Whether class 0 should be included.

    Returns
    -------
    float
        Mean precision.
    """
    prediction, target = _prepare_masks(
        prediction,
        target,
    )

    if num_classes is None:
        num_classes = int(
            max(
                prediction.max(initial=0),
                target.max(initial=0),
            )
            + 1
        )

    start_class = 0 if include_background else 1

    scores = []

    for class_id in range(
        start_class,
        num_classes,
    ):
        pred_class = prediction == class_id
        target_class = target == class_id

        scores.append(
            _binary_precision(
                pred_class,
                target_class,
            )
        )

    if not scores:
        return 0.0

    return float(np.mean(scores))


def segmentation_recall(
    prediction,
    target,
    num_classes: int | None = None,
    include_background: bool = False,
) -> float:
    """
    Compute mean segmentation recall.

    Parameters
    ----------
    prediction : array-like
        Predicted segmentation mask.

    target : array-like
        Ground-truth segmentation mask.

    num_classes : int, optional
        Number of segmentation classes.

    include_background : bool
        Whether class 0 should be included.

    Returns
    -------
    float
        Mean recall.
    """
    prediction, target = _prepare_masks(
        prediction,
        target,
    )

    if num_classes is None:
        num_classes = int(
            max(
                prediction.max(initial=0),
                target.max(initial=0),
            )
            + 1
        )

    start_class = 0 if include_background else 1

    scores = []

    for class_id in range(
        start_class,
        num_classes,
    ):
        pred_class = prediction == class_id
        target_class = target == class_id

        scores.append(
            _binary_recall(
                pred_class,
                target_class,
            )
        )

    if not scores:
        return 0.0

    return float(np.mean(scores))


def _surface_distances(
    prediction,
    target,
):
    """
    Compute symmetric surface distances using scipy.

    Returns
    -------
    tuple
        Forward and backward surface distances.
    """
    from scipy.ndimage import (
        binary_erosion,
        distance_transform_edt,
        generate_binary_structure,
    )

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    if not prediction.any() or not target.any():
        return None, None

    structure = generate_binary_structure(
        prediction.ndim,
        1,
    )

    prediction_surface = prediction ^ binary_erosion(
        prediction,
        structure=structure,
        border_value=1,
    )

    target_surface = target ^ binary_erosion(
        target,
        structure=structure,
        border_value=1,
    )

    target_distance = distance_transform_edt(
        ~target_surface
    )

    prediction_distance = distance_transform_edt(
        ~prediction_surface
    )

    prediction_to_target = target_distance[
        prediction_surface
    ]

    target_to_prediction = prediction_distance[
        target_surface
    ]

    return (
        prediction_to_target,
        target_to_prediction,
    )


def hd95(
    prediction,
    target,
    num_classes: int | None = None,
    include_background: bool = False,
) -> float:
    """
    Compute mean 95th-percentile Hausdorff distance.

    HD95 is calculated symmetrically from the surface distances
    between prediction and ground truth.

    Parameters
    ----------
    prediction : array-like
        Predicted segmentation mask.

    target : array-like
        Ground-truth segmentation mask.

    num_classes : int, optional
        Number of segmentation classes.

    include_background : bool
        Whether class 0 should be included.

    Returns
    -------
    float
        Mean HD95.

    Notes
    -----
    This implementation assumes isotropic pixel spacing. If physical
    voxel spacing is required, the distance transform should be
    modified to incorporate the actual spacing.
    """
    prediction, target = _prepare_masks(
        prediction,
        target,
    )

    if num_classes is None:
        num_classes = int(
            max(
                prediction.max(initial=0),
                target.max(initial=0),
            )
            + 1
        )

    start_class = 0 if include_background else 1

    distances = []

    for class_id in range(
        start_class,
        num_classes,
    ):
        pred_class = prediction == class_id
        target_class = target == class_id

        pred_dist, target_dist = _surface_distances(
            pred_class,
            target_class,
        )

        if pred_dist is None or target_dist is None:
            continue

        all_distances = np.concatenate(
            [
                pred_dist,
                target_dist,
            ]
        )

        if all_distances.size > 0:
            distances.append(
                np.percentile(
                    all_distances,
                    95,
                )
            )

    if not distances:
        return 0.0

    return float(np.mean(distances))


def segmentation_metrics(
    prediction,
    target,
    num_classes: int = 4,
    include_background: bool = False,
) -> dict:
    """
    Compute the complete segmentation metric set.

    Parameters
    ----------
    prediction : array-like
        Predicted segmentation mask.

    target : array-like
        Ground-truth segmentation mask.

    num_classes : int
        Number of segmentation classes.

    include_background : bool
        Whether class 0 should be included.

    Returns
    -------
    dict
        Dice, IoU, precision, recall, and HD95.
    """
    return {
        "dice": dice_score(
            prediction,
            target,
            num_classes=num_classes,
            include_background=include_background,
        ),
        "iou": iou_score(
            prediction,
            target,
            num_classes=num_classes,
            include_background=include_background,
        ),
        "precision": segmentation_precision(
            prediction,
            target,
            num_classes=num_classes,
            include_background=include_background,
        ),
        "recall": segmentation_recall(
            prediction,
            target,
            num_classes=num_classes,
            include_background=include_background,
        ),
        "hd95": hd95(
            prediction,
            target,
            num_classes=num_classes,
            include_background=include_background,
        ),
    }


__all__ = [
    "dice_score",
    "iou_score",
    "segmentation_precision",
    "segmentation_recall",
    "hd95",
    "segmentation_metrics",
]