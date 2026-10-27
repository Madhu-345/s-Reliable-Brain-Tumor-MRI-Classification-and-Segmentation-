"""
Evaluation metrics for the uncertainty-aware multi-task
brain tumor MRI classification and segmentation framework.

This package provides:
    - Classification metrics
    - Segmentation metrics
    - Uncertainty and calibration metrics
    - Selective prediction and reliability metrics
"""

from .classification import (
    accuracy,
    precision,
    recall,
    specificity,
    f1_score,
    auroc,
)

from .segmentation import (
    dice_score,
    iou_score,
    segmentation_precision,
    segmentation_recall,
    hd95,
)

from .uncertainty import (
    predictive_uncertainty,
    expected_calibration_error,
    negative_log_likelihood,
)

from .selective import (
    coverage,
    selective_accuracy,
    selective_risk,
)


__all__ = [
    # Classification
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1_score",
    "auroc",

    # Segmentation
    "dice_score",
    "iou_score",
    "segmentation_precision",
    "segmentation_recall",
    "hd95",

    # Uncertainty
    "predictive_uncertainty",
    "expected_calibration_error",
    "negative_log_likelihood",

    # Selective prediction
    "coverage",
    "selective_accuracy",
    "selective_risk",
]