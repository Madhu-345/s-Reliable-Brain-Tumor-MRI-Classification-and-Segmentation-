"""
Loss functions for the uncertainty-aware multi-task
brain tumor MRI classification and segmentation framework.

This package provides:
    - Evidential classification and segmentation losses
    - Dice loss
    - Dirichlet KL regularization
    - Calibration loss
    - Attention–uncertainty alignment loss
"""

from .evidential import (
    evidential_classification_loss,
    evidential_segmentation_loss,
)

from .dice import dice_loss

from .kl_regularization import dirichlet_kl_loss

from .calibration import calibration_loss

from .attention_uncertainty import attention_uncertainty_alignment_loss


__all__ = [
    "evidential_classification_loss",
    "evidential_segmentation_loss",
    "dice_loss",
    "dirichlet_kl_loss",
    "calibration_loss",
    "attention_uncertainty_alignment_loss",
]