"""
Model package for the uncertainty-aware multi-task
brain tumor MRI classification and segmentation framework.

This package provides:
    - ResNet-50 feature extraction backbone
    - Uncertainty-Calibrated Spatial Attention (UCSA)
    - Evidential classification and segmentation heads
    - Multi-scale segmentation decoder
    - Unified multi-task model
"""

from .backbone import ResNet50Backbone
from .ucsa import UncertaintyCalibratedSpatialAttention
from .evidential_heads import (
    EvidentialClassificationHead,
    EvidentialSegmentationHead,
)
from .decoder import MultiScaleDecoder
from .multitask_model import UncertaintyAwareMultiTaskModel


__all__ = [
    "ResNet50Backbone",
    "UncertaintyCalibratedSpatialAttention",
    "EvidentialClassificationHead",
    "EvidentialSegmentationHead",
    "MultiScaleDecoder",
    "UncertaintyAwareMultiTaskModel",
]from .multitask import UncertaintyAwareMultiTaskModel
