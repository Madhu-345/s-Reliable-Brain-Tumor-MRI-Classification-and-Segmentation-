"""
Uncertainty-Aware Multi-Task Evidential Model.

This module integrates:

    Input MRI
        ↓
    ResNet-50 Backbone
        ↓
    Multi-Scale Features
        ↓
    Uncertainty-Calibrated Spatial Attention
        ↓
    Reliability-Aware Features
        ├──────────────────────┐
        ↓                      ↓
    Classification          Segmentation
        ↓                      ↓
    Evidential Head         Decoder
        ↓                      ↓
    Dirichlet Head          Evidential Head
        ↓                      ↓
    Probability            Segmentation Mask
        ↓                      ↓
    Uncertainty            Pixel Uncertainty

The model supports classification-only, segmentation-only,
and joint multi-task inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import ResNet50Backbone
from .ucsa import UncertaintyCalibratedSpatialAttention
from .decoder import MultiScaleDecoder
from .evidential_heads import (
    EvidentialClassificationHead,
    EvidentialSegmentationHead,
)


class UncertaintyAwareMultiTaskModel(nn.Module):
    """
    Unified uncertainty-aware multi-task model.

    Parameters
    ----------
    num_classes_classification : int
        Number of classification classes.

    num_classes_segmentation : int
        Number of segmentation classes/regions.

    pretrained : bool
        Use ImageNet-pretrained ResNet-50 weights.

    trainable_backbone : bool
        Whether the ResNet-50 backbone is trainable.

    enable_classification : bool
        Enable classification branch.

    enable_segmentation : bool
        Enable segmentation branch.

    input_channels : int
        Number of input image channels.

    classification_channels : int
        Expected number of channels for the ResNet-50 input.

    attention_gamma : float
        Strength of uncertainty modulation.
    """

    def __init__(
        self,
        num_classes_classification: int = 4,
        num_classes_segmentation: int = 3,
        pretrained: bool = True,
        trainable_backbone: bool = True,
        enable_classification: bool = True,
        enable_segmentation: bool = True,
        input_channels: int = 3,
        classification_channels: int = 3,
        attention_gamma: float = 1.0,
    ) -> None:
        super().__init__()

        self.num_classes_classification = (
            num_classes_classification
        )

        self.num_classes_segmentation = (
            num_classes_segmentation
        )

        self.enable_classification = (
            enable_classification
        )

        self.enable_segmentation = (
            enable_segmentation
        )

        self.input_channels = input_channels
        self.classification_channels = classification_channels

        # --------------------------------------------------------
        # Input adaptation
        # --------------------------------------------------------
        #
        # ResNet-50 expects 3 channels because the ImageNet
        # pretrained first convolution has 3 input channels.
        #
        # BraTS provides 4 MRI modalities:
        # T1, T1ce, T2, FLAIR.
        #
        # A lightweight 1x1 projection adapts arbitrary input
        # channel counts to the 3-channel ResNet representation.
        # --------------------------------------------------------

        if input_channels != classification_channels:
            self.input_adapter = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    classification_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    classification_channels
                ),
                nn.ReLU(inplace=True),
            )
        else:
            self.input_adapter = nn.Identity()

        # --------------------------------------------------------
        # Shared ResNet-50 backbone
        # --------------------------------------------------------

        self.backbone = ResNet50Backbone(
            pretrained=pretrained,
            trainable=trainable_backbone,
        )

        # --------------------------------------------------------
        # UCSA modules
        #
        # One attention module is applied independently to each
        # multi-scale feature level.
        # --------------------------------------------------------

        self.ucsa = nn.ModuleDict(
            {
                "C2": UncertaintyCalibratedSpatialAttention(
                    in_channels=256,
                    gamma=attention_gamma,
                ),
                "C3": UncertaintyCalibratedSpatialAttention(
                    in_channels=512,
                    gamma=attention_gamma,
                ),
                "C4": UncertaintyCalibratedSpatialAttention(
                    in_channels=1024,
                    gamma=attention_gamma,
                ),
                "C5": UncertaintyCalibratedSpatialAttention(
                    in_channels=2048,
                    gamma=attention_gamma,
                ),
            }
        )

        # --------------------------------------------------------
        # Classification branch
        # --------------------------------------------------------

        if enable_classification:
            self.classification_head = (
                EvidentialClassificationHead(
                    in_channels=2048,
                    num_classes=num_classes_classification,
                )
            )

        # --------------------------------------------------------
        # Segmentation branch
        # --------------------------------------------------------

        if enable_segmentation:
            self.decoder = MultiScaleDecoder(
                output_channels=128,
            )

            self.segmentation_head = (
                EvidentialSegmentationHead(
                    in_channels=128,
                    num_classes=num_classes_segmentation,
                )
            )

    def refine_features(
        self,
        features: dict[str, torch.Tensor],
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """
        Apply UCSA to all multi-scale feature maps.

        Returns
        -------
        refined_features : dict
            Reliability-aware feature maps.

        attention_maps : dict
            Attention maps generated at each scale.
        """
        refined_features = {}
        attention_maps = {}

        for level, feature in features.items():
            refined, attention, uncertainty = self.ucsa[level](
                feature
            )

            refined_features[level] = refined

            attention_maps[level] = {
                "attention": attention,
                "uncertainty": uncertainty,
            }

        return refined_features, attention_maps

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> dict[str, object]:
        """
        Forward pass through the complete model.

        Parameters
        ----------
        x : torch.Tensor
            Input MRI tensor.

            Classification:
                [B, 3, H, W]

            BraTS segmentation:
                [B, 4, H, W]

        return_features : bool
            If True, include intermediate feature maps in the
            returned dictionary.

        Returns
        -------
        dict
            Model predictions and uncertainty information.
        """
        if x.ndim != 4:
            raise ValueError(
                "Input must have shape [B, C, H, W]."
            )

        # --------------------------------------------------------
        # Input adaptation
        # --------------------------------------------------------

        x = self.input_adapter(x)

        # --------------------------------------------------------
        # Shared feature extraction
        # --------------------------------------------------------

        features = self.backbone(x)

        # --------------------------------------------------------
        # Uncertainty-calibrated spatial attention
        # --------------------------------------------------------

        refined_features, attention_maps = (
            self.refine_features(features)
        )

        outputs = {
            "attention": attention_maps,
        }

        # --------------------------------------------------------
        # Classification branch
        # --------------------------------------------------------

        if self.enable_classification:
            deepest_feature = refined_features["C5"]

            pooled = torch.mean(
                deepest_feature,
                dim=(2, 3),
            )

            classification_output = (
                self.classification_head(pooled)
            )

            outputs["classification"] = (
                classification_output
            )

        # --------------------------------------------------------
        # Segmentation branch
        # --------------------------------------------------------

        if self.enable_segmentation:
            decoder_features = self.decoder(
                refined_features,
                output_size=x.shape[-2:],
            )

            segmentation_output = (
                self.segmentation_head(
                    decoder_features
                )
            )

            outputs["segmentation"] = (
                segmentation_output
            )

        # --------------------------------------------------------
        # Optional intermediate representations
        # --------------------------------------------------------

        if return_features:
            outputs["features"] = features
            outputs["refined_features"] = (
                refined_features
            )

        return outputs

    def classification_parameters(self):
        """
        Return parameters belonging to the classification branch.
        """
        if not self.enable_classification:
            return []

        return self.classification_head.parameters()

    def segmentation_parameters(self):
        """
        Return parameters belonging to the segmentation branch.
        """
        parameters = []

        if self.enable_segmentation:
            parameters.extend(
                self.decoder.parameters()
            )
            parameters.extend(
                self.segmentation_head.parameters()
            )

        return parameters


__all__ = [
    "UncertaintyAwareMultiTaskModel",
]