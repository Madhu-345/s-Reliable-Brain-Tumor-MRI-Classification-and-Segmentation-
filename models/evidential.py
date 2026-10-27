"""
Evidential prediction heads for the uncertainty-aware multi-task
brain tumor MRI classification and segmentation framework.

The heads convert network features into non-negative evidence and
represent predictions using Dirichlet distributions.

Classification:
    Features
        ↓
    Evidence
        ↓
    Dirichlet parameters
        ↓
    Class probabilities
        ↓
    Predictive uncertainty
        ↓
    Reliability

Segmentation:
    Pixel-wise features
        ↓
    Pixel-wise evidence
        ↓
    Pixel-wise Dirichlet parameters
        ↓
    Pixel-wise probabilities
        ↓
    Pixel-wise uncertainty
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialClassificationHead(nn.Module):
    """
    Evidential classification head.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.

    num_classes : int
        Number of classification classes.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 4,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.classifier = nn.Linear(
            in_channels,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Generate evidential classification predictions.

        Parameters
        ----------
        x : torch.Tensor
            Feature tensor of shape [B, C].

        Returns
        -------
        dict
            Contains:
                evidence
                alpha
                probabilities
                uncertainty
                reliability
                prediction
        """
        if x.ndim != 2:
            raise ValueError(
                "Classification head expects input "
                "with shape [B, C]."
            )

        logits = self.classifier(x)

        evidence = F.softplus(logits)

        alpha = evidence + 1.0

        strength = torch.sum(
            alpha,
            dim=1,
            keepdim=True,
        )

        probabilities = (
            alpha / strength
        )

        uncertainty = (
            float(self.num_classes) / strength
        )

        reliability = 1.0 - uncertainty

        prediction = torch.argmax(
            probabilities,
            dim=1,
        )

        return {
            "logits": logits,
            "evidence": evidence,
            "alpha": alpha,
            "probabilities": probabilities,
            "uncertainty": uncertainty.squeeze(1),
            "reliability": reliability.squeeze(1),
            "prediction": prediction,
        }


class EvidentialSegmentationHead(nn.Module):
    """
    Pixel-wise evidential segmentation head.

    Parameters
    ----------
    in_channels : int
        Number of decoder feature channels.

    num_classes : int
        Number of segmentation regions.

        For the proposed BraTS implementation:
            3 = WT, TC, ET
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 3,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.classifier = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Generate pixel-wise evidential segmentation predictions.

        Parameters
        ----------
        x : torch.Tensor
            Decoder features with shape [B, C, H, W].

        Returns
        -------
        dict
            Contains:
                logits
                evidence
                alpha
                probabilities
                uncertainty
                reliability
                prediction
        """
        if x.ndim != 4:
            raise ValueError(
                "Segmentation head expects input "
                "with shape [B, C, H, W]."
            )

        logits = self.classifier(x)

        evidence = F.softplus(logits)

        alpha = evidence + 1.0

        strength = torch.sum(
            alpha,
            dim=1,
            keepdim=True,
        )

        probabilities = (
            alpha / strength
        )

        uncertainty = (
            float(self.num_classes) / strength
        )

        reliability = 1.0 - uncertainty

        prediction = torch.argmax(
            probabilities,
            dim=1,
        )

        return {
            "logits": logits,
            "evidence": evidence,
            "alpha": alpha,
            "probabilities": probabilities,
            "uncertainty": uncertainty.squeeze(1),
            "reliability": reliability.squeeze(1),
            "prediction": prediction,
        }


__all__ = [
    "EvidentialClassificationHead",
    "EvidentialSegmentationHead",
]