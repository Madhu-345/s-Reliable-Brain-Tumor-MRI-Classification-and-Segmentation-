"""
ResNet-50 backbone for the uncertainty-aware multi-task
brain tumor MRI classification and segmentation framework.

The backbone extracts hierarchical multi-scale feature maps that
are subsequently refined by the Uncertainty-Calibrated Spatial
Attention (UCSA) module.

Feature levels returned:
    - C2: early spatial features
    - C3: intermediate features
    - C4: deep multi-scale features
    - C5: deepest semantic features
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)


class ResNet50Backbone(nn.Module):
    """
    ResNet-50 feature extraction backbone.

    Parameters
    ----------
    pretrained : bool
        If True, initialize the backbone with ImageNet-pretrained
        weights.

    trainable : bool
        If True, all backbone parameters are trainable.
    """

    def __init__(
        self,
        pretrained: bool = True,
        trainable: bool = True,
    ) -> None:
        super().__init__()

        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        backbone = resnet50(
            weights=weights
        )

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.out_channels = {
            "C2": 256,
            "C3": 512,
            "C4": 1024,
            "C5": 2048,
        }

        if not trainable:
            self.freeze()

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Extract hierarchical feature maps.

        Parameters
        ----------
        x : torch.Tensor
            Input MRI tensor with shape [B, 3, H, W].

        Returns
        -------
        dict[str, torch.Tensor]
            Multi-scale feature maps:

                C2: [B, 256, H/4,  W/4]
                C3: [B, 512, H/8,  W/8]
                C4: [B,1024, H/16, W/16]
                C5: [B,2048, H/32, W/32]
        """
        if x.ndim != 4:
            raise ValueError(
                "Input must have shape [B, C, H, W]."
            )

        x = self.stem(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return {
            "C2": c2,
            "C3": c3,
            "C4": c4,
            "C5": c5,
        }

    def freeze(self) -> None:
        """Freeze all backbone parameters."""
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters."""
        for parameter in self.parameters():
            parameter.requires_grad = True

    def trainable_parameters(self):
        """
        Return parameters that are currently trainable.
        """
        return (
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        )


__all__ = [
    "ResNet50Backbone",
]