"""
Uncertainty-Calibrated Spatial Attention (UCSA).

The module estimates:
    1. A base spatial attention map.
    2. A spatial uncertainty map.
    3. An uncertainty-modulated attention map.

The refined feature representation follows:

    A^l = sigmoid(phi_a(F^l))

    U^l = sigmoid(phi_u(F^l))

    A_U^l = A^l * (1 - U^l)^gamma

    F_R^l = F^l + A_U^l * F^l

The module returns the refined features together with the
attention and uncertainty maps for subsequent uncertainty-aware
loss computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UncertaintyCalibratedSpatialAttention(nn.Module):
    """
    Uncertainty-Calibrated Spatial Attention module.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input feature map.

    gamma : float
        Controls the strength of uncertainty modulation.

    reduction : int
        Channel reduction used in the internal attention
        representation.
    """

    def __init__(
        self,
        in_channels: int,
        gamma: float = 1.0,
        reduction: int = 4,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be greater than zero."
            )

        if gamma < 0:
            raise ValueError(
                "gamma must be non-negative."
            )

        hidden_channels = max(
            in_channels // reduction,
            1,
        )

        self.in_channels = in_channels
        self.gamma = gamma

        # Shared feature projection.
        self.feature_projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                hidden_channels
            ),
            nn.ReLU(inplace=True),
        )

        # Base spatial attention:
        #
        # A^l = sigmoid(phi_a(F^l))
        self.attention_projection = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
            bias=True,
        )

        # Spatial uncertainty:
        #
        # U^l = sigmoid(phi_u(F^l))
        self.uncertainty_projection = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Apply uncertainty-calibrated spatial attention.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape [B, C, H, W].

        Returns
        -------
        refined_features : torch.Tensor
            Uncertainty-aware refined feature map [B, C, H, W].

        attention : torch.Tensor
            Base spatial attention map [B, 1, H, W].

        uncertainty : torch.Tensor
            Spatial uncertainty map [B, 1, H, W].
        """
        if x.ndim != 4:
            raise ValueError(
                "UCSA expects input with shape [B, C, H, W]."
            )

        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"got {x.size(1)}."
            )

        projected = self.feature_projection(x)

        # --------------------------------------------------------
        # Base attention map
        # --------------------------------------------------------

        attention = torch.sigmoid(
            self.attention_projection(
                projected
            )
        )

        # --------------------------------------------------------
        # Spatial uncertainty map
        # --------------------------------------------------------

        uncertainty = torch.sigmoid(
            self.uncertainty_projection(
                projected
            )
        )

        # --------------------------------------------------------
        # Uncertainty-calibrated attention
        #
        # A_U = A * (1 - U)^gamma
        # --------------------------------------------------------

        uncertainty_weight = (
            1.0 - uncertainty
        ).pow(self.gamma)

        calibrated_attention = (
            attention * uncertainty_weight
        )

        # --------------------------------------------------------
        # Residual feature refinement
        #
        # F_R = F + A_U * F
        # --------------------------------------------------------

        refined_features = (
            x
            + calibrated_attention * x
        )

        return (
            refined_features,
            attention,
            uncertainty,
        )


__all__ = [
    "UncertaintyCalibratedSpatialAttention",
]