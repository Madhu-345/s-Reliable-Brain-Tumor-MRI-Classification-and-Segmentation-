"""
Multi-scale segmentation decoder for the uncertainty-aware
multi-task brain tumor MRI classification and segmentation framework.

The decoder progressively upsamples deep semantic features while
fusing them with higher-resolution backbone features through
skip connections.

Input feature hierarchy:
    C2: 256 channels
    C3: 512 channels
    C4: 1024 channels
    C5: 2048 channels

Output:
    Pixel-wise feature representation suitable for the
    evidential segmentation head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two-layer convolutional refinement block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Upsampling and skip-feature fusion block.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Lower-resolution decoder feature.

        skip : torch.Tensor
            Corresponding higher-resolution encoder feature.
        """
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        return self.conv(x)


class MultiScaleDecoder(nn.Module):
    """
    Progressive multi-scale segmentation decoder.

    The decoder follows:

        C5 -> C4 -> C3 -> C2 -> full-resolution features

    with skip connections at each stage.
    """

    def __init__(
        self,
        output_channels: int = 128,
    ) -> None:
        super().__init__()

        self.decoder_c4 = DecoderBlock(
            in_channels=2048,
            skip_channels=1024,
            out_channels=512,
        )

        self.decoder_c3 = DecoderBlock(
            in_channels=512,
            skip_channels=512,
            out_channels=256,
        )

        self.decoder_c2 = DecoderBlock(
            in_channels=256,
            skip_channels=256,
            out_channels=128,
        )

        self.final_refinement = ConvBlock(
            in_channels=128,
            out_channels=output_channels,
        )

        self.output_channels = output_channels

    def forward(
        self,
        features: dict[str, torch.Tensor],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Decode multi-scale backbone features.

        Parameters
        ----------
        features : dict
            Dictionary containing:

                C2: [B, 256, H/4,  W/4]
                C3: [B, 512, H/8,  W/8]
                C4: [B,1024, H/16, W/16]
                C5: [B,2048, H/32, W/32]

        output_size : tuple, optional
            Desired final spatial resolution (H, W).

        Returns
        -------
        torch.Tensor
            Pixel-wise decoder features.
        """
        required = {"C2", "C3", "C4", "C5"}

        missing = required.difference(features.keys())

        if missing:
            raise ValueError(
                f"Missing backbone feature maps: {sorted(missing)}"
            )

        c2 = features["C2"]
        c3 = features["C3"]
        c4 = features["C4"]
        c5 = features["C5"]

        x = self.decoder_c4(
            c5,
            c4,
        )

        x = self.decoder_c3(
            x,
            c3,
        )

        x = self.decoder_c2(
            x,
            c2,
        )

        x = self.final_refinement(x)

        if output_size is not None:
            x = F.interpolate(
                x,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        return x


__all__ = [
    "ConvBlock",
    "DecoderBlock",
    "MultiScaleDecoder",
]