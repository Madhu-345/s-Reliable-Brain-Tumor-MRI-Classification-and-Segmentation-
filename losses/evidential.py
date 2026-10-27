"""
Evidential learning losses for brain tumor classification and segmentation.

The losses are based on Dirichlet evidence modeling. Network outputs are
converted into non-negative evidence using Softplus, followed by
Dirichlet parameters:

    alpha_k = evidence_k + 1

The resulting Dirichlet distribution provides both class probabilities
and uncertainty estimates.

Implemented losses:
    - Evidential classification loss
    - Evidential segmentation loss
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _dirichlet_parameters(
    evidence: torch.Tensor,
) -> torch.Tensor:
    """
    Convert evidence into Dirichlet concentration parameters.

    Parameters
    ----------
    evidence : torch.Tensor
        Non-negative evidence tensor.

    Returns
    -------
    torch.Tensor
        Dirichlet concentration parameters alpha.
    """
    return evidence + 1.0


def _evidence_from_logits(
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Convert network logits into non-negative evidence.

    Softplus is used to ensure numerical positivity.

    Parameters
    ----------
    logits : torch.Tensor
        Raw network outputs.

    Returns
    -------
    torch.Tensor
        Non-negative evidence.
    """
    return F.softplus(logits)


def evidential_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Evidential classification loss.

    The loss follows the manuscript formulation:

        L_ECL = (1/K) sum_k [
            (y_k - p_k)^2
            + p_k(1-p_k)/(S+1)
        ]

    where:

        evidence_k = Softplus(logit_k)
        alpha_k    = evidence_k + 1
        p_k        = alpha_k / sum_j(alpha_j)
        S          = sum_j(alpha_j)

    Parameters
    ----------
    logits : torch.Tensor
        Classification logits of shape [B, K].

    targets : torch.Tensor
        Class labels of shape [B], containing integer class indices.

    Returns
    -------
    torch.Tensor
        Scalar evidential classification loss.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"Expected classification logits with shape [B, K], "
            f"got {tuple(logits.shape)}."
        )

    if targets.ndim != 1:
        raise ValueError(
            f"Expected classification targets with shape [B], "
            f"got {tuple(targets.shape)}."
        )

    if logits.size(0) != targets.size(0):
        raise ValueError(
            "Batch size of logits and targets must match."
        )

    evidence = _evidence_from_logits(logits)
    alpha = _dirichlet_parameters(evidence)

    strength = torch.sum(alpha, dim=1, keepdim=True)
    probabilities = alpha / strength

    num_classes = logits.size(1)

    one_hot_targets = F.one_hot(
        targets.long(),
        num_classes=num_classes,
    ).to(dtype=logits.dtype)

    squared_error = (
        one_hot_targets - probabilities
    ).pow(2)

    variance = (
        probabilities * (1.0 - probabilities)
    ) / (strength + 1.0)

    loss = squared_error + variance

    return loss.mean()


def evidential_segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Evidential segmentation loss.

    The loss is computed independently at every pixel:

        L_EDL^s = (1/|Omega|) sum_i (1/K) sum_k [
            (y_i,k - p_i,k)^2
            + p_i,k(1-p_i,k)/(S_i+1)
        ]

    Parameters
    ----------
    logits : torch.Tensor
        Segmentation logits of shape [B, K, H, W].

    targets : torch.Tensor
        Segmentation labels.

        Supported formats:
            [B, H, W]      - integer class labels
            [B, K, H, W]   - one-hot encoded labels

    Returns
    -------
    torch.Tensor
        Scalar evidential segmentation loss.
    """
    if logits.ndim != 4:
        raise ValueError(
            f"Expected segmentation logits with shape [B, K, H, W], "
            f"got {tuple(logits.shape)}."
        )

    batch_size, num_classes, height, width = logits.shape

    evidence = _evidence_from_logits(logits)
    alpha = _dirichlet_parameters(evidence)

    strength = torch.sum(
        alpha,
        dim=1,
        keepdim=True,
    )

    probabilities = alpha / strength

    if targets.ndim == 3:
        if targets.shape != (
            batch_size,
            height,
            width,
        ):
            raise ValueError(
                "Segmentation target spatial dimensions do not "
                "match the logits."
            )

        one_hot_targets = F.one_hot(
            targets.long(),
            num_classes=num_classes,
        )

        one_hot_targets = (
            one_hot_targets
            .permute(0, 3, 1, 2)
            .to(dtype=logits.dtype)
        )

    elif targets.ndim == 4:
        if targets.shape != logits.shape:
            raise ValueError(
                "One-hot segmentation targets must have the same "
                "shape as logits."
            )

        one_hot_targets = targets.to(dtype=logits.dtype)

    else:
        raise ValueError(
            "Segmentation targets must have shape [B,H,W] "
            "or [B,K,H,W]."
        )

    squared_error = (
        one_hot_targets - probabilities
    ).pow(2)

    variance = (
        probabilities * (1.0 - probabilities)
    ) / (strength + 1.0)

    loss = squared_error + variance

    return loss.mean()


__all__ = [
    "evidential_classification_loss",
    "evidential_segmentation_loss",
]