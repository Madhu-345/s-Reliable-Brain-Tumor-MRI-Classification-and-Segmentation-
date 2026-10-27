"""
Reliability metrics for the uncertainty-aware multi-task
brain tumor MRI classification and segmentation framework.

Reliability is derived from evidential uncertainty:

    U = K / S

where:
    K = number of classes
    S = total Dirichlet evidence strength.

The reliability score is:

    R = 1 - U

A prediction is accepted when:

    R >= threshold

otherwise it is classified as an abstention.
"""

from __future__ import annotations

import torch


def predictive_uncertainty(
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Compute evidential predictive uncertainty.

    Formula:

        U = K / S

    where:
        K = number of classes
        S = sum(alpha)

    Parameters
    ----------
    alpha : torch.Tensor
        Dirichlet parameters.

        Classification:
            [B, K]

        Segmentation:
            [B, K, H, W]

    Returns
    -------
    torch.Tensor
        Predictive uncertainty.
    """
    if alpha.ndim < 2:
        raise ValueError(
            "alpha must contain a class dimension."
        )

    num_classes = alpha.size(1)

    strength = torch.sum(
        alpha,
        dim=1,
        keepdim=True,
    )

    uncertainty = (
        float(num_classes) / strength
    )

    return uncertainty.squeeze(1)


def reliability_score(
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Compute reliability from Dirichlet uncertainty.

    Formula:

        R = 1 - U

    Parameters
    ----------
    alpha : torch.Tensor
        Dirichlet parameters.

    Returns
    -------
    torch.Tensor
        Reliability score.
    """
    uncertainty = predictive_uncertainty(alpha)

    return 1.0 - uncertainty


def reliability_decision(
    reliability: torch.Tensor,
    threshold: float = 0.80,
) -> torch.Tensor:
    """
    Generate accept/abstain decisions.

    Formula:

        Accept, if R >= threshold
        Abstain, otherwise

    Parameters
    ----------
    reliability : torch.Tensor
        Reliability scores.

    threshold : float
        Reliability threshold.

    Returns
    -------
    torch.Tensor
        Boolean tensor where True indicates acceptance.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Reliability threshold must be between 0 and 1."
        )

    return reliability >= threshold


def classification_reliability(
    alpha: torch.Tensor,
    threshold: float = 0.80,
) -> dict:
    """
    Compute classification reliability and accept/abstain decisions.

    Parameters
    ----------
    alpha : torch.Tensor
        Classification Dirichlet parameters [B, K].

    threshold : float
        Reliability threshold.

    Returns
    -------
    dict
        Uncertainty, reliability, and acceptance decisions.
    """
    if alpha.ndim != 2:
        raise ValueError(
            "Classification alpha must have shape [B, K]."
        )

    uncertainty = predictive_uncertainty(alpha)
    reliability = 1.0 - uncertainty
    accepted = reliability_decision(
        reliability,
        threshold,
    )

    return {
        "uncertainty": uncertainty,
        "reliability": reliability,
        "accepted": accepted,
        "abstained": ~accepted,
    }


def segmentation_reliability(
    alpha: torch.Tensor,
    threshold: float = 0.80,
    tumor_mask: torch.Tensor | None = None,
) -> dict:
    """
    Compute image-level segmentation reliability.

    Pixel-wise evidential uncertainty is first computed. When a tumor
    mask is supplied, reliability is averaged over the predicted tumor
    region. If the tumor region is empty, the uncertainty over the
    complete image is used as the fallback.

    Parameters
    ----------
    alpha : torch.Tensor
        Segmentation Dirichlet parameters [B, K, H, W].

    threshold : float
        Reliability threshold.

    tumor_mask : torch.Tensor, optional
        Boolean tumor-region mask [B, H, W].

    Returns
    -------
    dict
        Pixel uncertainty, image-level uncertainty, reliability,
        and accept/abstain decisions.
    """
    if alpha.ndim != 4:
        raise ValueError(
            "Segmentation alpha must have shape [B, K, H, W]."
        )

    pixel_uncertainty = predictive_uncertainty(alpha)

    if tumor_mask is not None:
        if tumor_mask.shape != pixel_uncertainty.shape:
            raise ValueError(
                "tumor_mask must have shape [B, H, W]."
            )

        tumor_mask = tumor_mask.bool()

        batch_size = pixel_uncertainty.size(0)

        image_uncertainty = []

        for i in range(batch_size):
            region = tumor_mask[i]

            if torch.any(region):
                uncertainty_i = pixel_uncertainty[i][region].mean()
            else:
                uncertainty_i = pixel_uncertainty[i].mean()

            image_uncertainty.append(uncertainty_i)

        image_uncertainty = torch.stack(
            image_uncertainty
        )

    else:
        image_uncertainty = pixel_uncertainty.flatten(
            start_dim=1
        ).mean(dim=1)

    reliability = 1.0 - image_uncertainty

    accepted = reliability_decision(
        reliability,
        threshold,
    )

    return {
        "pixel_uncertainty": pixel_uncertainty,
        "uncertainty": image_uncertainty,
        "reliability": reliability,
        "accepted": accepted,
        "abstained": ~accepted,
    }


def coverage(
    reliability: torch.Tensor,
    threshold: float = 0.80,
) -> float:
    """
    Compute selective prediction coverage.

    Formula:

        Coverage = N_accepted / N_total

    Parameters
    ----------
    reliability : torch.Tensor
        Reliability scores.

    threshold : float
        Reliability threshold.

    Returns
    -------
    float
        Coverage in the range [0, 1].
    """
    accepted = reliability_decision(
        reliability,
        threshold,
    )

    return float(
        accepted.float().mean().item()
    )


__all__ = [
    "predictive_uncertainty",
    "reliability_score",
    "reliability_decision",
    "classification_reliability",
    "segmentation_reliability",
    "coverage",
]