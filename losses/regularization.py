"""
Dirichlet KL regularization for evidential learning.

The regularization encourages the predicted Dirichlet distribution
to remain close to a uniform Dirichlet prior when appropriate.

Manuscript formulation:

    L_DKL = D_KL(
        Dir(alpha_tilde) || Dir(1)
    )

where alpha_tilde denotes the target-adjusted Dirichlet parameters
and Dir(1) is the uniform Dirichlet prior.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dirichlet_kl_loss(
    evidence: torch.Tensor,
    targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute Dirichlet KL-divergence regularization.

    Parameters
    ----------
    evidence : torch.Tensor
        Non-negative evidence tensor.

        Classification:
            [B, K]

        Segmentation:
            [B, K, H, W]

    targets : torch.Tensor, optional
        Optional target labels. When supplied, the evidence associated
        with the target class is excluded from the regularization
        adjustment, following the evidential learning formulation.

        Classification:
            [B]

        Segmentation:
            [B, H, W]

    Returns
    -------
    torch.Tensor
        Scalar KL regularization loss.
    """
    if evidence.ndim < 2:
        raise ValueError(
            "Evidence must have at least two dimensions "
            "(batch dimension and class dimension)."
        )

    # Convert evidence to Dirichlet concentration parameters.
    alpha = evidence + 1.0

    if targets is not None:
        alpha = _adjust_alpha_for_targets(
            alpha=alpha,
            targets=targets,
        )

    # Uniform Dirichlet prior: beta_k = 1.
    beta = torch.ones_like(alpha)

    alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
    beta_sum = torch.sum(beta, dim=1, keepdim=True)

    # KL(Dir(alpha) || Dir(beta))
    log_gamma_alpha_sum = torch.lgamma(alpha_sum)
    log_gamma_alpha = torch.lgamma(alpha)

    log_gamma_beta_sum = torch.lgamma(beta_sum)
    log_gamma_beta = torch.lgamma(beta)

    digamma_alpha = torch.digamma(alpha)
    digamma_alpha_sum = torch.digamma(alpha_sum)

    kl = (
        log_gamma_alpha_sum
        - torch.sum(log_gamma_alpha, dim=1, keepdim=True)
        - log_gamma_beta_sum
        + torch.sum(log_gamma_beta, dim=1, keepdim=True)
        + torch.sum(
            (alpha - beta)
            * (digamma_alpha - digamma_alpha_sum),
            dim=1,
            keepdim=True,
        )
    )

    return kl.mean()


def _adjust_alpha_for_targets(
    alpha: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Adjust Dirichlet parameters using the target labels.

    The target class retains its evidence while non-target evidence
    is regularized toward the uniform prior.

    Parameters
    ----------
    alpha : torch.Tensor
        Dirichlet parameters with shape [B,K,...].

    targets : torch.Tensor
        Integer target labels with shape [B,...].

    Returns
    -------
    torch.Tensor
        Target-adjusted Dirichlet parameters.
    """
    num_classes = alpha.size(1)

    if alpha.ndim == 2:
        if targets.ndim != 1:
            raise ValueError(
                "Classification targets must have shape [B]."
            )

        if targets.size(0) != alpha.size(0):
            raise ValueError(
                "Batch size of targets and evidence must match."
            )

        one_hot = F.one_hot(
            targets.long(),
            num_classes=num_classes,
        )

        one_hot = one_hot.to(
            dtype=alpha.dtype,
            device=alpha.device,
        )

        # Keep target-class evidence and regularize non-target evidence.
        evidence = alpha - 1.0
        adjusted_evidence = evidence * one_hot

        return adjusted_evidence + 1.0

    if alpha.ndim == 4:
        if targets.ndim != 3:
            raise ValueError(
                "Segmentation targets must have shape [B,H,W]."
            )

        if targets.shape != (
            alpha.size(0),
            alpha.size(2),
            alpha.size(3),
        ):
            raise ValueError(
                "Segmentation target dimensions do not match "
                "the evidence tensor."
            )

        one_hot = F.one_hot(
            targets.long(),
            num_classes=num_classes,
        )

        one_hot = one_hot.permute(
            0, 3, 1, 2
        ).to(
            dtype=alpha.dtype,
            device=alpha.device,
        )

        evidence = alpha - 1.0
        adjusted_evidence = evidence * one_hot

        return adjusted_evidence + 1.0

    raise ValueError(
        "Unsupported evidence shape. Expected [B,K] or [B,K,H,W]."
    )


__all__ = [
    "dirichlet_kl_loss",
]