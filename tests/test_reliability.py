"""
Unit tests for uncertainty and reliability calculations.

Tests:
    - Predictive uncertainty
    - Reliability score
    - Classification reliability
    - Segmentation reliability
    - Reliability threshold decisions
    - Coverage calculation
    - Edge cases
"""

from __future__ import annotations

import pytest
import torch

from metrics.reliability import (
    predictive_uncertainty,
    reliability_score,
    reliability_decision,
    classification_reliability,
    segmentation_reliability,
    coverage,
)


# ---------------------------------------------------------------------
# Predictive uncertainty
# ---------------------------------------------------------------------


def test_predictive_uncertainty_uniform_dirichlet():
    """
    With alpha = [1, 1, ..., 1]:

        U = K / S = 1

    Therefore, uncertainty should be maximal.
    """

    alpha = torch.ones(
        2,
        4,
    )

    uncertainty = predictive_uncertainty(
        alpha
    )

    expected = torch.ones(
        2
    )

    assert torch.allclose(
        uncertainty,
        expected,
        atol=1e-6,
    )


def test_predictive_uncertainty_high_evidence():
    """
    Higher total evidence should produce lower uncertainty.
    """

    alpha_low = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    alpha_high = torch.tensor(
        [
            [101.0, 1.0, 1.0, 1.0],
        ]
    )

    uncertainty_low = (
        predictive_uncertainty(
            alpha_low
        )
    )

    uncertainty_high = (
        predictive_uncertainty(
            alpha_high
        )
    )

    assert (
        uncertainty_high.item()
        < uncertainty_low.item()
    )


def test_predictive_uncertainty_range():
    """
    Evidential uncertainty should lie in [0, 1].
    """

    alpha = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 3.0, 4.0, 5.0],
            [100.0, 20.0, 10.0, 5.0],
        ]
    )

    uncertainty = predictive_uncertainty(
        alpha
    )

    assert torch.all(
        uncertainty >= 0.0
    )

    assert torch.all(
        uncertainty <= 1.0
    )


# ---------------------------------------------------------------------
# Reliability score
# ---------------------------------------------------------------------


def test_reliability_is_one_minus_uncertainty():
    """Reliability must equal 1 - uncertainty."""

    uncertainty = torch.tensor(
        [
            0.0,
            0.2,
            0.5,
            0.8,
            1.0,
        ]
    )

    reliability = reliability_score(
        uncertainty
    )

    expected = (
        1.0 - uncertainty
    )

    assert torch.allclose(
        reliability,
        expected,
        atol=1e-6,
    )


def test_reliability_range():
    """Reliability should lie in [0, 1]."""

    uncertainty = torch.tensor(
        [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
        ]
    )

    reliability = reliability_score(
        uncertainty
    )

    assert torch.all(
        reliability >= 0.0
    )

    assert torch.all(
        reliability <= 1.0
    )


# ---------------------------------------------------------------------
# Classification reliability
# ---------------------------------------------------------------------


def test_classification_reliability():
    """
    Test classification reliability directly from Dirichlet
    concentration parameters.
    """

    alpha = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [10.0, 1.0, 1.0, 1.0],
        ]
    )

    uncertainty, reliability = (
        classification_reliability(
            alpha
        )
    )

    assert uncertainty.shape == (
        2,
    )

    assert reliability.shape == (
        2,
    )

    assert torch.allclose(
        reliability,
        1.0 - uncertainty,
        atol=1e-6,
    )


def test_classification_reliability_increases_with_evidence():
    """
    A sample with stronger evidence should have higher reliability.
    """

    alpha_low = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    alpha_high = torch.tensor(
        [
            [100.0, 1.0, 1.0, 1.0],
        ]
    )

    _, reliability_low = (
        classification_reliability(
            alpha_low
        )
    )

    _, reliability_high = (
        classification_reliability(
            alpha_high
        )
    )

    assert (
        reliability_high.item()
        > reliability_low.item()
    )


# ---------------------------------------------------------------------
# Reliability threshold decision
# ---------------------------------------------------------------------


def test_reliability_decision_accept():
    """
    Reliability equal to or above the threshold should be accepted.
    """

    reliability = torch.tensor(
        [
            0.80,
            0.90,
            1.00,
        ]
    )

    decision = reliability_decision(
        reliability,
        threshold=0.80,
    )

    expected = torch.tensor(
        [
            True,
            True,
            True,
        ]
    )

    assert torch.equal(
        decision,
        expected,
    )


def test_reliability_decision_abstain():
    """
    Reliability below the threshold should be rejected/abstained.
    """

    reliability = torch.tensor(
        [
            0.10,
            0.50,
            0.79,
        ]
    )

    decision = reliability_decision(
        reliability,
        threshold=0.80,
    )

    expected = torch.tensor(
        [
            False,
            False,
            False,
        ]
    )

    assert torch.equal(
        decision,
        expected,
    )


def test_reliability_decision_mixed():
    """Test mixed accept/abstain decisions."""

    reliability = torch.tensor(
        [
            0.20,
            0.65,
            0.80,
            0.95,
        ]
    )

    decision = reliability_decision(
        reliability,
        threshold=0.80,
    )

    expected = torch.tensor(
        [
            False,
            False,
            True,
            True,
        ]
    )

    assert torch.equal(
        decision,
        expected,
    )


# ---------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------


def test_full_coverage():
    """All samples accepted should produce 100% coverage."""

    reliability = torch.tensor(
        [
            0.90,
            0.85,
            0.95,
            1.00,
        ]
    )

    result = coverage(
        reliability,
        threshold=0.50,
    )

    assert result == pytest.approx(
        100.0
    )


def test_zero_coverage():
    """No samples accepted should produce 0% coverage."""

    reliability = torch.tensor(
        [
            0.10,
            0.20,
            0.30,
            0.40,
        ]
    )

    result = coverage(
        reliability,
        threshold=0.50,
    )

    assert result == pytest.approx(
        0.0
    )


def test_partial_coverage():
    """
    Two of four samples satisfy the threshold:

        Coverage = 2 / 4 × 100 = 50%
    """

    reliability = torch.tensor(
        [
            0.40,
            0.60,
            0.70,
            0.30,
        ]
    )

    result = coverage(
        reliability,
        threshold=0.60,
    )

    assert result == pytest.approx(
        50.0
    )


# ---------------------------------------------------------------------
# Segmentation reliability
# ---------------------------------------------------------------------


def test_segmentation_reliability_with_tumor_mask():
    """
    Reliability should be computed over the predicted/target tumor
    region rather than the complete image when a tumor is present.
    """

    uncertainty = torch.tensor(
        [
            [
                [0.2, 0.2],
                [0.9, 0.9],
            ]
        ]
    )

    tumor_mask = torch.tensor(
        [
            [
                [1.0, 1.0],
                [0.0, 0.0],
            ]
        ]
    )

    result = segmentation_reliability(
        uncertainty,
        tumor_mask,
    )

    expected = 1.0 - 0.2

    assert result.item() == pytest.approx(
        expected,
        abs=1e-6,
    )


def test_segmentation_reliability_empty_mask():
    """
    If no tumor pixels are present, the implementation should use
    the full image as the fallback region.
    """

    uncertainty = torch.tensor(
        [
            [
                [0.2, 0.4],
                [0.6, 0.8],
            ]
        ]
    )

    tumor_mask = torch.zeros(
        1,
        2,
        2,
    )

    result = segmentation_reliability(
        uncertainty,
        tumor_mask,
    )

    expected = 1.0 - (
        (
            0.2
            + 0.4
            + 0.6
            + 0.8
        )
        / 4.0
    )

    assert result.item() == pytest.approx(
        expected,
        abs=1e-6,
    )


def test_segmentation_reliability_range():
    """Segmentation reliability must lie in [0, 1]."""

    uncertainty = torch.rand(
        2,
        8,
        8,
    )

    tumor_mask = (
        torch.rand(
            2,
            8,
            8,
        )
        > 0.5
    ).float()

    result = segmentation_reliability(
        uncertainty,
        tumor_mask,
    )

    assert torch.all(
        result >= 0.0
    )

    assert torch.all(
        result <= 1.0
    )


# ---------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------


def test_reliability_monotonically_decreases_with_uncertainty():
    """
    Greater uncertainty should never result in greater reliability.
    """

    uncertainty = torch.tensor(
        [
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
        ]
    )

    reliability = reliability_score(
        uncertainty
    )

    for i in range(
        len(reliability) - 1
    ):

        assert (
            reliability[i]
            >= reliability[i + 1]
        )


# ---------------------------------------------------------------------
# Threshold edge cases
# ---------------------------------------------------------------------


def test_threshold_zero_accepts_all():
    """Threshold zero should accept every valid reliability value."""

    reliability = torch.tensor(
        [
            0.0,
            0.2,
            0.5,
            1.0,
        ]
    )

    decision = reliability_decision(
        reliability,
        threshold=0.0,
    )

    assert torch.all(
        decision
    )


def test_threshold_one_accepts_only_perfect_reliability():
    """Threshold one should accept only reliability equal to one."""

    reliability = torch.tensor(
        [
            0.2,
            0.5,
            0.99,
            1.0,
        ]
    )

    decision = reliability_decision(
        reliability,
        threshold=1.0,
    )

    expected = torch.tensor(
        [
            False,
            False,
            False,
            True,
        ]
    )

    assert torch.equal(
        decision,
        expected,
    )


# ---------------------------------------------------------------------
# Invalid input tests
# ---------------------------------------------------------------------


def test_predictive_uncertainty_requires_positive_strength():
    """
    Alpha values below zero are invalid Dirichlet concentration
    parameters and should not be silently accepted.
    """

    alpha = torch.tensor(
        [
            [-1.0, 1.0, 1.0, 1.0],
        ]
    )

    with pytest.raises(
        (ValueError, RuntimeError, AssertionError),
    ):
        predictive_uncertainty(
            alpha
        )


def test_segmentation_reliability_shape_mismatch():
    """
    Uncertainty and tumor mask must have compatible spatial shapes.
    """

    uncertainty = torch.rand(
        1,
        8,
        8,
    )

    tumor_mask = torch.rand(
        1,
        4,
        4,
    )

    with pytest.raises(
        (ValueError, RuntimeError, AssertionError),
    ):
        segmentation_reliability(
            uncertainty,
            tumor_mask,
        )