"""
Unit and integration tests for the uncertainty-aware multi-task
brain tumor MRI classification and segmentation model.

Tests:
    1. Model construction
    2. Classification forward pass
    3. Segmentation forward pass
    4. Multi-task output structure
    5. Evidential parameter validity
    6. Probability normalization
    7. Uncertainty and reliability ranges
    8. Attention/uncertainty map generation
    9. Backward propagation
"""

from __future__ import annotations

import pytest
import torch

from models.multitask_model import (
    UncertaintyAwareMultiTaskModel,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def model() -> UncertaintyAwareMultiTaskModel:
    """
    Construct a small test instance of the proposed model.

    ImageNet weights are disabled so that the tests do not require
    downloading external pretrained weights.
    """

    model = UncertaintyAwareMultiTaskModel(
        num_classes=4,
        segmentation_classes=3,
        input_channels=4,
        classification_channels=3,
        pretrained=False,
    )

    model.eval()

    return model


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def get_classification_output(
    outputs: dict,
) -> dict:
    """Extract classification outputs."""

    if "classification" in outputs:
        value = outputs["classification"]

        if isinstance(value, dict):
            return value

    if "classification_alpha" in outputs:

        result = {}

        for key, value in outputs.items():

            if key.startswith("classification_"):

                result[
                    key.replace(
                        "classification_",
                        "",
                    )
                ] = value

        return result

    raise KeyError(
        "Classification output was not found."
    )


def get_segmentation_output(
    outputs: dict,
) -> dict:
    """Extract segmentation outputs."""

    if "segmentation" in outputs:
        value = outputs["segmentation"]

        if isinstance(value, dict):
            return value

    if "segmentation_alpha" in outputs:

        result = {}

        for key, value in outputs.items():

            if key.startswith("segmentation_"):

                result[
                    key.replace(
                        "segmentation_",
                        "",
                    )
                ] = value

        return result

    raise KeyError(
        "Segmentation output was not found."
    )


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------


def test_model_construction(model):
    """Test that the proposed model can be constructed."""

    assert isinstance(
        model,
        UncertaintyAwareMultiTaskModel,
    )


# ---------------------------------------------------------------------
# Classification forward pass
# ---------------------------------------------------------------------


@torch.no_grad()
def test_classification_forward(model):
    """
    Test classification forward propagation.

    Classification input:
        [B, 3, 224, 224]
    """

    images = torch.randn(
        2,
        3,
        224,
        224,
    )

    outputs = model(
        images,
        return_features=False,
    )

    classification = (
        get_classification_output(
            outputs
        )
    )

    assert "evidence" in classification
    assert "alpha" in classification
    assert "probabilities" in classification
    assert "uncertainty" in classification
    assert "reliability" in classification

    assert classification[
        "evidence"
    ].shape == (2, 4)

    assert classification[
        "alpha"
    ].shape == (2, 4)

    assert classification[
        "probabilities"
    ].shape == (2, 4)

    assert classification[
        "uncertainty"
    ].shape[0] == 2

    assert classification[
        "reliability"
    ].shape[0] == 2


# ---------------------------------------------------------------------
# Segmentation forward pass
# ---------------------------------------------------------------------


@torch.no_grad()
def test_segmentation_forward(model):
    """
    Test segmentation forward propagation.

    BraTS input:
        [B, 4, H, W]

    Segmentation output:
        [B, 3, H, W]

    where the three channels represent:
        WT, TC, ET
    """

    images = torch.randn(
        2,
        4,
        224,
        224,
    )

    outputs = model(
        images,
        return_features=False,
    )

    segmentation = (
        get_segmentation_output(
            outputs
        )
    )

    assert "evidence" in segmentation
    assert "alpha" in segmentation
    assert "probabilities" in segmentation
    assert "uncertainty" in segmentation
    assert "reliability" in segmentation

    assert segmentation[
        "evidence"
    ].shape == (
        2,
        3,
        224,
        224,
    )

    assert segmentation[
        "alpha"
    ].shape == (
        2,
        3,
        224,
        224,
    )

    assert segmentation[
        "probabilities"
    ].shape == (
        2,
        3,
        224,
        224,
    )

    assert segmentation[
        "uncertainty"
    ].shape == (
        2,
        3,
        224,
        224,
    )

    assert segmentation[
        "reliability"
    ].shape == (
        2,
        3,
        224,
        224,
    )


# ---------------------------------------------------------------------
# Four-channel input adapter
# ---------------------------------------------------------------------


@torch.no_grad()
def test_four_channel_input(model):
    """
    Verify that four-channel BraTS input can pass through the
    model's input adapter.
    """

    images = torch.randn(
        1,
        4,
        224,
        224,
    )

    outputs = model(
        images,
        return_features=False,
    )

    segmentation = (
        get_segmentation_output(
            outputs
        )
    )

    assert segmentation[
        "probabilities"
    ].shape[1] == 3


# ---------------------------------------------------------------------
# Classification probability normalization
# ---------------------------------------------------------------------


@torch.no_grad()
def test_classification_probability_normalization(
    model,
):
    """Classification probabilities must sum to one."""

    images = torch.randn(
        2,
        3,
        224,
        224,
    )

    outputs = model(
        images,
        return_features=False,
    )

    classification = (
        get_classification_output(
            outputs
        )
    )

    probabilities = classification[
        "probabilities"
    ]

    sums = probabilities.sum(
        dim=1
    )

    assert torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-5,
    )


# ---------------------------------------------------------------------
# Segmentation probability range
# ---------------------------------------------------------------------


@torch.no_grad()
def test_segmentation_probability_range(
    model,
):
    """
    Evidential segmentation probabilities must be in [0, 1].
    """

    images = torch.randn(
        1,
        4,
        224,
        224,
    )

    outputs = model(
        images,
        return_features=False,
    )

    segmentation = (
        get_segmentation_output(
            outputs
        )
    )

    probabilities = segmentation[
        "probabilities"
    ]

    assert torch.all(
        probabilities >= 0
    )

    assert torch.all(
        probabilities <= 1
    )


# ---------------------------------------------------------------------
# Evidential alpha validity
# ---------------------------------------------------------------------


@torch.no_grad()
def test_classification_alpha_validity(
    model,
):
    """
    Dirichlet concentration parameters must be >= 1 because:

        alpha = evidence + 1
    """

    images = torch.randn(
        2,
        3,
        224,
        224,
    )

    outputs = model(
        images,
        return_features=False,
    )

    classification = (
        get_classification_output(
            outputs
        )
    )

    alpha = classification[
        "alpha"
    ]

    assert torch.all(
        alpha >= 1.0
    )


@torch.no_grad()
def test_segmentation_alpha_validity(
    model,
):
    """Segmentation Dirichlet parameters must be >= 1."""

    images = torch.randn(
        1,
        4,
        224,
        224,
    )

    outputs = model(
        images,
        return_features=False,
    )

    segmentation = (
        get_segmentation_output(
            outputs
        )
    )

    alpha = segmentation[
        "alpha"
    ]

    assert torch.all(
        alpha >= 1.0
    )


# ---------------------------------------------------------------------
# Uncertainty range
# ---------------------------------------------------------------------


@torch.no_grad()
def test_classification_uncertainty_range(
    model,
):
    """
    Evidential uncertainty:

        U = K / S

    must lie in [0, 1].
    """

    images = torch.randn(
        2,
        3,
        224,
        224,
    )

    outputs = model(
        images,
        return_features=False,
    )

    classification = (
        get_classification_output(
            outputs
        )
    )

    uncertainty = classification[
        "uncertainty"
    ]

    assert torch.all(
        uncertainty >= 0
    )

    assert torch.all(
        uncertainty <= 1
    )


@torch.no_grad()
def test_segmentation_uncertainty_range(
    model,
):
    """Segmentation uncertainty must lie in [0, 1]."""

    images = torch.randn(
        1,
        4,
        224,
        224,
    )

    outputs = model(
        images,
        return_features=False,
    )

    segmentation = (
        get_segmentation_output(
            outputs
        )
    )

    uncertainty = segmentation[
        "uncertainty"
    ]

    assert torch.all(
        uncertainty >= 0
    )

    assert torch.all(
        uncertainty <= 1
    )


# ---------------------------------------------------------------------
# Reliability range
# ---------------------------------------------------------------------


@torch.no_grad()
def test_classification_reliability_range(
    model,
):
    """Classification reliability must lie in [0, 1]."""

    images = torch.randn(
        2,
        3,
        224,
        224,
    )

    outputs = model(
        images,
        return_features=False,
    )

    classification = (
        get_classification_output(
            outputs
        )
    )

    reliability = classification[
        "reliability"
    ]

    assert torch.all(
        reliability >= 0
    )

    assert torch.all(
        reliability <= 1
    )


@torch.no_grad()
def test_segmentation_reliability_range(
    model,
):
    """Segmentation reliability must lie in [0, 1]."""

    images = torch.randn(
        1,
        4,
        224,
        224,
    )

    outputs = model(
        images,
        return_features=False,
    )

    segmentation = (
        get_segmentation_output(
            outputs
        )
    )

    reliability = segmentation[
        "reliability"
    ]

    assert torch.all(
        reliability >= 0
    )

    assert torch.all(
        reliability <= 1
    )


# ---------------------------------------------------------------------
# Attention and uncertainty maps
# ---------------------------------------------------------------------


@torch.no_grad()
def test_attention_uncertainty_maps(
    model,
):
    """
    Verify that the UCSA modules produce attention and uncertainty
    maps for the multi-scale feature levels.
    """

    images = torch.randn(
        1,
        4,
        224,
        224,
    )

    outputs = model(
        images,
        return_features=False,
    )

    assert (
        "attention_maps"
        in outputs
    )

    assert (
        "uncertainty_maps"
        in outputs
    )

    attention_maps = outputs[
        "attention_maps"
    ]

    uncertainty_maps = outputs[
        "uncertainty_maps"
    ]

    assert len(
        attention_maps
    ) > 0

    assert len(
        attention_maps
    ) == len(
        uncertainty_maps
    )

    for attention, uncertainty in zip(
        attention_maps,
        uncertainty_maps,
    ):

        assert (
            attention.shape
            == uncertainty.shape
        )

        assert torch.all(
            attention >= 0
        )

        assert torch.all(
            attention <= 1
        )

        assert torch.all(
            uncertainty >= 0
        )

        assert torch.all(
            uncertainty <= 1
        )


# ---------------------------------------------------------------------
# Multi-task forward pass
# ---------------------------------------------------------------------


@torch.no_grad()
def test_multitask_forward(model):
    """
    Verify that the model produces both classification and
    segmentation branches from a four-channel input.
    """

    images = torch.randn(
        1,
        4,
        224,
        224,
    )

    outputs = model(
        images,
        return_features=False,
    )

    classification = (
        get_classification_output(
            outputs
        )
    )

    segmentation = (
        get_segmentation_output(
            outputs
        )
    )

    assert (
        classification[
            "probabilities"
        ].shape
        == (1, 4)
    )

    assert (
        segmentation[
            "probabilities"
        ].shape
        == (
            1,
            3,
            224,
            224,
        )
    )


# ---------------------------------------------------------------------
# Feature-return test
# ---------------------------------------------------------------------


@torch.no_grad()
def test_return_features(model):
    """
    Verify that optional intermediate features can be returned.
    """

    images = torch.randn(
        1,
        4,
        224,
        224,
    )

    outputs = model(
        images,
        return_features=True,
    )

    assert isinstance(
        outputs,
        dict,
    )

    assert (
        "classification"
        in outputs
    )

    assert (
        "segmentation"
        in outputs
    )


# ---------------------------------------------------------------------
# Backward propagation
# ---------------------------------------------------------------------


def test_backward_pass():
    """
    Verify that gradients can propagate through the complete
    multi-task model.
    """

    model = (
        UncertaintyAwareMultiTaskModel(
            num_classes=4,
            segmentation_classes=3,
            input_channels=4,
            classification_channels=3,
            pretrained=False,
        )
    )

    model.train()

    images = torch.randn(
        1,
        4,
        224,
        224,
    )

    outputs = model(
        images,
        return_features=False,
    )

    classification = (
        get_classification_output(
            outputs
        )
    )

    segmentation = (
        get_segmentation_output(
            outputs
        )
    )

    classification_loss = (
        classification[
            "probabilities"
        ].mean()
    )

    segmentation_loss = (
        segmentation[
            "probabilities"
        ].mean()
    )

    loss = (
        classification_loss
        + segmentation_loss
    )

    loss.backward()

    has_gradient = False

    for parameter in model.parameters():

        if parameter.requires_grad:

            if parameter.grad is not None:

                has_gradient = True

                assert torch.isfinite(
                    parameter.grad
                ).all()

    assert has_gradient


# ---------------------------------------------------------------------
# Parameter sanity check
# ---------------------------------------------------------------------


def test_model_has_trainable_parameters(
    model,
):
    """Verify that the model contains trainable parameters."""

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert len(
        trainable_parameters
    ) > 0

    total_parameters = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    assert (
        total_parameters > 0
    )