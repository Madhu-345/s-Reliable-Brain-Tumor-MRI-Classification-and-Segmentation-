"""
Classification metrics for the uncertainty-aware multi-task
brain tumor MRI classification framework.

Supported metrics:
    - Accuracy
    - Precision
    - Recall
    - Specificity
    - F1-score
    - AUROC

The functions accept either class-index predictions or probability
predictions where appropriate.
"""

from __future__ import annotations

import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score as sklearn_f1_score,
    roc_auc_score,
    confusion_matrix,
)


def _to_numpy(tensor):
    """Convert a PyTorch tensor or array-like object to NumPy."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def accuracy(
    targets,
    predictions,
) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    targets : array-like
        Ground-truth class indices.

    predictions : array-like
        Predicted class indices.

    Returns
    -------
    float
        Accuracy in the range [0, 1].
    """
    targets = _to_numpy(targets)
    predictions = _to_numpy(predictions)

    return float(
        accuracy_score(targets, predictions)
    )


def precision(
    targets,
    predictions,
    average: str = "macro",
) -> float:
    """
    Compute classification precision.

    Parameters
    ----------
    targets : array-like
        Ground-truth class indices.

    predictions : array-like
        Predicted class indices.

    average : str
        Averaging strategy. Default is ``macro``.

    Returns
    -------
    float
        Precision score.
    """
    targets = _to_numpy(targets)
    predictions = _to_numpy(predictions)

    return float(
        precision_score(
            targets,
            predictions,
            average=average,
            zero_division=0,
        )
    )


def recall(
    targets,
    predictions,
    average: str = "macro",
) -> float:
    """
    Compute classification recall.

    Parameters
    ----------
    targets : array-like
        Ground-truth class indices.

    predictions : array-like
        Predicted class indices.

    average : str
        Averaging strategy. Default is ``macro``.

    Returns
    -------
    float
        Recall score.
    """
    targets = _to_numpy(targets)
    predictions = _to_numpy(predictions)

    return float(
        recall_score(
            targets,
            predictions,
            average=average,
            zero_division=0,
        )
    )


def specificity(
    targets,
    predictions,
    average: str = "macro",
) -> float:
    """
    Compute multi-class specificity.

    Specificity for each class is:

        TN / (TN + FP)

    The returned value is the macro-average across classes.

    Parameters
    ----------
    targets : array-like
        Ground-truth class indices.

    predictions : array-like
        Predicted class indices.

    average : str
        Currently supports ``macro`` or ``weighted``.

    Returns
    -------
    float
        Specificity score.
    """
    targets = _to_numpy(targets)
    predictions = _to_numpy(predictions)

    labels = np.unique(
        np.concatenate([targets, predictions])
    )

    cm = confusion_matrix(
        targets,
        predictions,
        labels=labels,
    )

    specificities = []

    for i in range(len(labels)):
        true_positive = cm[i, i]
        false_positive = cm[:, i].sum() - true_positive
        false_negative = cm[i, :].sum() - true_positive
        true_negative = (
            cm.sum()
            - true_positive
            - false_positive
            - false_negative
        )

        denominator = true_negative + false_positive

        if denominator == 0:
            class_specificity = 0.0
        else:
            class_specificity = (
                true_negative / denominator
            )

        specificities.append(class_specificity)

    specificities = np.asarray(
        specificities,
        dtype=np.float64,
    )

    if average == "weighted":
        class_support = cm.sum(axis=1)
        if class_support.sum() == 0:
            return 0.0

        return float(
            np.average(
                specificities,
                weights=class_support,
            )
        )

    return float(
        np.mean(specificities)
    )


def f1_score(
    targets,
    predictions,
    average: str = "macro",
) -> float:
    """
    Compute classification F1-score.

    Parameters
    ----------
    targets : array-like
        Ground-truth class indices.

    predictions : array-like
        Predicted class indices.

    average : str
        Averaging strategy. Default is ``macro``.

    Returns
    -------
    float
        F1-score.
    """
    targets = _to_numpy(targets)
    predictions = _to_numpy(predictions)

    return float(
        sklearn_f1_score(
            targets,
            predictions,
            average=average,
            zero_division=0,
        )
    )


def auroc(
    targets,
    probabilities,
    num_classes: int | None = None,
) -> float:
    """
    Compute multi-class AUROC using one-vs-rest evaluation.

    Parameters
    ----------
    targets : array-like
        Ground-truth class indices of shape [N].

    probabilities : array-like
        Predicted class probabilities of shape [N, K].

    num_classes : int, optional
        Number of classes. If omitted, inferred from probabilities.

    Returns
    -------
    float
        Macro one-vs-rest AUROC.

    Notes
    -----
    AUROC requires probability scores rather than hard class
    predictions.
    """
    targets = _to_numpy(targets)
    probabilities = _to_numpy(probabilities)

    if probabilities.ndim != 2:
        raise ValueError(
            "Probabilities must have shape [N, K]."
        )

    if targets.ndim != 1:
        targets = targets.reshape(-1)

    if probabilities.shape[0] != targets.shape[0]:
        raise ValueError(
            "Number of probability samples and targets must match."
        )

    if num_classes is None:
        num_classes = probabilities.shape[1]

    if probabilities.shape[1] != num_classes:
        raise ValueError(
            "num_classes does not match the probability dimension."
        )

    try:
        return float(
            roc_auc_score(
                targets,
                probabilities,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        # AUROC is undefined when the evaluation set does not
        # contain sufficient class variation.
        return float("nan")


def classification_metrics(
    targets,
    predictions,
    probabilities=None,
) -> dict:
    """
    Compute the complete classification metric set.

    Parameters
    ----------
    targets : array-like
        Ground-truth class indices.

    predictions : array-like
        Predicted class indices.

    probabilities : array-like, optional
        Predicted class probabilities. Required for AUROC.

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall,
        specificity, F1, and AUROC.
    """
    results = {
        "accuracy": accuracy(
            targets,
            predictions,
        ),
        "precision": precision(
            targets,
            predictions,
        ),
        "recall": recall(
            targets,
            predictions,
        ),
        "specificity": specificity(
            targets,
            predictions,
        ),
        "f1": f1_score(
            targets,
            predictions,
        ),
    }

    if probabilities is not None:
        results["auroc"] = auroc(
            targets,
            probabilities,
        )
    else:
        results["auroc"] = float("nan")

    return results


__all__ = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1_score",
    "auroc",
    "classification_metrics",
]