"""
Compute selective prediction results for the uncertainty-aware
brain tumor MRI classification framework.

This script evaluates reliability-based selective prediction at
multiple thresholds and produces a table containing:

    - Reliability threshold
    - Coverage
    - Selective accuracy
    - Selective risk

The implementation follows the manuscript definitions:

    Coverage(tau)
        = N_accepted / N_total * 100

    Accuracy_sel(tau)
        = N_correct,accepted / N_accepted * 100

    Risk(tau)
        = N_incorrect,accepted / N_accepted * 100

The script can be used with saved prediction/reliability values
stored in a CSV file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
]


def compute_selective_metrics(
    correct: np.ndarray,
    reliability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """
    Compute selective prediction metrics at a reliability threshold.

    Parameters
    ----------
    correct : np.ndarray
        Boolean array indicating whether each prediction is correct.

    reliability : np.ndarray
        Reliability score for each prediction, expected to lie in [0, 1].

    threshold : float
        Reliability threshold used for acceptance.

    Returns
    -------
    dict[str, float]
        Threshold, coverage, selective accuracy, and selective risk.
    """

    correct = np.asarray(correct).astype(bool)
    reliability = np.asarray(reliability, dtype=float)

    if correct.ndim != 1 or reliability.ndim != 1:
        raise ValueError(
            "correct and reliability must be one-dimensional arrays."
        )

    if len(correct) != len(reliability):
        raise ValueError(
            "correct and reliability must contain the same number "
            "of samples."
        )

    if len(correct) == 0:
        raise ValueError("Input arrays must not be empty.")

    if np.any(~np.isfinite(reliability)):
        raise ValueError(
            "Reliability contains NaN or infinite values."
        )

    if np.any((reliability < 0.0) | (reliability > 1.0)):
        raise ValueError(
            "Reliability values must be within [0, 1]."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be within [0, 1]."
        )

    accepted = reliability >= threshold

    n_total = len(correct)
    n_accepted = int(accepted.sum())

    coverage_value = (
        100.0 * n_accepted / n_total
    )

    if n_accepted == 0:
        selective_accuracy = float("nan")
        selective_risk = float("nan")
    else:
        n_correct_accepted = int(
            np.logical_and(
                accepted,
                correct,
            ).sum()
        )

        selective_accuracy = (
            100.0
            * n_correct_accepted
            / n_accepted
        )

        selective_risk = (
            100.0
            - selective_accuracy
        )

    return {
        "threshold": float(threshold),
        "coverage": coverage_value,
        "selective_accuracy": selective_accuracy,
        "selective_risk": selective_risk,
    }


def compute_selective_table(
    correct: np.ndarray,
    reliability: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Compute a complete selective prediction table.

    Parameters
    ----------
    correct : np.ndarray
        Boolean correctness indicators.

    reliability : np.ndarray
        Reliability scores.

    thresholds : list[float], optional
        Reliability thresholds.

    Returns
    -------
    pandas.DataFrame
        Selective prediction results.
    """

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    rows = [
        compute_selective_metrics(
            correct=correct,
            reliability=reliability,
            threshold=threshold,
        )
        for threshold in thresholds
    ]

    return pd.DataFrame(rows)


def load_prediction_file(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load prediction correctness and reliability from CSV.

    Required columns
    ----------------
    correct
        1/0 or True/False indicating prediction correctness.

    reliability
        Reliability score in [0, 1].
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    dataframe = pd.read_csv(path)

    required_columns = {
        "correct",
        "reliability",
    }

    missing = required_columns.difference(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    correct = dataframe["correct"].to_numpy()

    if correct.dtype.kind in {"U", "S", "O"}:
        correct = (
            dataframe["correct"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": 1,
                    "false": 0,
                    "1": 1,
                    "0": 0,
                }
            )
            .to_numpy()
        )

        if np.any(pd.isna(correct)):
            raise ValueError(
                "Column 'correct' contains unsupported values."
            )

    correct = np.asarray(correct).astype(bool)

    reliability = dataframe[
        "reliability"
    ].to_numpy(dtype=float)

    return correct, reliability


def save_selective_table(
    table: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save the selective prediction table as CSV.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        output_path,
        index=False,
        float_format="%.4f",
    )


def parse_thresholds(
    value: str,
) -> list[float]:
    """
    Parse comma-separated thresholds.

    Example
    -------
    "0.50,0.60,0.70,0.80,0.90,0.95"
    """

    thresholds = [
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    ]

    if not thresholds:
        raise ValueError(
            "At least one threshold must be provided."
        )

    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Invalid threshold: {threshold}. "
                "Thresholds must be within [0, 1]."
            )

    return thresholds


def main() -> None:
    """
    Command-line entry point.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Compute reliability-based selective "
            "prediction metrics."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "CSV file containing 'correct' and "
            "'reliability' columns."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/selective_table.csv",
        help=(
            "Output CSV path. "
            "Default: outputs/selective_table.csv"
        ),
    )

    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.50,0.60,0.70,0.80,0.90,0.95",
        help=(
            "Comma-separated reliability thresholds."
        ),
    )

    args = parser.parse_args()

    thresholds = parse_thresholds(
        args.thresholds
    )

    correct, reliability = load_prediction_file(
        args.input
    )

    table = compute_selective_table(
        correct=correct,
        reliability=reliability,
        thresholds=thresholds,
    )

    save_selective_table(
        table,
        args.output,
    )

    print("\nSelective Prediction Table")
    print("=" * 72)
    print(
        table.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print(
        f"\nSaved to: {Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()