"""Forecast metrics. All metrics are lower-is-better."""

import numpy as np


def mae(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mase(actual, predicted, training_target, seasonality: int = 1) -> float:
    """Mean absolute scaled error using only the fold's training target."""
    training_target = np.asarray(training_target, dtype=float)
    seasonality = max(1, int(seasonality))
    if len(training_target) <= seasonality:
        seasonality = 1
    scale = np.mean(np.abs(training_target[seasonality:] - training_target[:-seasonality]))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = np.mean(np.abs(np.diff(training_target))) if len(training_target) > 1 else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return mae(actual, predicted) / float(scale)


def score(actual, predicted, training_target, seasonality: int = 1) -> dict[str, float]:
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mase": mase(actual, predicted, training_target, seasonality),
    }
