"""Causal feature construction for fixed-horizon forecasting."""

from typing import Any

import numpy as np
import pandas as pd


def build_supervised(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    target = config["target_column"]
    horizon = int(config["horizon"])
    seasonality = int(config.get("seasonality", 1))
    lags = sorted({int(value) for value in config.get("lags", [1]) if int(value) > 0})
    windows = sorted({int(value) for value in config.get("rolling_windows", []) if int(value) > 1})

    # Labels remain unfilled: a missing future observation is never invented.
    labels = frame[target].shift(-horizon).rename("forecast_target")
    causal = frame.ffill()
    output = pd.DataFrame(index=frame.index)
    output[f"{target}_current"] = causal[target]
    for lag in lags:
        output[f"{target}_lag_{lag}"] = causal[target].shift(lag)
    for window in windows:
        history = causal[target].shift(1).rolling(window, min_periods=window)
        output[f"{target}_rolling_mean_{window}"] = history.mean()
        output[f"{target}_rolling_std_{window}"] = history.std().fillna(0.0)

    for column in config.get("feature_columns", []):
        if column != target and column in causal:
            output[column] = pd.to_numeric(causal[column], errors="coerce")

    calendar = set(config.get("calendar_features", []))
    if "hour" in calendar:
        output["hour_sin"] = np.sin(2 * np.pi * output.index.hour / 24)
        output["hour_cos"] = np.cos(2 * np.pi * output.index.hour / 24)
    if "day_of_week" in calendar:
        output["weekday_sin"] = np.sin(2 * np.pi * output.index.dayofweek / 7)
        output["weekday_cos"] = np.cos(2 * np.pi * output.index.dayofweek / 7)
    if "day_of_year" in calendar:
        output["year_sin"] = np.sin(2 * np.pi * output.index.dayofyear / 365.25)
        output["year_cos"] = np.cos(2 * np.pi * output.index.dayofyear / 365.25)

    # Prediction-only columns implement baselines and are kept out of learned models.
    output["__persistence__"] = causal[target]
    seasonal_lag = max(0, seasonality - horizon)
    output["__seasonal_naive__"] = causal[target].shift(seasonal_lag) if seasonal_lag else causal[target]

    combined = output.join(labels).replace([np.inf, -np.inf], np.nan).dropna()
    y = combined.pop("forecast_target")
    profile = {
        "horizon": horizon,
        "seasonality": seasonality,
        "supervised_rows": int(len(combined)),
        "model_features": [name for name in combined.columns if not name.startswith("__")],
        "feature_count": sum(not name.startswith("__") for name in combined.columns),
    }
    return combined, y, profile
