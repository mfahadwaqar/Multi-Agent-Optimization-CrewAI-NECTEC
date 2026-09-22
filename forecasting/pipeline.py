"""Regularize timestamps and profile time-series data without future leakage."""

from typing import Any

import pandas as pd


def prepare_time_series(raw: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = config["data"]
    timestamp = source["timestamp_column"]
    target = source["target_column"]
    if timestamp not in raw or target not in raw:
        raise ValueError(f"Required columns are missing: {timestamp}, {target}")

    columns = [timestamp, target] + [name for name in source.get("feature_columns", []) if name in raw and name != target]
    frame = raw[columns].copy()
    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="coerce", dayfirst=source.get("dayfirst", False))
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    for column in source.get("feature_columns", []):
        if column in frame and column != target:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid_timestamps = int(frame[timestamp].isna().sum())
    frame = frame.dropna(subset=[timestamp]).sort_values(timestamp)
    duplicate_timestamps = int(frame[timestamp].duplicated(keep="last").sum())
    frame = frame.drop_duplicates(timestamp, keep="last").set_index(timestamp)
    observed_rows = len(frame)
    regular_index = pd.date_range(frame.index.min(), frame.index.max(), freq=source["frequency"])
    frame = frame.reindex(regular_index)
    frame.index.name = timestamp
    missing_timestamps = int(len(frame) - observed_rows)
    missing_by_column = {name: int(count) for name, count in frame.isna().sum().items() if count}
    profile = {
        "name": source["name"],
        "source": source["source"],
        "timestamp_column": timestamp,
        "target_column": target,
        "frequency": source["frequency"],
        "observed_rows": int(observed_rows),
        "regularized_rows": int(len(frame)),
        "start": frame.index.min().isoformat(),
        "end": frame.index.max().isoformat(),
        "invalid_timestamps": invalid_timestamps,
        "duplicate_timestamps": duplicate_timestamps,
        "missing_timestamps": missing_timestamps,
        "missing_values_by_column": missing_by_column,
        "causal_imputation": "forward fill for model inputs; missing future labels are dropped",
    }
    return frame, profile
