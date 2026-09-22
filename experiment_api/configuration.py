from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from forecasting.models import model_specs

from .models import ExperimentRequest


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return slug[:80] or "dataset"


def _validated_models(request: ExperimentRequest) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for model in request.models:
        if model.name in seen:
            raise ValueError("Model names must be unique")
        seen.add(model.name)
        params = dict(model.params)
        if model.name == "ridge":
            alpha = float(params.get("alpha", 1.0))
            if not 0.000001 <= alpha <= 1_000_000:
                raise ValueError("ridge alpha is outside the allowed range")
            params = {"alpha": alpha}
        else:
            estimators = int(params.get("n_estimators", 150))
            depth = params.get("max_depth", 12)
            leaf = int(params.get("min_samples_leaf", 2))
            if not 10 <= estimators <= 2_000 or not 1 <= leaf <= 1_000:
                raise ValueError("random forest parameters are outside the allowed range")
            if depth is not None and not 1 <= int(depth) <= 100:
                raise ValueError("random forest max_depth is outside the allowed range")
            params = {"n_estimators": estimators, "max_depth": depth, "min_samples_leaf": leaf}
        output.append({"name": model.name, "params": params})
    return output


def build_forecast_config(
    request: ExperimentRequest,
    dataset: dict[str, Any],
    *,
    data_path: str | None = None,
) -> dict[str, Any]:
    columns = set(dataset["columns"])
    selected = [request.timestamp_column, request.target_column, *request.feature_columns]
    missing = [column for column in selected if column not in columns]
    if missing:
        raise ValueError(f"Columns are not present in the dataset: {', '.join(missing)}")
    if len(set(request.feature_columns)) != len(request.feature_columns):
        raise ValueError("Feature columns must be unique")
    if request.target_column in request.feature_columns or request.timestamp_column in request.feature_columns:
        raise ValueError("Feature columns cannot include the timestamp or target column")
    return {
        "data": {
            "source": "csv",
            "name": f"api_{_safe_slug(dataset['dataset_id'])}",
            "path": data_path or dataset["path"],
            "timestamp_column": request.timestamp_column,
            "target_column": request.target_column,
            "feature_columns": request.feature_columns,
            "frequency": request.frequency,
            "dayfirst": False,
        },
        "forecast": {
            "horizon": request.horizon,
            "seasonality": request.seasonality,
            "lags": request.lags,
            "rolling_windows": request.rolling_windows,
            "calendar_features": request.calendar_features,
        },
        "validation": {
            "folds": request.folds,
            "window_size": request.validation_size,
            "minimum_train_size": request.minimum_train_size,
        },
        "test_fraction": request.test_fraction,
        "random_seed": request.random_seed,
        "models": _validated_models(request),
    }


def validate_dataset_values(request: ExperimentRequest, dataset: dict[str, Any]) -> None:
    """Validate configured semantic column types before queueing work."""
    selected = [request.timestamp_column, request.target_column, *request.feature_columns]
    try:
        frame = pd.read_csv(Path(dataset["path"]), usecols=selected)
    except ValueError:
        # Column membership is reported consistently by build_forecast_config.
        return
    timestamps = frame[request.timestamp_column].dropna()
    if timestamps.empty or pd.to_datetime(timestamps, errors="coerce", format="mixed").isna().any():
        raise ValueError("Timestamp column must contain parseable timestamps")
    for column in [request.target_column, *request.feature_columns]:
        values = frame[column].dropna()
        if values.empty or pd.to_numeric(values, errors="coerce").isna().any():
            raise ValueError(f"Column '{column}' must contain numeric values")


def build_manifest(
    config: dict[str, Any],
    config_path: str,
    project_root: str,
    run_id: str,
) -> dict[str, Any]:
    jobs = []
    for spec in model_specs(config):
        for fold in range(1, int(config["validation"]["folds"]) + 1):
            jobs.append({
                "job_id": f"{_safe_slug(run_id)}-{_safe_slug(spec['id'])}-fold-{fold}",
                "runner": "forecasting",
                "dataset_id": config["data"]["name"],
                "configuration_id": spec["id"],
                "fold_id": fold,
                "config_path": config_path,
                "project_root": project_root,
            })
    return {"bundle_size": 1, "jobs": jobs}

