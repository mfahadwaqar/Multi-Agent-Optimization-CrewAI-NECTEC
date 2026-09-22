"""Forecast baselines and compact learned-model registry."""

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASELINES = {"persistence", "seasonal_naive"}


def build_model(name: str, params: dict[str, Any], random_seed: int):
    if name == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=float(params.get("alpha", 1.0))),
        )
    if name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=int(params.get("n_estimators", 150)),
                max_depth=params.get("max_depth"),
                min_samples_leaf=int(params.get("min_samples_leaf", 2)),
                random_state=random_seed,
                n_jobs=-1,
            ),
        )
    raise ValueError(f"Unknown forecasting model: {name}")


def baseline_predictions(name: str, features) -> np.ndarray:
    column = "__persistence__" if name == "persistence" else "__seasonal_naive__"
    return features[column].to_numpy(dtype=float)


def model_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"id": "persistence", "name": "persistence", "kind": "baseline", "params": {}},
        {"id": "seasonal_naive", "name": "seasonal_naive", "kind": "baseline", "params": {}},
        *[
            {
                "id": item.get("id", item["name"]),
                "name": item["name"],
                "kind": "learned",
                "params": item.get("params", {}),
            }
            for item in config.get("models", [])
        ],
    ]
