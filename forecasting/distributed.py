"""Pure worker entry point suitable for Slurminade distribution."""

import json
from pathlib import Path

import numpy as np

from .backtesting import chronological_holdout, expanding_window_folds
from .data_sources import load_source
from .features import build_supervised
from .metrics import score
from .models import BASELINES, baseline_predictions, build_model, model_specs
from .pipeline import prepare_time_series


def _from_shared_config(job_spec: dict) -> dict:
    config_path = Path(job_spec["config_path"]).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = Path(job_spec.get("project_root", config_path.parent.parent)).resolve()
    if config["data"]["name"] != str(job_spec["dataset_id"]):
        raise ValueError("dataset_id does not match the shared configuration")
    if config["data"]["source"] == "usgs":
        cache = Path(config["data"]["cache_path"])
        cache = cache if cache.is_absolute() else project_root / cache
        if not cache.exists():
            raise FileNotFoundError("USGS cache must be prepared by the controller before SLURM dispatch")

    raw = load_source(config, project_root, refresh=False)
    regular, _ = prepare_time_series(raw, config)
    feature_config = {
        **config["forecast"],
        "target_column": config["data"]["target_column"],
        "feature_columns": config["data"].get("feature_columns", []),
    }
    features, target, _ = build_supervised(regular, feature_config)
    development, _ = chronological_holdout(
        len(features), config["test_fraction"], int(config["forecast"]["horizon"])
    )
    features, target = features.iloc[development], target.iloc[development]
    folds = expanding_window_folds(
        len(features), config["validation"]["folds"], config["validation"]["window_size"],
        int(config["forecast"]["horizon"]), config["validation"]["minimum_train_size"],
    )
    fold_number = int(job_spec["fold_id"])
    fold_index = fold_number - 1 if fold_number >= 1 else 0
    if fold_index >= len(folds):
        raise ValueError(f"fold_id {fold_number} is outside the configured fold range")
    fold = folds[fold_index]

    configuration_id = str(job_spec["configuration_id"])
    candidates = model_specs(config)
    spec = next(
        (item for item in candidates if str(item.get("id", item["name"])) == configuration_id),
        None,
    )
    if spec is None:
        raise ValueError(f"configuration_id is not defined in the shared config: {configuration_id}")
    train_x, validation_x = features.iloc[fold.train], features.iloc[fold.validation]
    train_y, validation_y = target.iloc[fold.train], target.iloc[fold.validation]
    if spec["name"] in BASELINES:
        prediction = baseline_predictions(spec["name"], validation_x)
    else:
        columns = [name for name in features.columns if not name.startswith("__")]
        model = build_model(spec["name"], spec["params"], int(config.get("random_seed", 42)))
        model.fit(train_x[columns], train_y)
        prediction = model.predict(validation_x[columns])
    return {
        "job_id": job_spec["job_id"],
        "model_name": spec["name"],
        "fold": fold_number,
        "train_rows": len(train_y),
        "validation_rows": len(validation_y),
        "metrics": score(validation_y, prediction, train_y, int(config["forecast"].get("seasonality", 1))),
    }


def run_fold_job(job_spec: dict) -> dict:
    """Run one fold from a shared config or JSON-safe arrays, without agents."""
    if "config_path" in job_spec:
        return _from_shared_config(job_spec)
    name = job_spec["model_name"]
    train_x = np.asarray(job_spec["train_features"], dtype=float)
    validation_x = np.asarray(job_spec["validation_features"], dtype=float)
    train_y = np.asarray(job_spec["train_target"], dtype=float)
    validation_y = np.asarray(job_spec["validation_target"], dtype=float)
    if name in BASELINES:
        baseline_index = int(job_spec["baseline_column_index"])
        prediction = validation_x[:, baseline_index]
        fitted = None
    else:
        fitted = build_model(name, job_spec.get("model_params", {}), int(job_spec.get("random_seed", 42)))
        fitted.fit(train_x, train_y)
        prediction = fitted.predict(validation_x)
    return {
        "job_id": job_spec["job_id"],
        "model_name": name,
        "metrics": score(validation_y, prediction, train_y, int(job_spec.get("seasonality", 1))),
    }
