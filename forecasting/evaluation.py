"""Walk-forward model comparison and untouched-test evaluation."""

from typing import Any

import numpy as np

from .backtesting import expanding_window_folds
from .metrics import score
from .models import BASELINES, baseline_predictions, build_model, model_specs


def _model_columns(features) -> list[str]:
    return [name for name in features.columns if not name.startswith("__")]


def compare_models(features, target, config: dict[str, Any]) -> list[dict[str, Any]]:
    validation = config["validation"]
    folds = expanding_window_folds(
        len(features),
        validation["folds"],
        validation["window_size"],
        int(config["forecast"]["horizon"]),
        validation["minimum_train_size"],
    )
    columns = _model_columns(features)
    results = []
    for spec in model_specs(config):
        fold_results = []
        for fold_number, fold in enumerate(folds, start=1):
            train_x = features.iloc[fold.train]
            validation_x = features.iloc[fold.validation]
            train_y = target.iloc[fold.train]
            validation_y = target.iloc[fold.validation]
            if spec["name"] in BASELINES:
                prediction = baseline_predictions(spec["name"], validation_x)
            else:
                model = build_model(spec["name"], spec["params"], config.get("random_seed", 42))
                model.fit(train_x[columns], train_y)
                prediction = model.predict(validation_x[columns])
            fold_results.append({
                "fold": fold_number,
                "train_rows": len(train_y),
                "validation_rows": len(validation_y),
                **score(validation_y, prediction, train_y, config["forecast"].get("seasonality", 1)),
            })
        results.append({
            **spec,
            "status": "completed",
            "validation_mae": float(np.mean([item["mae"] for item in fold_results])),
            "validation_rmse": float(np.mean([item["rmse"] for item in fold_results])),
            "validation_mase": float(np.mean([item["mase"] for item in fold_results])),
            "folds": fold_results,
        })
    return results


def choose_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in results if item.get("status") == "completed"]
    if not completed:
        raise ValueError("No completed forecasting results")
    return min(completed, key=lambda item: (item["validation_mase"], item["validation_mae"]))


def fit_and_test(best: dict[str, Any], train_x, train_y, test_x, test_y, config: dict[str, Any]):
    columns = _model_columns(train_x)
    if best["name"] in BASELINES:
        fitted = {"name": best["name"], "kind": "baseline"}
        prediction = baseline_predictions(best["name"], test_x)
    else:
        fitted = build_model(best["name"], best["params"], config.get("random_seed", 42))
        fitted.fit(train_x[columns], train_y)
        prediction = fitted.predict(test_x[columns])
    metrics = score(test_y, prediction, train_y, config["forecast"].get("seasonality", 1))
    return fitted, {f"test_{name}": value for name, value in metrics.items()}, prediction
