"""End-to-end forecasting coordinator."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .backtesting import chronological_holdout
from .data_sources import load_source
from .evaluation import choose_best, compare_models, fit_and_test
from .features import build_supervised
from .pipeline import prepare_time_series
from .reporting import build_report
from .seasonality import analyze_seasonality


ROOT = Path(__file__).resolve().parents[1]


def run_forecast(
    config: dict[str, Any],
    *,
    use_agents: bool = False,
    refresh: bool = False,
    root: Path = ROOT,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run local deterministic forecasting; optionally request a derived-data agent review."""
    raw = load_source(config, root, refresh=refresh)
    regular, data_profile = prepare_time_series(raw, config)
    feature_config = {
        **config["forecast"],
        "target_column": config["data"]["target_column"],
        "feature_columns": config["data"].get("feature_columns", []),
    }
    features, target, feature_profile = build_supervised(regular, feature_config)
    profile = {**data_profile, **feature_profile}

    gap = int(config["forecast"]["horizon"])
    development_slice, test_slice = chronological_holdout(
        len(features), config["test_fraction"], gap
    )
    development_x = features.iloc[development_slice]
    development_y = target.iloc[development_slice]
    test_x = features.iloc[test_slice]
    test_y = target.iloc[test_slice]
    profile.update({
        "development_rows": len(development_y),
        "test_rows": len(test_y),
        "test_gap_rows": gap,
        "test_start": test_x.index.min().isoformat(),
        "seasonality_analysis_scope": "development period only",
        "seasonality_analysis": analyze_seasonality(
            regular.loc[: development_x.index.max(), config["data"]["target_column"]],
            config["forecast"].get("seasonality", 1),
        ),
    })

    results = compare_models(development_x, development_y, config)
    best = choose_best(results).copy()
    fitted, test_metrics, predictions = fit_and_test(
        best, development_x, development_y, test_x, test_y, config
    )
    best.update(test_metrics)
    history = {
        "dataset": profile["name"],
        "selection_policy": "lowest expanding-window validation MASE, then MAE",
        "test_policy": "untouched chronological holdout evaluated once after selection",
        "results": results,
        "best_result": best,
    }

    destination = output_dir or root / "results" / "forecasting" / profile["name"]
    artifact_dir = destination / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    (destination / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    joblib.dump(fitted, destination / "best_model.joblib")
    pd.DataFrame({
        "timestamp": test_y.index,
        "actual": test_y.to_numpy(),
        "predicted": predictions,
    }).to_csv(destination / "test_predictions.csv", index=False)

    agent_review = None
    if use_agents:
        from .agent_team import review_forecast
        agent_review = review_forecast(profile, results, best)
    (destination / "summary.md").write_text(
        build_report(profile, history, agent_review), encoding="utf-8"
    )
    return history
