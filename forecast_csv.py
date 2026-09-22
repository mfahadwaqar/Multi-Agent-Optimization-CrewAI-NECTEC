"""Run the forecasting workflow on a user-supplied time-series CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import pandas as pd
from dotenv import load_dotenv

from experiment_api.configuration import build_forecast_config
from experiment_api.models import ExperimentRequest, ModelRequest
from forecasting.workflow import run_forecast


ROOT = Path(__file__).resolve().parent


def _dataset_id(path: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", path.stem).strip("-.") or "dataset"
    return f"csv-{name}"[:64]


def build_request_from_csv(
    path: str | Path,
    *,
    timestamp_column: str,
    target_column: str,
    feature_columns: Sequence[str] | None = None,
    separator: str = ",",
    encoding: str = "utf-8",
    frequency: str = "D",
    horizon: int = 1,
    seasonality: int = 1,
    lags: Sequence[int] = (1,),
    rolling_windows: Sequence[int] = (),
    calendar_features: Sequence[str] = (),
    folds: int = 3,
    validation_size: int = 30,
    minimum_train_size: int = 100,
    test_fraction: float = 0.2,
    models: Sequence[str] = ("ridge", "random_forest"),
) -> tuple[ExperimentRequest, dict]:
    """Inspect a CSV and build the same validated request used by the API."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"CSV file not found: {source}")
    try:
        frame = pd.read_csv(source, sep=separator, encoding=encoding)
    except Exception as exc:
        raise ValueError(f"CSV could not be parsed: {exc}") from exc
    columns = [str(column) for column in frame.columns]
    missing = [column for column in (timestamp_column, target_column) if column not in columns]
    if missing:
        raise ValueError(f"Required columns not found in CSV: {', '.join(missing)}")
    if feature_columns is None:
        selected_features = [
            str(column)
            for column in frame.select_dtypes(include="number").columns
            if str(column) not in {timestamp_column, target_column}
        ]
    else:
        selected_features = list(feature_columns)
        unknown = [column for column in selected_features if column not in columns]
        if unknown:
            raise ValueError(f"Feature columns not found in CSV: {', '.join(unknown)}")
    dataset = {
        "dataset_id": _dataset_id(source),
        "columns": columns,
        "path": str(source),
        "row_count": int(len(frame)),
        "missing_values": {str(column): int(value) for column, value in frame.isna().sum().items()},
    }
    request = ExperimentRequest(
        dataset_id=dataset["dataset_id"],
        timestamp_column=timestamp_column,
        target_column=target_column,
        feature_columns=selected_features,
        frequency=frequency,
        horizon=horizon,
        seasonality=seasonality,
        lags=list(lags),
        rolling_windows=list(rolling_windows),
        calendar_features=list(calendar_features),
        folds=folds,
        validation_size=validation_size,
        minimum_train_size=minimum_train_size,
        test_fraction=test_fraction,
        models=[ModelRequest(name=name) for name in models],
    )
    return request, dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forecast a numeric target in any standard time-series CSV"
    )
    parser.add_argument("--data", required=True, type=Path, help="Path to a CSV file")
    parser.add_argument("--timestamp", required=True, help="Timestamp column name")
    parser.add_argument("--target", required=True, help="Numeric target column name")
    parser.add_argument(
        "--features",
        nargs="*",
        help="Optional feature columns; omit to use all other numeric columns",
    )
    parser.add_argument("--separator", default=",")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--frequency", default="D", help="Pandas frequency such as H, D or 10min")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--seasonality", type=int, default=1)
    parser.add_argument("--lags", nargs="+", type=int, default=[1])
    parser.add_argument("--rolling-windows", nargs="*", type=int, default=[])
    parser.add_argument(
        "--calendar-features",
        nargs="*",
        choices=("hour", "day_of_week", "day_of_year"),
        default=[],
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--validation-size", type=int, default=30)
    parser.add_argument("--minimum-train-size", type=int, default=100)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("ridge", "random_forest"),
        default=["ridge", "random_forest"],
    )
    parser.add_argument("--use-agents", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    load_dotenv(ROOT / ".env", override=False)
    request, dataset = build_request_from_csv(
        arguments.data,
        timestamp_column=arguments.timestamp,
        target_column=arguments.target,
        feature_columns=arguments.features,
        separator=arguments.separator,
        encoding=arguments.encoding,
        frequency=arguments.frequency,
        horizon=arguments.horizon,
        seasonality=arguments.seasonality,
        lags=arguments.lags,
        rolling_windows=arguments.rolling_windows,
        calendar_features=arguments.calendar_features,
        folds=arguments.folds,
        validation_size=arguments.validation_size,
        minimum_train_size=arguments.minimum_train_size,
        test_fraction=arguments.test_fraction,
        models=arguments.models,
    )
    config = build_forecast_config(request, dataset)
    config["data"].update(separator=arguments.separator, encoding=arguments.encoding)
    history = run_forecast(config, use_agents=arguments.use_agents)
    best = history["best_result"]
    destination = ROOT / "results" / "forecasting" / config["data"]["name"]
    print(f"Dataset rows: {dataset['row_count']}")
    print(f"Features: {', '.join(request.feature_columns) or '(target history only)'}")
    print(f"Best model: {best['name']}")
    print(f"Test MAE: {best['test_mae']:.6f}")
    print(f"Test RMSE: {best['test_rmse']:.6f}")
    print(f"Test MASE: {best['test_mase']:.6f}")
    print(f"Report: {destination / 'summary.md'}")


if __name__ == "__main__":
    main()
