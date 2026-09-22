"""Load, inspect and prepare classification datasets without data leakage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_breast_cancer, load_digits, load_iris
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent


def _load_frame(spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series, str]:
    source = spec.get("source", "sklearn")
    name = spec.get("name", "digits")
    target_column = spec.get("target_column")

    if source == "sklearn":
        loaders = {"digits": load_digits, "iris": load_iris, "breast_cancer": load_breast_cancer}
        if name not in loaders:
            raise ValueError(f"Unsupported sklearn dataset: {name}")
        bunch = loaders[name](as_frame=True)
        return bunch.data, pd.Series(bunch.target, name="target"), "target"

    if source == "csv":
        path = Path(spec["path"])
        if not path.is_absolute():
            path = ROOT / path
        frame = pd.read_csv(path)
        if not target_column or target_column not in frame.columns:
            raise ValueError("dataset.target_column must name a column in the CSV")
        return frame.drop(columns=[target_column]), frame[target_column], target_column

    if source == "openml":
        bunch = fetch_openml(name=name, version=spec.get("version", "active"), as_frame=True)
        return bunch.data, pd.Series(bunch.target, name=target_column or "target"), target_column or "target"

    raise ValueError(f"Unsupported dataset source: {source}")


def _safe_split(X, y, test_size: float, seed: int):
    counts = pd.Series(y).value_counts()
    stratify = y if len(counts) > 1 and counts.min() >= 2 else None
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)


def prepare_dataset(config: dict[str, Any], output_dir: Path | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    output_dir = output_dir or ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    X, raw_y, target_column = _load_frame(config["dataset"])
    missing_targets = int(raw_y.isna().sum())
    usable = raw_y.notna()
    X, raw_y = X.loc[usable].reset_index(drop=True), raw_y.loc[usable].reset_index(drop=True)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(raw_y.astype(str))
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    transformer = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        remainder="drop",
    )

    training = config.get("training", {})
    seed = int(training.get("random_seed", 42))
    test_size = float(training.get("test_size", 0.2))
    validation_size = float(training.get("validation_size", 0.2))
    X_train_val, X_test, y_train_val, y_test = _safe_split(X, y, test_size, seed)
    relative_validation = validation_size / (1.0 - test_size)
    X_train, X_validation, y_train, y_validation = _safe_split(X_train_val, y_train_val, relative_validation, seed)

    X_train_array = transformer.fit_transform(X_train).astype(float)
    X_validation_array = transformer.transform(X_validation).astype(float)
    X_test_array = transformer.transform(X_test).astype(float)
    arrays = {
        "X_train": X_train_array,
        "X_validation": X_validation_array,
        "X_test": X_test_array,
        "y_train": np.asarray(y_train),
        "y_validation": np.asarray(y_validation),
        "y_test": np.asarray(y_test),
    }

    dataset_name = str(config["dataset"].get("name", "dataset"))
    fingerprint = json.dumps({"name": dataset_name, "columns": X.columns.tolist(), "rows": len(X)}, sort_keys=True)
    profile = {
        "source": config["dataset"].get("source", "sklearn"),
        "name": dataset_name,
        "target_column": target_column,
        "problem_type": "binary_classification" if len(label_encoder.classes_) == 2 else "multiclass_classification",
        "total_rows": int(len(X) + missing_targets),
        "usable_rows": int(len(X)),
        "rows_dropped_for_missing_target": missing_targets,
        "original_features": int(X.shape[1]),
        "input_features": int(X_train_array.shape[1]),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "missing_feature_values": int(X.isna().sum().sum()),
        "missing_values_by_column": {key: int(value) for key, value in X.isna().sum().items() if value},
        "classes": label_encoder.classes_.tolist(),
        "training_samples": int(len(y_train)),
        "validation_samples": int(len(y_validation)),
        "test_samples": int(len(y_test)),
        "dataset_key": hashlib.sha256(fingerprint.encode()).hexdigest()[:16],
    }

    np.savez(output_dir / "prepared_dataset.npz", **arrays)
    joblib.dump({"transformer": transformer, "label_encoder": label_encoder}, output_dir / "preprocessor.joblib")
    (output_dir / "dataset_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return arrays, profile
