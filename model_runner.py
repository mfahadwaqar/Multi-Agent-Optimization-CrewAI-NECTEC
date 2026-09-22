"""Train real baseline and neural-network classifiers."""

import time
from pathlib import Path
from typing import Any

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neural_network import MLPClassifier


def _metrics(model, X, y) -> dict[str, float]:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "loss": float(log_loss(y, probabilities, labels=model.classes_)),
    }


def _fit(name: str, model, arrays: dict, kind: str, configuration: dict | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        model.fit(arrays["X_train"], arrays["y_train"])
        validation = _metrics(model, arrays["X_validation"], arrays["y_validation"])
        return {
            "name": name,
            "kind": kind,
            "status": "completed",
            "configuration": configuration or {},
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "validation_loss": validation["loss"],
            "duration_seconds": round(time.perf_counter() - started, 3),
            "model": model,
        }
    except Exception as exc:
        return {"name": name, "kind": kind, "status": "failed", "error": str(exc), "configuration": configuration or {}}


def run_baselines(arrays: dict, seed: int) -> list[dict[str, Any]]:
    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
    }
    return [_fit(name, model, arrays, "baseline") for name, model in models.items()]


def run_neural_trial(arrays: dict, configuration: dict[str, Any], seed: int, number: int) -> dict[str, Any]:
    model = MLPClassifier(
        hidden_layer_sizes=tuple(configuration["layers"]),
        activation=configuration["activation"],
        solver=configuration["optimizer"],
        learning_rate_init=float(configuration["learning_rate"]),
        batch_size=int(configuration["batch_size"]),
        max_iter=int(configuration["epochs"]),
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        random_state=seed,
    )
    return _fit(f"neural_trial_{number:03d}", model, arrays, "neural_network", configuration)


def evaluate_and_save(result: dict[str, Any], arrays: dict, output_dir: Path) -> dict[str, Any]:
    final = {key: value for key, value in result.items() if key != "model"}
    test = _metrics(result["model"], arrays["X_test"], arrays["y_test"])
    final.update({f"test_{key}": value for key, value in test.items()})
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["model"], output_dir / "best_model.joblib")
    return final
