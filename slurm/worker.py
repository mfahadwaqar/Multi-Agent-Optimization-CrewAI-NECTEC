"""Execute one deterministic experiment and persist its result atomically."""

from __future__ import annotations

import json
import os
import re
import tempfile
import traceback
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Iterator


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SECRET_PARTS = ("api_key", "apikey", "password", "secret", "token", "credential")
_CLOUD_ENVIRONMENT_VARIABLES = (
    "OLLAMA_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "CREWAI_API_KEY",
)


def _validate_json(value: Any, location: str = "job") -> None:
    """Require values that Slurminade can transport without pickle."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be finite JSON-serializable data: {exc}") from exc


def _reject_secrets(value: Any, location: str = "job") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in _SECRET_PARTS):
                raise ValueError(f"Secret-like field is not allowed in worker input: {location}.{key}")
            _reject_secrets(nested, f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_secrets(nested, f"{location}[{index}]")


def validate_job(job: dict[str, Any]) -> None:
    if not isinstance(job, dict):
        raise ValueError("Each job must be a JSON object")
    required = ("job_id", "runner", "dataset_id", "configuration_id", "fold_id")
    missing = [key for key in required if key not in job]
    if missing:
        raise ValueError(f"Job is missing required fields: {', '.join(missing)}")
    if not _SAFE_ID.fullmatch(str(job["job_id"])):
        raise ValueError("job_id must contain only letters, digits, dots, dashes or underscores")
    for key in ("dataset_id", "configuration_id", "fold_id"):
        if not isinstance(job[key], (str, int)) or isinstance(job[key], bool):
            raise ValueError(f"{key} must be a string or integer identifier")
    if job["runner"] != "forecasting":
        raise ValueError(f"Unsupported deterministic runner: {job['runner']!r}")
    _reject_secrets(job)
    _validate_json(job)


def _forecasting_runner(job: dict[str, Any]) -> dict[str, Any]:
    module = import_module("forecasting.distributed")
    runner = getattr(module, "run_fold_job", None)
    if not callable(runner):
        raise RuntimeError("forecasting.distributed.run_fold_job is not available")
    return runner(job)


def _resolve_runner(job: dict[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if job["runner"] == "forecasting":
        return _forecasting_runner
    raise ValueError(f"Unsupported deterministic runner: {job['runner']!r}")


@contextmanager
def _without_cloud_credentials() -> Iterator[None]:
    """Keep agent credentials out of the deterministic worker call."""
    previous = {name: os.environ.pop(name) for name in _CLOUD_ENVIRONMENT_VARIABLES if name in os.environ}
    try:
        yield
    finally:
        os.environ.update(previous)


def _atomic_json_write(path: Path, data: dict[str, Any]) -> None:
    _validate_json(data, "result")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def execute_job(
    job: dict[str, Any],
    result_directory: str | Path,
    runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one job and write ``<job_id>.json`` even when the model fails."""
    validate_job(job)
    output_path = Path(result_directory).resolve() / f"{job['job_id']}.json"
    result: dict[str, Any]
    try:
        with _without_cloud_credentials():
            measured = (runner or _resolve_runner(job))(job)
        if not isinstance(measured, dict):
            raise TypeError("Experiment runner must return a dictionary")
        result = {
            "job_id": job["job_id"],
            "dataset_id": job["dataset_id"],
            "configuration_id": job["configuration_id"],
            "fold_id": job["fold_id"],
            "status": "completed",
            "result": measured,
        }
    except Exception as exc:
        result = {
            "job_id": job["job_id"],
            "dataset_id": job["dataset_id"],
            "configuration_id": job["configuration_id"],
            "fold_id": job["fold_id"],
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
    _atomic_json_write(output_path, result)
    return result
