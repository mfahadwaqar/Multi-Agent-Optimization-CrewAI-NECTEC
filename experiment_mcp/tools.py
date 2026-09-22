from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from experiment_api.models import ExperimentRequest
from experiment_api.service import ExperimentService


def _without(record: dict[str, Any], *private_keys: str) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in private_keys}


class ExperimentTools:
    """Safe, transport-independent tools backed by ExperimentService."""

    def __init__(self, service: ExperimentService):
        self.service = service

    def register_csv(self, filename: str, csv_content: str) -> dict[str, Any]:
        if not isinstance(csv_content, str):
            raise ValueError("csv_content must be UTF-8 text")
        try:
            content = csv_content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("csv_content must be valid UTF-8 text") from exc
        return _without(self.service.store.save_dataset(filename, content), "path")

    def get_dataset_profile(self, dataset_id: str) -> dict[str, Any]:
        try:
            return _without(self.service.store.get_dataset(dataset_id), "path")
        except KeyError as exc:
            raise KeyError("Dataset not found") from exc

    def submit_forecast(self, configuration: dict[str, Any]) -> dict[str, Any]:
        try:
            request = ExperimentRequest.model_validate(configuration)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc
        if request.backend == "slurm" and self.service.lanta_executor is None:
            raise ValueError("LANTA execution is not configured")
        try:
            return _without(self.service.submit(request), "request", "result_path")
        except KeyError as exc:
            raise KeyError("Dataset not found") from exc

    def get_experiment_status(self, run_id: str) -> dict[str, Any]:
        try:
            return _without(self.service.store.get_run(run_id), "request", "result_path")
        except KeyError as exc:
            raise KeyError("Experiment not found") from exc

    def get_experiment_results(self, run_id: str) -> dict[str, Any]:
        record = self.get_experiment_status(run_id)
        if record["status"] != "completed":
            raise RuntimeError(f"Experiment is {record['status']}")
        return self.service.store.get_result(run_id)

    def get_experiment_metrics(self, run_id: str) -> dict[str, Any]:
        best = self.get_experiment_results(run_id).get("best_result", {})
        metrics = {key: value for key, value in best.items() if "mae" in key or "rmse" in key or "mase" in key}
        if "name" in best:
            metrics = {"model": best["name"], **metrics}
        return metrics

    def get_experiment_report(self, run_id: str) -> str:
        record = self.get_experiment_status(run_id)
        if record["status"] != "completed":
            raise RuntimeError(f"Experiment report is not available while status is {record['status']}")
        return self.service.store.get_report(run_id)
