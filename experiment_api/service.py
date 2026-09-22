from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .configuration import build_forecast_config
from .executors import LantaExecutor, LocalExecutor
from .models import ExperimentRequest
from .storage import JsonStore


class InlineSubmitter:
    def submit(self, function: Callable[..., Any], *args: Any, **kwargs: Any):
        return function(*args, **kwargs)


class ExperimentService:
    def __init__(
        self,
        store: JsonStore,
        *,
        local_executor: Any,
        lanta_executor: Any | None,
        submitter: Any | None = None,
    ):
        self.store = store
        self.local_executor = local_executor
        self.lanta_executor = lanta_executor
        self.submitter = submitter or ThreadPoolExecutor(max_workers=2, thread_name_prefix="forecast-api")

    def submit(self, request: ExperimentRequest) -> dict[str, Any]:
        dataset = self.store.get_dataset(request.dataset_id)
        config = build_forecast_config(request, dataset)
        record = self.store.create_run(request.backend, request.dataset_id, request.model_dump(mode="json"))
        config_path = self.store.runs / record["run_id"] / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        self.submitter.submit(self._execute, record["run_id"], request, dataset, config)
        return record

    def _execute(
        self,
        run_id: str,
        request: ExperimentRequest,
        dataset: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        self.store.update_run(run_id, status="running")
        run_directory = self.store.runs / run_id
        try:
            if request.backend == "local":
                result = self.local_executor.execute(config, run_directory)
            else:
                if self.lanta_executor is None:
                    raise RuntimeError("LANTA execution is not configured")
                result = self.lanta_executor.execute(request, dataset, run_id, run_directory)
            self.store.save_result(run_id, result)
        except Exception as exc:
            self.store.update_run(run_id, status="failed", error=f"{type(exc).__name__}: {exc}")

