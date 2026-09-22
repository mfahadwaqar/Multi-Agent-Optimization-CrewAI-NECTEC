from __future__ import annotations

import io
import json
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


_GENERATED_ID = re.compile(r"^[0-9a-f]{32}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonStore:
    def __init__(self, root: Path, max_upload_bytes: int = 25 * 1024 * 1024):
        self.root = Path(root).resolve()
        self.datasets = self.root / "datasets"
        self.runs = self.root / "runs"
        self.datasets.mkdir(parents=True, exist_ok=True)
        self.runs.mkdir(parents=True, exist_ok=True)
        self.max_upload_bytes = int(max_upload_bytes)
        self._lock = threading.RLock()
        self._recover_interrupted_runs()

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise KeyError(path.parent.name)
        return json.loads(path.read_text(encoding="utf-8"))

    def save_dataset(self, filename: str, content: bytes) -> dict[str, Any]:
        if not filename.lower().endswith(".csv"):
            raise ValueError("Only CSV uploads are supported")
        if not content:
            raise ValueError("CSV upload is empty")
        if len(content) > self.max_upload_bytes:
            raise ValueError(f"CSV exceeds the maximum upload size of {self.max_upload_bytes} bytes")
        try:
            frame = pd.read_csv(io.BytesIO(content))
        except Exception as exc:
            raise ValueError(f"CSV could not be parsed: {exc}") from exc
        if frame.empty or not list(frame.columns):
            raise ValueError("CSV must contain a header and at least one data row")
        dataset_id = uuid.uuid4().hex
        directory = self.datasets / dataset_id
        directory.mkdir(parents=True, exist_ok=False)
        source = directory / "source.csv"
        source.write_bytes(content)
        record = {
            "dataset_id": dataset_id,
            "original_filename": Path(filename).name,
            "path": str(source),
            "row_count": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
            "missing_values": {str(column): int(count) for column, count in frame.isna().sum().items()},
            "created_at": _now(),
        }
        self._write_json(directory / "record.json", record)
        return record

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        if not _GENERATED_ID.fullmatch(dataset_id):
            raise KeyError(dataset_id)
        return self._read_json(self.datasets / dataset_id / "record.json")

    def create_run(self, backend: str, dataset_id: str, request: dict[str, Any]) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        record = {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "backend": backend,
            "status": "queued",
            "request": request,
            "created_at": _now(),
            "updated_at": _now(),
            "error": None,
            "result_path": None,
        }
        self._write_json(self.runs / run_id / "record.json", record)
        return record

    def get_run(self, run_id: str) -> dict[str, Any]:
        if not _GENERATED_ID.fullmatch(run_id):
            raise KeyError(run_id)
        return self._read_json(self.runs / run_id / "record.json")

    def update_run(self, run_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            record = self.get_run(run_id)
            record.update(updates)
            record["updated_at"] = _now()
            self._write_json(self.runs / run_id / "record.json", record)
            return record

    def save_result(self, run_id: str, result: dict[str, Any]) -> Path:
        path = self.runs / run_id / "result.json"
        self._write_json(path, result)
        self.update_run(run_id, result_path=str(path), status="completed")
        return path

    def get_result(self, run_id: str) -> dict[str, Any]:
        record = self.get_run(run_id)
        if not record.get("result_path"):
            raise RuntimeError("Experiment has no completed result")
        return self._read_json(Path(record["result_path"]))

    def _recover_interrupted_runs(self) -> None:
        for path in self.runs.glob("*/record.json"):
            try:
                record = self._read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if record.get("status") in {"queued", "running"}:
                record.update(
                    status="failed",
                    error="API service restarted before the experiment finished",
                    updated_at=_now(),
                )
                self._write_json(path, record)
