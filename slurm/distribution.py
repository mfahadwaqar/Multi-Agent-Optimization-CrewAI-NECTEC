"""Submit manifest jobs locally or through Slurminade, then aggregate files."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slurm.worker import execute_job, validate_job

try:  # Slurminade is deliberately optional for local development.
    import slurminade as _slurminade
except ImportError:  # pragma: no cover - expected on ordinary Windows machines
    _slurminade = None


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _atomic_json_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
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


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("jobs"), list):
        raise ValueError("Manifest must be an object containing a jobs list")
    if not manifest["jobs"]:
        raise ValueError("Manifest contains no jobs")
    seen: set[str] = set()
    for job in manifest["jobs"]:
        validate_job(job)
        job_id = str(job["job_id"])
        if job_id in seen:
            raise ValueError(f"Duplicate job_id in manifest: {job_id}")
        seen.add(job_id)
    return manifest


def aggregate_results(result_directory: str | Path, expected_job_ids: list[str]) -> dict[str, Any]:
    result_directory = Path(result_directory).resolve()
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    corrupt: list[dict[str, str]] = []
    for job_id in expected_job_ids:
        path = result_directory / f"{job_id}.json"
        if not path.exists():
            missing.append(job_id)
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("job_id") != job_id:
                raise ValueError("result job_id does not match its filename")
            records.append(value)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            corrupt.append({"job_id": job_id, "error": str(exc)})
    summary = {
        "expected_jobs": len(expected_job_ids),
        "completed_jobs": sum(item.get("status") == "completed" for item in records),
        "failed_jobs": sum(item.get("status") == "failed" for item in records),
        "missing_jobs": missing,
        "corrupt_jobs": corrupt,
        "results": records,
    }
    _atomic_json_write(result_directory / "aggregate.json", summary)
    return summary


if _slurminade is not None:
    @_slurminade.slurmify()
    def _distributed_experiment(job: dict[str, Any], result_directory: str) -> None:
        execute_job(job, result_directory)
else:
    _distributed_experiment = None


def dispatch_jobs(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    backend: str = "local",
    run_id: str | None = None,
    slurm_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch all jobs and return the file-based aggregate."""
    if backend not in {"local", "slurm"}:
        raise ValueError("backend must be 'local' or 'slurm'")
    manifest = load_manifest(manifest_path)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, digits, dots, dashes or underscores")
    result_directory = Path(output_root).resolve() / run_id
    result_directory.mkdir(parents=True, exist_ok=False)
    jobs = manifest["jobs"]

    if backend == "local":
        for job in jobs:
            execute_job(job, result_directory)
    else:
        if platform.system() == "Windows":
            raise RuntimeError("Real SLURM dispatch requires a Linux cluster; use --backend local on Windows")
        if _slurminade is None or _distributed_experiment is None:
            raise RuntimeError("Install slurminade before using --backend slurm")
        if slurm_options:
            _slurminade.update_default_configuration(**slurm_options)
        _slurminade.set_dispatcher(_slurminade.SlurmDispatcher())
        bundle_size = int(manifest.get("bundle_size", 1))
        if bundle_size < 1:
            raise ValueError("bundle_size must be at least 1")
        with _slurminade.JobBundling(max_size=bundle_size):
            for job in jobs:
                _distributed_experiment.distribute(job, str(result_directory))
        _slurminade.join()

    return aggregate_results(result_directory, [str(job["job_id"]) for job in jobs])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("results/distributed"), type=Path)
    parser.add_argument("--backend", choices=("local", "slurm"), default="local")
    parser.add_argument("--run-id")
    parser.add_argument("--slurm-options", type=Path, help="JSON object passed to Slurminade configuration")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    options = None
    if arguments.slurm_options:
        options = json.loads(arguments.slurm_options.read_text(encoding="utf-8"))
        if not isinstance(options, dict):
            raise ValueError("SLURM options file must contain a JSON object")
    summary = dispatch_jobs(
        arguments.manifest,
        arguments.output_root,
        backend=arguments.backend,
        run_id=arguments.run_id,
        slurm_options=options,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
