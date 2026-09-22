from __future__ import annotations

import json
import os
import posixpath
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from forecasting.workflow import run_forecast

from .configuration import build_forecast_config, build_manifest
from .models import ExperimentRequest


class LocalExecutor:
    def __init__(self, forecast_runner: Callable[..., dict[str, Any]] = run_forecast):
        self.forecast_runner = forecast_runner

    def execute(self, config: dict[str, Any], run_directory: Path) -> dict[str, Any]:
        return self.forecast_runner(
            config,
            use_agents=False,
            refresh=False,
            output_dir=run_directory / "artifacts",
        )


@dataclass(frozen=True)
class LantaSettings:
    host: str
    username: str
    remote_project: str
    ssh_executable: str = "ssh"
    scp_executable: str = "scp"
    python_path: str = ".venv/bin/python"
    slurm_options: str = "configs/slurm_lanta.json"
    timeout_seconds: int = 86_400

    @classmethod
    def from_env(cls) -> "LantaSettings":
        required = {
            "host": os.getenv("LANTA_HOST"),
            "username": os.getenv("LANTA_USERNAME"),
            "remote_project": os.getenv("LANTA_PROJECT_DIR"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(f"Missing LANTA settings: {', '.join(missing)}")
        return cls(
            **required,
            ssh_executable=os.getenv("LANTA_SSH", "ssh"),
            scp_executable=os.getenv("LANTA_SCP", "scp"),
            python_path=os.getenv("LANTA_PYTHON", ".venv/bin/python"),
            slurm_options=os.getenv("LANTA_SLURM_OPTIONS", "configs/slurm_lanta.json"),
            timeout_seconds=int(os.getenv("LANTA_TIMEOUT_SECONDS", "86400")),
        )


class LantaExecutor:
    def __init__(self, settings: LantaSettings, runner: Callable[..., Any] = subprocess.run):
        self.settings = settings
        self.runner = runner

    def _run(self, arguments: list[str]) -> None:
        completed = self.runner(
            arguments,
            capture_output=True,
            text=True,
            timeout=self.settings.timeout_seconds,
            check=False,
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "command failed").strip()
            raise RuntimeError(detail[-2000:])

    def execute(
        self,
        request: ExperimentRequest,
        dataset: dict[str, Any],
        run_id: str,
        run_directory: Path,
    ) -> dict[str, Any]:
        target = f"{self.settings.username}@{self.settings.host}"
        remote_run = posixpath.join(self.settings.remote_project.rstrip("/"), "api_runs", run_id)
        remote_input = posixpath.join(remote_run, "input")
        remote_output_root = posixpath.join(remote_run, "results")
        remote_config = posixpath.join(remote_input, "config.json")
        remote_manifest = posixpath.join(remote_input, "manifest.json")
        remote_dataset = posixpath.join(remote_input, "source.csv")
        remote_aggregate = posixpath.join(remote_output_root, run_id, "aggregate.json")

        staging = run_directory / "lanta_input"
        staging.mkdir(parents=True, exist_ok=True)
        config = build_forecast_config(request, dataset, data_path=remote_dataset)
        manifest = build_manifest(config, remote_config, self.settings.remote_project, run_id)
        config_file = staging / "config.json"
        manifest_file = staging / "manifest.json"
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        ssh_base = [self.settings.ssh_executable, "-o", "BatchMode=yes", target]
        self._run(ssh_base + [f"mkdir -p -- {shlex.quote(remote_input)} {shlex.quote(remote_output_root)}"])
        self._run([
            self.settings.scp_executable, "-o", "BatchMode=yes",
            dataset["path"], str(config_file), str(manifest_file), f"{target}:{remote_input}/",
        ])
        command = " && ".join([
            f"cd {shlex.quote(self.settings.remote_project)}",
            " ".join([
                shlex.quote(self.settings.python_path), "-m", "slurm.distribution",
                "--manifest", shlex.quote(remote_manifest),
                "--output-root", shlex.quote(remote_output_root),
                "--backend", "slurm",
                "--run-id", shlex.quote(run_id),
                "--slurm-options", shlex.quote(posixpath.join(self.settings.remote_project, self.settings.slurm_options)),
            ]),
        ])
        self._run(ssh_base + [command])
        local_aggregate = run_directory / "aggregate.json"
        self._run([
            self.settings.scp_executable, "-o", "BatchMode=yes",
            f"{target}:{remote_aggregate}", str(local_aggregate),
        ])
        return json.loads(local_aggregate.read_text(encoding="utf-8"))

