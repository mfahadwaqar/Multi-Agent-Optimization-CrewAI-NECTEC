import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from slurm.distribution import aggregate_results, dispatch_jobs, load_manifest
from slurm.worker import execute_job, validate_job


def _job(job_id: str = "weather-ridge-fold-0") -> dict:
    return {
        "job_id": job_id,
        "runner": "forecasting",
        "dataset_id": "weather-jena",
        "configuration_id": "ridge-v1",
        "fold_id": 0,
        "model_name": "ridge",
        "model_params": {"alpha": 1.0},
    }


class SlurmWorkerTests(unittest.TestCase):
    def test_worker_writes_unique_result_and_hides_cloud_key(self):
        observed = {}

        def runner(job):
            observed["key_visible"] = "OLLAMA_API_KEY" in os.environ
            return {"mae": 0.25, "fold": job["fold_id"]}

        with tempfile.TemporaryDirectory() as folder:
            os.environ["OLLAMA_API_KEY"] = "must-not-reach-worker"
            try:
                result = execute_job(_job(), folder, runner=runner)
                saved = json.loads((Path(folder) / "weather-ridge-fold-0.json").read_text(encoding="utf-8"))
            finally:
                os.environ.pop("OLLAMA_API_KEY", None)
        self.assertFalse(observed["key_visible"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(saved["result"]["mae"], 0.25)

    def test_secret_like_worker_input_is_rejected(self):
        job = _job()
        job["payload"] = {"ollama_api_key": "not-allowed"}
        with self.assertRaisesRegex(ValueError, "Secret-like field"):
            validate_job(job)

    def test_non_json_input_is_rejected(self):
        job = _job()
        job["payload"] = {"bad": {1, 2}}
        with self.assertRaisesRegex(ValueError, "JSON-serializable"):
            validate_job(job)


class DistributionTests(unittest.TestCase):
    def test_duplicate_job_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "manifest.json"
            manifest.write_text(json.dumps({"jobs": [_job(), _job()]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate job_id"):
                load_manifest(manifest)

    def test_local_dispatch_uses_forecasting_entry_point(self):
        runner = lambda job: {"mae": 0.1, "model": job["model_name"]}
        with patch("slurm.worker._resolve_runner", return_value=runner):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                manifest = root / "manifest.json"
                manifest.write_text(json.dumps({"jobs": [_job()]}), encoding="utf-8")
                summary = dispatch_jobs(manifest, root / "results", run_id="test-run")
                aggregate = root / "results" / "test-run" / "aggregate.json"
                self.assertTrue(aggregate.exists())
                self.assertEqual(summary["completed_jobs"], 1)
                self.assertEqual(summary["results"][0]["result"]["mae"], 0.1)

    def test_aggregation_reports_missing_and_corrupt_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "bad.json").write_text("not json", encoding="utf-8")
            summary = aggregate_results(root, ["bad", "missing"])
            self.assertEqual(summary["missing_jobs"], ["missing"])
            self.assertEqual(summary["corrupt_jobs"][0]["job_id"], "bad")

    def test_local_dispatch_runs_real_forecast_fold_from_shared_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            timestamps = pd.date_range("2024-01-01", periods=120, freq="h")
            values = np.sin(np.arange(120) * 2 * np.pi / 24) + np.arange(120) * 0.01
            pd.DataFrame({"time": timestamps, "level": values}).to_csv(root / "series.csv", index=False)
            config = {
                "data": {
                    "source": "csv", "name": "synthetic-water", "path": "series.csv",
                    "timestamp_column": "time", "target_column": "level",
                    "feature_columns": [], "frequency": "h",
                },
                "forecast": {
                    "horizon": 1, "seasonality": 24, "lags": [1, 24],
                    "rolling_windows": [3], "calendar_features": ["hour"],
                },
                "validation": {"folds": 2, "window_size": 12, "minimum_train_size": 30},
                "test_fraction": 0.2,
                "random_seed": 42,
                "models": [{"name": "ridge", "params": {"alpha": 1.0}}],
            }
            config_path = root / "forecast.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            job = {
                **_job("synthetic-ridge-fold-1"),
                "dataset_id": "synthetic-water",
                "configuration_id": "ridge",
                "fold_id": 1,
                "config_path": str(config_path),
                "project_root": str(root),
            }
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"jobs": [job]}), encoding="utf-8")
            summary = dispatch_jobs(manifest, root / "results", run_id="real-integration")
            measured = summary["results"][0]["result"]
            self.assertEqual(summary["completed_jobs"], 1)
            self.assertEqual(measured["model_name"], "ridge")
            self.assertIn("mae", measured["metrics"])


if __name__ == "__main__":
    unittest.main()
