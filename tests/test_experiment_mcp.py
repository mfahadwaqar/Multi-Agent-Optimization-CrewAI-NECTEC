import json
import tempfile
import unittest
from pathlib import Path

from experiment_api.service import ExperimentService, InlineSubmitter
from experiment_api.storage import JsonStore


class FakeExecutor:
    def execute(self, config, run_directory):
        (run_directory / "artifacts").mkdir(parents=True, exist_ok=True)
        (run_directory / "artifacts" / "summary.md").write_text("# Safe report", encoding="utf-8")
        return {
            "best_result": {"name": "ridge", "validation_mae": 1.25, "test_rmse": 2.5},
            "results": [{"name": "ridge", "validation_mae": 1.25}],
        }


class ExperimentMcpToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        store = JsonStore(Path(self.temporary.name), max_upload_bytes=100)
        service = ExperimentService(
            store,
            local_executor=FakeExecutor(),
            lanta_executor=None,
            submitter=InlineSubmitter(),
        )
        from experiment_mcp.tools import ExperimentTools
        self.tools = ExperimentTools(service)

    def tearDown(self):
        self.temporary.cleanup()

    def _register(self):
        return self.tools.register_csv(
            filename="../private-series.csv",
            csv_content="time,value\n2026-01-01,1\n2026-01-02,2\n",
        )

    def test_register_and_profile_hide_server_paths_and_sanitize_name(self):
        dataset = self._register()
        self.assertEqual(dataset["original_filename"], "private-series.csv")
        self.assertNotIn("path", dataset)
        profile = self.tools.get_dataset_profile(dataset["dataset_id"])
        self.assertEqual(profile["columns"], ["time", "value"])
        self.assertNotIn("path", profile)

    def test_register_rejects_non_csv_and_oversized_content(self):
        with self.assertRaisesRegex(ValueError, "Only CSV"):
            self.tools.register_csv("series.txt", "time,value\n1,2\n")
        with self.assertRaisesRegex(ValueError, "maximum upload size"):
            self.tools.register_csv("series.csv", "x" * 101)

    def test_submit_status_results_metrics_and_report_reuse_service(self):
        dataset = self._register()
        run = self.tools.submit_forecast({
            "dataset_id": dataset["dataset_id"],
            "backend": "local",
            "timestamp_column": "time",
            "target_column": "value",
            "folds": 1,
            "validation_size": 1,
            "minimum_train_size": 10,
        })
        self.tools.service.store.save_report(run["run_id"], "# Canonical report")
        status = self.tools.get_experiment_status(run["run_id"])
        self.assertEqual(status["status"], "completed")
        self.assertNotIn("request", status)
        results = self.tools.get_experiment_results(run["run_id"])
        self.assertEqual(results["best_result"]["name"], "ridge")
        self.assertEqual(
            self.tools.get_experiment_metrics(run["run_id"]),
            {"model": "ridge", "validation_mae": 1.25, "test_rmse": 2.5},
        )
        self.assertEqual(self.tools.get_experiment_report(run["run_id"]), "# Canonical report")

    def test_slurm_submission_is_rejected_when_lanta_is_not_configured(self):
        dataset = self._register()
        with self.assertRaisesRegex(ValueError, "LANTA execution is not configured"):
            self.tools.submit_forecast({
                "dataset_id": dataset["dataset_id"],
                "backend": "slurm",
                "timestamp_column": "time",
                "target_column": "value",
            })

    def test_unknown_and_unfinished_runs_have_safe_errors(self):
        with self.assertRaisesRegex(KeyError, "Experiment not found"):
            self.tools.get_experiment_status("../record")
        run = self.tools.service.store.create_run("local", "dataset", {})
        with self.assertRaisesRegex(RuntimeError, "queued"):
            self.tools.get_experiment_results(run["run_id"])
        with self.assertRaisesRegex(RuntimeError, "not available"):
            self.tools.get_experiment_report(run["run_id"])


if __name__ == "__main__":
    unittest.main()
