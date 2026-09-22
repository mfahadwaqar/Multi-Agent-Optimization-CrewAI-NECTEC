import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from experiment_api.app import create_app
from experiment_api.service import ExperimentService, InlineSubmitter
from experiment_api.storage import JsonStore


class FakeExecutor:
    def __init__(self, result):
        self.result = result

    def execute(self, *args, **kwargs):
        run_directory = args[-1]
        artifact_directory = run_directory / "artifacts"
        artifact_directory.mkdir(parents=True, exist_ok=True)
        (artifact_directory / "summary.md").write_text(
            "# Forecast report\n\nSynthetic test report.\n", encoding="utf-8"
        )
        return self.result


class ExperimentApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        store = JsonStore(Path(self.temporary.name))
        self.store = store
        service = ExperimentService(
            store,
            local_executor=FakeExecutor({
                "best_result": {"name": "ridge", "test_mae": 1.25},
                "results": [{"name": "ridge", "validation_mae": 1.5}],
            }),
            lanta_executor=FakeExecutor({"completed_jobs": 3}),
            submitter=InlineSubmitter(),
        )
        self.client = TestClient(create_app(service))

    def tearDown(self):
        self.temporary.cleanup()

    def _upload(self):
        response = self.client.post(
            "/datasets",
            files={"file": ("series.csv", b"time,value\n2026-01-01,1\n2026-01-02,2\n", "text/csv")},
        )
        self.assertEqual(response.status_code, 201)
        self.assertNotIn("path", response.json())
        return response.json()

    def test_health_upload_profile_run_and_results(self):
        self.assertEqual(self.client.get("/health").json()["backends"], ["local", "slurm"])
        dataset = self._upload()
        profile = self.client.get(f"/datasets/{dataset['dataset_id']}")
        self.assertEqual(profile.status_code, 200)
        self.assertNotIn("path", profile.json())
        response = self.client.post("/experiments", json={
            "dataset_id": dataset["dataset_id"],
            "backend": "local",
            "timestamp_column": "time",
            "target_column": "value",
            "folds": 1,
            "validation_size": 1,
            "minimum_train_size": 10,
        })
        self.assertEqual(response.status_code, 202)
        run_id = response.json()["run_id"]
        status_body = self.client.get(f"/experiments/{run_id}").json()
        self.assertEqual(status_body["status"], "completed")
        self.assertNotIn("result_path", status_body)
        self.assertEqual(
            self.client.get(f"/experiments/{run_id}/results").json()["best_result"]["name"],
            "ridge",
        )
        metrics = self.client.get(f"/experiments/{run_id}/metrics")
        self.assertEqual(metrics.status_code, 200)
        self.assertEqual(metrics.json()["best_result"]["test_mae"], 1.25)
        report = self.client.get(f"/experiments/{run_id}/report")
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.headers["content-type"], "text/markdown; charset=utf-8")
        self.assertIn("Synthetic test report", report.text)

    def test_unknown_ids_and_unfinished_results_have_distinct_errors(self):
        self.assertEqual(self.client.get("/datasets/missing").status_code, 404)
        self.assertEqual(self.client.get("/experiments/missing").status_code, 404)

        queued = self.store.create_run("local", "dataset", {})
        response = self.client.get(f"/experiments/{queued['run_id']}/results")
        self.assertEqual(response.status_code, 409)
        self.assertIn("queued", response.json()["detail"])

        self.store.update_run(queued["run_id"], status="failed", error="training failed")
        response = self.client.get(f"/experiments/{queued['run_id']}/results")
        self.assertEqual(response.status_code, 409)
        self.assertIn("failed", response.json()["detail"])

        self.assertEqual(
            self.client.get(f"/experiments/{queued['run_id']}/metrics").status_code, 409
        )
        self.assertEqual(
            self.client.get(f"/experiments/{queued['run_id']}/report").status_code, 409
        )

    def test_rejects_invalid_column_data_before_creating_a_run(self):
        invalid_time = self.client.post(
            "/datasets",
            files={"file": ("bad-time.csv", b"time,value\nnot-a-date,1\nstill-bad,2\n", "text/csv")},
        ).json()
        response = self.client.post("/experiments", json={
            "dataset_id": invalid_time["dataset_id"],
            "timestamp_column": "time",
            "target_column": "value",
        })
        self.assertEqual(response.status_code, 422)
        self.assertIn("timestamp", response.json()["detail"].lower())

        non_numeric = self.client.post(
            "/datasets",
            files={"file": ("bad-target.csv", b"time,value\n2026-01-01,low\n2026-01-02,high\n", "text/csv")},
        ).json()
        response = self.client.post("/experiments", json={
            "dataset_id": non_numeric["dataset_id"],
            "timestamp_column": "time",
            "target_column": "value",
        })
        self.assertEqual(response.status_code, 422)
        self.assertIn("numeric", response.json()["detail"].lower())

    def test_rejects_slurm_submission_when_lanta_is_not_configured(self):
        service = ExperimentService(
            self.store,
            local_executor=FakeExecutor({}),
            lanta_executor=None,
            submitter=InlineSubmitter(),
        )
        client = TestClient(create_app(service))
        dataset = self._upload()
        response = client.post("/experiments", json={
            "dataset_id": dataset["dataset_id"],
            "backend": "slurm",
            "timestamp_column": "time",
            "target_column": "value",
        })
        self.assertEqual(response.status_code, 503)
        self.assertEqual(list(self.store.runs.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
