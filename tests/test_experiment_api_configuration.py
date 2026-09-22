import unittest

from experiment_api.configuration import build_forecast_config, build_manifest
from experiment_api.models import ExperimentRequest, ModelRequest


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = {
            "dataset_id": "abc123",
            "columns": ["when", "level", "rain"],
            "path": "C:/server/datasets/abc123/source.csv",
        }

    def test_builds_existing_forecast_shape_and_manifest_per_model_fold(self):
        request = ExperimentRequest(
            dataset_id="abc123",
            backend="local",
            timestamp_column="when",
            target_column="level",
            feature_columns=["rain"],
            frequency="D",
            horizon=1,
            seasonality=7,
            lags=[1, 7],
            rolling_windows=[7],
            folds=2,
            validation_size=10,
            minimum_train_size=30,
            models=[ModelRequest(name="ridge", params={"alpha": 2.0})],
        )
        config = build_forecast_config(request, self.dataset)
        manifest = build_manifest(config, "input/config.json", ".", "run-1")

        self.assertEqual(config["data"]["path"], self.dataset["path"])
        self.assertEqual(config["data"]["target_column"], "level")
        self.assertEqual(config["models"][0]["name"], "ridge")
        self.assertEqual(len(manifest["jobs"]), 6)  # 2 baselines + ridge, two folds
        self.assertEqual({job["configuration_id"] for job in manifest["jobs"]},
                         {"persistence", "seasonal_naive", "ridge"})

    def test_rejects_unknown_or_duplicate_columns(self):
        with self.assertRaisesRegex(ValueError, "not present"):
            build_forecast_config(
                ExperimentRequest(
                    dataset_id="abc123", timestamp_column="when", target_column="missing"
                ),
                self.dataset,
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            build_forecast_config(
                ExperimentRequest(
                    dataset_id="abc123",
                    timestamp_column="when",
                    target_column="level",
                    feature_columns=["rain", "rain"],
                ),
                self.dataset,
            )


if __name__ == "__main__":
    unittest.main()
