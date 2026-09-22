import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.workflow import run_forecast


class ForecastWorkflowTests(unittest.TestCase):
    def test_complete_offline_forecast_writes_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            timestamps = pd.date_range("2023-01-01", periods=420, freq="h")
            values = 15 + 4 * np.sin(2 * np.pi * np.arange(420) / 24) + np.arange(420) * 0.005
            frame = pd.DataFrame({"time": timestamps, "temperature": values, "humidity": 60 + np.cos(np.arange(420) / 12)})
            frame.loc[[50, 151], "temperature"] = np.nan
            frame = frame.drop(index=200)
            data_path = root / "weather.csv"
            frame.to_csv(data_path, index=False)
            config = {
                "data": {
                    "source": "csv", "name": "synthetic_weather", "path": str(data_path),
                    "timestamp_column": "time", "target_column": "temperature",
                    "feature_columns": ["humidity"], "frequency": "h",
                },
                "forecast": {
                    "horizon": 3, "seasonality": 24, "lags": [1, 3, 24],
                    "rolling_windows": [6, 24], "calendar_features": ["hour", "day_of_week"],
                },
                "validation": {"folds": 2, "window_size": 30, "minimum_train_size": 120},
                "test_fraction": 0.2, "random_seed": 7,
                "models": [
                    {"name": "ridge", "params": {"alpha": 1.0}},
                    {"name": "random_forest", "params": {"n_estimators": 20, "max_depth": 8, "min_samples_leaf": 2}},
                ],
            }
            output = root / "output"
            history = run_forecast(config, root=root, output_dir=output)
            self.assertIn(history["best_result"]["name"], {"persistence", "seasonal_naive", "ridge", "random_forest"})
            self.assertIn("test_mase", history["best_result"])
            self.assertEqual(len(history["results"]), 4)
            for name in ["history.json", "summary.md", "best_model.joblib", "test_predictions.csv"]:
                self.assertTrue((output / name).exists(), name)
            self.assertTrue((output / "artifacts" / "profile.json").exists())
            saved = json.loads((output / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["selection_policy"], "lowest expanding-window validation MASE, then MAE")
            profile = json.loads((output / "artifacts" / "profile.json").read_text(encoding="utf-8"))
            self.assertGreater(profile["seasonality_analysis"]["lag_correlation"], 0.9)


if __name__ == "__main__":
    unittest.main()
