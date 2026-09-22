import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.backtesting import expanding_window_folds
from forecasting.features import build_supervised
from forecasting.pipeline import prepare_time_series


class ForecastPipelineTests(unittest.TestCase):
    def test_regularization_and_features_are_causal(self):
        timestamps = pd.date_range("2024-01-01", periods=80, freq="h")
        raw = pd.DataFrame({"time": timestamps, "value": np.arange(80, dtype=float)})
        raw = raw.drop(index=30).reset_index(drop=True)
        config = {
            "data": {
                "source": "csv", "name": "synthetic", "timestamp_column": "time",
                "target_column": "value", "feature_columns": [], "frequency": "h",
            }
        }
        regular, profile = prepare_time_series(raw, config)
        self.assertEqual(profile["missing_timestamps"], 1)
        features, _, _ = build_supervised(regular, {
            "target_column": "value", "feature_columns": [], "horizon": 2,
            "seasonality": 24, "lags": [1], "rolling_windows": [3], "calendar_features": ["hour"],
        })
        missing_time = timestamps[30]
        self.assertEqual(features.loc[missing_time, "value_current"], 29.0)
        self.assertEqual(features.loc[missing_time, "value_rolling_mean_3"], 28.0)

    def test_expanding_folds_keep_a_horizon_gap(self):
        folds = expanding_window_folds(200, folds=2, validation_size=30, gap=5, minimum_train_size=100)
        for fold in folds:
            self.assertEqual(fold.validation.start - fold.train.stop, 5)
            self.assertEqual(fold.train.start, 0)


if __name__ == "__main__":
    unittest.main()
