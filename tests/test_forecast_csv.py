import tempfile
import unittest
from pathlib import Path

from forecast_csv import build_request_from_csv


class GenericCsvForecastTests(unittest.TestCase):
    def test_build_request_accepts_unprepared_csv_and_detects_numeric_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "province.csv"
            path.write_text(
                "observed_at,water_level,rainfall,station_name\n"
                "2026-01-01,1.2,3.4,A\n"
                "2026-01-02,1.3,,A\n",
                encoding="utf-8",
            )

            request, dataset = build_request_from_csv(
                path,
                timestamp_column="observed_at",
                target_column="water_level",
                frequency="D",
                folds=1,
                validation_size=1,
                minimum_train_size=10,
            )

            self.assertEqual(request.feature_columns, ["rainfall"])
            self.assertEqual(dataset["columns"], ["observed_at", "water_level", "rainfall", "station_name"])
            self.assertEqual(dataset["path"], str(path.resolve()))

    def test_explicit_unknown_columns_are_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "series.csv"
            path.write_text("time,value\n2026-01-01,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not found"):
                build_request_from_csv(
                    path,
                    timestamp_column="missing_time",
                    target_column="value",
                )


if __name__ == "__main__":
    unittest.main()
