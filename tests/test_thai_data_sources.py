import tempfile
import unittest
from pathlib import Path

import pandas as pd

from forecasting.data_sources import load_thai_csv
from forecasting.pipeline import prepare_time_series


class ThaiDataSourceTests(unittest.TestCase):
    def test_cp874_station_filter_date_merge_and_buddhist_year(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "thai_water.csv"
            pd.DataFrame({
                "วันที่": ["17/08/2567", "17/08/2567", "18/08/2567"],
                "เวลา": ["00:00", "01:00", "00:00"],
                "ชื่อสถานี": ["สำแล", "อื่น", "สำแล"],
                "ระดับน้ำ": [1.2, 9.9, 1.4],
            }).to_csv(path, index=False, encoding="cp874")
            source = {
                "path": str(path), "encodings": ["utf-8", "cp874"],
                "filters": {"ชื่อสถานี": "สำแล"},
                "timestamp_from": ["วันที่", "เวลา"],
                "timestamp_column": "timestamp",
                "column_map": {"ระดับน้ำ": "water_level_m"},
                "buddhist_year": True,
            }
            loaded = load_thai_csv(source, root)
            self.assertEqual(len(loaded), 2)
            self.assertIn("2024", loaded["timestamp"].iloc[0])
            self.assertEqual(loaded["water_level_m"].tolist(), [1.2, 1.4])

    def test_normalized_thai_data_enters_standard_pipeline(self):
        raw = pd.DataFrame({
            "timestamp": ["01/01/2024", "02/01/2024", "04/01/2024"],
            "temperature_c": [30.0, 31.0, 33.0],
        })
        config = {"data": {
            "source": "thai_csv", "name": "thai_weather_test",
            "timestamp_column": "timestamp", "target_column": "temperature_c",
            "feature_columns": [], "frequency": "D", "dayfirst": True,
        }}
        regular, profile = prepare_time_series(raw, config)
        self.assertEqual(len(regular), 4)
        self.assertEqual(profile["missing_timestamps"], 1)

    def test_nasa_power_metadata_year_day_and_missing_sentinel(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "power.csv"
            path.write_text(
                "-BEGIN HEADER-\n"
                "NASA/POWER daily data\n"
                "-END HEADER-\n"
                "YEAR,DOY,T2M_MAX,T2M_MIN\n"
                "2024,60,34.5,22.1\n"
                "2024,61,-999,22.4\n",
                encoding="utf-8",
            )
            source = {
                "path": str(path),
                "encodings": ["utf-8"],
                "header_starts_with": "YEAR,DOY,",
                "missing_values": [-999],
                "timestamp_column": "timestamp",
                "timestamp_from_year_day": {"year": "YEAR", "day_of_year": "DOY"},
                "column_map": {"T2M_MAX": "temperature_max_c"},
            }
            loaded = load_thai_csv(source, root)
            self.assertEqual(str(loaded["timestamp"].iloc[0].date()), "2024-02-29")
            self.assertTrue(pd.isna(loaded["temperature_max_c"].iloc[1]))


if __name__ == "__main__":
    unittest.main()
