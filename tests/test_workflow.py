import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset_pipeline import prepare_dataset
from evaluation import choose_best
from optimization_loop import run


ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

    def test_pipeline_has_three_clean_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            arrays, profile = prepare_dataset(self.config, Path(folder))
            expected = {"X_train", "X_validation", "X_test", "y_train", "y_validation", "y_test"}
            self.assertEqual(set(arrays), expected)
            self.assertEqual(profile["usable_rows"], 1797)
            self.assertEqual(profile["input_features"], 64)
            self.assertTrue(all(np.isfinite(value).all() for key, value in arrays.items() if key.startswith("X_")))

    def test_selection_uses_validation_accuracy(self):
        results = [
            {"name": "A", "status": "completed", "validation_accuracy": 0.80, "test_accuracy": 0.99},
            {"name": "B", "status": "completed", "validation_accuracy": 0.85, "test_accuracy": 0.50},
        ]
        self.assertEqual(choose_best(results)["name"], "B")

    def test_complete_local_run_writes_outputs(self):
        self.config["optimization"]["iterations"] = 1
        self.config["search_space"]["epochs"] = [20]
        with tempfile.TemporaryDirectory() as folder:
            history = run(self.config, use_agents=False, output_dir=Path(folder))
            self.assertIn("test_accuracy", history["best_result"])
            self.assertTrue((Path(folder) / "history.json").exists())
            self.assertTrue((Path(folder) / "summary.md").exists())
            self.assertTrue((Path(folder) / "best_model.joblib").exists())


if __name__ == "__main__":
    unittest.main()
