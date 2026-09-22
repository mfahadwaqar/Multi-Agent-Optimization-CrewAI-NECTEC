import json
import tempfile
import unittest
from pathlib import Path

from experiment_api.storage import JsonStore


class JsonStoreTests(unittest.TestCase):
    def test_upload_profiles_csv_and_never_uses_client_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonStore(Path(directory), max_upload_bytes=1024)
            record = store.save_dataset(
                "../../passengers.csv",
                b"time,value,feature\n2026-01-01,1,\n2026-01-02,2,3\n",
            )

            self.assertEqual(record["row_count"], 2)
            self.assertEqual(record["missing_values"]["feature"], 1)
            self.assertEqual(record["columns"], ["time", "value", "feature"])
            stored = Path(record["path"])
            self.assertEqual(stored.parent.name, record["dataset_id"])
            self.assertEqual(stored.name, "source.csv")
            self.assertTrue(stored.is_file())

    def test_upload_rejects_non_csv_empty_and_oversized_content(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonStore(Path(directory), max_upload_bytes=12)
            with self.assertRaisesRegex(ValueError, "CSV"):
                store.save_dataset("data.txt", b"a,b\n1,2\n")
            with self.assertRaisesRegex(ValueError, "empty"):
                store.save_dataset("data.csv", b"")
            with self.assertRaisesRegex(ValueError, "maximum"):
                store.save_dataset("data.csv", b"a,b\n123456789\n")

    def test_interrupted_runs_are_marked_failed_on_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = JsonStore(root)
            run = store.create_run("local", "dataset-1", {"target_column": "value"})
            store.update_run(run["run_id"], status="running")

            restarted = JsonStore(root)
            recovered = restarted.get_run(run["run_id"])

            self.assertEqual(recovered["status"], "failed")
            self.assertIn("restarted", recovered["error"].lower())
            parsed = json.loads((root / "runs" / run["run_id"] / "record.json").read_text())
            self.assertEqual(parsed["status"], "failed")

    def test_lookup_rejects_non_generated_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = JsonStore(root)
            outside = root.parent / "outside" / "record.json"
            outside.parent.mkdir(exist_ok=True)
            outside.write_text('{"dataset_id": "escaped"}', encoding="utf-8")
            with self.assertRaises(KeyError):
                store.get_dataset("../../outside")
            with self.assertRaises(KeyError):
                store.get_run("not-a-uuid")


if __name__ == "__main__":
    unittest.main()
