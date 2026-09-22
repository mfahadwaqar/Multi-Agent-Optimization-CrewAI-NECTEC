import json
import tempfile
import unittest
from pathlib import Path

from experiment_api.executors import LantaExecutor, LantaSettings, LocalExecutor
from experiment_api.models import ExperimentRequest


class ExecutorTests(unittest.TestCase):
    def test_local_executor_delegates_with_isolated_output(self):
        calls = []
        expected = {"best_result": {"name": "ridge"}}
        executor = LocalExecutor(lambda config, **kwargs: calls.append((config, kwargs)) or expected)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = executor.execute({"data": {"name": "sample"}}, output)
        self.assertEqual(result, expected)
        self.assertEqual(calls[0][1]["output_dir"], output / "artifacts")
        self.assertFalse(calls[0][1]["use_agents"])

    def test_lanta_executor_uses_argument_lists_and_retrieves_aggregate(self):
        calls = []

        def runner(arguments, **kwargs):
            calls.append((arguments, kwargs))
            if arguments[0] == "scp" and str(arguments[-2]).endswith("aggregate.json"):
                Path(arguments[-1]).write_text('{"completed_jobs": 3}', encoding="utf-8")
            class Completed:
                returncode = 0
                stdout = "ok"
                stderr = ""
            return Completed()

        settings = LantaSettings(
            host="lanta.nstda.or.th",
            username="mwaqar",
            remote_project="/project/lt200304-dipmt/mwaqar/CrewAI-rebuild",
            ssh_executable="ssh",
            scp_executable="scp",
        )
        dataset = {"dataset_id": "data1", "columns": ["time", "value"], "path": "ignored"}
        request = ExperimentRequest(
            dataset_id="data1", backend="slurm", timestamp_column="time", target_column="value",
            folds=1, validation_size=10, minimum_train_size=20,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text("time,value\n2026-01-01,1\n", encoding="utf-8")
            dataset["path"] = str(source)
            result = LantaExecutor(settings, runner=runner).execute(request, dataset, "safe-run", root)

        self.assertEqual(result["completed_jobs"], 3)
        ssh_calls = [args for args, _ in calls if args[0] == "ssh"]
        self.assertTrue(ssh_calls)
        self.assertIn("BatchMode=yes", ssh_calls[0])
        self.assertTrue(all(isinstance(call[0], list) for call in calls))


if __name__ == "__main__":
    unittest.main()
