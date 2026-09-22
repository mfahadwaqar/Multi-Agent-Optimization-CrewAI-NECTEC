import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class McpStdioIntegrationTests(unittest.TestCase):
    def test_client_can_initialize_list_tools_and_call_upload(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            process = subprocess.Popen(
                [sys.executable, "mcp_main.py", "--data-root", directory],
                cwd=Path(__file__).resolve().parents[1],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            requests = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                    "name": "register_csv",
                    "arguments": {"filename": "sample.csv", "csv_content": "time,value\n2026-01-01,1\n"},
                }},
            ]
            stdout, stderr = process.communicate(
                "".join(json.dumps(item) + "\n" for item in requests), timeout=15
            )
            self.assertEqual(process.returncode, 0, stderr)
            responses = [json.loads(line) for line in stdout.splitlines()]
            self.assertEqual(responses[0]["result"]["protocolVersion"], "2024-11-05")
            names = {tool["name"] for tool in responses[1]["result"]["tools"]}
            self.assertEqual(names, {
                "register_csv", "get_dataset_profile", "submit_forecast",
                "get_experiment_status", "get_experiment_results",
                "get_experiment_metrics", "get_experiment_report",
            })
            content = json.loads(responses[2]["result"]["content"][0]["text"])
            self.assertEqual(content["row_count"], 1)
            self.assertNotIn("path", content)


if __name__ == "__main__":
    unittest.main()
