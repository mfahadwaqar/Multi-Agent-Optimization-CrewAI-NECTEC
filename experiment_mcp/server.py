from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from .tools import ExperimentTools


_TOOLS = [
    {"name": "register_csv", "description": "Register UTF-8 CSV content for forecasting.", "inputSchema": {"type": "object", "properties": {"filename": {"type": "string"}, "csv_content": {"type": "string"}}, "required": ["filename", "csv_content"], "additionalProperties": False}},
    {"name": "get_dataset_profile", "description": "Inspect columns and summary metadata for a registered dataset.", "inputSchema": {"type": "object", "properties": {"dataset_id": {"type": "string"}}, "required": ["dataset_id"], "additionalProperties": False}},
    {"name": "submit_forecast", "description": "Submit a validated local or configured LANTA forecast experiment.", "inputSchema": {"type": "object", "properties": {"configuration": {"type": "object"}}, "required": ["configuration"], "additionalProperties": False}},
    {"name": "get_experiment_status", "description": "Get an experiment's current status.", "inputSchema": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"], "additionalProperties": False}},
    {"name": "get_experiment_results", "description": "Get complete results for a completed experiment.", "inputSchema": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"], "additionalProperties": False}},
    {"name": "get_experiment_metrics", "description": "Get selected validation and test metrics.", "inputSchema": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"], "additionalProperties": False}},
    {"name": "get_experiment_report", "description": "Get the generated Markdown report for a completed experiment.", "inputSchema": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"], "additionalProperties": False}},
]


class StdioMcpServer:
    def __init__(self, tools: ExperimentTools):
        self.tools = tools

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        request_id = request.get("id")
        method = request.get("method")
        if request_id is None:
            return None
        try:
            if method == "initialize":
                result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "crewai-forecasting", "version": "1.0.0"}}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": _TOOLS}
            elif method == "tools/call":
                params = request.get("params") or {}
                result = self._call(params.get("name"), params.get("arguments") or {})
            else:
                return self._error(request_id, -32601, "Method not found")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            return self._error(request_id, -32602, str(exc))

    def _call(self, name: str | None, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "register_csv": self.tools.register_csv,
            "get_dataset_profile": self.tools.get_dataset_profile,
            "submit_forecast": self.tools.submit_forecast,
            "get_experiment_status": self.tools.get_experiment_status,
            "get_experiment_results": self.tools.get_experiment_results,
            "get_experiment_metrics": self.tools.get_experiment_metrics,
            "get_experiment_report": self.tools.get_experiment_report,
        }
        if name not in handlers:
            raise ValueError("Unknown tool")
        value = handlers[name](**arguments)
        text = value if isinstance(value, str) else json.dumps(value, allow_nan=False)
        return {"content": [{"type": "text", "text": text}], "isError": False}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def run(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        for line in stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                response = self.handle(request)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                response = self._error(None, -32700, f"Parse error: {exc}")
            if response is not None:
                stdout.write(json.dumps(response, allow_nan=False) + "\n")
                stdout.flush()
