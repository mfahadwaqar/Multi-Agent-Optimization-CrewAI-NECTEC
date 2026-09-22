# Forecasting MCP Server

The MCP server lets an AI agent use the time-series experiment workflow as tools. It is a thin adapter: dataset storage, validation, local execution, optional LANTA execution and result persistence all come from the same `experiment_api` service used by the REST API.

## Tools

- `register_csv`: register UTF-8 CSV text within configured size and column limits.
- `get_dataset_profile`: return columns, types, row count and missing-value counts without raw rows or server paths.
- `submit_forecast`: validate and submit a local or configured LANTA experiment.
- `get_experiment_status`: return queued, running, completed or failed status.
- `get_experiment_results`: return the completed structured result.
- `get_experiment_metrics`: return selected-model validation and test metrics.
- `get_experiment_report`: return the generated Markdown report.

The tools do not accept server file paths, shell commands, Python code, credentials, arbitrary model classes or SLURM accounts.

## Start the stdio server

```powershell
.\.venv\Scripts\Activate.ps1
python mcp_main.py --data-root api_data
```

Use absolute paths when configuring an MCP client:

```json
{
  "mcpServers": {
    "crewai-forecasting": {
      "command": "C:\\\\path\\\\to\\\\CrewAI-rebuild\\\\.venv\\\\Scripts\\\\python.exe",
      "args": [
        "C:\\\\path\\\\to\\\\CrewAI-rebuild\\\\mcp_main.py",
        "--data-root",
        "C:\\\\path\\\\to\\\\CrewAI-rebuild\\\\api_data"
      ]
    }
  }
}
```

## Agent workflow

1. Call `register_csv` with a safe filename and CSV text.
2. Read the returned `dataset_id` and inspect `get_dataset_profile`.
3. Call `submit_forecast` with that ID, timestamp and target columns, and forecasting settings.
4. Poll `get_experiment_status` until it is `completed` or `failed`.
5. Read `get_experiment_metrics`, `get_experiment_results` and `get_experiment_report`.

For `backend: "slurm"`, configure the LANTA environment variables described in `API_README.md`; otherwise submission is rejected without creating a run.

## Verification and limitation

Automated tests start the server as a real subprocess, initialize it, list tools and call CSV registration. Tool-level tests cover the full register, submit, status, result, metrics and report flow. The transport implements MCP 2024-11-05 JSON-RPC over stdio without an additional SDK dependency. It has not yet been tested with the official Python MCP SDK or a live external desktop-agent client.
