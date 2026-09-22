# CrewAI Automated ML and Time-Series Forecasting

This project provides reproducible machine-learning workflows for:

- tabular classification with deterministic preprocessing and model selection;
- weather and water-level time-series forecasting;
- optional CrewAI analysis through Ollama Cloud;
- local or distributed model/fold evaluation through SLURM and Slurminade; and
- a REST API and stdio MCP server for uploading time-series CSV files and retrieving experiment results.

The forecasting workflow uses chronological splitting, a horizon gap, expanding-window validation, causal lag/rolling features, and an untouched final test set. Raw observations are not sent to Ollama Cloud; agents receive only derived profiles, configuration choices and measured metrics.

## How the system works

The machine-learning pipeline performs the measurable work: it reads the data, handles missing timestamps and values, builds leakage-safe features, trains registered models, compares them on validation periods, evaluates the selected model once on the held-out test period, and saves reproducible artifacts. CrewAI is an optional review and decision layer around that deterministic pipeline; the language-model agents do not replace training code and do not invent metrics.

For classification, nine specialized agents are used across two stages:

- **Dataset Analyst** summarizes the derived dataset profile and identifies quality risks.
- **Feature Engineering Adviser** reviews preprocessing, missing data, encoding and leakage risks.
- **Baseline Analyst** interprets measured baseline performance and establishes the score to beat.
- **Neural Network Researcher** recommends an architecture appropriate for the dataset.
- **Neural Network Optimizer** selects one allowed configuration using previous measured results.
- **Experiment Engineer** checks that the proposed configuration is complete and executable.
- **Model Selection Scientist** verifies that the winner was selected using validation evidence rather than test results.
- **Experiment Critic** identifies weaknesses and limitations in the measured experiment.
- **Experiment Reporter** converts the evidence into a short human-readable summary.

For time-series forecasting, five agents review the completed deterministic experiment:

- **Time-Series and Seasonality Analyst** checks frequency, horizon, gaps and plausible seasonal structure.
- **Forecast Leakage Auditor** verifies that features and validation use only information available at prediction time.
- **Forecast Model Analyst** compares walk-forward MAE, RMSE and MASE results.
- **Weather and Water Forecast Critic** identifies limitations and useful follow-up experiments.
- **Forecast Experiment Reporter** produces a concise plain-language account without changing the measured results.

All agents run sequentially through CrewAI and receive derived profiles and metrics—not raw timestamp/value rows.

## Repository layout

| Path | Purpose |
| --- | --- |
| `forecasting/` | Time-series loading, preprocessing, features, validation, models and reports |
| `slurm/` | Deterministic distributed worker and local/SLURM dispatcher |
| `experiment_api/` | Development FastAPI service for external users and agents |
| `experiment_mcp/` | MCP tool adapter over the same experiment service |
| `configs/` | Dataset, distributed-job and SLURM example configurations |
| `data/thailand/` | Pathum Thani weather and water-level inputs used in the completed experiments |
| `tests/` | Classification, forecasting, Thai-data, SLURM and API tests |
| `forecast_main.py` | Direct forecasting command-line entry point |
| `forecast_csv.py` | Generic entry point for a user-supplied time-series CSV |
| `api_main.py` | REST API entry point |
| `mcp_main.py` | MCP stdio server entry point |

Generated `results/`, `artifacts/`, `api_data/`, virtual environments and `.env` files are intentionally ignored by Git.

## Requirements

- Python 3.11 or newer (verified with Python 3.13 on Windows)
- Git
- Windows, Linux or macOS for local runs
- A LANTA account and project allocation only for real SLURM runs
- An Ollama Cloud API key only when `--use-agents` is requested

## Quick start: local Windows run

```powershell
git clone <REPOSITORY_URL>
Set-Location CrewAI-rebuild
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Run the prepared Thai forecasting experiments:

```powershell
python forecast_main.py --config configs\thailand_pathum_weather.json
python forecast_main.py --config configs\thailand_pathum_water.json
```

Outputs are written under `results/forecasting/<dataset-name>/` and include the dataset profile, validation history, selected model, test predictions and Markdown report.

Test both distributed manifests locally before using a cluster:

```powershell
python -m slurm.distribution --manifest configs\distributed_pathum_weather.json --backend local --run-id pathum-weather-local
python -m slurm.distribution --manifest configs\distributed_pathum_water.json --backend local --run-id pathum-water-local
```

Each aggregate is written to `results/distributed/<run-id>/aggregate.json`.

## Run your own time-series dataset

Prepared project datasets are optional. `forecast_csv.py` accepts any standard CSV containing:

- one parseable timestamp column;
- one numeric target column to predict; and
- optionally, additional numeric feature columns.

Rows may contain ordinary missing numeric values. The workflow sorts and regularizes timestamps, forward-fills model inputs causally, and drops rows whose future target label is unavailable. It does not currently encode free-text or categorical features; omit those columns or preprocess them before running.

Example:

```powershell
python forecast_csv.py `
  --data "data\my_province_timeseries.csv" `
  --timestamp "timestamp" `
  --target "water_level" `
  --frequency "D" `
  --horizon 1 `
  --seasonality 7 `
  --lags 1 2 7 14 `
  --rolling-windows 7 30 `
  --calendar-features day_of_week day_of_year `
  --folds 3 `
  --validation-size 30 `
  --minimum-train-size 180
```

When `--features` is omitted, every other numeric column is selected automatically. To choose explicitly, add for example:

```powershell
--features rainfall temperature humidity
```

For hourly data use `--frequency H`; for ten-minute data use `--frequency 10min`. The dataset must contain enough chronological observations for the largest lag or rolling window, the validation folds and the final test split. Reduce those settings for a shorter dataset.

The development API offers the same generic-dataset capability through CSV upload. See [`API_README.md`](API_README.md).

## Optional CrewAI review with Ollama Cloud

Copy `.env.example` to `.env`, then set `OLLAMA_BASE_URL`, `OLLAMA_MODEL` and `OLLAMA_API_KEY`:

```powershell
Copy-Item .env.example .env
python forecast_main.py --config configs\thailand_pathum_weather.json --use-agents
```

Agent mode consumes API quota. Never commit `.env`; it is ignored by Git.

## Run on ThaiSC LANTA with SLURM

### 1. Clone and install on the login node

Connect to LANTA, move into storage belonging to your project, and clone the repository:

```bash
ssh YOUR_USERNAME@lanta.nstda.or.th
cd /project/YOUR_PROJECT_STORAGE/YOUR_USERNAME
git clone <REPOSITORY_URL> CrewAI-rebuild
cd CrewAI-rebuild
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-slurm.txt
```

The repository, environment, datasets and output directory must be visible from compute nodes. Use project storage, not a laptop-local path.

### 2. Configure the approved allocation

```bash
cp configs/slurm_lanta.example.json configs/slurm_lanta.json
```

Edit `configs/slurm_lanta.json` and replace `YOUR_PROJECT_ACCOUNT`. Confirm the permitted partition and limits with `sinfo`, `sbalance`, or the current ThaiSC documentation. The completed CPU experiments used two CPUs and 8 GB of memory per job; they did not require GPU resources.

Check Slurminade before the full run:

```bash
python -m slurminade.check --partition compute
```

### 3. Submit the experiments

```bash
python -m slurm.distribution \
  --manifest configs/distributed_pathum_weather.json \
  --backend slurm \
  --slurm-options configs/slurm_lanta.json \
  --run-id pathum-weather-lanta

python -m slurm.distribution \
  --manifest configs/distributed_pathum_water.json \
  --backend slurm \
  --slurm-options configs/slurm_lanta.json \
  --run-id pathum-water-lanta
```

Verify completion:

```bash
python -c "import json; r=json.load(open('results/distributed/pathum-weather-lanta/aggregate.json')); print({k:r[k] for k in ['expected_jobs','completed_jobs','failed_jobs','missing_jobs','corrupt_jobs']})"
python -c "import json; r=json.load(open('results/distributed/pathum-water-lanta/aggregate.json')); print({k:r[k] for k in ['expected_jobs','completed_jobs','failed_jobs','missing_jobs','corrupt_jobs']})"
```

Do not copy `.env` or cloud credentials to compute nodes. SLURM workers execute only deterministic forecasting jobs. More detail is available in [`SLURM_README.md`](SLURM_README.md).

## REST API

The API supports CSV upload/profile, asynchronous local forecasting, status polling, result retrieval, metrics, generated Markdown reports, and a constrained SSH/SLURM adapter for LANTA.

```powershell
.\.venv\Scripts\Activate.ps1
python api_main.py
```

Open `http://127.0.0.1:8000/docs` for the interactive API interface. Local execution and automated endpoint tests are working. Direct API-to-LANTA submission remains a development feature and requires non-interactive SSH-key or SSH-agent authentication plus an existing LANTA checkout. It should not yet be exposed as a public service.

See [`API_README.md`](API_README.md) for request examples, environment variables and current limitations.

## MCP server for AI agents

`mcp_main.py` exposes the same `ExperimentService` as MCP tools; it does not reimplement the forecasting pipeline. The server communicates over standard input/output and provides tools to register CSV text, inspect its derived profile, submit an experiment, check status, and retrieve results, metrics, or the generated report.

Run it manually:

```powershell
python mcp_main.py --data-root api_data
```

Example client configuration (replace the absolute paths):

```json
{
  "mcpServers": {
    "crewai-forecasting": {
      "command": "C:\\\\absolute\\\\path\\\\to\\\\CrewAI-rebuild\\\\.venv\\\\Scripts\\\\python.exe",
      "args": [
        "C:\\\\absolute\\\\path\\\\to\\\\CrewAI-rebuild\\\\mcp_main.py",
        "--data-root",
        "C:\\\\absolute\\\\path\\\\to\\\\CrewAI-rebuild\\\\api_data"
      ]
    }
  }
}
```

See [`MCP_README.md`](MCP_README.md) for the tool list, end-to-end flow, safety boundaries, and current compatibility status.

## Additional datasets

- `configs/weather_jena.json` supports the Jena climate dataset after it is downloaded and extracted into `data/`.
- `configs/groundwater_usgs.json` downloads and caches USGS groundwater observations when run with `--refresh`.
- `thai_csv` configurations support common Thai encodings, Buddhist-calendar conversion, station filters, column mapping and missing-value sentinels.

## Current status

- Classification, forecasting, Thai dataset adapters and local distributed execution are implemented.
- The Pathum Thani weather and water-level manifests were successfully executed on LANTA.
- Local and LANTA outputs were consistent within normal floating-point variation.
- The REST API local workflow and MCP subprocess workflow are covered by automated integration tests.
- Direct API-to-LANTA submission is implemented but still requires a live integration test using non-interactive SSH authentication.
- The MCP transport is dependency-free protocol-compatible stdio and has been tested with a subprocess client; validation with the official Python MCP SDK remains pending.
