# Time-Series Experiment API

This private API exposes the existing forecasting workflow to trusted programs
and agents. It accepts CSV files but never accepts Python code, shell commands,
server paths, credentials, SLURM accounts, or arbitrary model classes.

## 1. Install and start

From PowerShell in the project directory:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env  # only when .env does not exist
python api_main.py
```

The server listens only on `http://127.0.0.1:8000`. Interactive documentation
is available at `http://127.0.0.1:8000/docs`.

## 2. Upload a dataset

```powershell
curl.exe -X POST "http://127.0.0.1:8000/datasets" `
  -F "file=@data\thailand\pathum_weather.csv"
```

Copy the returned `dataset_id`. The response contains column names, inferred
types, row count, and missing-value counts; it does not echo raw rows.

## 3. Submit a local experiment

Create `request.json`:

```json
{
  "dataset_id": "PASTE_DATASET_ID",
  "backend": "local",
  "timestamp_column": "timestamp",
  "target_column": "temperature_max_c",
  "feature_columns": ["temperature_min_c", "rainfall_mm", "humidity_percent", "wind_speed_m_s"],
  "frequency": "D",
  "horizon": 1,
  "seasonality": 365,
  "lags": [1, 2, 7, 14, 30, 365],
  "rolling_windows": [7, 30, 90, 365],
  "calendar_features": ["day_of_week", "day_of_year"],
  "folds": 3,
  "validation_size": 90,
  "minimum_train_size": 730,
  "models": [
    {"name": "ridge", "params": {"alpha": 1.0}},
    {"name": "random_forest", "params": {"n_estimators": 150, "max_depth": 12, "min_samples_leaf": 3}}
  ]
}
```

An uploaded CSV must already contain the column names used in the request. The
API's generic CSV mode does not apply the special NASA POWER or Thai-agency
column transformations used by the hand-written project configurations.

Submit it:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/experiments" `
  -H "Content-Type: application/json" `
  --data-binary "@request.json"
```

Copy the returned `run_id`, then poll and retrieve results:

```powershell
curl.exe "http://127.0.0.1:8000/experiments/RUN_ID"
curl.exe "http://127.0.0.1:8000/experiments/RUN_ID/results"
curl.exe "http://127.0.0.1:8000/experiments/RUN_ID/metrics"
curl.exe "http://127.0.0.1:8000/experiments/RUN_ID/report"
```

Local runs perform model selection, refit the winner, evaluate the untouched
test set, and save predictions/model/report under `api_data/runs/<run_id>/`.

## 4. Enable the LANTA backend

Set these values in `.env` (or as Windows environment variables):

```dotenv
LANTA_HOST=lanta.nstda.or.th
LANTA_USERNAME=replace_with_your_username
LANTA_PROJECT_DIR=/project/YOUR_PROJECT_STORAGE/replace_with_your_username/CrewAI-rebuild
LANTA_PYTHON=.venv/bin/python
LANTA_SLURM_OPTIONS=configs/slurm_lanta.json
```

The Windows OpenSSH client must be able to connect without an interactive
password or OTP prompt. Configure a registered SSH public key or a working
SSH agent first, then verify:

```powershell
ssh -o BatchMode=yes YOUR_USERNAME@lanta.nstda.or.th "whoami"
```

The remote project must already contain this repository, its virtual
environment, Slurminade, and `configs/slurm_lanta.json`. Restart the API after
changing `.env`. Use the same experiment request with `"backend": "slurm"`.

The service uploads only the selected CSV plus generated JSON configuration,
submits one validation job for every registered model/fold, waits for the
Slurminade aggregate, and downloads `aggregate.json`. It does not upload the
Ollama API key. Version one's SLURM response contains distributed validation
metrics; it does not refit or download a final model.

## Endpoint summary

- `GET /health`
- `POST /datasets`
- `GET /datasets/{dataset_id}`
- `POST /experiments`
- `GET /experiments/{run_id}`
- `GET /experiments/{run_id}/results`
- `GET /experiments/{run_id}/metrics`
- `GET /experiments/{run_id}/report`

Uploads are limited by `EXPERIMENT_API_MAX_UPLOAD_BYTES` and
`EXPERIMENT_API_MAX_COLUMNS`. Duplicate or blank CSV headers, invalid timestamp
values, non-numeric configured targets or features, unknown columns, unsafe
model parameters and unconfigured SLURM submissions are rejected before
execution.

This is a localhost development service, not a public multi-user deployment.
Add authentication, TLS, per-user quotas, malware scanning, and a durable job
queue before exposing it to a network.
