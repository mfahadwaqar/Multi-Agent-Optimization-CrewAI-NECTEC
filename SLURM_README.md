# Optional SLURM experiment distribution

This layer sends independent deterministic forecast model/fold experiments to compute nodes. CrewAI and Ollama Cloud stay on the controller. Workers receive JSON-safe experiment data and identifiers, calculate real metrics, and save one atomic JSON file per job.

SLURM is useful when many model configurations and validation folds would take too long one after another. It schedules those independent jobs across cluster CPUs or GPUs. Slurminade is the small Python wrapper used here to submit ordinary top-level Python functions to SLURM. It does not create a cluster, train models itself, or return Python values from remote workers.

## Manifest contract

Every job requires:

```json
{
  "job_id": "weather-ridge-fold-1",
  "runner": "forecasting",
  "dataset_id": "jena_temperature_24h",
  "configuration_id": "ridge",
  "fold_id": 1,
  "config_path": "configs/weather_jena.json",
  "project_root": "."
}
```

Additional JSON-safe fields required by `forecasting.distributed.run_fold_job` may be included. Each `job_id` must be unique. Secret-like fields are rejected, and cloud API-key environment variables are hidden while worker code executes.

## Local test on Windows

The local backend needs no SLURM installation:

```powershell
python -m slurm.distribution --manifest configs/distributed_weather.json --backend local --run-id weather-local-001
```

Results are written under `results/distributed/weather-local-001/`. Each job has its own JSON file, and `aggregate.json` reports completed, failed, missing, and corrupt jobs.

## Linux SLURM cluster

Install the optional dependency in the environment shared by the controller and nodes:

```bash
python -m pip install -r requirements-slurm.txt
python -m slurminade.check --partition YOUR_PARTITION
```

Copy `configs/slurm_lanta.example.json` to the ignored
`configs/slurm_lanta.json`, then replace the project account with the allocation
assigned to you:

```json
{
  "partition": "compute",
  "account": "YOUR_PROJECT_ACCOUNT",
  "cpus_per_task": 2,
  "mem": "8G",
  "time": "01:00:00"
}
```

Submit and wait for all jobs:

```bash
python -m slurm.distribution \
  --manifest configs/distributed_weather.json \
  --backend slurm \
  --slurm-options configs/slurm_lanta.json \
  --run-id weather-cluster-001
```

The project directory, dataset files, result directory, and Python environment must be visible at the same paths on every worker. Choose site-approved partition, memory, CPU/GPU, and time values. Use `bundle_size` in the manifest for many very short tasks so the scheduler is not flooded with tiny jobs.

Do not copy `.env` to compute nodes. Generate any CrewAI recommendations on the controller first, convert the approved choices into a manifest, then distribute only deterministic training/evaluation jobs.
