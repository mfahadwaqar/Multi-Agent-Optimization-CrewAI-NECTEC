from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from dotenv import load_dotenv

from .executors import LantaExecutor, LantaSettings, LocalExecutor
from .models import ExperimentRequest
from .service import ExperimentService
from .storage import JsonStore


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


def _public_dataset(record: dict) -> dict:
    return {key: value for key, value in record.items() if key != "path"}


def _default_service() -> ExperimentService:
    root = Path(os.getenv("EXPERIMENT_API_DATA", "api_data"))
    store = JsonStore(root, max_upload_bytes=int(os.getenv("EXPERIMENT_API_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024))))
    try:
        lanta = LantaExecutor(LantaSettings.from_env())
    except RuntimeError:
        lanta = None
    return ExperimentService(store, local_executor=LocalExecutor(), lanta_executor=lanta)


def create_app(service: ExperimentService | None = None) -> FastAPI:
    app = FastAPI(title="Time-Series Experiment API", version="1.0.0")
    app.state.service = service or _default_service()

    @app.get("/health")
    def health():
        return {"status": "ok", "backends": ["local", "slurm"]}

    @app.post("/datasets", status_code=status.HTTP_201_CREATED)
    async def upload_dataset(file: UploadFile = File(...)):
        try:
            maximum = app.state.service.store.max_upload_bytes
            content = await file.read(maximum + 1)
            return _public_dataset(
                app.state.service.store.save_dataset(file.filename or "upload.csv", content)
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/datasets/{dataset_id}")
    def get_dataset(dataset_id: str):
        try:
            return _public_dataset(app.state.service.store.get_dataset(dataset_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc

    @app.post("/experiments", status_code=status.HTTP_202_ACCEPTED)
    def submit_experiment(request: ExperimentRequest):
        try:
            return app.state.service.submit(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/experiments/{run_id}")
    def get_experiment(run_id: str):
        try:
            record = app.state.service.store.get_run(run_id)
            return {key: value for key, value in record.items() if key != "request"}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Experiment not found") from exc

    @app.get("/experiments/{run_id}/results")
    def get_results(run_id: str):
        try:
            record = app.state.service.store.get_run(run_id)
            if record["status"] != "completed":
                raise HTTPException(status_code=409, detail=f"Experiment is {record['status']}")
            return app.state.service.store.get_result(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Experiment not found") from exc

    return app


app = create_app()
