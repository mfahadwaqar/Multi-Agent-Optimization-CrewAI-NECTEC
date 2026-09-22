from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelRequest(BaseModel):
    name: Literal["ridge", "random_forest"]
    params: dict[str, Any] = Field(default_factory=dict)


class ExperimentRequest(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=64)
    backend: Literal["local", "slurm"] = "local"
    timestamp_column: str = Field(min_length=1, max_length=256)
    target_column: str = Field(min_length=1, max_length=256)
    feature_columns: list[str] = Field(default_factory=list, max_length=100)
    frequency: str = Field(default="D", min_length=1, max_length=16)
    horizon: int = Field(default=1, ge=1, le=10_000)
    seasonality: int = Field(default=1, ge=1, le=100_000)
    lags: list[int] = Field(default_factory=lambda: [1], max_length=100)
    rolling_windows: list[int] = Field(default_factory=list, max_length=50)
    calendar_features: list[Literal["hour", "day_of_week", "day_of_year"]] = Field(default_factory=list)
    folds: int = Field(default=3, ge=1, le=20)
    validation_size: int = Field(default=30, ge=1, le=1_000_000)
    minimum_train_size: int = Field(default=100, ge=10, le=10_000_000)
    test_fraction: float = Field(default=0.2, gt=0.0, lt=0.5)
    random_seed: int = Field(default=42, ge=0, le=2_147_483_647)
    models: list[ModelRequest] = Field(
        default_factory=lambda: [ModelRequest(name="ridge"), ModelRequest(name="random_forest")],
        min_length=1,
        max_length=10,
    )

    @field_validator("lags")
    @classmethod
    def positive_unique_lags(cls, values: list[int]) -> list[int]:
        if any(value < 1 or value > 100_000 for value in values):
            raise ValueError("lags must be between 1 and 100000")
        if len(set(values)) != len(values):
            raise ValueError("lags must be unique")
        return values

    @field_validator("rolling_windows")
    @classmethod
    def valid_windows(cls, values: list[int]) -> list[int]:
        if any(value < 2 or value > 100_000 for value in values):
            raise ValueError("rolling windows must be between 2 and 100000")
        if len(set(values)) != len(values):
            raise ValueError("rolling windows must be unique")
        return values

    @model_validator(mode="after")
    def distinct_columns(self):
        if self.timestamp_column == self.target_column:
            raise ValueError("timestamp and target columns must differ")
        return self

