"""Small, interpretable seasonality diagnostics computed on development data only."""

import numpy as np
import pandas as pd


def analyze_seasonality(series: pd.Series, period: int) -> dict[str, float | int | None]:
    clean = pd.to_numeric(series, errors="coerce").ffill().dropna()
    period = max(1, int(period))
    if len(clean) <= period + 2:
        return {"period": period, "lag_correlation": None, "seasonal_naive_mae": None}
    current = clean.iloc[period:].to_numpy(dtype=float)
    lagged = clean.iloc[:-period].to_numpy(dtype=float)
    correlation = float(np.corrcoef(current, lagged)[0, 1])
    return {
        "period": period,
        "lag_correlation": correlation if np.isfinite(correlation) else None,
        "seasonal_naive_mae": float(np.mean(np.abs(current - lagged))),
    }
