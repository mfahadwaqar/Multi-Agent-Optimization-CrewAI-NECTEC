"""Leakage-safe forecasting workflow for weather and groundwater data."""


def run_forecast(*args, **kwargs):
    """Load the modeling stack only when an experiment is actually requested."""
    from .workflow import run_forecast as _run_forecast

    return _run_forecast(*args, **kwargs)


__all__ = ["run_forecast"]
