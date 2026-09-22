"""Command-line entry point for weather and water-level forecasting."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from forecasting.workflow import run_forecast


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a leakage-safe forecasting experiment")
    parser.add_argument("--config", required=True, help="Weather or groundwater JSON configuration")
    parser.add_argument("--use-agents", action="store_true", help="Add a CrewAI review through Ollama Cloud")
    parser.add_argument("--refresh", action="store_true", help="Refresh a remote USGS cache")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=True)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    history = run_forecast(config, use_agents=args.use_agents, refresh=args.refresh)
    best = history["best_result"]
    destination = ROOT / "results" / "forecasting" / config["data"]["name"]
    print(f"Best model: {best['name']}")
    print(f"Validation MASE: {best['validation_mase']:.4f}")
    print(f"Test MAE: {best['test_mae']:.4f}")
    print(f"Test RMSE: {best['test_rmse']:.4f}")
    print(f"Test MASE: {best['test_mase']:.4f}")
    print(f"Report: {destination / 'summary.md'}")


if __name__ == "__main__":
    main()
