import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from optimization_loop import run


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CrewAI classification experiment")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--use-agents", action="store_true", help="Use Ollama Cloud through CrewAI")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    history = run(config, use_agents=args.use_agents)
    best = history["best_result"]
    print(f"Best model: {best['name']}")
    print(f"Validation accuracy: {best['validation_accuracy']:.4f}")
    print(f"Test accuracy: {best['test_accuracy']:.4f}")
    print(f"Report: {ROOT / 'results' / 'summary.md'}")


if __name__ == "__main__":
    main()
