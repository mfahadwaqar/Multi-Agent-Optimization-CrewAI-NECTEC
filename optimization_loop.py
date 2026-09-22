"""Coordinate preparation, experiments, selection and reporting."""

import itertools
import json
from pathlib import Path
from typing import Any

from agent_team import create_agent_report, request_architecture
from dataset_pipeline import prepare_dataset
from evaluation import choose_best
from model_runner import evaluate_and_save, run_baselines, run_neural_trial
from reporting import build_report


ROOT = Path(__file__).resolve().parent


def _public(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "model"}


def _configurations(space: dict[str, Any]):
    for values in itertools.product(
        space["layers"], space["activations"], space["optimizers"],
        space["learning_rates"], space["batch_sizes"], space["epochs"],
    ):
        yield dict(zip(["layers", "activation", "optimizer", "learning_rate", "batch_size", "epochs"], values))


def _validate(candidate: dict[str, Any], space: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "layers": space["layers"], "activation": space["activations"],
        "optimizer": space["optimizers"], "learning_rate": space["learning_rates"],
        "batch_size": space["batch_sizes"], "epochs": space["epochs"],
    }
    for key, choices in allowed.items():
        if candidate.get(key) not in choices:
            raise ValueError(f"Agent returned invalid {key}: {candidate.get(key)!r}")
    return candidate


def run(config: dict[str, Any], use_agents: bool = False, output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays, profile = prepare_dataset(config, ROOT / "artifacts")
    seed = int(config.get("training", {}).get("random_seed", 42))
    results = run_baselines(arrays, seed)
    iterations = int(config.get("optimization", {}).get("iterations", 3))
    generated = _configurations(config["search_space"])

    for number in range(1, iterations + 1):
        if use_agents:
            candidate = request_architecture(profile, [_public(item) for item in results], config["search_space"])
            candidate = _validate(candidate, config["search_space"])
        else:
            candidate = next(generated)
        results.append(run_neural_trial(arrays, candidate, seed + number, number))

    minimum = float(config.get("optimization", {}).get("minimum_improvement", 0.0))
    best_with_model = choose_best(results, minimum)
    final = evaluate_and_save(best_with_model, arrays, output_dir)
    experiments = [_public(item) for item in results]
    history = {"dataset": profile, "experiments": experiments, "best_result": final}
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    agent_review = create_agent_report(profile, final) if use_agents else None
    count = sum(item["status"] == "completed" for item in experiments)
    (output_dir / "summary.md").write_text(build_report(profile, final, count, agent_review), encoding="utf-8")
    return history
