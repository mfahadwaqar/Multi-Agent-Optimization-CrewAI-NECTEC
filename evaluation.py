"""Rules for fair model comparison."""

from typing import Any


def choose_best(results: list[dict[str, Any]], minimum_improvement: float = 0.0) -> dict[str, Any]:
    completed = [result for result in results if result.get("status") == "completed"]
    if not completed:
        raise ValueError("No completed experiment results")
    best = completed[0]
    for candidate in completed[1:]:
        if candidate["validation_accuracy"] > best["validation_accuracy"] + minimum_improvement:
            best = candidate
    return best
