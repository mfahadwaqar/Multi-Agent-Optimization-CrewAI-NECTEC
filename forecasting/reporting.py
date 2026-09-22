"""Concise forecasting report."""

from typing import Any


def build_report(profile: dict[str, Any], history: dict[str, Any], agent_review: str | None = None) -> str:
    best = history["best_result"]
    lines = [
        f"# {profile['name']} forecasting result",
        "",
        f"The system predicted **{profile['target_column']}** {profile['horizon']} time step(s) ahead.",
        f"It regularized {profile['regularized_rows']} timestamps and detected {profile['missing_timestamps']} missing timestamps.",
        "Inputs were forward-filled only from earlier observations; missing future labels were excluded.",
        "",
        "## Selected model",
        "",
        f"- Model: **{best['name']}**",
        f"- Walk-forward validation MASE: **{best['validation_mase']:.4f}**",
        f"- Test MAE: **{best['test_mae']:.4f}**",
        f"- Test RMSE: **{best['test_rmse']:.4f}**",
        f"- Test MASE: **{best['test_mase']:.4f}**",
        "",
        "Lower values are better. A test MASE below 1 means its average error was smaller than the average seasonal change measured in the training period.",
    ]
    if agent_review:
        lines.extend(["", "## Agent review", "", agent_review.strip()])
    return "\n".join(lines) + "\n"
