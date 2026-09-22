"""Create a short report from measured results."""

from typing import Any


def build_report(profile: dict[str, Any], best: dict[str, Any], result_count: int, agent_review: str | None = None) -> str:
    lines = [
        "# Experiment summary", "",
        f"Dataset: **{profile['name']}** ({profile['usable_rows']} usable rows, {profile['input_features']} prepared features).",
        f"Missing feature values found: **{profile['missing_feature_values']}**.", "",
        "## Result", "",
        f"Compared {result_count} completed models using validation accuracy.",
        f"Selected **{best['name']}** with validation accuracy **{best['validation_accuracy']:.4f}**.",
        f"Final test accuracy: **{best['test_accuracy']:.4f}**; macro F1: **{best['test_macro_f1']:.4f}**.", "",
        "The test set was used only after model selection, so it remained an independent final check.",
    ]
    if agent_review:
        lines.extend(["", "## Agent review", "", agent_review.strip()])
    return "\n".join(lines) + "\n"
