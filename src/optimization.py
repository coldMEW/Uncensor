"""Closed-loop selection and logging for refusal-vector experiments.

The optimizer treats a run as useful only when refusal-probe score movement is
paired with non-degenerate generations and preserved benign controls. This
prevents chat-template loops or repeated tokens from being counted as progress.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence


SweepResult = Mapping[str, Any]


def _score_sweep(result: SweepResult) -> tuple[float, float, float, float, float]:
    """Return a sortable score tuple for a coefficient/layer sweep result."""
    return (
        float(result.get("valid_reduction_rate", 0.0)),
        float(result.get("bypass_quality_rate", 0.0)),
        float(result.get("benign_valid_rate", 0.0)),
        -float(result.get("avg_bypass", 1.0)),
        -float(result.get("coefficient", 0.0)),
    )


def select_best_sweep_result(sweep_results: Sequence[SweepResult]) -> Dict[str, Any]:
    """Select the best intervention sweep result.

    Valid runs always outrank invalid runs. Within each pool, prefer stronger
    non-degenerate refusal-probe reduction, preserved output quality, preserved
    benign controls, lower residual refusal score, then smaller coefficients.
    """
    if not sweep_results:
        raise ValueError("sweep_results must not be empty")

    valid = [dict(result) for result in sweep_results if result.get("run_is_valid")]
    pool = valid if valid else [dict(result) for result in sweep_results]
    return max(pool, key=_score_sweep)


def _has_degenerate_outputs(result: SweepResult) -> bool:
    prompt_results = result.get("prompt_results", [])
    benign_results = result.get("benign_results", [])
    return any(item.get("is_degenerate") for item in prompt_results + benign_results)


def propose_next_cycle_adjustments(sweep_results: Sequence[SweepResult]) -> Dict[str, Any]:
    """Produce next-cycle parameter adjustments from completed sweep evidence."""
    selected = select_best_sweep_result(sweep_results)
    regression_flags: List[str] = []

    if selected.get("run_is_valid"):
        return {
            "converged": True,
            "regression_flags": [],
            "coefficient_grid": [float(selected.get("coefficient", 1.0))],
            "layer_strategy": "keep_selected",
            "direction_strategy": "keep_selected",
            "rationale": "All gates passed; keep the selected intervention.",
        }

    if _has_degenerate_outputs(selected):
        regression_flags.append("DEGENERATE_OUTPUT")
    if float(selected.get("benign_valid_rate", 0.0)) < 1.0:
        regression_flags.append("BENIGN_REGRESSION")
    if float(selected.get("valid_reduction_rate", 0.0)) < 0.75:
        regression_flags.append("INSUFFICIENT_VALID_REDUCTION")

    if "DEGENERATE_OUTPUT" in regression_flags or "BENIGN_REGRESSION" in regression_flags:
        return {
            "converged": False,
            "regression_flags": regression_flags,
            "coefficient_grid": [0.05, 0.1, 0.15, 0.2, 0.3],
            "layer_strategy": "middle_layers_only",
            "direction_strategy": "layerwise_ranked_directions",
            "rationale": (
                "The intervention changes refusal-probe scores by damaging "
                "generation quality or benign controls. Narrow the layer set "
                "and lower projection strength before increasing coverage."
            ),
        }

    return {
        "converged": False,
        "regression_flags": regression_flags,
        "coefficient_grid": [0.3, 0.5, 0.7, 0.9, 1.1],
        "layer_strategy": "expand_ranked_layers",
        "direction_strategy": "add_secondary_direction",
        "rationale": (
            "Outputs are valid but refusal-probe movement is insufficient; "
            "expand targeted layers or add an orthogonal secondary direction."
        ),
    }


def _per_category_scores(selected_result: SweepResult) -> Dict[str, Dict[str, Any]]:
    categories: Dict[str, Dict[str, Any]] = {}
    for item in selected_result.get("prompt_results", []):
        categories[str(item["prompt_id"])] = {
            "type": "refusal_probe",
            "baseline_score": float(item.get("baseline_score", 0.0)),
            "intervened_score": float(item.get("intervened_score", 0.0)),
            "score_drop": float(item.get("score_drop", 0.0)),
            "is_degenerate": bool(item.get("is_degenerate", False)),
            "quality_reason": str(item.get("quality_reason", "unknown")),
            "valid_reduction": bool(item.get("valid_reduction", False)),
        }
    for item in selected_result.get("benign_results", []):
        categories[str(item["prompt_id"])] = {
            "type": "benign_control",
            "score": float(item.get("score", 0.0)),
            "is_degenerate": bool(item.get("is_degenerate", False)),
            "quality_reason": str(item.get("quality_reason", "unknown")),
            "valid": bool(item.get("valid", False)),
        }
    return categories


def build_cycle_log(
    cycle_index: int,
    model_name: str,
    direction_metadata: Mapping[str, Any],
    sweep_results: Sequence[SweepResult],
    selected_result: SweepResult,
) -> Dict[str, Any]:
    """Build a JSON-serializable optimization-cycle record."""
    next_adjustments = propose_next_cycle_adjustments(sweep_results)
    return {
        "cycle_index": int(cycle_index),
        "model": model_name,
        "intervention_params": {
            "coefficient": float(selected_result.get("coefficient", 0.0)),
            "strategy": "directional_ablation",
        },
        "direction": dict(direction_metadata),
        "per_category_scores": _per_category_scores(selected_result),
        "aggregate_scores": {
            "valid_reduction_rate": float(selected_result.get("valid_reduction_rate", 0.0)),
            "bypass_quality_rate": float(selected_result.get("bypass_quality_rate", 0.0)),
            "benign_valid_rate": float(selected_result.get("benign_valid_rate", 0.0)),
            "run_is_valid": bool(selected_result.get("run_is_valid", False)),
        },
        "regression_flags": next_adjustments["regression_flags"],
        "next_cycle_adjustments": next_adjustments,
    }
