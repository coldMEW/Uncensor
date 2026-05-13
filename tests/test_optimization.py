"""Tests for closed-loop intervention selection and cycle logging."""
from __future__ import annotations

from src.optimization import (
    build_cycle_log,
    propose_next_cycle_adjustments,
    select_best_sweep_result,
)


def _sweep(
    coefficient: float,
    valid_reduction_rate: float,
    bypass_quality_rate: float,
    benign_valid_rate: float,
    run_is_valid: bool = False,
):
    return {
        "coefficient": coefficient,
        "valid_reduction_rate": valid_reduction_rate,
        "bypass_quality_rate": bypass_quality_rate,
        "benign_valid_rate": benign_valid_rate,
        "avg_bypass": 0.0,
        "run_is_valid": run_is_valid,
        "prompt_results": [
            {
                "prompt_id": "refusal_probe_1",
                "baseline_score": 0.6,
                "intervened_score": 0.0,
                "score_drop": 0.6,
                "is_degenerate": bypass_quality_rate < 1.0,
                "quality_reason": "chat_template_loop" if bypass_quality_rate < 1.0 else "ok",
                "valid_reduction": valid_reduction_rate >= 1.0,
            }
        ],
        "benign_results": [
            {
                "prompt_id": "benign_probe_1",
                "score": 0.0,
                "is_degenerate": benign_valid_rate < 1.0,
                "quality_reason": "repeated_token" if benign_valid_rate < 1.0 else "ok",
                "valid": benign_valid_rate >= 1.0,
            }
        ],
    }


def test_select_best_sweep_result_prefers_valid_runs() -> None:
    invalid_high_delta = _sweep(1.0, 1.0, 0.75, 0.0)
    valid_lower_delta = _sweep(0.4, 0.75, 1.0, 1.0, run_is_valid=True)

    selected = select_best_sweep_result([invalid_high_delta, valid_lower_delta])

    assert selected["coefficient"] == 0.4
    assert selected["run_is_valid"] is True


def test_propose_next_cycle_adjustments_shrinks_when_generation_degenerates() -> None:
    sweeps = [
        _sweep(0.25, 0.0, 0.0, 0.0),
        _sweep(0.50, 0.0, 0.0, 0.0),
        _sweep(0.75, 0.25, 0.25, 0.0),
        _sweep(1.00, 0.75, 0.75, 0.0),
    ]

    adjustment = propose_next_cycle_adjustments(sweeps)

    assert adjustment["converged"] is False
    assert "DEGENERATE_OUTPUT" in adjustment["regression_flags"]
    assert "BENIGN_REGRESSION" in adjustment["regression_flags"]
    assert adjustment["coefficient_grid"] == [0.05, 0.1, 0.15, 0.2, 0.3]
    assert adjustment["layer_strategy"] == "middle_layers_only"


def test_build_cycle_log_records_categories_and_next_adjustment() -> None:
    sweeps = [_sweep(1.0, 0.75, 0.75, 0.0)]
    selected = select_best_sweep_result(sweeps)

    cycle = build_cycle_log(
        cycle_index=1,
        model_name="test-model",
        direction_metadata={"shape": [4], "norm": 1.0},
        sweep_results=sweeps,
        selected_result=selected,
    )

    assert cycle["cycle_index"] == 1
    assert cycle["intervention_params"]["coefficient"] == 1.0
    assert cycle["per_category_scores"]["refusal_probe_1"]["valid_reduction"] is False
    assert cycle["per_category_scores"]["benign_probe_1"]["valid"] is False
    assert "BENIGN_REGRESSION" in cycle["regression_flags"]
    assert cycle["next_cycle_adjustments"]["converged"] is False
