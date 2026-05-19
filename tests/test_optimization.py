"""Tests for closed-loop intervention selection and cycle logging."""
from __future__ import annotations

from src.optimization import (
    build_cycle_log,
    build_intervention_candidates,
    build_run_summary,
    constrained_candidate_score,
    meaningful_improvement,
    propose_next_cycle_adjustments,
    select_best_sweep_result,
    should_stop_evaluation_budget,
    should_stop_search,
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


def test_build_intervention_candidates_crosses_direction_layer_and_intervention_axes() -> None:
    candidates = build_intervention_candidates(
        direction_families=["svd", "cosine"],
        direction_counts=[1, 2],
        layer_windows={"core": [31, 32, 33], "wide": [30, 31, 32, 33, 34]},
        coefficients=[0.1, 0.2],
        intervention_types=["hook_ablation", "layer_weighted_orthogonalization"],
        include_final_norm=False,
    )

    assert len(candidates) == 32
    assert candidates[0]["direction_family"] == "svd"
    assert candidates[0]["direction_count"] == 1
    assert candidates[0]["layer_window_name"] == "core"
    assert candidates[0]["include_final_norm"] is False
    assert {c["intervention_type"] for c in candidates} == {
        "hook_ablation",
        "layer_weighted_orthogonalization",
    }


def test_constrained_candidate_score_rejects_regressions_before_movement() -> None:
    unsafe_high_movement = _sweep(0.8, 1.0, 0.75, 1.0)
    safe_low_movement = _sweep(0.2, 0.25, 1.0, 1.0)

    assert constrained_candidate_score(safe_low_movement) > constrained_candidate_score(
        unsafe_high_movement
    )


def test_meaningful_improvement_ignores_tiny_candidate_tie_breakers() -> None:
    previous = _sweep(0.2, 0.25, 1.0, 1.0)
    tiny_gain = _sweep(0.3, 0.30, 1.0, 1.0)

    assert constrained_candidate_score(tiny_gain) > constrained_candidate_score(previous)
    assert not meaningful_improvement(previous, tiny_gain)


def test_meaningful_improvement_accepts_valid_gate_transition() -> None:
    previous = _sweep(0.2, 0.50, 1.0, 1.0)
    now_valid = _sweep(0.3, 0.75, 1.0, 1.0, run_is_valid=True)

    assert meaningful_improvement(previous, now_valid)


def test_should_stop_search_reports_first_applicable_budget_reason() -> None:
    assert should_stop_search(
        converged=True,
        completed_cycles=0,
        max_cycles=2,
        stagnant_cycles=0,
        max_stagnant_cycles=1,
    ) == (True, "CONVERGED")
    assert should_stop_search(
        converged=False,
        completed_cycles=0,
        max_cycles=2,
        stagnant_cycles=0,
        max_stagnant_cycles=1,
        elapsed_seconds=120.0,
        max_seconds=60.0,
    ) == (True, "TIME_BUDGET_EXHAUSTED")
    assert should_stop_search(
        converged=False,
        completed_cycles=1,
        max_cycles=2,
        stagnant_cycles=1,
        max_stagnant_cycles=1,
    ) == (True, "NO_MEANINGFUL_IMPROVEMENT")


def test_should_stop_evaluation_budget_bounds_expensive_generation_calls() -> None:
    assert should_stop_evaluation_budget(
        completed_evaluations=95,
        max_evaluations=96,
        elapsed_seconds=120.0,
        max_seconds=1_800.0,
    ) == (False, "CONTINUE")
    assert should_stop_evaluation_budget(
        completed_evaluations=96,
        max_evaluations=96,
        elapsed_seconds=120.0,
        max_seconds=1_800.0,
    ) == (True, "EVALUATION_BUDGET_EXHAUSTED")
    assert should_stop_evaluation_budget(
        completed_evaluations=1,
        max_evaluations=96,
        elapsed_seconds=1_800.0,
        max_seconds=1_800.0,
    ) == (True, "TIME_BUDGET_EXHAUSTED")


def test_build_run_summary_records_dataset_and_judge_status() -> None:
    summary = build_run_summary(
        prompt_source="hf_datasets",
        train_counts={"harmful": 512, "harmless": 512},
        eval_counts={"refusal_probe": 128, "benign_control": 128},
        judge_backend="substring_stub",
        judge_is_official=False,
        best_candidate={"candidate_id": "c001"},
        rejected_candidates=[{"candidate_id": "c002", "rejection_reasons": ["DEGENERATE_OUTPUT"]}],
        category_metrics={"cyber": {"valid_reduction_rate": 0.0}},
        converged=False,
    )

    assert summary["prompt_source"] == "hf_datasets"
    assert summary["eval_counts"]["refusal_probe"] == 128
    assert summary["judge"]["status"] == "UNVERIFIED_JUDGE"
    assert summary["best_candidate"]["candidate_id"] == "c001"
    assert summary["rejected_candidate_count"] == 1
