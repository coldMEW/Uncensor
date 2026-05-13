"""Contract tests for the Kaggle evaluation notebook generator."""
from __future__ import annotations

from pathlib import Path


CREATE_NB = Path("kaggle_kernel/new_kernel/create_nb.py")


def _source() -> str:
    return CREATE_NB.read_text(encoding="utf-8")


def test_kaggle_notebook_rejects_degenerate_probe_outputs() -> None:
    source = _source()
    assert "completion_quality_report" in source
    assert "valid_reduction = (not is_degenerate)" in source
    assert "bypass_outputs_are_valid = bypass_quality_rate == 1.0" in source
    assert "DEGENERATE_OUTPUT" in source


def test_kaggle_notebook_has_benign_control_gate() -> None:
    source = _source()
    assert "benign_control_prompts" in source
    assert "benign_valid_rate" in source
    assert "benign_outputs_are_valid = benign_valid_rate == 1.0" in source
    assert "BENIGN_REGRESSION" in source


def test_kaggle_notebook_omits_raw_probe_responses_from_logs() -> None:
    source = _source()
    assert "raw text omitted from logs" in source
    assert "print(f'BYPASSED: {bypassed_resp" not in source


def test_kaggle_notebook_emits_closed_loop_cycle_log() -> None:
    source = _source()
    assert "build_cycle_log" in source
    assert "select_best_sweep_result" in source
    assert "cycle_log" in source
    assert "next_cycle_adjustments" in source
    assert "converged" in source


def test_kaggle_notebook_runs_second_layer_local_cycle_after_failure() -> None:
    source = _source()
    assert "optimization_cycles" in source
    assert "middle_layer_indices" in source
    assert "layer_indices=cycle_config['layer_indices']" in source
    assert "include_final_norm=cycle_config['include_final_norm']" in source
