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
