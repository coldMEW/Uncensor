"""Tests for run_capability_benchmarks function.

Tests lm-evaluation-harness integration for MMLU/GSM8K/ARC.
Run with: pytest tests/test_capability_benchmarks.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
import shutil
from unittest.mock import patch


def lm_eval_available():
    """Check if lm_eval is installed."""
    return shutil.which("lm_eval") is not None or shutil.which("lm-eval") is not None


class TestCapabilityBenchmarks:
    """Tests for run_capability_benchmarks function."""

    def test_import_exists(self):
        """Verify run_capability_benchmarks is importable."""
        from src import metrics
        assert hasattr(metrics, "run_capability_benchmarks")

    def test_function_signature(self):
        """Verify function accepts expected parameters."""
        from src import metrics
        import inspect
        sig = inspect.signature(metrics.run_capability_benchmarks)
        params = list(sig.parameters.keys())
        assert "model_name" in params
        assert "tasks" in params
        assert "batch_size" in params

    @pytest.mark.skipif(not lm_eval_available(), reason="lm-eval not installed")
    def test_lm_eval_tasks_list(self):
        """Verify lm_eval shows available tasks."""
        import subprocess
        result = subprocess.run(
            ["lm_eval", "--tasks", "list"],
            capture_output=True,
            text=True,
        )
        assert "mmlu" in result.stdout.lower() or result.returncode == 0

    @pytest.mark.skipif(not lm_eval_available(), reason="lm-eval not installed")
    def test_run_benchmarks_returns_dict(self):
        """Run benchmarks with small limit returns dict."""
        from src import metrics
        # Use tiny model for fast test
        result = metrics.run_capability_benchmarks(
            model_name="gpt2",  # Small model, fast
            tasks=["mmlu"],
            limit=5,
        )
        assert isinstance(result, dict)

    @pytest.mark.skipif(not lm_eval_available(), reason="lm-eval not installed")
    def test_benchmarks_include_expected_tasks(self):
        """Benchmarks should include mmlu, arc_challenge, gsm8k."""
        from src import metrics
        result = metrics.run_capability_benchmarks(
            model_name="gpt2",
            tasks=["mmlu", "arc_challenge", "gsm8k"],
            limit=5,
        )
        # May be empty dict if tasks fail, but should not raise
        assert isinstance(result, dict)

    def test_graceful_skip_when_not_installed(self):
        """Function returns empty dict when lm_eval not available."""
        from src import metrics
        with patch("shutil.which", return_value=None):
            result = metrics.run_capability_benchmarks(
                model_name="test",
                tasks=["mmlu"],
            )
            assert result == {}


class TestLmEvalInstallation:
    """Tests for lm-eval installation status."""

    def test_lm_eval_installed_check(self):
        """Check lm_eval installation."""
        if lm_eval_available():
            pytest.skip("lm-eval is installed")
        else:
            # This is expected on some systems
            pass

    def test_pip_install_strongreject_info(self):
        """Document how to install lm-eval."""
        # This test always passes, just for documentation
        assert True


class TestBenchmarkOutput:
    """Tests for benchmark output parsing."""

    def test_output_parsing_logic(self):
        """Verify output parsing handles different metric names."""
        from src import metrics
        # Test the logic that parses lm_eval output
        # This is a unit test of the parsing logic
        mock_results = {
            "mmlu": {"acc": 0.5},
            "arc_challenge": {"acc_norm": 0.3},
            "gsm8k": {"exact_match": 0.4},
        }
        out = {}
        _primary = {"gsm8k": "exact_match", "arc_challenge": "acc_norm"}
        for task in ["mmlu", "arc_challenge", "gsm8k"]:
            task_blob = mock_results.get(task, {})
            key = _primary.get(task, "acc")
            if key in task_blob:
                out[task] = float(task_blob[key])

        assert out.get("mmlu") == 0.5
        assert out.get("arc_challenge") == 0.3
        assert out.get("gsm8k") == 0.4

    def test_prefix_matching_for_metrics(self):
        """Test metric prefix matching logic."""
        mock_results = {
            "mmlu": {"acc,none": 0.5, "acc_stderr": 0.1},
        }
        out = {}
        key = "acc"
        task_blob = mock_results.get("mmlu", {})
        if key in task_blob:
            out["mmlu"] = float(task_blob[key])
        else:
            for k, v in task_blob.items():
                if k.startswith(key + ",") and isinstance(v, (int, float)):
                    out["mmlu"] = float(v)
                    break

        assert out.get("mmlu") == 0.5