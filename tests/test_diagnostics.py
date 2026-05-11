"""Tests for diagnostics module."""

import pytest
import torch
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import json

from src.diagnostics import (
    DiagnosticLogger,
    DiagnosticEntry,
    DiagnosticReport,
    ErrorCategory,
    Severity,
    ResultComparator,
    ValidationResult,
    generate_diagnostic_report,
    get_logger,
    log_error,
)


class TestDiagnosticLogger:
    """Test DiagnosticLogger functionality."""

    def test_init_creates_session_id(self):
        logger = DiagnosticLogger()
        assert logger.session_id is not None
        assert len(logger.session_id) == 12

    def test_init_with_custom_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DiagnosticLogger(output_dir=Path(tmpdir))
            assert logger.output_dir == Path(tmpdir)

    def test_log_creates_entry(self):
        logger = DiagnosticLogger()
        entry = logger.log(
            ErrorCategory.MODEL_LOAD,
            Severity.ERROR,
            "Test error",
            {"key": "value"},
        )
        assert entry.category == ErrorCategory.MODEL_LOAD.value
        assert entry.severity == Severity.ERROR.value
        assert entry.message == "Test error"
        assert entry.context == {"key": "value"}

    def test_log_error_creates_error_entry(self):
        logger = DiagnosticLogger()
        entry = logger.log_error(
            ErrorCategory.CUDA_OOM,
            "Out of memory",
            {"available": "0GB", "required": "14GB"},
        )
        assert entry.severity == Severity.ERROR.value
        assert entry.category == ErrorCategory.CUDA_OOM.value

    def test_log_warning_creates_warning_entry(self):
        logger = DiagnosticLogger()
        entry = logger.log_warning(
            ErrorCategory.VALIDATION,
            "Missing parameter",
            {"param": "coefficient"},
        )
        assert entry.severity == Severity.WARNING.value

    def test_log_info_creates_info_entry(self):
        logger = DiagnosticLogger()
        entry = logger.log_info("Test info message", {"test": True})
        assert entry.severity == Severity.INFO.value

    def test_log_output_mismatch(self):
        logger = DiagnosticLogger()
        entry = logger.log_output_mismatch(
            expected=0.8,
            actual=0.3,
            context={"bypass_rate": "low"},
        )
        assert entry.category == ErrorCategory.OUTPUT_MISMATCH.value
        assert entry.context["expected"] == 0.8
        assert entry.context["actual"] == 0.3

    def test_capture_system_info(self):
        logger = DiagnosticLogger()
        info = logger.capture_system_info()
        assert "python_version" in info
        assert "torch_version" in info
        assert "cuda_available" in info

    def test_generate_report(self):
        logger = DiagnosticLogger()
        logger.log_error(ErrorCategory.MODEL_LOAD, "Test error")
        report = logger.generate_report()
        assert report.session_id == logger.session_id
        assert len(report.entries) == 1
        assert report.entries[0].message == "Test error"

    def test_save_report_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DiagnosticLogger(output_dir=Path(tmpdir))
            logger.log_error(ErrorCategory.MODEL_LOAD, "Test error")
            path = logger.save_report(format="json")
            assert path.suffix == ".json"
            assert path.exists()

            with open(path) as f:
                data = json.load(f)
                assert data["session_id"] == logger.session_id

    def test_save_report_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DiagnosticLogger(output_dir=Path(tmpdir))
            logger.log_error(ErrorCategory.MODEL_LOAD, "Test error")
            path = logger.save_report(format="md")
            assert path.suffix == ".md"
            assert path.exists()

            content = path.read_text(encoding="utf-8")
            assert "# Diagnostic Report" in content
            assert "Test error" in content


class TestDiagnosticReport:
    """Test DiagnosticReport generation."""

    def test_to_dict(self):
        report = DiagnosticReport(
            report_id="test123",
            created_at="2024-01-01T00:00:00",
            session_id="abc",
            entries=[],
            system_info={},
            pipeline_state={},
            metrics_summary={},
        )
        data = report.to_dict()
        assert data["report_id"] == "test123"
        assert data["session_id"] == "abc"

    def test_to_markdown_generates_suggestions(self):
        entry = DiagnosticEntry(
            timestamp="2024-01-01T00:00:00",
            category=ErrorCategory.CUDA_OOM.value,
            severity=Severity.ERROR.value,
            message="Out of memory",
        )
        report = DiagnosticReport(
            report_id="test123",
            created_at="2024-01-01T00:00:00",
            session_id="abc",
            entries=[entry],
        )
        md = report.to_markdown()
        assert "## Suggested Actions" in md
        assert "CUDA OOM" in md


class TestResultComparator:
    """Test ResultComparator functionality."""

    def test_validate_bypass_rate_pass(self):
        comparator = ResultComparator()
        result = comparator.validate_bypass_rate(actual=0.5, expected_min=0.3)
        assert result.passed
        assert result.metric == "bypass_rate"

    def test_validate_bypass_rate_fail(self):
        comparator = ResultComparator()
        result = comparator.validate_bypass_rate(actual=0.2, expected_min=0.3)
        assert not result.passed

    def test_validate_refusal_score_pass(self):
        comparator = ResultComparator()
        result = comparator.validate_refusal_score(
            actual=0.3, baseline=0.9, expected_drop=0.5
        )
        assert result.passed
        assert abs(result.actual - 0.6) < 0.001  # 0.9 - 0.3 (allow float precision)

    def test_validate_refusal_score_fail(self):
        comparator = ResultComparator()
        result = comparator.validate_refusal_score(
            actual=0.7, baseline=0.9, expected_drop=0.5
        )
        assert not result.passed
        assert abs(result.actual - 0.2) < 0.001  # 0.9 - 0.7

    def test_validate_capability_preservation_pass(self):
        comparator = ResultComparator()
        result = comparator.validate_capability_preservation(
            before=0.50, after=0.48, max_drop=0.05
        )
        assert result.passed
        assert abs(result.actual - 0.02) < 0.001  # 0.50 - 0.48

    def test_validate_capability_preservation_fail(self):
        comparator = ResultComparator()
        result = comparator.validate_capability_preservation(
            before=0.50, after=0.35, max_drop=0.05
        )
        assert not result.passed

    def test_validate_direction_shape_pass(self):
        comparator = ResultComparator()
        direction = torch.randn(2048)
        result = comparator.validate_direction_shape(direction, expected_dim=2048)
        assert result.passed

    def test_validate_direction_shape_fail(self):
        comparator = ResultComparator()
        direction = torch.randn(2048)
        result = comparator.validate_direction_shape(direction, expected_dim=4096)
        assert not result.passed

    def test_validate_direction_norm_pass(self):
        comparator = ResultComparator()
        direction = torch.randn(2048)
        direction = direction / direction.norm()
        result = comparator.validate_direction_norm(direction)
        assert result.passed

    def test_validate_direction_norm_fail(self):
        comparator = ResultComparator()
        direction = torch.randn(2048) * 2.5
        result = comparator.validate_direction_norm(direction)
        assert not result.passed

    def test_summary(self):
        comparator = ResultComparator()
        comparator.validate_bypass_rate(actual=0.5, expected_min=0.3)
        comparator.validate_bypass_rate(actual=0.2, expected_min=0.3)
        summary = comparator.summary()
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1


class TestGlobalFunctions:
    """Test module-level convenience functions."""

    def test_get_logger_returns_singleton(self):
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2

    def test_log_error_uses_global_logger(self):
        entry = log_error(
            ErrorCategory.MODEL_LOAD,
            "Global test error",
        )
        assert entry.category == ErrorCategory.MODEL_LOAD.value


class TestIntegration:
    """Integration tests for diagnostics with pipeline."""

    def test_diagnostics_capture_pipeline_state(self):
        logger = DiagnosticLogger()
        state = {
            "model_name": "Qwen/Qwen2-0.5B-Instruct",
            "direction_extracted": True,
            "layer": 12,
            "position": -1,
        }
        logger.capture_pipeline_state(state)
        report = logger.generate_report()
        assert report.pipeline_state == state

    def test_diagnostics_capture_metrics(self):
        logger = DiagnosticLogger()
        metrics = {
            "bypass_rate": 0.65,
            "baseline_refusal": 0.92,
            "modified_refusal": 0.35,
        }
        logger.capture_metrics_summary(metrics)
        report = logger.generate_report()
        assert report.metrics_summary == metrics

    def test_full_diagnostic_workflow(self):
        """Test complete diagnostic workflow from error to report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DiagnosticLogger(output_dir=Path(tmpdir))
            logger.capture_system_info()

            # Simulate pipeline errors
            logger.log_error(
                ErrorCategory.MODEL_LOAD,
                "Failed to load model",
                {"model": "Qwen/Qwen2-0.5B-Instruct", "error": "CUDA OOM"},
            )
            logger.log_warning(
                ErrorCategory.VALIDATION,
                "Direction norm outside expected range",
                {"norm": 0.85, "expected": 1.0},
            )

            # Generate and save report
            json_path = logger.save_report(format="json")
            md_path = logger.save_report(format="md")

            # Verify JSON report
            with open(json_path) as f:
                data = json.load(f)
                assert len(data["entries"]) == 2
                assert data["system_info"]["cuda_available"] is not None

            # Verify markdown report
            md_content = md_path.read_text(encoding="utf-8")
            assert "## Errors/Warnings" in md_content
            assert "## Suggested Actions" in md_content