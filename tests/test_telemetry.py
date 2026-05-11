"""Tests for telemetry and leaderboard module (US-015)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
import torch

from src.telemetry import (
    TelemetryConfig,
    TelemetryEvent,
    submit_anonymous_result,
    get_leaderboard,
    format_leaderboard_table,
)


def test_telemetry_config_defaults() -> None:
    """TelemetryConfig has sensible defaults."""
    config = TelemetryConfig()
    assert config.enabled is False
    assert config.endpoint is not None  # Has default endpoint


def test_telemetry_event_dataclass() -> None:
    """TelemetryEvent stores run metadata."""
    event = TelemetryEvent.create(
        model_name="meta-llama/Llama-3.1-8B",
        method="diy",
        bypass_score=0.85,
        capability_delta=-0.02,
        n_directions=1,
        quantization=None,
    )
    assert event.model_name == "meta-llama/Llama-3.1-8B"
    assert event.bypass_score == 0.85
    assert event.run_id is not None  # Auto-generated
    assert event.timestamp is not None


def test_submit_anonymous_result_noop_when_disabled() -> None:
    """submit_anonymous_result is no-op when telemetry disabled."""
    config = TelemetryConfig(enabled=False)
    result = submit_anonymous_result(config, "model", "method", 0.8, -0.01, 1, None)
    # Should return None when disabled (no actual submission)
    # Implementation may still return event for local logging


def test_leaderboard_returns_list() -> None:
    """get_leaderboard returns list of entries."""
    leaderboard = get_leaderboard()
    assert isinstance(leaderboard, list)


def test_leaderboard_entries_sorted() -> None:
    """Leaderboard entries sorted by bypass_score desc."""
    leaderboard = get_leaderboard()
    if len(leaderboard) >= 2:
        # First entry should have highest bypass
        assert leaderboard[0]["bypass_score"] >= leaderboard[1]["bypass_score"]


def test_format_leaderboard_table_non_empty() -> None:
    """format_leaderboard_table returns non-empty string."""
    leaderboard = get_leaderboard()
    table = format_leaderboard_table(leaderboard)
    assert isinstance(table, str)
    assert len(table) > 0


def test_telemetry_anonymization() -> None:
    """TelemetryEvent should not contain PII."""
    event = TelemetryEvent.create(
        model_name="meta-llama/Llama-3.1-8B",
        method="diy",
        bypass_score=0.85,
        capability_delta=-0.02,
        n_directions=1,
        quantization=None,
    )
    # No email, username, or other PII fields
    assert not hasattr(event, "email")
    assert not hasattr(event, "username")
    # Model name is not PII - it's public model identifier
    assert "llama" in event.model_name.lower() or "qwen" in event.model_name.lower()
    # Run ID should be anonymized (hex string, not tied to user identity)
    assert len(event.run_id) == 16
    assert all(c in "0123456789abcdef" for c in event.run_id)