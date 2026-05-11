"""Pytest configuration — adds the project root to sys.path so tests can
import ``src.*`` directly without having to install the package.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest


@pytest.fixture(scope="session")
def mock_refusal_model():
    """Session-scoped fixture returning a MockRefusalModel for integration tests."""
    from tests.test_integration import MockRefusalModel
    return MockRefusalModel()
