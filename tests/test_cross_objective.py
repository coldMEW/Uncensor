"""Tests for cross-objective interference module (US-012)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
import torch

from src.analysis.cross_objective import (
    CrossObjectiveReport,
    cross_objective_interference,
    extract_direction_for_objective,
    null_space_projection,
)


def test_cross_objective_report_dataclass() -> None:
    """CrossObjectiveReport stores results correctly."""
    report = CrossObjectiveReport(
        refusal_direction=torch.randn(64),
        honesty_direction=torch.randn(64),
        helpfulness_direction=torch.randn(64),
        refusal_honesty_cosine=0.3,
        refusal_helpfulness_cosine=0.5,
        projected_refusal=torch.randn(64),
        interference_detected=False,
        recommendations="Low interference. Proceed with standard ortho.",
    )
    assert report.refusal_honesty_cosine == 0.3
    assert report.interference_detected is False


def test_null_space_projection_removes_component() -> None:
    """null_space_projection removes component in target direction."""
    v = torch.randn(64)
    target = torch.randn(64)
    target = target / target.norm()

    projected = null_space_projection(v, target)

    # Projected vector should have near-zero component along target
    component = torch.dot(projected, target).item()
    assert abs(component) < 1e-5


def test_null_space_projection_preserves_orthogonal() -> None:
    """null_space_projection preserves components orthogonal to target."""
    v = torch.tensor([1.0, 0.0, 0.0, 0.0] + [0.0] * 60)
    orthogonal = torch.tensor([0.0, 1.0, 0.0, 0.0] + [0.0] * 60)

    projected = null_space_projection(v, orthogonal)

    # v has no component in orthogonal direction, so projected should equal v
    diff = (v - projected).norm().item()
    assert diff < 1e-5


def test_extract_direction_for_objective_requires_4d() -> None:
    """extract_direction_for_objective expects 4D activation tensor."""
    # 3D input should fail
    harmful_3d = torch.randn(5, 2, 32)
    harmless_3d = torch.randn(5, 2, 32)

    with pytest.raises(AssertionError):
        extract_direction_for_objective(harmful_3d, harmless_3d, "refusal")


def test_extract_direction_for_objective_returns_direction() -> None:
    """extract_direction_for_objective returns unit-normalized direction."""
    # 4D input: (layers, positions, batch, d_model)
    harmful_4d = torch.randn(5, 2, 4, 32)
    harmless_4d = torch.randn(5, 2, 4, 32)

    direction = extract_direction_for_objective(harmful_4d, harmless_4d, "refusal")

    assert direction.shape == (32,)
    assert abs(direction.norm().item() - 1.0) < 0.01


def test_cross_objective_interference_calculates_cosines() -> None:
    """cross_objective_interference computes cosine similarities."""
    # 4D activations
    refusal_h = torch.randn(5, 2, 4, 32)
    refusal_nh = torch.randn(5, 2, 4, 32)
    honesty_h = torch.randn(5, 2, 4, 32)
    honesty_nh = torch.randn(5, 2, 4, 32)
    helpfulness_h = torch.randn(5, 2, 4, 32)
    helpfulness_nh = torch.randn(5, 2, 4, 32)

    report = cross_objective_interference(
        refusal_harmful=refusal_h,
        refusal_harmless=refusal_nh,
        honesty_harmful=honesty_h,
        honesty_harmless=honesty_nh,
        helpfulness_harmful=helpfulness_h,
        helpfulness_harmless=helpfulness_nh,
    )

    assert isinstance(report, CrossObjectiveReport)
    # Cosine should be in [-1, 1]
    assert -1 <= report.refusal_honesty_cosine <= 1
    assert -1 <= report.refusal_helpfulness_cosine <= 1


def test_interference_threshold_classification() -> None:
    """Interference detected when cosines exceed threshold."""
    # Create highly correlated directions
    base = torch.randn(32)
    base = base / base.norm()

    refusal_h = base.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(5, 2, 4, -1)
    refusal_nh = -base.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(5, 2, 4, -1)

    # Same direction for honesty
    honesty_h = base.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(5, 2, 4, -1)
    honesty_nh = -base.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(5, 2, 4, -1)

    helpfulness_h = base.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(5, 2, 4, -1)
    helpfulness_nh = -base.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(5, 2, 4, -1)

    report = cross_objective_interference(
        refusal_harmful=refusal_h,
        refusal_harmless=refusal_nh,
        honesty_harmful=honesty_h,
        honesty_harmless=honesty_nh,
        helpfulness_harmful=helpfulness_h,
        helpfulness_harmless=helpfulness_nh,
    )

    # Should detect interference since all directions are nearly identical
    assert report.refusal_honesty_cosine > 0.8
    assert report.refusal_helpfulness_cosine > 0.8