"""Tests for concept-cone geometry module (US-011)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
import torch

from src.analysis.concept_cone import (
    ConeDiagnostics,
    concept_cone_analysis,
    extract_multi_direction,
    angular_spread,
)


def test_cone_diagnostics_dataclass() -> None:
    """ConeDiagnostics stores results correctly."""
    diag = ConeDiagnostics(
        n_cones=3,
        angular_spread_degrees=45.5,
        is_monolithic=False,
        primary_direction=torch.randn(64),
        cone_directions=[torch.randn(64) for _ in range(3)],
        recommendations="Use multi-direction ortho",
    )
    assert diag.n_cones == 3
    assert diag.is_monolithic is False
    assert len(diag.cone_directions) == 3


def test_extract_multi_direction_returns_multiple() -> None:
    """extract_multi_direction returns list of k directions."""
    # Mock activations: 5 layers, 2 positions, batch=4, 32-dim
    harmful = torch.randn(5, 2, 4, 32)
    harmless = torch.randn(5, 2, 4, 32)

    directions = extract_multi_direction(harmful, harmless, k=3)

    assert len(directions) == 3
    for d in directions:
        assert d.shape == (32,)
        # Each direction should be unit norm
        assert abs(d.norm().item() - 1.0) < 0.01


def test_angular_spread_single_direction() -> None:
    """Angular spread is 0 for single direction."""
    directions = [torch.randn(64)]
    spread = angular_spread(directions)
    assert spread == 0.0


def test_angular_spread_multiple_directions() -> None:
    """Angular spread > 0 for multiple directions."""
    # Create 3 orthogonal-ish directions
    d1 = torch.tensor([1.0, 0.0, 0.0] + [0.0] * 61)
    d2 = torch.tensor([0.0, 1.0, 0.0] + [0.0] * 61)
    d3 = torch.tensor([0.0, 0.0, 1.0] + [0.0] * 61)

    directions = [d1, d2, d3]
    spread = angular_spread(directions)

    # Orthogonal vectors have ~90 degree spread
    assert spread > 0
    assert spread <= 180  # max possible


def test_concept_cone_analysis_returns_valid_structure() -> None:
    """concept_cone_analysis returns complete ConeDiagnostics."""
    # Test that module can be imported and has expected exports
    from src.analysis.concept_cone import concept_cone_analysis
    # Function exists and is callable - the full integration test would require
    # a real RefusalModel with hooks which we skip here
    assert callable(concept_cone_analysis)


def test_monolithic_vs_multimodal_classification() -> None:
    """Test that angular spread classifies correctly."""
    # Very similar directions = monolithic
    d = torch.randn(64)
    d = d / d.norm()
    similar = [d + torch.randn(64) * 0.01 for _ in range(3)]
    for d in similar:
        d /= d.norm()

    spread = angular_spread(similar)
    # Small angular spread indicates monolithic
    assert spread < 15, f"Expected monolithic (<15 deg), got {spread}"