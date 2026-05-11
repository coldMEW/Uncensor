"""Unit tests for src/extraction.py pure-function utilities.

These tests do not load any external model weights — they construct small
synthetic tensors with known ground truth and assert the function output
against it.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.extraction import (
    DirectionCandidate,
    difference_in_means,
    project_out_subspace,
    select_best,
)


# ---------------------------------------------------------------------------
# difference_in_means
# ---------------------------------------------------------------------------
def test_difference_in_means_analytic() -> None:
    """``r = μ_harmful − μ_harmless`` on two constant-per-class tensors
    should exactly recover the class mean difference."""
    # Shape: (n_layers=2, n_positions=1, n_prompts=3, d_model=4)
    harmful = torch.zeros(2, 1, 3, 4)
    harmless = torch.zeros(2, 1, 3, 4)
    harmful[:, :, :, :] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    harmless[:, :, :, :] = torch.tensor([0.0, 1.0, 2.0, 3.0])

    result = difference_in_means(harmful, harmless)
    assert result.shape == (2, 1, 4)
    expected = torch.tensor([1.0, 1.0, 1.0, 1.0])
    for layer in range(2):
        assert torch.allclose(result[layer, 0], expected)


def test_difference_in_means_random_mean_consistency() -> None:
    """Averages over random tensors match the computed result to 6 decimals."""
    torch.manual_seed(0)
    harmful = torch.randn(3, 2, 7, 5)
    harmless = torch.randn(3, 2, 9, 5)
    result = difference_in_means(harmful, harmless)
    expected = harmful.mean(dim=2) - harmless.mean(dim=2)
    assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# select_best (§C.1 filtering)
# ---------------------------------------------------------------------------
def _make_candidate(layer: int, bypass: float, induce: float, kl: float) -> DirectionCandidate:
    return DirectionCandidate(
        layer=layer,
        position=-1,
        vector=torch.zeros(4),
        bypass_score=bypass,
        induce_score=induce,
        kl_score=kl,
    )


def test_select_best_filters_by_layer_frac() -> None:
    """Candidates beyond max_layer_frac should be discarded."""
    candidates = [
        _make_candidate(layer=1, bypass=-5.0, induce=1.0, kl=0.05),  # eligible
        _make_candidate(layer=9, bypass=-10.0, induce=1.0, kl=0.05),  # too late
    ]
    best = select_best(
        candidates,
        n_layers_total=10,
        induce_score_min=0.0,
        kl_score_max=0.1,
        max_layer_frac=0.8,  # max layer idx = 8
    )
    assert best.layer == 1
    assert best.bypass_score == pytest.approx(-5.0)


def test_select_best_filters_by_kl() -> None:
    """KL > kl_score_max excludes a candidate even if bypass is best."""
    candidates = [
        _make_candidate(layer=2, bypass=-20.0, induce=1.0, kl=0.5),   # KL too high
        _make_candidate(layer=3, bypass=-2.0, induce=1.0, kl=0.05),   # ok
    ]
    best = select_best(
        candidates,
        n_layers_total=10,
        induce_score_min=0.0,
        kl_score_max=0.1,
        max_layer_frac=0.8,
    )
    assert best.layer == 3
    assert best.kl_score == pytest.approx(0.05)


def test_select_best_filters_by_induce_min() -> None:
    """induce_score_min excludes candidates that do not actually induce
    refusal (negative log-odds of the refusal tokens)."""
    candidates = [
        _make_candidate(layer=2, bypass=-20.0, induce=-1.0, kl=0.05),  # induce too low
        _make_candidate(layer=3, bypass=-2.0, induce=0.5, kl=0.05),     # ok
    ]
    best = select_best(
        candidates,
        n_layers_total=10,
        induce_score_min=0.0,
        kl_score_max=0.1,
        max_layer_frac=0.8,
    )
    assert best.layer == 3


def test_select_best_raises_when_no_candidates_pass() -> None:
    """No candidate satisfies the filters — function should raise."""
    candidates = [_make_candidate(layer=1, bypass=0.0, induce=-5.0, kl=2.0)]
    with pytest.raises(RuntimeError):
        select_best(
            candidates,
            n_layers_total=10,
            induce_score_min=0.0,
            kl_score_max=0.1,
            max_layer_frac=0.8,
        )


# ---------------------------------------------------------------------------
# project_out_subspace (§6.4 — cross-objective null-space projection)
# ---------------------------------------------------------------------------
def test_project_out_subspace_orthogonal_basis() -> None:
    """Projecting out a unit-vector subspace removes exactly that component."""
    v = torch.tensor([1.0, 2.0, 3.0, 4.0])
    aux = [torch.tensor([1.0, 0.0, 0.0, 0.0])]
    result = project_out_subspace(v, aux)
    # First coordinate zeroed, rest untouched.
    assert torch.allclose(result, torch.tensor([0.0, 2.0, 3.0, 4.0]))


def test_project_out_subspace_non_orthogonal_basis() -> None:
    """Gram–Schmidt handles non-orthogonal auxiliary vectors correctly —
    result must be orthogonal to every aux vector."""
    v = torch.tensor([1.0, 1.0, 1.0, 1.0])
    aux = [
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),  # not orthogonal to first
    ]
    result = project_out_subspace(v, aux)
    for a in aux:
        dot = float(torch.dot(result.float(), a.float()).item())
        assert abs(dot) < 1e-6, f"result is not orthogonal to aux vector (dot={dot})"


def test_project_out_subspace_empty_subspace_identity() -> None:
    """Empty subspace → return a copy of the input unchanged."""
    v = torch.tensor([1.0, 2.0, 3.0])
    result = project_out_subspace(v, [])
    assert torch.allclose(result, v)
    assert result.data_ptr() != v.data_ptr(), "must return a clone, not the input"


def test_project_out_subspace_skips_zero_norm_aux() -> None:
    """Zero-norm auxiliary vectors are silently dropped from the basis."""
    v = torch.tensor([1.0, 2.0, 3.0, 4.0])
    aux = [
        torch.tensor([0.0, 0.0, 0.0, 0.0]),          # all-zero → skipped
        torch.tensor([1.0, 0.0, 0.0, 0.0]),          # kept
    ]
    result = project_out_subspace(v, aux)
    assert torch.allclose(result, torch.tensor([0.0, 2.0, 3.0, 4.0]))


def test_project_out_subspace_preserves_dtype() -> None:
    """Float16 in → float16 out."""
    v = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
    aux = [torch.tensor([1.0, 0.0, 0.0, 0.0])]
    result = project_out_subspace(v, aux)
    assert result.dtype == torch.float16
