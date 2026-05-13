"""Unit tests for src/interventions.py.

Covers the low-level weight-orthogonalization primitives (_ortho_linear,
_ortho_embedding) and the post-quantization correction pass. These tests
do not need a full language model — they construct ``nn.Linear`` /
``nn.Embedding`` modules directly.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from src.interventions import _ortho_embedding, _ortho_linear, directional_ablation


def _random_unit(d: int, *, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    return v / v.norm()


# ---------------------------------------------------------------------------
# _ortho_linear
# ---------------------------------------------------------------------------
def test_ortho_linear_projects_out_direction() -> None:
    """After _ortho_linear, every row of W has zero component along r̂."""
    torch.manual_seed(0)
    d_out, d_in = 16, 32
    layer = nn.Linear(d_in, d_out, bias=False)
    r_hat = _random_unit(d_out, seed=1)

    _ortho_linear(layer, r_hat)

    # W projected onto r̂: r̂ᵀ W should be ~0 (for every column).
    projections = r_hat @ layer.weight.data  # (d_in,)
    assert torch.allclose(projections, torch.zeros_like(projections), atol=1e-5)


def test_ortho_linear_zeros_bias_component() -> None:
    """The bias component along r̂ is zeroed when the layer has a bias."""
    torch.manual_seed(0)
    d_out, d_in = 8, 16
    layer = nn.Linear(d_in, d_out, bias=True)
    r_hat = _random_unit(d_out, seed=2)

    _ortho_linear(layer, r_hat)

    proj = float(torch.dot(r_hat, layer.bias.data).item())
    assert abs(proj) < 1e-5


def test_ortho_linear_preserves_orthogonal_component() -> None:
    """Components of W orthogonal to r̂ must be unchanged."""
    torch.manual_seed(0)
    d_out, d_in = 8, 4
    layer = nn.Linear(d_in, d_out, bias=False)
    r_hat = _random_unit(d_out, seed=3)

    W_before = layer.weight.data.clone()
    _ortho_linear(layer, r_hat)
    W_after = layer.weight.data

    # Decompose the change: Δ = W_before − W_after should be rank-1 along r̂.
    delta = W_before - W_after
    # Project each column of delta onto r̂⊥ — must be zero (no orthogonal change).
    for col in range(d_in):
        col_vec = delta[:, col]
        orth_component = col_vec - float(torch.dot(r_hat, col_vec).item()) * r_hat
        assert torch.allclose(
            orth_component, torch.zeros_like(orth_component), atol=1e-5
        )


# ---------------------------------------------------------------------------
# _ortho_embedding
# ---------------------------------------------------------------------------
def test_ortho_embedding_rows_orthogonal_to_direction() -> None:
    """Every row of the embedding table should have zero component along r̂."""
    torch.manual_seed(0)
    vocab, d_model = 64, 32
    embed = nn.Embedding(vocab, d_model)
    r_hat = _random_unit(d_model, seed=4)

    _ortho_embedding(embed, r_hat)

    # Dot product of every row with r̂ should be ~0.
    dots = embed.weight.data @ r_hat  # (vocab,)
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-5)


# ---------------------------------------------------------------------------
# Device-awareness (CPU smoke test — verifies the .to(device=W.device)
# branch does not crash on CPU even though CPU is trivially "same device")
# ---------------------------------------------------------------------------
def test_ortho_linear_device_match_cpu() -> None:
    """r_hat originally on CPU + layer on CPU → result correct (no crash)."""
    torch.manual_seed(0)
    layer = nn.Linear(8, 8, bias=False)
    r_hat = _random_unit(8, seed=5)
    _ortho_linear(layer, r_hat)
    assert torch.allclose(
        r_hat @ layer.weight.data,
        torch.zeros(8),
        atol=1e-5,
    )


def test_directional_ablation_accepts_coefficient_argument() -> None:
    """Kaggle notebook sweeps intervention strength via this public argument."""
    import inspect

    signature = inspect.signature(directional_ablation)
    assert "coefficient" in signature.parameters
    assert signature.parameters["coefficient"].default == 1.0
