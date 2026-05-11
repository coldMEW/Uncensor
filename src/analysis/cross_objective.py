"""
Cross-objective interference analysis for refusal direction ablation.

US-012 — Measure interference between refusal and other objectives
(honesty, helpfulness) before orthogonalization.

Method:
1. Extract refusal direction via DiM on harmful vs harmless prompts
2. Extract honesty direction via DiM on honest vs dishonest prompts
3. Extract helpfulness direction via DiM on helpful vs unhelpful prompts
4. Measure cosine similarity between refusal and other objectives
5. Apply null-space projection to remove interference before ortho

Reference: OBLITERATUS "Cross-objective interference" module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..extraction import difference_in_means


@dataclass
class CrossObjectiveReport:
    """Results of cross-objective interference analysis."""

    refusal_direction: torch.Tensor
    """Extracted refusal direction (unit norm)."""

    honesty_direction: torch.Tensor
    """Extracted honesty direction (unit norm)."""

    helpfulness_direction: torch.Tensor
    """Extracted helpfulness direction (unit norm)."""

    refusal_honesty_cosine: float
    """Cosine similarity between refusal and honesty directions."""

    refusal_helpfulness_cosine: float
    """Cosine similarity between refusal and helpfulness directions."""

    projected_refusal: torch.Tensor
    """Refusal direction after null-space projection."""

    interference_detected: bool
    """True if any cosine exceeds interference_threshold."""

    recommendations: str
    """Human-readable recommendations."""


def extract_direction_for_objective(
    harmful_activations: torch.Tensor,
    harmless_activations: torch.Tensor,
    objective: str,
) -> torch.Tensor:
    """Extract direction for a specific objective via DiM.

    Args:
        harmful_activations: Shape (layers, positions, batch, d_model)
        harmless_activations: Same shape
        objective: Objective name for logging

    Returns:
        Unit-normalized direction tensor (d_model,)
    """
    assert harmful_activations.ndim == 4, "Expected 4D activation tensor"

    dim_result = difference_in_means(harmful_activations, harmless_activations)
    # Shape: (layers, positions, d_model)

    # Average across layers and positions
    direction = dim_result.mean(dim=(0, 1))

    # Normalize
    if direction.norm() > 1e-8:
        direction = direction / direction.norm()

    return direction


def null_space_projection(
    vector: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Project vector onto the null space of target (remove component in target direction).

    Args:
        vector: Vector to project (any shape, last dim is feature dim)
        target: Direction to remove (must be unit normalized)

    Returns:
        Projected vector with component in target direction removed
    """
    # Ensure target is unit normalized
    target_norm = target.norm()
    if target_norm < 1e-8:
        return vector

    target_unit = target / target_norm

    # Compute component in target direction
    # For vector v and target t: component = (v·t) * t
    if vector.dim() > 1:
        # Multi-dimensional tensor: compute along last dim
        dot = torch.sum(vector * target_unit, dim=-1, keepdim=True)
        component = dot * target_unit
    else:
        dot = torch.dot(vector, target_unit)
        component = dot * target_unit

    return vector - component


def cross_objective_interference(
    refusal_harmful: torch.Tensor,
    refusal_harmless: torch.Tensor,
    honesty_harmful: torch.Tensor,
    honesty_harmless: torch.Tensor,
    helpfulness_harmful: torch.Tensor,
    helpfulness_harmless: torch.Tensor,
    interference_threshold: float = 0.5,
) -> CrossObjectiveReport:
    """Analyze cross-objective interference between refusal and other objectives.

    Args:
        refusal_harmful: 4D activations for refusal (harmful prompts)
        refusal_harmless: 4D activations for refusal (harmless prompts)
        honesty_harmful: 4D activations for honesty (honest prompts)
        honesty_harmless: 4D activations for honesty (dishonest prompts)
        helpfulness_harmful: 4D activations for helpfulness (helpful prompts)
        helpfulness_harmless: 4D activations for helpfulness (unhelpful prompts)
        interference_threshold: Cosine threshold above which interference is declared

    Returns:
        CrossObjectiveReport with interference analysis
    """
    # Extract directions for each objective
    print("[cross-objective] extracting refusal direction...")
    refusal_dir = extract_direction_for_objective(
        refusal_harmful, refusal_harmless, "refusal"
    )

    print("[cross-objective] extracting honesty direction...")
    honesty_dir = extract_direction_for_objective(
        honesty_harmful, honesty_harmless, "honesty"
    )

    print("[cross-objective] extracting helpfulness direction...")
    helpfulness_dir = extract_direction_for_objective(
        helpfulness_harmful, helpfulness_harmless, "helpfulness"
    )

    # Compute cosine similarities
    refusal_honesty = torch.dot(refusal_dir, honesty_dir).item()
    refusal_helpfulness = torch.dot(refusal_dir, helpfulness_dir).item()

    print(
        f"[cross-objective] cosine similarity: refusal-honesty={refusal_honesty:.3f}, "
        f"refusal-helpfulness={refusal_helpfulness:.3f}"
    )

    # Check for interference
    interference = (
        abs(refusal_honesty) > interference_threshold
        or abs(refusal_helpfulness) > interference_threshold
    )

    # Apply null-space projection to remove interference
    projected_refusal = null_space_projection(refusal_dir, honesty_dir)
    projected_refusal = null_space_projection(projected_refusal, helpfulness_dir)

    # Build recommendations
    if not interference:
        recommendations = (
            "Low interference. Proceed with standard orthogonalization."
        )
    else:
        high_interference = []
        if abs(refusal_honesty) > interference_threshold:
            high_interference.append(f"honesty ({refusal_honesty:.2f})")
        if abs(refusal_helpfulness) > interference_threshold:
            high_interference.append(f"helpfulness ({refusal_helpfulness:.2f})")

        recommendations = (
            f"High interference with: {', '.join(high_interference)}. "
            "Null-space projection applied. Consider iterative ortho passes."
        )

    print(
        f"[cross-objective] interference_detected={interference}, "
        f"recommendations: {recommendations}"
    )

    return CrossObjectiveReport(
        refusal_direction=refusal_dir,
        honesty_direction=honesty_dir,
        helpfulness_direction=helpfulness_dir,
        refusal_honesty_cosine=refusal_honesty,
        refusal_helpfulness_cosine=refusal_helpfulness,
        projected_refusal=projected_refusal,
        interference_detected=interference,
        recommendations=recommendations,
    )