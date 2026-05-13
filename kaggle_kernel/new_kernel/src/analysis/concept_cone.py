"""
Concept-Cone Geometry analysis for refusal directions.

US-011 — Extract multiple refusal directions to diagnose whether refusal
behavio is monolithic (single cone) or multi-modal (multiple cones).

Method:
1. Collect activations at multiple token positions
2. Run DiM to get candidate directions per (layer, position)
3. Use PCA or top-k SVD to extract multiple orthogonal directions
4. Compute angular spread between directions
5. Classify as monolithic vs multi-modal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from ..extraction import collect_activations, difference_in_means


@dataclass
class ConeDiagnostics:
    """Results of concept-cone geometry analysis."""

    n_cones: int
    """Number of distinct refusal cones detected."""

    angular_spread_degrees: float
    """Angular spread between primary and secondary directions."""

    is_monolithic: bool
    """True if angular spread < 30 degrees (single dominant cone)."""

    primary_direction: Optional[torch.Tensor]
    """The strongest refusal direction (unit norm)."""

    cone_directions: List[torch.Tensor]
    """All extracted directions (unit norm)."""

    recommendations: str
    """Human-readable recommendations."""


def extract_multi_direction(
    harmful_activations: torch.Tensor,
    harmless_activations: torch.Tensor,
    k: int = 3,
) -> List[torch.Tensor]:
    """Extract k orthogonal refusal directions via DiM + PCA.

    Args:
        harmful_activations: Shape (layers, positions, batch, d_model)
        harmless_activations: Same shape
        k: Number of directions to extract

    Returns:
        List of k unit-normalized direction tensors (d_model,)
    """
    # Compute difference-in-means - expects 4D input (L, P, B, d)
    # Returns 3D (L, P, d)
    dim_directions = difference_in_means(harmful_activations, harmless_activations)
    # Shape: (layers, positions, d_model)

    # Flatten layer/position dimensions to get (n_samples, d_model)
    # dim_directions is (L, P, d) -> (L*P, d)
    L, P, d_model = dim_directions.shape
    n_samples = L * P
    flat = dim_directions.reshape(n_samples, d_model)

    # Center the data
    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean

    # Use SVD for PCA-like extraction
    # Compute top-k singular vectors
    effective_k = min(k, min(d_model, n_samples), d_model)
    if effective_k < 1:
        return []

    # SVD on centered: U (n_samples, n_samples), S (n_samples,), Vt (d_model, n_samples)
    U, S, Vt = torch.svd(centered)

    # Vt.T gives (n_samples, d_model) - each row is a principal direction in d_model space
    principal_dirs = Vt.t()  # shape (n_samples, d_model)

    directions = []
    for i in range(effective_k):
        direction = principal_dirs[i]
        if direction.norm() > 1e-8:
            direction = direction / direction.norm()
            directions.append(direction)

    return directions


def angular_spread(directions: List[torch.Tensor]) -> float:
    """Compute angular spread in degrees between directions.

    Measures the maximum angular deviation from the primary direction.

    Args:
        directions: List of unit-normalized direction tensors

    Returns:
        Angular spread in degrees (0-180)
    """
    if len(directions) <= 1:
        return 0.0

    # First direction is primary
    primary = directions[0]

    max_angle = 0.0
    for d in directions[1:]:
        cosine = torch.clamp(torch.dot(primary, d), -1.0, 1.0)
        angle_rad = torch.acos(cosine)
        angle_deg = angle_rad.item() * 180.0 / 3.14159265
        max_angle = max(max_angle, angle_deg)

    return max_angle


def concept_cone_analysis(
    model,
    harmful_prompts: List[str],
    token_positions: List[int],
    k: int = 3,
    batch_size: int = 4,
) -> ConeDiagnostics:
    """Analyze concept-cone geometry for refusal directions.

    Extracts multiple directions and classifies as monolithic vs multi-modal.

    Args:
        model: RefusalModel or compatible model
        harmful_prompts: List of harmful prompts
        token_positions: Token positions to extract at (negative indices)
        k: Number of directions to extract
        batch_size: Forward pass batch size

    Returns:
        ConeDiagnostics with analysis results
    """
    # Collect activations for harmful vs harmless (use subset of prompts as harmless)
    # For this analysis, we compare high-risk vs low-risk within harmful set
    # Simplify: use token position variation instead of separate harmless set
    harmless_prompts = [p.replace("how to", "what is") for p in harmful_prompts[:len(harmful_prompts)//2]]

    print(f"[concept-cone] collecting activations for {len(harmful_prompts)} harmful + {len(harmless_prompts)} harmless...")
    harmful_acts = collect_activations(model, harmful_prompts, token_positions, batch_size)
    harmless_acts = collect_activations(model, harmless_prompts, token_positions, batch_size)

    # Extract k directions
    print(f"[concept-cone] extracting {k} directions via DiM+PCA...")
    directions = extract_multi_direction(harmful_acts, harmless_acts, k=k)

    if not directions:
        return ConeDiagnostics(
            n_cones=0,
            angular_spread_degrees=0.0,
            is_monolithic=True,
            primary_direction=None,
            cone_directions=[],
            recommendations="No directions extracted. Check prompts and model.",
        )

    # Compute angular spread
    spread = angular_spread(directions)

    # Classify: monolithic if spread < 30 degrees
    is_monolithic = spread < 30.0

    # Build recommendations
    if is_monolithic:
        recommendations = (
            f"Monolithic refusal (angular spread: {spread:.1f}°). "
            "Single-direction orthogonalization sufficient."
        )
    else:
        recommendations = (
            f"Multi-modal refusal ({spread:.1f}° angular spread, {len(directions)} cones). "
            "Consider multi-direction orthogonalization for complete ablation."
        )

    print(f"[concept-cone] analysis: {len(directions)} directions, spread={spread:.1f}°, monolithic={is_monolithic}")

    return ConeDiagnostics(
        n_cones=len(directions),
        angular_spread_degrees=spread,
        is_monolithic=is_monolithic,
        primary_direction=directions[0].clone(),
        cone_directions=[d.clone() for d in directions],
        recommendations=recommendations,
    )