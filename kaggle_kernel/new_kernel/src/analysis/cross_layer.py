"""
Cross-Layer Alignment Analyzer for refusal direction ablation.

Analyzes how the refusal direction evolves across transformer layers to identify
which layers are most critical for refusal behavior.

This helps:
- Identify "safety-critical" layers (highest alignment with refusal direction)
- Understand refusal geometry across the model depth
- Target specific layers for surgical ablation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LayerAlignmentResult:
    """Results from cross-layer alignment analysis."""

    layer_indices: List[int]
    """List of layer indices analyzed."""

    alignment_scores: List[float]
    """Cosine similarity of refusal direction at each layer (0=embed, L=final)."""

    refusal_critical_layers: List[int]
    """Layer indices where alignment > threshold (most refusal-critical)."""

    peak_layer: int
    """Layer with highest alignment score."""

    peak_score: float
    """Highest alignment score."""

    threshold: float
    """Threshold used to identify critical layers."""

def compute_layer_alignment_scores(
    model,
    direction: torch.Tensor,
    prompts: List[str],
    batch_size: int = 4,
) -> LayerAlignmentResult:
    """Compute cosine similarity of refusal direction across all transformer layers.

    For each layer l, we:
    1. Hook the residual stream output at that layer
    2. Collect activations for all prompts
    3. Compute DiM direction at that layer
    4. Calculate cosine similarity with the provided direction

    Args:
        model: RefusalModel wrapper
        direction: Reference refusal direction (unit-norm, shape (d_model,))
        prompts: Prompts to analyze
        batch_size: Forward pass batch size

    Returns:
        LayerAlignmentResult with per-layer alignment scores
    """
    from ..extraction import collect_activations, difference_in_means

    d_model = model.d_model
    n_layers = model.n_layers
    threshold = 0.5

    # Normalize reference direction
    ref_dir = direction.to(dtype=torch.float32)
    ref_dir = ref_dir / ref_dir.norm()

    # Collect activations at all layers
    token_positions = [-1]  # Last token only for efficiency
    activations = collect_activations(model, prompts, token_positions, batch_size)
    # Shape: (n_layers, n_positions, n_prompts, d_model)

    # Compute DiM direction per layer
    alignment_scores: List[float] = []
    refusal_critical_layers: List[int] = []

    for layer_idx in range(n_layers):
        # Get activations for this layer
        layer_acts = activations[layer_idx, 0]  # (n_prompts, d_model)

        # Compute mean activation as simple layer representation
        mean_act = layer_acts.mean(dim=0)  # (d_model,)

        # Compute cosine similarity with reference direction
        cos_sim = F.cosine_similarity(
            mean_act.unsqueeze(0),
            ref_dir.unsqueeze(0).to(device=mean_act.device),
            dim=-1,
        )
        score = float(cos_sim.item())
        alignment_scores.append(score)

        # Identify critical layers
        if score > threshold:
            refusal_critical_layers.append(layer_idx)

    # Find peak
    peak_layer = int(max(range(n_layers), key=lambda i: alignment_scores[i]))
    peak_score = alignment_scores[peak_layer]

    return LayerAlignmentResult(
        layer_indices=list(range(n_layers)),
        alignment_scores=alignment_scores,
        refusal_critical_layers=refusal_critical_layers,
        peak_layer=peak_layer,
        peak_score=peak_score,
        threshold=threshold,
    )


def plot_alignment_heatmap(
    result: LayerAlignmentResult,
    title: str = "Refusal Direction Alignment by Layer",
    save_path: Optional[str] = None,
) -> None:
    """Plot alignment scores as a heatmap across layers.

    Args:
        result: From compute_layer_alignment_scores()
        title: Plot title
        save_path: Optional path to save PNG
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[cross_layer] matplotlib not installed; skipping heatmap")
        return

    fig, ax = plt.subplots(figsize=(12, 4))

    scores = np.array(result.alignment_scores)
    layers = np.array(result.layer_indices)

    # Create bar plot with color gradient
    colors = plt.cm.RdYlGn_r(scores)  # Red=high (critical), Green=low

    ax.bar(layers, scores, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=result.threshold, color="red", linestyle="--", label=f"Threshold ({result.threshold})")
    ax.axhline(y=result.peak_score, color="blue", linestyle=":", label=f"Peak ({result.peak_score:.3f})")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Alignment Score (Cosine Similarity)")
    ax.set_title(title)
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[cross_layer] saved heatmap to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def batch_cross_layer_analysis(
    model,
    directions: List[torch.Tensor],
    prompts: List[str],
    batch_size: int = 4,
) -> List[LayerAlignmentResult]:
    """Analyze alignment for multiple directions across layers.

    Args:
        model: RefusalModel wrapper
        directions: List of refusal directions
        prompts: Prompts to analyze
        batch_size: Forward pass batch size

    Returns:
        List of LayerAlignmentResult, one per direction
    """
    results: List[LayerAlignmentResult] = []

    for direction in directions:
        result = compute_layer_alignment_scores(model, direction, prompts, batch_size)
        results.append(result)

    return results
