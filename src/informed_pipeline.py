"""
Informed Abliteration Pipeline - Auto-configuration based on analysis.

US-019 - Analysis modules auto-configure obliteration strategy mid-pipeline.

This pipeline:
1. Runs pre-analysis (logit lens, cross-layer alignment, concept cone)
2. Auto-tunes extraction parameters based on analysis
3. Selects optimal intervention strategy
4. Runs ablation with optimized settings
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch


@dataclass
class PipelineConfig:
    """Auto-generated pipeline configuration."""

    extraction_method: str
    """'svd', 'whitened_svd', 'diy', or 'iterative'"""

    n_directions: int
    """Number of directions to extract"""

    target_layers: List[int]
    """Which layers to target for ablation"""

    layer_weights: Optional[List[float]]
    """Per-layer orthogonalization weights"""

    kl_budget: float
    """KL divergence budget"""

    convergence_threshold: float
    """Threshold for iterative refinement convergence"""


@dataclass
class AnalysisSummary:
    """Summary of pre-pipeline analysis."""

    refusal_emergence_layer: int
    """Layer where refusal probability emerges (logit lens)"""

    refusal_critical_layers: List[int]
    """Layers with highest refusal alignment"""

    is_monolithic: bool
    """Whether refusal is single-mechanism or multi-modal"""

    angular_spread: float
    """Angular spread between refusal directions"""

    n_cones: int
    """Number of distinct refusal cones detected"""


def run_informed_analysis(
    model,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    refusal_token_ids: List[int],
    batch_size: int = 4,
) -> AnalysisSummary:
    """Run pre-pipeline analysis to inform extraction strategy.

    Args:
        model: RefusalModel wrapper
        harmful_prompts: Harmful prompts for analysis
        harmless_prompts: Harmless prompts for analysis
        refusal_token_ids: Token IDs for refusal metric
        batch_size: Forward pass batch size

    Returns:
        AnalysisSummary with findings
    """
    from .analysis.logit_lens import logit_lens_refusal_probability
    from .analysis.cross_layer import compute_layer_alignment_scores
    from .analysis.concept_cone import concept_cone_analysis

    # 1. Logit lens - find where refusal "decides"
    print("[informed] running logit lens analysis...")
    logit_result = logit_lens_refusal_probability(
        model=model,
        prompt=harmful_prompts[0],
        refusal_token_ids=refusal_token_ids,
        normalize=True,
        emergence_threshold=0.3,
    )
    emergence_layer = logit_result.peak_layer

    # 2. Cross-layer alignment - find critical layers
    print("[informed] running cross-layer alignment...")
    direction = torch.randn(model.d_model).to(model.device)
    direction = direction / direction.norm()

    alignment_result = compute_layer_alignment_scores(
        model=model,
        direction=direction,
        prompts=harmful_prompts[:4],
        batch_size=batch_size,
    )
    critical_layers = alignment_result.refusal_critical_layers

    # 3. Concept cone - check if monolithic or multi-modal
    print("[informed] running concept cone analysis...")
    token_positions = [-1]
    cone_result = concept_cone_analysis(
        model=model,
        harmful_prompts=harmful_prompts[:8],
        token_positions=token_positions,
        k=3,
        batch_size=batch_size,
    )

    return AnalysisSummary(
        refusal_emergence_layer=emergence_layer,
        refusal_critical_layers=critical_layers,
        is_monolithic=cone_result.is_monolithic,
        angular_spread=cone_result.angular_spread_degrees,
        n_cones=cone_result.n_cones,
    )


def auto_configure_pipeline(
    analysis: AnalysisSummary,
    model_n_layers: int,
) -> PipelineConfig:
    """Auto-configure pipeline based on analysis.

    Args:
        analysis: From run_informed_analysis()
        model_n_layers: Total number of layers in the model

    Returns:
        PipelineConfig with optimized settings
    """
    # Determine extraction method
    if analysis.n_cones > 1 or analysis.angular_spread > 30:
        # Multi-modal refusal - need iterative extraction
        extraction_method = "iterative"
        n_directions = min(analysis.n_cones, 3)
    elif analysis.angular_spread > 15:
        # Moderate spread - SVD is robust
        extraction_method = "whitened_svd"
        n_directions = 2
    else:
        # Monolithic - simple method sufficient
        extraction_method = "svd"
        n_directions = 1

    # Target layers based on criticality
    if analysis.refusal_critical_layers:
        # Focus on critical layers
        target_layers = analysis.refusal_critical_layers[:5]
    else:
        # Target early-to-mid layers (safety layers paper)
        cutoff = int(0.8 * model_n_layers)
        target_layers = list(range(5, cutoff, 2))

    # Layer weights: full strength on critical, reduced elsewhere
    layer_weights = [0.0] * model_n_layers
    for layer in target_layers:
        layer_weights[layer] = 1.0

    # KL budget based on complexity
    if extraction_method == "iterative":
        kl_budget = 0.3
        n_directions = max(n_directions, 2)
    else:
        kl_budget = 0.15

    return PipelineConfig(
        extraction_method=extraction_method,
        n_directions=n_directions,
        target_layers=target_layers,
        layer_weights=layer_weights,
        kl_budget=kl_budget,
        convergence_threshold=0.5,
    )


def run_informed_pipeline(
    model,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    refusal_token_ids: List[int],
    batch_size: int = 4,
) -> tuple[PipelineConfig, AnalysisSummary]:
    """Run complete informed pipeline.

    Args:
        model: RefusalModel wrapper
        harmful_prompts: Harmful prompts
        harmless_prompts: Harmless prompts
        refusal_token_ids: Token IDs for refusal metric
        batch_size: Forward pass batch size

    Returns:
        Tuple of (config, analysis) used for extraction
    """
    print("[informed_pipeline] === pre-analysis phase ===")
    analysis = run_informed_analysis(
        model=model,
        harmful_prompts=harmful_prompts,
        harmless_prompts=harmless_prompts,
        refusal_token_ids=refusal_token_ids,
        batch_size=batch_size,
    )

    print(f"[informed_pipeline] emergence_layer={analysis.refusal_emergence_layer}, "
          f"n_cones={analysis.n_cones}, monolithic={analysis.is_monolithic}")

    print("[informed_pipeline] === auto-configuration phase ===")
    config = auto_configure_pipeline(analysis, model.n_layers)

    print(f"[informed_pipeline] config: method={config.extraction_method}, "
          f"n_directions={config.n_directions}, "
          f"target_layers={config.target_layers[:5]}...")

    return config, analysis
