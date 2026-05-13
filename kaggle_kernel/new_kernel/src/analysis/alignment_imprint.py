"""
Alignment-imprint detection for refusal direction analysis.

Different alignment training methods leave characteristic signatures in
the model's activation geometry:

SFT (Supervised Fine-Tuning):
  - Refusal direction emerges early (first 30% of layers)
  - High direction concentration (top-1 explains >70% of variance)
  - Sharp, localized refusal signal
  → Best intervention: single-direction orthogonalization at middle layers

DPO (Direct Preference Optimization):
  - Refusal direction is diffuse across layers (emerges at 40-60%)
  - Lower concentration (top-1 explains 50-70% of variance)
  - Multi-modal harmful activation distribution
  → Best intervention: multi-direction iterative deflation (n_directions=3)

RLHF (Reinforcement Learning from Human Feedback):
  - Refusal direction emerges late (last 50% of layers)
  - Moderate concentration (60-75%)
  - Strong self-repair tendency
  → Best intervention: surgical head ablation + self-repair mitigation

Reference: Meade et al. (2024) "Alignment Tax" — AFT vs APO classification.
           Arditi et al. (2024) Table 1 — models classified by alignment type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class AlignmentImprint:
    """Result of alignment-imprint detection analysis."""

    alignment_type: str              # 'sft', 'dpo', 'rlhf', 'unknown'
    confidence: float                # 0.0 to 1.0 — how confident we are
    refusal_emergence_layer: int     # First layer where refusal probability > 0.3
    refusal_emergence_frac: float    # emergence_layer / n_layers
    direction_concentration: float   # top-1 singular value ratio
    top5_concentration: float        # sum of top-5 singular value ratios
    recommended_preset: str          # 'basic', 'standard', 'deep', 'surgical', 'nuclear'
    recommended_n_directions: int    # Suggested n_directions for config
    reasoning: str                   # Human-readable explanation


# =============================================================================
# Internal helpers
# =============================================================================

def _find_emergence_layer_logit_lens(
    model,
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
    refusal_token_ids: List[int],
    threshold: float = 0.3,
) -> int:
    """Use logit-lens to find the first layer where refusal probability > threshold.

    Falls back to norm-based proxy if logit_lens module is unavailable.

    Args:
        model: RefusalModel instance.
        harmful_acts: ``(n_layers, n_positions, n_prompts, d_model)``
        harmless_acts: ``(n_layers, n_positions, n_prompts, d_model)``
        refusal_token_ids: Token IDs that represent refusal tokens.
        threshold: Refusal probability threshold (default 0.3).

    Returns:
        Index of the first emergence layer (0-indexed). Returns the last layer
        if no layer crosses the threshold.
    """
    n_layers = harmful_acts.shape[0]

    # Try logit-lens path first.
    try:
        from ..analysis.logit_lens import logit_lens_refusal_probability  # type: ignore[import]

        for layer_idx in range(n_layers):
            # Use mean over positions and prompts at this layer.
            # harmful_acts[layer_idx]: (n_positions, n_prompts, d_model)
            mean_act = harmful_acts[layer_idx].mean(dim=(0, 1))  # (d_model,)
            prob = logit_lens_refusal_probability(
                model, mean_act, refusal_token_ids, layer_idx,
            )
            if prob > threshold:
                return layer_idx
        return n_layers - 1

    except (ImportError, AttributeError, TypeError):
        pass

    # Fallback: proxy using per-layer mean norm difference between harmful and
    # harmless. The layer where the norm gap is highest is treated as the
    # primary refusal signal; emergence is where the gap first exceeds 30% of
    # its peak value.
    norm_gaps = []
    for layer_idx in range(n_layers):
        harmful_mean_norm = (
            harmful_acts[layer_idx].mean(dim=(0, 1)).norm().item()
        )
        harmless_mean_norm = (
            harmless_acts[layer_idx].mean(dim=(0, 1)).norm().item()
        )
        norm_gaps.append(abs(harmful_mean_norm - harmless_mean_norm))

    peak = max(norm_gaps) if norm_gaps else 1.0
    if peak < 1e-8:
        return n_layers - 1

    emergence_threshold = threshold * peak
    for layer_idx, gap in enumerate(norm_gaps):
        if gap >= emergence_threshold:
            return layer_idx

    return n_layers - 1


def _classify_alignment(
    emergence_frac: float,
    concentration: float,
    top5_concentration: float,
) -> Tuple[str, float]:
    """Classify alignment type from geometric features.

    Returns ``(alignment_type, confidence)`` where confidence ∈ [0, 1].

    Decision rules:
    - SFT:     emergence_frac < 0.35 AND concentration > 0.65 → high confidence
    - RLHF:    emergence_frac > 0.50 AND 0.55 < concentration < 0.80
    - DPO:     concentration < 0.60 OR top5_concentration > 0.90 (diffuse signal)
    - unknown: doesn't fit any pattern cleanly
    """
    sft_match = emergence_frac < 0.35 and concentration > 0.65
    rlhf_match = emergence_frac > 0.50 and 0.55 < concentration < 0.80
    dpo_match = concentration < 0.60 or top5_concentration > 0.90

    # Score each type to pick the best match and compute confidence.
    if sft_match and not rlhf_match:
        # Confidence scales with how strongly both conditions are met.
        emergence_margin = (0.35 - emergence_frac) / 0.35          # 0→1
        concentration_margin = (concentration - 0.65) / 0.35       # 0→1
        confidence = min(1.0, 0.5 + 0.25 * emergence_margin + 0.25 * concentration_margin)
        # Reduce confidence if DPO features also appear (ambiguous).
        if dpo_match:
            confidence = max(0.0, confidence - 0.2)
        return "sft", round(confidence, 3)

    if rlhf_match and not sft_match:
        emergence_margin = (emergence_frac - 0.50) / 0.50          # 0→1
        # Concentration should be in [0.55, 0.80]; score by distance from midpoint 0.675.
        conc_closeness = 1.0 - abs(concentration - 0.675) / 0.125  # 0→1 within range
        confidence = min(1.0, 0.5 + 0.2 * emergence_margin + 0.2 * conc_closeness)
        if dpo_match:
            confidence = max(0.0, confidence - 0.15)
        return "rlhf", round(confidence, 3)

    if dpo_match and not sft_match and not rlhf_match:
        # DPO signal: low concentration or very diffuse top-5.
        conc_margin = max(0.0, (0.60 - concentration) / 0.60)
        diffuse_margin = max(0.0, (top5_concentration - 0.90) / 0.10)
        confidence = min(1.0, 0.5 + 0.3 * conc_margin + 0.2 * diffuse_margin)
        return "dpo", round(confidence, 3)

    # Ambiguous: multiple or no patterns match.
    return "unknown", 0.3


def _get_recommendation(
    alignment_type: str,
    confidence: float,
    concentration: float,
) -> Tuple[str, int, str]:
    """Return ``(recommended_preset, recommended_n_directions, reasoning)``.

    Mapping:
    - SFT     → standard (n_directions=2): single direction usually sufficient;
                 n_directions=2 adds safety margin.
    - DPO     → deep (n_directions=3): multi-direction needed for diffuse refusal.
    - RLHF    → surgical (head-targeted): self-repair likely, head ablation better.
    - unknown or low confidence → deep (conservative, catches most cases).
    """
    if alignment_type == "sft" and confidence >= 0.6:
        preset = "standard"
        n_directions = 2
        reasoning = (
            "SFT models show sharp, early-layer refusal with high direction "
            "concentration ({:.0%} top-1 variance). The standard preset with "
            "n_directions=2 applies single-direction orthogonalization at the "
            "middle layers, which is sufficient for localized SFT refusal "
            "circuits. A second direction provides a safety margin against "
            "minor backup pathways.".format(concentration)
        )

    elif alignment_type == "dpo" and confidence >= 0.5:
        preset = "deep"
        n_directions = 3
        reasoning = (
            "DPO models exhibit diffuse refusal signal spread across multiple "
            "directions (top-1 concentration {:.0%}). The deep preset with "
            "n_directions=3 applies iterative subspace deflation (§5.1) to "
            "remove the primary direction and reveal dormant backup circuits "
            "that a single-direction approach would miss.".format(concentration)
        )

    elif alignment_type == "rlhf" and confidence >= 0.5:
        preset = "surgical"
        n_directions = 2
        reasoning = (
            "RLHF models develop late-layer refusal with strong self-repair "
            "tendency (concentration {:.0%}, late emergence). The surgical "
            "preset performs targeted attention-head ablation to disable the "
            "specific refusal-mediating heads while minimising KL divergence "
            "from the clean distribution, mitigating self-repair.".format(concentration)
        )

    else:
        # Low confidence or unknown: fall back to conservative deep preset.
        preset = "deep"
        n_directions = 3
        if alignment_type == "unknown":
            reasoning = (
                "Alignment type could not be determined with confidence "
                "(no geometric signature matched cleanly). Defaulting to the "
                "deep preset with n_directions=3 as the most conservative "
                "choice that handles SFT, DPO, and RLHF refusal patterns."
            )
        else:
            reasoning = (
                "Detected {} alignment type but with low confidence ({:.0%}). "
                "Defaulting to the deep preset with n_directions=3 to handle "
                "potential misclassification conservatively.".format(
                    alignment_type.upper(), confidence
                )
            )

    return preset, n_directions, reasoning


# =============================================================================
# Public API
# =============================================================================

def alignment_imprint_detection(
    model,
    harmful_train: List[str],
    harmless_train: List[str],
    token_positions: List[int],
    batch_size: int,
    refusal_token_ids: List[int],
) -> AlignmentImprint:
    """Detect the alignment training type from activation geometry.

    Steps:
    1. Collect activations on harmful and harmless prompts.
    2. Compute explained variance ratio (top-1 and top-5) at the middle layer.
    3. Compute logit-lens refusal probability to find emergence_layer.
    4. Apply heuristics to classify alignment type.
    5. Return recommendation.

    The logit-lens step is done via a simple inline computation if
    ``src.analysis.logit_lens`` is available; otherwise uses activation norms
    as a proxy.

    Args:
        model: A :class:`~src.model.RefusalModel` instance.
        harmful_train: List of harmful training prompts.
        harmless_train: List of harmless training prompts.
        token_positions: Post-instruction token positions (negative indices).
        batch_size: Forward-pass batch size.
        refusal_token_ids: Token IDs used to compute the refusal metric.

    Returns:
        An :class:`AlignmentImprint` dataclass with the detected type,
        confidence, geometric features, and recommended ablation strategy.
    """
    # Lazy imports to avoid circular dependencies.
    from ..extraction import (
        collect_activations,
        compute_explained_variance_ratio,
    )

    n_layers = model.n_layers

    # ------------------------------------------------------------------
    # 1. Collect activations on training sets.
    # ------------------------------------------------------------------
    harmful_acts = collect_activations(
        model, harmful_train, token_positions, batch_size,
    )
    harmless_acts = collect_activations(
        model, harmless_train, token_positions, batch_size,
    )

    # ------------------------------------------------------------------
    # 2. Compute variance ratios at the middle layer.
    #    The middle layer is the most stable point for concentration
    #    analysis — early layers capture embedding effects, late layers
    #    capture output formatting, middle layers carry refusal signal.
    # ------------------------------------------------------------------
    mid_layer = n_layers // 2
    pos_idx = 0  # Use the first (and typically only) token position.

    top1_ratio, ratios = compute_explained_variance_ratio(
        harmful_acts,
        harmless_acts,
        layer_idx=mid_layer,
        pos_idx=pos_idx,
        top_k=5,
    )
    top5_ratio = sum(ratios)

    # ------------------------------------------------------------------
    # 3. Find refusal emergence layer via logit-lens or norm proxy.
    # ------------------------------------------------------------------
    emergence_layer = _find_emergence_layer_logit_lens(
        model,
        harmful_acts,
        harmless_acts,
        refusal_token_ids,
        threshold=0.3,
    )
    emergence_frac = emergence_layer / max(n_layers - 1, 1)

    # ------------------------------------------------------------------
    # 4. Classify alignment type.
    # ------------------------------------------------------------------
    alignment_type, confidence = _classify_alignment(
        emergence_frac=emergence_frac,
        concentration=top1_ratio,
        top5_concentration=top5_ratio,
    )

    # ------------------------------------------------------------------
    # 5. Get recommendation.
    # ------------------------------------------------------------------
    preset, n_directions, reasoning = _get_recommendation(
        alignment_type=alignment_type,
        confidence=confidence,
        concentration=top1_ratio,
    )

    return AlignmentImprint(
        alignment_type=alignment_type,
        confidence=confidence,
        refusal_emergence_layer=emergence_layer,
        refusal_emergence_frac=emergence_frac,
        direction_concentration=top1_ratio,
        top5_concentration=top5_ratio,
        recommended_preset=preset,
        recommended_n_directions=n_directions,
        reasoning=reasoning,
    )


def print_alignment_report(imprint: AlignmentImprint) -> None:
    """Pretty-print the alignment imprint analysis to stdout."""
    width = 72
    sep = "=" * width
    thin = "-" * width

    print(sep)
    print("  ALIGNMENT IMPRINT DETECTION REPORT")
    print(sep)
    print(f"  Alignment type   : {imprint.alignment_type.upper()}")
    print(f"  Confidence       : {imprint.confidence:.1%}")
    print(thin)
    print("  GEOMETRIC FEATURES")
    print(thin)
    print(f"  Emergence layer  : {imprint.refusal_emergence_layer}"
          f"  ({imprint.refusal_emergence_frac:.1%} of layers)")
    print(f"  Top-1 conc.      : {imprint.direction_concentration:.1%}"
          "  (fraction of variance in top singular direction)")
    print(f"  Top-5 conc.      : {imprint.top5_concentration:.1%}"
          "  (cumulative top-5 directions)")
    print(thin)
    print("  RECOMMENDATION")
    print(thin)
    print(f"  Preset           : {imprint.recommended_preset}")
    print(f"  n_directions     : {imprint.recommended_n_directions}")
    print()
    # Word-wrap the reasoning to fit within the column width.
    words = imprint.reasoning.split()
    line: List[str] = []
    for word in words:
        if sum(len(w) for w in line) + len(line) + len(word) > width - 4:
            print("  " + " ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        print("  " + " ".join(line))
    print(sep)
