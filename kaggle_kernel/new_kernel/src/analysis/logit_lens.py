"""
Logit-lens analysis for refusal direction ablation.

The logit-lens projects intermediate residual stream activations through
the model's LM head (unembedding matrix) to compute per-layer token
probability distributions. For refusal analysis this reveals exactly
which transformer layer causes the model to "decide" to refuse.

Reference: Nostalgebraist (2020) "interpreting GPT: the logit lens"
           Arditi et al. (2024) §5 — mechanistic analysis of adversarial suffixes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


# =============================================================================
# Result dataclass
# =============================================================================

@dataclass
class LogitLensResult:
    """Per-layer refusal probability profile for a single prompt."""

    prompt: str
    refusal_probs: List[float]    # One entry per layer (0 = embed, L = final)
    emergence_layer: int           # First layer where refusal_prob > emergence_threshold
    peak_layer: int                # Layer with highest refusal_prob
    emergence_threshold: float     # Threshold used to find emergence_layer
    n_layers: int


# =============================================================================
# Internal helpers
# =============================================================================

def _get_lm_head(model_obj) -> Optional[nn.Module]:
    """Retrieve the LM head (unembedding matrix) from the HF model.

    Tries the inner `model.lm_head` attribute first, then falls back to
    the top-level model object.  Returns ``None`` if nothing is found so
    the caller can handle gracefully.
    """
    inner = getattr(model_obj, "model", None)
    if inner is not None:
        lm_head = getattr(inner, "lm_head", None)
        if lm_head is not None:
            return lm_head
    return getattr(model_obj, "lm_head", None)


def _get_final_norm(model_obj) -> Optional[nn.Module]:
    """Return the final RMSNorm / LayerNorm before lm_head.

    Mirrors the logic in ``model.get_final_norm_layer`` but operates
    directly on a HF PreTrainedModel so this module has no hard dependency
    on the RefusalModel wrapper.

    Path 1: Llama-style  → ``hf.model.norm``
    Path 2: GPT-2-style  → ``hf.transformer.ln_f``
    Path 3: Direct       → ``hf.norm``
    """
    inner = getattr(model_obj, "model", None)
    if inner is not None:
        norm = getattr(inner, "norm", None)
        if norm is not None:
            return norm

    transformer = getattr(model_obj, "transformer", None)
    if transformer is not None:
        norm = getattr(transformer, "ln_f", None)
        if norm is not None:
            return norm

    return getattr(model_obj, "norm", None)


def _get_decoder_layers(model_obj) -> nn.ModuleList:
    """Return the ModuleList of transformer decoder layers.

    Covers Llama-style (``model.model.layers``) and GPT-2-style
    (``model.transformer.h``).
    """
    inner = getattr(model_obj, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    transformer = getattr(model_obj, "transformer", None)
    if transformer is not None and hasattr(transformer, "h"):
        return transformer.h
    raise RuntimeError(
        f"Cannot locate decoder layers on {type(model_obj).__name__}; "
        "extend _get_decoder_layers() in logit_lens.py to support this architecture."
    )


def _compute_refusal_prob(
    hidden: torch.Tensor,          # (d_model,) float32
    lm_head: nn.Module,
    final_norm: Optional[nn.Module],
    refusal_token_ids: Sequence[int],
    normalize: bool,
) -> float:
    """Project one hidden state through (optional norm +) lm_head → P(refusal).

    All computation is done in float32 for numerical stability.
    """
    h = hidden.float().unsqueeze(0)  # (1, d_model)

    with torch.no_grad():
        if normalize and final_norm is not None:
            h = final_norm(h)
        logits = lm_head(h)           # (1, vocab_size)
        probs = F.softmax(logits, dim=-1)  # (1, vocab_size)
        refusal_prob = probs[0, list(refusal_token_ids)].sum().item()

    return float(refusal_prob)


# =============================================================================
# Core function
# =============================================================================

def logit_lens_refusal_probability(
    model,
    prompt: str,
    refusal_token_ids: Sequence[int],
    normalize: bool = True,
    emergence_threshold: float = 0.3,
) -> LogitLensResult:
    """Compute per-layer refusal probability via the logit lens.

    For each layer l, we:
    1. Hook the pre-norm residual stream output at layer l
    2. Apply the model's final layer norm to it (if available)
    3. Pass through lm_head to get logits
    4. Compute P(refusal) = sum of softmax probs over refusal_token_ids

    This is done in a single forward pass by collecting intermediate
    activations at all layers simultaneously.

    Args:
        model: RefusalModel wrapper (or any object with ``.model``,
               ``.format()``, and ``.tokenize()`` attributes).
        prompt: The prompt to analyze (will be formatted via model.format).
        refusal_token_ids: Token IDs that indicate refusal.
        normalize: If True, apply layer norm before unembedding.
                   If False, use raw residual stream (faster but noisier).
        emergence_threshold: Threshold used to determine ``emergence_layer``.

    Returns:
        LogitLensResult with per-layer refusal probabilities and
        the estimated refusal emergence layer.
    """
    hf_model = model.model  # underlying HF PreTrainedModel

    lm_head = _get_lm_head(hf_model)
    if lm_head is None:
        raise RuntimeError(
            "Could not locate lm_head on the model. "
            "Ensure the model has a `lm_head` attribute."
        )

    final_norm = _get_final_norm(hf_model) if normalize else None
    decoder_layers = _get_decoder_layers(hf_model)
    n_layers = len(decoder_layers)

    # Buffer: layer_idx -> last-token residual (1, d_model) on CPU
    cache: Dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(_module, inputs):
            # inputs[0]: (batch, seq_len, d_model) — residual stream entering layer
            x = inputs[0]
            # Grab the last token position (left-padded, so -1 is always valid).
            cache[layer_idx] = x[:, -1, :].detach().float().cpu()
            return None  # do not modify inputs
        return hook

    handles = []
    for idx, layer in enumerate(decoder_layers):
        handles.append(layer.register_forward_pre_hook(make_hook(idx)))

    try:
        formatted = model.format(prompt)
        enc = model.tokenize([formatted])
        with torch.no_grad():
            hf_model(**enc)
    finally:
        for h in handles:
            h.remove()

    # Build per-layer refusal probability list.
    # Layer 0 corresponds to the residual entering the first decoder block
    # (i.e., post-embedding, pre-layer-0 attention).
    refusal_probs: List[float] = []
    for layer_idx in range(n_layers):
        hidden = cache[layer_idx].squeeze(0)  # (d_model,)
        prob = _compute_refusal_prob(
            hidden, lm_head, final_norm, refusal_token_ids, normalize
        )
        refusal_probs.append(prob)

    # Derive summary statistics.
    peak_layer = int(max(range(n_layers), key=lambda i: refusal_probs[i]))

    emergence_layer = n_layers  # sentinel: never exceeded threshold
    for i, p in enumerate(refusal_probs):
        if p > emergence_threshold:
            emergence_layer = i
            break

    return LogitLensResult(
        prompt=prompt,
        refusal_probs=refusal_probs,
        emergence_layer=emergence_layer,
        peak_layer=peak_layer,
        emergence_threshold=emergence_threshold,
        n_layers=n_layers,
    )


# =============================================================================
# Batch helper
# =============================================================================

def batch_logit_lens(
    model,
    prompts: List[str],
    refusal_token_ids: Sequence[int],
    batch_size: int = 4,
    normalize: bool = True,
    emergence_threshold: float = 0.3,
) -> List[LogitLensResult]:
    """Run logit-lens analysis on multiple prompts.

    Processes prompts sequentially in chunks of ``batch_size``.  Each chunk
    runs a single forward pass; per-layer probabilities are computed by taking
    the last-token residual from each prompt in the batch.

    Args:
        model: RefusalModel wrapper.
        prompts: List of instruction strings to analyze.
        refusal_token_ids: Token IDs that indicate refusal.
        batch_size: Number of prompts per forward pass.
        normalize: Whether to apply final layer norm before unembedding.
        emergence_threshold: Threshold for emergence_layer computation.

    Returns:
        List of LogitLensResult, one per prompt, in the same order.
    """
    results: List[LogitLensResult] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        for prompt in batch_prompts:
            result = logit_lens_refusal_probability(
                model=model,
                prompt=prompt,
                refusal_token_ids=refusal_token_ids,
                normalize=normalize,
                emergence_threshold=emergence_threshold,
            )
            results.append(result)

    return results


# =============================================================================
# Emergence layer query
# =============================================================================

def refusal_emergence_layer(
    result: LogitLensResult,
    threshold: float = 0.3,
) -> int:
    """Return the first layer where refusal probability exceeds threshold.

    If no layer exceeds the threshold, returns ``result.n_layers`` as a
    sentinel value indicating that refusal never emerged above the given
    threshold in this result.

    Args:
        result: A LogitLensResult from ``logit_lens_refusal_probability``.
        threshold: Probability threshold (default 0.3).

    Returns:
        Index of the first layer exceeding ``threshold``, or ``n_layers``
        if none do.
    """
    for i, p in enumerate(result.refusal_probs):
        if p > threshold:
            return i
    return result.n_layers


# =============================================================================
# Visualization
# =============================================================================

def plot_refusal_trajectory(
    results: List[LogitLensResult],
    labels: Optional[List[str]] = None,
    title: str = "Refusal Probability by Layer (Logit Lens)",
    save_path: Optional[str] = None,
) -> None:
    """Plot per-layer refusal probability for one or more prompts.

    Uses matplotlib. Each result is one line on the chart.
    X-axis: layer index. Y-axis: refusal probability [0, 1].
    Draws a horizontal dashed line at 0.5 (decision boundary).

    If save_path is provided, saves as PNG. Otherwise calls plt.show().
    Gracefully skips if matplotlib is not installed.

    Args:
        results: List of LogitLensResult objects to plot.
        labels: Optional list of legend labels (one per result). Defaults
                to truncated prompt text when not provided.
        title: Plot title.
        save_path: Optional filesystem path to save the figure as PNG.
                   When ``None``, the figure is displayed interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "[logit_lens] matplotlib is not installed; "
            "skipping plot_refusal_trajectory. "
            "Install with: pip install matplotlib"
        )
        return

    if not results:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, result in enumerate(results):
        if labels is not None and i < len(labels):
            label = labels[i]
        else:
            # Truncate prompt to 40 chars for a readable default label.
            label = result.prompt[:40] + ("..." if len(result.prompt) > 40 else "")

        x = list(range(result.n_layers))
        ax.plot(x, result.refusal_probs, marker="o", markersize=3, label=label)

    # Decision-boundary reference line.
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1.0, alpha=0.7,
               label="Decision boundary (0.5)")

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Refusal probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize="small")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[logit_lens] saved trajectory plot to: {save_path}")
    else:
        plt.show()

    plt.close(fig)
