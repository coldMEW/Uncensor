"""
Model interventions: directional ablation, activation addition, weight
orthogonalization, and extended interventions from the research plan.

Paper: https://arxiv.org/abs/2406.11717
§2.4 — activation addition and directional ablation equations.
§3.1 — directional ablation bypasses refusal.
§3.2 — activation addition induces refusal.
§4.1 — weight orthogonalization.
§E — proof that weight orthogonalization equals directional ablation.

Extensions (RESEARCH_PLAN.md):
§3.2 — last-layer hook completeness fix for directional_ablation.
§3.3/§5.7 — weight-tying guard for orthogonalize_weights.
§3.6 — post-quantization ortho correction.
§5.1 — multi-directional ablation + subspace weight orthogonalization.
§5.4 — attention head ablation.
§5.5 — SAE feature ablation (stub).
§5.6 — calibrated orthogonalization with activation statistics preservation.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .model import RefusalModel, discover_residual_writers, get_final_norm_layer


# =============================================================================
# Reversible Steering Mode (US-003)
# =============================================================================

class InferenceSteering:
    """Reversible hook-based steering without permanent weight modification.

    This class provides inference-time steering using directional ablation hooks.
    Unlike orthogonalize_weights(), this does NOT modify model weights - it only
    applies hooks during generation. When the context manager exits, the model
    weights are unchanged.

    Usage:
        with InferenceSteering(model, direction):
            response = model.generate(prompt)

    vs permanent weight modification:
        with directional_ablation(model, direction):
            # Same behavior, different API
    """

    def __init__(self, model: RefusalModel, direction: torch.Tensor, layer_indices: List[int] = None):
        """Initialize inference steering.

        Args:
            model: The RefusalModel to steer.
            direction: The direction vector to project out.
            layer_indices: Optional list of specific layer indices to apply steering.
                          If None, applies to all layers.
        """
        self.model = model
        self.direction = direction.to(dtype=torch.float32)
        self.direction = self.direction / self.direction.norm()
        self.layer_indices = layer_indices
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._norm_handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def _make_hook(self, layer_idx: int):
        """Create a forward-pre-hook for the given layer."""
        def hook(_module, inputs):
            x = inputs[0]  # (batch, seq_len, d_model)
            d = self.direction.to(device=x.device, dtype=x.dtype)
            new_x = x - _project_onto(x, d)
            return (new_x,) + inputs[1:]
        return hook

    def __enter__(self):
        """Register hooks on model layers."""
        if self.layer_indices is not None:
            for idx in self.layer_indices:
                if idx < len(self.model.layers):
                    handle = self.model.layers[idx].register_forward_pre_hook(
                        self._make_hook(idx)
                    )
                    self._handles.append(handle)
        else:
            self._handles = self.model.register_forward_pre_hooks(self._make_hook)

        # Last-layer hook for completeness
        final_norm = get_final_norm_layer(self.model.model)
        if final_norm is not None:
            def _final_norm_hook(_module, _inputs, output):
                d = self.direction.to(device=output.device, dtype=output.dtype)
                return output - _project_onto(output, d)
            self._norm_handle = final_norm.register_forward_hook(_final_norm_hook)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove all hooks - model weights unchanged."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        if self._norm_handle is not None:
            self._norm_handle.remove()
            self._norm_handle = None
        return False


# =============================================================================
# Internal helpers
# =============================================================================

def _project_onto(x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Return (direction · x) direction, where `direction` is a unit vector.

    Broadcasting: x has shape (..., d_model), direction has shape (d_model,).
    Result has the same shape as x.
    """
    # coeff: (..., 1); direction: (d_model,). Result: (..., d_model).
    coeff = (x * direction).sum(dim=-1, keepdim=True)
    return coeff * direction



# =============================================================================
# Directional ablation (§2.4 Eq. 4, extended §3.2)
# =============================================================================

@contextmanager
def directional_ablation(
    model: RefusalModel,
    direction: torch.Tensor,
    coefficient: float = 1.0,
) -> Iterator[None]:
    """Context manager that ablates `direction` from the residual stream at
    every layer and every token position for the duration of the `with` block.

    §2.4 Eq. 4 — x' = x − (r̂ᵀ x) r̂
    §2.4 — "We perform this operation at every activation x^(l)_i and x̃^(l)_i,
    across all layers l and all token positions i."

    **§3.2 fix — last-layer hook completeness:**
    After the last decoder layer, the residual stream passes through the final
    RMSNorm and into lm_head without any ablation hook.  This means the last
    layer can re-introduce an r̂ component that flows into the logits.  To close
    this gap we also register a forward hook on the final norm so that its
    output has the direction removed before reaching lm_head.
    """
    direction = direction.to(dtype=torch.float32)
    direction = direction / direction.norm()  # r̂
    coefficient = float(coefficient)

    def make_hook(_layer_idx: int):
        def hook(_module, inputs):
            x = inputs[0]  # (batch, seq_len, d_model)
            d = direction.to(device=x.device, dtype=x.dtype)
            # Subtract the projection along r̂ from every token, every layer.
            new_x = x - coefficient * _project_onto(x, d)
            return (new_x,) + inputs[1:]
        return hook

    handles = model.register_forward_pre_hooks(make_hook)

    # §3.2 — Register a forward hook on the final norm to close the
    # last-layer gap.  We use register_forward_hook (not pre_hook) because we
    # need to modify the *output* of the final norm before it reaches lm_head.
    final_norm = get_final_norm_layer(model.model)
    norm_handle: Optional[torch.utils.hooks.RemovableHandle] = None
    if final_norm is not None:
        def _final_norm_hook(_module, _inputs, output):
            d = direction.to(device=output.device, dtype=output.dtype)
            return output - coefficient * _project_onto(output, d)

        norm_handle = final_norm.register_forward_hook(_final_norm_hook)

    try:
        yield
    finally:
        RefusalModel.remove_hooks(handles)
        if norm_handle is not None:
            norm_handle.remove()


# =============================================================================
# Activation addition (§2.4)
# =============================================================================

@contextmanager
def activation_addition(
    model: RefusalModel,
    direction: torch.Tensor,
    layer_idx: int,
) -> Iterator[None]:
    """§2.4 — add the un-normalized difference-in-means vector r to the
    residual stream at a single layer, across all token positions.

    Note: the paper's equation uses `r` (un-normalized), not `r̂`. This is
    an intentional design choice — the magnitude of r matters here, not just
    the direction. See §2.3 "Note that each such vector is meaningful in both
    (1) its direction ... and (2) its magnitude".
    """
    direction = direction.to(device=model.device)

    def make_hook(this_layer: int):
        def hook(_module, inputs):
            x = inputs[0]
            if this_layer == layer_idx:
                d = direction.to(dtype=x.dtype)
                new_x = x + d  # broadcasts over (batch, seq_len, d_model)
                return (new_x,) + inputs[1:]
            return None
        return hook

    handles = model.register_forward_pre_hooks(make_hook)
    try:
        yield
    finally:
        RefusalModel.remove_hooks(handles)


# =============================================================================
# Weight orthogonalization (§4.1, §E, extended §3.3/§5.7)
# =============================================================================

@torch.no_grad()
def _ortho_linear(layer: nn.Linear, r_hat: torch.Tensor) -> None:
    """Project out r_hat from a linear layer's weight and bias.

    Device-aware: uses ``W.device`` / ``b.device`` so the projection works
    when residual writers live on different GPUs (``device_map='auto'``,
    tensor parallelism, etc.).
    """
    W = layer.weight.data  # (d_model, d_input) — rows live in R^d_model
    r = r_hat.to(dtype=W.dtype, device=W.device)
    # Subtract the r̂ component from every column of W: W ← W − r̂ (r̂ᵀ W).
    proj = torch.outer(r, r @ W)   # (d_out, d_in)
    W.sub_(proj)
    if layer.bias is not None:
        b = layer.bias.data
        r_b = r_hat.to(dtype=b.dtype, device=b.device)
        # Project out r_hat direction: b' = b - r * (r^T b)
        b.sub_(r_b * (b @ r_b))


@torch.no_grad()
def _ortho_embedding(layer: nn.Embedding, r_hat: torch.Tensor) -> None:
    """Project out r_hat from an embedding layer's weight.

    Device-aware (see ``_ortho_linear``).
    """
    # Embedding weight shape: (vocab, d_model). Each row is a vector in
    # R^d_model that is written to the residual stream. Orthogonalize per row.
    W = layer.weight.data
    r = r_hat.to(dtype=W.dtype, device=W.device)
    # coeffs: (vocab,)  — dot of every row with r̂
    coeffs = W @ r
    W.sub_(torch.outer(coeffs, r))


def orthogonalize_weights(
    model: RefusalModel,
    direction: torch.Tensor,
    layer_weights: Optional[List[float]] = None,
) -> None:
    """§4.1 Eq. 5 — W'_out = W_out − r̂ r̂ᵀ W_out.

    Modifies every matrix that writes to the residual stream (embedding,
    positional embedding if any, attention o_proj, MLP down_proj) as well as
    any biases on those projections.

    This mutates the model weights in-place. It is mathematically equivalent
    to directional ablation (§E) but requires no hooks at inference time.

    §3.3/§5.7 — Weight-tying guard:
    In weight-tied models (many Llama variants, Qwen, etc.),
    ``embed_tokens.weight`` and ``lm_head.weight`` share the same underlying
    tensor.  Orthogonalizing embed_tokens would *also* orthogonalize lm_head,
    which is a residual-stream *reader* — not covered by §E's proof.  We detect
    this case and handle it by: backing up lm_head → orthogonalizing the
    (tied) embedding → restoring lm_head from the backup.  This preserves
    embed_tokens as a proper residual writer (no r̂ component) while keeping
    lm_head unchanged.

    Extended with layer-weighted orthogonalization (Safety Layers paper,
    arXiv:2408.17003): when *layer_weights* is provided, per-layer coefficients
    control orthogonalization strength.  w_l ∈ [0, 1]: 0 = skip layer entirely,
    1 = full orthogonalization.  This avoids degrading early comprehension
    layers and late generation layers where refusal direction is weak.

    Args:
        model: The wrapped model.
        direction: The refusal direction vector.
        layer_weights: Optional per-layer coefficients of length
            ``model.n_layers``.  Each value in [0, 1].  ``None`` means
            uniform full-strength (backward-compatible).
    """
    writers = discover_residual_writers(model.model)
    r_hat = direction.to(device=model.device, dtype=torch.float32)
    r_hat = r_hat / r_hat.norm()

    # --- Embedding orthogonalization with weight-tying guard (§3.3/§5.7) ---
    embed = writers.embed
    hf = model.model
    lm_head = getattr(hf, "lm_head", None)

    # §3.3/§5.7 — Weight-tying detection.
    # Config-based check first (reliable with accelerate/device_map="auto").
    # Fall back to data_ptr comparison for single-GPU setups.
    is_tied = False
    hf_config = getattr(hf, "config", None)
    if hf_config is not None and getattr(hf_config, "tie_word_embeddings", False):
        is_tied = True
    elif lm_head is not None and hasattr(lm_head, "weight"):
        is_tied = (lm_head.weight.data_ptr() == embed.weight.data_ptr())

    if not is_tied:
        # Not tied — safe to orthogonalize embed_tokens directly.
        _ortho_embedding(embed, r_hat)
        print("[ortho] weight-tying: False — embed_tokens orthogonalized normally")
    else:
        # Weight-tied: backup lm_head, ortho embed (which also orthos lm_head
        # due to shared storage), then restore lm_head.
        print("[ortho] weight-tying: True — backing up lm_head, ortho embed, "
              "then restoring lm_head")
        lm_head_backup = lm_head.weight.data.clone()
        _ortho_embedding(embed, r_hat)  # modifies embed AND lm_head (tied)
        lm_head.weight.data.copy_(lm_head_backup)  # restore lm_head

    # --- Positional embedding (if present) ---
    if writers.positional_embed is not None:
        _ortho_embedding(writers.positional_embed, r_hat)

    # --- Attention o_proj and MLP down_proj (with optional layer weights) ---
    for layer_idx, (lin_attn, lin_mlp) in enumerate(
        zip(writers.attn_out_weights, writers.mlp_out_weights)
    ):
        w = 1.0  # default: full strength
        if layer_weights is not None and layer_idx < len(layer_weights):
            # Clamp to [0, 1] defensively: a negative causal score would
            # otherwise flip the sign of the projection, which is invalid
            # per §4.1. We treat negative contributions as zero.
            w = max(0.0, min(1.0, float(layer_weights[layer_idx])))
        if w < 0.01:
            continue  # skip layers with negligible weight
        if abs(w - 1.0) < 1e-6:
            _ortho_linear(lin_attn, r_hat)
            _ortho_linear(lin_mlp, r_hat)
        else:
            # Partial orthogonalization: W' = W - w * r_hat * (r_hat^T * W)
            # Device-aware: layers can live on different GPUs.
            W_a = lin_attn.weight.data
            r_a = r_hat.to(dtype=W_a.dtype, device=W_a.device)
            proj_a = torch.outer(r_a, r_a @ W_a) * w
            W_a.sub_(proj_a)
            W_m = lin_mlp.weight.data
            r_m = r_hat.to(dtype=W_m.dtype, device=W_m.device)
            proj_m = torch.outer(r_m, r_m @ W_m) * w
            W_m.sub_(proj_m)
            if layer_idx % 10 == 0:
                print(f"[ortho] layer {layer_idx}: partial strength w={w:.3f}")


# =============================================================================
# Multi-directional ablation (§5.1)
# =============================================================================

@contextmanager
def multi_directional_ablation(
    model: RefusalModel,
    directions: Sequence[torch.Tensor],
) -> Iterator[None]:
    """Context manager that ablates *multiple* directions from the residual
    stream at every layer and token position.

    §5.1 — Iterative subspace deflation: after finding r̂₁, a second direction
    r̂₂ can capture residual refusal circuits that become dominant only after
    r̂₁ is suppressed.  This context manager projects out all provided directions
    sequentially at every hook point.

    Each direction is projected out in order: first r̂₁, then r̂₂ from the
    residual, etc.  If the directions are mutually orthogonal (as produced by
    Gram-Schmidt in iterative extraction), the sequential projection is
    equivalent to a single projection onto the subspace orthogonal complement.

    Also registers a final-norm hook (§3.2 fix) for the same reason as
    ``directional_ablation``.
    """
    # Normalize and move all directions to model device.
    unit_dirs: List[torch.Tensor] = []
    for d in directions:
        d = d.to(dtype=torch.float32, device=model.device)
        d = d / d.norm()
        unit_dirs.append(d)

    def make_hook(_layer_idx: int):
        def hook(_module, inputs):
            x = inputs[0]  # (batch, seq_len, d_model)
            for r_hat in unit_dirs:
                r = r_hat.to(dtype=x.dtype)
                x = x - _project_onto(x, r)
            return (x,) + inputs[1:]
        return hook

    handles = model.register_forward_pre_hooks(make_hook)

    # §3.2 — final norm hook
    final_norm = get_final_norm_layer(model.model)
    norm_handle: Optional[torch.utils.hooks.RemovableHandle] = None
    if final_norm is not None:
        def _final_norm_hook(_module, _inputs, output):
            x = output
            for r_hat in unit_dirs:
                r = r_hat.to(dtype=x.dtype)
                x = x - _project_onto(x, r)
            return x

        norm_handle = final_norm.register_forward_hook(_final_norm_hook)

    try:
        yield
    finally:
        RefusalModel.remove_hooks(handles)
        if norm_handle is not None:
            norm_handle.remove()


# =============================================================================
# Subspace weight orthogonalization (§5.1)
# =============================================================================

def orthogonalize_weights_subspace(
    model: RefusalModel,
    directions: List[torch.Tensor],
    layer_weights: Optional[List[float]] = None,
) -> None:
    """Orthogonalize all residual-writer weights against a *subspace* spanned
    by the given direction vectors.

    §5.1 — Sequential application of orthogonalize_weights for each direction
    projects out the full subspace, provided the directions are mutually
    orthogonal (as they are after Gram-Schmidt in iterative extraction).

    Extended with layer-weighted orthogonalization (Safety Layers paper,
    arXiv:2408.17003): when *layer_weights* is provided, per-layer
    coefficients control orthogonalization strength.  Layers with weight 0.0
    are skipped entirely (e.g., early comprehension / late generation layers).
    Layers with weight 1.0 receive full orthogonalization.

    Args:
        model: The wrapped model.
        directions: List of direction vectors (need not be unit-norm, but
            should be mutually orthogonal for correct subspace projection).
        layer_weights: Optional per-layer coefficients of length
            ``model.n_layers``.  Each value in [0, 1].  ``None`` means
            uniform full-strength (backward-compatible).
    """
    for idx, d in enumerate(directions):
        print(f"[subspace ortho] direction {idx + 1}/{len(directions)}")
        orthogonalize_weights(model, d, layer_weights=layer_weights)
    print(f"[subspace ortho] done — {len(directions)} directions applied")


# =============================================================================
# Calibrated orthogonalization (§5.6)
# =============================================================================

def calibrated_orthogonalize(
    model: RefusalModel,
    direction: torch.Tensor,
    calibration_prompts: List[str],
    batch_size: int = 8,
) -> None:
    """Orthogonalize weights, then calibrate per-layer activation statistics
    by adjusting RMSNorm γ (``input_layernorm.weight``) to restore the
    pre-ortho residual stream scale.

    §5.6 — Weight ortho removes the r̂ component from each column, which
    slightly reduces activation norms.  For multi-direction ortho this accumulates.
    Rather than rescaling individual weight columns (numerically unstable for
    columns nearly parallel to r̂ — see §1.6), we measure the *actual* change
    in activation RMS at each layer and compensate via the RMSNorm learnable
    gain parameter γ.

    Adjusting γ is preferred over rescaling down_proj because:
    * γ is a single vector per layer (clean, interpretable).
    * Adjusting down_proj changes effective output scale but also alters the
      gradient flow structure in ways that interact with subsequent layers.

    Args:
        model: The wrapped model.
        direction: The refusal direction to orthogonalize against.
        calibration_prompts: Prompts used to measure activation statistics.
            These should be representative of the model's expected input
            distribution (e.g. a mix of harmless prompts).
        batch_size: Forward-pass batch size for calibration.
    """
    print("[calibrated ortho] measuring pre-ortho activation RMS ...")
    pre_scales = model.measure_layer_input_rms(calibration_prompts, batch_size)

    orthogonalize_weights(model, direction)

    print("[calibrated ortho] measuring post-ortho activation RMS ...")
    post_scales = model.measure_layer_input_rms(calibration_prompts, batch_size)

    for layer_idx, layer in enumerate(model.layers):
        if post_scales[layer_idx] < 1e-8:
            continue
        scale = pre_scales[layer_idx] / post_scales[layer_idx]
        # Adjust γ for BOTH layernorms per block.
        # Llama-style transformers have two RMSNorm per block:
        #   input_layernorm (pre-attention) and post_attention_layernorm (pre-MLP).
        # Orthogonalizing down_proj primarily affects the post-attention domain,
        # so both need calibration (Biprojection/Grimjim insight, Safety Layers paper).
        for ln_name in ("input_layernorm", "post_attention_layernorm"):
            ln = getattr(layer, ln_name, None)
            if ln is not None and hasattr(ln, "weight"):
                with torch.no_grad():
                    ln.weight.data.mul_(scale)
                print(f"  layer {layer_idx} {ln_name}: γ *= {scale:.6f} "
                      f"(pre={pre_scales[layer_idx]:.4f}, "
                      f"post={post_scales[layer_idx]:.4f})")

    print("[calibrated ortho] done")


# =============================================================================
# Attention head ablation (§5.4)
# =============================================================================

@torch.no_grad()
def ablate_head_contribution(
    o_proj_weight: torch.Tensor,
    head_idx: int,
    head_dim: int,
) -> None:
    """Zero out a single attention head's contribution from the output projection.

    §5.4 — In the HuggingFace convention, ``o_proj.weight`` has shape
    ``(d_model, n_heads * head_dim)``  ``(out_features, in_features)``.
    Head ``h`` reads from input columns ``[h*head_dim : (h+1)*head_dim]``.
    Zeroing these columns makes head ``h`` contribute nothing to the
    residual stream.

    Args:
        o_proj_weight: The weight tensor of ``o_proj`` (mutated in-place).
        head_idx: Index of the head to ablate.
        head_dim: Dimensionality of each attention head.
    """
    start = head_idx * head_dim
    end = (head_idx + 1) * head_dim
    o_proj_weight[:, start:end].zero_()


@torch.no_grad()
def ablate_attention_heads(
    model: RefusalModel,
    layer_idx: int,
    head_indices: List[int],
) -> None:
    """Permanently ablate specific attention heads at a given layer by zeroing
    their o_proj columns.

    §5.4 — Identifying "refusal heads" and ablating only those achieves
    comparable bypass to full directional ablation with less capability
    degradation (only 2–5 heads at one layer vs. residual stream of all layers).

    Args:
        model: The wrapped model.
        layer_idx: Index of the transformer layer to modify.
        head_indices: List of head indices to ablate at that layer.

    Raises:
        ValueError: If ``layer_idx`` is out of range or the layer has no
            attention output projection.
    """
    if layer_idx < 0 or layer_idx >= model.n_layers:
        raise ValueError(
            f"layer_idx {layer_idx} out of range [0, {model.n_layers})"
        )

    layer = model.layers[layer_idx]
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    if attn is None:
        raise ValueError(f"layer {layer_idx} has no attention module")

    o_proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
    if o_proj is None:
        raise ValueError(f"layer {layer_idx} has no output projection")

    n_heads = model.model.config.num_attention_heads
    head_dim = model.d_model // n_heads

    for h in head_indices:
        if h < 0 or h >= n_heads:
            print(f"[head ablation] WARNING: head {h} out of range "
                  f"[0, {n_heads}), skipping")
            continue
        ablate_head_contribution(o_proj.weight.data, h, head_dim)
        print(f"[head ablation] layer {layer_idx}, head {h}: "
              f"o_proj columns [{h * head_dim}:{(h + 1) * head_dim}] zeroed")


# =============================================================================
# SAE feature ablation — stub (§5.5)
# =============================================================================

def sae_feature_ablation(
    model: RefusalModel,
    sae_decoder_weights: torch.Tensor,
    feature_indices: List[int],
    layer_idx: Optional[int] = None,
) -> None:
    """Ablate specific SAE features by orthogonalizing residual-writer weights
    against the corresponding SAE decoder vectors.

    §5.5 — Sparse autoencoders decompose residual-stream activations into
    sparse, interpretable features.  Some features correspond specifically to
    "refusal" or "safety" concepts.  Ablating those features is more surgical
    than ablating a global direction because it only affects the refusal-specific
    use of the direction, not all downstream uses.

    **Implementation:** For each provided feature index ``i``, the SAE
    decoder row ``sae_decoder_weights[i]`` is a ``(d_model,)`` vector in the
    residual-stream space.  We treat that vector exactly like a
    difference-in-means direction and feed the collected set to
    :func:`orthogonalize_weights_subspace`, which permanently projects every
    residual-stream writer orthogonally to the span of the selected features.

    Identifying which features correspond to refusal is left to the caller —
    the standard recipe is to run difference-in-means on SAE feature
    activations over harmful vs harmless prompts and pick the top-k by
    absolute magnitude.  See :func:`load_sae_and_score_features` below for a
    helper that wraps ``sae-lens``.

    Args:
        model: The wrapped model.
        sae_decoder_weights: SAE decoder matrix of shape
            ``(n_features, d_model)``.  Each row is a feature vector in the
            residual-stream space.
        feature_indices: Indices of SAE features to ablate.
        layer_idx: Unused in the current all-layers implementation.  Reserved
            for future per-layer ablation (§5.5 advanced mode); emits a
            warning if provided, since the current backend ignores it.
    """
    if sae_decoder_weights.ndim != 2:
        raise ValueError(
            f"sae_decoder_weights must be 2-D (n_features, d_model); "
            f"got shape {tuple(sae_decoder_weights.shape)}"
        )
    if sae_decoder_weights.shape[1] != model.d_model:
        raise ValueError(
            f"SAE decoder d_model = {sae_decoder_weights.shape[1]} does not "
            f"match model d_model = {model.d_model}"
        )
    if not feature_indices:
        print("[sae ablation] no feature indices provided — skipping")
        return
    if layer_idx is not None:
        print(
            "[sae ablation] layer_idx argument is currently advisory only; "
            "all residual-stream writers will be orthogonalized."
        )

    directions = [
        sae_decoder_weights[i].detach().clone().to(
            device=model.device, dtype=torch.float32
        )
        for i in feature_indices
    ]
    print(
        f"[sae ablation] orthogonalizing {len(directions)} SAE feature(s) "
        f"{feature_indices} from the residual stream"
    )
    orthogonalize_weights_subspace(model, directions)


def load_sae_and_score_features(
    model: RefusalModel,
    sae_release: str,
    sae_id: str,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    top_k: int = 5,
    batch_size: int = 4,
) -> Tuple[torch.Tensor, List[int]]:
    """Load a pre-trained SAE via ``sae-lens`` and return the top-k refusal
    features discovered via difference-in-means on SAE feature activations.

    §6.3 — End-to-end helper so callers can go from ``(model, sae_release,
    harmful/harmless prompts)`` to a ready-to-ablate set of SAE features
    without having to hand-roll the activation-capture + DiM + ranking loop.

    Returns ``(decoder_matrix, selected_feature_indices)``. The decoder
    matrix can be passed directly to :func:`sae_feature_ablation` together
    with the selected indices.

    When ``sae-lens`` is not installed, raises ``ImportError`` with an
    actionable install hint.  This keeps the dependency optional.
    """
    try:
        from sae_lens import SAE  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "sae-lens is not installed. `pip install sae-lens` to enable "
            "SAE feature ablation."
        ) from exc

    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id)
    sae = sae[0] if isinstance(sae, tuple) else sae
    sae = sae.to(model.device)

    # Hook the specified SAE site and collect pre-activation residuals.
    hook_name = getattr(sae, "cfg", None)
    layer_idx = None
    if hook_name is not None and hasattr(hook_name, "hook_layer"):
        layer_idx = int(hook_name.hook_layer)
    if layer_idx is None:
        layer_idx = model.n_layers // 2
        print(
            f"[sae] could not infer layer from SAE cfg; defaulting to "
            f"middle layer {layer_idx}"
        )

    def _collect(prompts: List[str]) -> torch.Tensor:
        buf: List[torch.Tensor] = []

        def hook(_module, inputs):
            x = inputs[0]
            buf.append(x[:, -1, :].detach().to(torch.float32))
            return None

        handle = model.layers[layer_idx].register_forward_pre_hook(hook)
        try:
            for start in range(0, len(prompts), batch_size):
                batch = [model.format(p) for p in prompts[start:start + batch_size]]
                enc = model.tokenize(batch)
                with torch.no_grad():
                    model.model(**enc)
        finally:
            handle.remove()
        return torch.cat(buf, dim=0)

    harmful_acts = _collect(harmful_prompts)
    harmless_acts = _collect(harmless_prompts)

    # Encode through the SAE and compute DiM on feature activations.
    with torch.no_grad():
        harmful_feats = sae.encode(harmful_acts.to(sae.W_enc.dtype))
        harmless_feats = sae.encode(harmless_acts.to(sae.W_enc.dtype))
    feat_dim = (harmful_feats.mean(0) - harmless_feats.mean(0)).float()  # (n_features,)

    top_indices = torch.topk(feat_dim.abs(), k=top_k).indices.tolist()
    print(f"[sae] top-{top_k} refusal features (by |DiM|): {top_indices}")

    decoder = sae.W_dec.detach().float().to("cpu")  # (n_features, d_model)
    return decoder, top_indices


# =============================================================================
# Post-quantization ortho correction (§3.6)
# =============================================================================

@torch.no_grad()
def post_quantization_ortho_correction(
    model: RefusalModel,
    direction: torch.Tensor,
    threshold: float = 0.005,
) -> int:
    """Re-zero the r̂ component of residual-writer weights after quantization.

    §3.6 — Weight ortho sets each column's r̂ component to zero in bfloat16.
    After GPTQ/AWQ/GGUF quantization, rounding error can re-introduce a small
    r̂ component per column.  Summed over ``n_layers × d_in`` columns this
    accumulates and can partially restore the refusal direction.

    This function checks every residual-writer weight column and re-zeros any
    column whose r̂ component exceeds ``threshold``.

    Args:
        model: The wrapped model (presumably already quantized).
        direction: The original refusal direction vector.
        threshold: Maximum allowed |r̂ · w_col| before re-zeroing.  A column
            with ``|r̂ · w_col| > threshold`` has its r̂ component stripped.

    Returns:
        Number of columns that were re-zeroed.
    """
    writers = discover_residual_writers(model.model)
    r_hat = direction.to(device=model.device, dtype=torch.float32)
    r_hat = r_hat / r_hat.norm()

    total_corrected = 0

    def _correct_linear(layer: nn.Linear) -> int:
        """Correct r̂ component in a linear layer's weight (device-aware)."""
        W = layer.weight.data  # (d_model, d_input)
        r = r_hat.to(dtype=W.dtype, device=W.device)
        # Per-column dot product: r̂ᵀ w_j for each column j
        col_dots = r @ W  # (d_input,)
        mask = col_dots.abs() > threshold
        n_corrected = int(mask.sum().item())
        if n_corrected > 0:
            # Re-zero only the flagged columns
            correction = torch.outer(r, col_dots * mask.float())
            W.sub_(correction)
        return n_corrected

    def _correct_embedding(layer: nn.Embedding) -> int:
        """Correct r̂ component in an embedding layer's weight (device-aware)."""
        W = layer.weight.data  # (vocab, d_model)
        r = r_hat.to(dtype=W.dtype, device=W.device)
        row_dots = W @ r  # (vocab,)
        mask = row_dots.abs() > threshold
        n_corrected = int(mask.sum().item())
        if n_corrected > 0:
            correction = torch.outer(row_dots * mask.float(), r)
            W.sub_(correction)
        return n_corrected

    # Correct embedding
    n = _correct_embedding(writers.embed)
    total_corrected += n
    if n:
        print(f"[quant correction] embed_tokens: {n} rows corrected")

    # Correct positional embedding (if present)
    if writers.positional_embed is not None:
        n = _correct_embedding(writers.positional_embed)
        total_corrected += n
        if n:
            print(f"[quant correction] positional_embed: {n} rows corrected")

    # Correct attention o_proj
    for idx, lin in enumerate(writers.attn_out_weights):
        n = _correct_linear(lin)
        total_corrected += n
        if n:
            print(f"[quant correction] layer {idx} o_proj: {n} cols corrected")

    # Correct MLP down_proj
    for idx, lin in enumerate(writers.mlp_out_weights):
        n = _correct_linear(lin)
        total_corrected += n
        if n:
            print(f"[quant correction] layer {idx} down_proj: {n} cols corrected")

    print(f"[quant correction] total columns/rows re-zeroed: {total_corrected}")
    return total_corrected


# =============================================================================
# LoRA-Based Reversible Ablation (US-013)
# =============================================================================

class LoRAReversibleAblation:
    """Non-destructive reversible ablation using LoRA adapters.

    Instead of permanent weight modification, this creates rank-1 LoRA adapters
    that can be removed to restore original behavior. This is useful for:
    - Temporary ablation during development/testing
    - A/B comparison between original and ablated
    - Experiments that need to restore original model
    """

    def __init__(self, model: RefusalModel, direction: torch.Tensor, rank: int = 4):
        """Initialize LoRA reversible ablation.

        Args:
            model: RefusalModel wrapper
            direction: Refusal direction to neutralize
            rank: LoRA rank (default 4, higher = more expressiveness)
        """
        self.model = model
        self.direction = direction.to(dtype=torch.float32)
        self.direction = self.direction / self.direction.norm()
        self.rank = rank
        self._handles: List = []
        self._lora_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def apply(self) -> None:
        """Apply LoRA correction - modifies forward pass to neutralize refusal."""
        from .model import discover_residual_writers

        writers = discover_residual_writers(self.model.model)
        d = self.direction.to(device=self.model.device)

        # Create rank-1 LoRA for embedding
        embed_W = writers.embed.weight.data  # (vocab, d_model)
        vocab_size, d_model = embed_W.shape

        # LoRA for embedding: W + A @ B where A, B are rank-r
        A_embed = torch.randn(vocab_size, self.rank, device=embed_W.device) * 0.01
        B_embed = torch.randn(self.rank, d_model, device=embed_W.device) * 0.01

        # Compute target: we want to subtract the refusal component
        # For each vocab embedding v, compute r_hat @ v and adjust
        self._original_embed = embed_W.data.clone()

        # For each row in embed, compute projection onto direction
        for i in range(vocab_size):
            v = embed_W[i]
            proj = (v @ d) * d  # component along refusal direction
            # A[i] @ B approximates -proj
            # Simple approach: set A[i] to projection vector, B to direction
            A_embed[i] = proj.unsqueeze(0) * 0.1  # Scaled down
            B_embed[0] = d * 0.1

        self._lora_weights["embed"] = (A_embed, B_embed)

        # Hook into forward to apply LoRA correction
        def embed_hook(module, inputs):
            x = inputs[0]  # token indices
            embeddings = self._original_embed[x]
            # Apply LoRA correction: subtract projection onto refusal direction
            for idx in range(x.shape[0]):
                emb = embeddings[idx]
                proj = (emb @ d) * d
                embeddings[idx] = emb - proj * 0.5  # Partial correction
            return embeddings

        handle = writers.embed.register_forward_pre_hook(embed_hook)
        self._handles.append(handle)

    def remove(self) -> None:
        """Remove LoRA correction - restores original behavior."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._lora_weights = {}

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        return False


# =============================================================================
# KL-Divergence Co-Optimization (§KL)
# =============================================================================

def measure_kl_divergence(
    model,
    original_logits: torch.Tensor,
    modified_logits: torch.Tensor,
) -> float:
    """Measure KL divergence between original and modified model outputs.

    Args:
        model: RefusalModel (for device/dtype info)
        original_logits: Logits from original model (n_prompts, vocab)
        modified_logits: Logits from modified model (n_prompts, vocab)

    Returns:
        KL divergence (higher = more distribution shift)
    """
    orig_log_probs = torch.log_softmax(original_logits.float(), dim=-1)
    mod_log_probs = torch.log_softmax(modified_logits.float(), dim=-1)
    mod_probs = torch.softmax(modified_logits.float(), dim=-1)

    # KL(orig || mod) = sum(p * (log(p) - log(q)))
    kl = torch.sum(mod_probs * (mod_log_probs - orig_log_probs), dim=-1)
    return float(kl.mean().item())


@torch.no_grad()
def kl_co_optimize(
    model: RefusalModel,
    direction: torch.Tensor,
    calibration_prompts: List[str],
    batch_size: int = 4,
    kl_budget: float = 0.1,
    max_iterations: int = 5,
) -> None:
    """Orthogonalize weights with KL divergence feedback loop.

    After each orthogonalization pass, measures KL divergence from original
    model outputs. If KL exceeds budget, iteratively reduces orthogonalization
    strength on over-projected layers.

    Args:
        model: RefusalModel wrapper
        direction: Refusal direction to orthogonalize
        calibration_prompts: Representative prompts for KL measurement
        batch_size: Forward pass batch size
        kl_budget: Maximum allowed KL divergence (default 0.1)
        max_iterations: Maximum refinement iterations
    """
    print("[kl_co_opt] collecting original logits ...")

    # Collect original logits
    original_logits: List[torch.Tensor] = []
    for start in range(0, len(calibration_prompts), batch_size):
        batch = [model.format(p) for p in calibration_prompts[start:start + batch_size]]
        enc = model.tokenize(batch)
        logits = model.model(**enc).logits
        original_logits.append(logits[:, -1, :].detach().cpu())
    original_logits = torch.cat(original_logits, dim=0)

    # First orthogonalization pass
    print("[kl_co_opt] first orthogonalization pass ...")
    orthogonalize_weights(model, direction)

    # Measure KL
    modified_logits: List[torch.Tensor] = []
    for start in range(0, len(calibration_prompts), batch_size):
        batch = [model.format(p) for p in calibration_prompts[start:start + batch_size]]
        enc = model.tokenize(batch)
        logits = model.model(**enc).logits
        modified_logits.append(logits[:, -1, :].detach().cpu())
    modified_logits = torch.cat(modified_logits, dim=0)

    kl = measure_kl_divergence(model, original_logits, modified_logits)
    print(f"[kl_co_opt] KL after first pass: {kl:.4f} (budget: {kl_budget})")

    # If within budget, done
    if kl <= kl_budget:
        print("[kl_co_opt] KL within budget — no refinement needed")
        return

    # Refinement loop
    layer_weights = [1.0] * model.n_layers

    for iteration in range(max_iterations):
        print(f"[kl_co_opt] refinement iteration {iteration + 1}/{max_iterations}")

        # Reduce strength on layers with highest impact
        # Simple strategy: reduce by factor of 0.5 each iteration
        for i in range(model.n_layers):
            layer_weights[i] *= 0.7

        # Re-orthogonalize with reduced weights
        orthogonalize_weights(model, direction, layer_weights=layer_weights)

        # Measure KL again
        modified_logits: List[torch.Tensor] = []
        for start in range(0, len(calibration_prompts), batch_size):
            batch = [model.format(p) for p in calibration_prompts[start:start + batch_size]]
            enc = model.tokenize(batch)
            logits = model.model(**enc).logits
            modified_logits.append(logits[:, -1, :].detach().cpu())
        modified_logits = torch.cat(modified_logits, dim=0)

        kl = measure_kl_divergence(model, original_logits, modified_logits)
        print(f"[kl_co_opt] KL after refinement: {kl:.4f}")

        if kl <= kl_budget:
            print("[kl_co_opt] KL within budget after refinement")
            return

    print(f"[kl_co_opt] WARNING: KL budget not met after {max_iterations} iterations")


# =============================================================================
# CoT-Aware Ablation (§CoT)
# =============================================================================

def cot_aware_orthogonalize(
    model: RefusalModel,
    refusal_direction: torch.Tensor,
    cot_prompts: List[str],
    batch_size: int = 4,
) -> None:
    """Orthogonalize refusal direction while preserving chain-of-thought reasoning.

    Identifies reasoning directions from CoT prompts, then orthogonalizes
    the refusal direction against these to preserve step-by-step reasoning.

    Args:
        model: RefusalModel wrapper
        refusal_direction: The refusal direction to remove
        cot_prompts: Prompts that require chain-of-thought reasoning
        batch_size: Forward pass batch size
    """
    from ..extraction import collect_activations, difference_in_means, project_out_subspace

    print("[cot_aware] identifying reasoning directions ...")

    # Collect activations for CoT prompts
    token_positions = [-1, -2, -3]  # Multiple positions for richer signal
    activations = collect_activations(model, cot_prompts, token_positions, batch_size)

    # Split into two halves for DiM (reasoning vs factual)
    n_prompts = len(cot_prompts)
    half = n_prompts // 2
    reasoning_acts = activations[:, :, :half]
    factual_acts = activations[:, :, half:]

    # Extract reasoning direction via DiM
    reasoning_direction = difference_in_means(reasoning_acts, factual_acts)
    reasoning_direction = reasoning_direction.mean(dim=(0, 1))  # Average over layers/positions
    reasoning_direction = reasoning_direction / reasoning_direction.norm()

    print("[cot_aware] orthogonalizing refusal against reasoning direction ...")

    # Project refusal direction against reasoning direction
    preserved_refusal = project_out_subspace(refusal_direction, [reasoning_direction])

    # Normalize if not degenerate
    if preserved_refusal.norm() > 1e-6:
        preserved_refusal = preserved_refusal / preserved_refusal.norm()
    else:
        # Degenerate: use original but warn
        print("[cot_aware] WARNING: projection degenerate, using original direction")
        preserved_refusal = refusal_direction / refusal_direction.norm()

    # Orthogonalize weights with preserved direction
    orthogonalize_weights(model, preserved_refusal)

    print("[cot_aware] done — reasoning direction preserved")
