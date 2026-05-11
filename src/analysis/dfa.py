"""
Direct Feature Attribution (DFA) for refusal direction analysis.

DFA measures each component's direct contribution to the refusal direction
by projecting its output onto r̂ (the refusal direction unit vector).

For attention head h at layer l:
    DFA(h, l) = mean over prompts of [ (head_output_h @ r̂) ]

For MLP at layer l:
    DFA(mlp, l) = mean over prompts of [ (mlp_output @ r̂) ]

High positive DFA → component strongly writes the refusal direction.
High negative DFA → component suppresses the refusal direction.

This is more principled than activation patching: it measures the direct
causal contribution of each component to the refusal signal, not the
indirect causal effect of ablating the component.

Reference: Arditi et al. (2024) §5.2 — Direct Feature Attribution.
           Kissane et al. (2024) — DFA methodology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


# -----------------------------------------------------------------------------
# Result dataclass
# -----------------------------------------------------------------------------

@dataclass
class DFAResult:
    """DFA scores for all attention heads and MLPs across all layers."""
    # Shape: (n_layers, n_heads) — DFA of each head
    per_layer_attn: torch.Tensor
    # Shape: (n_layers,) — DFA of each MLP
    per_layer_mlp: torch.Tensor
    n_layers: int
    n_heads: int
    d_model: int
    prompts_used: int
    direction_norm: float  # Norm of the input direction (before normalization)


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _get_decoder_layers(model) -> nn.ModuleList:
    """Return the ModuleList of transformer decoder layers.

    Mirrors the logic from model.py without importing it, to avoid
    circular imports.
    """
    hf = model.model  # RefusalModel.model is the PreTrainedModel
    inner = getattr(hf, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    # GPT-2 style
    transformer = getattr(hf, "transformer", None)
    if transformer is not None and hasattr(transformer, "h"):
        return transformer.h
    # Direct layers on the HF model (some architectures)
    if hasattr(hf, "layers"):
        return hf.layers
    raise RuntimeError(
        f"Cannot locate decoder layers on {type(hf).__name__}; "
        "extend _get_decoder_layers() in dfa.py to support this architecture."
    )


# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------

def compute_dfa_scores(
    model,                          # RefusalModel
    direction: torch.Tensor,        # Refusal direction (need not be unit-norm)
    harmful_prompts: List[str],     # Prompts to measure DFA over
    batch_size: int = 4,
) -> DFAResult:
    """Compute Direct Feature Attribution scores for all components.

    For each forward pass, we:
    1. Hook each attention module to capture its output (pre-o_proj result × o_proj)
    2. Hook each MLP to capture its output (post-down_proj)
    3. Project each component's output onto r̂
    4. Average over all prompts and positions

    Implementation uses output hooks on self_attn and mlp modules.
    For attention: hook self_attn output (shape: batch, seq, d_model),
    then split into per-head contributions using o_proj.weight.
    For MLP: hook mlp output directly (shape: batch, seq, d_model),
    then project onto r̂ for the aggregate MLP DFA.

    For per-head DFA: the head h contribution to the output is
    head_out_h @ o_proj.weight[:, h*head_dim:(h+1)*head_dim].T
    then project that onto r̂.

    Args:
        model: A RefusalModel instance.
        direction: The refusal direction vector of shape (d_model,).
            Need not be unit-norm; the norm is recorded in the result.
        harmful_prompts: List of harmful instruction strings.
        batch_size: Number of prompts per forward pass.

    Returns:
        DFAResult with per-layer attention and MLP DFA scores.
    """
    direction_norm = float(direction.norm().item())

    # Work in float32 throughout for numerical stability.
    r_hat = direction.to(device=model.device, dtype=torch.float32)
    r_hat = r_hat / r_hat.norm()  # (d_model,)

    layers = _get_decoder_layers(model)
    n_layers = len(layers)

    config = model.model.config
    n_heads: int = config.num_attention_heads
    head_dim: int = model.d_model // n_heads

    # Accumulators: sum of DFA contributions across all (prompt, position) pairs.
    # We will divide by the total token count at the end.
    attn_acc = torch.zeros(n_layers, n_heads, dtype=torch.float64)
    mlp_acc = torch.zeros(n_layers, dtype=torch.float64)
    token_count: int = 0  # total number of (prompt × position) pairs accumulated

    # Storage for hook captures, keyed by layer index.
    # Each forward pass populates these before we read them.
    attn_captures: Dict[int, torch.Tensor] = {}
    mlp_captures: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Build hooks.  We register output hooks on each self_attn and mlp
    # sub-module so we capture the post-projection output written to the
    # residual stream.
    # ------------------------------------------------------------------
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_attn_hook(layer_idx: int):
        """Capture the attention module's output (batch, seq, d_model)."""
        def hook(_module, _inputs, output):
            # HuggingFace attention modules return a tuple:
            # (attn_output, attn_weights_or_None, past_key_value_or_None, ...)
            # The first element is the residual-stream contribution.
            out = output[0] if isinstance(output, tuple) else output
            attn_captures[layer_idx] = out.detach().to(dtype=torch.float32)
        return hook

    def _make_mlp_hook(layer_idx: int):
        """Capture the MLP module's output (batch, seq, d_model)."""
        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            mlp_captures[layer_idx] = out.detach().to(dtype=torch.float32)
        return hook

    # Cache o_proj weight slices per layer to avoid re-fetching inside the loop.
    o_proj_weights: List[Optional[torch.Tensor]] = []

    for layer_idx, layer in enumerate(layers):
        attn = (
            getattr(layer, "self_attn", None)
            or getattr(layer, "attention", None)
        )
        mlp = (
            getattr(layer, "mlp", None)
            or getattr(layer, "feed_forward", None)
        )

        if attn is not None:
            handles.append(attn.register_forward_hook(_make_attn_hook(layer_idx)))
            # Grab o_proj weight once (shape: d_model × (n_heads * head_dim)).
            o_proj = (
                getattr(attn, "o_proj", None)
                or getattr(attn, "out_proj", None)
            )
            if isinstance(o_proj, nn.Linear):
                # Detach and cast to float32 for DFA arithmetic.
                o_proj_weights.append(
                    o_proj.weight.detach().to(dtype=torch.float32)
                )
            else:
                o_proj_weights.append(None)
        else:
            o_proj_weights.append(None)

        if mlp is not None:
            handles.append(mlp.register_forward_hook(_make_mlp_hook(layer_idx)))

    try:
        for start in range(0, len(harmful_prompts), batch_size):
            batch_raw = harmful_prompts[start : start + batch_size]
            batch_fmt = [model.format(p) for p in batch_raw]
            enc = model.tokenize(batch_fmt)

            # seq_len varies per batch due to left-padding; capture it after.
            with torch.no_grad():
                model.model(**enc)

            # Determine number of tokens in this batch for averaging.
            # We average over all sequence positions (all token positions
            # contribute to the residual stream and carry DFA signal).
            # enc["input_ids"]: (batch, seq_len)
            seq_len = enc["input_ids"].shape[1]
            b_actual = enc["input_ids"].shape[0]
            n_tokens = b_actual * seq_len
            token_count += n_tokens

            # r_hat on CPU for fast matmul (captures are already float32).
            r_hat_cpu = r_hat.cpu()  # (d_model,)

            # ----------------------------------------------------------
            # Accumulate attention DFA per head.
            # ----------------------------------------------------------
            for layer_idx in range(n_layers):
                if layer_idx not in attn_captures:
                    continue
                attn_out = attn_captures[layer_idx].cpu()  # (b, seq, d_model)
                W_o = o_proj_weights[layer_idx]  # (d_model, n_heads * head_dim) or None

                if W_o is None:
                    # No o_proj found; skip per-head attribution for this layer.
                    continue

                W_o_cpu = W_o.cpu()  # (d_model, n_heads * head_dim)

                # attn_out is the *already-projected* output (post o_proj).
                # To recover per-head contributions we need the pre-o_proj
                # value (the concatenated head outputs).  We don't have direct
                # access to it via an output hook on self_attn, so we use the
                # linear relationship:
                #
                #   attn_out = concat_heads @ W_o^T  (+ bias, ignored for DFA)
                #
                # The contribution of head h to the residual stream is:
                #   head_contrib_h = v_h @ W_o_h^T
                # where v_h is the (batch, seq, head_dim) value-head output and
                # W_o_h = W_o[:, h*head_dim:(h+1)*head_dim]  (d_model × head_dim).
                #
                # DFA(h) = r̂ · head_contrib_h  (projected onto the direction)
                #
                # Since we only have the combined attn_out, we approximate by
                # decomposing the o_proj column blocks:
                #   attn_out ≈ Σ_h  (attn_out @ W_o_h_pinv)  @ W_o_h^T
                # But this requires solving a system.  Instead we use the direct
                # approach: the pre-projection head outputs can be recovered as
                #   v_h = attn_out @ W_o_h (W_o_h^T W_o_h)^{-1}
                # which is the pseudo-inverse projection.  For orthogonal W_o
                # column blocks this simplifies nicely, but in general we use:
                #
                #   head_contrib_h = (attn_out @ W_o_h) / (W_o_h.norm()^2) * W_o_h^T ...
                #
                # The cleanest correct decomposition without the pre-proj tensor:
                # We project attn_out onto the column space of each head's W_o
                # block, giving the component of attn_out that head h contributes.
                #
                # Concretely:
                #   a_h = attn_out @ W_o_h   (b, seq, head_dim) - coordinates in head space
                #   contrib_h = a_h @ W_o_h^T  (b, seq, d_model)
                # is NOT correct because W_o_h columns are not orthonormal in general.
                #
                # The correct attribution is simpler than it looks:
                # r̂ · attn_out = Σ_h  r̂ · (v_h @ W_o_h^T)
                # and r̂ · (v_h @ W_o_h^T) = (W_o_h @ r̂) · v_h
                # so DFA(h) = mean_over_tokens[ (W_o_h @ r̂) · v_h ]
                #
                # We still need v_h (the pre-o_proj concatenated heads).
                # Since attn_out = concat(v_0,...,v_{H-1}) @ W_o^T, and
                # if W_o has full column rank, we can recover concat_heads via
                # the pseudoinverse: concat_heads = attn_out @ W_o^{+T}
                # which equals attn_out @ W_o (W_o^T W_o)^{-1} when W_o is tall.
                #
                # For large d_model >> n_heads*head_dim, W_o is (d_model × D_v)
                # with D_v = n_heads * head_dim.  The pseudoinverse is:
                #   W_o^+ = (W_o^T W_o)^{-1} W_o^T   shape (D_v × d_model)
                # so: concat_heads = attn_out @ W_o^{+T} = attn_out @ W_o (W_o^T W_o)^{-1}
                # shape: (b, seq, d_model) @ (d_model, D_v) @ (D_v, D_v) -> (b, seq, D_v)
                #
                # This is exact when attn_out lies in col(W_o), which it does by
                # construction (no bias or exact; with bias it's an approximation
                # that removes the bias term — acceptable for DFA purposes).
                #
                # For efficiency, pre-compute r̂ projected through each head's
                # W_o columns: r_h = W_o_h^T @ r̂  (head_dim,) per head h.
                # Then DFA(h) = mean[ v_h · r_h ] over (b, seq).
                # We can compute this without inverting W_o^T W_o by
                # noting: v_h · r_h = (attn_out @ W_o^{+T})_h · (W_o_h^T r̂)
                # But this circles back to needing concat_heads.
                #
                # Pragmatic approach used here (matches Kissane et al. DFA):
                # Compute r̂^T W_o_h  (head_dim,) for each head, then
                # recover per-head pre-proj output via pseudoinverse.
                # For typical LLaMA models W_o is (4096, 4096) and computing
                # the pseudoinverse once is feasible.

                D_v = n_heads * head_dim  # total value dimension
                # W_o_cpu: (d_model, D_v)
                # Project r̂ through W_o^T to get coordinates in head-output space.
                # r_in_head_space = W_o_cpu^T @ r̂_cpu  (D_v,)
                r_in_head_space = W_o_cpu.t() @ r_hat_cpu  # (D_v,)

                # Recover concat_heads (b, seq, D_v) via pseudoinverse of W_o.
                # Use lstsq for numerical stability: solve W_o^T X^T = attn_out^T
                # i.e., for each token vector t (d_model,), find x (D_v,) s.t. W_o x ≈ t.
                # Equivalently: concat_heads = attn_out @ pinv(W_o^T)
                # pinv(W_o^T) = (W_o W_o^T)^+ W_o = W_o^+ shape (D_v, d_model)
                # We solve: (b*seq, d_model) @ (d_model, D_v) weighted by pseudoinverse.
                #
                # Efficient: concat_heads ≈ attn_out @ W_o @ (W_o^T @ W_o)^{-1}
                # using torch.linalg.lstsq with small D_v system.
                b_seq = b_actual * seq_len
                attn_flat = attn_out.reshape(b_seq, model.d_model)  # (b*seq, d_model)

                # Solve W_o @ x = attn_flat^T for x  →  x has shape (D_v, b*seq)
                # Equivalently in the transposed form: attn_flat @ W_o^{+} (b*seq, D_v)
                # Use lstsq: min ||attn_flat - concat_heads @ W_o^T||  → concat_heads = attn_flat @ W_o^{+T}
                # W_o^{+T} = (W_o^T)^+ = W_o (W_o^T W_o)^{-1}   (d_model, D_v) → (d_model, D_v)
                # So concat_heads = attn_flat @ W_o (W_o^T W_o)^{-1}  (b*seq, D_v)
                #
                # Skip full lstsq — use the closed form for the typical square case (D_v == d_model):
                if D_v == model.d_model:
                    # Square W_o: directly solve.
                    # concat_heads = attn_flat @ inv(W_o^T)
                    try:
                        concat_heads = torch.linalg.solve(W_o_cpu.t(), attn_flat.t()).t()  # (b*seq, D_v)
                    except Exception:
                        # Fallback: use lstsq
                        concat_heads = torch.linalg.lstsq(W_o_cpu, attn_flat.t()).solution.t()
                else:
                    # Non-square: use lstsq.
                    concat_heads = torch.linalg.lstsq(W_o_cpu, attn_flat.t()).solution.t()  # (b*seq, D_v)

                # concat_heads: (b*seq, D_v) = (b*seq, n_heads * head_dim)
                # r_in_head_space: (D_v,) = (n_heads * head_dim,)
                # Per-head DFA = mean over tokens of dot(v_h, r_h)
                for h in range(n_heads):
                    v_h = concat_heads[:, h * head_dim : (h + 1) * head_dim]  # (b*seq, head_dim)
                    r_h = r_in_head_space[h * head_dim : (h + 1) * head_dim]  # (head_dim,)
                    head_dfa = (v_h @ r_h).sum().item()  # scalar (sum, divide by token_count at end)
                    attn_acc[layer_idx, h] += head_dfa

            # ----------------------------------------------------------
            # Accumulate MLP DFA.
            # ----------------------------------------------------------
            for layer_idx in range(n_layers):
                if layer_idx not in mlp_captures:
                    continue
                mlp_out = mlp_captures[layer_idx].cpu()  # (b, seq, d_model)
                mlp_flat = mlp_out.reshape(b_actual * seq_len, model.d_model)  # (b*seq, d_model)
                # DFA = r̂ · mlp_out, summed over tokens.
                mlp_dfa = (mlp_flat @ r_hat_cpu).sum().item()
                mlp_acc[layer_idx] += mlp_dfa

            attn_captures.clear()
            mlp_captures.clear()

    finally:
        for h in handles:
            h.remove()

    # Normalize by total token count.
    if token_count > 0:
        attn_scores = (attn_acc / token_count).float()
        mlp_scores = (mlp_acc / token_count).float()
    else:
        attn_scores = attn_acc.float()
        mlp_scores = mlp_acc.float()

    return DFAResult(
        per_layer_attn=attn_scores,
        per_layer_mlp=mlp_scores,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=model.d_model,
        prompts_used=len(harmful_prompts),
        direction_norm=direction_norm,
    )


# -----------------------------------------------------------------------------
# Top-k components
# -----------------------------------------------------------------------------

def top_dfa_components(
    result: DFAResult,
    top_k: int = 8,
    component_type: str = "all",  # "attn", "mlp", or "all"
) -> List[Dict]:
    """Return the top-k components by absolute DFA score.

    Returns a list of dicts, each with:
    - 'type': 'attn' or 'mlp'
    - 'layer': layer index
    - 'head': head index (None for MLP)
    - 'dfa_score': the raw DFA value
    - 'abs_dfa': abs(dfa_score)
    - 'rank': 1-indexed rank

    Args:
        result: A DFAResult returned by compute_dfa_scores.
        top_k: Number of top components to return.
        component_type: Which components to include — "attn", "mlp", or "all".

    Returns:
        List of dicts sorted by abs_dfa descending.
    """
    entries: List[Dict] = []

    if component_type in ("attn", "all"):
        for layer_idx in range(result.n_layers):
            for head_idx in range(result.n_heads):
                score = float(result.per_layer_attn[layer_idx, head_idx].item())
                entries.append({
                    "type": "attn",
                    "layer": layer_idx,
                    "head": head_idx,
                    "dfa_score": score,
                    "abs_dfa": abs(score),
                })

    if component_type in ("mlp", "all"):
        for layer_idx in range(result.n_layers):
            score = float(result.per_layer_mlp[layer_idx].item())
            entries.append({
                "type": "mlp",
                "layer": layer_idx,
                "head": None,
                "dfa_score": score,
                "abs_dfa": abs(score),
            })

    entries.sort(key=lambda e: e["abs_dfa"], reverse=True)
    top = entries[:top_k]

    for rank, entry in enumerate(top, start=1):
        entry["rank"] = rank

    return top


# -----------------------------------------------------------------------------
# Heatmap visualization
# -----------------------------------------------------------------------------

def dfa_heatmap(
    result: DFAResult,
    title: str = "DFA Scores: Contribution to Refusal Direction",
    save_path: Optional[str] = None,
) -> None:
    """Plot a heatmap of attention head DFA scores (layers × heads).

    Also overlays MLP DFA as a separate bar chart below.
    Uses matplotlib. Gracefully skips if not installed.
    Positive DFA → red (writing refusal), negative → blue (suppressing).

    Args:
        result: A DFAResult returned by compute_dfa_scores.
        title: Title for the figure.
        save_path: If provided, save the figure to this path; otherwise show
            interactively. None by default.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print(
            "[dfa_heatmap] matplotlib is not installed. "
            "Install it with `pip install matplotlib` to generate heatmaps."
        )
        return

    attn = result.per_layer_attn.numpy()  # (n_layers, n_heads)
    mlp = result.per_layer_mlp.numpy()    # (n_layers,)

    fig, (ax_heat, ax_bar) = plt.subplots(
        2, 1,
        figsize=(max(10, result.n_heads * 0.5), result.n_layers * 0.4 + 3),
        gridspec_kw={"height_ratios": [4, 1]},
    )

    # Symmetric color scale centered at zero.
    vmax = max(abs(attn).max(), 1e-6)
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    im = ax_heat.imshow(
        attn,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax_heat.set_title(title, fontsize=12, pad=8)
    ax_heat.set_xlabel("Attention Head")
    ax_heat.set_ylabel("Layer")
    ax_heat.set_xticks(range(result.n_heads))
    ax_heat.set_yticks(range(result.n_layers))
    fig.colorbar(im, ax=ax_heat, label="DFA score", shrink=0.8)

    # MLP bar chart.
    bar_colors = [
        "#d73027" if v > 0 else "#4575b4"
        for v in mlp
    ]
    ax_bar.bar(range(result.n_layers), mlp, color=bar_colors, edgecolor="none")
    ax_bar.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax_bar.set_xlabel("Layer")
    ax_bar.set_ylabel("MLP DFA")
    ax_bar.set_xticks(range(result.n_layers))
    ax_bar.set_title("MLP DFA per Layer", fontsize=10)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[dfa_heatmap] saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# -----------------------------------------------------------------------------
# Head selection for surgical ablation
# -----------------------------------------------------------------------------

def select_refusal_heads_by_dfa(
    result: DFAResult,
    top_k: int = 5,
    min_dfa_percentile: float = 90.0,
) -> List[Tuple[int, int]]:
    """Return (layer, head) pairs for the top refusal heads by DFA.

    Used by the surgical preset to target only high-DFA heads for ablation,
    rather than the full direction-based orthogonalization.

    Returns list of (layer_idx, head_idx) sorted by DFA score descending.

    Args:
        result: A DFAResult returned by compute_dfa_scores.
        top_k: Maximum number of heads to return.
        min_dfa_percentile: Only include heads whose DFA score is at or above
            this percentile of all attention head DFA scores. Default 90.0.

    Returns:
        List of (layer_idx, head_idx) tuples sorted by DFA descending.
    """
    attn = result.per_layer_attn  # (n_layers, n_heads)

    # Flatten to compute percentile threshold.
    flat = attn.reshape(-1)  # (n_layers * n_heads,)
    threshold = float(torch.quantile(flat, min_dfa_percentile / 100.0).item())

    # Collect all (layer, head) pairs above the threshold.
    candidates: List[Tuple[int, int, float]] = []
    for layer_idx in range(result.n_layers):
        for head_idx in range(result.n_heads):
            score = float(attn[layer_idx, head_idx].item())
            if score >= threshold:
                candidates.append((layer_idx, head_idx, score))

    # Sort by DFA descending and take top_k.
    candidates.sort(key=lambda x: x[2], reverse=True)
    return [(layer_idx, head_idx) for layer_idx, head_idx, _ in candidates[:top_k]]
