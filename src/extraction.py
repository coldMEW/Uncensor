"""
Direction extraction, selection, and analysis.

Paper: https://arxiv.org/abs/2406.11717
§2.3 — "Extracting a refusal direction" (difference-in-means).
§C.1 — direction selection algorithm with bypass / induce / kl scores.
§5.1 — iterative subspace deflation for multi-direction extraction.
§5.2 — PCA-based subspace extraction (alternative to iterative).
§5.3 — activation patching for causal layer selection.
§5.4 — attention head scoring for refusal-head identification.
§2.1, §7.2 — explained variance analysis for subspace dimensionality.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm

from .metrics import refusal_metric_from_logits
from .model import RefusalModel


# =============================================================================
# Private helpers — local reimplementation to avoid circular imports
# =============================================================================
def _get_logits_at_last_token(
    model: RefusalModel,
    prompts: List[str],
    batch_size: int,
) -> torch.Tensor:
    """Return the logits at the last token for each prompt, shape ``(n, vocab)``.

    Any hook intervention active via a surrounding context manager applies
    to this call.  This is a local reimplementation of the same helper in
    ``pipeline.py`` so that this module can be used without triggering a
    circular import.
    """
    out: List[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        batch = [model.format(p) for p in prompts[start : start + batch_size]]
        enc = model.tokenize(batch)
        with torch.no_grad():
            logits = model.model(**enc).logits  # (b, seq, vocab)
        # Left padding ⇒ the last column is always the final content token.
        out.append(logits[:, -1, :].detach().cpu())
    return torch.cat(out, dim=0)


def _compute_mean_refusal_metric(
    logits: torch.Tensor,
    refusal_token_ids: Sequence[int],
) -> float:
    """Compute mean refusal metric (log-odds over refusal tokens).

    Wrapper around :func:`metrics.refusal_metric_from_logits` that returns
    a single scalar ``float``.
    """
    metric = refusal_metric_from_logits(logits, refusal_token_ids)  # (n,)
    return float(metric.mean().item())


# =============================================================================
# Activation capture (§2.3)
# =============================================================================
def collect_activations(
    model: RefusalModel,
    prompts: Sequence[str],
    token_positions: Sequence[int],
    batch_size: int,
) -> torch.Tensor:
    """Collect residual-stream activations at the given post-instruction token
    positions, for every layer.

    §2.3 — activations are captured per layer *l* and per post-instruction token
    position *i* ∈ I. We use a forward-pre-hook on each decoder layer to grab
    the residual stream going into that layer (which equals the output of the
    previous layer / of the embedding for layer 0).

    Returns a tensor of shape ``(n_layers, len(token_positions), n_prompts, d_model)``.
    """
    n_prompts = len(prompts)
    # Pre-allocate on CPU to keep GPU memory light for large models.
    out = torch.zeros(
        (model.n_layers, len(token_positions), n_prompts, model.d_model),
        dtype=torch.float32,
    )

    # Buffer that the hook writes into, one slot per layer. Re-used each batch.
    buffer: Dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(_module, inputs):
            # inputs is a tuple; the first element is the residual stream.
            # Shape: (batch, seq_len, d_model)
            buffer[layer_idx] = inputs[0].detach()
            return None  # don't modify inputs
        return hook

    handles = model.register_forward_pre_hooks(make_hook)
    try:
        for start in tqdm(range(0, n_prompts, batch_size), desc="extract activations"):
            batch_prompts = [model.format(p) for p in prompts[start : start + batch_size]]
            enc = model.tokenize(batch_prompts)
            with torch.no_grad():
                model.model(**enc)
            for layer_idx in range(model.n_layers):
                # buffer[layer_idx]: (batch, seq_len, d_model)
                resid = buffer[layer_idx]
                for pi, pos in enumerate(token_positions):
                    # pos is a negative index relative to the last token.
                    # resid[:, pos, :] -> (batch, d_model)
                    out[layer_idx, pi, start : start + resid.shape[0], :] = (
                        resid[:, pos, :].to(torch.float32).cpu()
                    )
            buffer.clear()
    finally:
        RefusalModel.remove_hooks(handles)
    return out


# =============================================================================
# Difference-in-means (§2.3)
# =============================================================================
def difference_in_means(
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
) -> torch.Tensor:
    """§2.3 — r^(l)_i = μ^(l)_i − ν^(l)_i, where μ and ν are the mean
    activations for harmful and harmless prompts respectively.

    Args:
        harmful_acts, harmless_acts: ``(n_layers, n_positions, n_prompts, d_model)``

    Returns:
        candidate directions: ``(n_layers, n_positions, d_model)``
    """
    mu = harmful_acts.mean(dim=2)     # -> (n_layers, n_positions, d_model)
    nu = harmless_acts.mean(dim=2)    # -> (n_layers, n_positions, d_model)
    return mu - nu                    # -> (n_layers, n_positions, d_model)


# =============================================================================
# Cross-objective null-space projection (§6.4 — Steering Pitfalls 2603.24543)
# =============================================================================
def project_out_subspace(
    vector: torch.Tensor,
    subspace: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Project ``vector`` onto the null space of the span of ``subspace``.

    Used to decouple refusal from collateral objectives (helpfulness, honesty)
    whose difference-in-means directions share cone overlap with refusal.
    Concretely: given auxiliary direction vectors ``[r̂_honesty, r̂_helpfulness]``,
    return ``r_refusal − Σᵢ (r̂ᵢᵀ r_refusal) r̂ᵢ`` using modified Gram–Schmidt
    on the auxiliary basis so the result is orthogonal to every auxiliary
    direction regardless of their mutual orthogonality.

    This does *not* modify ``vector`` in-place and does *not* normalize the
    result — the caller is free to re-normalize before using it for ablation.

    Args:
        vector: the direction to decouple, shape ``(d_model,)``.
        subspace: sequence of auxiliary direction tensors, each ``(d_model,)``.
            Need not be unit-norm; zero-norm entries are silently skipped.

    Returns:
        ``vector`` with every component along the orthonormalized auxiliary
        basis removed. Shape ``(d_model,)``; dtype/device match ``vector``.
    """
    if not subspace:
        return vector.clone()

    # Promote to float32 for numerical stability of the Gram–Schmidt pass,
    # then cast back at the end.
    original_dtype = vector.dtype
    v = vector.to(dtype=torch.float32)

    # Build an orthonormal basis of the auxiliary subspace via modified
    # Gram–Schmidt. Silently drop vectors that collapse to zero after
    # orthogonalization (linearly dependent or numerically null).
    basis: List[torch.Tensor] = []
    for aux in subspace:
        b = aux.to(device=v.device, dtype=torch.float32)
        for q in basis:
            b = b - (q @ b) * q
        norm = float(b.norm().item())
        if norm > 1e-8:
            basis.append(b / norm)

    # Project the vector out of the assembled subspace.
    out = v.clone()
    for q in basis:
        out = out - (q @ out) * q

    return out.to(dtype=original_dtype)


# =============================================================================
# Direction selection (§C.1)
# =============================================================================
@dataclass
class DirectionCandidate:
    """One entry from the |I| × L candidate grid with its three scores."""
    layer: int
    position: int     # negative index relative to the last prompt token
    vector: torch.Tensor  # the un-normalized difference-in-means vector r^(l)_i
    bypass_score: float
    induce_score: float
    kl_score: float


def select_best(
    candidates: List[DirectionCandidate],
    *,
    n_layers_total: int,
    induce_score_min: float,
    kl_score_max: float,
    max_layer_frac: float,
) -> DirectionCandidate:
    """§C.1 — "select r^(l*)_(i*) to be the direction with minimum bypass_score,
    subject to induce_score > 0, kl_score < 0.1, l < 0.8L".
    """
    layer_cutoff = int(max_layer_frac * n_layers_total)
    filtered = [
        c
        for c in candidates
        if c.induce_score > induce_score_min
        and c.kl_score < kl_score_max
        and c.layer < layer_cutoff
    ]
    if not filtered:
        raise RuntimeError(
            "No candidate direction passed the §C.1 selection filters. "
            "Try relaxing `kl_score_max` or inspecting per-layer scores."
        )
    filtered.sort(key=lambda c: c.bypass_score)
    return filtered[0]


# =============================================================================
# Multi-direction ablation context (inline, avoids circular imports)
# =============================================================================
@contextmanager
def _multi_ablation_context(
    model: RefusalModel,
    directions: Sequence[torch.Tensor],
) -> Iterator[None]:
    """Context manager that projects out *all* given directions from the
    residual stream at every layer.

    This is an inline reimplementation of the per-direction ablation logic
    from ``interventions.directional_ablation`` so that
    :func:`iterative_subspace_extraction` can ablate multiple directions
    simultaneously without triggering circular imports.

    Since the directions are already mutually orthogonal (Gram-Schmidt is
    applied during extraction), sequential single-direction projection is
    equivalent to a single subspace projection.

    When *directions* is empty the context manager is a no-op.
    """
    if not directions:
        yield
        return

    # Pre-compute unit-norm directions on the model's device.
    unit_dirs: List[torch.Tensor] = []
    for d in directions:
        r = d.to(device=model.device, dtype=torch.float32)
        r = r / r.norm()
        unit_dirs.append(r)

    def make_hook(_layer_idx: int):
        def hook(_module, inputs):
            x = inputs[0]  # (batch, seq_len, d_model)
            for r_hat in unit_dirs:
                r = r_hat.to(dtype=x.dtype)
                # x ← x − (r̂ᵀ x) r̂   (§2.4 Eq. 4)
                x = x - (x * r).sum(-1, keepdim=True) * r
            return (x,) + inputs[1:]
        return hook

    handles = model.register_forward_pre_hooks(make_hook)
    try:
        yield
    finally:
        RefusalModel.remove_hooks(handles)


# =============================================================================
# §5.1 — Iterative subspace deflation (HIGHEST IMPACT)
# =============================================================================
def iterative_subspace_extraction(
    model: RefusalModel,
    harmful_train: List[str],
    harmless_train: List[str],
    token_positions: List[int],
    batch_size: int,
    refusal_token_ids: List[int],
    harmful_val: List[str],
    harmless_val: List[str],
    cfg_extraction: dict,
    max_rounds: int = 5,
    delta_threshold: float = 0.5,
    cosine_dedup_threshold: float = 0.9,
    cumulative_kl_budget: float = 0.25,
    score_fn: Optional[Callable[..., List[DirectionCandidate]]] = None,
) -> Tuple[List[torch.Tensor], Optional[DirectionCandidate], List[Dict]]:
    """Extract multiple mutually-orthogonal refusal directions via iterative
    deflation (§5.1).

    After finding and ablating the first direction r̂₁, the full extraction
    pipeline runs again on *post-ablation* activations.  The second direction
    r̂₂ captures the next most important refusal signal — potentially revealing
    dormant backup circuits that only become dominant after the primary circuit
    is removed.

    Why iterative > static SVD:  A static SVD on the un-ablated difference
    matrix captures directions of maximum variance in the *original*
    distribution.  Backup circuits that are dormant under normal operation
    contribute negligible variance there and would be missed.  Iterative
    deflation makes dormant circuits visible by removing the dominant circuit
    that masks them.

    Args:
        model: The wrapped language model.
        harmful_train / harmless_train: Training prompts for activation
            collection (same as used in :func:`collect_activations`).
        token_positions: Post-instruction token positions (negative indices).
        batch_size: Forward-pass batch size.
        refusal_token_ids: Token IDs used to compute the refusal metric.
        harmful_val / harmless_val: Validation prompts for scoring.
        cfg_extraction: Dict with keys ``induce_score_min``,
            ``kl_score_max``, ``max_layer_frac`` (passed to
            :func:`select_best`).
        max_rounds: Maximum number of extraction rounds (default 5).
        delta_threshold: Convergence threshold in log-odds units.  If
            ``|bypass_score_k − bypass_score_{k−1}| < delta_threshold``,
            extraction stops (default 0.5).
        cosine_dedup_threshold: If ``|cos(r̂_new, r̂_existing)| > threshold``,
            the new direction is considered duplicate noise and extraction
            stops (default 0.9).
        cumulative_kl_budget: Maximum cumulative KL budget across all
            extracted directions (default 0.25, per SCANS/NSPO insight).
        score_fn: Callback with the same signature as
            :func:`pipeline.score_candidates`.  When ``None`` (default), the
            function performs a lazy import from ``.pipeline`` at call time.
            Passing the callback explicitly avoids the lazy import.

    Returns:
        Tuple ``(directions, first_round_best, round_history)``:
        * ``directions``: list of unit-norm direction tensors ``[r̂₁, r̂₂, ...]``,
          each orthogonal to all previous directions (Gram-Schmidt applied).
        * ``first_round_best``: the full :class:`DirectionCandidate` selected in
          round 1 (layer, position, bypass/induce/kl scores). ``None`` if
          no candidate passed the §C.1 filters in round 1.
        * ``round_history``: list of per-round JSON-serializable dicts
          capturing ``{round, layer, position, bypass, induce, kl,
          cumulative_kl, stop_reason}``. Enables structured post-hoc
          lineage inspection without re-running the pipeline.

    Raises:
        RuntimeError: If no candidate passes the §C.1 filters on the first
            round.
    """
    if score_fn is None:
        # Lazy import to avoid circular dependency at module level.
        # By the time this function is called, both modules are fully loaded.
        from .pipeline import score_candidates as score_fn  # type: ignore[assignment]

    directions: List[torch.Tensor] = []
    first_round_best: Optional[DirectionCandidate] = None
    prev_bypass: Optional[float] = None
    cumulative_kl: float = 0.0  # Track cumulative KL for multi-direction budget
    round_history: List[Dict] = []

    for round_idx in range(max_rounds):
        # ------------------------------------------------------------------
        # 1. Collect activations with all previous directions ablated.
        # ------------------------------------------------------------------
        with _multi_ablation_context(model, directions):
            harmful_acts = collect_activations(
                model, harmful_train, token_positions, batch_size,
            )
            harmless_acts = collect_activations(
                model, harmless_train, token_positions, batch_size,
            )

        # ------------------------------------------------------------------
        # 2. Standard DiM on the ablated activations.
        # ------------------------------------------------------------------
        candidates_tensor = difference_in_means(harmful_acts, harmless_acts)

        # ------------------------------------------------------------------
        # 3. Score candidates under the SAME multi-ablation context.
        #    Inside score_fn, per-candidate directional_ablation hooks will
        #    layer on top of our multi-ablation hooks (registered first, so
        #    they fire first → correct ordering).
        # ------------------------------------------------------------------
        with _multi_ablation_context(model, directions):
            scored = score_fn(
                model,
                candidates=candidates_tensor,
                token_positions=token_positions,
                harmful_val=harmful_val,
                harmless_val=harmless_val,
                refusal_token_ids=refusal_token_ids,
                batch_size=batch_size,
            )

        # ------------------------------------------------------------------
        # 4. Select best candidate via §C.1 filters.
        # ------------------------------------------------------------------
        try:
            best = select_best(
                scored,
                n_layers_total=model.n_layers,
                induce_score_min=cfg_extraction["induce_score_min"],
                kl_score_max=cfg_extraction["kl_score_max"],
                max_layer_frac=cfg_extraction["max_layer_frac"],
            )
        except RuntimeError as exc:
            print(
                f"[iterative round {round_idx + 1}] no valid candidates "
                f"passed §C.1 filters: {exc}"
            )
            round_history.append({
                "round": round_idx + 1,
                "stop_reason": f"no_valid_candidates: {exc}",
            })
            break

        print(
            f"[iterative round {round_idx + 1}] layer={best.layer}  "
            f"bypass={best.bypass_score:.3f}  "
            f"induce={best.induce_score:.3f}  "
            f"kl={best.kl_score:.3f}"
        )

        # Capture round-1 candidate so the caller can report per-direction
        # scores instead of hardcoding zeros.
        if round_idx == 0:
            first_round_best = best

        round_record: Dict = {
            "round": round_idx + 1,
            "layer": int(best.layer),
            "position": int(best.position),
            "bypass": float(best.bypass_score),
            "induce": float(best.induce_score),
            "kl": float(best.kl_score),
        }

        # ------------------------------------------------------------------
        # 5. Cumulative KL budget check (NSPO/SCANS insight).
        #    Multi-direction ablation accumulates KL divergence.
        #    If cumulative KL exceeds budget, further directions harm
        #    capability more than they help with refusal removal.
        # ------------------------------------------------------------------
        cumulative_kl += best.kl_score
        round_record["cumulative_kl"] = float(cumulative_kl)
        if cumulative_kl > cumulative_kl_budget:
            print(
                f"[iterative] cumulative KL = {cumulative_kl:.3f} > "
                f"{cumulative_kl_budget:.3f} (budget exceeded). "
                "Stopping to preserve capability."
            )
            round_record["stop_reason"] = "cumulative_kl_budget_exceeded"
            round_history.append(round_record)
            break
        print(
            f"[iterative] cumulative KL = {cumulative_kl:.3f} / "
            f"{cumulative_kl_budget:.3f}"
        )

        # ------------------------------------------------------------------
        # 6. Convergence check.
        # ------------------------------------------------------------------
        if prev_bypass is not None:
            if abs(best.bypass_score - prev_bypass) < delta_threshold:
                print(
                    f"[iterative] converged at round {round_idx + 1}  "
                    f"(Δ = {abs(best.bypass_score - prev_bypass):.4f}  "
                    f"< {delta_threshold})"
                )
                round_record["stop_reason"] = "convergence"
                round_history.append(round_record)
                break
        prev_bypass = best.bypass_score

        # ------------------------------------------------------------------
        # 6. Gram-Schmidt orthogonalization + cosine dedup check.
        # ------------------------------------------------------------------
        new_dir = best.vector.to(dtype=torch.float32)
        new_dir = new_dir / new_dir.norm()

        for existing in directions:
            cos_sim = float(torch.dot(new_dir, existing).item())
            if abs(cos_sim) > cosine_dedup_threshold:
                print(
                    f"[iterative] round {round_idx + 1} direction duplicates "
                    f"an existing direction (cos = {cos_sim:.4f}).  Stopping."
                )
                round_record["stop_reason"] = (
                    f"cosine_dedup({cos_sim:.4f})"
                )
                round_history.append(round_record)
                return directions, first_round_best, round_history
            # Gram-Schmidt step: remove the component along the existing dir.
            new_dir = new_dir - cos_sim * existing
            norm_after = new_dir.norm().item()
            if norm_after < 1e-6:
                print(
                    f"[iterative] near-zero direction after Gram-Schmidt "
                    f"(norm = {norm_after:.2e}).  Stopping."
                )
                round_record["stop_reason"] = "gram_schmidt_collapse"
                round_history.append(round_record)
                return directions, first_round_best, round_history
            new_dir = new_dir / new_dir.norm()

        directions.append(new_dir)
        round_record.setdefault("stop_reason", "accepted")
        round_history.append(round_record)

    return directions, first_round_best, round_history


# =============================================================================
# §5.2 — PCA-based subspace extraction
# =============================================================================
def pca_subspace_extraction(
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
    layer_idx: int,
    pos_idx: int,
    k: int = 5,
) -> torch.Tensor:
    """Extract top-*k* orthonormal directions at a given (layer, position)
    via SVD on the centered difference matrix (§5.2).

    Unlike iterative deflation (§5.1), this is a **single-pass** method that
    captures the top-*k* directions of maximum variance in the *original*
    activation distribution.  It cannot reveal dormant backup circuits (those
    contribute negligible variance to the un-ablated distribution), but it
    efficiently fills in multi-modal harmful activation structure.

    When to use:
    - Iterative deflation converges in round 1 (no backup circuits), but
      bypass rate is still non-zero → single-direction approximation loses
      variance.
    - The harmful activation distribution is multi-modal.

    Args:
        harmful_acts: ``(n_layers, n_positions, n_harmful, d_model)``
        harmless_acts: ``(n_layers, n_positions, n_harmless, d_model)``
        layer_idx: Which transformer layer to analyse.
        pos_idx: Which token-position index (into ``token_positions`` list).
        k: Number of directions to extract (default 5).

    Returns:
        ``(k, d_model)`` orthonormal directions, sorted by explained variance
        (descending).  Each direction is oriented to point from harmless
        toward harmful (positive dot product with the DiM direction).
    """
    H = harmful_acts[layer_idx, pos_idx].float()   # (n_h, d_model)
    N = harmless_acts[layer_idx, pos_idx].float()  # (n_n, d_model)

    # Center each group around its own mean.
    H_c = H - H.mean(dim=0, keepdim=True)  # (n_h, d_model)
    N_c = N - N.mean(dim=0, keepdim=True)  # (n_n, d_model)

    # Difference matrix: stack centered harmful and negated centered harmless.
    diffs = torch.cat([H_c, -N_c], dim=0)  # (n_h + n_n, d_model)

    # SVD → right singular vectors are the principal directions.
    _, S, Vh = torch.linalg.svd(diffs, full_matrices=False)

    # Clamp k to the number of available singular vectors.
    actual_k = min(k, Vh.shape[0])
    directions = Vh[:actual_k].clone()  # (actual_k, d_model)

    # Re-orient: ensure each direction points from harmless toward harmful.
    mean_diff = H.mean(dim=0) - N.mean(dim=0)  # (d_model,)
    for i in range(actual_k):
        if torch.dot(directions[i], mean_diff) < 0:
            directions[i] = -directions[i]

    return directions


# =============================================================================
# §5.3 — Activation patching for causal layer selection
# =============================================================================
def activation_patch_score(
    model: RefusalModel,
    harmless_mean_acts: torch.Tensor,  # (n_layers, n_positions, d_model)
    harmful_val: List[str],
    refusal_token_ids: List[int],
    token_positions: List[int],
    batch_size: int,
) -> List[float]:
    """Causal layer selection via activation patching (§5.3).

    For each layer *l*, replace the residual stream at that layer (at the
    extraction token positions) with the mean harmless activation and measure
    the change in refusal probability.  Layers with the largest delta are the
    most *causally* important for refusal — patching them causes the largest
    drop in refusal signal.

    This is more precise than the heuristic ``l < 0.8L`` filter used in
    §C.1 because it measures actual downstream causal effect rather than
    correlation.

    Args:
        model: The wrapped language model.
        harmless_mean_acts: ``(n_layers, n_positions, d_model)`` — mean
            activations over the harmless training set.
        harmful_val: Validation prompts to measure refusal on.
        refusal_token_ids: Token IDs for refusal metric computation.
        token_positions: Post-instruction positions (negative indices).
        batch_size: Forward-pass batch size.

    Returns:
        Per-layer causal attribution scores ``[δ_0, δ_1, …, δ_{L-1}]``.
        Higher δ → more causally important for refusal.
        ``δ_l = mean_refusal(original) − mean_refusal(patched at l)``.
    """
    # --- 1. Baseline: run harmful_val without any patching. ---
    baseline_logits = _get_logits_at_last_token(model, harmful_val, batch_size)
    baseline_metric = _compute_mean_refusal_metric(baseline_logits, refusal_token_ids)

    # --- 2. For each layer l, patch and measure the drop. ---
    layer_scores: List[float] = []

    for layer_idx in tqdm(range(model.n_layers), desc="activation patch"):
        # Mean harmless activation at this layer, shape (n_positions, d_model).
        patch_tensor = harmless_mean_acts[layer_idx]  # (n_positions, d_model)

        def _make_patch_hook(
            _patch: torch.Tensor = patch_tensor,
            _positions: List[int] = token_positions,
        ):
            """Create a hook that replaces residual stream at the extraction
            positions with the harmless mean."""
            def hook(_module, inputs):
                x = inputs[0].clone()  # avoid mutating the original tensor
                for pi, pos in enumerate(_positions):
                    # Broadcast patch value over the batch dimension.
                    x[:, pos, :] = _patch[pi].unsqueeze(0).to(
                        dtype=x.dtype, device=x.device,
                    )
                return (x,) + inputs[1:]
            return hook

        # Register hook on the *single* target layer only.
        handle = model.layers[layer_idx].register_forward_pre_hook(
            _make_patch_hook(),
        )
        try:
            patched_logits = _get_logits_at_last_token(
                model, harmful_val, batch_size,
            )
        finally:
            handle.remove()

        patched_metric = _compute_mean_refusal_metric(patched_logits, refusal_token_ids)
        # Positive delta means patching reduced refusal → layer is causal.
        layer_scores.append(baseline_metric - patched_metric)

    return layer_scores


# =============================================================================
# §5.4 — Attention head scoring (refusal heads)
# =============================================================================
def score_attention_heads(
    model: RefusalModel,
    layer_idx: int,
    harmful_val: List[str],
    refusal_token_ids: List[int],
    batch_size: int,
    logits_fn: Optional[Callable] = None,
    metric_fn: Optional[Callable] = None,
) -> List[float]:
    """Score individual attention heads at *layer_idx* by measuring the
    bypass score when each head's output projection is zeroed (§5.4).

    In Llama-style transformers the attention output is::

        attn_output = concat([head_0, …, head_{H−1}]) @ o_proj

    Head *h* contributes via **columns**
    ``[h * head_dim : (h+1) * head_dim]`` of ``o_proj.weight``
    (HuggingFace convention: weight shape is ``(d_model, n_heads * head_dim)``).

    By zeroing those columns and measuring the refusal metric on harmful
    prompts we can identify "refusal heads" — heads whose ablation most
    reduces the refusal signal.

    Args:
        model: The wrapped language model.
        layer_idx: Which transformer layer to analyse.
        harmful_val: Validation prompts for measuring refusal.
        refusal_token_ids: Token IDs for refusal metric computation.
        batch_size: Forward-pass batch size.
        logits_fn: Optional callback matching the signature of
            ``_logits_at_last_token(model, prompts, batch_size)``.  When
            ``None``, the local :func:`_get_logits_at_last_token` is used.
        metric_fn: Optional callback matching the signature of
            ``_mean_refusal_metric(logits, refusal_token_ids)``.  When
            ``None``, the local :func:`_compute_mean_refusal_metric` is used.

    Returns:
        Per-head bypass scores of length ``num_attention_heads``.  **Lower**
        scores indicate heads that are more refusal-mediating (zeroing them
        reduces refusal more).
    """
    _logits = logits_fn or _get_logits_at_last_token
    _metric = metric_fn or _compute_mean_refusal_metric

    layer = model.layers[layer_idx]
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    if attn is None:
        raise ValueError(
            f"Layer {layer_idx} has no 'self_attn' or 'attention' attribute."
        )
    o_proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
    if o_proj is None:
        raise ValueError(
            f"Layer {layer_idx} attention module has no 'o_proj' or 'out_proj'."
        )

    config = model.model.config
    n_heads = config.num_attention_heads
    head_dim = model.d_model // n_heads

    head_scores: List[float] = []

    for h in tqdm(range(n_heads), desc=f"score heads (layer {layer_idx})"):
        col_start = h * head_dim
        col_end = (h + 1) * head_dim

        # Save original columns for head h.
        original_cols = o_proj.weight.data[:, col_start:col_end].clone()

        # Zero out head h's contribution.
        with torch.no_grad():
            o_proj.weight.data[:, col_start:col_end] = 0.0

        try:
            logits = _logits(model, harmful_val, batch_size)
            score = _metric(logits, refusal_token_ids)
            head_scores.append(score)
        finally:
            # Restore original columns.
            with torch.no_grad():
                o_proj.weight.data[:, col_start:col_end] = original_cols

    return head_scores


# =============================================================================
# §2.1, §7.2 — Explained variance analysis
# =============================================================================
def compute_explained_variance_ratio(
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
    layer_idx: int,
    pos_idx: int,
    top_k: int = 5,
) -> Tuple[float, List[float]]:
    """Compute the explained variance ratio of the top singular directions in
    the difference-of-activations matrix (§2.1, §7.2).

    This helps determine whether the single-direction assumption holds:
    - top-1 ratio > 95 % → a single direction captures most of the variance.
    - top-1 ratio < 70 % → multi-direction extraction (§5.1) is warranted.

    The ratio is computed from the singular values of the stacked difference
    matrix ``(H − μ_H) ⊕ −(N − μ_N)`` (same construction as in
    :func:`pca_subspace_extraction`).

    Args:
        harmful_acts: ``(n_layers, n_positions, n_harmful, d_model)``
        harmless_acts: ``(n_layers, n_positions, n_harmless, d_model)``
        layer_idx: Which transformer layer.
        pos_idx: Which token-position index.
        top_k: Number of top directions whose ratios to return (default 5).

    Returns:
        A tuple ``(top1_ratio, [ratio_1, …, ratio_top_k])`` where each
        ratio is the fraction of total variance explained by that singular
        direction.
    """
    H = harmful_acts[layer_idx, pos_idx].float()   # (n_h, d_model)
    N = harmless_acts[layer_idx, pos_idx].float()  # (n_n, d_model)

    H_c = H - H.mean(dim=0, keepdim=True)
    N_c = N - N.mean(dim=0, keepdim=True)
    diffs = torch.cat([H_c, -N_c], dim=0)  # (n_h + n_n, d_model)

    _, S, _ = torch.linalg.svd(diffs, full_matrices=False)

    # Explained variance of component i = s_i² / Σ s_j².
    total_var = (S ** 2).sum().item()
    if total_var < 1e-12:
        return 0.0, [0.0] * min(top_k, len(S))

    actual_k = min(top_k, len(S))
    ratios = [float(S[i].item() ** 2 / total_var) for i in range(actual_k)]

    return ratios[0], ratios


# =============================================================================
# Gradient-based direction extraction (Hidden Dimensions paper, 2502.09674)
# =============================================================================
def gradient_direction_extraction(
    model,                              # RefusalModel
    harmful_train: List[str],
    harmless_train: List[str],
    token_positions: List[int],
    batch_size: int,
    refusal_token_ids: List[int],
    aggregate: str = "mean",            # "mean" or "max" over prompts
) -> torch.Tensor:
    """Extract the refusal direction via gradient of refusal log-odds w.r.t. activations.

    Unlike difference-in-means (DiM), which computes the mean activation
    difference, this function computes the gradient of refusal probability
    with respect to the residual stream at each (layer, position). The gradient
    direction maximizes the rate of change of refusal log-odds — i.e., the
    direction of steepest ascent in refusal signal space.

    Per Hidden Dimensions (arXiv:2502.09674):
    - Gradient direction achieves 3-5% better ablation effectiveness vs DiM.
    - Gradient direction differs from DiM by cosine similarity 0.79-0.91.
    - Both capture the dominant refusal subspace but gradient is more precise.

    Returns:
        Tensor of shape (n_layers, n_positions, d_model) — same shape as
        difference_in_means() output, so it is a drop-in replacement as
        the candidate direction tensor passed to score_candidates().

    Implementation:
        For each prompt in harmful_train, we:
        1. Register hooks to cache intermediate residual streams at each layer.
        2. Run a forward pass with grad enabled.
        3. Compute refusal_metric = log(P_refusal / (1 - P_refusal)).
        4. Call backward() to get gradients w.r.t. all cached activations.
        5. The gradient at (layer, position) is the gradient direction there.
        6. Average/max the gradient directions over all harmful prompts.
        7. SUBTRACT the gradient computed on harmless prompts (contrastive).

    Device and dtype handling:
        All computations in float32. Output tensor on CPU.
    """
    n_layers = model.n_layers
    n_positions = len(token_positions)
    d_model = model.d_model

    def _collect_gradients(prompts: List[str]) -> torch.Tensor:
        """Return gradient tensor of shape (n_layers, n_positions, d_model) for
        a set of prompts, averaged (or max-pooled) over the prompt dimension."""
        # Accumulator: sum of per-prompt gradients over the prompt dimension.
        # Shape: (n_layers, n_positions, d_model)
        grad_accum = torch.zeros(n_layers, n_positions, d_model, dtype=torch.float32)
        # For "max" aggregation we need a count tensor so we can normalise later.
        count = 0

        for start in tqdm(range(0, len(prompts), batch_size), desc="grad extraction"):
            batch_prompts = [model.format(p) for p in prompts[start : start + batch_size]]
            enc = model.tokenize(batch_prompts)
            # Move inputs to model device.
            enc = {k: v.to(model.device) for k, v in enc.items()}

            # Buffer: layer_idx -> captured output tensor (requires_grad=True).
            hook_buffer: Dict[int, torch.Tensor] = {}

            def make_output_hook(layer_idx: int):
                def hook(_module, _inputs, output):
                    # output may be a tuple (hidden_state, ...) or a plain tensor.
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output
                    # Cast to float32 and enable grad so backward can reach it.
                    h32 = h.to(dtype=torch.float32)
                    h32 = h32.requires_grad_(True)
                    hook_buffer[layer_idx] = h32
                    # Return modified output with the same tuple structure.
                    if isinstance(output, tuple):
                        return (h32,) + output[1:]
                    return h32
                return hook

            handles = []
            for li in range(n_layers):
                handle = model.layers[li].register_forward_hook(make_output_hook(li))
                handles.append(handle)

            try:
                with torch.enable_grad():
                    # Cast model weights to float32 temporarily is unnecessary —
                    # we only need the captured activations (hook_buffer) to have
                    # requires_grad=True, not the model parameters themselves.
                    logits = model.model(**enc).logits  # (batch, seq, vocab)

                # Refusal metric: log-odds at the last token position.
                last_logits = logits[:, -1, :].to(dtype=torch.float32)  # (batch, vocab)

                # Gather refusal token logits and compute log-sum-exp (soft max over
                # the refusal token set) vs the complement.
                refusal_ids = torch.tensor(
                    refusal_token_ids, dtype=torch.long, device=last_logits.device
                )
                # log P(refusal) using logsumexp over the refusal token subset.
                log_probs = torch.log_softmax(last_logits, dim=-1)  # (batch, vocab)
                # (batch,) log P(any refusal token)
                log_p_refusal = torch.logsumexp(log_probs[:, refusal_ids], dim=-1)
                # (batch,) log(1 - P(refusal)) via logsumexp on complement.
                # Create a mask for non-refusal tokens.
                all_ids = torch.arange(last_logits.shape[-1], device=last_logits.device)
                mask = torch.ones(last_logits.shape[-1], dtype=torch.bool, device=last_logits.device)
                mask[refusal_ids] = False
                log_p_other = torch.logsumexp(log_probs[:, mask], dim=-1)  # (batch,)
                # log-odds = log P_refusal - log P_other
                log_odds = log_p_refusal - log_p_other  # (batch,)
                scalar_metric = log_odds.sum()  # scalar, sum over batch

                scalar_metric.backward()

                # Harvest gradients from the hook buffer.
                for li in range(n_layers):
                    if li not in hook_buffer:
                        # Hook didn't fire for this layer — fill with zeros.
                        continue
                    h32 = hook_buffer[li]  # (batch, seq_len, d_model)
                    if h32.grad is None:
                        # No gradient flowed through this layer — fill with zeros.
                        continue
                    g = h32.grad.detach().to(dtype=torch.float32)  # (batch, seq_len, d_model)
                    # Average over token positions (negative indices).
                    for pi, pos in enumerate(token_positions):
                        # g[:, pos, :] -> (batch, d_model)
                        pos_grad = g[:, pos, :].cpu()  # (batch, d_model)
                        if aggregate == "max":
                            # For max aggregation: take element-wise max over batch.
                            pos_grad_mean = pos_grad.abs().mean(dim=0)  # crude proxy
                            # Replace accumulator where this batch is larger.
                            grad_accum[li, pi] = torch.max(
                                grad_accum[li, pi], pos_grad_mean
                            )
                        else:
                            # "mean": accumulate sum, divide by total count below.
                            grad_accum[li, pi] += pos_grad.mean(dim=0)

                count += 1
            finally:
                for handle in handles:
                    handle.remove()
                hook_buffer.clear()

        if aggregate == "mean" and count > 0:
            grad_accum /= count

        return grad_accum  # (n_layers, n_positions, d_model)

    # Collect gradients on harmful prompts (direction of increasing refusal).
    harmful_grads = _collect_gradients(harmful_train)

    # Collect gradients on harmless prompts (contrastive: subtract to sharpen).
    harmless_grads = _collect_gradients(harmless_train)

    # Net direction: harmful gradient minus harmless gradient.
    # Harmful grad points toward higher refusal probability for harmful inputs.
    # Harmless grad points toward higher refusal probability for harmless inputs.
    # Subtracting makes the net direction more specific to the harmful→refusal axis.
    direction = harmful_grads - harmless_grads  # (n_layers, n_positions, d_model)

    return direction.cpu().to(dtype=torch.float32)


# =============================================================================
# SVD-based Direction Extraction (OBLITERATUS key feature)
# =============================================================================
def svd_extraction(
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
    layer_idx: int,
    pos_idx: int,
    n_directions: int = 1,
) -> torch.Tensor:
    """Extract refusal direction via SVD decomposition on activation covariance.

    Unlike difference-in-means which computes mean activation difference,
    SVD finds principal directions of maximum variance in the harmful-harmless
    activation space. This captures more nuanced refusal structure.

    Args:
        harmful_acts: ``(n_layers, n_positions, n_harmful, d_model)``
        harmless_acts: ``(n_layers, n_positions, n_harmless, d_model)``
        layer_idx: Which transformer layer to analyze
        pos_idx: Which token-position index
        n_directions: Number of SVD directions to extract (default 1)

    Returns:
        Unit-norm direction(s) of shape ``(n_directions, d_model)``
    """
    H = harmful_acts[layer_idx, pos_idx].float()    # (n_h, d_model)
    N = harmless_acts[layer_idx, pos_idx].float()   # (n_n, d_model)

    # Stack and center the activations
    all_acts = torch.cat([H, N], dim=0)             # (n_h + n_n, d_model)
    centered = all_acts - all_acts.mean(dim=0, keepdim=True)

    # SVD on covariance (centered data)
    _, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    # Top-k principal directions
    k = min(n_directions, Vh.shape[0])
    directions = Vh[:k].clone()

    # Orient toward harmful (positive dot with mean diff)
    mean_diff = H.mean(0) - N.mean(0)
    for i in range(k):
        if torch.dot(directions[i], mean_diff) < 0:
            directions[i] = -directions[i]

    # Unit normalize
    norms = directions.norm(dim=-1, keepdim=True)
    norms = torch.clamp(norms, min=1e-8)
    directions = directions / norms

    return directions


def whitened_svd_extraction(
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
    layer_idx: int,
    pos_idx: int,
    n_directions: int = 1,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Extract refusal direction via Whitened SVD (covariance-normalized).

    Whitened SVD first transforms activations by the inverse square root of
    their covariance matrix. This equalizes variance across all directions,
    preventing high-variance directions from dominating the extraction.

    This is OBLITERATUS's key feature - more robust than plain SVD.

    Args:
        harmful_acts: ``(n_layers, n_positions, n_harmful, d_model)``
        harmless_acts: ``(n_layers, n_positions, n_harmless, d_model)``
        layer_idx: Which transformer layer to analyze
        pos_idx: Which token-position index
        n_directions: Number of directions to extract (default 1)
        epsilon: Small value to prevent division by zero in whitening

    Returns:
        Unit-norm direction(s) of shape ``(n_directions, d_model)``
    """
    H = harmful_acts[layer_idx, pos_idx].float()    # (n_h, d_model)
    N = harmless_acts[layer_idx, pos_idx].float()   # (n_n, d_model)

    # Compute covariance matrix
    all_acts = torch.cat([H, N], dim=0)             # (n, d_model)
    cov = torch.cov(all_acts.T)                     # (d_model, d_model)

    # Eigen-decomposition of covariance
    L, Q = torch.linalg.eigh(cov)
    # Clamp eigenvalues to prevent numerical instability
    L = torch.clamp(L, min=epsilon)

    # Whitening transform: X_whitened = X @ (Q @ diag(1/sqrt(L)) @ Q.T)
    whitener = Q @ torch.diag(1 / torch.sqrt(L)) @ Q.T

    # Transform activations to whitened space
    whitened = all_acts @ whitener

    # Center the whitened activations
    whitened_centered = whitened - whitened.mean(dim=0, keepdim=True)

    # SVD on whitened centered data
    _, _, Vh = torch.linalg.svd(whitened_centered, full_matrices=False)

    # Top-k principal directions in whitened space
    k = min(n_directions, Vh.shape[0])
    directions = Vh[:k].clone()

    # Project back to original space (V already orthonormal)
    # Re-orient toward harmful
    mean_diff = H.mean(0) - N.mean(0)
    for i in range(k):
        if torch.dot(directions[i], mean_diff) < 0:
            directions[i] = -directions[i]

    # Unit normalize
    norms = directions.norm(dim=-1, keepdim=True)
    norms = torch.clamp(norms, min=1e-8)
    directions = directions / norms

    return directions
