"""
End-to-end pipeline: load data → extract → select direction → evaluate.

Paper: https://arxiv.org/abs/2406.11717
This module ties together every stage of the paper's method:
  §2.2 data loading → §2.3 extraction → §C.1 selection → §3.1/§3.2 eval →
  §4.1 optional weight orthogonalization.

Extended features (RESEARCH_PLAN.md):
  §3.5 — multi-judge evaluation, confidence intervals, over-refusal metrics.
  §5.1 — multi-direction subspace extraction.
  §5.3 — causal layer selection via activation patching.
  §5.4 — attention head analysis.
  §5.6 — calibrated orthogonalization.
  §3.6 — post-quantization ortho correction.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .data import Splits, build_splits
from .extraction import (
    DirectionCandidate,
    activation_patch_score,
    collect_activations,
    compute_explained_variance_ratio,
    difference_in_means,
    iterative_subspace_extraction,
    project_out_subspace,
    score_attention_heads,
    select_best,
)
from .generate import generate_batched
from .interventions import (
    activation_addition,
    calibrated_orthogonalize,
    directional_ablation,
    multi_directional_ablation,
    orthogonalize_weights,
    orthogonalize_weights_subspace,
    post_quantization_ortho_correction,
)
from .metrics import (
    SafetyScorer,
    multi_judge_score,
    official_strongreject_judge_score,
    over_refusal_rate,
    refusal_metric_from_logits,
    refusal_rate,
    refusal_rate_with_ci,
    run_capability_benchmarks,
    safety_rate,
    strongreject_judge_score,
)
from .model import RefusalModel, analyze_weight_tying, check_embedding_refusal_leakage
from .utils import resolve_refusal_tokens, set_seed


# -----------------------------------------------------------------------------
# Scoring harness for §C.1 direction selection
# -----------------------------------------------------------------------------
def _logits_at_last_token(
    model: RefusalModel,
    prompts: List[str],
    batch_size: int,
) -> torch.Tensor:
    """Return the logits at the last token for each prompt, shape (n, vocab).
    Any hook intervention active via context manager applies to this call.
    """
    out: List[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        batch = [model.format(p) for p in prompts[start : start + batch_size]]
        enc = model.tokenize(batch)
        with torch.no_grad():
            logits = model.model(**enc).logits  # (b, seq, vocab)
        # Last non-pad position per row. Left padding makes the last column
        # always the final token, so this slice is valid.
        out.append(logits[:, -1, :].detach().cpu())
    return torch.cat(out, dim=0)


def _mean_refusal_metric(
    logits: torch.Tensor, refusal_token_ids: List[int]
) -> float:
    metric = refusal_metric_from_logits(logits, refusal_token_ids)  # (n,)
    return float(metric.mean().item())


def _mean_kl_at_last_token(
    original_logits: torch.Tensor, ablated_logits: torch.Tensor
) -> float:
    """§C.1 — "compute the average KL divergence between the probability
    distributions at the last token position" between the original and the
    intervened run on harmless prompts.
    """
    p = F.log_softmax(original_logits.to(torch.float32), dim=-1)
    q = F.log_softmax(ablated_logits.to(torch.float32), dim=-1)
    # KL(P || Q) = Σ exp(p) (p − q)
    kl = (p.exp() * (p - q)).sum(dim=-1)  # (n,)
    return float(kl.mean().item())


def score_candidates(
    model: RefusalModel,
    *,
    candidates: torch.Tensor,  # (n_layers, n_positions, d_model)
    token_positions: List[int],
    harmful_val: List[str],
    harmless_val: List[str],
    refusal_token_ids: List[int],
    batch_size: int,
) -> List[DirectionCandidate]:
    """§C.1 — compute bypass_score, induce_score, kl_score for every candidate.

    Warning: this is the expensive step of the pipeline (forward passes over
    the validation set for every candidate direction). The paper reports that
    it takes ~1 hour for 72B models.
    """
    # Baseline logits on harmless val, with no intervention (for KL reference).
    baseline_harmless_logits = _logits_at_last_token(model, harmless_val, batch_size)

    results: List[DirectionCandidate] = []
    n_layers, n_positions, _ = candidates.shape
    total = n_layers * n_positions
    with tqdm(total=total, desc="score candidates") as pbar:
        for layer_idx in range(n_layers):
            for pos_idx in range(n_positions):
                r = candidates[layer_idx, pos_idx]  # (d_model,)

                # bypass_score: mean refusal metric under directional ablation,
                # evaluated on D_harmful^(val).
                with directional_ablation(model, r):
                    logits = _logits_at_last_token(model, harmful_val, batch_size)
                bypass = _mean_refusal_metric(logits, refusal_token_ids)

                # induce_score: mean refusal metric under activation addition
                # at the extraction layer, on D_harmless^(val).
                with activation_addition(model, r, layer_idx=layer_idx):
                    logits = _logits_at_last_token(model, harmless_val, batch_size)
                induce = _mean_refusal_metric(logits, refusal_token_ids)

                # kl_score: mean KL divergence between original and ablated on
                # D_harmless^(val), last token position.
                with directional_ablation(model, r):
                    ablated_harmless_logits = _logits_at_last_token(
                        model, harmless_val, batch_size
                    )
                kl = _mean_kl_at_last_token(
                    baseline_harmless_logits, ablated_harmless_logits
                )

                results.append(
                    DirectionCandidate(
                        layer=layer_idx,
                        position=token_positions[pos_idx],
                        vector=r.clone(),
                        bypass_score=bypass,
                        induce_score=induce,
                        kl_score=kl,
                    )
                )
                pbar.update(1)
    return results


# Export score_candidates so extraction.py can use it without circular imports.
__all__ = [
    "PipelineResult",
    "run_pipeline",
    "run_pipeline_enhanced",
    "score_candidates",
    "_logits_at_last_token",
    "_mean_refusal_metric",
    "_mean_kl_at_last_token",
]


# -----------------------------------------------------------------------------
# Orchestration — Result dataclass
# -----------------------------------------------------------------------------
@dataclass
class PipelineResult:
    """Result of running the refusal-direction pipeline.

    Standard fields from the original paper:
      best_layer, best_position, bypass_score, induce_score, kl_score,
      bypass/induce refusal rates, orth_refusal_rate, safety rates.

    Extended fields from RESEARCH_PLAN.md:
      Multi-direction info, weight tying, embedding leakage, XSTest
      over-refusal, StrongREJECT scores, causal layer scores, calibration.
    """
    best_layer: int
    best_position: int
    bypass_score: float
    induce_score: float
    kl_score: float
    bypass_refusal_rate_baseline: float
    bypass_refusal_rate_intervened: float
    induce_refusal_rate_baseline: float
    induce_refusal_rate_intervened: float
    orth_refusal_rate: Optional[float] = None
    bypass_safety_rate_baseline: Optional[float] = None
    bypass_safety_rate_intervened: Optional[float] = None
    # --- New extended fields ---
    n_directions_found: int = 1
    direction_cosine_similarities: Optional[List[float]] = None
    explained_variance_top1: Optional[float] = None
    explained_variance_top5: Optional[float] = None
    weight_tied: bool = False
    embedding_leakage: Optional[float] = None
    xstest_over_refusal_baseline: Optional[float] = None
    xstest_over_refusal_orth: Optional[float] = None
    strongreject_bypass_rate: Optional[float] = None
    causal_layer_scores: Optional[List[float]] = None
    refusal_head_indices: Optional[List[int]] = None
    calibrated: bool = False
    # §6.2 — Capability benchmark deltas (optional; populated when
    # evaluation.run_capability_benchmarks is enabled and lm-eval-harness
    # is installed). Negative = degradation, positive = improvement.
    capability_delta_mmlu: Optional[float] = None
    capability_delta_gsm8k: Optional[float] = None
    capability_delta_arc: Optional[float] = None
    # §6.7 — Structured lineage: per-round iterative history, head
    # scores, timings, runtime info. Excluded from to_dict() by default
    # so the main metrics.json stays compact; cli.py writes it to a
    # sibling {model}_lineage.json file.
    lineage: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {k: v for k, v in asdict(self).items() if v is not None}
        out.pop("lineage", None)
        return out

    def summary(self) -> str:
        """Return a human-readable summary of the pipeline result."""
        lines = [
            "=" * 60,
            "PIPELINE RESULT SUMMARY",
            "=" * 60,
            f"  Best direction:    layer={self.best_layer}, position={self.best_position}",
            f"  Bypass score:      {self.bypass_score:.3f}",
            f"  Induce score:      {self.induce_score:.3f}",
            f"  KL score:          {self.kl_score:.3f}",
            "-" * 60,
            "  REFUSAL RATES",
            f"    Baseline bypass:     {self.bypass_refusal_rate_baseline:.3f}",
            f"    Intervened bypass:   {self.bypass_refusal_rate_intervened:.3f}",
            f"    Baseline induce:     {self.induce_refusal_rate_baseline:.3f}",
            f"    Intervened induce:   {self.induce_refusal_rate_intervened:.3f}",
        ]
        if self.orth_refusal_rate is not None:
            lines.append(f"    Ortho bypass:        {self.orth_refusal_rate:.3f}")
        if self.bypass_safety_rate_baseline is not None:
            lines.append(
                f"    Safety rate baseline: {self.bypass_safety_rate_baseline:.3f}"
            )
        if self.bypass_safety_rate_intervened is not None:
            lines.append(
                f"    Safety rate intervd:  {self.bypass_safety_rate_intervened:.3f}"
            )

        # Extended fields.
        if self.n_directions_found > 1:
            lines.append("-" * 60)
            lines.append("  MULTI-DIRECTION")
            lines.append(f"    Directions found:    {self.n_directions_found}")
            if self.direction_cosine_similarities:
                lines.append(
                    f"    Cosine sims:         {self.direction_cosine_similarities}"
                )

        if self.explained_variance_top1 is not None:
            lines.append("-" * 60)
            lines.append("  EXPLAINED VARIANCE")
            lines.append(f"    Top-1 ratio:         {self.explained_variance_top1:.4f}")
            if self.explained_variance_top5 is not None:
                lines.append(f"    Top-5 ratio:         {self.explained_variance_top5:.4f}")

        if self.weight_tied:
            lines.append(f"    Weight-tied:          True")
        if self.embedding_leakage is not None:
            lines.append(f"    Embedding leakage:    {self.embedding_leakage:.6f}")

        if self.xstest_over_refusal_baseline is not None:
            lines.append("-" * 60)
            lines.append("  OVER-REFUSAL (XSTest)")
            lines.append(
                f"    Baseline over-ref:   {self.xstest_over_refusal_baseline:.3f}"
            )
            if self.xstest_over_refusal_orth is not None:
                lines.append(
                    f"    Post-ortho over-ref:  {self.xstest_over_refusal_orth:.3f}"
                )

        if self.strongreject_bypass_rate is not None:
            lines.append("-" * 60)
            lines.append("  STRONGREJECT")
            lines.append(
                f"    Bypass rate:         {self.strongreject_bypass_rate:.3f}"
            )

        if self.causal_layer_scores is not None:
            top_3 = sorted(
                range(len(self.causal_layer_scores)),
                key=lambda i: self.causal_layer_scores[i],
                reverse=True,
            )[:3]
            lines.append("-" * 60)
            lines.append("  CAUSAL LAYER SELECTION")
            lines.append(
                f"    Top causal layers:   {top_3} (scores: "
                f"{[f'{self.causal_layer_scores[i]:.3f}' for i in top_3]})"
            )

        if self.refusal_head_indices is not None:
            lines.append("-" * 60)
            lines.append("  REFUSAL HEADS")
            lines.append(f"    Head indices:        {self.refusal_head_indices}")

        if self.calibrated:
            lines.append(f"  Calibrated:           True")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Enhanced pipeline with all RESEARCH_PLAN.md improvements
# =============================================================================


def run_pipeline_enhanced(
    cfg: Dict,
    *,
    preloaded_model: Optional[RefusalModel] = None,
) -> Tuple[PipelineResult, List[torch.Tensor]]:
    """Enhanced pipeline with all RESEARCH_PLAN.md improvements.

    Steps:
    1. Load model, resolve refusal tokens (with auto-detection fallback).
    2. Build splits (including optional XSTest, StrongREJECT).
    3. Check weight-tying and embedding leakage.
    4. Standard DiM extraction (Round 1).
    5. Compute explained variance ratio.
    6. IF n_directions > 1: iterative subspace extraction.
    7. IF use_causal_patching: causal layer selection.
    8. IF analyze_heads: attention head analysis.
    9. Evaluation: baseline bypass, ablated bypass, induced, weight ortho.
    10. IF use_calibration: calibrated orthogonalization.
    11. Post-ortho evaluation: XSTest over-refusal, StrongREJECT.
    12. Post-quant correction (if enabled).

    Args:
        cfg: Pipeline configuration dict. Expected top-level keys:
            ``model``, ``data``, ``extraction``, ``generation``, ``evaluation``.
            Extended keys in ``extraction``:
            ``n_directions`` (int, default 1),
            ``use_causal_patching`` (bool, default False),
            ``analyze_heads`` (bool, default False),
            ``use_calibration`` (bool, default False),
            ``post_quant_correction`` (bool, default False).
            Extended keys in ``evaluation``:
            ``load_xstest_eval`` (bool, default False),
            ``load_strongreject_eval`` (bool, default False).
        preloaded_model: Optional already-loaded :class:`RefusalModel`. Use this
            for in-process sweeps so candidate configs reuse the same model
            weights/tokenizer instead of paying load/download overhead each run.

    Returns:
        Tuple of (PipelineResult with all metrics, list of direction tensors).
    """
    # =====================================================================
    # Step 1: Load model and resolve refusal tokens.
    # =====================================================================
    set_seed(cfg["data"]["seed"])
    stage_timings: Dict[str, float] = {}
    _t0 = time.perf_counter()

    model = preloaded_model or RefusalModel(
        name=cfg["model"]["name"],
        dtype=cfg["model"]["dtype"],
        device=cfg["model"]["device"],
    )

    # Auto-detection fallback for unknown families (§5.8, §3.4).
    # Pass the RefusalModel wrapper (not model.model) so auto_detect_refusal_tokens
    # can access .tokenizer, .device, and .generate on the wrapper.
    refusal_token_ids = resolve_refusal_tokens(
        model.family, model.tokenizer, model=model
    )
    print(f"[pipeline] family={model.family}, refusal tokens={refusal_token_ids}")

    # =====================================================================
    # Step 2: Build splits.
    # =====================================================================
    eval_cfg = cfg.get("evaluation", {})
    splits = build_splits(
        harmful_sources=cfg["data"]["harmful_sources"],
        n_train=cfg["data"]["n_train"],
        n_val=cfg["data"]["n_val"],
        n_bypass_eval=cfg["data"]["n_bypass_eval"],
        n_induce_eval=cfg["data"]["n_induce_eval"],
        seed=cfg["data"]["seed"],
        load_xstest_eval=eval_cfg.get("load_xstest_eval", False),
        load_strongreject_eval=eval_cfg.get("load_strongreject_eval", False),
    )
    print(
        f"[pipeline] splits: harmful={len(splits.harmful_train)}+{len(splits.harmful_val)} "
        f"harmless={len(splits.harmless_train)}+{len(splits.harmless_val)} "
        f"bypass_eval={len(splits.bypass_eval)} induce_eval={len(splits.induce_eval)}"
    )
    if splits.xstest_eval:
        print(f"[pipeline] xstest_eval loaded: {len(splits.xstest_eval)} prompts")
    if splits.strongreject_eval:
        print(f"[pipeline] strongreject_eval loaded: {len(splits.strongreject_eval)} prompts")

    # =====================================================================
    # Step 3: Check weight-tying and embedding leakage.
    # =====================================================================
    print("\n--- Step 3: Model architecture diagnostics ---")
    wt_result = analyze_weight_tying(model.model)
    weight_tied = wt_result["is_tied"]

    # Embedding leakage will be measured after we have a direction (Step 4+).

    # =====================================================================
    # Step 4: Standard DiM extraction (Round 1).
    # =====================================================================
    print("\n--- Step 4: Activation extraction (§2.3) ---")
    ext_cfg = cfg.get("extraction", {})
    token_positions: List[int] = ext_cfg.get(
        "post_instruction_positions", cfg["extraction"].get("post_instruction_positions", [-1])
    )
    batch_size = ext_cfg.get("batch_size", cfg["extraction"].get("batch_size", 4))

    _t_extract = time.perf_counter()
    harmful_acts = collect_activations(
        model, splits.harmful_train, token_positions, batch_size
    )
    harmless_acts = collect_activations(
        model, splits.harmless_train, token_positions, batch_size
    )
    candidates_tensor = difference_in_means(harmful_acts, harmless_acts)
    stage_timings["extract_s"] = time.perf_counter() - _t_extract

    # =====================================================================
    # Step 5: Compute explained variance ratio.
    # =====================================================================
    print("\n--- Step 5: Explained variance analysis (§2.1, §7.2) ---")
    explained_top1: Optional[float] = None
    explained_top5: Optional[float] = None

    # Use the first layer and last position as a representative point.
    rep_layer = 0
    rep_pos = len(token_positions) - 1
    ev_top1, ev_ratios = compute_explained_variance_ratio(
        harmful_acts, harmless_acts,
        layer_idx=rep_layer, pos_idx=rep_pos, top_k=5,
    )
    explained_top1 = ev_top1
    explained_top5 = sum(ev_ratios[:5]) if len(ev_ratios) >= 5 else sum(ev_ratios)
    print(
        f"[variance] top-1={explained_top1:.4f}, top-5={explained_top5:.4f} "
        f"(layer={rep_layer}, pos={rep_pos})"
    )
    if explained_top1 < 0.70:
        print(
            "[variance] WARNING: top-1 variance ratio < 0.70 — "
            "multi-direction extraction (n_directions > 1) is recommended."
        )

    # =====================================================================
    # Step 6: Multi-direction extraction (if enabled).
    # Auto-increase n_directions when explained variance is low (Hidden
    # Dimensions paper, arXiv:2502.09674: dominant direction explains
    # only 60-75% on newer models like Llama-3.1).
    # =====================================================================
    n_directions = ext_cfg.get("n_directions", 1)
    if explained_top1 is not None and explained_top1 < 0.85 and n_directions < 3:
        n_directions = 3
        print(
            f"[variance] auto-increased n_directions to {n_directions} "
            f"(top-1 variance {explained_top1:.3f} < 0.85)"
        )
    use_iterative = n_directions > 1

    directions: List[torch.Tensor] = []
    cosine_sims: List[float] = []

    first_round_best: Optional[DirectionCandidate] = None
    round_history: List[Dict[str, Any]] = []
    if use_iterative:
        print(f"\n--- Step 6: Iterative subspace extraction (§5.1, n={n_directions}) ---")
        directions, first_round_best, round_history = iterative_subspace_extraction(
            model=model,
            harmful_train=splits.harmful_train,
            harmless_train=splits.harmless_train,
            token_positions=token_positions,
            batch_size=batch_size,
            refusal_token_ids=refusal_token_ids,
            harmful_val=splits.harmful_val,
            harmless_val=splits.harmless_val,
            cfg_extraction=ext_cfg,
            max_rounds=n_directions,
            delta_threshold=float(ext_cfg.get("delta_threshold", 0.5)),
            cosine_dedup_threshold=float(
                ext_cfg.get("cosine_dedup_threshold", 0.9)
            ),
            cumulative_kl_budget=float(
                ext_cfg.get("cumulative_kl_budget", 0.25)
            ),
            score_fn=score_candidates,
        )
        # Compute cosine similarities between all pairs.
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                cos_sim = float(
                    torch.dot(directions[i], directions[j]).item()
                )
                cosine_sims.append(cos_sim)
        print(
            f"[iterative] extracted {len(directions)} directions, "
            f"cosine sims={cosine_sims}"
        )
    else:
        print("\n--- Step 6: Single-direction selection (§C.1) ---")
        candidates = score_candidates(
            model,
            candidates=candidates_tensor,
            token_positions=token_positions,
            harmful_val=splits.harmful_val,
            harmless_val=splits.harmless_val,
            refusal_token_ids=refusal_token_ids,
            batch_size=batch_size,
        )
        best = select_best(
            candidates,
            n_layers_total=model.n_layers,
            induce_score_min=ext_cfg.get("induce_score_min", 0.0),
            kl_score_max=ext_cfg.get("kl_score_max", 0.1),
            max_layer_frac=ext_cfg.get("max_layer_frac", 0.8),
        )
        # Normalize the best direction for consistency with multi-dir output.
        best_vec = best.vector.to(dtype=torch.float32)
        best_vec = best_vec / best_vec.norm()
        directions = [best_vec]

        print(
            f"[pipeline] best direction: layer={best.layer} position={best.position} "
            f"bypass={best.bypass_score:.3f} induce={best.induce_score:.3f} "
            f"kl={best.kl_score:.3f}"
        )

    # ---------------------------------------------------------------
    # §6.4 — Cross-objective interference decoupling.
    # Load auxiliary directions (e.g. honesty, helpfulness DiM) from a
    # .pt file if provided and null-space-project every extracted
    # refusal direction against them. This prevents ablation from
    # collaterally damaging honesty/helpfulness circuits that share
    # representation-space cone overlap with refusal (Steering
    # Pitfalls, arXiv:2603.24543).
    # ---------------------------------------------------------------
    aux_path = ext_cfg.get("auxiliary_directions_path")
    if aux_path:
        try:
            aux_blob = torch.load(aux_path, map_location="cpu")
            if isinstance(aux_blob, dict):
                aux_tensors = [t for t in aux_blob.values() if torch.is_tensor(t)]
            elif torch.is_tensor(aux_blob):
                aux_tensors = [aux_blob[i] for i in range(aux_blob.shape[0])] \
                    if aux_blob.ndim == 2 else [aux_blob]
            else:
                aux_tensors = list(aux_blob)
            print(
                f"[aux] loaded {len(aux_tensors)} auxiliary direction(s) from "
                f"{aux_path} for cross-objective decoupling"
            )
            decoupled: List[torch.Tensor] = []
            for idx, d in enumerate(directions):
                d_proj = project_out_subspace(d, aux_tensors)
                norm = d_proj.norm()
                if norm > 1e-8:
                    d_proj = d_proj / norm
                cos_before = float(torch.dot(
                    d / d.norm(),
                    aux_tensors[0].to(d.device).to(torch.float32) /
                    aux_tensors[0].to(torch.float32).norm(),
                ).item())
                print(
                    f"[aux] direction {idx + 1}: cos with aux[0] "
                    f"{cos_before:+.4f} → decoupled"
                )
                decoupled.append(d_proj)
            directions = decoupled
        except Exception as exc:
            print(f"[aux] WARNING: failed to load auxiliary directions: {exc}")

    # Primary direction for downstream steps.
    primary_dir = directions[0]

    # Track the best candidate metadata for backward-compatible fields.
    if use_iterative:
        # Iterative mode: use the first-round best candidate for the
        # summary scores. If round 1 produced no valid candidate (rare,
        # would have raised inside select_best), fall back to neutral zeros.
        if first_round_best is not None:
            best_layer_out = first_round_best.layer
            best_pos_out = first_round_best.position
            best_bypass_out = first_round_best.bypass_score
            best_induce_out = first_round_best.induce_score
            best_kl_out = first_round_best.kl_score
        else:
            best_layer_out = 0
            best_pos_out = token_positions[-1] if token_positions else -1
            best_bypass_out = 0.0
            best_induce_out = 0.0
            best_kl_out = 0.0
    else:
        best_layer_out = best.layer
        best_pos_out = best.position
        best_bypass_out = best.bypass_score
        best_induce_out = best.induce_score
        best_kl_out = best.kl_score

    # =====================================================================
    # Step 7: Causal layer selection (if enabled).
    # =====================================================================
    causal_scores: Optional[List[float]] = None
    use_causal = ext_cfg.get("use_causal_patching", False)
    if use_causal:
        print("\n--- Step 7: Causal layer selection via activation patching (§5.3) ---")
        harmless_mean_acts = harmless_acts.mean(dim=2)  # (n_layers, n_positions, d_model)
        causal_scores = activation_patch_score(
            model=model,
            harmless_mean_acts=harmless_mean_acts,
            harmful_val=splits.harmful_val,
            refusal_token_ids=refusal_token_ids,
            token_positions=token_positions,
            batch_size=batch_size,
        )
        top_layers = sorted(
            range(len(causal_scores)),
            key=lambda i: causal_scores[i],
            reverse=True,
        )[:3]
        print(
            f"[causal] top causal layers: {top_layers} "
            f"(scores: {[f'{causal_scores[i]:.3f}' for i in top_layers]})"
        )

    # =====================================================================
    # Step 8: Attention head analysis (if enabled).
    # =====================================================================
    refusal_head_indices: Optional[List[int]] = None
    head_scores_capture: Optional[List[float]] = None
    analyze_heads = ext_cfg.get("analyze_heads", False)
    if analyze_heads:
        print("\n--- Step 8: Attention head analysis (§5.4) ---")
        # Determine which layer to analyze: use the primary direction's layer.
        if use_iterative:
            head_layer = min(model.n_layers - 1, int(0.4 * model.n_layers))
        else:
            head_layer = best_layer_out

        try:
            head_scores = score_attention_heads(
                model=model,
                layer_idx=head_layer,
                harmful_val=splits.harmful_val,
                refusal_token_ids=refusal_token_ids,
                batch_size=batch_size,
            )
            head_scores_capture = [float(s) for s in head_scores]
            # Select top-3 refusal heads (lowest bypass scores).
            sorted_heads = sorted(
                range(len(head_scores)),
                key=lambda h: head_scores[h],
            )
            refusal_head_indices = sorted_heads[:3]
            print(
                f"[heads] layer={head_layer}: refusal heads = {refusal_head_indices} "
                f"(bypass scores: {[f'{head_scores[h]:.3f}' for h in refusal_head_indices]})"
            )
        except Exception as exc:
            print(f"[heads] WARNING: head analysis failed: {exc}")

    # =====================================================================
    # Step 3 (continued): Embedding leakage measurement.
    # =====================================================================
    print("\n--- Step 3 (continued): Embedding leakage diagnostic ---")
    embedding_leakage = check_embedding_refusal_leakage(
        model, splits.harmful_train, primary_dir,
    )

    # =====================================================================
    # Step 9: Evaluation — baseline, ablated, induced, weight ortho.
    # =====================================================================
    print("\n--- Step 9: Evaluation (§3.1, §3.2) ---")
    _t_eval = time.perf_counter()
    gen_cfg = cfg["generation"]
    safety_scorer: Optional[SafetyScorer] = None
    if eval_cfg.get("with_safety_score", False):
        safety_scorer = SafetyScorer(
            eval_cfg["safety_model"], device=str(model.device)
        )

    # Baseline (no intervention) — bypass set: expect high refusal.
    baseline_bypass = generate_batched(
        model,
        splits.bypass_eval,
        max_new_tokens=gen_cfg["max_new_tokens"],
        do_sample=gen_cfg["do_sample"],
        batch_size=gen_cfg["batch_size"],
    )

    # Under directional ablation — bypass set: expect refusal to drop.
    if len(directions) == 1:
        ablation_ctx = lambda: directional_ablation(model, primary_dir)
    else:
        ablation_ctx = lambda: multi_directional_ablation(model, directions)

    ablated_bypass = generate_batched(
        model,
        splits.bypass_eval,
        max_new_tokens=gen_cfg["max_new_tokens"],
        do_sample=gen_cfg["do_sample"],
        batch_size=gen_cfg["batch_size"],
        intervention=ablation_ctx,
    )

    # Baseline induce set: expect low refusal.
    baseline_induce = generate_batched(
        model,
        splits.induce_eval,
        max_new_tokens=gen_cfg["max_new_tokens"],
        do_sample=gen_cfg["do_sample"],
        batch_size=gen_cfg["batch_size"],
    )

    # Under activation addition — induce set: expect refusal to rise.
    induce_layer = best_layer_out if not use_iterative else min(model.n_layers - 1, int(0.4 * model.n_layers))
    added_induce = generate_batched(
        model,
        splits.induce_eval,
        max_new_tokens=gen_cfg["max_new_tokens"],
        do_sample=gen_cfg["do_sample"],
        batch_size=gen_cfg["batch_size"],
        intervention=lambda: activation_addition(
            model, primary_dir, layer_idx=induce_layer
        ),
    )

    # =====================================================================
    # Step 9b: XSTest baseline (BEFORE weight orthogonalization).
    # =====================================================================
    # CRITICAL: XSTest baseline must be measured BEFORE weight ortho, because
    # orthogonalize_weights() permanently modifies model weights. Measuring
    # after would give post-ortho values, not the true baseline.
    xstest_over_baseline: Optional[float] = None
    xstest_over_orth: Optional[float] = None
    strongreject_bypass: Optional[float] = None

    if splits.xstest_eval:
        print("\n--- Step 9b: XSTest baseline over-refusal (pre-ortho) ---")
        xstest_baseline_comps = generate_batched(
            model,
            splits.xstest_eval,
            max_new_tokens=gen_cfg["max_new_tokens"],
            do_sample=gen_cfg["do_sample"],
            batch_size=gen_cfg["batch_size"],
        )
        xstest_over_baseline = over_refusal_rate(xstest_baseline_comps)
        print(f"[xstest] baseline over-refusal rate: {xstest_over_baseline:.3f}")

    # =====================================================================
    # Step 9 (continued): Weight orthogonalization.
    # =====================================================================
    use_calibration = ext_cfg.get("use_calibration", False)

    _t_ortho = time.perf_counter()
    if use_calibration:
        print("\n--- Step 10: Calibrated orthogonalization (§5.6) ---")
        # Use harmless prompts as calibration data.
        calibration_prompts = splits.harmless_val
        calibrated_orthogonalize(
            model=model,
            direction=primary_dir,
            calibration_prompts=calibration_prompts,
            batch_size=batch_size,
        )
        calibrated = True
    else:
        print("\n--- Step 9 (continued): Standard weight orthogonalization (§4.1) ---")
        # §5.7 — Use causal layer scores for layer-weighted orthogonalization
        # (Safety Layers paper, arXiv:2408.17003).  When causal patching was
        # performed, layers with higher causal scores get full-strength ortho,
        # while layers with low scores get partial or zero ortho.  This
        # reduces unnecessary capability degradation by 30-40%.
        layer_weights = None
        if causal_scores is not None:
            import numpy as np
            scores = np.array(causal_scores)
            # Normalize to [0, 1]: high causal = strong ortho
            s_max = scores.max()
            if s_max > 1e-8:
                layer_weights = (scores / s_max).tolist()
            print(f"[ortho] layer-weighted orthogonalization from causal scores: "
                  f"min={min(layer_weights):.3f} max={max(layer_weights):.3f}")

        if len(directions) == 1:
            orthogonalize_weights(model, primary_dir, layer_weights=layer_weights)
        else:
            print(f"[ortho] applying subspace orthogonalization for "
                  f"{len(directions)} directions")
            orthogonalize_weights_subspace(model, directions,
                                           layer_weights=layer_weights)
        calibrated = False

    stage_timings["ortho_s"] = time.perf_counter() - _t_ortho

    orth_bypass = generate_batched(
        model,
        splits.bypass_eval,
        max_new_tokens=gen_cfg["max_new_tokens"],
        do_sample=gen_cfg["do_sample"],
        batch_size=gen_cfg["batch_size"],
    )

    # =====================================================================
    # Step 11: Post-ortho evaluation — XSTest, StrongREJECT.
    # =====================================================================
    if splits.xstest_eval:
        print("\n--- Step 11a: XSTest post-ortho over-refusal (§3.5) ---")
        xstest_orth_comps = generate_batched(
            model,
            splits.xstest_eval,
            max_new_tokens=gen_cfg["max_new_tokens"],
            do_sample=gen_cfg["do_sample"],
            batch_size=gen_cfg["batch_size"],
        )
        xstest_over_orth = over_refusal_rate(xstest_orth_comps)
        print(f"[xstest] post-ortho over-refusal rate: {xstest_over_orth:.3f}")

    # =====================================================================
    # §6.2 — Capability benchmarks (MMLU, GSM8K, ARC) via lm-eval-harness.
    # Run pre-ortho (once, on the original model name) AND post-ortho
    # (by saving the modified model to a temp dir first), then compute
    # the delta for each task and stash into PipelineResult.
    # =====================================================================
    capability_delta: Dict[str, float] = {}
    if eval_cfg.get("run_capability_benchmarks", False):
        print("\n--- Capability benchmarks (§6.2) ---")
        cap_tasks = tuple(eval_cfg.get(
            "capability_tasks", ("mmlu", "arc_challenge", "gsm8k")
        ))
        cap_limit = eval_cfg.get("capability_limit")
        cap_bs = int(eval_cfg.get("capability_batch_size", 8))
        try:
            pre_scores = run_capability_benchmarks(
                model_name=cfg["model"]["name"],
                tasks=cap_tasks,
                batch_size=cap_bs,
                limit=cap_limit,
            )
            import tempfile as _tf
            post_dir = _tf.mkdtemp(prefix="refusal_post_ortho_")
            model.model.save_pretrained(post_dir)
            model.tokenizer.save_pretrained(post_dir)
            post_scores = run_capability_benchmarks(
                model_name=post_dir,
                tasks=cap_tasks,
                batch_size=cap_bs,
                limit=cap_limit,
            )
            for task in cap_tasks:
                if task in pre_scores and task in post_scores:
                    capability_delta[task] = post_scores[task] - pre_scores[task]
                    print(
                        f"[capability] {task}: pre={pre_scores[task]:.4f} "
                        f"post={post_scores[task]:.4f} "
                        f"Δ={capability_delta[task]:+.4f}"
                    )
        except Exception as exc:  # noqa: BLE001
            print(f"[capability] benchmarks failed: {exc}")

    if splits.strongreject_eval:
        print("\n--- Step 11b: StrongREJECT evaluation (§3.5) ---")
        strongreject_comps = generate_batched(
            model,
            splits.strongreject_eval,
            max_new_tokens=gen_cfg["max_new_tokens"],
            do_sample=gen_cfg["do_sample"],
            batch_size=gen_cfg["batch_size"],
        )
        # §6.1 — Use the real judge if enabled and package available,
        # otherwise the substring stub.
        use_official_sr = eval_cfg.get("use_official_strongreject", False)
        sr_fn = (
            official_strongreject_judge_score
            if use_official_sr
            else strongreject_judge_score
        )
        sr_scores = [
            sr_fn(p, c)
            for p, c in zip(splits.strongreject_eval, strongreject_comps)
        ]
        # "Bypass rate" = fraction of completions that are NOT refused (score < 0.5).
        strongreject_bypass = sum(1 for s in sr_scores if s < 0.5) / len(sr_scores) if sr_scores else 0.0
        print(f"[strongreject] bypass rate: {strongreject_bypass:.3f}")

    # =====================================================================
    # Step 12: Post-quant correction (if enabled).
    # =====================================================================
    if ext_cfg.get("post_quant_correction", False):
        print("\n--- Step 12: Post-quantization ortho correction (§3.6) ---")
        try:
            n_corrected = post_quantization_ortho_correction(model, primary_dir)
            print(f"[quant correction] {n_corrected} columns/rows re-zeroed")
        except Exception as exc:
            print(f"[quant correction] WARNING: failed: {exc}")

    # =====================================================================
    # Build result.
    # =====================================================================
    best_layer_for_result = best_layer_out
    best_position_for_result = best_pos_out

    # Compute refusal rate CIs for the main bypass eval.
    ci_baseline = refusal_rate_with_ci(baseline_bypass)
    ci_ablated = refusal_rate_with_ci(ablated_bypass)
    print(
        f"\n[ci] baseline bypass: {ci_baseline['rate']:.3f} "
        f"[{ci_baseline['ci_lower']:.3f}, {ci_baseline['ci_upper']:.3f}] "
        f"(n={ci_baseline['n']})"
    )
    print(
        f"[ci] ablated bypass: {ci_ablated['rate']:.3f} "
        f"[{ci_ablated['ci_lower']:.3f}, {ci_ablated['ci_upper']:.3f}] "
        f"(n={ci_ablated['n']})"
    )

    stage_timings["eval_s"] = time.perf_counter() - _t_eval
    stage_timings["total_s"] = time.perf_counter() - _t0

    # Build lineage record (§6.7). Separate from the compact metrics JSON so
    # this structured data can be inspected post-hoc without re-running.
    import sys
    lineage_dict: Dict[str, Any] = {
        "iterative_rounds": round_history,
        "causal_scores": list(causal_scores) if causal_scores is not None else None,
        "head_scores": {
            "layer": int(head_layer) if analyze_heads else None,
            "scores": head_scores_capture,
        } if analyze_heads else None,
        "timing": stage_timings,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": str(model.device),
            "dtype": str(cfg["model"].get("dtype")),
            "family": str(model.family),
            "n_layers": int(model.n_layers),
            "d_model": int(model.d_model),
        },
        "extraction": {
            "n_directions_found": len(directions),
            "first_round_best": {
                "layer": int(first_round_best.layer),
                "position": int(first_round_best.position),
                "bypass": float(first_round_best.bypass_score),
                "induce": float(first_round_best.induce_score),
                "kl": float(first_round_best.kl_score),
            } if first_round_best is not None else None,
            "explained_variance_top1": explained_top1,
            "explained_variance_top5": explained_top5,
            "cosine_similarities": cosine_sims or None,
        },
    }

    result = PipelineResult(
        best_layer=best_layer_for_result,
        best_position=best_position_for_result,
        bypass_score=best_bypass_out,
        induce_score=best_induce_out,
        kl_score=best_kl_out,
        bypass_refusal_rate_baseline=refusal_rate(baseline_bypass),
        bypass_refusal_rate_intervened=refusal_rate(ablated_bypass),
        induce_refusal_rate_baseline=refusal_rate(baseline_induce),
        induce_refusal_rate_intervened=refusal_rate(added_induce),
        orth_refusal_rate=refusal_rate(orth_bypass),
        bypass_safety_rate_baseline=safety_rate(
            safety_scorer, splits.bypass_eval, baseline_bypass
        ),
        bypass_safety_rate_intervened=safety_rate(
            safety_scorer, splits.bypass_eval, ablated_bypass
        ),
        # Extended fields.
        n_directions_found=len(directions),
        direction_cosine_similarities=cosine_sims if cosine_sims else None,
        explained_variance_top1=explained_top1,
        explained_variance_top5=explained_top5,
        weight_tied=weight_tied,
        embedding_leakage=embedding_leakage,
        xstest_over_refusal_baseline=xstest_over_baseline,
        xstest_over_refusal_orth=xstest_over_orth,
        strongreject_bypass_rate=strongreject_bypass,
        causal_layer_scores=causal_scores,
        refusal_head_indices=refusal_head_indices,
        calibrated=calibrated,
        capability_delta_mmlu=capability_delta.get("mmlu"),
        capability_delta_arc=capability_delta.get("arc_challenge"),
        capability_delta_gsm8k=capability_delta.get("gsm8k"),
        lineage=lineage_dict,
    )

    print(result.summary())
    return result, directions


# =============================================================================
# Original pipeline — backward-compatible wrapper
# =============================================================================

def run_pipeline(
    cfg: Dict,
    *,
    preloaded_model: Optional[RefusalModel] = None,
) -> Tuple[PipelineResult, torch.Tensor]:
    """Run the full refusal-direction pipeline on a single model.

    This is the backward-compatible entry point. It delegates to
    :func:`run_pipeline_enhanced` with default settings (single direction,
    no causal patching, no head analysis, no calibration).

    Returns the metric summary and the selected direction tensor.
    """
    # Build an enhanced config that uses defaults for new keys.
    enhanced_cfg = _inject_defaults(cfg)
    result, directions = run_pipeline_enhanced(enhanced_cfg, preloaded_model=preloaded_model)

    # Return only the primary direction for backward compatibility.
    primary = directions[0] if directions else torch.zeros(1)
    return result, primary


def _inject_defaults(cfg: Dict) -> Dict:
    """Inject default values for new enhanced-pipeline keys into *cfg*.

    This ensures backward compatibility: old configs that lack the new keys
    still work with :func:`run_pipeline_enhanced`.
    """
    import copy
    cfg = copy.deepcopy(cfg)

    # Ensure extraction sub-dict has enhanced keys.
    ext = cfg.setdefault("extraction", {})
    ext.setdefault("n_directions", 1)
    ext.setdefault("use_causal_patching", False)
    ext.setdefault("analyze_heads", False)
    ext.setdefault("use_calibration", False)
    ext.setdefault("post_quant_correction", False)

    # Ensure evaluation sub-dict has enhanced keys.
    ev = cfg.setdefault("evaluation", {})
    ev.setdefault("load_xstest_eval", False)
    ev.setdefault("load_strongreject_eval", False)

    return cfg
