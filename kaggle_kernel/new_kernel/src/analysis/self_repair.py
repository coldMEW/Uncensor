"""
Self-repair (Ouroboros) detection and mitigation for refusal direction ablation.

After weight orthogonalization, some models partially recover their refusal
behavior through self-repair: other components re-express the refusal direction
to compensate for the ablated components. This module detects whether this
has occurred and offers a mitigation pass.

Method:
1. After orthogonalization, run a fresh DiM extraction on the modified model.
2. If the new dominant direction is highly aligned with the original (cosine > threshold),
   self-repair is occurring.
3. Mitigation: orthogonalize the modified model against the re-grown direction.

Reference: OBLITERATUS "Ouroboros/self-repair prediction" module.
           Arditi et al. (2024) §7 Discussion — limitation of single-round ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


# =============================================================================
# Report dataclass
# =============================================================================

@dataclass
class SelfRepairReport:
    """Results of self-repair detection analysis."""

    detected: bool
    """True if self-repair was detected."""

    regrowth_cosine: float
    """Cosine similarity of re-grown direction with original."""

    regrowth_bypass_score: float
    """Bypass score of re-grown direction (lower = stronger refusal recovery)."""

    regrowth_layer: int
    """Layer where re-grown direction is strongest."""

    regrowth_direction: Optional[torch.Tensor]
    """The re-grown direction vector (unit-norm, float32, CPU)."""

    detection_threshold: float
    """Cosine threshold used for detection."""

    recommendation: str
    """Human-readable recommendation."""


# =============================================================================
# Detection
# =============================================================================

def detect_self_repair(
    model,
    original_direction: torch.Tensor,
    harmful_train: List[str],
    harmless_train: List[str],
    token_positions: List[int],
    batch_size: int,
    refusal_token_ids: List[int],
    harmful_val: List[str],
    harmless_val: List[str],
    cosine_threshold: float = 0.5,
) -> SelfRepairReport:
    """Detect whether the model has re-grown a refusal direction after orthogonalization.

    Runs a fresh DiM extraction on the modified model and checks whether any
    extracted direction is cosine-similar to the original ablated direction.

    This is more conservative than OBLITERATUS's Ouroboros module: we check
    cosine similarity to the *original* direction rather than just checking
    for any new strong direction, avoiding false positives from unrelated
    directions becoming more prominent.

    Args:
        model: RefusalModel (already orthogonalized).
        original_direction: The direction that was orthogonalized out.
        harmful_train: Harmful prompts for activation collection.
        harmless_train: Harmless prompts for activation collection.
        token_positions: Post-instruction token positions (negative indices).
        batch_size: Forward-pass batch size.
        refusal_token_ids: Token IDs for the refusal metric.
        harmful_val: Harmful validation prompts for bypass scoring.
        harmless_val: Harmless validation prompts for bypass scoring.
        cosine_threshold: Cosine similarity threshold above which self-repair
            is declared (default 0.5).

    Returns:
        SelfRepairReport with detection result and diagnostics.
    """
    # Lazy imports to avoid circular imports — both modules are fully loaded
    # by the time this function is called.
    from ..extraction import collect_activations, difference_in_means
    from ..interventions import directional_ablation

    # ------------------------------------------------------------------
    # 1. Collect fresh activations from the already-modified model.
    # ------------------------------------------------------------------
    print("[self-repair] collecting activations on modified model ...")
    harmful_acts = collect_activations(model, harmful_train, token_positions, batch_size)
    harmless_acts = collect_activations(model, harmless_train, token_positions, batch_size)

    # ------------------------------------------------------------------
    # 2. DiM to get per-(layer, position) candidate directions.
    #    Shape: (n_layers, n_positions, d_model)
    # ------------------------------------------------------------------
    candidates = difference_in_means(harmful_acts, harmless_acts)  # (L, P, d_model)

    n_layers, n_positions, d_model = candidates.shape

    # Unit-normalize the original direction (CPU, float32).
    orig = original_direction.to(dtype=torch.float32, device="cpu")
    orig_norm = orig.norm()
    if orig_norm < 1e-8:
        # Degenerate original direction — cannot detect anything meaningful.
        return SelfRepairReport(
            detected=False,
            regrowth_cosine=0.0,
            regrowth_bypass_score=0.0,
            regrowth_layer=0,
            regrowth_direction=None,
            detection_threshold=cosine_threshold,
            recommendation="No self-repair detected. Single orthogonalization pass is sufficient.",
        )
    orig_hat = orig / orig_norm

    # ------------------------------------------------------------------
    # 3. Find the candidate most aligned with the original direction.
    #    We search all (layer, position) pairs and pick the one with the
    #    highest |cosine| similarity.
    # ------------------------------------------------------------------
    best_cosine = 0.0
    best_layer = 0
    best_pos_idx = 0
    best_vec: Optional[torch.Tensor] = None

    for l in range(n_layers):
        for p in range(n_positions):
            vec = candidates[l, p].to(dtype=torch.float32, device="cpu")
            vec_norm = vec.norm()
            if vec_norm < 1e-8:
                continue
            vec_hat = vec / vec_norm
            cos = float(torch.dot(orig_hat, vec_hat).item())
            if abs(cos) > abs(best_cosine):
                best_cosine = cos
                best_layer = l
                best_pos_idx = p
                best_vec = vec_hat.clone()

    # Orient best_vec to have positive cosine with orig (sign convention).
    if best_vec is not None and best_cosine < 0.0:
        best_vec = -best_vec
        best_cosine = -best_cosine

    # ------------------------------------------------------------------
    # 4. Compute bypass score for the re-grown direction (if above threshold).
    #    Lower bypass_score = the direction captures stronger refusal signal.
    # ------------------------------------------------------------------
    bypass_score = 0.0
    if best_vec is not None and best_cosine >= cosine_threshold:
        print(
            f"[self-repair] candidate at layer={best_layer} pos_idx={best_pos_idx} "
            f"has cosine={best_cosine:.4f} >= threshold={cosine_threshold:.2f}. "
            "Computing bypass score ..."
        )
        # Import metrics helper inline (no circular risk — metrics has no
        # dependencies on extraction/interventions).
        from ..metrics import refusal_metric_from_logits

        # Reuse the private logits helper from extraction (already available
        # there as a local reimplementation safe from circular imports).
        from ..extraction import collect_activations as _ca  # noqa: F401

        # Build a small local logits function mirroring extraction's helper.
        def _logits_at_last_token(prompts: List[str]) -> torch.Tensor:
            out: List[torch.Tensor] = []
            for start in range(0, len(prompts), batch_size):
                batch = [model.format(p) for p in prompts[start : start + batch_size]]
                enc = model.tokenize(batch)
                with torch.no_grad():
                    logits = model.model(**enc).logits
                out.append(logits[:, -1, :].detach().cpu())
            return torch.cat(out, dim=0)

        with directional_ablation(model, best_vec):
            logits = _logits_at_last_token(harmful_val)
        metric = refusal_metric_from_logits(logits, refusal_token_ids)
        bypass_score = float(metric.mean().item())
        print(f"[self-repair] re-grown direction bypass_score={bypass_score:.4f}")

    # ------------------------------------------------------------------
    # 5. Build report.
    # ------------------------------------------------------------------
    detected = best_cosine >= cosine_threshold

    if not detected:
        recommendation = (
            "No self-repair detected. Single orthogonalization pass is sufficient."
        )
    elif best_cosine > 0.7:
        recommendation = (
            f"Strong self-repair (cos={best_cosine:.2f}). "
            "Run mitigate_self_repair() or use nuclear preset."
        )
    else:
        recommendation = (
            f"Moderate self-repair (cos={best_cosine:.2f}). "
            "Consider a second orthogonalization pass."
        )

    print(
        f"[self-repair] detection result: detected={detected}, "
        f"cosine={best_cosine:.4f}, layer={best_layer}, "
        f"bypass={bypass_score:.4f}"
    )

    return SelfRepairReport(
        detected=detected,
        regrowth_cosine=best_cosine,
        regrowth_bypass_score=bypass_score,
        regrowth_layer=best_layer,
        regrowth_direction=best_vec,
        detection_threshold=cosine_threshold,
        recommendation=recommendation,
    )


# =============================================================================
# Mitigation
# =============================================================================

def mitigate_self_repair(
    model,
    report: SelfRepairReport,
) -> bool:
    """Apply a second orthogonalization pass targeting the re-grown direction.

    Modifies model weights in-place.

    Args:
        model: RefusalModel (already orthogonalized).
        report: From detect_self_repair().

    Returns:
        True if mitigation was applied, False if no self-repair was detected
        (report.detected == False) or re-grown direction is None.
    """
    if not report.detected:
        print("[self-repair mitigation] no self-repair detected — skipping")
        return False

    if report.regrowth_direction is None:
        print(
            "[self-repair mitigation] regrowth_direction is None — skipping"
        )
        return False

    # Lazy import to avoid circular imports.
    from ..interventions import orthogonalize_weights

    print(
        f"[self-repair mitigation] applying second orthogonalization pass "
        f"(layer={report.regrowth_layer}, cosine={report.regrowth_cosine:.4f}) ..."
    )
    orthogonalize_weights(model, report.regrowth_direction)
    print("[self-repair mitigation] done")
    return True


# =============================================================================
# Combined convenience entry-point
# =============================================================================

def full_self_repair_analysis(
    model,
    original_direction: torch.Tensor,
    harmful_train: List[str],
    harmless_train: List[str],
    token_positions: List[int],
    batch_size: int,
    refusal_token_ids: List[int],
    harmful_val: List[str],
    harmless_val: List[str],
    auto_mitigate: bool = True,
    cosine_threshold: float = 0.5,
) -> SelfRepairReport:
    """Detect and optionally mitigate self-repair in one call.

    If auto_mitigate=True and self-repair is detected, calls
    mitigate_self_repair() and re-runs detection to verify mitigation success.

    Args:
        model: RefusalModel (already orthogonalized).
        original_direction: The direction that was orthogonalized out.
        harmful_train: Harmful prompts for activation collection.
        harmless_train: Harmless prompts for activation collection.
        token_positions: Post-instruction token positions (negative indices).
        batch_size: Forward-pass batch size.
        refusal_token_ids: Token IDs for the refusal metric.
        harmful_val: Harmful validation prompts for bypass scoring.
        harmless_val: Harmless validation prompts for bypass scoring.
        auto_mitigate: If True and self-repair is detected, automatically
            applies mitigate_self_repair() and re-runs detection.
        cosine_threshold: Cosine similarity threshold for detection.

    Returns:
        SelfRepairReport from the final detection run (post-mitigation if
        mitigation was applied).
    """
    _shared_kwargs = dict(
        original_direction=original_direction,
        harmful_train=harmful_train,
        harmless_train=harmless_train,
        token_positions=token_positions,
        batch_size=batch_size,
        refusal_token_ids=refusal_token_ids,
        harmful_val=harmful_val,
        harmless_val=harmless_val,
        cosine_threshold=cosine_threshold,
    )

    print("[self-repair] === detection pass ===")
    report = detect_self_repair(model=model, **_shared_kwargs)

    if not report.detected or not auto_mitigate:
        return report

    print("[self-repair] === mitigation pass ===")
    applied = mitigate_self_repair(model, report)

    if applied:
        print("[self-repair] === verification pass (post-mitigation) ===")
        report = detect_self_repair(model=model, **_shared_kwargs)
        if report.detected:
            print(
                f"[self-repair] WARNING: self-repair persists after mitigation "
                f"(cosine={report.regrowth_cosine:.4f}). "
                "Consider using the nuclear preset or additional ablation rounds."
            )
        else:
            print("[self-repair] mitigation successful — self-repair no longer detected")

    return report
