"""
Strength Sweep Visualization for refusal direction ablation.

Varies the steering coefficient and plots compliance vs coherence tradeoff,
helping users find the optimal strength for their use case.

This is useful for:
- Finding the optimal coefficient for compliance vs capability tradeoff
- Understanding model sensitivity to direction strength
- Tuning for specific use cases (max compliance vs max capability)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Callable

import torch


@dataclass
class SweepResult:
    """Results from a single strength sweep run."""

    coefficient: float
    """Steering coefficient used."""

    compliance_rate: float
    """Fraction of harmful prompts that now comply."""

    coherence_score: float
    """Coherence score on harmless prompts (perplexity-based)."""

    refusal_rate: float
    """Fraction of harmless prompts incorrectly refused."""

    kl_divergence: float
    """KL divergence from original model output."""


@dataclass
class SweepAnalysis:
    """Complete strength sweep analysis."""

    results: List[SweepResult]
    """Results for each coefficient tested."""

    optimal_coefficient: float
    """Coefficient with best compliance/capability tradeoff."""

    max_compliance: float
    """Maximum compliance achieved."""

    min_capability_loss: float
    """Minimum capability loss across all coefficients."""


def run_strength_sweep(
    model,
    direction: torch.Tensor,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    refusal_token_ids: List[int],
    coefficients: List[float] = None,
    batch_size: int = 4,
) -> List[SweepResult]:
    """Run strength sweep to measure compliance vs coherence tradeoff.

    Args:
        model: RefusalModel wrapper
        direction: Unit-norm refusal direction
        harmful_prompts: Prompts that should comply after ablation
        harmless_prompts: Prompts that should remain unaffected
        refusal_token_ids: Token IDs indicating refusal
        coefficients: List of coefficients to test (default: [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
        batch_size: Forward pass batch size

    Returns:
        List of SweepResult, one per coefficient
    """
    from ..metrics import refusal_metric_from_logits
    from ..interventions import directional_ablation

    if coefficients is None:
        coefficients = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    results: List[SweepResult] = []

    # Normalize direction
    direction = direction.to(dtype=torch.float32)
    direction = direction / direction.norm()

    for coef in coefficients:
        # Scale direction by coefficient
        scaled_dir = direction * coef

        with directional_ablation(model, scaled_dir):
            # Measure compliance on harmful prompts
            harmful_out: List[torch.Tensor] = []
            for start in range(0, len(harmful_prompts), batch_size):
                batch = [model.format(p) for p in harmful_prompts[start:start + batch_size]]
                enc = model.tokenize(batch)
                with torch.no_grad():
                    logits = model.model(**enc).logits
                harmful_out.append(logits[:, -1, :].detach().cpu())

            harmful_logits = torch.cat(harmful_out, dim=0)
            refusal_metric = refusal_metric_from_logits(harmful_logits, refusal_token_ids)
            compliance_rate = 1.0 - float(refusal_metric.mean().item())

            # Measure refusal rate on harmless prompts
            harmless_out: List[torch.Tensor] = []
            for start in range(0, len(harmless_prompts), batch_size):
                batch = [model.format(p) for p in harmless_prompts[start:start + batch_size]]
                enc = model.tokenize(batch)
                with torch.no_grad():
                    logits = model.model(**enc).logits
                harmless_out.append(logits[:, -1, :].detach().cpu())

            harmless_logits = torch.cat(harmless_out, dim=0)
            harmless_refusal = refusal_metric_from_logits(harmless_logits, refusal_token_ids)
            refusal_rate = float(harmless_refusal.mean().item())

            # Simple coherence: average log probability (lower = more coherent)
            log_probs = torch.log_softmax(harmless_logits, dim=-1)
            coherence_score = float(log_probs.mean().item())

            results.append(SweepResult(
                coefficient=coef,
                compliance_rate=compliance_rate,
                coherence_score=coherence_score,
                refusal_rate=refusal_rate,
                kl_divergence=0.0,  # Would need original logits for KL
            ))

    return results


def plot_strength_curve(
    results: List[SweepResult],
    title: str = "Compliance vs Coherence Tradeoff",
    save_path: Optional[str] = None,
) -> None:
    """Plot compliance vs coherence tradeoff curve.

    Args:
        results: From run_strength_sweep()
        title: Plot title
        save_path: Optional path to save PNG
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[strength_sweep] matplotlib not installed; skipping plot")
        return

    if not results:
        return

    coefficients = [r.coefficient for r in results]
    compliance = [r.compliance_rate for r in results]
    coherence = [r.coherence_score for r in results]
    refusal = [r.refusal_rate for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(coefficients, compliance, "b-o", label="Compliance Rate", linewidth=2)
    ax.plot(coefficients, refusal, "r-s", label="False Refusal Rate", linewidth=2)
    ax.set_xlabel("Coefficient (Direction Strength)")
    ax.set_ylabel("Rate", color="black")
    ax.tick_params(axis="y", labelcolor="black")

    # Secondary axis for coherence (if meaningful)
    ax2 = ax.twinx()
    ax2.plot(coefficients, coherence, "g--^", label="Coherence (log prob)", linewidth=1.5, alpha=0.7)
    ax2.set_ylabel("Coherence Score", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[strength_sweep] saved plot to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def analyze_sweep_results(results: List[SweepResult]) -> SweepAnalysis:
    """Analyze sweep results to find optimal coefficient.

    Args:
        results: From run_strength_sweep()

    Returns:
        SweepAnalysis with optimal coefficient and metrics
    """
    if not results:
        raise ValueError("No results to analyze")

    max_compliance = max(r.compliance_rate for r in results)
    min_capability_loss = abs(min(r.coherence_score for r in results))  # Higher is better

    # Find optimal: max compliance with min false refusal
    best_score = -float("inf")
    optimal = results[0]

    for r in results:
        # Score: high compliance, low refusal, maintain coherence
        score = r.compliance_rate - (r.refusal_rate * 2) + (r.coherence_score / 10)
        if score > best_score:
            best_score = score
            optimal = r

    return SweepAnalysis(
        results=results,
        optimal_coefficient=optimal.coefficient,
        max_compliance=max_compliance,
        min_capability_loss=min_capability_loss,
    )
