"""Method registry for refusal-direction intervention experiments.

The registry keeps notebook and CLI search spaces explicit. It does not claim
that every method is fully implemented in every runtime; it records safe
defaults and intended intervention families so unsupported methods can be
reported transparently instead of silently falling back to all-layer edits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class MethodSpec:
    name: str
    direction_family: str
    intervention_type: str
    default_layer_radius: int
    default_direction_count: int
    include_final_norm_by_default: bool = False
    description: str = ""
    supported_on_kaggle: bool = True


METHOD_REGISTRY: Dict[str, MethodSpec] = {
    "basic": MethodSpec(
        name="basic",
        direction_family="dim",
        intervention_type="hook_ablation",
        default_layer_radius=1,
        default_direction_count=1,
        description="Single direction difference-in-means baseline.",
    ),
    "svd_multi": MethodSpec(
        name="svd_multi",
        direction_family="svd_multi",
        intervention_type="hook_ablation",
        default_layer_radius=2,
        default_direction_count=3,
        description="Robust multi-direction SVD basis over contrastive activations.",
    ),
    "cosmic_ranked": MethodSpec(
        name="cosmic_ranked",
        direction_family="cosine_ranked",
        intervention_type="hook_ablation",
        default_layer_radius=2,
        default_direction_count=3,
        description="Layer ranking by contrastive cosine separation before candidate search.",
    ),
    "layer_weighted": MethodSpec(
        name="layer_weighted",
        direction_family="svd_multi",
        intervention_type="layer_weighted_hook_ablation",
        default_layer_radius=4,
        default_direction_count=3,
        description="Layer-window ablation with per-layer weights from validation ranking.",
    ),
    "orthogonalized": MethodSpec(
        name="orthogonalized",
        direction_family="svd_multi",
        intervention_type="weight_orthogonalization",
        default_layer_radius=2,
        default_direction_count=3,
        description="Rank-1 residual-writer weight orthogonalization export path.",
        supported_on_kaggle=False,
    ),
}


def kaggle_supported_methods() -> Tuple[MethodSpec, ...]:
    return tuple(spec for spec in METHOD_REGISTRY.values() if spec.supported_on_kaggle)


def build_method_search_space(
    *,
    layer_windows: Mapping[str, Sequence[int]],
    coefficients: Sequence[float],
    methods: Sequence[MethodSpec] | None = None,
) -> Tuple[Dict[str, Any], ...]:
    """Build a deterministic candidate grid from registered methods.

    The method registry is the single source of truth for Kaggle robust mode.
    Unsupported methods stay visible in ``METHOD_REGISTRY`` for export/offline
    use, but they are intentionally excluded from this reversible search grid.
    """
    active_methods = tuple(methods or kaggle_supported_methods())
    candidates = []
    candidate_index = 0
    for spec in active_methods:
        for layer_window_name, layer_indices in layer_windows.items():
            for coefficient in coefficients:
                candidate_index += 1
                candidates.append(
                    {
                        "candidate_id": f"m{candidate_index:04d}",
                        "method_name": spec.name,
                        "direction_family": spec.direction_family,
                        "direction_count": int(spec.default_direction_count),
                        "layer_window_name": str(layer_window_name),
                        "layer_indices": [int(idx) for idx in layer_indices],
                        "coefficient": float(coefficient),
                        "intervention_type": spec.intervention_type,
                        "include_final_norm": bool(spec.include_final_norm_by_default),
                    }
                )
    return tuple(candidates)
