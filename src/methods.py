"""Method registry for refusal-direction intervention experiments.

The registry keeps notebook and CLI search spaces explicit. It does not claim
that every method is fully implemented in every runtime; it records safe
defaults and intended intervention families so unsupported methods can be
reported transparently instead of silently falling back to all-layer edits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


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


def select_diverse_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    max_candidates: int,
    preferred_layer_window_names: Sequence[str] = (),
    preferred_coefficients: Sequence[float] = (),
    preferred_direction_counts: Sequence[int] = (),
) -> Tuple[Dict[str, Any], ...]:
    """Select a bounded candidate set without collapsing to one method.

    The raw grid is deterministic but grouped by method, window, and
    coefficient.  A naive ``grid[:N]`` can therefore spend the whole Kaggle
    budget on near-duplicate baseline candidates.  This selector does a stable
    round-robin over method names so robust mode exercises each registered
    strategy before repeating any one strategy.  Optional preferences reorder
    each method bucket without changing method diversity; this lets a run feed
    back observed useful layer windows or coefficient ranges into candidate
    triage.
    """
    if max_candidates <= 0:
        return tuple()

    preferred_windows = tuple(str(name) for name in preferred_layer_window_names)
    preferred_coeffs = tuple(float(value) for value in preferred_coefficients)
    preferred_counts = tuple(int(value) for value in preferred_direction_counts)

    def preference_rank(candidate: Mapping[str, Any]) -> tuple[float, float, float]:
        layer_name = str(candidate.get("layer_window_name", ""))
        direction_count = int(candidate.get("direction_count", 0))
        coefficient = float(candidate.get("coefficient", 0.0))
        layer_rank = (
            preferred_windows.index(layer_name)
            if layer_name in preferred_windows
            else len(preferred_windows)
        )
        count_rank = (
            preferred_counts.index(direction_count)
            if direction_count in preferred_counts
            else len(preferred_counts)
        )
        if preferred_coeffs:
            coeff_rank = min(abs(coefficient - target) for target in preferred_coeffs)
        else:
            coeff_rank = 0.0
        return (float(layer_rank), float(count_rank), float(coeff_rank))

    grouped: Dict[str, list[Dict[str, Any]]] = {}
    method_order: list[str] = []
    for candidate in candidates:
        item = dict(candidate)
        method_name = str(item.get("method_name", "unknown"))
        if method_name not in grouped:
            grouped[method_name] = []
            method_order.append(method_name)
        grouped[method_name].append(item)

    if preferred_windows or preferred_coeffs or preferred_counts:
        for bucket in grouped.values():
            bucket.sort(key=preference_rank)

    selected: list[Dict[str, Any]] = []
    while len(selected) < max_candidates:
        added_this_round = False
        for method_name in method_order:
            bucket = grouped[method_name]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            added_this_round = True
            if len(selected) >= max_candidates:
                break
        if not added_this_round:
            break
    return tuple(selected)
