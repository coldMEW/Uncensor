from __future__ import annotations

from src.methods import METHOD_REGISTRY, MethodSpec


def test_method_registry_includes_requested_abliteration_research_methods() -> None:
    assert set(METHOD_REGISTRY) >= {
        "basic",
        "svd_multi",
        "cosmic_ranked",
        "layer_weighted",
        "orthogonalized",
    }
    assert all(isinstance(spec, MethodSpec) for spec in METHOD_REGISTRY.values())


def test_method_registry_defaults_avoid_unbounded_all_layer_projection() -> None:
    assert all(not spec.include_final_norm_by_default for spec in METHOD_REGISTRY.values())
    assert all(spec.default_layer_radius <= 4 for spec in METHOD_REGISTRY.values())
