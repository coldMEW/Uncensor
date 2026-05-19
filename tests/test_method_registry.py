from __future__ import annotations

from src.methods import METHOD_REGISTRY, MethodSpec, build_method_search_space, kaggle_supported_methods


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


def test_kaggle_supported_methods_exclude_permanent_weight_edits() -> None:
    names = {spec.name for spec in kaggle_supported_methods()}
    assert {"basic", "svd_multi", "cosmic_ranked", "layer_weighted"} <= names
    assert "orthogonalized" not in names


def test_method_search_space_builds_reversible_registered_candidates() -> None:
    candidates = build_method_search_space(
        layer_windows={"core": [31, 32, 33]},
        coefficients=[0.1, 0.2],
    )

    assert len(candidates) == 8
    assert {candidate["candidate_id"] for candidate in candidates} == {
        f"m{i:04d}" for i in range(1, 9)
    }
    assert {candidate["method_name"] for candidate in candidates} >= {
        "basic",
        "svd_multi",
        "cosmic_ranked",
        "layer_weighted",
    }
    assert all(candidate["intervention_type"] != "weight_orthogonalization" for candidate in candidates)
