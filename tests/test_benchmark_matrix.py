from __future__ import annotations

from src.benchmark_matrix import build_benchmark_matrix, matrix_counts, matrix_is_dataset_scale


def test_benchmark_matrix_separates_safety_overrefusal_and_utility() -> None:
    matrix = build_benchmark_matrix(
        refusal_probe_count=128,
        benign_control_count=100,
        xstest_count=250,
        strongreject_count=100,
        jailbreakbench_count=100,
        harmbench_count=100,
        utility_count=32,
        judge_backend="official_strongreject",
        judge_is_verified=True,
    )

    assert matrix["judge"]["verified"] is True
    assert matrix["safety_retention"]["strongreject"]["count"] == 100
    assert matrix["safety_retention"]["harmbench"]["count"] == 100
    assert matrix["over_refusal"]["xstest"]["count"] == 250
    assert matrix["utility"]["general_utility"]["count"] == 32


def test_matrix_is_dataset_scale_requires_verified_judge_and_100_prompt_gates() -> None:
    counts = matrix_counts(
        refusal_probe_count=100,
        benign_control_count=100,
        xstest_count=100,
        strongreject_count=100,
        jailbreakbench_count=100,
        harmbench_count=100,
        utility_count=0,
    )

    assert matrix_is_dataset_scale(counts, judge_is_verified=True) is True
    assert matrix_is_dataset_scale(counts, judge_is_verified=False) is False
    assert matrix_is_dataset_scale({**counts, "benign_control": 99}, judge_is_verified=True) is False
