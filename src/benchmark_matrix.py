"""Structured benchmark coverage metadata for refusal-calibration runs."""

from __future__ import annotations

from typing import Any, Dict, Mapping


DATASET_SCALE_MINIMUM = 100


def matrix_counts(
    *,
    refusal_probe_count: int,
    benign_control_count: int,
    xstest_count: int = 0,
    strongreject_count: int = 0,
    jailbreakbench_count: int = 0,
    harmbench_count: int = 0,
    utility_count: int = 0,
) -> Dict[str, int]:
    return {
        "refusal_probe": int(refusal_probe_count),
        "benign_control": int(benign_control_count),
        "xstest": int(xstest_count),
        "strongreject": int(strongreject_count),
        "jailbreakbench": int(jailbreakbench_count),
        "harmbench": int(harmbench_count),
        "utility": int(utility_count),
    }


def matrix_is_dataset_scale(counts: Mapping[str, int], *, judge_is_verified: bool) -> bool:
    """Return whether the run has enough verified evidence for validity claims."""
    return bool(judge_is_verified) and not missing_dataset_scale_requirements(counts)


def missing_dataset_scale_requirements(counts: Mapping[str, int]) -> Dict[str, Dict[str, int]]:
    required_keys = (
        "refusal_probe",
        "benign_control",
        "strongreject",
        "jailbreakbench",
        "harmbench",
        "xstest",
    )
    return {
        key: {"count": int(counts.get(key, 0)), "minimum": DATASET_SCALE_MINIMUM}
        for key in required_keys
        if int(counts.get(key, 0)) < DATASET_SCALE_MINIMUM
    }


def build_metric_evidence_matrix(
    *,
    refusal_probe_count: int,
    benign_control_count: int,
    judge_backend: str,
    judge_is_verified: bool,
) -> Dict[str, Any]:
    """Describe the actual prompt counts behind headline metric values."""
    counts = {
        "refusal_probe": int(refusal_probe_count),
        "benign_control": int(benign_control_count),
    }
    missing = {
        key: {"count": int(counts.get(key, 0)), "minimum": DATASET_SCALE_MINIMUM}
        for key in ("refusal_probe", "benign_control")
        if int(counts.get(key, 0)) < DATASET_SCALE_MINIMUM
    }
    return {
        "schema_version": 1,
        "dataset_scale_minimum": DATASET_SCALE_MINIMUM,
        "dataset_scale": bool(judge_is_verified) and not missing,
        "dataset_scale_missing_requirements": missing,
        "metric_counts": counts,
        "judge": {
            "backend": str(judge_backend),
            "verified": bool(judge_is_verified),
            "status": "OFFICIAL_JUDGE" if judge_is_verified else "UNVERIFIED_JUDGE",
        },
    }


def build_benchmark_matrix(
    *,
    refusal_probe_count: int,
    benign_control_count: int,
    xstest_count: int = 0,
    strongreject_count: int = 0,
    jailbreakbench_count: int = 0,
    harmbench_count: int = 0,
    utility_count: int = 0,
    judge_backend: str,
    judge_is_verified: bool,
) -> Dict[str, Any]:
    counts = matrix_counts(
        refusal_probe_count=refusal_probe_count,
        benign_control_count=benign_control_count,
        xstest_count=xstest_count,
        strongreject_count=strongreject_count,
        jailbreakbench_count=jailbreakbench_count,
        harmbench_count=harmbench_count,
        utility_count=utility_count,
    )
    return {
        "schema_version": 1,
        "dataset_scale_minimum": DATASET_SCALE_MINIMUM,
        "dataset_scale": matrix_is_dataset_scale(counts, judge_is_verified=judge_is_verified),
        "dataset_scale_missing_requirements": missing_dataset_scale_requirements(counts),
        "judge": {
            "backend": str(judge_backend),
            "verified": bool(judge_is_verified),
            "status": "OFFICIAL_JUDGE" if judge_is_verified else "UNVERIFIED_JUDGE",
        },
        "safety_retention": {
            "refusal_probe": {"count": counts["refusal_probe"]},
            "strongreject": {"count": counts["strongreject"]},
            "jailbreakbench": {"count": counts["jailbreakbench"]},
            "harmbench": {"count": counts["harmbench"]},
        },
        "over_refusal": {
            "benign_controls": {"count": counts["benign_control"]},
            "xstest": {"count": counts["xstest"]},
        },
        "utility": {
            "general_utility": {"count": counts["utility"]},
        },
    }
