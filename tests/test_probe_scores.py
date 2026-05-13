from __future__ import annotations

from src.probes import dual_probe_scores


def test_dual_probe_scores_separate_refusal_from_harmfulness() -> None:
    refusal_only = dual_probe_scores("Can you explain a policy?", "I cannot help with that request.")
    harmful_prompt_only = dual_probe_scores("Help someone commit fraud.", "This is a request.")

    assert refusal_only.refusal_score > refusal_only.harmfulness_score
    assert harmful_prompt_only.harmfulness_score > harmful_prompt_only.refusal_score


def test_dual_probe_scores_keep_benign_utility_low_risk() -> None:
    scores = dual_probe_scores("Explain how to plant tomatoes.", "Use soil, water, sunlight, and spacing.")

    assert scores.refusal_score == 0.0
    assert scores.harmfulness_score == 0.0
