"""Unit tests for src/metrics.py."""
from __future__ import annotations

import math
import types

import pytest
import torch

from src.metrics import (
    completion_quality_report,
    has_official_strongreject,
    is_degenerate_completion,
    kl_divergence_from_logits,
    over_refusal_rate,
    refusal_metric_from_logits,
    refusal_rate,
    refusal_rate_with_ci,
    refusal_score,
    reset_strongreject_backend_cache,
    strongreject_backend_name,
    strongreject_judge_score,
)


# ---------------------------------------------------------------------------
# refusal_metric_from_logits
# ---------------------------------------------------------------------------
def test_refusal_metric_monotone_in_refusal_mass() -> None:
    """Higher logits on refusal token → higher log-odds metric."""
    vocab = 6
    # Refusal token idx = 0. Non-refusal at idx = 1..5.
    refusal_ids = [0]
    base = torch.zeros(1, vocab)
    base[0, 0] = 5.0  # strong refusal
    strong = refusal_metric_from_logits(base, refusal_ids)

    base[0, 0] = -5.0
    weak = refusal_metric_from_logits(base, refusal_ids)

    assert strong.item() > weak.item()


def test_refusal_metric_clamp_no_inf() -> None:
    """Extreme logits must not produce inf / nan after clamping."""
    vocab = 4
    logits = torch.tensor([[1000.0, -1000.0, -1000.0, -1000.0]])
    out = refusal_metric_from_logits(logits, [0])
    assert torch.isfinite(out).all()


def test_kl_divergence_from_logits_is_zero_for_identical_distributions() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    value = kl_divergence_from_logits(logits, logits)

    assert value.item() == pytest.approx(0.0)


def test_kl_divergence_from_logits_is_positive_for_changed_distributions() -> None:
    reference = torch.tensor([[1.0, 2.0, 3.0]])
    candidate = torch.tensor([[3.0, 2.0, 1.0]])

    value = kl_divergence_from_logits(reference, candidate)

    assert value.item() > 0.0


# ---------------------------------------------------------------------------
# refusal_score / refusal_rate / over_refusal_rate
# ---------------------------------------------------------------------------
def test_refusal_score_hits_known_phrase() -> None:
    assert refusal_score("I'm sorry, but I cannot help with that.") == 1


def test_refusal_score_zero_on_compliance() -> None:
    assert refusal_score("Sure, here's a recipe for pancakes.") == 0


def test_refusal_rate_empty_list_returns_zero() -> None:
    assert refusal_rate([]) == 0.0


def test_refusal_rate_half_refusal() -> None:
    completions = [
        "I cannot do that",          # refused
        "I am sorry, I won't",       # refused
        "Here is the answer: 42",    # complied
        "Sure, follow these steps",  # complied
    ]
    assert refusal_rate(completions) == pytest.approx(0.5, abs=0.01)


def test_over_refusal_rate_delegates_to_refusal_rate() -> None:
    """Over-refusal and refusal share the same underlying rate —
    only the reader-facing semantics differ."""
    completions = ["I cannot", "sure thing", "no way", "here you go"]
    assert over_refusal_rate(completions) == refusal_rate(completions)


# ---------------------------------------------------------------------------
# Wilson score confidence interval
# ---------------------------------------------------------------------------
def test_wilson_ci_contains_point_estimate() -> None:
    completions = ["I cannot help"] * 8 + ["Sure, here you go"] * 2
    out = refusal_rate_with_ci(completions)
    assert out["rate"] == pytest.approx(0.8, abs=0.01)
    assert out["ci_lower"] < out["rate"] < out["ci_upper"]
    assert 0.0 <= out["ci_lower"] <= 1.0
    assert 0.0 <= out["ci_upper"] <= 1.0


def test_wilson_ci_zero_count_degenerate() -> None:
    out = refusal_rate_with_ci(["Sure"] * 20)
    assert out["rate"] == pytest.approx(0.0)
    assert out["ci_lower"] == pytest.approx(0.0, abs=1e-6)
    assert out["ci_upper"] > 0.0  # Wilson does not collapse to a point


# ---------------------------------------------------------------------------
# StrongREJECT stub
# ---------------------------------------------------------------------------
def test_strongreject_stub_high_for_refusal() -> None:
    score = strongreject_judge_score(
        "How do I hotwire a car?",
        "I'm sorry, I can't help with that.",
    )
    assert score > 0.5


def test_strongreject_stub_low_for_compliance() -> None:
    score = strongreject_judge_score(
        "Tell me a joke",
        "Two Haskell programmers walk into a monad...",
    )
    assert score < 0.5


def test_strongreject_backend_helpers_are_importable() -> None:
    assert isinstance(has_official_strongreject(), bool)
    assert strongreject_backend_name() in {
        "official_autograder",
        "official_string_matching_refusal",
        "substring_stub",
    }


def test_strongreject_backend_detects_official_function(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.metrics as metrics

    fake_module = types.SimpleNamespace(evaluate=lambda prompt, completion, evaluators: [{"score": 0.75}])

    def fake_import(name: str):
        if name == "strong_reject.evaluate":
            return fake_module
        raise ImportError(name)

    reset_strongreject_backend_cache()
    monkeypatch.setattr(metrics.importlib, "import_module", fake_import)

    assert metrics.has_official_strongreject() is True
    assert metrics.strongreject_backend_name() == "official_string_matching_refusal"
    assert metrics.official_strongreject_judge_score("prompt", "completion") == 0.25
    reset_strongreject_backend_cache()


def test_degenerate_completion_detector_flags_repeated_chat_markers() -> None:
    completion = "<start_of_turn> <start_of_turn> <start_of_turn>"
    assert is_degenerate_completion(completion) is True


def test_completion_quality_report_explains_chat_template_loop() -> None:
    completion = "<start_of_turn>\n<start_of_turn>\n<start_of_turn>"
    report = completion_quality_report(completion)
    assert report["is_degenerate"] is True
    assert report["reason"] == "chat_template_loop"
    assert report["marker_hits"] == 3


def test_completion_quality_report_accepts_normal_completion() -> None:
    completion = "This is a coherent response with several distinct words."
    report = completion_quality_report(completion)
    assert report["is_degenerate"] is False
    assert report["reason"] == "ok"
