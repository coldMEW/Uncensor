from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

import src.pipeline as pipeline
from src.pipeline import PipelineResult


def _minimal_result(*, intervened: float = 0.2, kl: float = 0.01) -> PipelineResult:
    return PipelineResult(
        best_layer=1,
        best_position=-1,
        bypass_score=intervened,
        induce_score=0.0,
        kl_score=kl,
        bypass_refusal_rate_baseline=0.8,
        bypass_refusal_rate_intervened=intervened,
        induce_refusal_rate_baseline=0.0,
        induce_refusal_rate_intervened=0.0,
    )


def test_run_pipeline_enhanced_accepts_preloaded_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_integration import MockRefusalModel

    preloaded_model = MockRefusalModel()

    def fail_model_load(*args, **kwargs):
        raise AssertionError("RefusalModel should not be constructed when preloaded_model is supplied")

    def stop_after_model_reuse(*args, **kwargs):
        raise RuntimeError("split boundary reached")

    monkeypatch.setattr(pipeline, "RefusalModel", fail_model_load)
    monkeypatch.setattr(pipeline, "resolve_refusal_tokens", lambda *args, **kwargs: [1])
    monkeypatch.setattr(pipeline, "build_splits", stop_after_model_reuse)

    cfg = {
        "model": {"name": "unused", "dtype": "float32", "device": "cpu"},
        "data": {
            "seed": 0,
            "harmful_sources": [],
            "n_train": 1,
            "n_val": 1,
            "n_bypass_eval": 1,
            "n_induce_eval": 1,
        },
        "extraction": {"post_instruction_positions": [-1], "batch_size": 1},
        "generation": {},
        "evaluation": {},
    }

    with pytest.raises(RuntimeError, match="split boundary reached"):
        pipeline.run_pipeline_enhanced(cfg, preloaded_model=preloaded_model)


def test_autoresearch_sweep_reuses_one_model_and_logs_results(tmp_path: Path) -> None:
    from src.autoresearch import run_config_sweep

    shared_model = object()
    seen_models = []

    def runner(cfg, *, preloaded_model=None):
        seen_models.append(preloaded_model)
        return _minimal_result(intervened=cfg["evaluation"]["expected_intervened"]), [torch.ones(1)]

    base_cfg = {
        "model": {"name": "mock", "dtype": "float32", "device": "cpu"},
        "evaluation": {"expected_intervened": 0.6},
    }
    candidates = [
        {"tag": "baseline", "overrides": {}},
        {"tag": "lower_refusal", "overrides": {"evaluation": {"expected_intervened": 0.2}}},
    ]
    results_path = tmp_path / "results.tsv"

    records = run_config_sweep(
        base_cfg,
        candidates,
        shared_model=shared_model,
        runner=runner,
        results_path=results_path,
    )

    assert seen_models == [shared_model, shared_model]
    assert [record.tag for record in records] == ["baseline", "lower_refusal"]
    assert records[1].status == "keep"

    with results_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert rows[0]["tag"] == "baseline"
    assert rows[1]["status"] == "keep"
