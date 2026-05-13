from __future__ import annotations

import pytest

import src.data as data


def test_load_harmful_can_skip_failed_sources_when_partial_loading_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_loader() -> list[str]:
        raise RuntimeError("gated")

    monkeypatch.setattr(
        data,
        "HARMFUL_LOADERS",
        {
            "gated/source": broken_loader,
            "open/source": lambda: ["open prompt 1", "open prompt 2"],
        },
    )

    prompts = data.load_harmful(["gated/source", "open/source"], seed=0, allow_partial_sources=True)

    assert sorted(prompts) == ["open prompt 1", "open prompt 2"]


def test_load_harmful_still_raises_when_every_partial_source_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        data,
        "HARMFUL_LOADERS",
        {"gated/source": lambda: (_ for _ in ()).throw(RuntimeError("gated"))},
    )

    with pytest.raises(RuntimeError, match="No harmful prompts loaded"):
        data.load_harmful(["gated/source"], seed=0, allow_partial_sources=True)


def test_build_splits_passes_partial_loading_to_harmful_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_harmful(sources: list[str], seed: int, *, allow_partial_sources: bool = False) -> list[str]:
        assert allow_partial_sources is True
        return [f"harmful-{idx}" for idx in range(12)]

    monkeypatch.setattr(data, "load_harmful", fake_load_harmful)
    monkeypatch.setattr(data, "load_harmless", lambda seed: [f"harmless-{idx}" for idx in range(20)])
    monkeypatch.setattr(data, "load_jailbreakbench", lambda n: [f"eval-{idx}" for idx in range(n)])

    splits = data.build_splits(
        harmful_sources=["gated/source", "open/source"],
        n_train=5,
        n_val=2,
        n_bypass_eval=3,
        n_induce_eval=4,
        seed=0,
        allow_partial_sources=True,
    )

    assert len(splits.harmful_train) == 5
    assert len(splits.induce_eval) == 4


def test_build_splits_can_downshift_train_size_for_partial_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "load_harmful", lambda sources, seed, *, allow_partial_sources=False: [f"harmful-{idx}" for idx in range(100)])
    monkeypatch.setattr(data, "load_harmless", lambda seed: [f"harmless-{idx}" for idx in range(300)])
    monkeypatch.setattr(data, "load_jailbreakbench", lambda n: [f"eval-{idx}" for idx in range(n)])

    splits = data.build_splits(
        harmful_sources=["open/source"],
        n_train=512,
        n_val=32,
        n_bypass_eval=100,
        n_induce_eval=100,
        seed=0,
        allow_partial_sources=True,
        min_partial_train=64,
    )

    assert len(splits.harmful_train) == 68
    assert len(splits.harmful_val) == 32
    assert len(splits.harmless_train) == 68
    assert len(splits.harmless_val) == 32
