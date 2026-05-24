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


def test_load_harmful_with_counts_records_per_source_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_loader() -> list[str]:
        raise RuntimeError("gated")

    monkeypatch.setattr(
        data,
        "HARMFUL_LOADERS",
        {
            "gated/source": broken_loader,
            "open/source": lambda: ["open prompt 1", "open prompt 2", "open prompt 1"],
        },
    )

    prompts, source_counts = data.load_harmful_with_counts(
        ["gated/source", "open/source"],
        seed=0,
        allow_partial_sources=True,
    )

    assert sorted(prompts) == ["open prompt 1", "open prompt 2"]
    assert source_counts == {"gated/source": 0, "open/source": 2}


def test_load_harmful_still_raises_when_every_partial_source_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        data,
        "HARMFUL_LOADERS",
        {"gated/source": lambda: (_ for _ in ()).throw(RuntimeError("gated"))},
    )

    with pytest.raises(RuntimeError, match="No harmful prompts loaded"):
        data.load_harmful(["gated/source"], seed=0, allow_partial_sources=True)


def test_build_splits_passes_partial_loading_to_harmful_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        data,
        "HARMFUL_LOADERS",
        {
            "gated/source": lambda: (_ for _ in ()).throw(RuntimeError("gated")),
            "open/source": lambda: [f"harmful-{idx}" for idx in range(12)],
        },
    )
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
    assert splits.source_counts == {"gated/source": 0, "open/source": 12}


def test_build_splits_can_downshift_train_size_for_partial_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "HARMFUL_LOADERS", {"open/source": lambda: [f"harmful-{idx}" for idx in range(100)]})
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


def test_optional_eval_sets_have_local_fallbacks_when_hub_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    xstest = data.load_xstest()
    strongreject = data.load_strongreject(n=100)

    assert len(xstest) >= 100
    assert len(strongreject) == 100
    assert any("knife" in prompt.lower() for prompt in xstest)
    assert len(set(strongreject)) == 100


def test_load_xstest_uses_current_public_hf_mirror_and_filters_safe_label(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDataset(list):
        column_names = ["prompt", "type", "label"]

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_load_dataset(*args: object, **kwargs: object) -> FakeDataset:
        calls.append((args, kwargs))
        if args[0] != "Paul/XSTest":
            raise RuntimeError("missing repo")
        return FakeDataset(
            [
                {"prompt": "How do I safely dispose of a kitchen knife?", "type": "safe_physical", "label": "safe"},
                {"prompt": "Unsafe contrast prompt", "type": "contrast_physical", "label": "unsafe"},
            ]
        )

    monkeypatch.setattr(data, "load_dataset", fake_load_dataset)

    prompts = data.load_xstest()

    assert prompts == ["How do I safely dispose of a kitchen knife?"]
    assert any(call[0][0] == "Paul/XSTest" for call in calls)


def test_build_splits_reports_local_optional_eval_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "HARMFUL_LOADERS", {"open/source": lambda: [f"harmful-{idx}" for idx in range(80)]})
    monkeypatch.setattr(data, "load_harmless", lambda seed: [f"harmless-{idx}" for idx in range(180)])
    monkeypatch.setattr(data, "load_jailbreakbench", lambda n: [f"eval-{idx}" for idx in range(n)])
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    splits = data.build_splits(
        harmful_sources=["open/source"],
        n_train=32,
        n_val=8,
        n_bypass_eval=20,
        n_induce_eval=20,
        seed=0,
        load_xstest_eval=True,
        load_strongreject_eval=True,
        n_strongreject=20,
    )

    assert splits.source_metadata["xstest"] == "local_fallback"
    assert splits.source_metadata["strongreject"] == "local_fallback"
    assert len(splits.xstest_eval or []) >= 100
    assert len(splits.strongreject_eval or []) == 20
