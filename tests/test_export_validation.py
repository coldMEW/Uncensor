from __future__ import annotations

import json

from src.export_validation import (
    build_export_manifest,
    build_model_card_metadata,
    hash_file,
    validate_export_roundtrip,
    write_export_manifest,
)


def test_hash_file_returns_stable_sha256(tmp_path) -> None:
    file_path = tmp_path / "weights.safetensors"
    file_path.write_bytes(b"abc")

    assert hash_file(file_path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_write_export_manifest_records_files_and_metadata(tmp_path) -> None:
    weights = tmp_path / "weights.safetensors"
    config = tmp_path / "config.json"
    weights.write_bytes(b"weights")
    config.write_text("{}", encoding="utf-8")

    manifest = build_export_manifest(
        model_name="test/model",
        method="svd_multi",
        files=[weights, config],
        metadata={"base_revision": "abc123"},
    )
    manifest_path = write_export_manifest(tmp_path / "manifest.json", manifest)

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["model_name"] == "test/model"
    assert loaded["method"] == "svd_multi"
    assert loaded["metadata"]["base_revision"] == "abc123"
    assert {entry["name"] for entry in loaded["files"]} == {"weights.safetensors", "config.json"}


def test_build_model_card_metadata_records_base_model_and_calibration_limits() -> None:
    metadata = build_model_card_metadata(
        base_model="google/gemma-4-E4B-it",
        method="svd_multi",
        judge_status="UNVERIFIED_JUDGE",
        benchmark_matrix={"safety_retention": {"strongreject": {"count": 100}}},
        intended_use="refusal-calibration research",
    )

    assert metadata["base_model"] == "google/gemma-4-E4B-it"
    assert metadata["method"] == "svd_multi"
    assert metadata["judge_status"] == "UNVERIFIED_JUDGE"
    assert metadata["intended_use"] == "refusal-calibration research"
    assert metadata["limitations"]


def test_validate_export_roundtrip_checks_required_files_hashes_and_smoke_eval(tmp_path) -> None:
    weights = tmp_path / "model.safetensors"
    config = tmp_path / "config.json"
    tokenizer = tmp_path / "tokenizer_config.json"
    weights.write_bytes(b"weights")
    config.write_text("{}", encoding="utf-8")
    tokenizer.write_text("{}", encoding="utf-8")

    manifest = build_export_manifest(
        model_name="test/model",
        method="svd_multi",
        files=[weights, config, tokenizer],
    )
    write_export_manifest(tmp_path / "edit_manifest.json", manifest)

    report = validate_export_roundtrip(
        tmp_path,
        load_fn=lambda export_dir: ("model", "tokenizer"),
        smoke_eval_fn=lambda model, tokenizer: {"load_ok": True, "generate_ok": True},
    )

    assert report["ok"] is True
    assert report["manifest_ok"] is True
    assert report["hashes_ok"] is True
    assert report["roundtrip_ok"] is True


def test_validate_export_roundtrip_rejects_empty_manifest_file_list(tmp_path) -> None:
    weights = tmp_path / "model.safetensors"
    config = tmp_path / "config.json"
    tokenizer = tmp_path / "tokenizer_config.json"
    weights.write_bytes(b"weights")
    config.write_text("{}", encoding="utf-8")
    tokenizer.write_text("{}", encoding="utf-8")
    write_export_manifest(
        tmp_path / "edit_manifest.json",
        {"schema_version": 1, "files": []},
    )

    report = validate_export_roundtrip(
        tmp_path,
        load_fn=lambda export_dir: ("model", "tokenizer"),
        smoke_eval_fn=lambda model, tokenizer: {"load_ok": True, "generate_ok": True},
    )

    assert report["ok"] is False
    assert report["manifest_ok"] is False
    assert report["hashes_ok"] is False


def test_validate_export_roundtrip_rejects_stale_manifest_hash(tmp_path) -> None:
    weights = tmp_path / "model.safetensors"
    config = tmp_path / "config.json"
    tokenizer = tmp_path / "tokenizer_config.json"
    weights.write_bytes(b"weights")
    config.write_text("{}", encoding="utf-8")
    tokenizer.write_text("{}", encoding="utf-8")

    manifest = build_export_manifest(
        model_name="test/model",
        method="svd_multi",
        files=[weights, config, tokenizer],
    )
    write_export_manifest(tmp_path / "edit_manifest.json", manifest)
    weights.write_bytes(b"tampered")

    report = validate_export_roundtrip(
        tmp_path,
        load_fn=lambda export_dir: ("model", "tokenizer"),
        smoke_eval_fn=lambda model, tokenizer: {"load_ok": True, "generate_ok": True},
    )

    assert report["ok"] is False
    assert report["manifest_ok"] is True
    assert report["hashes_ok"] is False
