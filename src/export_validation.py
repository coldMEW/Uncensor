"""Export manifest helpers for edited model artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_export_manifest(
    *,
    model_name: str,
    method: str,
    files: Iterable[str | Path],
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    file_entries = []
    for file_path in files:
        path = Path(file_path)
        file_entries.append(
            {
                "name": path.name,
                "path": str(path),
                "sha256": hash_file(path),
                "bytes": path.stat().st_size,
            }
        )

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "method": method,
        "files": file_entries,
        "metadata": dict(metadata or {}),
    }


def write_export_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    manifest_path = Path(path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def build_model_card_metadata(
    *,
    base_model: str,
    method: str,
    judge_status: str,
    benchmark_matrix: Mapping[str, Any],
    intended_use: str,
    license_note: str | None = None,
) -> Dict[str, Any]:
    """Build model-card metadata for calibrated exports.

    This metadata is intentionally conservative: it records judge status and
    limitations so unverified diagnostic runs cannot be mistaken for validated
    model releases.
    """
    return {
        "schema_version": 1,
        "base_model": str(base_model),
        "method": str(method),
        "intended_use": str(intended_use),
        "judge_status": str(judge_status),
        "benchmark_matrix": dict(benchmark_matrix),
        "license_note": license_note or "Derivative artifacts inherit the base model license and terms.",
        "limitations": [
            "This artifact is for refusal-calibration research, not blanket safety removal.",
            "Safety claims require a verified judge and dataset-scale benchmark matrix.",
            "Raw harmful-probe completions should not be published in run logs.",
        ],
    }


def _manifest_hashes_ok(export_dir: Path, manifest: Mapping[str, Any]) -> bool:
    entries = list(manifest.get("files", []))
    if not entries:
        return False
    for entry in entries:
        file_path = Path(entry.get("path", ""))
        if not file_path.is_absolute():
            file_path = export_dir / str(entry.get("name", ""))
        if not file_path.exists():
            return False
        if hash_file(file_path) != entry.get("sha256"):
            return False
    return True


def _manifest_lists_required_files(export_dir: Path, manifest: Mapping[str, Any]) -> bool:
    entries = list(manifest.get("files", []))
    names = {str(entry.get("name", "")) for entry in entries}
    weight_names = {path.name for path in export_dir.glob("*.safetensors")}
    required_names = {"config.json", "tokenizer_config.json"} | weight_names
    return bool(entries) and required_names.issubset(names)


def validate_export_roundtrip(
    export_dir: str | Path,
    *,
    load_fn: Callable[[Path], Any],
    smoke_eval_fn: Callable[[Any, Any], Mapping[str, Any]],
) -> Dict[str, Any]:
    """Validate exported files, manifest hashes, reload, and smoke generation."""
    root = Path(export_dir)
    manifest_path = root / "edit_manifest.json"
    report: Dict[str, Any] = {
        "config_exists": (root / "config.json").exists(),
        "tokenizer_exists": (root / "tokenizer_config.json").exists(),
        "weights_present": any(root.glob("*.safetensors")),
        "manifest_exists": manifest_path.exists(),
        "manifest_ok": False,
        "hashes_ok": False,
        "roundtrip_ok": False,
        "ok": False,
    }

    if not all(
        [
            report["config_exists"],
            report["tokenizer_exists"],
            report["weights_present"],
            report["manifest_exists"],
        ]
    ):
        return report

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report["manifest_ok"] = int(manifest.get("schema_version", 0)) >= 1 and _manifest_lists_required_files(root, manifest)
        report["hashes_ok"] = _manifest_hashes_ok(root, manifest)
        model, tokenizer = load_fn(root)
        smoke = dict(smoke_eval_fn(model, tokenizer))
        report.update(smoke)
        report["roundtrip_ok"] = bool(smoke.get("load_ok", False) and smoke.get("generate_ok", False))
    except Exception as exc:
        report["error"] = str(exc)
        return report

    report["ok"] = bool(report["manifest_ok"] and report["hashes_ok"] and report["roundtrip_ok"])
    return report
