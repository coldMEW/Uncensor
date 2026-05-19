"""Contract tests for Kaggle GPU package metadata.

Run 59 failed before model load because a Kaggle submission was executed
without GPU.  Every GPU entrypoint in this repo must therefore request the
same T4 shape, not rely on legacy ``gpu`` metadata aliases.
"""
from __future__ import annotations

import json
from pathlib import Path


GPU_METADATA_FILES = [
    Path("kernel-metadata.json"),
    Path("kaggle_kernel/kernel-metadata.json"),
    Path("kaggle_kernel/new_kernel/kernel-metadata.json"),
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_all_kaggle_gpu_metadata_files_request_t4_x2() -> None:
    for path in GPU_METADATA_FILES:
        metadata = _read_json(path)
        assert metadata["enable_gpu"] is True, path
        assert metadata["machine_shape"] == "NvidiaTeslaT4", path
        assert "gpu" not in metadata, path


def test_generated_gpu_notebooks_request_t4_x2() -> None:
    for path in [
        Path("uncensor-refusal-pipeline-test.ipynb"),
        Path("kaggle_kernel/notebook.ipynb"),
        Path("kaggle_kernel/new_kernel/uncensor-refusal-pipeline-test.ipynb"),
    ]:
        nb = _read_json(path)
        kaggle = nb["metadata"]["kaggle"]
        assert kaggle["accelerator"] == "GPU T4 x2", path
        assert kaggle["enable_gpu"] is True, path
        assert kaggle.get("gpu") is not True, path


def test_legacy_notebook_generator_no_longer_emits_ambiguous_gpu_metadata() -> None:
    source = Path("kaggle_kernel/create_nb.py").read_text(encoding="utf-8")
    assert '"accelerator": "GPU T4 x2"' in source
    assert '"enable_gpu": True' in source
    assert '"gpu": True' not in source


def test_root_gpu_notebook_matches_canonical_generated_notebook() -> None:
    root = _read_json(Path("uncensor-refusal-pipeline-test.ipynb"))
    canonical = _read_json(Path("kaggle_kernel/new_kernel/uncensor-refusal-pipeline-test.ipynb"))
    assert root == canonical
