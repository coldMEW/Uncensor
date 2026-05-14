import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CPU_SMOKE_DIR = ROOT / "kaggle_kernel" / "cpu_smoke"
GPU_KERNEL_DIR = ROOT / "kaggle_kernel" / "new_kernel"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_cpu_smoke_kernel_is_isolated_from_gpu_pipeline():
    create_nb = CPU_SMOKE_DIR / "create_nb.py"
    notebook = CPU_SMOKE_DIR / "uncensor-refusal-pipeline-test-cpu-smoke.ipynb"
    metadata = CPU_SMOKE_DIR / "kernel-metadata.json"

    assert create_nb.exists()
    assert notebook.exists()
    assert metadata.exists()

    metadata_json = json.loads(_read(metadata))
    assert metadata_json["id"] == "coldmew/uncensor-refusal-pipeline-test-cpu-smoke"
    assert metadata_json["code_file"] == "uncensor-refusal-pipeline-test-cpu-smoke.ipynb"
    assert metadata_json["enable_gpu"] is False
    assert metadata_json["enable_internet"] is True
    assert "machine_shape" not in metadata_json

    combined = _read(create_nb) + "\n" + _read(notebook)
    required_markers = [
        "CPU_SMOKE_MODE",
        "CPU_SMOKE_PASS",
        "sshleifer/tiny-gpt2",
        "should_stop_search",
        "meaningful_improvement",
        "completion_quality_report",
        "dual_probe_scores",
    ]
    for marker in required_markers:
        assert marker in combined

    forbidden_markers = [
        "multi_directional_ablation",
        "build_intervention_candidates",
        "=== SEARCH EVALUATION ===",
        "=== FINAL VERIFICATION",
        "DEGENERATE_OUTPUT",
        "INSUFFICIENT_VALID_REDUCTION",
        "MODEL_NAME = 'google/gemma-4-E4B-it'",
        'MODEL_NAME = "google/gemma-4-E4B-it"',
    ]
    for marker in forbidden_markers:
        assert marker not in combined


def test_gpu_kernel_contract_remains_gpu_only():
    gpu_metadata = json.loads(_read(GPU_KERNEL_DIR / "kernel-metadata.json"))
    gpu_generator = _read(GPU_KERNEL_DIR / "create_nb.py")

    assert gpu_metadata["id"] == "coldmew/uncensor-refusal-pipeline-test"
    assert gpu_metadata["enable_gpu"] is True
    assert gpu_metadata["machine_shape"] == "NvidiaTeslaT4"
    assert "GPU T4 x2" in gpu_generator
    assert "google/gemma-4-E4B-it" in gpu_generator
