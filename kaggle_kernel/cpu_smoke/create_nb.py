import json
from pathlib import Path


NOTEBOOK_NAME = "uncensor-refusal-pipeline-test-cpu-smoke.ipynb"


def code_cell(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kaggle": {
            "accelerator": "None",
            "enable_gpu": False,
            "language": "python",
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
    },
    "cells": [
        markdown_cell(
            "intro",
            """# Uncensor CPU Smoke Test

This notebook is intentionally CPU-only. It validates Kaggle setup, repository clone, core imports,
optimizer stop controls, probe helpers, quality checks, and a tiny Hugging Face generation path.

It does not run the Gemma refusal-direction research pipeline and must not be used for refusal metrics.
""",
        ),
        code_cell(
            "cpu-guard",
            """import json
import os
import platform
from datetime import datetime
from pathlib import Path

import torch

CPU_SMOKE_MODE = True
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Assigned GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []}")
print("CPU smoke guard passed: GPU is not required for this run")
""",
        ),
        code_cell(
            "setup",
            """!pip -q install "transformers>=4.40.0" "accelerate>=0.25.0" "huggingface_hub>=0.20.0" "numpy>=1.24.0"

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/coldMEW/Uncensor.git"
REPO_DIR = Path("/kaggle/working/uncensor")

if REPO_DIR.exists():
    print(f"Using existing source path: {REPO_DIR}")
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)

commit = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"], text=True).strip()
print(f"Repo commit: {commit}")
sys.path.insert(0, str(REPO_DIR))
""",
        ),
        code_cell(
            "core-contracts",
            """from src.metrics import completion_quality_report
from src.optimization import meaningful_improvement, should_stop_search
from src.probes import dual_probe_scores

assert should_stop_search(
    converged=True,
    completed_cycles=0,
    max_cycles=3,
    stagnant_cycles=0,
    max_stagnant_cycles=2,
) == (True, "CONVERGED")

assert should_stop_search(
    converged=False,
    completed_cycles=0,
    max_cycles=3,
    stagnant_cycles=0,
    max_stagnant_cycles=2,
    elapsed_seconds=60,
    max_seconds=60,
) == (True, "TIME_BUDGET_EXHAUSTED")

assert meaningful_improvement(
    {"run_is_valid": False, "valid_reduction_rate": 0.0, "avg_score_drop": 0.0},
    {
        "run_is_valid": True,
        "valid_reduction_rate": 0.75,
        "avg_score_drop": 0.2,
        "bypass_quality_rate": 1.0,
        "benign_valid_rate": 1.0,
    },
)

quality = completion_quality_report("This is a short but coherent CPU smoke completion.")
probe = dual_probe_scores("Explain safe lab ventilation.", "Use a fume hood and follow the safety data sheet.")
print(f"Quality ok: {quality['ok']} reason={quality['reason']}")
print(f"Dual probe refusal={probe.refusal_score:.3f} harmfulness={probe.harmfulness_score:.3f}")
""",
        ),
        code_cell(
            "tiny-generation",
            """from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "sshleifer/tiny-gpt2"
print(f"Loading {MODEL_NAME} for CPU smoke generation...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

prompt = "CPU smoke test:"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(f"Tiny generation: {completion!r}")
assert isinstance(completion, str)
""",
        ),
        code_cell(
            "results",
            """results = {
    "status": "CPU_SMOKE_PASS",
    "status_reasons": [],
    "cpu_smoke_mode": CPU_SMOKE_MODE,
    "purpose": (
        "Validates Kaggle CPU environment, repo clone, imports, tiny model generation, "
        "optimizer stop controls, probe helpers, and quality checks. Not a refusal-metric run."
    ),
    "repo_commit": commit,
    "model": MODEL_NAME,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "python": platform.python_version(),
    "timestamp": datetime.utcnow().isoformat() + "Z",
}

out_path = Path("/kaggle/working/uncensor_cpu_smoke_results.json")
out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
print(f"Results saved to {out_path}")
""",
        ),
    ],
}


out = Path(__file__).with_name(NOTEBOOK_NAME)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out}")
