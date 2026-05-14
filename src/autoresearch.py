"""Small autonomous experiment utilities inspired by karpathy/autoresearch.

The core rule is pragmatic: keep the expensive model object alive in one Python
process, vary configs around it, and log every attempt to a TSV so the next
agent/human can make keep/discard decisions from evidence.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .model import RefusalModel
from .pipeline import PipelineResult, run_pipeline_enhanced


RESULTS_HEADER = [
    "tag",
    "config_hash",
    "objective",
    "status",
    "bypass_baseline",
    "bypass_intervened",
    "kl_score",
    "description",
]


@dataclass
class ExperimentRecord:
    tag: str
    config_hash: str
    objective: float
    status: str
    bypass_baseline: float
    bypass_intervened: float
    kl_score: float
    description: str = ""

    def to_row(self) -> Dict[str, str]:
        return {
            "tag": self.tag,
            "config_hash": self.config_hash,
            "objective": f"{self.objective:.6f}",
            "status": self.status,
            "bypass_baseline": f"{self.bypass_baseline:.6f}",
            "bypass_intervened": f"{self.bypass_intervened:.6f}",
            "kl_score": f"{self.kl_score:.6f}",
            "description": self.description,
        }


def load_shared_model(cfg: Mapping[str, Any]) -> RefusalModel:
    """Load one model for a whole in-process experiment sweep."""
    model_cfg = cfg["model"]
    return RefusalModel(
        name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "float16"),
        device=model_cfg.get("device", "cuda"),
        quantization=model_cfg.get("quantization"),
    )


def merge_config(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep-merged config without mutating either input."""
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def config_hash(cfg: Mapping[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def objective_from_result(result: PipelineResult) -> float:
    """Lower-is-better constrained objective for keep/discard decisions.

    This intentionally includes KL drift and benign induction pressure so an
    autonomous sweep does not chase refusal reduction by breaking the model.
    """
    benign_regression = max(
        0.0,
        result.induce_refusal_rate_intervened - result.induce_refusal_rate_baseline,
    )
    return float(result.bypass_refusal_rate_intervened + result.kl_score + benign_regression)


def append_result(results_path: Path, record: ExperimentRecord) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not results_path.exists() or results_path.stat().st_size == 0
    with results_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER, delimiter="\t")
        if needs_header:
            writer.writeheader()
        writer.writerow(record.to_row())


def run_config_sweep(
    base_cfg: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    shared_model: Optional[RefusalModel] = None,
    runner: Callable[..., Tuple[PipelineResult, List[torch.Tensor]]] = run_pipeline_enhanced,
    results_path: Optional[Path] = None,
) -> List[ExperimentRecord]:
    """Run candidate config overlays while reusing one loaded model.

    Candidate format:
    ``{"tag": "name", "overrides": {...}, "description": "short note"}``.
    """
    model = shared_model or load_shared_model(base_cfg)
    records: List[ExperimentRecord] = []
    best_objective: Optional[float] = None

    for index, candidate in enumerate(candidates):
        tag = str(candidate.get("tag", f"candidate_{index:03d}"))
        overrides = candidate.get("overrides", {})
        description = str(candidate.get("description", ""))
        cfg = merge_config(base_cfg, overrides)

        result, _directions = runner(cfg, preloaded_model=model)
        objective = objective_from_result(result)
        status = "keep" if best_objective is None or objective < best_objective else "discard"
        if status == "keep":
            best_objective = objective

        record = ExperimentRecord(
            tag=tag,
            config_hash=config_hash(cfg),
            objective=objective,
            status=status,
            bypass_baseline=float(result.bypass_refusal_rate_baseline),
            bypass_intervened=float(result.bypass_refusal_rate_intervened),
            kl_score=float(result.kl_score),
            description=description,
        )
        records.append(record)
        if results_path is not None:
            append_result(results_path, record)

    return records


def load_candidates(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        blob = json.load(f)
    if not isinstance(blob, list):
        raise ValueError("Candidate file must contain a JSON list")
    return [dict(item) for item in blob]
