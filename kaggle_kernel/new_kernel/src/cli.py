"""
Command-line entry point — extended with RESEARCH_PLAN.md flags.

Usage:
    python -m src.cli --config configs/base.yaml
    python -m src.cli --config configs/base.yaml --model Qwen/Qwen1.5-1.8B-Chat
    python -m src.cli --config configs/base.yaml --n-directions 3 --causal-patching --analyze-heads
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
import yaml

from .pipeline import run_pipeline


def _git_sha() -> str:
    """Return the current git HEAD sha, or 'unknown' if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refusal-direction pipeline (extended with RESEARCH_PLAN.md improvements)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="YAML config file (see configs/base.yaml for the schema).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override the model name from the config.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override the device (cuda / cpu).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Where to write the selected direction and the metric summary.",
    )
    # --- RESEARCH_PLAN.md extended flags ---
    parser.add_argument(
        "--n-directions", type=int, default=None,
        help="§5.1 — max iterative subspace extraction rounds (default: config value, typically 1).",
    )
    parser.add_argument(
        "--causal-patching", action="store_true", default=None,
        help="§5.3 — enable causal layer selection via activation patching.",
    )
    parser.add_argument(
        "--no-causal-patching", action="store_true", default=None,
        help="§5.3 — disable causal layer selection.",
    )
    parser.add_argument(
        "--analyze-heads", action="store_true", default=None,
        help="§5.4 — enable attention head analysis at the best layer.",
    )
    parser.add_argument(
        "--no-analyze-heads", action="store_true", default=None,
        help="§5.4 — disable attention head analysis.",
    )
    parser.add_argument(
        "--calibration", action="store_true", default=None,
        help="§5.6 — enable calibrated orthogonalization (adjusts RMSNorm gamma).",
    )
    parser.add_argument(
        "--no-calibration", action="store_true", default=None,
        help="§5.6 — disable calibrated orthogonalization.",
    )
    parser.add_argument(
        "--xstest", action="store_true", default=None,
        help="§3.5 — enable XSTest over-refusal evaluation.",
    )
    parser.add_argument(
        "--no-xstest", action="store_true", default=None,
        help="§3.5 — disable XSTest evaluation.",
    )
    parser.add_argument(
        "--strongreject", action="store_true", default=None,
        help="§3.5 — enable StrongREJECT bypass evaluation.",
    )
    parser.add_argument(
        "--no-strongreject", action="store_true", default=None,
        help="§3.5 — disable StrongREJECT evaluation.",
    )
    parser.add_argument(
        "--post-quant-correct", action="store_true", default=None,
        help="§3.6 — enable post-quantization ortho correction.",
    )
    parser.add_argument(
        "--mode", type=str, choices=["inference", "permanent"], default=None,
        help="Execution mode: 'inference' for reversible hook-based steering, "
             "'permanent' for weight orthogonalization (default: permanent).",
    )
    parser.add_argument(
        "--preset", type=str, choices=["basic", "moderate", "aggressive", "nuclear"], default=None,
        help="Pipeline preset: basic (1 direction), moderate (2), aggressive (3), nuclear (5). "
             "Overrides individual extraction flags.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # Model overrides.
    if args.model is not None:
        cfg["model"]["name"] = args.model
    if args.device is not None:
        cfg["model"]["device"] = args.device

    # Extraction overrides (§5.1, §5.3, §5.4, §5.6, §3.6).
    ext = cfg.setdefault("extraction", {})
    if args.n_directions is not None:
        ext["n_directions"] = args.n_directions
    if args.causal_patching:
        ext["use_causal_patching"] = True
    elif args.no_causal_patching is not None:
        ext["use_causal_patching"] = False
    if args.analyze_heads:
        ext["analyze_heads"] = True
    elif args.no_analyze_heads is not None:
        ext["analyze_heads"] = False
    if args.calibration:
        ext["use_calibration"] = True
    elif args.no_calibration is not None:
        ext["use_calibration"] = False
    if args.post_quant_correct:
        ext["post_quant_correction"] = True
    if args.mode is not None:
        ext["execution_mode"] = args.mode

    # US-014: Escalation presets
    if args.preset is not None:
        preset_configs = {
            "basic": {"n_directions": 1, "delta_threshold": 0.5, "kl_score_max": 0.1},
            "moderate": {"n_directions": 2, "delta_threshold": 0.4, "kl_score_max": 0.15},
            "aggressive": {"n_directions": 3, "delta_threshold": 0.3, "kl_score_max": 0.2},
            "nuclear": {"n_directions": 5, "delta_threshold": 0.2, "kl_score_max": 0.3},
        }
        preset = preset_configs[args.preset]
        ext["n_directions"] = preset["n_directions"]
        ext["delta_threshold"] = preset["delta_threshold"]
        ext["kl_score_max"] = preset["kl_score_max"]

    # Evaluation overrides (§3.5).
    ev = cfg.setdefault("evaluation", {})
    if args.xstest:
        ev["load_xstest_eval"] = True
    elif args.no_xstest is not None:
        ev["load_xstest_eval"] = False
    if args.strongreject:
        ev["load_strongreject_eval"] = True
    elif args.no_strongreject is not None:
        ev["load_strongreject_eval"] = False

    result, direction = run_pipeline(cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = cfg["model"]["name"].replace("/", "__")
    torch.save(direction, out_dir / f"{safe_name}_refusal_direction.pt")
    with open(out_dir / f"{safe_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    print("[pipeline] metrics:")
    print(json.dumps(result.to_dict(), indent=2))
    if hasattr(result, "summary"):
        print(result.summary())

    # §6.7 — Structured lineage JSON (separate from the compact metrics
    # file). Captures git sha, resolved config (after CLI overrides),
    # iterative round history, causal/head scores, timing, runtime.
    lineage = getattr(result, "lineage", None) or {}
    lineage["git_sha"] = _git_sha()
    lineage["resolved_config"] = cfg
    lineage_path = out_dir / f"{safe_name}_lineage.json"
    with open(lineage_path, "w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2, default=str)
    print(f"[pipeline] lineage written to {lineage_path}")


if __name__ == "__main__":
    main()
