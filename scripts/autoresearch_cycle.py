"""Run an in-process config sweep with one shared loaded model.

Example:
    python scripts/autoresearch_cycle.py \
        --config configs/base.yaml \
        --candidates autoresearch_candidates.json \
        --results autoresearch_results.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.autoresearch import load_candidates, run_config_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autonomous config sweeps with one shared model")
    parser.add_argument("--config", default="configs/base.yaml", help="Base YAML config")
    parser.add_argument("--candidates", required=True, help="JSON list of config overlays")
    parser.add_argument("--results", default="autoresearch_results.tsv", help="TSV experiment log")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    candidates = load_candidates(Path(args.candidates))
    records = run_config_sweep(base_cfg, candidates, results_path=Path(args.results))
    for record in records:
        print(
            f"{record.tag}: objective={record.objective:.6f} "
            f"status={record.status} hash={record.config_hash}"
        )


if __name__ == "__main__":
    main()
