# Uncensor Autoresearch Program

This file is the operating contract for autonomous experiments in this repo.
It is inspired by `karpathy/autoresearch`, adapted to refusal-calibration
research where safety, utility, and drift constraints matter.

## Scope

Allowed experiment surfaces:
- Config overlays for extraction, method registry, layer windows, coefficients,
  benchmark sizes, and ranking weights.
- `src/optimization.py`, `src/methods.py`, and narrow evaluation glue when the
  experiment requires code.
- `kaggle_kernel/new_kernel/create_nb.py` only when the Kaggle verifier needs a
  new observable or guardrail.

Protected surfaces:
- Do not weaken benchmark gates.
- Do not remove dual probes, KL/logit drift, benign controls, export validation,
  or judge-backend metadata.
- Do not hardcode secrets. Hugging Face and Kaggle credentials must come from
  environment variables or Kaggle Secrets.
- Do not edit raw benchmark prompts to make metrics easier.

## Objective

Prefer candidates that reduce measured refusal on the targeted refusal probes
while preserving:
- benign valid rate,
- output quality,
- harmfulness/refusal probe separation,
- low KL/logit drift,
- verified judge/backend metadata.

Lower autonomous objective is better:

`bypass_refusal_rate_intervened + kl_score + benign_refusal_regression`

Kaggle full runs remain the source of truth. Local/in-process sweeps are for
fast candidate triage.

## Experiment Loop

1. Start from a clean commit.
2. Load the model once with `src.autoresearch.load_shared_model()` or use
   `scripts/autoresearch_cycle.py`.
3. Run candidate config overlays in one process so the same `RefusalModel` is
   reused across candidates.
4. Log every run to `autoresearch_results.tsv`.
5. Keep only candidates that improve the objective without triggering quality,
   benign, judge, or dataset-scale regressions.
6. Promote the best candidate to the Kaggle notebook and run a full Kaggle
   verification.
7. If Kaggle disagrees with the local sweep, trust Kaggle and log the failure.

## Candidate File Format

Use a JSON list:

```json
[
  {
    "tag": "baseline",
    "description": "unchanged config",
    "overrides": {}
  },
  {
    "tag": "layer-window-radius-4",
    "description": "wider layer-local search",
    "overrides": {
      "extraction": {
        "n_directions": 3
      }
    }
  }
]
```

Run:

```bash
python scripts/autoresearch_cycle.py --config configs/base.yaml --candidates autoresearch_candidates.json
```

## Model Reuse Reality

Within one Python process, model reuse is real: the sweep passes a single
preloaded `RefusalModel` into every pipeline run.

Across separate Kaggle kernel versions, model reuse in VRAM is not possible:
Kaggle starts a fresh worker process. The best mitigation there is to use
Kaggle Secrets for auth, persistent Hugging Face cache where the platform
allows it, and fewer full Kaggle submissions by doing local triage first.
