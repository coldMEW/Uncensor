# Running Guide — refusal_direction pipeline

Exact, ordered, verified command list for taking the `refusal_direction/`
project from a fresh machine to a completed run with inspectable outputs.
Every command is meant to be run in a **bash shell** (Git Bash on Windows)
from inside the project directory.

This guide complements `README.md` (paper reproduction overview) and
`REPRODUCTION_NOTES.md` (unspecified-detail audit). Read those before
trusting any number this code produces.

---

## Current machine state (verified)

- Python 3.11.0 at `C:\Users\DSU\AppData\Local\Programs\Python\Python311\python.exe`
- pip 22.3 available
- `torch` / `transformers` / `datasets` / etc. — **not installed**
- **No NVIDIA GPU detected** (`nvidia-smi` not on PATH) → CPU-only unless
  a CUDA toolkit is installed separately
- Project root contains `configs/`, `src/`, `tests/`, `requirements.txt`

---

## Step 0 — `cd` into the project

```bash
cd /c/Users/DSU/Desktop/Projects/Uncensor/uncensor_new/uncensor_research_rewrite/uncensor/refusal_direction
pwd
```

**Expected output:**
```
/c/Users/DSU/Desktop/Projects/Uncensor/uncensor_new/uncensor_research_rewrite/uncensor/refusal_direction
```

Do not `cd` out of this directory for the rest of the steps.

---

## Step 1 — Create a virtual environment

```bash
python -m venv .venv
```

**Expected:** silent success. A new `.venv/` directory appears. Takes ~15
seconds.

---

## Step 2 — Activate the venv

```bash
source .venv/Scripts/activate
python -c "import sys; print(sys.executable)"
```

**Expected:** path that starts with
`...refusal_direction/.venv/Scripts/python.exe`. If it still points at
`AppData\Local\Programs\Python\Python311`, the activate did not work —
check that you are in bash, not cmd.

---

## Step 3 — Upgrade pip

```bash
python -m pip install --upgrade pip
```

**Expected:** `Successfully installed pip-XX.X.X` (22.3 → latest). Takes
~30 seconds.

---

## Step 4 — Install PyTorch (CPU build)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Expected:** downloads ~200 MB, finishes with
`Successfully installed torch-2.x.x`. Takes 1–5 minutes.

If you *do* have a CUDA-capable GPU, replace the above with the CUDA
12.1 wheel index instead:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Verify:**
```bash
python -c "import torch; print(torch.__version__, 'cuda=', torch.cuda.is_available())"
```

**Expected:** `2.x.x cuda= False` (or `True` if you installed the CUDA
build).

---

## Step 5 — Install the rest of the requirements

```bash
pip install -r requirements.txt
```

**Expected:** installs `transformers`, `datasets`, `huggingface_hub`,
`numpy`, `pyyaml`, `tqdm`, `scipy`, `accelerate`. Finishes with
`Successfully installed ...`. Takes 2–5 minutes.

**Verify:**
```bash
python -c "import torch, transformers, datasets, yaml, accelerate, tqdm, scipy; print('all imports OK')"
```

**Expected:** `all imports OK`

---

## Step 6 — Verify the package itself imports

```bash
python -c "from src.pipeline import run_pipeline; print('pipeline import OK')"
```

**Expected:** `pipeline import OK`. If this fails with
`ModuleNotFoundError: No module named 'src'`, you are not in the
`refusal_direction/` directory — re-run Step 0.

---

## Step 7 — Shrink the config for a CPU smoke test

**CPU warning.** With the default config (`n_train=128`, `n_bypass_eval=100`,
`n_induce_eval=100`, 3 iterative rounds) on CPU with a 1.8B model, expect
**2–6 hours**. For a first run that finishes in **10–20 minutes** on CPU,
make a shrunk copy of the config:

```bash
cp configs/base.yaml configs/smoke.yaml
python - <<'PY'
import yaml, pathlib
p = pathlib.Path("configs/smoke.yaml")
cfg = yaml.safe_load(p.read_text())
cfg["model"]["device"] = "cpu"
cfg["model"]["dtype"] = "float32"          # bfloat16 is slow on CPU
cfg["data"]["n_train"] = 16
cfg["data"]["n_val"] = 8
cfg["data"]["n_bypass_eval"] = 8
cfg["data"]["n_induce_eval"] = 8
cfg["extraction"]["n_directions"] = 1
cfg["extraction"]["batch_size"] = 2
cfg["generation"]["max_new_tokens"] = 32
cfg["generation"]["batch_size"] = 2
p.write_text(yaml.safe_dump(cfg))
print("wrote", p)
PY
```

**Expected:** `wrote configs/smoke.yaml`

---

## Step 8 — First real run (CPU smoke test)

```bash
python -m src.cli --config configs/smoke.yaml
```

**What happens, in order:**

1. **Dataset downloads** from HuggingFace on first run: AdvBench,
   MaliciousInstruct, HarmBench, Alpaca, JBB-Behaviors. A few hundred MB
   the first time, cached to `%USERPROFILE%\.cache\huggingface\`.
2. **Model download**: `Qwen/Qwen1.5-1.8B-Chat`, ~3.5 GB on first run.
3. **Activation capture** — tqdm bar labelled
   `collect harmful` / `collect harmless`.
4. **Difference-in-means + selection** — prints the best candidate
   `(layer, position)` and its three scores.
5. **Generation under each intervention** — another tqdm bar.
6. **Weight orthogonalization** — fast.
7. **Metrics printed** to stdout as JSON.
8. **Summary table** printed.
9. **Lineage file** written.

**Expected terminal tail (schematic):**
```
[pipeline] metrics:
{
  "best_layer": <int in [6, 20]>,
  "best_position": -1 or -2 or -3,
  "bypass_score": <float, typically 0.5-0.95>,
  "induce_score": <float, typically 0.4-0.9>,
  "kl_score": <float, should be ≤ 0.10>,
  "bypass_refusal_rate_baseline": <float, should be ≥ 0.6 with only 8 eval prompts>,
  "bypass_refusal_rate_intervened": <float, should be much lower than baseline>,
  "induce_refusal_rate_baseline": <float, should be low>,
  "induce_refusal_rate_intervened": <float, should be higher than baseline>,
  "orth_refusal_rate": <close to bypass_refusal_rate_intervened>,
  "n_directions_found": 1,
  ...
}
============================================================
PIPELINE RESULT SUMMARY
============================================================
  Best direction:    layer=<int>, position=<int>
  Bypass score:      0.xxx
  ...
[pipeline] lineage written to outputs\Qwen__Qwen1.5-1.8B-Chat_lineage.json
```

**Small-N caveat.** With only 8 eval prompts per split, refusal-rate
numbers are noisy (resolution = 1/8 = 0.125). The rank-order and sign of
the change should still match the trends below; the absolute values are
not publication-quality. That is the cost of running on CPU; it is fine
for validating the pipeline works.

---

## Step 9 — Inspect the outputs

```bash
ls -la outputs/
cat outputs/Qwen__Qwen1.5-1.8B-Chat_metrics.json
```

**Expected files:**
```
Qwen__Qwen1.5-1.8B-Chat_refusal_direction.pt    # torch tensor, shape (d_model,) = (2048,)
Qwen__Qwen1.5-1.8B-Chat_metrics.json            # PipelineResult.to_dict()
Qwen__Qwen1.5-1.8B-Chat_lineage.json            # structured lineage (§9.7)
```

**Programmatic sanity checks.** Paste into a `python` REPL:

```python
import json, torch
m = json.load(open("outputs/Qwen__Qwen1.5-1.8B-Chat_metrics.json"))
d = torch.load("outputs/Qwen__Qwen1.5-1.8B-Chat_refusal_direction.pt")

# Shape + norm
print("direction shape:", d.shape)                  # expect torch.Size([2048])
print("direction norm: ", float(d.norm()))          # expect ~1.0 (unit vector)

# Sign-of-effect checks (the method's raison d'être)
print("bypass drop:  ", m["bypass_refusal_rate_baseline"] - m["bypass_refusal_rate_intervened"])  # expect > 0
print("induce rise:  ", m["induce_refusal_rate_intervened"] - m["induce_refusal_rate_baseline"])  # expect > 0
print("KL ok:        ", m["kl_score"] <= 0.10)

# Equivalence check (paper §4.1 / §E)
print("ortho gap:    ", abs(m["orth_refusal_rate"] - m["bypass_refusal_rate_intervened"]))        # expect ≤ 0.15 at N=8
```

---

## Step 10 — "Is this going in the positive direction?" read on the smoke run

With only 8 prompts per eval, pass thresholds are relaxed:

| Check                                         | Smoke (N=8)    | Paper-faithful (N≥100) |
|-----------------------------------------------|----------------|------------------------|
| `bypass_drop = baseline − intervened`         | ≥ 0.25         | ≥ 0.30                 |
| `induce_rise = intervened − baseline`         | ≥ 0.25         | ≥ 0.30                 |
| `kl_score`                                    | ≤ 0.10         | ≤ 0.10                 |
| `\|orth_refusal_rate − bypass_intervened\|`   | ≤ 0.15         | ≤ 0.05                 |
| `best_layer ∈ [n_layers/4, 3*n_layers/4]`     | [6, 18]        | [6, 18]                |

All five pass → pipeline is correct, ready for a real run.
Any fail → do **not** jump to a real run. Open `*_lineage.json` and
attribute the failing metric to a specific stage (extraction, selection,
generation, or orthogonalization) before touching anything else.

---

## Step 11 — Promote to the real run (only after Step 10 passes)

**If you have a GPU** (re-install torch with the CUDA wheel index first,
then verify `torch.cuda.is_available() == True`):

```bash
python -m src.cli --config configs/base.yaml \
    --n-directions 3 \
    --causal-patching \
    --analyze-heads \
    --calibration \
    --xstest \
    --strongreject
```

**If you are still on CPU** and want full-size eval, just run
`configs/base.yaml` with the defaults and expect 2–6 hours:

```bash
python -m src.cli --config configs/base.yaml
```

**To run on a different model**, add `--model <hf-id>`:

```bash
python -m src.cli --config configs/base.yaml \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --n-directions 3 --causal-patching --xstest
```

Gated models (Llama, Gemma) require a HuggingFace token:
```bash
huggingface-cli login
# paste a read-scope token from https://huggingface.co/settings/tokens
```

---

## Output file reference

For every run, three files are written into `outputs/` under the safe
name `{org}__{model_name}` (slashes replaced with double-underscore):

| File | Content |
|---|---|
| `{safe_name}_refusal_direction.pt` | `torch.save` of the selected direction tensor, shape `(d_model,)`. Load with `torch.load(...)`. |
| `{safe_name}_metrics.json`         | `PipelineResult.to_dict()` — compact results (best layer, all three scores, refusal rates pre/post intervention, optional extended fields). |
| `{safe_name}_lineage.json`         | Structured audit trail: git sha, resolved config after CLI overrides, per-round iterative history, causal/head scores, stage timings, runtime triple. |

---

## Quick reference — CLI flags (from `src/cli.py`)

| Flag                       | Effect |
|----------------------------|--------|
| `--config PATH`            | YAML config file (default: `configs/base.yaml`). |
| `--model HF_ID`            | Override `model.name`. |
| `--device {cpu,cuda}`      | Override `model.device`. |
| `--output-dir DIR`         | Override output directory (default `outputs`). |
| `--n-directions N`         | §5.1 — iterative subspace extraction rounds. |
| `--causal-patching`        | §5.3 — causal layer selection via activation patching. |
| `--no-causal-patching`     | Disable §5.3. |
| `--analyze-heads`          | §5.4 — attention head analysis at best layer. |
| `--no-analyze-heads`       | Disable §5.4. |
| `--calibration`            | §5.6 — calibrated orthogonalization (adjusts RMSNorm γ). |
| `--no-calibration`         | Disable §5.6. |
| `--xstest`                 | §3.5 — XSTest over-refusal evaluation. |
| `--no-xstest`              | Disable XSTest. |
| `--strongreject`           | §3.5 — StrongREJECT bypass evaluation. |
| `--no-strongreject`        | Disable StrongREJECT. |
| `--post-quant-correct`     | §3.6 — post-quantization ortho correction. |

---

## Quick reference — important config keys (from `configs/base.yaml`)

```yaml
model:
  name: "Qwen/Qwen1.5-1.8B-Chat"
  dtype: "bfloat16"        # float32 for CPU
  device: "cuda"           # cpu for CPU-only

data:
  n_train: 128             # shrink for smoke test
  n_val: 32
  n_bypass_eval: 100
  n_induce_eval: 100

extraction:
  n_directions: 3          # §5.1 iterative rounds
  kl_score_max: 0.1        # §C.1 selection filter
  max_layer_frac: 0.8      # reject direction from top 20% of layers
  use_causal_patching: false
  analyze_heads: false
  use_calibration: false
  auxiliary_directions_path: ""   # §6.4 cross-objective projection (leave empty for paper-faithful)

evaluation:
  load_xstest_eval: false
  load_strongreject_eval: false
  use_official_strongreject: false    # §9.1 — requires `pip install strongreject`
  run_capability_benchmarks: false    # §9.2 — requires `pip install lm-eval`
  capability_tasks: ["mmlu", "arc_challenge", "gsm8k"]
```

---

## Optional round-2 features (require extra installs)

### §9.1 Real StrongREJECT judge

```bash
pip install strongreject
```
Then set `evaluation.use_official_strongreject: true` in the config.
Falls back to the keyword-stub judge if the package is missing.

### §9.2 Capability benchmarks (MMLU / ARC / GSM8K)

```bash
pip install lm-eval
```
Then set `evaluation.run_capability_benchmarks: true`. **This eval takes
30–120 minutes on a 7B model** — it calls the `lm_eval` CLI twice (pre
and post orthogonalization) and computes `capability_delta_*` fields.

### §9.4 Cross-objective projection (`auxiliary_directions_path`)

This requires you to build a separate `.pt` file containing one or more
auxiliary direction tensors (e.g. honesty, helpfulness) computed by
running the DiM procedure on non-refusal datasets. Not shipped; see the
inline notes in `configs/base.yaml` for the accepted file shapes:
- `(k, d_model)` tensor, or
- dict of `(d_model,)` tensors, or
- list of `(d_model,)` tensors.

Once built, point at it:
```yaml
extraction:
  auxiliary_directions_path: "outputs/aux_directions.pt"
```
The pipeline will run modified Gram–Schmidt on the aux basis and project
every candidate refusal direction into the null space of that subspace
before scoring — decoupling refusal from the collateral objectives.

### §9.3 SAE feature ablation

`src/interventions.py::sae_feature_ablation` is a library function; it
is **not** wired through any config or CLI surface. To use it, call it
programmatically after `run_pipeline`. Requires `pip install sae-lens`.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`** — you are not in the
`refusal_direction/` directory. Re-run Step 0.

**`OSError: We couldn't connect to huggingface.co`** — check your
network / proxy. Datasets and models are fetched on first run.

**`OSError: You are trying to access a gated repo`** — you are trying to
use a gated model (Llama, Gemma) without a HuggingFace token. Run
`huggingface-cli login` and paste a read-scope token.

**`RuntimeError: CUDA out of memory`** — your GPU is too small for the
model + activations + gradients. Switch to a smaller model, set
`model.dtype: "float16"`, or move to CPU (`model.device: "cpu"`).

**`kl_score > 0.1` in results** — the selection filter should have
rejected this. Either `extraction.kl_score_max` was overridden in the
config, or `select_best` is filtering incorrectly. Check
`_lineage.json["resolved_config"]` to see which.

**`orth_refusal_rate` differs sharply from `bypass_refusal_rate_intervened`** —
the two interventions should be equivalent up to numerical noise (paper
§4.1 / §E). Sharp disagreement means the weight-tying guard or a norm
helper is broken. Do not trust any metric from this run.

**Pipeline reports `n_directions_found = 1` when config says 3** — the
iterative extractor converged early (bypass-score delta fell below
`extraction.delta_threshold`, or cosine similarity to a prior direction
exceeded `extraction.cosine_dedup_threshold`). This is expected
behavior on small models.

---

## What this guide does not verify

- Whether HuggingFace datasets and models actually download without
  rate-limit or schema drift *today*. Only a real run tells you that.
- Whether `Qwen/Qwen1.5-1.8B-Chat` still loads on the current
  `transformers` version — the model is public, but library churn is a
  real risk.
- Actual wall-clock timings on *your* specific CPU.
- Whether the pytest test suite actually passes end-to-end — the
  modules compile, but compile ≠ works.

The only way to resolve these unknowns is to run Step 8 and read the
output. If something breaks, fix the guide from evidence rather than
from reading code.
