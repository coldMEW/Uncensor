# Reproduction Notes: Refusal in Language Models Is Mediated by a Single Direction

> This document records every implementation choice, whether it was specified by
> the paper, and what alternatives exist. **Read this before trusting any number
> this code produces.**

---

## Paper

- **Title:** Refusal in Language Models Is Mediated by a Single Direction
- **Authors:** Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda
- **Year:** 2024 (NeurIPS)
- **ArXiv:** https://arxiv.org/abs/2406.11717
- **Official code:** https://github.com/andyrdt/refusal_direction

---

## What this implements

The full pipeline: dataset construction (§2.2) → residual-stream activation
capture (§2.1, §2.3) → difference-in-means direction extraction (§2.3) →
direction selection over the `(layer × post-instruction-position)` grid (§C.1)
→ three interventions (directional ablation §2.4, activation addition §2.4,
weight orthogonalization §4.1) → evaluation via greedy generation and
`refusal_score` string matching (§D.1), with an optional Llama Guard 2
`safety_score` (§D.2). No training is involved.

---

## Verified against

- [x] Paper equations (§2.4 Eq. 4, §4.1 Eq. 5)
- [x] §C.1 direction selection algorithm (prose pseudocode)
- [x] §B refusal metric definition
- [x] §D.1 refusal substring list (Figure 12)
- [x] Official repo layout: https://github.com/andyrdt/refusal_direction (used as a reference for ambiguity resolution; no code transcribed)
- [ ] End-to-end reproduction of paper numbers — not run as part of this scaffold

---

## Unspecified choices

| Component | Our Choice | Alternatives | Paper Quote | Section |
|-----------|------------|--------------|-------------|---------|
| Post-instruction positions `I` | `[-5, -4, -3, -2, -1]` | `[-1]` only, `[-1, -2]` | "post-instruction token position i ∈ I" — `I` undefined; Table 5 `i*` uses `-1..-5` | §2.3, §C.3 |
| Model dtype | `bfloat16` | `float32`, `float16` | — | — |
| Activation extraction batch size | 8 | 1 (memory-light), 32 (faster) | — | — |
| Sampling seed for dataset splits | 42 | any | — | — |
| `max_new_tokens` for generation | 256 `[FROM_OFFICIAL_CODE]` | 128, 512 | — | — |
| Generation batch size | 4 | higher with more VRAM | — | — |
| Pad token | reuse EOS | — | — | — |
| Clamp epsilon in log-odds | `1e-8` | `1e-6` | — | §B |
| Alpaca row filter | drop rows with non-empty `input` field | keep all | — | §2.2 |

---

## Partially specified, resolved from official code

| Component | Paper says | Our value | Resolution |
|-----------|------------|-----------|------------|
| Token positions `I` | undefined in §2.3 | `[-5..-1]` | `[FROM_OFFICIAL_CODE]` — repo iterates the last 5 positions of the formatted prompt |
| Greedy vs sampled generation | "generating a full completion using greedy decoding" (§B) | `do_sample=False` | `[SPECIFIED]` by §B + repo defaults |
| Activation hook point | "residual stream activations" — Python hook location unspecified | forward-pre-hook on each transformer block | canonical residual-stream location (pre-block = post previous block's residual add) |

---

## Known deviations

| Deviation | Paper says | We do | Reason |
|-----------|------------|-------|--------|
| TDC2023 source | ADVBENCH + MALICIOUSINSTRUCT + TDC2023 + HARMBENCH | ADVBENCH + MALICIOUSINSTRUCT + HARMBENCH | No HF-hosted TDC2023 mirror; HarmBench is TDC2023's direct successor / superset |
| Positional-embedding orthogonalization | §4.1 lists "positional embedding matrix" | skipped for Llama/Qwen/Gemma/Yi | These models use RoPE and have no learned positional embedding — correct omission, not an ambiguity |
| Activation addition scaling | `x ← x + r` (un-normalized) | implemented exactly as §2.4 says | flagged because some readers would assume `r̂ · c` |

---

## Expected results

Reproducing the paper's headline numbers is hardware-expensive and out of
scope for this scaffold. Qualitative sanity checks you should see on a small
Qwen/Llama chat model after the pipeline runs:

| Metric | Expected trend | Paper anchor |
|--------|----------------|--------------|
| `bypass_refusal_rate` baseline | ~0.90–1.00 (model refuses) | Figure 1 |
| `bypass_refusal_rate` intervened (directional ablation) | ~0.00–0.10 (model complies) | Figure 1 |
| `induce_refusal_rate` baseline | ~0.00–0.10 (model complies) | Figure 3 |
| `induce_refusal_rate` intervened (activation addition) | ~0.80–1.00 (model refuses harmless) | Figure 3 |
| `orth_refusal_rate` | ≈ `bypass_refusal_rate` intervened | §4.1, §E equivalence |
| `bypass_score` at selected direction | strongly negative, e.g. `-4..-14` | Table 5 |
| `induce_score` at selected direction | `> 0`, e.g. `1..7` | Table 5 |
| `kl_score` at selected direction | `< 0.1` | Table 5 |

**Note:** Exact paper numbers (Table 5 per-model values) require matching all
unspecified choices above and running on the exact model weights the paper
used. Small deviations are normal.

---

## Debugging tips

1. **bypass_refusal_rate not dropping under ablation** — double-check you are
   hooking every decoder layer (not just one) and that the ablation subtracts
   the projection along `r̂`, not `r`. §2.4 Eq. 4 uses the unit vector.
2. **induce_refusal_rate not rising under addition** — activation addition
   uses the **un-normalized** `r` and is applied only at the extraction layer
   `l*`, not at every layer. §2.4 is explicit on both points.
3. **Weight orthogonalization produces different numbers than directional
   ablation** — verify you orthogonalized every `o_proj`, every `down_proj`,
   and the `embed_tokens` weights. Missing any one of these breaks the
   equivalence proven in §E.
4. **`kl_score` is huge for every candidate** — you may be capturing
   activations at the wrong token position. The post-instruction tokens are
   the chat-template suffix tokens (e.g. `<|eot_id|>`, `\n\n`), which means
   `[-5..-1]` on the formatted prompt, not on the raw instruction.
5. **`refusal_score` is 1 on every completion even after ablation** — check
   the refusal substring list is case-insensitive and operates on `completion.lower()`.

---

## Scope decisions

### Implemented
- Difference-in-means extraction — core contribution (§2.3).
- Direction selection with the three filters — core contribution (§C.1).
- Directional ablation — core contribution (§2.4).
- Activation addition — core contribution (§2.4).
- Weight orthogonalization — core contribution (§4.1).
- Refusal metric, refusal_score, safety_score — evaluation (§B, §D).
- Full data pipeline for AdvBench / MaliciousInstruct / HarmBench / Alpaca /
  JailbreakBench — `full` mode requirement.
- End-to-end CLI driver.
- Walkthrough notebook.

### Intentionally excluded
- GCG / PAIR / other adversarial-suffix baselines (§4.2, §5) — baselines, not
  the contribution. Reference `nanogcg` or the paper's repo if you need them.
- MMLU / ARC / GSM8K / TruthfulQA coherence evaluation (§4.3, §G) — should
  be run via `lm-evaluation-harness` against the orthogonalized checkpoint.
- Fine-tuning-based refusal removal (§I.2) — comparison, not the contribution.
- Adversarial suffix mechanistic analysis (§5) — separate research question.
- Distributed inference / sharding for 70B+ models — engineering only.

### Needed for full reproduction (not included)
- Hardware: an 80GB GPU for the larger models. §N ("compute statement")
  describes their setup.
- Llama Guard 2 weights for `safety_score` — gated on HuggingFace; set
  `with_safety_score: true` and log in with `huggingface-cli`.
- TDC2023-only prompts (not individually sourced; HarmBench covers them).

---

## References

- Arditi et al., 2024 — the paper itself. https://arxiv.org/abs/2406.11717
- Belrose, 2023 — origin of difference-in-means (referenced in §2.3).
- Marks & Tegmark, 2023 — prior application of difference-in-means for
  feature directions (referenced in §2.3).
- Chao et al., 2024 — JailbreakBench (§3.1 evaluation set).
- Meta Llama Guard 2 model card (§D.2 evaluator).
- Zou et al., 2023b — AdvBench.
- Taori et al., 2023 — Alpaca.
- Mazeika et al., 2024 — HarmBench.
- Huang et al., 2023 — MaliciousInstruct.
