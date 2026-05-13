"""
Dataset loaders for harmful / harmless / evaluation splits.

Paper: https://arxiv.org/abs/2406.11717
§2.2 — dataset sources.
§A — dataset details.

All loaders return `list[str]` of raw instructions. Prompt formatting is done
by `src/utils.py::format_prompt` at call sites so the datasets stay
model-agnostic.

These loaders call HuggingFace `datasets`. They DO NOT auto-download anything
beyond the default HF cache; users may need `huggingface-cli login` for gated
datasets. Full reproduction requires the four harmful sources named in §2.2;
TDC2023 is the precursor of HarmBench so HarmBench alone is a reasonable
superset.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from datasets import load_dataset

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Harmful instructions (§2.2)
# -----------------------------------------------------------------------------
def _load_advbench() -> List[str]:
    """ADVBENCH harmful behaviors (Zou et al., 2023b), §2.2."""
    ds = load_dataset("walledai/AdvBench", split="train")
    # Column is "prompt" on walledai/AdvBench.
    col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
    return [row[col] for row in ds]


def _load_malicious_instruct() -> List[str]:
    """MALICIOUSINSTRUCT (Huang et al., 2023), §2.2."""
    ds = load_dataset("walledai/MaliciousInstruct", split="train")
    col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
    return [row[col] for row in ds]


def _load_harmbench() -> List[str]:
    """HARMBENCH (Mazeika et al., 2024), §2.2. Covers TDC2023 as well.

    The walledai/HarmBench dataset ships with a "standard" config on some
    versions of the HF Hub. We try that first, then fall back to the default
    (no config name) so the loader works regardless of the hub snapshot.
    """
    col_candidates = ("prompt", "Behavior", "behavior", "instruction")
    split_candidates = ("train", "test", "validation")

    # Try with "standard" config (original paper2code default).
    for config in ("standard", None):
        for split in split_candidates:
            try:
                load_kwargs = {"split": split}
                if config is not None:
                    ds = load_dataset("walledai/HarmBench", config, **load_kwargs)
                else:
                    ds = load_dataset("walledai/HarmBench", **load_kwargs)
                # Find the right column.
                col = next((c for c in col_candidates if c in ds.column_names), None)
                if col is None:
                    col = ds.column_names[0]
                return [row[col] for row in ds]
            except Exception:
                continue

    # Last-resort: load without any split specification.
    try:
        ds = load_dataset("walledai/HarmBench")
        # datasets returns a DatasetDict; grab the first split.
        split_key = list(ds.keys())[0]
        ds = ds[split_key]
        col = next((c for c in col_candidates if c in ds.column_names), ds.column_names[0])
        return [row[col] for row in ds]
    except Exception as exc:
        raise RuntimeError(
            "Failed to load walledai/HarmBench. "
            "Try `huggingface-cli login` or check the dataset page."
        ) from exc


HARMFUL_LOADERS = {
    "walledai/AdvBench": _load_advbench,
    "walledai/MaliciousInstruct": _load_malicious_instruct,
    "walledai/HarmBench": _load_harmbench,
}


def load_harmful(
    sources: Sequence[str],
    seed: int,
    *,
    allow_partial_sources: bool = False,
) -> List[str]:
    """Load and shuffle the union of the specified harmful sources (§2.2).

    Some public benchmark mirrors are gated or intermittently unavailable on
    Kaggle. When ``allow_partial_sources`` is enabled, one failed source is
    logged and skipped instead of collapsing the whole dataset-scale run.
    """
    prompts: List[str] = []
    for src in sources:
        if src not in HARMFUL_LOADERS:
            raise ValueError(f"Unknown harmful source: {src}")
        try:
            loaded = HARMFUL_LOADERS[src]()
        except Exception as exc:
            if not allow_partial_sources:
                raise
            logger.warning("Skipping harmful source %s after load failure: %s", src, exc)
            continue
        prompts.extend(loaded)
    # Deduplicate on exact string.
    prompts = list(dict.fromkeys(prompts))
    if not prompts:
        raise RuntimeError(
            "No harmful prompts loaded from requested sources. "
            "Check dataset access or disable partial loading to see the first source error."
        )
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


# -----------------------------------------------------------------------------
# Harmless instructions (§2.2 — Alpaca)
# -----------------------------------------------------------------------------
def load_harmless(seed: int) -> List[str]:
    """Load harmless instructions from ALPACA (§2.2, Taori et al., 2023).

    We keep only rows whose `input` field is empty — these are pure instructions
    and match the style of the harmful sets. [UNSPECIFIED] — the paper does not
    say whether it keeps rows with inputs; we exclude them for cleanliness.
    """
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = [row["instruction"] for row in ds if not row["input"].strip()]
    prompts = list(dict.fromkeys(prompts))
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


# -----------------------------------------------------------------------------
# Evaluation sets
# -----------------------------------------------------------------------------
def load_jailbreakbench(n: int) -> List[str]:
    """JAILBREAKBENCH harmful behaviors (Chao et al., 2024), §3.1 — used as the
    bypass-refusal evaluation set.

    The JailbreakBench dataset is available on HuggingFace as
    `JailbreakBench/JBB-Behaviors`. We try the canonical config/split, then
    fall back gracefully so the loader works across hub versions.
    """
    goal_col_candidates = ("Goal", "goal", "prompt", "instruction", "behavior", "Behavior")

    # Canonical: config="behaviors", split="harmful" (original paper2code default).
    attempts = [
        dict(name="JailbreakBench/JBB-Behaviors", config="behaviors", split="harmful"),
        dict(name="JailbreakBench/JBB-Behaviors", config=None, split="harmful"),
        dict(name="JailbreakBench/JBB-Behaviors", config="behaviors", split="train"),
        dict(name="JailbreakBench/JBB-Behaviors", config=None, split="train"),
    ]

    for attempt in attempts:
        try:
            name = attempt["name"]
            config = attempt["config"]
            split = attempt["split"]
            if config is not None:
                ds = load_dataset(name, config, split=split)
            else:
                ds = load_dataset(name, split=split)
            col = next((c for c in goal_col_candidates if c in ds.column_names), None)
            if col is None:
                col = ds.column_names[0]
            prompts = [row[col] for row in ds]
            return prompts[:n]
        except Exception:
            continue

    # Last resort: load without specifying a split.
    try:
        ds_dict = load_dataset("JailbreakBench/JBB-Behaviors")
        # Try to find a split that looks "harmful" or just take the first.
        split_key = next(
            (k for k in ds_dict.keys() if "harm" in k.lower()),
            list(ds_dict.keys())[0],
        )
        ds = ds_dict[split_key]
        col = next((c for c in goal_col_candidates if c in ds.column_names), ds.column_names[0])
        return [row[col] for row in ds][:n]
    except Exception as exc:
        raise RuntimeError(
            "Failed to load JailbreakBench/JBB-Behaviors. "
            "Check your network / HF cache, or try `huggingface-cli login`."
        ) from exc


# -----------------------------------------------------------------------------
# XSTest safe prompts (§3.5, §5.3 Step 3)
# -----------------------------------------------------------------------------
def load_xstest() -> List[str]:
    """Load XSTest safe prompts (Röttger et al. 2023).

    XSTest has 250 safe prompts that superficially resemble harmful ones but
    have benign intents.  These are used as an additional safety evaluation to
    ensure the model does not refuse benign instructions that *sound* dangerous
    (§3.5, §5.3 Step 3).

    Returns
    -------
    list[str]
        Safe instruction strings from the XSTest benchmark.
    """
    # Known HuggingFace repository names for XSTest, in priority order.
    repo_candidates = [
        "xai-org/xstest",
        "hlilab/xstest",
        "AlexWortega/xstest",
    ]

    prompt_col_candidates = ("prompt", "question", "instruction", "text")
    type_col_candidates = ("type", "category", "label", "split")

    for repo in repo_candidates:
        for split in ("train", "test", "validation"):
            try:
                ds = load_dataset(repo, split=split)
            except Exception:
                continue

            # Find the prompt column.
            col = next(
                (c for c in prompt_col_candidates if c in ds.column_names),
                None,
            )
            if col is None:
                col = ds.column_names[0]

            # If there is a "type" column, filter to only "safe" rows.
            type_col = next(
                (c for c in type_col_candidates if c in ds.column_names),
                None,
            )
            if type_col is not None:
                safe_prompts = [
                    row[col] for row in ds
                    if str(row[type_col]).strip().lower() == "safe"
                ]
            else:
                # No type column — return everything and hope for the best.
                safe_prompts = [row[col] for row in ds]

            if safe_prompts:
                return safe_prompts

    # Last resort: try loading without specifying a split.
    for repo in repo_candidates:
        try:
            ds_dict = load_dataset(repo)
            split_key = list(ds_dict.keys())[0]
            ds = ds_dict[split_key]
            col = next(
                (c for c in prompt_col_candidates if c in ds.column_names),
                ds.column_names[0],
            )
            type_col = next(
                (c for c in type_col_candidates if c in ds.column_names),
                None,
            )
            if type_col is not None:
                safe_prompts = [
                    row[col] for row in ds
                    if str(row[type_col]).strip().lower() == "safe"
                ]
            else:
                safe_prompts = [row[col] for row in ds]
            if safe_prompts:
                return safe_prompts
        except Exception:
            continue

    logger.warning(
        "Could not load XSTest from any known HuggingFace repository "
        "(tried: %s). Returning empty list.",
        repo_candidates,
    )
    return []


# -----------------------------------------------------------------------------
# StrongREJECT evaluation prompts (§3.5)
# -----------------------------------------------------------------------------
def load_strongreject(n: int = 100) -> List[str]:
    """Load StrongREJECT evaluation prompts (arXiv:2402.10260).

    StrongREJECT is a harder bypass benchmark than JailbreakBench, focusing
    on multi-turn and obfuscated attack strategies (§3.5).

    Parameters
    ----------
    n:
        Maximum number of prompts to return.

    Returns
    -------
    list[str]
        Harmful instruction strings from the StrongREJECT benchmark.
    """
    # Known HuggingFace repository names / configs for StrongREJECT.
    repo_candidates = [
        "strongreject/strongreject_prompts",
        "allenai/strongreject",
        "walledai/StrongREJECT",
    ]

    prompt_col_candidates = ("prompt", "question", "instruction", "forbidden_prompt", "text")
    split_candidates = ("train", "test", "validation")

    for repo in repo_candidates:
        for config in (None,):
            for split in split_candidates:
                try:
                    if config is not None:
                        ds = load_dataset(repo, config, split=split)
                    else:
                        ds = load_dataset(repo, split=split)
                except Exception:
                    continue

                col = next(
                    (c for c in prompt_col_candidates if c in ds.column_names),
                    None,
                )
                if col is None:
                    col = ds.column_names[0]

                prompts = [row[col] for row in ds]
                if prompts:
                    return prompts[:n]

    # Last resort: load without specifying a split.
    for repo in repo_candidates:
        try:
            ds_dict = load_dataset(repo)
            split_key = list(ds_dict.keys())[0]
            ds = ds_dict[split_key]
            col = next(
                (c for c in prompt_col_candidates if c in ds.column_names),
                ds.column_names[0],
            )
            prompts = [row[col] for row in ds]
            if prompts:
                return prompts[:n]
        except Exception:
            continue

    logger.warning(
        "Could not load StrongREJECT from any known HuggingFace repository "
        "(tried: %s). Returning empty list.",
        repo_candidates,
    )
    return []


# -----------------------------------------------------------------------------
# WildJailbreak evaluation prompts
# -----------------------------------------------------------------------------
def load_wildjailbreak(n: int = 100) -> List[str]:
    """Load WildJailbreak evaluation prompts (arXiv:2406.18510).

    WildJailbreak is a diverse collection of real-world jailbreak prompts
    gathered from online sources, covering a wide variety of attack strategies.

    Parameters
    ----------
    n:
        Maximum number of prompts to return.

    Returns
    -------
    list[str]
        Harmful instruction strings from the WildJailbreak benchmark.
    """
    # Known HuggingFace repository names for WildJailbreak.
    repo_candidates = [
        "walledai/WildJailbreak",
        "akhaliq/wildjailbreak",
        "allenai/wildjailbreak",
    ]

    prompt_col_candidates = ("prompt", "question", "instruction", "forbidden_prompt", "text")
    split_candidates = ("train", "test", "validation")

    for repo in repo_candidates:
        for split in split_candidates:
            try:
                ds = load_dataset(repo, split=split)
            except Exception:
                continue

            col = next(
                (c for c in prompt_col_candidates if c in ds.column_names),
                None,
            )
            if col is None:
                col = ds.column_names[0]

            prompts = [row[col] for row in ds]
            if prompts:
                return prompts[:n]

    # Last resort: load without specifying a split.
    for repo in repo_candidates:
        try:
            ds_dict = load_dataset(repo)
            split_key = list(ds_dict.keys())[0]
            ds = ds_dict[split_key]
            col = next(
                (c for c in prompt_col_candidates if c in ds.column_names),
                ds.column_names[0],
            )
            prompts = [row[col] for row in ds]
            if prompts:
                return prompts[:n]
        except Exception:
            continue

    logger.warning(
        "Could not load WildJailbreak from any known HuggingFace repository "
        "(tried: %s). Returning empty list.",
        repo_candidates,
    )
    return []


# -----------------------------------------------------------------------------
# Dataset splits wrapper
# -----------------------------------------------------------------------------
@dataclass
class Splits:
    """The canonical splits used throughout the pipeline.

    §2.2 — "Each dataset consists of train and validation splits of 128 and 32
    samples, respectively."
    §3.1 — 100 JailbreakBench harmful instructions for bypass eval.
    §3.2 — 100 held-out harmless instructions for induce eval.
    §3.5 — Additional optional eval sets: XSTest, StrongREJECT, WildJailbreak.
    """
    harmful_train: List[str]
    harmful_val: List[str]
    harmless_train: List[str]
    harmless_val: List[str]
    bypass_eval: List[str]
    induce_eval: List[str]
    xstest_eval: Optional[List[str]] = None
    strongreject_eval: Optional[List[str]] = None
    wildjailbreak_eval: Optional[List[str]] = None


def build_splits(
    *,
    harmful_sources: Sequence[str],
    n_train: int,
    n_val: int,
    n_bypass_eval: int,
    n_induce_eval: int,
    seed: int,
    load_xstest_eval: bool = False,
    load_strongreject_eval: bool = False,
    load_wildjailbreak_eval: bool = False,
    n_strongreject: int = 100,
    n_wildjailbreak: int = 100,
    allow_partial_sources: bool = False,
    min_partial_train: Optional[int] = None,
) -> Splits:
    """Build the paper's canonical splits (§2.2, §3.1, §3.2) with optional
    additional evaluation sets (§3.5).

    Parameters
    ----------
    harmful_sources:
        HuggingFace repo ids of harmful-instruction datasets.
    n_train:
        Size of the training split (harmful and harmless).
    n_val:
        Size of the validation split (harmful and harmless).
    n_bypass_eval:
        Number of JailbreakBench prompts for bypass evaluation.
    n_induce_eval:
        Number of harmless prompts for induce evaluation.
    seed:
        Random seed for reproducibility.
    load_xstest_eval:
        If ``True``, load XSTest safe prompts for false-positive evaluation.
    load_strongreject_eval:
        If ``True``, load StrongREJECT prompts for harder bypass evaluation.
    load_wildjailbreak_eval:
        If ``True``, load WildJailbreak prompts for diverse bypass evaluation.
    n_strongreject:
        Maximum number of StrongREJECT prompts to load.
    n_wildjailbreak:
        Maximum number of WildJailbreak prompts to load.
    allow_partial_sources:
        If ``True``, skip individual harmful sources that fail to load while
        preserving the rest of the dataset-backed run.
    min_partial_train:
        Optional lower bound for a smaller dataset-backed training split when
        partial source loading returns fewer than ``n_train + n_val`` harmful
        prompts. If omitted, split sizes remain strict.
    """
    harmful = load_harmful(
        harmful_sources,
        seed,
        allow_partial_sources=allow_partial_sources,
    )
    harmless = load_harmless(seed)
    bypass = load_jailbreakbench(n_bypass_eval)

    # Ensure train/val splits do not overlap with evaluation sets (§2.2).
    bypass_set = set(bypass)
    harmful = [p for p in harmful if p not in bypass_set]

    if len(harmful) < n_train + n_val:
        if allow_partial_sources and min_partial_train is not None:
            adjusted_train = len(harmful) - n_val
            if adjusted_train >= min_partial_train:
                logger.warning(
                    "Downshifting harmful train split from %s to %s after partial dataset loading.",
                    n_train,
                    adjusted_train,
                )
                n_train = adjusted_train
            else:
                raise RuntimeError(
                    "Not enough harmful instructions after filtering for partial split: "
                    f"{len(harmful)} available, need at least {min_partial_train + n_val}"
                )
        else:
            raise RuntimeError(
                f"Not enough harmful instructions after filtering: {len(harmful)} < {n_train + n_val}"
            )
    if len(harmless) < n_train + n_val + n_induce_eval:
        raise RuntimeError(
            f"Not enough harmless instructions: need {n_train + n_val + n_induce_eval}, got {len(harmless)}"
        )

    harmful_train = harmful[:n_train]
    harmful_val = harmful[n_train : n_train + n_val]
    harmless_train = harmless[:n_train]
    harmless_val = harmless[n_train : n_train + n_val]
    induce_eval = harmless[n_train + n_val : n_train + n_val + n_induce_eval]

    # Optional additional eval sets (§3.5).
    xstest_eval: Optional[List[str]] = None
    strongreject_eval: Optional[List[str]] = None
    wildjailbreak_eval: Optional[List[str]] = None

    if load_xstest_eval:
        try:
            xstest_eval = load_xstest()
        except Exception as exc:
            logger.warning("Failed to load XSTest eval set: %s", exc)
            xstest_eval = None

    if load_strongreject_eval:
        try:
            strongreject_eval = load_strongreject(n=n_strongreject)
        except Exception as exc:
            logger.warning("Failed to load StrongREJECT eval set: %s", exc)
            strongreject_eval = None

    if load_wildjailbreak_eval:
        try:
            wildjailbreak_eval = load_wildjailbreak(n=n_wildjailbreak)
        except Exception as exc:
            logger.warning("Failed to load WildJailbreak eval set: %s", exc)
            wildjailbreak_eval = None

    return Splits(
        harmful_train=harmful_train,
        harmful_val=harmful_val,
        harmless_train=harmless_train,
        harmless_val=harmless_val,
        bypass_eval=bypass,
        induce_eval=induce_eval,
        xstest_eval=xstest_eval,
        strongreject_eval=strongreject_eval,
        wildjailbreak_eval=wildjailbreak_eval,
    )
