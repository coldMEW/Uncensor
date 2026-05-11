"""
Generation utilities: run a batch of prompts through the model and return the
completion strings, optionally under an intervention context manager.

Paper: https://arxiv.org/abs/2406.11717
§B — "generating a full completion using greedy decoding".
§3.1, §3.2 — generation under interventions.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, ContextManager, List, Optional, Sequence

import torch
from tqdm.auto import tqdm

from .model import RefusalModel


def generate_batched(
    model: RefusalModel,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    do_sample: bool,
    batch_size: int,
    intervention: Optional[Callable[[], ContextManager]] = None,
) -> List[str]:
    """Generate completions for a list of raw (not-yet-templated) instructions.

    §B — greedy decoding by default.
    §3.1 / §3.2 — generation happens under a directional_ablation or
    activation_addition context manager (passed as `intervention`).

    Args:
        prompts: raw user instructions (no chat template applied yet).
        intervention: zero-arg callable that returns a context manager, e.g.
            `lambda: directional_ablation(model, r)`. If None, runs vanilla.
    """
    completions: List[str] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="generate"):
        batch = [model.format(p) for p in prompts[start : start + batch_size]]
        enc = model.tokenize(batch)
        ctx = intervention() if intervention is not None else nullcontext()
        with ctx, torch.no_grad():
            output = model.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=model.tokenizer.pad_token_id,
            )
        # Strip the prompt tokens so only the completion remains.
        prompt_len = enc["input_ids"].shape[1]
        for row in output:
            completion_ids = row[prompt_len:]
            text = model.tokenizer.decode(completion_ids, skip_special_tokens=True)
            completions.append(text)
    return completions
