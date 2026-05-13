"""
Wrapper around a HuggingFace causal LM that exposes the residual stream for
activation capture and for hook-based interventions.

Paper: https://arxiv.org/abs/2406.11717
§2.1 Background — residual-stream notation x^(l)_i ∈ R^d_model.
§2.3 — activations are captured per layer l and post-instruction token position i.
§2.4 — interventions are implemented as hooks on the residual stream.
§4.1 — weight orthogonalization modifies matrices that write to the residual stream.

Design note: we hook the *input* of each transformer decoder layer, which is
the residual-stream activation going into that layer (pre-norm or post-norm,
the definition of "residual stream" in the paper). This matches the paper's
convention where x^(l) is the pre-layer residual (§2.1 background description
of the transformer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .utils import detect_family, format_prompt, resolve_device, resolve_dtype


# -----------------------------------------------------------------------------
# Decoder-layer discovery
# -----------------------------------------------------------------------------
def _get_decoder_layers(model: PreTrainedModel) -> nn.ModuleList:
    """Return the ModuleList of transformer decoder layers for this model.

    Covers the architectures the paper studies (Llama, Qwen, Gemma, Yi are all
    Llama-style; the path is `model.model.layers`). [UNSPECIFIED] — the paper
    does not dictate a specific Python hook point; the input of each layer is
    the canonical residual-stream location.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Fallback paths for other HF architectures.
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError(
        f"Cannot locate decoder layers on {type(model).__name__}; "
        "extend _get_decoder_layers() to support this architecture."
    )


# -----------------------------------------------------------------------------
# Residual-stream writer discovery (§4.1)
# -----------------------------------------------------------------------------
@dataclass
class ResidualWriters:
    """Handles to every parameter that writes to the residual stream.

    §4.1 — "the matrices that write to the residual stream are: the embedding
    matrix, the positional embedding matrix, attention out matrices, and MLP
    out matrices. Orthogonalizing all of these matrices, as well as any output
    biases, with respect to the direction r̂ effectively prevents the model
    from ever writing r̂ to its residual stream."

    Llama-family models (including Qwen, Gemma, Yi) use RoPE and therefore
    have no learned positional embedding matrix — that is a correct omission,
    not an ambiguity.
    """

    embed: nn.Embedding
    attn_out_weights: List[nn.Linear] = field(default_factory=list)
    attn_out_biases: List[torch.nn.Parameter] = field(default_factory=list)
    mlp_out_weights: List[nn.Linear] = field(default_factory=list)
    mlp_out_biases: List[torch.nn.Parameter] = field(default_factory=list)
    positional_embed: Optional[nn.Embedding] = None


def discover_residual_writers(model: PreTrainedModel) -> ResidualWriters:
    """Find every weight that writes to the residual stream.

    We look for the standard names used in Llama-style configs:
      - `embed_tokens` — token embedding
      - `self_attn.o_proj` — attention output projection (writes to residual)
      - `mlp.down_proj` — MLP down projection (writes to residual)
    Biases along these paths are collected separately so orthogonalization can
    zero out the ˆr component of each bias too.
    """
    writers = ResidualWriters(embed=model.get_input_embeddings())
    # Positional embedding matrix if the model has one (§4.1).
    pe = getattr(getattr(model, "model", model), "embed_positions", None)
    if isinstance(pe, nn.Embedding):
        writers.positional_embed = pe

    for layer in _get_decoder_layers(model):
        # Attention output projection.
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        if attn is not None:
            o_proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
            if isinstance(o_proj, nn.Linear):
                writers.attn_out_weights.append(o_proj)
                if o_proj.bias is not None:
                    writers.attn_out_biases.append(o_proj.bias)
        # MLP output / down projection.
        mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if mlp is not None:
            down = getattr(mlp, "down_proj", None) or getattr(mlp, "c_proj", None)
            if isinstance(down, nn.Linear):
                writers.mlp_out_weights.append(down)
                if down.bias is not None:
                    writers.mlp_out_biases.append(down.bias)

    return writers


# -----------------------------------------------------------------------------
# §3.2 — Get Final Norm Layer
# -----------------------------------------------------------------------------

def get_final_norm_layer(model: PreTrainedModel) -> Optional[nn.Module]:
    """Find the final RMSNorm / LayerNorm before lm_head.

    §3.2 — After the last decoder layer, the residual stream passes through
    a final normalization layer (RMSNorm in Llama-family, LayerNorm in GPT-2)
    before reaching lm_head. This is the hook point for the §3.2 fix that
    closes the "last-layer gap" in directional ablation. Shared by
    ``interventions.directional_ablation`` and ``multi_directional_ablation``.
    """
    # Path 1: Llama-style (hf.model is the inner LlamaModel)
    inner = getattr(model, "model", None)
    if inner is not None:
        norm = getattr(inner, "norm", None)
        if norm is not None:
            return norm

    # Path 2: GPT-2 style (hf.transformer.ln_f)
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        norm = getattr(transformer, "ln_f", None)
        if norm is not None:
            return norm

    # Path 3: Direct norm on hf (some tiny models)
    return getattr(model, "norm", None)


# -----------------------------------------------------------------------------
# §3.3 — Analyze Weight Tying
# -----------------------------------------------------------------------------

def analyze_weight_tying(model: PreTrainedModel) -> Dict[str, bool]:
    """Check if embed_tokens and lm_head share weights.

    §3.3 — In weight-tied models (many Llama variants, Qwen, etc.),
    ``embed_tokens.weight`` and ``lm_head.weight`` share the same underlying
    storage.  Orthogonalizing embed_tokens would *also* modify lm_head, which
    is a residual-stream *reader* — not covered by §E's proof.

    Knowing whether weight tying is active is essential for correctly handling
    embedding orthogonalization and the weight-tying guard in
    ``orthogonalize_weights()``.

    Args:
        model: A HuggingFace PreTrainedModel.

    Returns:
        Dict with keys:
        - ``is_tied``: ``True`` if embed_tokens and lm_head share storage.
        - ``embed_exists``: ``True`` if the model has an embed_tokens layer.
        - ``lm_head_exists``: ``True`` if the model has an lm_head layer.
    """
    embed = model.get_input_embeddings()
    lm_head = getattr(model, "lm_head", None)

    embed_exists = embed is not None
    lm_head_exists = lm_head is not None

    is_tied = False
    if embed_exists and lm_head_exists:
        if hasattr(lm_head, "weight") and hasattr(embed, "weight"):
            is_tied = (lm_head.weight.data_ptr() == embed.weight.data_ptr())

    print(
        f"[weight tying] embed_exists={embed_exists}, "
        f"lm_head_exists={lm_head_exists}, is_tied={is_tied}"
    )

    return {
        "is_tied": is_tied,
        "embed_exists": embed_exists,
        "lm_head_exists": lm_head_exists,
    }


# -----------------------------------------------------------------------------
# §3.3 — Embedding Refusal Leakage Check
# -----------------------------------------------------------------------------

def check_embedding_refusal_leakage(
    model: RefusalModel,
    harmful_train: List[str],
    direction: torch.Tensor,
    n_samples: int = 20,
) -> float:
    """Measure mean |r̂ · embed(tokens)| for harmful prompt tokens.

    §3.3 — After applying the weight-tying guard, embed_tokens is orthogonalized
    (r̂ component removed).  However, if the model was *not* weight-tied and
    embed_tokens ortho was skipped for some reason, this function measures how
    much r̂ component the raw embeddings contribute for typical harmful prompts.

    Lives in model.py because it operates on model-level concepts (embeddings,
    tokenization); the single source of truth used across the pipeline.

    If the returned value > 0.01, embedding orthogonalization matters for this
    model and the refusal direction has a non-negligible projection onto the
    raw token embeddings.

    Args:
        model: The wrapped model.
        harmful_train: List of harmful instruction strings (training set).
        direction: The refusal direction vector (need not be unit-norm).
        n_samples: Number of prompts to sample.

    Returns:
        Mean absolute dot product |r̂ · embed(t)| averaged over all tokens
        in the sampled prompts.
    """
    r_hat = direction.to(device=model.device, dtype=torch.float32)
    r_hat = r_hat / r_hat.norm()
    embed_fn = model.model.get_input_embeddings()
    leakages: List[float] = []

    with torch.no_grad():
        for prompt in harmful_train[:n_samples]:
            input_ids = model.tokenize([model.format(prompt)])["input_ids"][0]
            embeds = embed_fn(input_ids).float()  # (seq, d_model)
            dots = (embeds @ r_hat).abs()  # (seq,)
            leakages.append(dots.mean().item())

    mean_leakage = float(torch.tensor(leakages).mean()) if leakages else 0.0
    print(f"[leakage] mean |r̂ · embed(tokens)| = {mean_leakage:.6f} "
          f"(n={min(n_samples, len(harmful_train))} prompts)")
    return mean_leakage


# -----------------------------------------------------------------------------
# Wrapper
# -----------------------------------------------------------------------------
class RefusalModel:
    """Thin wrapper that owns the HF model + tokenizer and exposes the hook
    points needed by extraction.py and interventions.py.
    """

    def __init__(
        self,
        name: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        quantization: str = None,
    ) -> None:
        """Initialize RefusalModel.

        Args:
            name: HuggingFace model name.
            dtype: Model dtype (bfloat16, float16, float32).
            device: Device (cuda, cpu).
            quantization: Optional quantization ('8bit', '4bit'). Requires bitsandbytes.
        """
        self.name = name
        self.family = detect_family(name)
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype)
        self.quantization = quantization

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            name, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            # [UNSPECIFIED] pad token not discussed; we reuse EOS which matches
            # HuggingFace's default for left-padded causal LM batching.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Load model with optional quantization
        self.model = self._load_model(name, quantization)
        self.model.eval()

        self.layers = _get_decoder_layers(self.model)
        self.n_layers = len(self.layers)
        self.d_model = self.model.config.hidden_size

    def _load_model(self, name: str, quantization: str = None) -> PreTrainedModel:
        """Load model with optional quantization support (US-006)."""
        if quantization is None:
            return AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            ).to(self.device)

        # Quantization mode - requires bitsandbytes
        # Handle different import paths for different versions
        try:
            from bitsandbytes import BitsAndBytesConfig
        except ImportError:
            try:
                from bitsandbytes.nn import BitsAndBytesConfig
            except ImportError:
                raise ImportError(
"quantization requires `bitsandbytes>=0.41.0` with BitsAndBytesConfig. "
                    "Current version lacks this class. Install: pip install --upgrade bitsandbytes"
                )

        if quantization == "8bit":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            raise ValueError(f"Unknown quantization: {quantization}. Use '8bit' or '4bit'")

        return AutoModelForCausalLM.from_pretrained(
            name,
            quantization_config=quant_config,
            trust_remote_code=True,
        )

    # -------------------------------------------------------------------------
    # Prompt handling
    # -------------------------------------------------------------------------
    def format(self, instruction: str) -> str:
        """Apply the paper's chat template (§C.3, Table 6)."""
        return format_prompt(instruction, self.family, self.tokenizer)

    def tokenize(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of already-formatted prompts (left-padded)."""
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in enc.items()}

    # -------------------------------------------------------------------------
    # Hook helpers
    # -------------------------------------------------------------------------
    def register_forward_pre_hooks(
        self,
        make_hook: Callable[[int], Callable],
    ) -> List[torch.utils.hooks.RemovableHandle]:
        """Register a forward-pre-hook on every transformer block.

        §2.4 — directional ablation "at every activation x^(l)_i ... across all
        layers l and all token positions i". A forward-pre-hook on each block
        sees the residual-stream tensor going into that block and can mutate
        it before the block runs.
        """
        handles: List[torch.utils.hooks.RemovableHandle] = []
        for layer_idx, layer in enumerate(self.layers):
            handles.append(layer.register_forward_pre_hook(make_hook(layer_idx)))
        return handles

    @staticmethod
    def remove_hooks(handles: List[torch.utils.hooks.RemovableHandle]) -> None:
        for h in handles:
            h.remove()

    # -------------------------------------------------------------------------
    # §5.6 — Measure Layer Input RMS
    # -------------------------------------------------------------------------
    def measure_layer_input_rms(
        self,
        prompts: List[str],
        batch_size: int = 8,
    ) -> List[float]:
        """Capture mean RMS of residual stream at each layer input.

        §5.6 — This is used by calibrated orthogonalization to measure how
        much the residual stream scale changes after weight orthogonalization,
        and to compensate via RMSNorm γ adjustments.

        Args:
            prompts: Calibration prompt strings (typically harmless).
            batch_size: Number of prompts per forward pass.

        Returns:
            List of length ``self.n_layers`` with the mean RMS at each layer.
            Layers with no recorded activations default to ``1.0`` (no scaling).
        """
        rms_per_layer: List[List[float]] = [[] for _ in range(self.n_layers)]

        def make_hook(l: int):
            def hook(_module, inputs):
                x = inputs[0]  # (batch, seq, d_model)
                # RMS = sqrt(mean(x²)) over d_model, then mean over batch & seq.
                rms_per_layer[l].append(
                    x.float().pow(2).mean(dim=-1).sqrt().mean().item()
                )
            return hook

        handles = self.register_forward_pre_hooks(make_hook)
        try:
            for i in range(0, len(prompts), batch_size):
                batch = [self.format(p) for p in prompts[i : i + batch_size]]
                enc = self.tokenize(batch)
                with torch.no_grad():
                    self.model(**enc)
        finally:
            self.remove_hooks(handles)

        return [
            float(sum(v) / len(v)) if v else 1.0
            for v in rms_per_layer
        ]
