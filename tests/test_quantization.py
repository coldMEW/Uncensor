"""Tests for quantization support (int8/int4).

Run with: pytest tests/test_quantization.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
from types import SimpleNamespace

import torch
from torch import nn


class TestQuantizationConfig:
    """Tests for quantization configuration."""

    def test_refusal_model_accepts_quantization_param(self):
        """Verify RefusalModel.__init__ accepts quantization parameter."""
        from src.model import RefusalModel
        import inspect
        sig = inspect.signature(RefusalModel.__init__)
        params = list(sig.parameters.keys())
        assert "quantization" in params

    def test_quantization_values_accepted(self):
        """Verify quantization accepts '8bit' and '4bit' values."""
        from src.model import RefusalModel
        import inspect
        source = inspect.getsource(RefusalModel._load_model)
        assert "8bit" in source
        assert "4bit" in source

    def test_quantization_uses_transformers_bitsandbytes_config(self, monkeypatch):
        """BitsAndBytesConfig belongs to Transformers, not bitsandbytes."""
        from src import model as model_module
        from src.model import RefusalModel

        captured = {}

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(name, **kwargs):
                captured["name"] = name
                captured.update(kwargs)
                return object()

        monkeypatch.setattr(model_module, "AutoModelForCausalLM", FakeAutoModel)

        shim = SimpleNamespace(dtype="float16", device="cuda")
        loaded = RefusalModel._load_model(shim, "test/model", quantization="8bit")

        assert loaded is not None
        assert captured["name"] == "test/model"
        assert captured["trust_remote_code"] is True
        assert captured["quantization_config"].load_in_8bit is True

    def test_quantization_raises_on_unknown(self):
        """Verify unknown quantization raises ValueError."""
        from src.model import RefusalModel

        shim = SimpleNamespace(dtype="float16", device="cuda")
        with pytest.raises(ValueError, match="Unknown quantization"):
            RefusalModel._load_model(shim, "test/model", quantization="invalid")


class TestArchitectureDiscovery:
    """Regression tests for Gemma 4 / conditional generation wrappers."""

    def test_decoder_layers_support_nested_language_model(self):
        from src.model import _get_decoder_layers

        layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        fake_model = SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(layers=layers))
        )

        assert _get_decoder_layers(fake_model) is layers

    def test_hidden_size_supports_text_config(self):
        from src.model import _get_hidden_size

        fake_model = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2560))
        )

        assert _get_hidden_size(fake_model) == 2560

    def test_tokenize_uses_embedding_device_for_device_map_models(self):
        from src.model import _get_input_device

        embedding = nn.Embedding(8, 4)
        fake_model = SimpleNamespace(get_input_embeddings=lambda: embedding)

        assert _get_input_device(fake_model, "cuda") == embedding.weight.device


class TestQuantizationDocs:
    """Tests verifying quantization documentation."""

    def test_load_model_docstring_mentions_quantization(self):
        """Verify _load_model docstring mentions quantization."""
        from src.model import RefusalModel
        doc = RefusalModel._load_model.__doc__ or ""
        assert "quantization" in doc.lower()

    def test_model_catalog_contains_quantization_info(self):
        """Verify model catalog has vram_int8 and vram_int4 fields."""
        import json
        catalog_path = Path("configs/model_catalog.json")
        with open(catalog_path) as f:
            catalog = json.load(f)

        # Check first tier has int8/int4 info
        first_tier = list(catalog.values())[0]
        models = first_tier.get("models", [])
        assert len(models) > 0
        model = models[0]
        assert "vram_int8_gb" in model
        assert "vram_int4_gb" in model
