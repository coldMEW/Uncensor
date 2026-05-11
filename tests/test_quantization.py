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

    def test_quantization_raises_on_unknown(self):
        """Verify unknown quantization raises ValueError."""
        from src.model import RefusalModel
        try:
            import bitsandbytes
        except ImportError:
            pytest.skip("bitsandbytes not installed")
        # Unknown quantization should raise
        model = RefusalModel(name="gpt2", device="cpu", quantization="invalid")
        with pytest.raises(ValueError, match="Unknown quantization"):
            model.load()


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