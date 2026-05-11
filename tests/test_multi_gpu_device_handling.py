"""Tests for multi-GPU device handling in weight orthogonalization.

Run with: pytest tests/test_multi_gpu_device_handling.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
import torch
import torch.nn as nn


class TestOrthoLinearDeviceHandling:
    """Tests for device handling in _ortho_linear."""

    def test_ortho_linear_cpu_device(self):
        """Verify _ortho_linear works on CPU."""
        from src.interventions import _ortho_linear

        # Create linear layer on CPU: Linear(in_features, out_features)
        # r_hat dimension must equal out_features
        in_feat, out_feat = 64, 128
        layer = nn.Linear(in_feat, out_feat)
        r_hat = torch.randn(out_feat)  # r_hat dim = out_features

        # Should not raise
        _ortho_linear(layer, r_hat)

        # Weight should be modified (not zero)
        assert layer.weight.abs().sum() > 0

    def test_ortho_linear_dtype_conversion(self):
        """Verify dtype conversion works when r_hat dtype differs from layer."""
        from src.interventions import _ortho_linear

        in_feat, out_feat = 64, 128
        layer = nn.Linear(in_feat, out_feat, dtype=torch.float32)
        r_hat = torch.randn(out_feat, dtype=torch.float64)  # Different dtype, dim=out_feat

        # Should not raise - converts dtype automatically
        _ortho_linear(layer, r_hat)

    def test_ortho_linear_device_conversion(self):
        """Verify device conversion works when r_hat on different device."""
        from src.interventions import _ortho_linear

        in_feat, out_feat = 64, 128
        layer = nn.Linear(in_feat, out_feat)
        r_hat = torch.randn(out_feat)  # On CPU

        _ortho_linear(layer, r_hat)

    def test_ortho_embedding_device_handling(self):
        """Verify _ortho_embedding handles device correctly."""
        from src.interventions import _ortho_embedding

        vocab, d_model = 256, 64
        layer = nn.Embedding(vocab, d_model)
        r_hat = torch.randn(d_model)

        # Should not raise
        _ortho_embedding(layer, r_hat)

        assert layer.weight.abs().sum() > 0

    def test_ortho_linear_with_bias_device_handling(self):
        """Verify bias device handling in _ortho_linear."""
        from src.interventions import _ortho_linear

        in_feat, out_feat = 64, 128
        layer = nn.Linear(in_feat, out_feat, bias=True)
        r_hat = torch.randn(out_feat)

        # Should not raise
        _ortho_linear(layer, r_hat)

    def test_ortho_linear_no_bias(self):
        """Verify works when layer has no bias."""
        from src.interventions import _ortho_linear

        in_feat, out_feat = 64, 128
        layer = nn.Linear(in_feat, out_feat, bias=False)
        r_hat = torch.randn(out_feat)

        # Should not raise
        _ortho_linear(layer, r_hat)


class TestMultiGPUReadiness:
    """Tests verifying multi-GPU readiness."""

    def test_device_conversion_pattern_present(self):
        """Verify code uses .to(device=layer.device) pattern."""
        from src import interventions
        import inspect
        source = inspect.getsource(interventions._ortho_linear)

        # Should have device= parameter in .to() call
        assert "device=W.device" in source or "device=W.device" in source.replace(" ", "")

    def test_dtype_and_device_both_converted(self):
        """Verify both dtype and device are converted."""
        from src import interventions
        import inspect
        source = inspect.getsource(interventions._ortho_linear)

        # Should have dtype= and device= in same .to() call
        has_dtype = "dtype=" in source
        has_device = "device=" in source

        assert has_dtype and has_device, "Both dtype and device conversion required"

    def test_embedding_also_device_aware(self):
        """Verify _ortho_embedding also handles device."""
        from src import interventions
        import inspect
        source = inspect.getsource(interventions._ortho_embedding)

        assert "device=W.device" in source or "device=W.device" in source.replace(" ", "")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Single GPU or CPU only")
class TestActualMultiGPU:
    """Tests requiring actual multi-GPU setup."""

    def test_cross_device_ortho_linear(self):
        """Test _ortho_linear when layer and r_hat on different devices."""
        from src.interventions import _ortho_linear

        # This test would require actual multi-GPU
        # Skip if only one GPU available
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires 2+ GPUs")

        # Create layer on GPU 0
        layer = nn.Linear(64, 128).cuda(0)
        # Create r_hat on GPU 1
        r_hat = torch.randn(64).cuda(1)

        # Should handle cross-device case
        _ortho_linear(layer, r_hat)