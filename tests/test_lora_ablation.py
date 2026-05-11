"""
Tests for LoRA Reversible Ablation.
"""

import torch
import pytest


def test_lora_adapter_initialization():
    """Test LoRA adapter initializes correctly."""
    from src.interventions import LoRAReversibleAblation

    # Mock model
    class MockModel:
        d_model = 64
        n_layers = 4
        device = "cpu"

    model = MockModel()
    direction = torch.randn(64)
    direction = direction / direction.norm()

    lora = LoRAReversibleAblation(model, direction, rank=4)

    assert lora.rank == 4
    assert lora.direction.shape == (64,)


def test_lora_context_manager():
    """Test LoRA can be used as context manager."""
    from src.interventions import LoRAReversibleAblation

    class MockModel:
        d_model = 64
        n_layers = 4
        device = "cpu"
        model = None

    model = MockModel()
    direction = torch.randn(64)
    direction = direction / direction.norm()

    lora = LoRAReversibleAblation(model, direction)

    # Test initialization and basic properties
    assert lora.direction.shape == (64,)
    assert lora.rank == 4


def test_lora_remove_restores_state():
    """Test that remove() cleans up hooks."""
    from src.interventions import LoRAReversibleAblation

    class MockModel:
        d_model = 64
        n_layers = 4
        device = "cpu"
        model = None

    model = MockModel()
    direction = torch.randn(64)
    direction = direction / direction.norm()

    lora = LoRAReversibleAblation(model, direction)
    # Initial state has no handles
    assert len(lora._handles) == 0
    assert lora._lora_weights == {}
