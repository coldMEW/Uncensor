"""
Tests for Strength Sweep Visualization.
"""

import torch
import pytest


def test_sweep_result_dataclass():
    """Test SweepResult dataclass fields."""
    from src.analysis.strength_sweep import SweepResult

    result = SweepResult(
        coefficient=1.0,
        compliance_rate=0.8,
        coherence_score=-1.5,
        refusal_rate=0.1,
        kl_divergence=0.05,
    )

    assert result.coefficient == 1.0
    assert result.compliance_rate == 0.8


def test_sweep_analysis_dataclass():
    """Test SweepAnalysis dataclass."""
    from src.analysis.strength_sweep import SweepResult, SweepAnalysis

    results = [
        SweepResult(0.5, 0.6, -1.0, 0.1, 0.05),
        SweepResult(1.0, 0.8, -1.2, 0.15, 0.08),
        SweepResult(1.5, 0.9, -1.5, 0.2, 0.12),
    ]

    analysis = SweepAnalysis(
        results=results,
        optimal_coefficient=1.0,
        max_compliance=0.9,
        min_capability_loss=1.5,
    )

    assert analysis.optimal_coefficient == 1.0
    assert analysis.max_compliance == 0.9


def test_analyze_sweep_results():
    """Test sweep result analysis."""
    from src.analysis.strength_sweep import (
        SweepResult,
        analyze_sweep_results,
    )

    results = [
        SweepResult(0.5, 0.5, -1.0, 0.2, 0.05),
        SweepResult(1.0, 0.8, -1.5, 0.1, 0.1),
        SweepResult(1.5, 0.9, -2.0, 0.3, 0.15),
    ]

    analysis = analyze_sweep_results(results)

    assert analysis.max_compliance == 0.9
    assert analysis.optimal_coefficient in [0.5, 1.0, 1.5]


def test_plot_strength_curve_callable():
    """Test that plot function is callable without errors."""
    from src.analysis.strength_sweep import SweepResult, plot_strength_curve

    results = [
        SweepResult(0.5, 0.6, -1.0, 0.1, 0.05),
        SweepResult(1.0, 0.8, -1.2, 0.15, 0.08),
    ]

    # Should not raise even without matplotlib
    plot_strength_curve(results, save_path=None)
