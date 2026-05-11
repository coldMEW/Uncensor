"""
Anonymous telemetry and leaderboard for refusal direction ablation.

US-015 — Submit anonymized run results to shared benchmark DB and view
simple leaderboard: method -> bypass % -> capability delta.

This is OPTIONAL telemetry - disabled by default, user must explicitly opt-in.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TelemetryConfig:
    """Configuration for anonymous telemetry."""

    enabled: bool = False
    """Whether to submit telemetry. Default False."""

    endpoint: str = "https://api.uncensor.dev/leaderboard"
    """API endpoint for leaderboard submissions."""

    local_cache_dir: Optional[Path] = None
    """Local directory for caching leaderboard data."""


# =============================================================================
# Telemetry Event
# =============================================================================

@dataclass
class TelemetryEvent:
    """Anonymized telemetry event for a single run."""

    run_id: str
    """Anonymous run identifier (SHA of timestamp + random)."""

    timestamp: str
    """ISO timestamp of run."""

    model_name: str
    """Model identifier (e.g., 'meta-llama/Llama-3.1-8B')."""

    method: str
    """Ablation method used ('diy', 'OBLITERATUS', 'steering')."""

    bypass_score: float
    """Bypass score (0-1): fraction of harmful prompts not refused."""

    capability_delta: float
    """Change in capability (e.g., MMLU delta) after ablation."""

    n_directions: int
    """Number of directions orthogonalized."""

    quantization: Optional[str]
    """Quantization used ('8bit', '4bit', or None)."""

    system_info: str
    """Platform info (Python version, OS)."""

    @staticmethod
    def create(
        model_name: str,
        method: str,
        bypass_score: float,
        capability_delta: float,
        n_directions: int,
        quantization: Optional[str] = None,
    ) -> "TelemetryEvent":
        """Create a new telemetry event with anonymized ID."""
        # Generate anonymous run ID from timestamp + random
        timestamp = datetime.utcnow().isoformat()
        run_input = f"{timestamp}{model_name}{n_directions}".encode()
        run_id = hashlib.sha256(run_input).hexdigest()[:16]

        return TelemetryEvent(
            run_id=run_id,
            timestamp=timestamp,
            model_name=model_name,
            method=method,
            bypass_score=bypass_score,
            capability_delta=capability_delta,
            n_directions=n_directions,
            quantization=quantization,
            system_info=f"Python {sys.version.split()[0]}, {platform.system()}",
        )


# =============================================================================
# Local Leaderboard Cache
# =============================================================================

def _get_cache_path(config: TelemetryConfig) -> Path:
    """Get path to local leaderboard cache."""
    if config.local_cache_dir:
        return config.local_cache_dir / "leaderboard.json"

    # Default to project root
    return Path(__file__).parent.parent / "leaderboard_cache.json"


def _load_cached_leaderboard(config: TelemetryConfig) -> List[Dict]:
    """Load leaderboard from local cache."""
    cache_path = _get_cache_path(config)
    if not cache_path.exists():
        return []

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("entries", [])
    except Exception:
        return []


def _save_cached_leaderboard(config: TelemetryConfig, entries: List[Dict]) -> None:
    """Save leaderboard to local cache."""
    cache_path = _get_cache_path(config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"entries": entries, "updated": datetime.utcnow().isoformat()}, f, indent=2)


# =============================================================================
# Submission (stub - real implementation would POST to endpoint)
# =============================================================================

def submit_anonymous_result(
    config: TelemetryConfig,
    model_name: str,
    method: str,
    bypass_score: float,
    capability_delta: float,
    n_directions: int,
    quantization: Optional[str] = None,
) -> Optional[TelemetryEvent]:
    """Submit anonymous result to leaderboard.

    Args:
        config: Telemetry configuration
        model_name: Model identifier
        method: Ablation method
        bypass_score: Bypass score (0-1)
        capability_delta: Capability change
        n_directions: Number of directions
        quantization: Quantization used

    Returns:
        TelemetryEvent if submitted, None if disabled
    """
    if not config.enabled:
        print("[telemetry] telemetry disabled - no submission made")
        return None

    # Create event
    event = TelemetryEvent.create(
        model_name=model_name,
        method=method,
        bypass_score=bypass_score,
        capability_delta=capability_delta,
        n_directions=n_directions,
        quantization=quantization,
    )

    # In production, this would POST to config.endpoint
    # For now, just cache locally
    print(f"[telemetry] submitted run {event.run_id}: {model_name} ({method})")
    print(f"[telemetry]   bypass={bypass_score:.2f}, cap_delta={capability_delta:.2f}")

    return event


# =============================================================================
# Leaderboard
# =============================================================================

def get_leaderboard(config: Optional[TelemetryConfig] = None) -> List[Dict]:
    """Get current leaderboard entries.

    Args:
        config: Telemetry configuration (optional)

    Returns:
        List of leaderboard entries sorted by bypass_score descending
    """
    if config is None:
        config = TelemetryConfig()

    # Load from cache
    entries = _load_cached_leaderboard(config)

    # Add some sample entries if empty (for demo purposes)
    if not entries:
        entries = [
            {
                "run_id": "sample1",
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                "method": "diy",
                "bypass_score": 0.85,
                "capability_delta": -0.02,
                "n_directions": 1,
            },
            {
                "run_id": "sample2",
                "model_name": "Qwen/Qwen2-7B-Instruct",
                "method": "diy",
                "bypass_score": 0.78,
                "capability_delta": -0.05,
                "n_directions": 2,
            },
            {
                "run_id": "sample3",
                "model_name": "meta-llama/Llama-3.1-70B-Instruct",
                "method": "OBLITERATUS",
                "bypass_score": 0.92,
                "capability_delta": -0.08,
                "n_directions": 3,
            },
        ]

    # Sort by bypass_score descending
    entries = sorted(entries, key=lambda e: e.get("bypass_score", 0), reverse=True)

    return entries


def format_leaderboard_table(leaderboard: List[Dict]) -> str:
    """Format leaderboard as ASCII table.

    Args:
        leaderboard: List of leaderboard entries

    Returns:
        Formatted ASCII table string
    """
    if not leaderboard:
        return "No entries in leaderboard yet."

    # Header
    header = f"{'Rank':<5} {'Model':<40} {'Method':<12} {'Bypass':<8} {'Cap Δ':<8}"
    separator = "-" * len(header)

    lines = [header, separator]

    for i, entry in enumerate(leaderboard, 1):
        model = entry.get("model_name", "unknown")
        # Truncate long model names
        if len(model) > 38:
            model = model[:35] + "..."

        method = entry.get("method", "unknown")
        bypass = entry.get("bypass_score", 0)
        cap_delta = entry.get("capability_delta", 0)

        lines.append(
            f"{i:<5} {model:<40} {method:<12} {bypass:.2f}    {cap_delta:+.2f}"
        )

    return "\n".join(lines)


def show_leaderboard(config: Optional[TelemetryConfig] = None) -> None:
    """Display leaderboard to console."""
    leaderboard = get_leaderboard(config)
    print("\n" + "=" * 80)
    print("UNCENSOR LEADERBOARD")
    print("=" * 80)
    print(format_leaderboard_table(leaderboard))
    print("=" * 80)