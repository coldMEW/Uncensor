"""
Gradio UI for Uncensor - Browser-based refusal direction ablation.

Usage:
    python app.py                    # Launch local UI
    python app.py --share            # Create public share link
    python app.py --port 7860        # Custom port

This provides a browser interface to:
- Select models from the catalog or enter custom HF path
- Configure pipeline with preset selectors
- View results with metrics and charts
- Export to disk or push to HF Hub
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr
import torch
import yaml

from src.model import RefusalModel
from src.utils import filter_models_by_tier, compute_tier_for_vram


# =============================================================================
# Catalog & Config Helpers
# =============================================================================

def get_tier_choices() -> List[str]:
    """Get available compute tiers."""
    return [
        "tier_1_consumer (≤8GB)",
        "tier_2_enthusiast (8-16GB)",
        "tier_3_hobbyist (16-24GB)",
        "tier_4_professional (24-80GB)",
        "tier_5_cluster (>80GB)",
    ]


def get_model_choices(tier: str = None) -> List[str]:
    """Get model choices from catalog."""
    tier_key = tier.split(" ")[0] if tier else None
    models = filter_models_by_tier(tier=tier_key)
    return [m["name"] for m in models]


def get_preset_choices() -> Dict[str, Dict]:
    """Get pipeline presets."""
    return {
        "basic": {
            "n_directions": 1,
            "delta_threshold": 0.5,
            "kl_score_max": 0.1,
            "description": "Single direction, conservative KL budget",
        },
        "moderate": {
            "n_directions": 2,
            "delta_threshold": 0.4,
            "kl_score_max": 0.15,
            "description": "Two directions, moderate intervention",
        },
        "aggressive": {
            "n_directions": 3,
            "delta_threshold": 0.3,
            "kl_score_max": 0.2,
            "description": "Three directions, stronger bypass",
        },
        "nuclear": {
            "n_directions": 5,
            "delta_threshold": 0.2,
            "kl_score_max": 0.3,
            "description": "Maximum directions, highest bypass",
        },
    }


# =============================================================================
# Pipeline Runners
# =============================================================================

def run_pipeline_ui(
    model_name: str,
    custom_model: str,
    tier: str,
    preset: str,
    quantization: str,
    use_calibration: bool,
    use_causal_patching: bool,
    use_xstest: bool,
    progress=gr.Progress(),
):
    """Run the ablation pipeline and return results."""
    # Determine model name
    if custom_model and custom_model.strip():
        actual_model = custom_model.strip()
    else:
        actual_model = model_name

    if not actual_model:
        raise gr.Error("Please select a model or enter a custom model name")

    progress(0, desc="Loading model...")
    try:
        # Determine dtype based on quantization
        dtype = "bfloat16"
        quant = None
        if quantization != "none":
            quant = quantization
            # Quantized models often work better with fp16
            dtype = "float16"

        model = RefusalModel(
            name=actual_model,
            dtype=dtype,
            device="cuda" if torch.cuda.is_available() else "cpu",
            quantization=quant,
        )
    except Exception as e:
        raise gr.Error(f"Failed to load model: {str(e)}")

    # Load config
    config_path = Path("configs/base.yaml")
    if not config_path.exists():
        config_path = Path(__file__).parent / "configs/base.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply preset
    presets = get_preset_choices()
    if preset in presets:
        p = presets[preset]
        config["extraction"]["n_directions"] = p["n_directions"]
        config["extraction"]["delta_threshold"] = p["delta_threshold"]
        config["extraction"]["kl_score_max"] = p["kl_score_max"]

    # Apply settings
    config["model"]["name"] = actual_model
    config["extraction"]["use_calibration"] = use_calibration
    config["extraction"]["use_causal_patching"] = use_causal_patching
    config["evaluation"]["load_xstest_eval"] = use_xstest

    # Run extraction
    progress(0.2, desc="Extracting refusal direction...")
    from src.pipeline import run_pipeline
    try:
        result, direction = run_pipeline(config)
    except Exception as e:
        raise gr.Error(f"Pipeline failed: {str(e)}")

    # Save outputs
    progress(0.9, desc="Saving outputs...")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    safe_name = actual_model.replace("/", "__")
    torch.save(direction, output_dir / f"{safe_name}_refusal_direction.pt")

    # Format results for display
    metrics_text = f"""# Results for {actual_model}

## Pipeline Configuration
- Preset: {preset}
- Quantization: {quantization}
- Calibration: {'Yes' if use_calibration else 'No'}
- Causal Patching: {'Yes' if use_causal_patching else 'No'}

## Direction Extraction
- Best Layer: {result.best_layer}
- Best Position: {result.best_position}
- Bypass Score: {result.bypass_score:.4f}
- Induce Score: {result.induce_score:.4f}
- KL Score: {result.kl_score:.4f}

## Safety Metrics
- Refusal Rate (harmful): {result.refusal_rate_harmful:.2%}
- Safety Rate: {result.safety_rate:.2%}

Direction saved to: {output_dir / f'{safe_name}_refusal_direction.pt'}
"""
    progress(1.0, desc="Complete!")
    return metrics_text


def export_to_hub(
    model_name: str,
    direction_path: str,
    hub_repo: str,
    progress=gr.Progress(),
):
    """Export orthogonalized model to HuggingFace Hub."""
    if not hub_repo:
        raise gr.Error("Please specify a Hub repo ID (e.g., user/uncensored-model)")

    progress(0, desc="Loading direction...")
    direction = torch.load(direction_path)

    # TODO: Implement model_to_hub() - push modified weights
    # This requires implementing post-orthogonalization weight pushing
    raise gr.Error("Hub export not yet implemented - coming soon!")


# =============================================================================
# UI Layout
# =============================================================================

def create_ui():
    """Create the Gradio interface with 4 tabs."""

    with gr.Blocks(title="Uncensor - Refusal Direction Ablation") as app:
        gr.Markdown("# Uncensor - Browser Interface")
        gr.Markdown("Remove refusal directions from language models without retraining.")

        with gr.Tab("Model Selection"):
            with gr.Row():
                with gr.Column():
                    tier_select = gr.Dropdown(
                        label="Compute Tier (based on your GPU)",
                        choices=get_tier_choices(),
                        value="tier_1_consumer (≤8GB)",
                    )
                    model_select = gr.Dropdown(
                        label="Model from Catalog",
                        choices=get_model_choices("tier_1_consumer"),
                    )
                    custom_model = gr.Textbox(
                        label="Or enter custom HuggingFace model path",
                        placeholder="e.g., meta-llama/Llama-3.2-1B-Instruct",
                    )
                    quantization = gr.Dropdown(
                        label="Quantization (optional)",
                        choices=["none", "8bit", "4bit"],
                        value="none",
                    )

                with gr.Column():
                    gr.Markdown("### VRAM Requirements")
                    gr.Markdown("Select your GPU tier to see compatible models. Quantization reduces VRAM usage.")
                    vram_info = gr.JSON(
                        label="Selected Model VRAM",
                    )

            tier_select.change(
                get_model_choices,
                inputs=[tier_select],
                outputs=[model_select],
            )

            def update_vram(model_name):
                from src.utils import get_model_vram
                vram = get_model_vram(model_name)
                return vram if vram else {"info": "Custom model - VRAM unknown"}

            model_select.change(update_vram, inputs=[model_select], outputs=[vram_info])

        with gr.Tab("Pipeline Config"):
            with gr.Row():
                with gr.Column():
                    preset = gr.Dropdown(
                        label="Intervention Preset",
                        choices=list(get_preset_choices().keys()),
                        value="moderate",
                    )
                    preset_desc = gr.Markdown()

                    use_calibration = gr.Checkbox(
                        label="Use calibrated orthogonalization (preserves capability)",
                        value=True,
                    )
                    use_causal_patching = gr.Checkbox(
                        label="Enable causal layer selection",
                        value=False,
                    )
                    use_xstest = gr.Checkbox(
                        label="Evaluate over-refusal (XSTest)",
                        value=True,
                    )

            def update_preset_desc(preset_name):
                p = get_preset_choices().get(preset_name, {})
                return f"**{preset_name}**: {p.get('description', '')}"

            preset.change(update_preset_desc, inputs=[preset], outputs=[preset_desc])

        with gr.Tab("Results"):
            with gr.Row():
                run_btn = gr.Button("Run Pipeline", variant="primary")

            results = gr.Markdown("Run pipeline to see results...")

            run_btn.click(
                run_pipeline_ui,
                inputs=[
                    model_select,
                    custom_model,
                    tier_select,
                    preset,
                    quantization,
                    use_calibration,
                    use_causal_patching,
                    use_xstest,
                ],
                outputs=[results],
            )

        with gr.Tab("Export"):
            gr.Markdown("### Export Options")

            with gr.Row():
                export_path = gr.Textbox(
                    label="Direction file path",
                    placeholder="outputs/...",
                )
                export_btn = gr.Button("Save to Disk", variant="secondary")

            gr.Markdown("---")
            gr.Markdown("### HuggingFace Hub Export")

            with gr.Row():
                hub_repo = gr.Textbox(
                    label="Hub Repo ID",
                    placeholder="user/my-uncensored-model",
                )
                hub_export_btn = gr.Button("Push to Hub", variant="primary")

            hub_export_btn.click(
                export_to_hub,
                inputs=[model_select, export_path, hub_repo],
                outputs=[gr.Markdown()],
            )

    return app


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Uncensor Gradio UI")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    app = create_ui()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )