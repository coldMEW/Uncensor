"""
Kaggle Kernel Quick Test Script
Same as colab_quick_test.py but with Kaggle-specific additions
"""
# ============= COPY BELOW THIS LINE =============

# Step 1: Install (Kaggle specific)
!pip install -q strongreject lm-eval transformers accelerate bitsandbytes

# Step 2: Import
import sys
sys.path.insert(0, '.')
import torch
from src.model import RefusalModel
from src.extraction import difference_in_means
from src.interventions import directional_ablation, orthogonalize_weights
from src import metrics

# Step 3: Config - Choose model based on GPU
# T4 15GB: Qwen/Qwen2-0.5B-Instruct or Qwen/Qwen2-1.5B-Instruct
# T4 x2 30GB: Qwen/Qwen2-7B-Instruct (use int8)
# P100/V100 16GB: Qwen/Qwen2.5-7B-Instruct (use int8)

MODEL = "Qwen/Qwen2-0.5B-Instruct"  # Start with smallest
QUANTIZE = None  # Or "int8" for larger models

# Step 4: Load Model
print(f"Loading {MODEL}...")
model = RefusalModel(
    name=MODEL,
    device="cuda",
    quantization=QUANTIZE
)
model.load()
print(f"Loaded: {MODEL}")

# Step 5-7: Same as colab_quick_test.py
# ... (run the same test prompts section) ...

# Kaggle specific: Save to output
print("\nSaving model to output...")
model.model.save_pretrained('/kaggle/working/obliterated_model')
print("✅ Saved to /kaggle/working/obliterated_model")

# Run benchmarks
print("\nRunning MMLU benchmark...")
results = metrics.run_capability_benchmarks(
    model_name=MODEL,
    tasks=["mmlu"],
    limit=10
)
print(f"MMLU results: {results}")