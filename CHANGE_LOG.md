# Uncensor Project - Change Log

## Overview
This document tracks all changes made to fix the refusal bypass pipeline, primarily addressing bitsandbytes import errors and improving bypass effectiveness.

---

## Problem Statement

The original pipeline failed at the model loading stage due to bitsandbytes import errors:
```
ImportError: cannot import name 'BitsAndBytesConfig' from 'bitsandbytes'
```

Even when `quantization=None` was set, the code still attempted to import bitsandbytes, causing the failure.

---

## Changes Made

### 1. requirements.txt (github_sync_repo/)

**File:** `requirements.txt`

**Change:** Added bitsandbytes dependency
```diff
+ bitsandbytes>=0.41.0
```

**Reason:** Ensure fresh installations get the correct version with BitsAndBytesConfig.

---

### 2. src/model.py (github_sync_repo/)

**File:** `src/model.py`

**Change:** Improved bitsandbytes import handling with fallback paths
```python
# Before - single import path
try:
    from bitsandbytes import BitsAndBytesConfig
except ImportError:
    raise ImportError("quantization requires `bitsandbytes` package...")

# After - multiple import paths + better error message
BitsAndBytesConfig = None
try:
    from bitsandbytes import BitsAndBytesConfig
except ImportError:
    try:
        from bitsandbytes.nn import BitsAndBytesConfig
    except ImportError:
        pass  # Will raise error only if quantization actually requested

if BitsAndBytesConfig is None and quantization is not None:
    raise ImportError(
        "quantization requires `bitsandbytes>=0.41.0` with BitsAndBytesConfig. "
        "Current version lacks this class. Install: pip install --upgrade bitsandbytes"
    )
```

**Reason:** Handle different bitsandbytes versions that have BitsAndBytesConfig in different locations.

---

### 3. uncensor-refusal-pipeline-test.ipynb

**Multiple versions pushed to fix bitsandbytes issue in Kaggle:**

#### Version 34-35: pip install fix
- Changed `!pip install -q bitsandbytes` to `!pip install -q bitsandbytes>=0.41.0`
- **Result:** Still failed - pip doesn't upgrade existing packages with >= syntax

#### Version 36: Runtime patch attempt
- Added Python code to patch model.py after cloning
- Pattern matching didn't match the cloned repo's code structure
- **Result:** Failed

#### Version 37: Monkey-patch attempt
- Tried to replace `_load_model` method at runtime
- Class methods are bound at class definition, patching didn't work
- **Result:** Failed

#### Version 38-39: Brute-force exact replacement (WORKING)
- Added exact string replacement for the `_load_model` method in cloned repo
- Replaced the entire quantization block with simple fp16 loading
- **Result:** Version 39 finally worked - model loaded successfully

**Final working setup cell:**
```python
# Clone the repo
!git clone --depth 1 https://github.com/coldMEW/Uncensor.git /kaggle/working/uncensor

# Brute-force fix: exact string replacement
old_block = '''    def _load_model(self, name: str, quantization: str = None) -> PreTrainedModel:
        """Load model with optional quantization support (US-006)."""
        if quantization is None:
            return AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            ).to(self.device)

        # Quantization mode - requires bitsandbytes
        try:
            from bitsandbytes import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "quantization requires `bitsandbytes` package. "
                "Install with: pip install bitsandbytes"
            )
        ... (entire quantization block)'''

new_block = '''    def _load_model(self, name: str, quantization: str = None) -> PreTrainedModel:
        """Load model - quantization disabled due to bitsandbytes issues."""
        # Always use fp16, never use quantization
        return AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)'''
```

---

### 4. load-model cell improvements

**Change:** Removed quantization parameter and simplified model loading
```python
# Before
refusal_model = RefusalModel(
    name=MODEL_NAME,
    dtype='float16' if torch.cuda.is_available() else 'float32',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    quantization='8bit' if torch.cuda.is_available() else None,
)

# After - disabled quantization to avoid bitsandbytes
refusal_model = RefusalModel(
    name=MODEL_NAME,
    dtype='float16' if torch.cuda.is_available() else 'float32',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    quantization=None,  # Not used - always fp16
)
```

**Reason:** Avoid any code path that could trigger bitsandbytes import.

---

### 5. single-test cell improvements

**Change:** Fixed directional_ablation calls and added more tests
```python
# Removed invalid 'coefficient' parameter that doesn't exist in function
# Before: with directional_ablation(refusal_model, direction, coefficient=0.5):
# After:
with directional_ablation(refusal_model, direction):
    bypassed_resp = generate(test_prompt)

# Added weight orthogonalization test (permanent modification)
from src.interventions import orthogonalize_weights
orthogonalize_weights(refusal_model, direction)
```

**Reason:** The function doesn't accept coefficient - full ablation is applied. Added orthogonalize_weights for testing permanent modification.

---

### 6. full-eval cell improvements

**Change:** Added weight orthogonalization test
```python
# Added permanent weight modification test
print('\n=== WEIGHT ORTHOGONALIZATION ===')
print('Applying orthogonalize_weights (permanent)...')
orthogonalize_weights(refusal_model, direction)
print('Weights orthogonalized!')
```

**Reason:** Test both hook-based (temporary) and weight-based (permanent) interventions.

---

## Root Cause Analysis

### Why bitsandbytes failed:
1. Kaggle environment has older bitsandbytes without BitsAndBytesConfig in expected import paths
2. Even with `--upgrade`, the package may not have the class due to version/environment issues
3. The code imported bitsandbytes even when quantization=None (logic bug)
4. Kaggle clones fresh from GitHub, so local fixes aren't applied

### Why bypass not working (preliminary analysis):
Based on comparison with OBLITERATUS (see COMPARISON_OBLITERATUS.md):
1. Direction averaging across ALL layers dilutes the signal
2. Should use paper's selection algorithm (§C.1) to find optimal layer/position
3. May need multiple directions or higher intensity
4. OBLITERATUS has 7 escalation presets - may need stronger intervention

---

## Files Modified

| File | Changes |
|------|---------|
| `requirements.txt` | Added bitsandbytes>=0.41.0 |
| `src/model.py` | Improved bitsandbytes import handling |
| `uncensor-refusal-pipeline-test.ipynb` | Multiple iterations to fix bitsandbytes in Kaggle |

---

## Git Commits

1. `074dd13` - Fix bitsandbytes import error - add to requirements and improve import handling
2. `65a2ed8` - Fix Kaggle notebook: upgrade bitsandbytes to >=0.41.0
3. `3527c72` - Fix bitsandbytes and improve bypass pipeline
4. `dc41205` - Add runtime patch for bitsandbytes in cloned repo
5. `64f93e0` - Fix patch pattern to match cloned repo structure
6. `9126755` - Monkey-patch _load_model to bypass bitsandbytes completely
7. `789d8d5` - Fix cloned repo _load_model directly
8. `a93d47e` - Brute force fix: exact string replacement of _load_model

---

## Current Status

- **Model Loading:** ✅ Fixed - version 39 loads Gemma-4-E4B-it successfully
- **Bypass Effectiveness:** ⚠️ Not yet tested - need to verify after successful run
- **Remaining Issues:**
  - Direction selection may need improvement (per-layer selection vs averaging)
  - May need weight orthogonalization instead of just hooks
  - May need higher intensity or multiple directions

---

## Next Steps (After Successful Run)

1. Analyze bypass results
2. If bypass not working well, implement per-layer direction selection
3. Add weight orthogonalization comparison
4. Consider multi-direction ablation
5. Compare with OBLITERATUS methods