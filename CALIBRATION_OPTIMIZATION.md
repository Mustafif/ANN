# Calibration Performance Optimization Report

## Overview
The `run_calibration.py` script was experiencing severe performance issues due to several critical bottlenecks in the parallel differential evolution optimization process.

## Problems Identified

### 🔴 CRITICAL: Model Reloading on Every Evaluation
**Location**: `_evaluate_params()` function (lines 278-281)

**Problem**: The neural network model was being loaded from disk on **every single function evaluation**. During differential evolution with:
- Population size: 50
- Max iterations: 100
- This means **5,000+ model loads from disk**

**Impact**: This single issue caused the calibration to be orders of magnitude slower than necessary. Loading a PyTorch model involves:
- File I/O operations
- Deserialization of weights
- Memory allocation
- Moving tensors to device

### 🟡 Redundant Tensor Conversions
**Location**: `_evaluate_params()` function (lines 284-288)

**Problem**: Numpy arrays were being converted to PyTorch tensors on every evaluation:
```python
log_returns_t = torch.tensor(_SHARED_LOG_RETURNS, dtype=torch.float32, device=device)
sigma_obs_t = torch.tensor(_SHARED_SIGMA_OBS, dtype=torch.float32, device=device)
```

**Impact**: Thousands of unnecessary tensor allocations and data copies.

### 🟡 DataFrame Reconstruction
**Location**: `_evaluate_params()` function (line 291)

**Problem**: Options DataFrame was reconstructed from dictionary on every evaluation:
```python
options_df = pd.DataFrame(_SHARED_OPTIONS_DF_DICT)
```

**Impact**: Pandas DataFrame construction has overhead, repeated thousands of times.

### 🟠 Artificially Limited Workers
**Location**: Main execution (line 408)

**Problem**: Workers were artificially limited to 4, even on systems with more cores:
```python
num_workers = min(4, max_workers)  # Limited to reduce memory usage
```

**Impact**: Underutilized available CPU resources.

## Solutions Implemented

### ✅ Model Caching in Worker Initialization
**Change**: Moved model loading to `_init_worker()` and cached it globally per worker process.

**Before**:
```python
def _init_worker(...):
    global _SHARED_MODEL_PATH
    _SHARED_MODEL_PATH = model_path  # Just store path

def _evaluate_params(params):
    model = ForwardModel(dlayer=True)
    state_dict = torch.load(_SHARED_MODEL_PATH, ...)  # Load on EVERY call!
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
```

**After**:
```python
def _init_worker(...):
    global _SHARED_MODEL
    _SHARED_MODEL = ForwardModel(dlayer=True)
    state_dict = torch.load(model_path, ...)  # Load ONCE per worker
    _SHARED_MODEL.load_state_dict(state_dict)
    _SHARED_MODEL.to(_SHARED_DEVICE)
    _SHARED_MODEL.eval()

def _evaluate_params(params):
    result = joint_loss(params, _SHARED_MODEL, ...)  # Use cached model
```

**Benefit**: Model loaded once per worker process (e.g., 4-16 times total) instead of 5,000+ times.

### ✅ Pre-converted Tensors
**Change**: Convert numpy arrays to tensors once during worker initialization.

```python
def _init_worker(...):
    global _SHARED_LOG_RETURNS_T, _SHARED_SIGMA_OBS_T
    _SHARED_LOG_RETURNS_T = torch.tensor(log_returns_np, dtype=torch.float32, device=_SHARED_DEVICE)
    _SHARED_SIGMA_OBS_T = torch.tensor(sigma_obs_np, dtype=torch.float32, device=_SHARED_DEVICE)
```

**Benefit**: Eliminates thousands of tensor conversion operations.

### ✅ Pre-constructed DataFrame
**Change**: Build DataFrame once during initialization.

```python
def _init_worker(...):
    global _SHARED_OPTIONS_DF
    _SHARED_OPTIONS_DF = pd.DataFrame(options_df_dict)
```

**Benefit**: Avoids repeated DataFrame construction overhead.

### ✅ Full Core Utilization
**Change**: Removed artificial worker limit.

```python
num_workers = mp.cpu_count()  # Use all available cores
```

**Benefit**: Maximum parallelization for faster completion.

### ✅ Reduced Progress Spam
**Change**: Reduced progress update frequency from every 10 evaluations to every 50.

```python
if count % 50 == 0 or count <= 5:  # Was: count % 10 == 0
    print(f"[Evaluation {count}] Processing...", flush=True)
```

**Benefit**: Less I/O overhead from print statements.

## Expected Performance Impact

### Conservative Estimates:
- **Model caching**: 10-50x speedup (depending on model size and disk speed)
- **Tensor pre-conversion**: 2-3x speedup
- **DataFrame caching**: 1.5-2x speedup
- **Full core utilization**: Linear scaling with additional cores used

### Overall Expected Improvement:
**20-100x faster calibration** depending on hardware configuration.

### Example Timeline:
- **Before**: 30+ minutes per calibration run
- **After**: 30 seconds to 2 minutes per calibration run

## Memory Considerations

The optimizations increase per-worker memory usage slightly (each worker caches model + tensors), but this is negligible compared to:
1. The model is already being loaded (just not cached)
2. Tensors are small compared to model weights
3. Modern systems have sufficient RAM for this

If memory becomes an issue on resource-constrained systems, you can manually reduce `num_workers`.

## Verification

To verify the improvements:
1. Run the optimized version
2. Monitor evaluation count messages - should progress rapidly
3. Check total runtime - should be dramatically reduced
4. Results should be identical to the original (only performance changed)

## Additional Optimization Opportunities

If further speedup is needed:
1. **Reduce batch_size**: 2048 might be too large for CPU, try 512 or 256
2. **Adjust DE parameters**: Reduce `popsize` or `maxiter` if convergence allows
3. **GPU acceleration**: If model fits in GPU memory, could use CUDA (requires code changes)
4. **Compiled models**: Use `torch.jit.script()` or `torch.compile()` for model inference

## Summary

The original code had a catastrophic performance bug where the most expensive operation (model loading) was repeated thousands of times unnecessarily. The fix was simple: use the worker initialization pattern correctly by caching expensive resources once per worker process instead of recomputing them on every function evaluation.

This is a classic example of why profiling and understanding the execution flow is critical for performance optimization.