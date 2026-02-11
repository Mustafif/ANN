#!/usr/bin/env python3
"""
Quick test to verify verbose output and progress tracking.
Runs a mini calibration with just 2 iterations to check visibility.
"""

import multiprocessing as mp
import sys
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
from scipy.optimize import differential_evolution

from ann import ForwardModel

# Configuration
MODEL_PATH = "trained_model_HN_100K_with_dlayer.pth"
ASSET_PRICES_PATH = "datasets/assetprices.csv"
OPTIONS_DATA_PATH = "datasets/scalable_hn_dataset_250x60.csv"

# Shared variables for workers
_SHARED_MODEL_PATH = None
_SHARED_LOG_RETURNS = None
_SHARED_OPTIONS_DF_DICT = None
_SHARED_SIGMA_OBS = None
_SHARED_N = None
_SHARED_M = None
_SHARED_BATCH_SIZE = None
_EVAL_COUNTER = None

def _init_worker(model_path, log_returns_np, options_df_dict, sigma_obs_np, N, M, batch_size, eval_counter):
    """Initialize worker process"""
    global _SHARED_MODEL_PATH, _SHARED_LOG_RETURNS, _SHARED_OPTIONS_DF_DICT
    global _SHARED_SIGMA_OBS, _SHARED_N, _SHARED_M, _SHARED_BATCH_SIZE, _EVAL_COUNTER

    _SHARED_MODEL_PATH = model_path
    _SHARED_LOG_RETURNS = log_returns_np
    _SHARED_OPTIONS_DF_DICT = options_df_dict
    _SHARED_SIGMA_OBS = sigma_obs_np
    _SHARED_N = N
    _SHARED_M = M
    _SHARED_BATCH_SIZE = batch_size
    _EVAL_COUNTER = eval_counter

    print(f"[Worker {mp.current_process().name}] Initialized", flush=True)

def _evaluate_params(params):
    """Evaluate parameters with verbose output"""
    try:
        # Progress counter
        if _EVAL_COUNTER is not None:
            with _EVAL_COUNTER.get_lock():
                _EVAL_COUNTER.value += 1
                count = _EVAL_COUNTER.value
                print(f"[Evaluation {count}] Worker {mp.current_process().name} processing...", flush=True)

        # Load model
        device = torch.device("cpu")
        model = ForwardModel(dlayer=True)
        state_dict = torch.load(_SHARED_MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Simple loss computation (just return random for testing)
        result = float(np.random.rand() * 1000)

        if _EVAL_COUNTER is not None and _EVAL_COUNTER.value % 10 == 0:
            print(f"[Evaluation {_EVAL_COUNTER.value}] Loss = {result:.2f}", flush=True)

        return result
    except Exception as e:
        print(f"ERROR in worker: {e}", flush=True)
        return 1e9

def progress_callback(xk, convergence=0):
    """Show progress after each iteration"""
    if not hasattr(progress_callback, 'iteration'):
        progress_callback.iteration = 0
        progress_callback.start_time = time.time()

    progress_callback.iteration += 1
    elapsed = time.time() - progress_callback.start_time

    print(f"\n{'='*60}", flush=True)
    print(f"ITERATION {progress_callback.iteration} COMPLETE", flush=True)
    print(f"Elapsed time: {elapsed:.1f} seconds", flush=True)
    print(f"Best parameters so far: {xk[:3]}...", flush=True)
    print(f"{'='*60}\n", flush=True)

    return False

def main():
    print("="*60, flush=True)
    print("VERBOSE OUTPUT TEST - Mini Calibration", flush=True)
    print("="*60, flush=True)

    # Load data
    print("\n[1/5] Loading data...", flush=True)
    prices_df = pd.read_csv(ASSET_PRICES_PATH)
    prices = prices_df.values.flatten()
    log_returns = np.log(prices[1:] / prices[:-1])

    options_df = pd.read_csv(OPTIONS_DATA_PATH)
    options_df = options_df[options_df["V"] > 0.5].reset_index(drop=True)

    # Subsample for quick test
    options_df = options_df.sample(n=min(1000, len(options_df)), random_state=42)

    N = len(log_returns)
    M = len(options_df)

    print(f"✓ Loaded {N} returns, {M} options", flush=True)

    # Prepare data
    print("\n[2/5] Preparing data...", flush=True)
    sigma_obs_np = options_df["sigma"].values
    options_df_dict = options_df.to_dict('list')
    batch_size = 512
    print("✓ Data prepared", flush=True)

    # Test single evaluation
    print("\n[3/5] Testing single evaluation...", flush=True)
    eval_counter = mp.Value('i', 0)
    _init_worker(MODEL_PATH, log_returns, options_df_dict, sigma_obs_np, N, M, batch_size, eval_counter)
    test_params = np.array([1e-6, 1.33e-6, 0.8, 5.0, 0.2, 0.01])
    test_result = _evaluate_params(test_params)
    print(f"✓ Test evaluation complete: {test_result:.2f}", flush=True)

    # Setup parallel
    print("\n[4/5] Setting up parallel workers...", flush=True)
    num_workers = 2  # Use only 2 workers for quick test
    eval_counter = mp.Value('i', 0)

    initializer = partial(_init_worker, MODEL_PATH, log_returns,
                         options_df_dict, sigma_obs_np, N, M, batch_size, eval_counter)

    # Bounds
    bounds = [
        (1e-7, 1e-6),
        (1.15e-6, 1.50e-6),
        (0.0, 0.99),
        (0.0, 10.0),
        (0.0, 1.0),
        (1e-5, 0.1)
    ]

    print(f"✓ Using {num_workers} workers", flush=True)

    # Run mini optimization
    print("\n[5/5] Running mini optimization (2 iterations only)...", flush=True)
    print("="*60, flush=True)
    print("STARTING DIFFERENTIAL EVOLUTION", flush=True)
    print("="*60, flush=True)

    t0 = time.time()

    with mp.Pool(processes=num_workers, initializer=initializer, initargs=()) as pool:
        print("\n✓ Pool created, starting optimization...\n", flush=True)

        result = differential_evolution(
            _evaluate_params,
            bounds=bounds,
            strategy='best1bin',
            maxiter=2,  # Only 2 iterations for quick test
            popsize=5,   # Small population
            tol=1.0,
            mutation=(0.5, 1),
            recombination=0.8,
            disp=True,
            callback=progress_callback,
            polish=False,  # Skip polish for speed
            init='latinhypercube',
            workers=pool.map,
            updating='deferred'
        )

    t1 = time.time()

    print("\n" + "="*60, flush=True)
    print("TEST COMPLETE!", flush=True)
    print("="*60, flush=True)
    print(f"Total time: {t1-t0:.1f} seconds", flush=True)
    print(f"Total evaluations: {eval_counter.value}", flush=True)
    print(f"Success: {result.success}", flush=True)
    print(f"Final loss: {result.fun:.2f}", flush=True)
    print("\n✓ Verbose output is working!", flush=True)
    print("\nNow you can run the full calibration with:", flush=True)
    print("  python run_calibration.py", flush=True)

if __name__ == "__main__":
    main()
