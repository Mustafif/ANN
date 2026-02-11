#!/usr/bin/env python3
"""
Diagnostic script to test parallel calibration setup.
Tests model loading, data processing, and parallel execution.
"""

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from ann import ForwardModel

# Configuration
MODEL_PATH = "trained_model_HN_100K_with_dlayer.pth"
ASSET_PRICES_PATH = "datasets/assetprices.csv"
OPTIONS_DATA_PATH = "datasets/scalable_hn_dataset_250x60.csv"

def test_1_model_loading():
    """Test 1: Can we load the model?"""
    print("\n" + "="*60)
    print("TEST 1: Model Loading")
    print("="*60)
    try:
        device = torch.device("cpu")
        model = ForwardModel(dlayer=True)
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("✓ Model loaded successfully")
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_2_data_loading():
    """Test 2: Can we load the data?"""
    print("\n" + "="*60)
    print("TEST 2: Data Loading")
    print("="*60)
    try:
        # Load prices
        prices_df = pd.read_csv(ASSET_PRICES_PATH)
        prices = prices_df.values.flatten()
        log_returns = np.log(prices[1:] / prices[:-1])
        print(f"✓ Returns loaded: {len(log_returns)} samples")

        # Load options
        options_df = pd.read_csv(OPTIONS_DATA_PATH)
        options_df = options_df[options_df["V"] > 0.5].reset_index(drop=True)
        print(f"✓ Options loaded: {len(options_df)} samples")

        # Check required columns
        required_cols = ["S0", "m", "r", "T", "corp", "sigma"]
        missing = [col for col in required_cols if col not in options_df.columns]
        if missing:
            print(f"✗ Missing columns: {missing}")
            return False
        print(f"✓ All required columns present")

        return True, log_returns, options_df
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_3_single_evaluation(log_returns, options_df):
    """Test 3: Can we evaluate a single parameter set?"""
    print("\n" + "="*60)
    print("TEST 3: Single Evaluation")
    print("="*60)
    try:
        device = torch.device("cpu")

        # Load model
        model = ForwardModel(dlayer=True)
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Test parameters (close to true values)
        test_params = np.array([1e-6, 1.33e-6, 0.8, 5.0, 0.2, 0.01])

        # Convert data
        log_returns_t = torch.tensor(log_returns, dtype=torch.float32, device=device)
        sigma_obs_t = torch.tensor(options_df["sigma"].values, dtype=torch.float32, device=device)

        # Simple loss computation (just test the infrastructure)
        from run_calibration import joint_loss
        N = len(log_returns)
        M = len(options_df)

        print("  Evaluating loss function...")
        start = time.time()
        loss = joint_loss(test_params, model, log_returns_t, options_df,
                         sigma_obs_t, N, M, batch_size=2048)
        elapsed = time.time() - start

        print(f"✓ Evaluation successful")
        print(f"  Loss value: {loss:.6f}")
        print(f"  Time: {elapsed:.3f} seconds")

        if np.isnan(loss) or np.isinf(loss):
            print("✗ WARNING: Loss is NaN or Inf!")
            return False

        return True
    except Exception as e:
        print(f"✗ Single evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def worker_test_function(params_idx):
    """Simple worker function for testing multiprocessing"""
    params, idx = params_idx
    try:
        # Try to load model in worker
        device = torch.device("cpu")
        model = ForwardModel(dlayer=True)
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Simulate computation
        time.sleep(0.1)

        return f"Worker {idx}: Success"
    except Exception as e:
        return f"Worker {idx}: Failed - {e}"

def test_4_multiprocessing():
    """Test 4: Can we use multiprocessing with model loading?"""
    print("\n" + "="*60)
    print("TEST 4: Multiprocessing")
    print("="*60)

    try:
        num_workers = min(4, multiprocessing.cpu_count())
        print(f"  Testing with {num_workers} workers")

        # Create test data
        test_params = [
            (np.array([1e-6, 1.33e-6, 0.8, 5.0, 0.2, 0.01]), i)
            for i in range(num_workers * 2)
        ]

        # Try parallel execution
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(worker_test_function, test_params, timeout=30))

        # Check results
        successes = sum(1 for r in results if "Success" in r)
        failures = sum(1 for r in results if "Failed" in r)

        print(f"✓ Multiprocessing test completed")
        print(f"  Successes: {successes}/{len(test_params)}")
        print(f"  Failures: {failures}/{len(test_params)}")

        if failures > 0:
            print("  Failed workers:")
            for r in results:
                if "Failed" in r:
                    print(f"    {r}")

        return failures == 0
    except Exception as e:
        print(f"✗ Multiprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_5_memory_usage():
    """Test 5: Check memory usage"""
    print("\n" + "="*60)
    print("TEST 5: Memory Usage")
    print("="*60)

    try:
        import psutil
        process = psutil.Process()

        # Initial memory
        mem_before = process.memory_info().rss / 1024**2  # MB
        print(f"  Memory before model load: {mem_before:.1f} MB")

        # Load model
        device = torch.device("cpu")
        model = ForwardModel(dlayer=True)
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)

        mem_after = process.memory_info().rss / 1024**2  # MB
        print(f"  Memory after model load: {mem_after:.1f} MB")
        print(f"  Model size: {mem_after - mem_before:.1f} MB")

        # Estimate memory for N workers
        num_workers = multiprocessing.cpu_count()
        estimated = (mem_after - mem_before) * num_workers + mem_before
        print(f"\n  Estimated memory for {num_workers} workers: {estimated:.1f} MB")

        available = psutil.virtual_memory().available / 1024**2
        print(f"  Available memory: {available:.1f} MB")

        if estimated > available * 0.8:
            print(f"✗ WARNING: May not have enough memory for {num_workers} workers!")
            print(f"  Recommend using {max(1, int(available * 0.8 / (mem_after - mem_before)))} workers")
        else:
            print(f"✓ Memory should be sufficient")

        return True
    except ImportError:
        print("  psutil not installed, skipping memory test")
        print("  Install with: pip install psutil")
        return True
    except Exception as e:
        print(f"✗ Memory test failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("="*60)
    print("PARALLEL CALIBRATION DIAGNOSTIC TESTS")
    print("="*60)

    results = {}

    # Test 1: Model loading
    results['model_loading'] = test_1_model_loading()

    if not results['model_loading']:
        print("\n" + "!"*60)
        print("CRITICAL: Model loading failed. Fix this first!")
        print("!"*60)
        return

    # Test 2: Data loading
    test2_result = test_2_data_loading()
    if isinstance(test2_result, tuple):
        results['data_loading'], log_returns, options_df = test2_result
    else:
        results['data_loading'] = test2_result
        log_returns, options_df = None, None

    if not results['data_loading']:
        print("\n" + "!"*60)
        print("CRITICAL: Data loading failed. Fix this first!")
        print("!"*60)
        return

    # Test 3: Single evaluation
    results['single_eval'] = test_3_single_evaluation(log_returns, options_df)

    # Test 4: Multiprocessing
    results['multiprocessing'] = test_4_multiprocessing()

    # Test 5: Memory
    results['memory'] = test_5_memory_usage()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("="*60)

    if all_passed:
        print("\n✓ All tests passed! Parallel calibration should work.")
    else:
        print("\n✗ Some tests failed. Review the output above.")
        print("\nRecommendations:")
        if not results.get('multiprocessing', True):
            print("  - Set FORCE_CPU_FOR_PARALLEL = False to use single-process mode")
        if not results.get('single_eval', True):
            print("  - Check data preprocessing and model compatibility")

    print()

if __name__ == "__main__":
    main()
