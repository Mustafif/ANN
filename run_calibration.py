import multiprocessing as mp
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
from arch import arch_model
from scipy.optimize import differential_evolution, minimize

from ann import ForwardModel

# --- Configuration ---
#
# DIFFERENTIAL EVOLUTION PARAMETER TUNING GUIDE:
# -----------------------------------------------
# Total evaluations ≈ popsize × (maxiter + 1) + polish_evals
#
# Current settings (line ~395-396):
#   popsize=15, maxiter=30 → ~615 evaluations (~1-2 minutes with optimizations)
#
# Recommended presets:
#   FAST:     popsize=10,  maxiter=20  → ~210 evals  (30-60 seconds)
#   BALANCED: popsize=15,  maxiter=30  → ~615 evals  (1-2 minutes)  ← CURRENT
#   THOROUGH: popsize=25,  maxiter=50  → ~1,425 evals (3-5 minutes)
#   ORIGINAL: popsize=50,  maxiter=100 → ~5,200 evals (10-15 minutes even optimized!)
#
# Note: With the model caching fix, each evaluation is now ~100x faster,
# so even "THOROUGH" mode is reasonable.
#
# Force CPU for multiprocessing compatibility (set to False to use CUDA in single-process mode)
FORCE_CPU_FOR_PARALLEL = True

if FORCE_CPU_FOR_PARALLEL:
    DEVICE = torch.device("cpu")
    print("Note: Running in CPU mode for multiprocessing support")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "trained_model_HN_100K_with_dlayer.pth"
ASSET_PRICES_PATH = "datasets/assetprices.csv"
OPTIONS_DATA_PATH = "datasets/scalable_hn_dataset_250x60.csv"

# True params (adding sigma_eps=0.01 as default)
true_params = np.array([1e-6, 1.33e-6, 0.8, 5.0, 0.2, 0.01])

# Bounds exactly as in calibration.py
BOUNDS = [
    (1e-7, 1e-6),       # omega
    (1.15e-6, 1.50e-6), # alpha
    (0.0, 0.99),        # beta
    (0.0, 10.0),        # gamma
    (0.0, 1.0),         # lambda
    (1e-5, 0.1)         # sigma_eps
]

# --- 1. Load Data ---
def load_data():
    print(f"Loading asset prices from {ASSET_PRICES_PATH}...")
    prices_df = pd.read_csv(ASSET_PRICES_PATH)
    prices = prices_df.values.flatten()

    # Calculate Log Returns
    log_returns = np.log(prices[1:] / prices[:-1])

    print(f"Loading options data from {OPTIONS_DATA_PATH}...")
    options_df = pd.read_csv(OPTIONS_DATA_PATH)
    options_df = options_df[options_df["V"] > 0.5].reset_index(drop=True)

    return log_returns, options_df

# --- 2. Load Model ---
def load_model(path, device):
    print(f"Loading neural network from {path}...")
    model = ForwardModel(dlayer=True)
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    model.to(device)
    model.eval()
    return model

# --- 3. Loss Functions (Matching calibration.py) ---

def returns_loss(params, log_returns, r=0.05/252.0):
    # Only unpack first 5 for GARCH
    omega, alpha, beta, gamma, lambda_ = params[:5]

    # Ensure tensor
    if not isinstance(log_returns, torch.Tensor):
        log_returns = torch.tensor(log_returns, dtype=torch.float32, device=DEVICE)

    size = len(log_returns)
    h = torch.zeros(size, device=DEVICE)
    h[0] = torch.var(log_returns)

    # Exact recursion loop from calibration.py
    # Added minimal protection to prevent immediate crash on sqrt(negative)
    for i in range(size-1):
        h_val = h[i]
        if h_val <= 0:
            h_val = torch.tensor(1e-9, device=DEVICE)

        # Stability check for large values during DE exploration
        if h_val > 1000:
            h_val = torch.tensor(1000.0, device=DEVICE)

        term = (log_returns[i] - r - lambda_ * h_val) / torch.sqrt(h_val) - gamma * torch.sqrt(h_val)
        h[i+1] = omega + beta * h_val + alpha * (term ** 2)

    # Exact return formula from calibration.py
    loss = -0.5 * torch.sum(torch.log(h) + (log_returns - torch.pow((r+lambda_*h),2))/h)
    return loss

def options_loss(sigma_obs, sigma_model, sigma_eps=0.01):
    """
    Vectorized log-likelihood for option-implied volatilities.
    """
    device = sigma_obs.device
    sigma_eps_tensor = torch.tensor(sigma_eps, device=device)
    sigma_obs = sigma_obs.view(-1)
    sigma_model = sigma_model.view(-1)

    # Exact return formula from calibration.py
    return -0.5 * torch.sum(
        2 * torch.log(sigma_eps_tensor)
        + ((sigma_obs - sigma_model) / sigma_eps_tensor) ** 2
    )

def get_predicted_implied_vols(model, options_df, params, batch_size=2048):
    """
    Predict implied volatilities with batch processing for efficiency.
    """
    omega, alpha, beta, gamma, lambda_ = params[:5]
    N_opts = len(options_df)

    # Extract static features
    S0 = options_df["S0"].values
    m = options_df["m"].values
    r = options_df["r"].values
    T = options_df["T"].values
    callput = options_df["corp"].values

    alpha_arr = np.full(N_opts, alpha)
    beta_arr = np.full(N_opts, beta)
    omega_arr = np.full(N_opts, omega)
    gamma_arr = np.full(N_opts, gamma)
    lambda_arr = np.full(N_opts, lambda_)

    base_vals = np.column_stack([S0, m, r, T, callput, alpha_arr, beta_arr, omega_arr, gamma_arr, lambda_arr])

    eps = 1e-8
    log_vals = np.column_stack([
        np.log(alpha_arr + eps), np.log(beta_arr + eps),
        np.log(omega_arr + eps), np.log(gamma_arr + eps), np.log(lambda_arr + eps)
    ])

    X_np = np.hstack([base_vals, log_vals]).astype(np.float32)
    X = torch.tensor(X_np, device=DEVICE)

    # Batch prediction for efficiency
    predictions = []
    with torch.no_grad():
        for i in range(0, N_opts, batch_size):
            batch = X[i:min(i+batch_size, N_opts)]
            pred = model(batch)
            predictions.append(pred)

    sigma_pred = torch.cat(predictions, dim=0)
    return sigma_pred.view(-1)

def joint_loss(params, model, log_returns_t, options_df, sigma_obs_t, N, M, batch_size=2048):
    lr = returns_loss(params, log_returns_t)
    sigma_eps = params[5]

    sigma_model = get_predicted_implied_vols(model, options_df, params, batch_size=batch_size)
    lo = options_loss(sigma_obs_t, sigma_model, sigma_eps=sigma_eps)

    # Exact weighting from calibration.py
    joint = ((N+M)/(2*N))*lr + ((N+M)/(2*M))*lo

    # Essential NaN handling for optimizer
    if torch.isnan(joint) or torch.isinf(joint):
        return 1e9

    return -joint.item()

# --- 4. Initial Guess Logic ---
def initial_ll(params, log_returns_t):
    # Wrapper for minimize using only first 5 params
    return -returns_loss(params, log_returns_t).item()

def initial_guess(log_returns_np, log_returns_t):
    print("\n" + "="*60)
    print("STAGE 1: Initial Parameter Estimation")
    print("="*60)
    print("Running standard GARCH(1,1) for initial guess...")

    garch11 = arch_model(log_returns_np, vol="GARCH", p=1, q=1, dist="normal")
    res = garch11.fit(disp='off')

    # Map GARCH(1,1) to HN parameters roughly
    # omega, alpha, beta, gamma, lambda
    initial_params_5 = np.array([
        res.params["omega"],
        res.params["alpha[1]"],
        res.params["beta[1]"],
        0.0, # gamma start
        0.0  # lambda start
    ])

    print("GARCH(1,1) initial: omega={:.2e}, alpha={:.2e}, beta={:.4f}".format(
        initial_params_5[0], initial_params_5[1], initial_params_5[2]))

    # Optimize solely on returns first
    # Bounds for minimize: first 5 bounds
    bounds_5 = BOUNDS[:5]

    result = minimize(
        lambda p: initial_ll(p, log_returns_t),
        initial_params_5,
        bounds=bounds_5,
        method='L-BFGS-B',
        options={'disp': False}
    )

    params_5 = result.x
    loss = result.fun

    # Append default sigma_eps for the full parameter set
    full_params = np.append(params_5, 0.01) # Default sigma_eps=0.01

    print("\nRefined parameters after L-BFGS-B:")
    print("  omega:  {:.8e}".format(params_5[0]))
    print("  alpha:  {:.8e}".format(params_5[1]))
    print("  beta:   {:.6f}".format(params_5[2]))
    print("  gamma:  {:.6f}".format(params_5[3]))
    print("  lambda: {:.6f}".format(params_5[4]))
    print("\nInitial optimization loss: {:.4f}".format(loss))
    print("Two-norm error vs true params: {:.6f}".format(
        np.linalg.norm(true_params - full_params, ord=2)))

    return full_params

# --- 5. Parallel execution helper ---
# Module-level variables for worker processes (cached per worker)
_SHARED_MODEL = None
_SHARED_DEVICE = None
_SHARED_LOG_RETURNS_T = None
_SHARED_OPTIONS_DF = None
_SHARED_SIGMA_OBS_T = None
_SHARED_N = None
_SHARED_M = None
_SHARED_BATCH_SIZE = None
_EVAL_COUNTER = None

def _init_worker(model_path, log_returns_np, options_df_dict, sigma_obs_np, N, M, batch_size, eval_counter=None):
    """Initialize worker process with shared data - loads model ONCE per worker"""
    global _SHARED_MODEL, _SHARED_LOG_RETURNS_T, _SHARED_OPTIONS_DF
    global _SHARED_SIGMA_OBS_T, _SHARED_N, _SHARED_M, _SHARED_BATCH_SIZE, _EVAL_COUNTER, _SHARED_DEVICE

    # Setup device in worker process
    _SHARED_DEVICE = torch.device("cpu")

    # Load model ONCE per worker process (not on every evaluation!)
    _SHARED_MODEL = ForwardModel(dlayer=True)
    state_dict = torch.load(model_path, map_location=_SHARED_DEVICE, weights_only=False)
    _SHARED_MODEL.load_state_dict(state_dict)
    _SHARED_MODEL.to(_SHARED_DEVICE)
    _SHARED_MODEL.eval()

    # Convert to tensors ONCE
    _SHARED_LOG_RETURNS_T = torch.tensor(log_returns_np, dtype=torch.float32, device=_SHARED_DEVICE)
    _SHARED_SIGMA_OBS_T = torch.tensor(sigma_obs_np, dtype=torch.float32, device=_SHARED_DEVICE)

    # Convert to DataFrame ONCE
    _SHARED_OPTIONS_DF = pd.DataFrame(options_df_dict)

    _SHARED_N = N
    _SHARED_M = M
    _SHARED_BATCH_SIZE = batch_size
    _EVAL_COUNTER = eval_counter

def _evaluate_params(params):
    """
    Evaluate parameters in worker process.
    Uses shared data (including cached model) initialized by _init_worker.
    """
    try:
        # Increment counter if available
        if _EVAL_COUNTER is not None:
            with _EVAL_COUNTER.get_lock():
                _EVAL_COUNTER.value += 1
                count = _EVAL_COUNTER.value
                if count % 50 == 0 or count <= 5:
                    print(f"[Evaluation {count}] Processing...", flush=True)

        # Use pre-loaded model and pre-converted tensors from worker initialization
        # This avoids expensive I/O and conversions on every evaluation
        result = joint_loss(params, _SHARED_MODEL, _SHARED_LOG_RETURNS_T, _SHARED_OPTIONS_DF,
                           _SHARED_SIGMA_OBS_T, _SHARED_N, _SHARED_M, batch_size=_SHARED_BATCH_SIZE)

        # Check for invalid results
        if np.isnan(result) or np.isinf(result):
            return 1e9

        return result
    except Exception as e:
        # Return large penalty on error
        print(f"Worker error for params {params}: {e}", flush=True)
        return 1e9

# --- 6. Progress Callback ---
def progress_callback(xk, convergence=0):
    """Callback function to display progress during optimization"""
    global _iteration_counter, _last_update_time, _best_loss

    if not hasattr(progress_callback, 'iteration'):
        progress_callback.iteration = 0
        progress_callback.last_time = time.time()
        progress_callback.best_loss = float('inf')

    progress_callback.iteration += 1
    current_time = time.time()
    elapsed = current_time - progress_callback.last_time

    # Update every iteration or every 5 seconds
    if progress_callback.iteration == 1 or elapsed > 5.0:
        print(f"Iteration {progress_callback.iteration}: Evaluating population...", flush=True)
        progress_callback.last_time = current_time

    return False  # Don't stop optimization

# --- 7. Results Reporting ---
def report_results(result, elapsed_time, true_params, func_evals):
    """
    Comprehensive results reporting inspired by reference code.
    """
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Optimization time: {elapsed_time:.2f} seconds")
    print(f"Iterations: {result.nit}")
    print(f"Function evaluations: {func_evals}")
    print(f"Final objective value: {result.fun:.8f}")

    names = ["omega", "alpha", "beta", "gamma", "lambda", "sigma_eps"]

    print("\n" + "-"*60)
    print("PARAMETER COMPARISON")
    print("-"*60)
    print(f"{'Parameter':<10} {'True Value':<15} {'Calibrated':<15} {'Offset':<15}")
    print("-"*60)

    offsets = []
    for i, name in enumerate(names):
        offset = true_params[i] - result.x[i]
        offsets.append(offset)
        print(f"{name:<10} {true_params[i]:<15.8e} {result.x[i]:<15.8e} {offset:<15.8e}")

    print("-"*60)

    # Error metrics
    two_norm = np.linalg.norm(np.array(offsets), ord=2)
    max_abs = np.max(np.abs(offsets))

    print(f"\nTwo-norm error: {two_norm:.8f}")
    print(f"Max absolute offset: {max_abs:.8e}")

    status = {
        'time': elapsed_time,
        'func_evals': func_evals,
        'iterations': result.nit,
        'final_loss': result.fun,
        'two_norm_error': two_norm,
        'max_abs_error': max_abs
    }

    return offsets, status

# --- 8. Main ---
if __name__ == "__main__":
    print("="*60, flush=True)
    print("GARCH-HN CALIBRATION WITH PARALLEL DIFFERENTIAL EVOLUTION", flush=True)
    print("="*60, flush=True)

    # Load data and model
    log_returns_np, options_df = load_data()
    model = load_model(MODEL_PATH, DEVICE)

    N = len(log_returns_np)
    M = len(options_df)

    print("\nData Summary:", flush=True)
    print(f"  Returns samples (N): {N}", flush=True)
    print(f"  Options samples (M): {M}", flush=True)
    print(f"  Device: {DEVICE}", flush=True)

    log_returns_t = torch.tensor(log_returns_np, dtype=torch.float32, device=DEVICE)
    sigma_obs_t = torch.tensor(options_df["sigma"].values, dtype=torch.float32, device=DEVICE)

    # Get Initial Guess
    x0 = initial_guess(log_returns_np, log_returns_t)

    # Configuration for DE
    batch_size = 2048
    strategy = 'best1bin'
    maxiter = 30      # Reduced from 100 (saves ~3,500 evaluations)
    popsize = 15      # Reduced from 50 (fewer evaluations per iteration)

    # Determine parallelization strategy
    # Use custom parallel implementation to avoid PyTorch pickling issues
    use_parallel = (DEVICE.type == 'cpu' and FORCE_CPU_FOR_PARALLEL)

    if use_parallel:
        # Use all available CPU cores for maximum performance
        max_workers = mp.cpu_count()
        num_workers = max_workers
        print(f"  Available CPU cores: {max_workers}")
        print(f"  Using workers: {num_workers} (all cores)")
    else:
        num_workers = 1

    print("\n" + "="*60, flush=True)
    if use_parallel:
        print("STAGE 2: Parallel Differential Evolution (CPU Multi-Core)", flush=True)
    else:
        mode_str = "CUDA" if DEVICE.type == "cuda" else "CPU Single-Core"
        print(f"STAGE 2: Differential Evolution ({mode_str})", flush=True)
    print("="*60, flush=True)
    print(f"Strategy: {strategy}", flush=True)
    print(f"Max iterations: {maxiter}", flush=True)
    print(f"Population size: {popsize}", flush=True)
    print(f"Estimated function evaluations: ~{popsize * (maxiter + 1) + 150}", flush=True)
    if use_parallel:
        print(f"Workers: {num_workers}", flush=True)
    else:
        print(f"Workers: 1 (single process)", flush=True)
    print("Batch size: {}".format(batch_size), flush=True)
    print(f"Bounds: {BOUNDS}", flush=True)
    print("Initial guess provided: Yes", flush=True)

    # Prepare data for parallel processing (convert to numpy for pickling)
    sigma_obs_np = options_df["sigma"].values
    options_df_dict = options_df.to_dict('list')

    # Test single evaluation first to catch errors early
    print("\nTesting objective function...", flush=True)
    try:
        # Setup shared data for test
        _init_worker(MODEL_PATH, log_returns_np, options_df_dict, sigma_obs_np, N, M, batch_size, None)
        test_result = _evaluate_params(x0)
        print(f"✓ Test evaluation successful: loss = {test_result:.6f}", flush=True)
    except Exception as e:
        print(f"✗ WARNING: Test evaluation failed: {e}", flush=True)
        print("This may cause issues during optimization.", flush=True)
        import traceback
        traceback.print_exc()

    t0 = time.time()

    # Run differential evolution
    print("\nStarting optimization...", flush=True)
    print("(Progress updates every ~5 seconds)\n", flush=True)

    try:
        if use_parallel:
            print("Attempting parallel execution with custom pool...", flush=True)
            print(f"Initializing {num_workers} worker processes...", flush=True)

            # Create shared counter for progress tracking
            eval_counter = mp.Value('i', 0)

            # Create multiprocessing pool with initializer
            initializer = partial(_init_worker, MODEL_PATH, log_returns_np,
                                options_df_dict, sigma_obs_np, N, M, batch_size, eval_counter)

            with mp.Pool(processes=num_workers, initializer=initializer, initargs=()) as pool:
                print("✓ Worker pool created successfully!", flush=True)
                print("Starting differential evolution...", flush=True)
                print("Watch for '[Evaluation N]' messages to see progress\n", flush=True)

                # Use pool as workers for differential_evolution
                result = differential_evolution(
                    _evaluate_params,
                    bounds=BOUNDS,
                    strategy=strategy,
                    maxiter=maxiter,
                    popsize=popsize,
                    tol=0.01,
                    mutation=(0.5, 1),
                    recombination=0.8,
                    disp=True,
                    callback=progress_callback,
                    polish=True,
                    init='latinhypercube',
                    x0=x0,
                    workers=pool.map,
                    updating='deferred'
                )
        else:
            # Single-process mode - initialize shared data
            print("Using single-process mode...", flush=True)
            eval_counter = mp.Value('i', 0)
            _init_worker(MODEL_PATH, log_returns_np, options_df_dict, sigma_obs_np, N, M, batch_size, eval_counter)
            print("Starting differential evolution...\n", flush=True)
            result = differential_evolution(
                _evaluate_params,
                bounds=BOUNDS,
                strategy=strategy,
                maxiter=maxiter,
                popsize=popsize,
                tol=0.01,
                mutation=(0.5, 1),
                recombination=0.8,
                disp=True,
                callback=progress_callback,
                polish=True,
                init='latinhypercube',
                x0=x0,
                workers=1
            )
    except Exception as e:
        import traceback
        if use_parallel:
            print(f"\nWarning: Parallel execution failed ({type(e).__name__}: {e})", flush=True)
            traceback.print_exc()
            print("\nFalling back to single-process mode...", flush=True)

            # Retry with single-process
            eval_counter = mp.Value('i', 0)
            _init_worker(MODEL_PATH, log_returns_np, options_df_dict, sigma_obs_np, N, M, batch_size, eval_counter)
            print("Starting differential evolution (single-process)...\n", flush=True)
            result = differential_evolution(
                _evaluate_params,
                bounds=BOUNDS,
                strategy=strategy,
                maxiter=maxiter,
                popsize=popsize,
                tol=0.01,
                mutation=(0.5, 1),
                recombination=0.8,
                disp=True,
                callback=progress_callback,
                polish=True,
                init='latinhypercube',
                x0=x0,
                workers=1
            )
        else:
            traceback.print_exc()
            raise

    t1 = time.time()

    elapsed_time = t1 - t0

    # Report results
    offsets, status = report_results(result, elapsed_time, true_params, result.nfev)

    print("\n" + "="*60, flush=True)
    print("CALIBRATION COMPLETE", flush=True)
    print("="*60, flush=True)
