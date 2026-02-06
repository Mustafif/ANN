import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from arch import arch_model
from scipy.optimize import differential_evolution, minimize

from ann import ForwardModel

# --- Configuration ---
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
        if h_val <= 0: h_val = torch.tensor(1e-9, device=DEVICE)

        # Stability check for large values during DE exploration
        if h_val > 1000: h_val = torch.tensor(1000.0, device=DEVICE)

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

def get_predicted_implied_vols(model, options_df, params):
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

    with torch.no_grad():
        sigma_pred = model(X)

    return sigma_pred.view(-1)

def joint_loss(params, model, log_returns_t, options_df, sigma_obs_t, N, M):
    lr = returns_loss(params, log_returns_t)
    sigma_eps = params[5]

    sigma_model = get_predicted_implied_vols(model, options_df, params)
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
    print("\nRunning initial guess using standard GARCH...")
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

    # Optimize solely on returns first
    # Bounds for minimize: first 5 bounds
    bounds_5 = BOUNDS[:5]

    result = minimize(
        lambda p: initial_ll(p, log_returns_t),
        initial_params_5,
        bounds=bounds_5,
        method='L-BFGS-B'
    )

    params_5 = result.x
    loss = result.fun

    # Append default sigma_eps for the full parameter set
    full_params = np.append(params_5, 0.01) # Default sigma_eps=0.01

    print(f"Initial optimization loss: {loss:.4f}")
    print("Initial Two-norm error (vs true_params):")
    # Compare with true params
    print(np.linalg.norm(true_params - full_params, ord=2))

    return full_params

# --- 5. Main ---
if __name__ == "__main__":
    log_returns_np, options_df = load_data()
    model = load_model(MODEL_PATH, DEVICE)

    N = len(log_returns_np)
    M = len(options_df)

    print(f"N (returns) = {N}")
    print(f"M (options) = {M}")

    log_returns_t = torch.tensor(log_returns_np, dtype=torch.float32, device=DEVICE)
    sigma_obs_t = torch.tensor(options_df["sigma"].values, dtype=torch.float32, device=DEVICE)

    # Get Initial Guess
    x0 = initial_guess(log_returns_np, log_returns_t)
    print(f"Initial Guess Params: {x0}")

    print("\nStarting Differential Evolution...")
    print(f"Bounds: {BOUNDS}")

    def obj_func(params):
        return joint_loss(params, model, log_returns_t, options_df, sigma_obs_t, N, M)

    result = differential_evolution(
        obj_func,
        bounds=BOUNDS,
        strategy='best1bin',
        maxiter=100,
        popsize=50,
        tol=1e-6,
        disp=True,
        x0=x0
    )

    print("\n" + "="*40)
    print("Calibration Complete")
    print("="*40)
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Final Objective Value (-JointLL): {result.fun:.6f}")

    names = ["omega", "alpha", "beta", "gamma", "lambda", "sigma_eps"]
    print("\nCalibrated Parameters:")
    for n, v in zip(names, result.x):
        print(f"{n: <8}: {v:.8f}")

    print("\nFinal Two-Norm Error:")
    print(np.linalg.norm(result.x - true_params, ord=2))
