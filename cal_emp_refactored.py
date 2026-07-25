#!/usr/bin/env python3
"""
Refactored calibration script that supports:
- Iterating through multiple folders
- Configurable GARCH model type (Duan or HN)
- Better organization and modularity
"""

import json
import time
import warnings
import os
from typing import List, Dict, Tuple, Optional

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from arch import arch_model
from scipy.optimize import NonlinearConstraint, differential_evolution, minimize
from torch.utils.data import DataLoader, RandomSampler

from ann import ForwardModel
from main import SimDataset


def stationarity_fn(x, *args):
    """Define the stationarity condition: beta + alpha * gamma^2"""
    alpha, beta, omega, gamma, lambda_, sigma_eps = x
    return beta + alpha * (gamma**2)


def get_garch_config(garch_model: str) -> Tuple[List[Tuple[float, float]], float, float]:
    """Get configuration for different GARCH models"""
    if garch_model == "hn":
        bounds = [
            (1e-6, 1.50e-6),  # alpha
            (0.2, 0.99),  # beta
            (1e-7, 1e-6),  # omega
            (1, 7),  # gamma
            (0.1, 1),  # lambda
            (1e-3, 3e-1),  # sigma epsilon
        ]
        scale = 10.0
        scale2 = 0.05
    else:  # duan
        bounds = [
            (1e-6, 1.50e-6),  # alpha
            (0.5, 0.99),  # beta
            (1e-8, 1e-5),  # omega
            (0.25, 0.5),  # gamma
            (0.3, 0.6),  # lambda
            (1e-2, 3e-1),  # sigma epsilon
        ]
        scale = 1
        scale2 = 1
    return bounds, scale, scale2


def get_model_path(garch_model: str) -> str:
    """Get the appropriate model path based on GARCH type"""
    if garch_model == "hn":
        return "trained_model_HN_100K_with_dlayer.pth"
    else:
        return "trained_model_dataset_duan_with_dlayer.pth"


def load_data(assets: str, options_data: str) -> Tuple[np.ndarray, pd.DataFrame, float]:
    """Load asset prices and options data"""
    prices_df = pd.read_csv(assets)
    prices = prices_df.values.flatten()
    log_returns = np.log(prices[1:] / prices[:-1])

    options_df = pd.read_csv(options_data)

    if "r" in options_df.columns and len(options_df) > 0:
        r_vals = options_df["r"].values
        if np.allclose(r_vals, r_vals[0]):
            r_scalar = float(r_vals[0])
        else:
            r_scalar = float(np.mean(r_vals))
    else:
        r_scalar = 0.05 / 252.0

    return log_returns, options_df, r_scalar


def load_model(path: str, device: torch.device) -> ForwardModel:
    """Load the neural network model"""
    print(f"Loading neural network from {path}...")
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model = ForwardModel()
    model.load_state_dict(state_dict)

    model.to(device)
    model.double()  # Cast model to float64 to match inputs
    model.eval()
    return model


def returns_loss(params: np.ndarray, log_returns: np.ndarray, r: float, garch_model: str) -> torch.Tensor:
    """Calculate returns loss for GARCH model"""
    alpha, beta, omega, gamma, lambda_, _ = params
    size = len(log_returns)
    log_returns = torch.tensor(log_returns)
    h = torch.zeros(size)
    h[0] = torch.var(log_returns)

    for i in range(size - 1):
        if garch_model == "hn":
            h[i + 1] = (
                omega
                + beta * h[i]
                + alpha
                * (
                    (log_returns[i] - r - lambda_ * h[i]) / torch.sqrt(h[i])
                    - gamma * torch.sqrt(h[i])
                )
                ** 2
            )
        else:
            h[i + 1] = (
                omega
                + beta * h[i]
                + (
                    alpha
                    * h[i]
                    * ((log_returns[i] - r - lambda_ * h[i]) / torch.sqrt(h[i]) - gamma)
                    ** 2
                )
            )

    return -0.5 * torch.sum(torch.log(h) + ((log_returns - (r + lambda_ * h)) ** 2) / h)


def initial_ll(params: np.ndarray, log_returns: np.ndarray, r: float, garch_model: str) -> float:
    """Initial log-likelihood function"""
    return -returns_loss(params, log_returns, r, garch_model).item()


def initial_guess(log_returns: np.ndarray, bounds: List[Tuple[float, float]], garch_model: str, true_params: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    """Generate initial guess for parameters using GARCH(1,1) fit"""
    variance = float(np.var(log_returns))
    if not np.isfinite(variance) or variance <= 0.0:
        variance = 1.0
    # ARCH recommends data scale in [1, 1000]; target variance ~100.
    target_variance = 100.0
    return_scale = np.sqrt(target_variance / variance)
    scaled_returns = log_returns * return_scale

    garch11 = arch_model(scaled_returns, vol="GARCH", p=1, q=1, dist="normal")
    res = garch11.fit(disp="off")

    initial_params = np.array(
        [
            res.params["alpha[1]"],
            res.params["beta[1]"],
            res.params["omega"] / (return_scale**2),
            0.0,
            0.3,
            0.1,
        ]
    )

    # Refine initial guess
    result = minimize(
        lambda x: initial_ll(x, log_returns, 0.05/252.0, garch_model),
        initial_params,
        bounds=bounds,
        method="SLSQP",
        constraints=(NonlinearConstraint(stationarity_fn, 0.0, 0.999),),
    )

    params = result.x
    loss = result.fun
    print("Initial Two-norm error:")
    print(np.linalg.norm(true_params - params[:5], ord=2))
    print(f"Initial Params: {params}")
    print(f"Loss: {loss}")
    k = len(params)
    n = len(log_returns)
    aic = 2 * k - 2 * np.log(loss)
    bic = k * np.log(n) - 2 * np.log(loss)

    print(f"AIC: {aic}")
    print(f"BIC: {bic}")

    return params, [aic, bic, loss]


def calibration_HN_GARCH(
    assets: str,
    options_data: str,
    garch_params_path: str,
    model: ForwardModel,
    device: torch.device,
    garch_model: str,
    bounds: List[Tuple[float, float]],
    scale: float,
    scale2: float,
    batch_size: int = 2048,
    seed: Optional[int] = 4234532,
    strategy: str = "best1bin",
    polish: bool = False,
) -> Dict:
    """Main calibration function"""
    
    # Load data
    log_returns, options_df, r_scalar = load_data(assets, options_data)
    
    # Load true parameters
    garch_params = pd.read_csv(garch_params_path)
    true_params = np.array([
        garch_params["alpha"].iloc[0],
        garch_params["beta"].iloc[0],
        garch_params["omega"].iloc[0],
        garch_params["gamma"].iloc[0],
        garch_params["lambda"].iloc[0],
    ])
    
    # Extract options data
    S0 = options_df["S0"].values
    m = options_df["m"].values
    r = options_df["r"].values
    T = options_df["T"].values
    corp = options_df["corp"].values
    V = options_df["V"].values
    sigma_obs = options_df["sigma"].values

    # Get initial guess
    x0, crit = initial_guess(log_returns, bounds, garch_model, true_params)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    Y1_vals = []
    Y2_vals = []

    # Pre-calculate constant base features and move to target device
    base_vals = np.column_stack([S0, m, r, T, corp]).astype(np.float64)
    base_tensors = torch.tensor(base_vals, dtype=torch.float64, device=device)

    sigma_obs_tensor = torch.tensor(sigma_obs, dtype=torch.float64, device=device)

    lr_size = len(log_returns)
    lr_tensor = torch.tensor(log_returns, dtype=torch.float64, device=device)
    r_val = torch.tensor(r_scalar, dtype=torch.float64, device=device)

    N_obs = len(sigma_obs_tensor)

    def objective_fn(
        x,
        lr=log_returns,
        S0=S0,
        m=m,
        r=r,
        T=T,
        corp=corp,
        V=V,
        sigma_obs=sigma_obs,
    ):
        x = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
        alpha, beta, omega, gamma, lambda_, sigma_eps = x
        eps = 1e-8

        # 1. Dynamic Features
        dyn_vals = torch.tensor(
            [alpha, beta, omega, gamma, lambda_], dtype=torch.float64, device=device
        )
        dyn_tensors = dyn_vals.expand(N_obs, 5)

        # 2. Log Features
        log_vals = torch.log(dyn_vals + eps)
        log_tensors = log_vals.expand(N_obs, 5)

        # 3. Concatenate all features (Shape: N x 15)
        X = torch.cat([base_tensors, dyn_tensors, log_tensors], dim=1)

        sigma_model = []
        with torch.no_grad():
            for i in range(0, N_obs, batch_size):
                batch_X = X[i : i + batch_size]
                output = model(batch_X)
                sigma_model.append(output)

        sigma_model_tensor = torch.cat(sigma_model).flatten()

        h = torch.zeros(lr_size, dtype=torch.float64, device=device)
        if garch_model == "hn":
            h[0] = (omega + alpha) / (1.0 - beta - alpha * gamma**2)
        else:
            h[0] = omega / (1 - alpha - beta)

        for i in range(lr_size - 1):
            if garch_model == "hn":
                h[i + 1] = (
                    omega
                    + beta * h[i]
                    + alpha
                    * (
                        (lr_tensor[i] - r_val - lambda_ * h[i]) / torch.sqrt(h[i])
                        - gamma * torch.sqrt(h[i])
                    )
                    ** 2
                )
            else:
                h[i + 1] = (
                    omega
                    + beta * h[i]
                    + (
                        alpha
                        * h[i]
                        * (
                            ((lr_tensor[i] - r_val - lambda_ * h[i]) / torch.sqrt(h[i]))
                            - gamma
                        )
                        ** 2
                    )
                )

        Y1 = (
            -0.5
            * scale
            * torch.sum(torch.log(h) + ((lr_tensor - (r_val + lambda_ * h)) ** 2) / h)
        )
        Y1_vals.append(Y1.item())

        sigma_eps_t = torch.tensor(sigma_eps, dtype=torch.float64, device=device)

        Y2 = (
            -0.5
            * scale2
            * torch.sum(
                2 * torch.log(sigma_eps_t)
                + ((sigma_obs_tensor - sigma_model_tensor) / sigma_eps_t) ** 2
            )
        )

        Y2_vals.append(Y2.item())

        if garch_model == "hn":
            joint = ((lr_size + N_obs) / (2 * lr_size)) * Y1 + (
                (lr_size + N_obs) / (2 * N_obs)
            ) * Y2
        else:
            joint = Y1 + Y2
        return -joint.item()

    # Set up differential evolution
    popsize_multiplier = 20
    kwargs = dict(
        args=(),
        strategy=strategy,
        maxiter=500,
        popsize=popsize_multiplier,
        tol=1e-3,
        mutation=(0.5, 1),
        recombination=0.8,
        seed=seed,
        callback=None,
        disp=True,
        polish=polish,
        constraints=(NonlinearConstraint(stationarity_fn, 0.0, 0.999),),
    )

    if not polish:
        # Create initial population dynamically sized based on popsize multiplier
        pop_size_total = popsize_multiplier * len(bounds)
        init_pop = np.random.rand(pop_size_total, len(bounds))
        for i in range(pop_size_total):
            for j in range(len(bounds)):
                init_pop[i, j] = bounds[j][0] + init_pop[i, j] * (
                    bounds[j][1] - bounds[j][0]
                )
        init_pop[0] = x0
        kwargs["init"] = init_pop
    else:
        kwargs["x0"] = x0

    # Run optimization
    t0 = time.time()
    result = differential_evolution(objective_fn, bounds=bounds, **kwargs)
    t1 = time.time()

    case_time = t1 - t0
    x = np.array(result.x)
    alpha, beta, omega, gamma, lambda_, sigma_eps = result.x
    alpha_true, beta_true, omega_true, gamma_true, lambda_true = true_params
    eps = 1e-8

    # Print results
    print(f"Calibration Time: {case_time:.2f} seconds")
    print(f"Alpha Calibrated: {alpha} | Alpha True: {alpha_true}")
    print(
        f"Alpha Error: {alpha - alpha_true} | Perc Error: {(alpha - alpha_true) / (alpha_true + eps) * 100:.2f}%"
    )

    print(f"Beta Calibrated: {beta} | Beta True: {beta_true}")
    print(
        f"Beta Error: {beta - beta_true} | Perc Error: {(beta - beta_true) / (beta_true + eps) * 100:.2f}%"
    )

    print(f"Omega Calibrated: {omega} | Omega True: {omega_true}")
    print(
        f"Omega Error: {omega - omega_true} | Perc Error: {(omega - omega_true) / (omega_true + eps) * 100:.2f}%"
    )

    print(f"Gamma Calibrated: {gamma} | Gamma True: {gamma_true}")
    print(
        f"Gamma Error: {gamma - gamma_true} | Perc Error: {(gamma - gamma_true) / (gamma_true + eps) * 100:.2f}%"
    )

    print(f"Lambda Calibrated: {lambda_} | Lambda True: {lambda_true}")
    print(
        f"Lambda Error: {lambda_ - lambda_true} | Perc Error: {(lambda_ - lambda_true) / (lambda_true + eps) * 100:.2f}%"
    )

    print(f"Sigma_Eps: {sigma_eps}")

    print(f"Two Norm Error: {np.linalg.norm(x[:5] - true_params, ord=2)}")
    print(f"Average Y1: {np.mean(np.array(Y1_vals))}")
    print(f"Average Y2: {np.mean(np.array(Y2_vals))}")
    print(
        f"Stationarity Check (beta + alpha * gamma^2): {beta + alpha * gamma**2} | True: {beta_true + alpha_true * gamma_true**2} | Init Check: {x0[1] + x0[0] * x0[3] ** 2}"
    )

    k = 5
    loss = result.fun
    n = len(options_data)
    aic = 2 * k - 2 * np.log(loss)
    bic = k * np.log(n) - 2 * np.log(loss)

    print(f"AIC: {aic}")
    print(f"BIC: {bic}")
    print(f"Loss: {loss}")

    # Save Results into JSON File
    results = {
        "strategy": strategy,
        "alpha": alpha,
        "beta": beta,
        "omega": omega,
        "gamma": gamma,
        "lambda": lambda_,
        "alpha_true": alpha_true,
        "beta_true": beta_true,
        "omega_true": omega_true,
        "gamma_true": gamma_true,
        "lambda_true": lambda_true,
        "alpha_init": x0[0],
        "beta_init": x0[1],
        "omega_init": x0[2],
        "gamma_init": x0[3],
        "lambda_init": x0[4],
        "two_norm_error": np.linalg.norm(x[:5] - true_params, ord=2),
        "aic": aic,
        "bic": bic,
        "aic_init": crit[0],
        "bic_init": crit[1],
        "loss": loss,
        "loss_init": crit[2],
    }

    return results


def get_file_paths(folder: str, garch_model: str) -> Tuple[str, str, str]:
    """Get file paths for a folder, handling different naming conventions"""
    # Asset prices file
    asset_prices_path = f"{folder}/asset_prices.csv"
    if not os.path.exists(asset_prices_path):
        asset_prices_path = f"{folder}/asset_prices_set_1.csv"
        if not os.path.exists(asset_prices_path):
            return None, None, None
    
    # Options data file
    options_data_path = f"{folder}/dataset_{garch_model}.csv"
    if not os.path.exists(options_data_path):
        options_data_path = f"{folder}/dataset.csv"
        if not os.path.exists(options_data_path):
            return None, None, None
            
    # GARCH parameters file
    garch_params_path = f"{folder}/garch_parameters_{garch_model}.csv"
    if not os.path.exists(garch_params_path):
        garch_params_path = f"{folder}/garch_parameters.csv"
        if not os.path.exists(garch_params_path):
            return None, None, None
    
    return asset_prices_path, options_data_path, garch_params_path


def process_folder(
    folder: str,
    garch_model: str,
    device: torch.device,
    strategy: str = "best1bin",
    polish: bool = False
) -> Dict:
    """Process a single folder with given GARCH model"""
    print(f"\n=== Processing folder: {folder} with {garch_model.upper()} model ===")
    
    # Get file paths
    asset_prices_path, options_data_path, garch_params_path = get_file_paths(folder, garch_model)
    if not all([asset_prices_path, options_data_path, garch_params_path]):
        print(f"Warning: Required files not found in {folder}, skipping...")
        return None
    
    # Get configuration
    bounds, scale, scale2 = get_garch_config(garch_model)
    model_path = get_model_path(garch_model)
    
    # Load model
    model = load_model(model_path, device)
    
    # Run calibration
    results = calibration_HN_GARCH(
        assets=asset_prices_path,
        options_data=options_data_path,
        garch_params_path=garch_params_path,
        model=model,
        device=device,
        garch_model=garch_model,
        bounds=bounds,
        scale=scale,
        scale2=scale2,
        strategy=strategy,
        polish=polish
    )
    
    # Save results
    os.makedirs("strats", exist_ok=True)
    with open(f"strats/results_{strategy}_{folder}_{garch_model}.json", "w") as f:
        json.dump(results, f)
    
    return results


def get_available_folders(base_path: str = ".") -> List[str]:
    """Get list of available folders that contain the required files"""
    folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            # Check if folder has at least some of the required files
            has_asset_prices = any(
                os.path.exists(os.path.join(item_path, f))
                for f in ["asset_prices.csv", "asset_prices_set_1.csv"]
            )
            has_dataset = any(
                os.path.exists(os.path.join(item_path, f))
                for f in ["dataset.csv", "dataset_duan.csv", "dataset_hn.csv"]
            )
            has_garch_params = any(
                os.path.exists(os.path.join(item_path, f))
                for f in ["garch_parameters.csv", "garch_parameters_duan.csv", "garch_parameters_hn.csv"]
            )
            
            if has_asset_prices and has_dataset and has_garch_params:
                folders.append(item)
    
    return sorted(folders)


def main():
    """Main function to iterate through folders and GARCH models"""
    
    # Configuration - can be modified or passed as command line arguments
    config = {
        "folders": None,  # None means auto-detect, or specify list like ["OEX", "Duan_Set1"]
        "garch_models": ["duan", "hn"],  # Models to test
        "strategy": "best1bin",
        "polish": False,
        "base_path": "."  # Base directory to search for folders
    }
    
    # Set up device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps:0"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    print(f"Using device: {device}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Get folders to process
    if config["folders"] is None:
        folders = get_available_folders(config["base_path"])
        print(f"Auto-detected folders: {folders}")
    else:
        folders = config["folders"]
        print(f"Using specified folders: {folders}")
    
    if not folders:
        print("No valid folders found. Exiting.")
        return
    
    # Process each folder and model combination
    all_results = {}
    for folder in folders:
        folder_results = {}
        for garch_model in config["garch_models"]:
            try:
                print(f"\nProcessing {folder} with {garch_model.upper()} model...")
                results = process_folder(
                    folder=folder,
                    garch_model=garch_model,
                    device=device,
                    strategy=config["strategy"],
                    polish=config["polish"]
                )
                if results:
                    folder_results[garch_model] = results
                else:
                    folder_results[garch_model] = {"status": "skipped", "reason": "missing_files"}
            except Exception as e:
                print(f"Error processing {folder} with {garch_model}: {e}")
                folder_results[garch_model] = {"error": str(e), "status": "failed"}
        
        all_results[folder] = folder_results
    
    # Save summary results
    os.makedirs("strats", exist_ok=True)
    summary_path = f"strats/summary_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n=== Processing Summary ===")
    for folder, folder_results in all_results.items():
        print(f"\nFolder: {folder}")
        for garch_model, result in folder_results.items():
            if "error" in result:
                print(f"  {garch_model}: FAILED - {result['error']}")
            elif "status" in result and result["status"] == "skipped":
                print(f"  {garch_model}: SKIPPED - {result.get('reason', 'unknown')}")
            else:
                print(f"  {garch_model}: COMPLETED")
    
    print(f"\n=== All processing complete ===")
    print(f"Detailed results saved to: {summary_path}")
    print(f"Individual results saved to strats/ directory")


if __name__ == "__main__":
    main()