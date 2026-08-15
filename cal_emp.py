import json
import os
import time
import warnings

# HN: scale1 = 10; scale2 = 0.05
# Duan: scale1 = 20; scale2 = 1
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


# Define the stationarity condition: beta + alpha * gamma^2
def stationarity_fn(x, *args):
    alpha, beta, omega, gamma, lambda_, sigma_eps = x
    return beta + alpha * (gamma**2)


# Constraint: 0 < beta + alpha * gamma^2 < 0.999
nlc = NonlinearConstraint(stationarity_fn, 0.0, 0.999)

SRC2_ROOT = "src2"
REPORT_DIR = "strats"
garch_model = "duan"

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
else:
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

strategy = "best1bin"

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps:0"
    if torch.backends.mps.is_available()
    else "cpu"
)

model_path = (
    "trained_model_dataset_hn_with_out_dlayer.pth"
    if garch_model == "hn"
    else "trained_model_dataset_duan_with_dlayer.pth"
)

true_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])


def load_data(assets, options_data):
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


def load_model(path, device):
    print(f"Loading neural network from {path}...")
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model = ForwardModel()
    model.load_state_dict(state_dict)

    model.to(device)
    model.double()  # Cast model to float64 to match inputs
    model.eval()
    return model


def initial_ll(params, log_returns, r):
    return -returns_loss(params, log_returns, r)


def returns_loss(params, log_returns, r):
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


def initial_guess(log_returns, r):
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

    result = minimize(
        lambda p: initial_ll(p, log_returns, r),
        initial_params,
        bounds=bounds,
        method="SLSQP",
        constraints=(nlc,),
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
    assets,
    options_data,
    model,
    batch_size=2048,
    seed=4234532,
    strategy="best1bin",
    bounds=bounds,
    polish=False,
):
    log_returns, options_df, r_scalar = load_data(assets, options_data)
    S0 = options_df["S0"].values
    m = options_df["m"].values
    r = options_df["r"].values
    T = options_df["T"].values
    corp = options_df["corp"].values
    V = options_df["V"].values
    sigma_obs = options_df["sigma"].values

    x0, crit = initial_guess(log_returns, r_scalar)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    Y1_vals = []
    Y2_vals = []

    base_vals = np.column_stack([S0, m, r, T, corp]).astype(np.float64)
    base_tensors = torch.tensor(base_vals, dtype=torch.float64, device=device)

    sigma_obs_tensor = torch.tensor(sigma_obs, dtype=torch.float64, device=device)

    lr_size = len(log_returns)
    lr_tensor = torch.tensor(log_returns, dtype=torch.float64, device=device)
    r_val = torch.tensor(r_scalar, dtype=torch.float64, device=device)

    N_obs = len(sigma_obs_tensor)
    last_sigma_model = []
    def objective_fn(x):
        x = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
        alpha, beta, omega, gamma, lambda_, sigma_eps = x
        eps = 1e-8

        dyn_vals = torch.tensor(
            [alpha, beta, omega, gamma, lambda_], dtype=torch.float64, device=device
        )
        dyn_tensors = dyn_vals.expand(N_obs, 5)

        log_vals = torch.log(dyn_vals + eps)
        log_tensors = log_vals.expand(N_obs, 5)

        X = torch.cat([base_tensors, dyn_tensors, log_tensors], dim=1)

        sigma_model = []
        with torch.no_grad():
            for i in range(0, N_obs, batch_size):
                batch_X = X[i : i + batch_size]
                output = model(batch_X)
                sigma_model.append(output)

        sigma_model_tensor = torch.cat(sigma_model).flatten()
        last_sigma_model.clear()
        last_sigma_model.append(sigma_model_tensor.detach())

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
        constraints=(nlc,),
    )

    if not polish:
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

    t0 = time.time()
    result = differential_evolution(objective_fn, bounds=bounds, **kwargs)
    t1 = time.time()

    objective_fn(result.x)  # one extra forward pass to refresh sigma_model_tensor at the winning params
    sigma_model_final = last_sigma_model[0]
    mean_iv_mse = torch.mean((sigma_obs_tensor - sigma_model_final) ** 2).item()
    mean_iv_mae = torch.mean(torch.abs(sigma_obs_tensor - sigma_model_final)).item()
    print(f"Mean IV MSE: {mean_iv_mse}, Mean IV MAE: {mean_iv_mae}")


    case_time = t1 - t0
    x = np.array(result.x)
    alpha, beta, omega, gamma, lambda_, sigma_eps = result.x
    alpha_true, beta_true, omega_true, gamma_true, lambda_true = true_params
    eps = 1e-8
    print(f"Calibration Time: {case_time:.2f} seconds")
    print(f"Two Norm Error: {np.linalg.norm(x[:5] - true_params, ord=2)}")
    print(f"Average Y1: {np.mean(np.array(Y1_vals))}")
    print(f"Average Y2: {np.mean(np.array(Y2_vals))}")

    k = 5
    loss = result.fun
    n = len(options_df)
    aic = 2 * k - 2 * np.log(loss)
    bic = k * np.log(n) - 2 * np.log(loss)

    print(f"AIC: {aic}")
    print(f"BIC: {bic}")
    print(f"Loss: {loss}")

    return {
        "strategy": strategy,
        "alpha": alpha,
        "beta": beta,
        "omega": omega,
        "gamma": gamma,
        "lambda": lambda_,
        "sigma_eps": sigma_eps,
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
        "calibration_time_sec": case_time,
        "n_options": n,
        "n_returns": lr_size,
        "mean_iv_mse": mean_iv_mse,
        "mean_iv_mae": mean_iv_mae,
    }


def find_ticker_periods(src2_root):
    tickers = sorted(
        d for d in os.listdir(src2_root)
        if os.path.isdir(os.path.join(src2_root, d))
    )
    for ticker in tickers:
        ticker_dir = os.path.join(src2_root, ticker)
        periods = sorted(
            d for d in os.listdir(ticker_dir)
            if os.path.isdir(os.path.join(ticker_dir, d))
        )
        for period in periods:
            yield ticker, period, os.path.join(ticker_dir, period)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    model = load_model(model_path, device)

    rows = []
    for ticker, period, period_dir in find_ticker_periods(SRC2_ROOT):
        assets_path = os.path.join(period_dir, "asset_prices.csv")
        options_path = os.path.join(period_dir, "dataset.csv")

        print(f"\n=== Calibrating {ticker}/{period} ===")
        row = {"ticker": ticker, "period": period}
        try:
            result = calibration_HN_GARCH(
                assets_path,
                options_path,
                model=model,
                strategy=strategy,
            )
            row.update(result)
            row["status"] = "ok"
        except Exception as e:
            row["status"] = "failed"
            row["error"] = str(e)
            print(f"FAILED {ticker}/{period}: {e}")

        rows.append(row)

    report_df = pd.DataFrame(rows)
    csv_path = os.path.join(REPORT_DIR, f"calibration_report_{garch_model}.csv")
    report_df.to_csv(csv_path, index=False)
    print(f"\nWrote combined report to {csv_path}")


if __name__ == "__main__":
    main()
