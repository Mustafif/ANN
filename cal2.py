import json
import time

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

true_params = np.array([1.33e-6, 0.8, 1e-6, 5.0, 0.2])
bounds = [
    (1.15e-6, 1.50e-6),  # alpha
    (0, 0.99),  # beta
    (1e-7, 1e-6),  # omega
    (0, 10),  # gamma
    (0, 1),  # lambda
    (1e-1, 3e-1),  # sigma epsilon
]


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps:0"
    if torch.backends.mps.is_available()
    else "cpu"
)
model_path = "trained_model_HN_100K_with_dlayer.pth"
asset_prices_path = "datasets/assetprices.csv"
options_data_path = "datasets/scalable_hn_dataset_250x60.csv"


def load_data(assets, options_data):
    prices_df = pd.read_csv(assets)
    prices = prices_df.values.flatten()
    log_returns = np.log(prices[1:] / prices[:-1])

    options_df = pd.read_csv(options_data)

    return log_returns, options_df


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


LR, _ = load_data(asset_prices_path, options_data_path)


def initial_ll(params):
    return -returns_loss(params)


def returns_loss(params, log_returns=LR, r=0.05 / 252.0):
    alpha, beta, omega, gamma, lambda_, _ = params
    size = len(log_returns)
    log_returns = torch.tensor(log_returns)
    h = torch.zeros(size)
    h[0] = torch.var(log_returns)

    for i in range(size - 1):
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

    return -0.5 * torch.sum(
        torch.log(h) + (log_returns - torch.pow((r + lambda_ * h), 2)) / h
    )


def initial_guess(log_returns):
    garch11 = arch_model(log_returns, vol="GARCH", p=1, q=1, dist="normal")
    res = garch11.fit(disp="off")

    initial_params = np.array(
        [
            res.params["alpha[1]"],
            res.params["beta[1]"],
            res.params["omega"],
            0.0,
            0.0,
            0.1,
        ]
    )

    result = minimize(
        initial_ll,
        initial_params,
        bounds=bounds,
        method="L-BFGS-B",
    )

    params = result.x
    loss = -result.fun
    return params


hn_garch_model = load_model(model_path, device)


def calibration_HN_GARCH(
    assets,
    options_data,
    model=hn_garch_model,
    batch_size=2048,
    seed=None,
    strategy="best1bin",
    bounds=bounds,
    polish=True,
):
    log_returns, options_df = load_data(assets, options_data)
    S0 = options_df["S0"].values
    m = options_df["m"].values
    r = options_df["r"].values
    T = options_df["T"].values
    corp = options_df["corp"].values
    V = options_df["V"].values
    sigma_obs = options_df["sigma"].values

    x0 = initial_guess(log_returns)
    if x0[3] == 0.0:
        x0[3] = 5.0
    elif x0[4] == 0.0:
        x0[4] = 0.2
    Y1_vals = []
    Y2_vals = []

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
        alpha, beta, omega, gamma, lambda_, sigma_eps = x
        eps = 1e-8

        df = pd.DataFrame(
            {
                "S0": S0,
                "m": m,
                "r": r,
                "T": T,
                "callput": corp,
                "alpha": alpha,
                "beta": beta,
                "omega": omega,
                "gamma": gamma,
                "lambda": lambda_,
                "sigma": sigma_obs,
                "V": V,
            }
        )

        dataset = SimDataset(df)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        criterion = nn.HuberLoss().to(device)
        sigma_model = []
        with torch.no_grad():
            for X, _ in data_loader:
                X = X.to(device)
                output = model(X)
                sigma_model.extend(output.cpu().numpy().flatten())

        sigma_model = np.array(sigma_model)
        lr_size = len(lr)
        lr = torch.tensor(lr)
        h = torch.zeros(lr_size)
        h[0] = (omega + alpha) / (1.0 - beta - alpha * gamma**2)
        r_tensor = torch.tensor(0.05 / 252.0)

        for i in range(lr_size - 1):
            h[i + 1] = (
                omega
                + beta * h[i]
                + alpha
                * (
                    (lr[i] - r_tensor - lambda_ * h[i]) / torch.sqrt(h[i])
                    - gamma * torch.sqrt(h[i])
                )
                ** 2
            )

        Y1 = -0.5 * torch.sum(
            torch.log(h) + (lr - torch.pow((r_tensor + lambda_ * h), 2)) / h
        )
        Y1_vals.append(Y1)

        sigma_eps_tensor = torch.tensor(sigma_eps)

        Y2 = -0.5 * torch.sum(
            2 * torch.log(sigma_eps_tensor)
            + ((sigma_obs - sigma_model) / sigma_eps_tensor) ** 2
        )
        Y2_vals.append(Y2)

        N = lr_size
        M = len(sigma_obs)

        joint = ((N + M) / (2 * N)) * Y1 + ((N + M) / (2 * M)) * Y2
        return -joint

    kwargs = dict(
        args=(),
        strategy=strategy,
        maxiter=200,
        popsize=15,
        tol=1e-2,
        mutation=(0.5, 1),
        recombination=0.8,
        seed=seed,
        callback=None,
        disp=False,  # Turned off per-strategy logspam
        polish=polish,
        x0=x0,
        constraints=(nlc,),
    )
    t0 = time.time()
    result = differential_evolution(objective_fn, bounds=bounds, **kwargs)
    t1 = time.time()

    case_time = t1 - t0
    x = np.array(result.x)
    two_norm_error = np.linalg.norm(x[:5] - true_params, ord=2)

    results = {
        "strategy": strategy,
        "alpha": x[0],
        "beta": x[1],
        "omega": x[2],
        "gamma": x[3],
        "lambda": x[4],
        "sigma_eps": x[5],
        "two_norm_error": two_norm_error,
        "time": case_time,
        "fun": result.fun,
    }

    with open(f"results_{strategy}.json", "w") as f:
        json.dump(results, f)

    return two_norm_error, case_time, result.fun


if __name__ == "__main__":
    # Standard Scipy Differential Evolution Strategies
    strategies = [
        "best1bin",
        "best1exp",
        "rand1exp",
        "randtobest1exp",
        "currenttobest1exp",
        "best2exp",
        "rand2exp",
        "randtobest1bin",
        "currenttobest1bin",
        "best2bin",
        "rand2bin",
        "rand1bin",
    ]

    results = {}
    best_strategy = None
    best_error = float("inf")

    print("Starting Differential Evolution grid test configurations...")

    for s in strategies:
        print(f"\nEvaluating strategy: {s}...")
        try:
            error, time_taken, fun_obj = calibration_HN_GARCH(
                asset_prices_path, options_data_path, strategy=s
            )
            results[s] = {"error": error, "time": time_taken, "fun": fun_obj}
            print(f"[{s}] -> L2 Error: {error:.6f} | Time: {time_taken:.2f}s")

            if error < best_error:
                best_error = error
                best_strategy = s
        except Exception as e:
            print(f"[{s}] Failed with error: {e}")

    print("\n" + "=" * 60)
    print("TEST CONFIGURATIONS SUMMARY")
    print("=" * 60)
    for s, metrics in results.items():
        print(
            f"Strategy: {s:18} | Two-Norm Error: {metrics['error']:.6f} | Time: {metrics['time']:.2f}s | Objective: {metrics['fun']:.4f}"
        )

    print("=" * 60)
    print(
        f"Optimal Strategy (Lowest Norm Error): {best_strategy} with an error of {best_error:.6f}"
    )
