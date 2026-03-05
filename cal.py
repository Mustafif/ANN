import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from arch import arch_model
from scipy.optimize import differential_evolution, minimize
from torch.utils.data import DataLoader, RandomSampler

from ann import ForwardModel
from main import SimDataset

true_params = np.array([1.33e-6, 0.8, 1e-6, 5.0, 0.2])
bounds = [
    (1.15e-6, 1.50e-6),  # alpha
    (0, 0.99),  # beta
    (1e-7, 1e-6),  # omega
    (0, 10),  # gamma
    (0, 1),  # lambda
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
    alpha, beta, omega, gamma, lambda_ = params
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
        ]
    )

    result = minimize(initial_ll, initial_params, bounds=bounds, method="L-BFGS-B")

    params = result.x
    loss = -result.fun
    print("Initial Two-norm error:")
    print(np.linalg.norm(true_params - params, ord=2))
    print(f"Initial Params: {params}")
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
        sigma_eps = 1e-6
        alpha, beta, omega, gamma, lambda_ = x
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
        # sampler = RandomSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            # sampler=sampler,
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
        h[0] = torch.var(lr)
        r = torch.tensor(0.05 / 252.0)

        for i in range(lr_size - 1):
            h[i + 1] = (
                omega
                + beta * h[i]
                + alpha
                * (
                    (lr[i] - r - lambda_ * h[i]) / torch.sqrt(h[i])
                    - gamma * torch.sqrt(h[i])
                )
                ** 2
            )

        Y1 = -0.5 * torch.sum(torch.log(h) + (lr - torch.pow((r + lambda_ * h), 2)) / h)

        sigma_eps = torch.tensor(sigma_eps)
        # sigma_obs = sigma_obs
        # sigma_model = sigma_model.view(-1)

        Y2 = -0.5 * torch.sum(
            2 * torch.log(sigma_eps) + ((sigma_obs - sigma_model) / sigma_eps) ** 2
        )

        N = lr_size
        M = len(sigma_obs)

        joint = ((N + M) / (2 * N)) * Y1 + ((N + M) / (2 * M)) * Y2
        return -joint

    # init_pop = np.random.rand(15, 5)
    # for i in range(15):
    #     for j in range(5):
    #         init_pop[i, j] = bounds[j][0] + init_pop[i, j] * (bounds[j][1] - bounds[j][0])
    # init_pop[0] = x0
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
        disp=True,
        polish=polish,
        x0=x0,
    )
    t0 = time.time()
    result = differential_evolution(objective_fn, bounds=bounds, **kwargs)
    t1 = time.time()

    case_time = t1 - t0
    x = np.array(result.x)
    alpha, beta, omega, gamma, lambda_ = result.x
    alpha_true, beta_true, omega_true, gamma_true, lambda_true = true_params

    print(f"Calibration Time: {case_time:.2f} seconds")
    print(f"Alpha Calibrated: {alpha} | Alpha True: {alpha_true}")
    print(f"Alpha Error: {alpha - alpha_true}")

    print(f"Beta Calibrated: {beta} | Beta True: {beta_true}")
    print(f"Beta Error: {beta - beta_true}")

    print(f"Omega Calibrated: {omega} | Omega True: {omega_true}")
    print(f"Omega Error: {omega - omega_true}")

    print(f"Gamma Calibrated: {gamma} | Gamma True: {gamma_true}")
    print(f"Gamma Error: {gamma - gamma_true}")

    print(f"Lambda Calibrated: {lambda_} | Lambda True: {lambda_true}")
    print(f"Lambda Error: {lambda_ - lambda_true}")

    print(f"Two Norm Error: {np.linalg.norm(x - true_params, ord=2)}")

    # Save Results into JSON File
    results = {
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
        "two_norm_error": np.linalg.norm(x - true_params, ord=2),
    }
    with open("results.json", "w") as f:
        json.dump(results, f)


calibration_HN_GARCH(
    asset_prices_path,
    options_data_path,
)
