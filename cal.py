import json
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

folder = "src2/BSX/Period1"
garch_model = "duan"

# bounds = [
#     (1e-6, 1.50e-6),  # alpha
#     (0.2, 0.99),  # beta
#     (1e-7, 1e-6),  # omega
#     (1, 7),  # gamma
#     (0.1, 1),  # lambda
#     (1e-2, 1e-1),  # sigma epsilon
# ]
if garch_model == "hn":
    bounds = [
        (1e-6, 1.50e-6),  # alpha
        (0.2, 0.99),  # beta
        (1e-7, 1e-6),  # omega
        (1, 7),  # gamma
        (0.1, 1),  # lambda
        (1e-3, 3e-1),  # sigma epsilon
    ]
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
    "trained_model_HN_100K_with_dlayer.pth"
    if garch_model == "hn"
    else "trained_model_dataset_duan_with_dlayer.pth"
)
asset_prices_path = f"{folder}/asset_prices_set_1.csv"
options_data_path = f"{folder}/dataset_{garch_model}.csv"
garch_params_path = f"{folder}/garch_parameters_{garch_model}.csv"
garch_params = pd.read_csv(garch_params_path)
true_params = np.array(
    [
        garch_params["alpha"].iloc[0],
        garch_params["beta"].iloc[0],
        garch_params["omega"].iloc[0],
        garch_params["gamma"].iloc[0],
        garch_params["lambda"].iloc[0],
    ]
)


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


LR, _, R_DATA = load_data(asset_prices_path, options_data_path)


def initial_ll(params):
    return -returns_loss(params)


def returns_loss(params, log_returns=LR, r=None):
    if r is None:
        r = R_DATA
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


def initial_guess(log_returns):
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

    # try to calibrate initial also using differential evolution

    result = minimize(
        initial_ll,
        initial_params,
        bounds=bounds,
        method="SLSQP",
        constraints=(nlc,),
    )

    # result = differential_evolution(
    #     initial_ll,
    #     bounds=bounds,
    #     strategy="best1bin",
    #     maxiter=50,
    #     popsize=10,
    #     tol=1e-2,
    #     mutation=(0.5, 1),
    #     recombination=0.8,
    #     seed=42,
    #     callback=None,
    #     disp=True,
    #     polish=False,
    #     init="latinhypercube",
    #     constraints=(nlc,),
    # )

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


hn_garch_model = load_model(model_path, device)


# def calibration_HN_GARCH(
#     assets,
#     options_data,
#     model=hn_garch_model,
#     batch_size=2048,
#     seed=None,
#     strategy="best1bin",
#     bounds=bounds,
#     polish=False,
# ):
#     log_returns, options_df = load_data(assets, options_data)
#     S0 = options_df["S0"].values
#     m = options_df["m"].values
#     r = options_df["r"].values
#     T = options_df["T"].values
#     corp = options_df["corp"].values
#     V = options_df["V"].values
#     sigma_obs = options_df["sigma"].values

#     x0 = initial_guess(log_returns)
#     if x0[3] == 0.0:
#         x0[3] = 5.0
#     elif x0[4] == 0.0:
#         x0[4] = 0.2

#     # Clip x0 to ensure it's strictly within bounds
#     x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

#     # x0 = np.array([1.33e-6, 0.8, 1e-6, 5.0, 0.2, 0.1])
#     Y1_vals = []
#     Y2_vals = []

#     def objective_fn(
#         x,
#         lr=log_returns,
#         S0=S0,
#         m=m,
#         r=r,
#         T=T,
#         corp=corp,
#         V=V,
#         sigma_obs=sigma_obs,
#     ):
#         x = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
#         # sigma_eps = 1e-6
#         alpha, beta, omega, gamma, lambda_, sigma_eps = x
#         eps = 1e-8

#         df = pd.DataFrame(
#             {
#                 "S0": S0,
#                 "m": m,
#                 "r": r,
#                 "T": T,
#                 "callput": corp,
#                 "alpha": alpha,
#                 "beta": beta,
#                 "omega": omega,
#                 "gamma": gamma,
#                 "lambda": lambda_,
#                 "sigma": sigma_obs,
#                  "V": V,
#             }
#         )

#         dataset == SimDataset(df)
#         # sampler = RandomSampler(dataset)
#         data_loader = DataLoader(
#             dataset,
#             batch_size=batch_size,
#             # sampler=sampler,
#             shuffle=False,
#             pin_memory=True,
#         )

#         criterion = nn.HuberLoss().to(device)
#         sigma_model = []
#         with torch.no_grad():
#             for X, _ in data_loader:
#                 X = X.to(device)
#                 output = model(X)
#                 sigma_model.extend(output.cpu().numpy().flatten())

#         sigma_model = np.array(sigma_model)
#         lr_size = len(lr)
#         lr = torch.tensor(lr)
#         h = torch.zeros(lr_size)
#         # h[0] = torch.var(lr)
#         h[0] = (omega + alpha) / (1.0 - beta - alpha * gamma**2)
#         r = torch.tensor(0.05 / 252.0)

#         for i in range(lr_size - 1):
#             h[i + 1] = (
#                 omega
#                 + beta * h[i]
#                 + alpha
#                 * (
#                     (lr[i] - r - lambda_ * h[i]) / torch.sqrt(h[i])
#                     - gamma * torch.sqrt(h[i])
#                 )
#                 ** 2
#             )

#         Y1 = -0.5 * torch.sum(torch.log(h) + ((lr - (r + lambda_ * h)) ** 2) / h)
#         # print(f"Y1: {Y1}")
#         Y1_vals.append(Y1)

#         sigma_eps = torch.tensor(sigma_eps)
#         # sigma_obs = sigma_obs
#         # sigma_model = sigma_model.view(-1)

#         Y2 = -0.5 * torch.sum(
#             2 * torch.log(sigma_eps) + ((sigma_obs - sigma_model) / sigma_eps) ** 2
#         )
#         # Y2 = -torch.mean((torch.tensor(sigma_obs - sigma_model) / torch.tensor(sigma_obs)) ** 2)
#         # Y2 = -0.5 * torch.mean(
#         #     torch.tensor(((sigma_obs - sigma_model) / sigma_obs)) ** 2
#         # )
#         # Y2 = torch.nn.HuberLoss()(torch.tensor(sigma_obs), torch.tensor(sigma_model).view(-1))
#         Y2_vals.append(Y2)

#         # print(f"Y2: {Y2}")

#         N = lr_size
#         M = len(sigma_obs)

#         joint = ((N + M) / (2 * N)) * Y1 + ((N + M) / (2 * M)) * Y2
#         return -joint

#     # init_pop = np.random.rand(15, 5)
#     # for i in range(15):
#     #     for j in range(5):
#     #         init_pop[i, j] = bounds[j][0] + init_pop[i, j] * (bounds[j][1] - bounds[j][0])
#     # init_pop[0] = x0
#     kwargs = dict(
#         args=(),
#         strategy=strategy,
#         maxiter=200,
#         popsize=15,
#         tol=1e-2,
#         mutation=(0.5, 1),
#         recombination=0.8,
#         seed=seed,
#         callback=None,
#         disp=True,
#         polish=polish,
#         constraints=(nlc,),
#         # init="array",
#     )
#     if not polish:
#         # Create initial population with x0 as first member
#         init_pop = np.random.rand(15, len(bounds))
#         for i in range(15):
#             for j in range(len(bounds)):
#                 init_pop[i, j] = bounds[j][0] + init_pop[i, j] * (
#                     bounds[j][1] - bounds[j][0]
#                 )
#         init_pop[0] = x0
#         kwargs["init"] = init_pop
#     else:
#         kwargs["x0"] = x0
#     t0 = time.time()
#     result = differential_evolution(objective_fn, bounds=bounds, **kwargs)
#     t1 = time.time()


#     case_time = t1 - t0
#     x = np.array(result.x)
#     alpha, beta, omega, gamma, lambda_, sigma_eps = result.x
#     alpha_true, beta_true, omega_true, gamma_true, lambda_true = true_params
def calibration_HN_GARCH(
    assets,
    options_data,
    model=hn_garch_model,
    batch_size=2048,
    seed=4234532,
    # seed=1111111111,
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

    x0, crit = initial_guess(log_returns)
    # if x0[3] == 0.0:
    #     x0[3] = 5.0
    # elif x0[4] == 0.0:
    #     x0[4] = 0.2

    # Clip x0 to ensure it's strictly within bounds
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
        # h[0] = torch.var(lr_tensor)

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

        # joint = ((lr_size + N_obs) / (2 * lr_size)) * Y1 + (
        #     (lr_size + N_obs) / (2 * N_obs)
        # ) * Y2
        joint = Y1 + Y2
        return -joint.item()

    # kwargs = dict(
    #     args=(),
    #     strategy=strategy,
    #     maxiter=200,
    #     popsize=15,
    #     tol=1e-2,
    #     mutation=(0.5, 1),
    #     recombination=0.8,
    #     seed=seed,
    #     callback=None,
    #     disp=True,
    #     polish=polish,
    #     constraints=(nlc,),
    # )
    #
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
        # init="latinhypercube",
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

    t0 = time.time()
    result = differential_evolution(objective_fn, bounds=bounds, **kwargs)
    # result = minimize(
    #     objective_fn,
    #     result.x,
    #     method="Nelder-Mead",
    #     bounds=bounds,
    #     tol=1e-2,
    #     constraints=(nlc,),
    # )
    t1 = time.time()

    case_time = t1 - t0
    x = np.array(result.x)
    alpha, beta, omega, gamma, lambda_, sigma_eps = result.x
    alpha_true, beta_true, omega_true, gamma_true, lambda_true = true_params
    eps = 1e-8
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
    with open(f"strats/results_{strategy}_{folder}.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    strategies = [
        "best1bin",
        # "best1exp",
        # "rand1exp",
        # "randtobest1exp",
        # "currenttobest1exp",
        # "best2exp",
        # "rand2exp",
        # "randtobest1bin",
        # "currenttobest1bin",
        # "best2bin",
        # "rand2bin",
        # "rand1bin",
    ]

    calibration_HN_GARCH(
        asset_prices_path,
        options_data_path,
        strategy=strategy,
    )

    from mc_duan import run_moment_analysis

    mt, re = run_moment_analysis(
        json_path=f"strats/results_best1bin_{folder}.json",
        plot_prefix=f"Figs/{folder}_best1bin",
    )
