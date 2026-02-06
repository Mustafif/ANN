import numpy as np
import pandas as pd
import scipy.optimize as opt
import torch
from arch import arch_model

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps:0"
    if torch.backends.mps.is_available()
    else "cpu"
)

# true_params = {
#   "omega": 1e-6,
#   "alpha": 1.33e-6,
#   "beta": 0.8,
#   "gamma": 5.0,
#   "lambda": 0.2
# }

true_params = [1e-6, 1.33e-6, 0.8, 5.0, 0.2]

trained_model_path = "trained_model_HN_100K_with_dlayer.pth"
cal_dataset_path = "datasets/scalable_hn_dataset_250x60.csv"
asset_prices_path = "datasets/assetprices.csv"

bounds = [
    (1e-7, 1e-6), # omega
    (1.15e-6, 1.50e-6), # alpha
    (0, 0.99), # beta
    (0, 10), # gamma
    (0, 1), # lambda
]

def log_returns(asset_prices_path: str):
    prices = pd.read_csv(asset_prices_path,header=None).values
    log_returns = np.log(prices[1:]/prices[:-1])
    return log_returns

LR = log_returns(asset_prices_path)

N = len(LR)
M = len(pd.read_csv(cal_dataset_path)["sigma"])

def returns_loss(params, log_returns=LR, r=0.05/252.0):
    omega, alpha, beta, gamma, lambda_ = params
    size = len(log_returns)
    log_returns = torch.tensor(log_returns)
    h = torch.zeros(size)
    h[0] = torch.var(log_returns)

    for i in range(size-1):
        h[i+1] = omega + beta*h[i] + alpha*((log_returns[i]-r-lambda_*h[i])/torch.sqrt(h[i]) - gamma*torch.sqrt(h[i]))**2

    return -0.5 * torch.sum(torch.log(h) + (log_returns - torch.pow((r+lambda_*h),2))/h)

def options_loss(sigma_obs, sigma_model, sigma_eps=0.01):
    """
    Vectorized log-likelihood for option-implied volatilities.
    """
    device = sigma_obs.device
    sigma_eps_tensor = torch.tensor(sigma_eps, device=device)
    sigma_obs = sigma_obs.view(-1)
    sigma_model = sigma_model.view(-1)
    return -0.5 * torch.sum(
        2 * torch.log(sigma_eps_tensor)
        + ((sigma_obs - sigma_model) / sigma_eps_tensor) ** 2
    )

def joint_loss(params, sigma_obs, sigma_model,  log_returns=LR, r=0.05/252.0, sigma_eps=0.01):
    lr = returns_loss(params, log_returns, r)
    lo = options_loss(sigma_obs, sigma_model, sigma_eps)

    joint = ((N+M)/(2*N))*lr + ((N+M)/(2*M))*lo
    return -joint



def initial_ll(params):
    return -returns_loss(params)

def initial_guess(log_returns):
    garch11 = arch_model(log_returns, vol="GARCH", p=1, q=1, dist="normal")
    res = garch11.fit(disp='off')

    initial_params = np.array([
        res.params["omega"],
        res.params["alpha[1]"],
        res.params["beta[1]"],
        0.0,
        0.0
    ])

    result = opt.minimize(
        initial_ll, initial_params, bounds=bounds, method='L-BFGS-B'
    )

    params = result.x
    loss = -result.fun
    print("Initial Two-norm error:")
    print(np.linalg.norm(true_params - params, ord=2))
    return params, loss

initial_params, initial_loss = initial_guess(LR)
