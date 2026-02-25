# alpha, beta, omega, gamma, lambda
import json

import numpy as np
from numpy import mean, var
from scipy.stats import kurtosis, skew

initial = np.array([1.15000000e-06, 3.53482404e-03, 1.01702514e-07, 9.26301851e+00, 9.26301851e-01])
calibrated = np.array([1.3944573119856933e-06, 0.9114257132020877,8.783501598499955e-07, 4.08100741350143, 0.011796298424632812])
true = np.array([1.33e-6, 0.8, 1e-6, 5.0, 0.2])

r = 0.05
M = 10
N = 512
TN = [10, 50, 100, 252, 365, 512]
dt = 1 / N
Z = np.random.normal(0, 1, (N+1, M))

def mc(alpha, beta,omega, gamma, lambda_):
    # Ensure all parameters are scalar floats
    omega = float(omega)
    alpha = float(alpha)
    beta = float(beta)
    gamma = float(gamma)
    lambda_ = float(lambda_)

    num_point = N+1
    Rt = np.zeros((num_point, M))
    ht = np.zeros((num_point, M))

    # Calculate initial variance - ensure scalar operations
    initial_var = (omega + alpha)/(1.0 - beta - alpha * gamma**2)
    ht[0] = initial_var
    Rt[0] = 0

    for i in range(1, num_point):
        ht[i] = omega + beta*ht[i-1] + alpha*(Z[i-1] - gamma*np.sqrt(ht[i-1]))**2
        Rt[i] = r + lambda_*ht[i] + np.sqrt(ht[i])*Z[i]
    return Rt

def compute_moments(Rt, TN):
    """Compute four moments for given time points"""
    moments = {}
    for t in TN:
        Rt_t = Rt[:t+1].flatten()  # Ensure 1D array for moment calculations
        moments[str(t)] = {  # Convert t to string for JSON compatibility
            'mean': float(np.mean(Rt_t)),
            'variance': float(np.var(Rt_t)),
            'skewness': float(skew(Rt_t)),
            'kurtosis': float(kurtosis(Rt_t))
        }
    return moments

# Calculate mc and moments for each initial, calibrated, and true values

for param in [initial, calibrated, true]:
    Rt = mc(param[0], param[1], param[2], param[3], param[4])
    moments = compute_moments(Rt, TN)
    # print(f"Moments for {param}:")
    print(json.dumps(moments, indent=4))
