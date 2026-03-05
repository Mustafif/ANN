# alpha, beta, omega, gamma, lambda
import numpy as np
import pandas as pd
from numpy import mean, var
from scipy.stats import kurtosis, skew

initial = np.array(
    [1.15000000e-06, 3.53482404e-03, 1.01702514e-07, 9.26301851e00, 9.26301851e-01]
)
# calibrated = np.array(
#     [
#         1.3944573119856933e-06,
#         0.9114257132020877,
#         8.783501598499955e-07,
#         4.08100741350143,
#         0.011796298424632812,
#     ]
# )

calibrated = np.array(
    [
        1.15e-06,
        0.825541575411115,
        6.241474135503389e-07,
        5.532631955992681,
        0.018021270355291285,
    ]
)
true = np.array([1.33e-6, 0.8, 1e-6, 5.0, 0.2])

r = 0.05
M = 1
N = 2048
TN = [10, 50, 100, 252, 365, 512]
dt = 1 / N
Z = np.random.normal(0, 1, (N + 1, M))


def mc(alpha, beta, omega, gamma, lambda_):
    # Ensure all parameters are scalar floats
    omega = float(omega)
    alpha = float(alpha)
    beta = float(beta)
    gamma = float(gamma)
    lambda_ = float(lambda_)

    num_point = N + 1
    S = np.zeros((num_point, M))
    ht = np.zeros((num_point, M))

    # Calculate initial variance - ensure scalar operations
    initial_var = (omega + alpha) / (1.0 - beta - alpha * gamma**2)
    ht[0] = initial_var
    S[0] = 100

    for i in range(1, num_point):
        # ht[i] = (
        #     omega
        #     + beta * ht[i - 1]
        #     + alpha * (Z[i - 1] - gamma * np.sqrt(ht[i - 1])) ** 2
        # )
        # Rt = r + lambda_ * ht[i] + np.sqrt(ht[i]) * Z[i]

        ht[i] = (
            omega
            + beta * ht[i - 1]
            + alpha * (Z[i - 1] - (gamma + lambda_ + 0.5) * np.sqrt(ht[i - 1])) ** 2
        )
        Rt = r - 0.5 * ht[i] + Z[i] * np.sqrt(ht[i])
        S[i] = S[i - 1] * np.exp(Rt)
    return S


def compute_moments(Rt, TN):
    """Compute four moments for given time points"""
    moments = {}
    for t in TN:
        Rt_t = Rt[: t + 1].flatten()  # Ensure 1D array for moment calculations
        moments[str(t)] = {  # Convert t to string for JSON compatibility
            "mean": float(np.mean(Rt_t)),
            "variance": float(np.var(Rt_t)),
            "skewness": float(skew(Rt_t)),
            "kurtosis": float(kurtosis(Rt_t)),
        }
    return moments


if __name__ == "__main__":
    # Calculate mc and moments for each initial, calibrated, and true values

    results = []
    param_sets = {"Initial": initial, "Calibrated": calibrated, "True": true}

    for name, param in param_sets.items():
        Rt = mc(param[0], param[1], param[2], param[3], param[4])
        moments = compute_moments(Rt, TN)

        for t_str, metrics in moments.items():
            row = {"Parameter Set": name, "Time Horizon (T)": int(t_str)}
            row.update(metrics)
            results.append(row)

    df = pd.DataFrame(results)

    # Pivot to compare results side-by-side for each time horizon
    pivot_df = df.pivot(
        index="Time Horizon (T)",
        columns="Parameter Set",
        values=["mean", "variance", "skewness", "kurtosis"],
    )

    # Reorder columns to be cleaner if needed, but default sort is okay.
    # We want to see Mean(Calibrated) vs Mean(True) vs Mean(Initial) easily.

    print("Comparison of Moments for Initial, Calibrated, and True Parameters:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", "{:.6e}".format)

    # pivot_df.to_csv(
    #     "moment_comparison.csv"
    # )  # Save to CSV for easier analysis if needed
    print(pivot_df)

    # Calculate differences (Estimated - True)
    diff_data = []
    metrics = ["mean", "variance", "skewness", "kurtosis"]

    # pivot_df columns are MultiIndex: (metric, parameter_set)
    # We iterate through the index (Time Horizon)
    for t in pivot_df.index:
        row = {"Time Horizon (T)": t}
        for metric in metrics:
            true_val = pivot_df.loc[t, (metric, "True")]
            calib_val = pivot_df.loc[t, (metric, "Calibrated")]
            init_val = pivot_df.loc[t, (metric, "Initial")]

            row[(metric, "Calib - True")] = calib_val - true_val
            row[(metric, "Init - True")] = init_val - true_val
        diff_data.append(row)

    # Create DataFrame for differences
    # We need to restructure the dictionary keys to match a MultiIndex or just flatten it
    # Flattening for easier display: e.g., "mean (Calib - True)"

    flat_diff_data = []
    for d in diff_data:
        flat_row = {"Time Horizon (T)": d["Time Horizon (T)"]}
        for k, v in d.items():
            if k == "Time Horizon (T)":
                continue
            metric, diff_type = k
            flat_row[f"{metric} ({diff_type})"] = v
        flat_diff_data.append(flat_row)

    diff_df = pd.DataFrame(flat_diff_data)
    diff_df.set_index("Time Horizon (T)", inplace=True)

    # Reorder columns to group by metric
    # Columns are like: "mean (Calib - True)", "mean (Init - True)", "variance ...", ...
    ordered_cols = []
    for metric in metrics:
        ordered_cols.append(f"{metric} (Calib - True)")
        ordered_cols.append(f"{metric} (Init - True)")

    diff_df = diff_df[ordered_cols]

    print("\nDifferences from True Parameters (Estimated - True):")
    print(diff_df)

    # Calculate Absolute True Error
    abs_error_data = []

    for d in diff_data:
        abs_row = {"Time Horizon (T)": d["Time Horizon (T)"]}
        for k, v in d.items():
            if k == "Time Horizon (T)":
                continue
            metric, diff_type = k
            # diff_type is "Calib - True" or "Init - True"
            # We want "Abs Error (Calib)" or "Abs Error (Init)"

            error_type = "Calib" if "Calib" in diff_type else "Init"
            abs_row[f"{metric} (Abs Err {error_type})"] = abs(v)

        abs_error_data.append(abs_row)

    abs_error_df = pd.DataFrame(abs_error_data)
    abs_error_df.set_index("Time Horizon (T)", inplace=True)

    # Reorder columns
    abs_ordered_cols = []
    for metric in metrics:
        abs_ordered_cols.append(f"{metric} (Abs Err Calib)")
        abs_ordered_cols.append(f"{metric} (Abs Err Init)")

    abs_error_df = abs_error_df[abs_ordered_cols]

    print("\nAbsolute True Error (|Estimated - True|):")
    print(abs_error_df)

    abs_error_df.to_csv(
        "absolute_error_comparison2.csv"
    )  # Save to CSV for easier analysis if needed

    # Calculate "Winner" table (Which has less error?)
    winner_data = []

    for d in diff_data:
        winner_row = {"Time Horizon (T)": d["Time Horizon (T)"]}
        for metric in metrics:
            calib_err = abs(d[(metric, "Calib - True")])
            init_err = abs(d[(metric, "Init - True")])

            if calib_err < init_err:
                winner_row[metric] = "Calibrated"
            elif init_err < calib_err:
                winner_row[metric] = "Initial"
            else:
                winner_row[metric] = "Tie"
        winner_data.append(winner_row)

    winner_df = pd.DataFrame(winner_data)
    winner_df.set_index("Time Horizon (T)", inplace=True)
    winner_df = winner_df[metrics]  # Reorder columns

    print("\nBest Parameter Set (Lowest Error):")
    print(winner_df)
