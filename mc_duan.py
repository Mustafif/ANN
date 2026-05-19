import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# ── MC / moment helpers ───────────────────────────────────────────────────────


def mc_duan(N: int, alpha, beta, omega, gamma, lam, M: int = 1000, r: float = 0.05):
    """Monte-Carlo simulation of the Duan GARCH return process."""
    dt = 1 / N
    Rt = np.zeros((N + 1, M))
    ht = np.zeros((N + 1, M))
    Z = np.random.randn(N + 1, M)

    ht[0, :] = 1e-6

    for i in range(1, N):
        ht[i, :] = (
            omega
            + beta * ht[i - 1, :]
            + alpha * ht[i - 1, :] * (Z[i - 1, :] - gamma) ** 2
        )
        Rt[i, :] = (
            r * dt
            - 0.5 * ht[i, :]
            + lam * np.sqrt(ht[i, :])
            + Z[i, :] * np.sqrt(ht[i, :])
        )

    return Rt[1:], ht[1:]


def four_moments(N: int, alpha, beta, omega, gamma, lam) -> np.ndarray:
    """Return [mean, variance, skewness, kurtosis] of simulated returns."""
    Rt, _ = mc_duan(N, alpha, beta, omega, gamma, lam)
    flat = Rt.ravel()
    return np.array(
        [
            flat.mean(),
            flat.var(ddof=1),  # unbiased  (MATLAB var(...,0))
            skew(flat, bias=False),  # MATLAB skewness(...,0)
            kurtosis(
                flat, bias=False, fisher=False
            ),  # Pearson   (MATLAB kurtosis(...,0))
        ]
    )


def rel_err(x_p: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.abs(x_p - x) / np.abs(x)


# ── main analysis function ────────────────────────────────────────────────────


def run_moment_analysis(
    json_path: str = "strats/results_best1bin_duan.json",
    plot_prefix: str = "Figs/lambda_1",
    N_vals: list = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load calibration results from *json_path*, compute the first four moments
    of the Duan GARCH process for each horizon in *N_vals*, write per-moment
    CSVs, and save two figures:

        <plot_prefix>_RelErr.png   – relative error (calibrated vs initial)
        <plot_prefix>_Moments.png  – absolute moments (true vs calib vs init)

    Returns
    -------
    moment_table : pd.DataFrame  – moments for all three parameter sets
    rel_err_df   : pd.DataFrame  – relative errors vs true
    """
    if N_vals is None:
        N_vals = [30, 60, 120, 252, 512, 1024]

    os.makedirs(os.path.dirname(plot_prefix) or ".", exist_ok=True)

    # ── load JSON ─────────────────────────────────────────────────────────────
    with open(json_path) as f:
        data = json.load(f)

    calib_p = np.array(
        [data["alpha"], data["beta"], data["omega"], data["gamma"], data["lambda"]]
    )
    true_p = np.array(
        [
            data["alpha_true"],
            data["beta_true"],
            data["omega_true"],
            data["gamma_true"],
            data["lambda_true"],
        ]
    )
    init_p = np.array(
        [
            data["alpha_init"],
            data["beta_init"],
            data["omega_init"],
            data["gamma_init"],
            data["lambda_init"],
        ]
    )

    # ── compute moments ───────────────────────────────────────────────────────
    T = len(N_vals)
    X_true = np.zeros((T, 4))
    X_calib = np.zeros((T, 4))
    X_init = np.zeros((T, 4))

    for i, n in enumerate(N_vals):
        X_true[i] = four_moments(n, *true_p)
        X_calib[i] = four_moments(n, *calib_p)
        X_init[i] = four_moments(n, *init_p)

    rel_err_cal = rel_err(X_calib, X_true)
    rel_err_init = rel_err(X_init, X_true)

    # ── build moment table ────────────────────────────────────────────────────
    mom_labels = ["Mean", "Variance", "Skewness", "Kurtosis"]
    row_idx = [str(n) for n in N_vals]

    cols, col_names = [], []
    for grp, arr in [("True", X_true), ("Calib", X_calib), ("Init", X_init)]:
        for j, m in enumerate(mom_labels):
            cols.append(arr[:, j])
            col_names.append(f"{grp}_{m}")

    moment_table = pd.DataFrame(np.column_stack(cols), columns=col_names, index=row_idx)

    # per-moment CSVs
    for m in mom_labels:
        sub = moment_table[[f"True_{m}", f"Calib_{m}", f"Init_{m}"]].copy()
        sub.index.name = "N"
        sub.to_csv(f"{m.lower()}_moment_table.csv")

    # ── relative-error table ──────────────────────────────────────────────────
    var_names = [f"{m} Cal" for m in mom_labels] + [f"{m} Init" for m in mom_labels]
    rel_err_df = pd.DataFrame(
        np.hstack([rel_err_cal, rel_err_init]),
        columns=var_names,
        index=row_idx,
    )

    # ── plot 1: relative error ────────────────────────────────────────────────
    N_arr = np.array(N_vals)
    colors = plt.cm.tab10([0, 1])

    fig, axes = plt.subplots(4, 1, figsize=(8, 12), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for m_idx, (ax, mom) in enumerate(zip(axes, mom_labels)):
        ax.plot(
            N_arr,
            rel_err_cal[:, m_idx],
            "-",
            color=colors[0],
            linewidth=1.5,
            label="Calibrated",
        )
        ax.plot(
            N_arr,
            rel_err_init[:, m_idx],
            "--",
            color=colors[1],
            linewidth=1.5,
            label="Initial",
        )
        ax.set_ylabel(mom)
        ax.grid(True)
        ax.legend(loc="best")
        if m_idx == 0:
            ax.set_title("Relative Error by Moment: Calibrated vs Initial")
        if m_idx == 3:
            ax.set_xlabel("Time Period")

    out_rel = f"{plot_prefix}_RelErr.png"
    fig.savefig(out_rel, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_rel}")

    # ── plot 2: moments ───────────────────────────────────────────────────────
    colors3 = plt.cm.tab10([0, 1, 2])

    fig, axes = plt.subplots(4, 1, figsize=(8, 12), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for m_idx, (ax, mom) in enumerate(zip(axes, mom_labels)):
        ax.plot(
            N_arr, X_true[:, m_idx], "-", color=colors3[0], linewidth=1.5, label="True"
        )
        ax.plot(
            N_arr,
            X_calib[:, m_idx],
            "--",
            color=colors3[1],
            linewidth=1.5,
            label="Calibrated",
        )
        ax.plot(
            N_arr,
            X_init[:, m_idx],
            ":",
            color=colors3[2],
            linewidth=1.5,
            label="Initial",
        )
        ax.set_ylabel(mom)
        ax.grid(True)
        ax.legend(loc="best")
        if m_idx == 0:
            ax.set_title("Moments: True vs Calibrated vs Initial")
        if m_idx == 3:
            ax.set_xlabel("Time Period")

    out_mom = f"{plot_prefix}_Moments.png"
    fig.savefig(out_mom, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_mom}")

    return moment_table, rel_err_df


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    moment_table, rel_err_df = run_moment_analysis(
        json_path="strats/results_best1bin_duan.json",
        plot_prefix="Figs/lambda_1",
    )

    print("\nMoment table (first row):")
    print(moment_table.iloc[0].to_string())
    print("\nRelative error table (first row):")
    print(rel_err_df.iloc[0].to_string())
