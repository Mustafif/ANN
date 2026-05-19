"""
convert_to_duan.py
------------------
Converts three real market data tables into the Duan GARCH dataset format:
    S0, m, r, T, corp, alpha, beta, omega, gamma, lambda, sigma, V

GARCH parameters (alpha, beta, omega, gamma, lambda) are set to 0.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — edit these
# ---------------------------------------------------------------------------

OPTIONS_PATH = "OP_1996_to_2002_109497.csv"  # Table 1: options data
RATES_PATH = "ZC_1996_to_2002_109497.csv"  # Table 2: risk-free rates
STOCKS_PATH = "SP_1996_to_2002_109497.csv"  # Table 3: stock prices
OUTPUT_PATH = "dataset_duan.csv"

# ---------------------------------------------------------------------------
# Column name maps — edit if your CSVs use different headers
# ---------------------------------------------------------------------------

MAP_OPT = {
    "securityid": "securityid",
    "date": "date",
    "strike": "strike",
    "expiration": "expiration",
    "callput": "callput",  # 'C' or 'P'
    "bestbid": "bestbid",
    "bestoffer": "bestoffer",
    "impliedvol": "impliedvolatility",
}

MAP_RATE = {
    "date": "date",
    "days": "days",
    "rate": "rate",  # annualised %, e.g. 5.76
}

MAP_STK = {
    "securityid": "securityid",
    "date": "date",
    "close": "closeprice",
}

# ---------------------------------------------------------------------------


def convert():
    # ── load ────────────────────────────────────────────────────────────────
    print("Loading data...")
    opt = pd.read_csv(OPTIONS_PATH, low_memory=False)
    rat = pd.read_csv(RATES_PATH, low_memory=False)
    stk = pd.read_csv(STOCKS_PATH, low_memory=False)

    # ── rename to canonical names ────────────────────────────────────────────
    opt = opt.rename(columns={v: k for k, v in MAP_OPT.items()})
    rat = rat.rename(columns={v: k for k, v in MAP_RATE.items()})
    stk = stk.rename(columns={v: k for k, v in MAP_STK.items()})

    # ── parse dates ──────────────────────────────────────────────────────────
    opt["date"] = pd.to_datetime(opt["date"], errors="coerce")
    opt["expiration"] = pd.to_datetime(opt["expiration"], errors="coerce")
    rat["date"] = pd.to_datetime(rat["date"], errors="coerce")
    stk["date"] = pd.to_datetime(stk["date"], errors="coerce")

    # ── compute T (days to expiry) ───────────────────────────────────────────
    opt["T"] = (opt["expiration"] - opt["date"]).dt.days

    # ── filter bad rows ──────────────────────────────────────────────────────
    n_before = len(opt)
    opt = opt[
        opt["T"].notna()
        & (opt["T"] > 0)
        & opt["bestbid"].notna()
        & (opt["bestbid"] > 0)
        & opt["bestoffer"].notna()
        & (opt["bestoffer"] > 0)
        & opt["strike"].notna()
        & (opt["strike"] > 0)
        & opt["impliedvol"].notna()
        & (opt["impliedvol"] > 0)
    ].copy()
    print(f"  Options: {n_before} rows → {len(opt)} after filtering illiquid/bad rows")

    # ── option mid-price ─────────────────────────────────────────────────────
    opt["V"] = (opt["bestbid"] + opt["bestoffer"]) / 2.0

    # ── callput flag ─────────────────────────────────────────────────────────
    opt["corp"] = opt["callput"].str.strip().str.upper().map({"C": 1, "P": 0})
    opt = opt[opt["corp"].notna()].copy()

    # ── merge S0 from stock close prices ─────────────────────────────────────
    stk_close = stk[["securityid", "date", "close"]].drop_duplicates(
        subset=["securityid", "date"]
    )
    opt = opt.merge(stk_close, on=["securityid", "date"], how="left")
    opt = opt[opt["close"].notna() & (opt["close"] > 0)].copy()
    opt.rename(columns={"close": "S0"}, inplace=True)
    print(f"  Options after S0 merge: {len(opt)}")

    # ── moneyness ────────────────────────────────────────────────────────────
    opt["m"] = (opt["strike"] / opt["contractsize"]) / opt["S0"]

    # ── risk-free rate ────────────────────────────────────────────────────────
    print("  Interpolating risk-free rates...")

    rat_unique_dates = rat["date"].unique()

    # Map each option date to the nearest available rate date
    opt_dates = opt["date"].unique()
    date_map = {
        d: rat_unique_dates[np.argmin(np.abs(rat_unique_dates - d))] for d in opt_dates
    }
    opt["_rate_date"] = opt["date"].map(date_map)

    # Pre-build rate curves: date -> (days array, rates array)
    rate_curves = {
        d: grp.sort_values("days")[["days", "rate"]].values
        for d, grp in rat.groupby("date")
    }

    # Interpolate once per unique (_rate_date, T) pair then merge back
    unique_pairs = opt[["_rate_date", "T"]].drop_duplicates().copy()
    unique_pairs["r"] = unique_pairs.apply(
        lambda row: float(
            np.interp(
                row["T"],
                rate_curves[row["_rate_date"]][:, 0],
                rate_curves[row["_rate_date"]][:, 1],
            )
        )
        / 100.0
        / 365.0,
        axis=1,
    )

    opt = opt.merge(unique_pairs, on=["_rate_date", "T"], how="left")
    opt.drop(columns=["_rate_date"], inplace=True)

    # ── GARCH parameters set to 0 ─────────────────────────────────────────────
    for col in ["alpha", "beta", "omega", "gamma", "lambda"]:
        opt[col] = 0.0

    # ── final output ─────────────────────────────────────────────────────────
    opt.rename(columns={"impliedvol": "sigma"}, inplace=True)
    opt["T"] = opt["T"].astype(int)

    out_cols = [
        "S0",
        "m",
        "r",
        "T",
        "corp",
        "alpha",
        "beta",
        "omega",
        "gamma",
        "lambda",
        "sigma",
        "V",
    ]
    result = opt[out_cols].dropna()

    print(f"\nFinal dataset: {len(result)} rows")
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}")
    print("\nSample output:")
    print(result.head(5).to_string(index=False))


if __name__ == "__main__":
    convert()
