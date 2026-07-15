import pandas as pd
from scipy.interpolate import interp1d

# -----------------------------
# Load files
# -----------------------------
opt = pd.read_csv("OptionPrices/OEX_cleaned.csv")
spot = pd.read_csv("StockPrices/SP_OEX.csv")
zc = pd.read_csv("StockPrices/zero_curve.csv")

opt["date"] = pd.to_datetime(opt["date"])
spot["date"] = pd.to_datetime(spot["date"])
zc["date"] = pd.to_datetime(zc["date"])

# -----------------------------
# Adjusted spot price
# -----------------------------
spot["S0"] = spot["closeprice"] * spot["adjustmentfactor2"]

spot = spot[["date", "S0"]]

# -----------------------------
# Merge spot onto options
# -----------------------------
df = opt.merge(spot, on="date", how="left")


# -----------------------------
# Risk-free interpolation
# -----------------------------
def get_rate(row):
    curve = zc[zc["date"] == row["date"]]

    if len(curve) == 0:
        return None

    f = interp1d(curve["days"], curve["rate"], fill_value="extrapolate")

    return float(f(row["tau"])) / 100.0


df["r"] = df.apply(get_rate, axis=1)

# -----------------------------
# Time to maturity
# -----------------------------
df["T"] = df["tau"]

# -----------------------------
# Moneyness
# -----------------------------
df["strike"] = df["strike"] / df["contractsize"]

df["m"] = df["S0"] / df["strike"]

# -----------------------------
# Call/Put indicator
# -----------------------------
df["corp"] = (df["callput"] == "C").astype(int)

# -----------------------------
# Volatility
# -----------------------------
df["sigma"] = df["impliedvolatility"]

# -----------------------------
# Option mid-price
# -----------------------------
df["V"] = (df["bestbid"] + df["bestoffer"]) / 2

final_df = pd.DataFrame(
    {
        "S0": df["S0"],
        "m": df["m"],
        "r": df["r"],
        "T": df["T"],
        "corp": df["corp"],
        "alpha": 0.0,
        "beta": 0.0,
        "omega": 0.0,
        "gamma": 0.0,
        "lambda": 0.0,
        "sigma": df["sigma"],
        "V": df["V"],
    }
)

final_df.to_csv("OEX/dataset_duan.csv", index=False)
