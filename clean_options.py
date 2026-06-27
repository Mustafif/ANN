import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

folder = "OptionPrices"
files = glob.glob(os.path.join(folder, "*.csv"))

for file in files:
    if "_cleaned" in file:
        continue

    ticker = os.path.basename(file).replace(".csv", "")
    filter_path = f"{folder}/{ticker}_cleaned.csv"
    df = pd.read_csv(file)

    # Convert dates to pandas datetime objects
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])

    # --------------------------------------------------
    # 1. Remove all options with zero volume
    # --------------------------------------------------
    df = df[df["volume"] > 10].copy()

    # --------------------------------------------------
    # 2. Keep only contracts expiring on the third Friday of any month
    # --------------------------------------------------
    # def is_third_friday(d: pd.Timestamp) -> bool:
    #     # Friday is weekday == 4 (Monday=0, ..., Sunday=6)
    #     if d.weekday() != 4:
    #         return False
    #     # Find the first day of the month
    #     first_of_month = d.replace(day=1)
    #     # First Friday of the month
    #     first_friday_offset = (4 - first_of_month.weekday()) % 7
    #     first_friday = first_of_month + pd.Timedelta(days=first_friday_offset)
    #     # Third Friday = first Friday + 14 days
    #     third_friday = first_friday + pd.Timedelta(days=14)
    #     return d == third_friday

    # df["is_third_friday"] = df["expiration"].apply(is_third_friday)
    # df = df[df["is_third_friday"]].copy()
    # df.drop(columns="is_third_friday", inplace=True)

    # --------------------------------------------------
    # 3. Keep contracts with maturities ≈ 30, 91, 182, 365 days
    #    with windows:
    #      - 30 or 91: [τ-2, τ+2]
    #      - 182 or 365: [τ-5, τ+5]
    # --------------------------------------------------

    # Compute days to maturity τ = (expiry - trade_date)
    df["tau"] = (df["expiration"] - df["date"]).dt.days

    def in_maturity_bucket(tau: int) -> bool:
        # 30 or 91 days: [τ-2, τ+2]
        if 28 <= tau <= 32:  # 30 ± 2
            return True
        if 89 <= tau <= 93:  # 91 ± 2
            return True
        # 182 or 365 days: [τ-5, τ+5]
        if 177 <= tau <= 187:  # 182 ± 5
            return True
        if 360 <= tau <= 370:  # 365 ± 5
            return True
        return False

    df = df[df["tau"].apply(in_maturity_bucket)].copy()

    # --------------------------------------------------
    # 4. For each contract, exclude trading days
    #    with fewer than 40 available strikes
    # --------------------------------------------------
    # Interpretation: for a given underlying & trade_date,
    # keep only days where you observe at least 40 distinct strikes.
    # If you have just one underlying, you can omit 'underlying'.

    # group_cols = ["date"]  # or ['underlying', 'trade_date']
    # strike_count = (
    #     df.groupby(group_cols)["strike"].nunique().rename("n_strikes").reset_index()
    # )

    # # Keep days with ≥ 40 strikes
    # good_days = strike_count[strike_count["n_strikes"] >= 40][group_cols]

    # # Merge back and filter
    # df = df.merge(good_days, on=group_cols, how="inner")

    df.to_csv(filter_path, index=False)
