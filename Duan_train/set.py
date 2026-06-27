import numpy as np
import pandas as pd

sets = pd.read_csv("garch_parameters_duan.csv")
set1 = sets.loc[sets["set_id"] == 1].iloc[0]

omega = set1["omega"]
print(omega)
alpha = set1["alpha"]
beta = set1["beta"]
gamma = set1["gamma"]
lambda_ = set1["lambda"]

df = pd.read_csv("dataset_duan.csv")
print(len(df))
df_set1 = df[
    np.isclose(df["omega"], omega)
    & np.isclose(df["alpha"], alpha)
    & np.isclose(df["beta"], beta)
    & np.isclose(df["lambda"], lambda_)
]
print(len(df_set1))

# Save filtered dataset
df_set1.to_csv("dataset_duan_set1.csv", index=False)

# Save unique asset prices (S0) for set 1
asset_prices_set1 = df_set1["S0"].drop_duplicates().reset_index(drop=True)
asset_prices_set1.to_frame(name="S0").to_csv("asset_prices_set_1.csv", index=False)

# Save GARCH parameters derived from the filtered dataset
params_set1 = (
    df_set1[["omega", "alpha", "beta", "gamma", "lambda"]]
    .drop_duplicates()
    .assign(set_id=1)[["set_id", "omega", "alpha", "beta", "gamma", "lambda"]]
)
params_set1.to_csv("garch_parameters_duan_set1.csv", index=False)
