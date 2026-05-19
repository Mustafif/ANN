import pandas as pd

df = pd.read_csv("dataset_duan.csv")

# Extract asset prices
asset_prices = df["S0"]
unique = asset_prices.unique().tolist()

# save into asset_prices_set_1.csv
unique_df = pd.DataFrame(unique, columns=["S0"])
unique_df.to_csv("asset_prices_set_1.csv", index=False)
print(f"Unique asset prices: {asset_prices.unique()}")
