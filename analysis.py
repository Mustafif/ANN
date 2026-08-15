import pandas as pd
it = [
    "AAPL",
    "CSCO",
    "DELL",
    "AMD",
    "INTC",
    "NVDA",
    "ORCL",
    "CRM",
    "NTAP",
    "MSFT",
    "MU",
    "IBM",
    "AVGO",
    "ADBE",
    "HPE",
    "QCOM",
    "TXN",
    "MSI",
    "ADSK",
    "ADI",
]

finance = [
    "BLK",
    "BX",
    "MA",
    "JPM",
    "PYPL",
    "MS",
    "FIS",
    "C",
    "COF",
    "WFC",
    "BAC",
    "AFL",
    "ALL",
    "AXP",
    "AIG",
    "GS",
    "HIG",
    "HBAN",
    "L",
    "MTB",
]

healthcare = [
    "BIIB",
    "BSX",
    "BMY",
    "CAH",
    "COR",
    "CVS",
    "DHR",
    "VTRS",
    "WAT",
    "AMGN",
    "BAX",
    "BDX",
    "JNJ",
    "PFE",
    "SYK",
    "TMO",
    "UNH",
    "ABT",
    "LH",
    "LLY",
]

industrial = [
    "SNA",
    "LUV",
    "SWK",
    "TXT",
    "DOV",
    "ETN",
    "EMR",
    "EFX",
    "GD",
    "HON",
    "PCAR",
    "PH",
    "PAYX",
    "ITW",
    "GE",
    "FDX",
    "CAT",
    "CTAS",
    "GWW",
    "WM",
]

cd  = [
    "AMZN",
    "AZO",
    "BBY",
    "CCL",
    "DRI",
    "DHI",
    "EBAY",
    "F",
    "GPC",
    "HAS",
    "HD",
    "LEN",
    "LOW",
    "MAR",
    "MCD",
    "NKE",
    "PHM",
    "SBUX",
    "TPR",
    "TJX",
]

df = pd.read_csv("calibration_report_duan.csv")
map = {}
for ticker in it:
    map[ticker] = "IT"
for ticker in finance:
    map[ticker] = "Finance"
for ticker in healthcare:
    map[ticker] = "Health"
for ticker in industrial:
    map[ticker] = "Industrial"
for ticker in cd:
    map[ticker] = "Retail"

df['sector'] = df['ticker'].map(map)
df_success = df[df['status'] == 'ok'].copy()
metrics_to_aggregate = {
    'aic': 'mean',
    'bic': 'mean',
    'aic_init': 'mean',
    'bic_init': 'mean',
    'loss': 'mean',
    'loss_init': 'mean',
    'n_options': 'mean',
    'n_returns': 'mean'
}

grouped = df_success.groupby(['sector', 'period'])

# Create summary dataframe
summary = grouped.agg(metrics_to_aggregate).reset_index()

# Round for readability
summary[['aic', 'bic', 'aic_init', 'bic_init']] = summary[[
   'aic', 'bic', 'aic_init', 'bic_init']].round(2)
summary[['loss', 'loss_init']] = summary[['loss', 'loss_init']].round(2)
summary[['n_options', 'n_returns']] = summary[['n_options', 'n_returns']].astype(int)
summary['n_samples'] = grouped.size().values  # Count samples per group

print("=" * 80)
print("MODEL METRICS BY SECTOR AND PERIOD")
print("=" * 80)
print(summary.to_string(index=False))

summary.to_excel("analysis.xlsx", index=False)
