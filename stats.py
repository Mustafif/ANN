import pandas as pd
import numpy as np

df = pd.read_csv("calibration_report_duan.csv")

# Configurable: which direction indicates improvement?
# For AIC/BIC: lower is better (< 0 is improvement)
# For loss: higher is better
metric_configs = {
    "AIC":   {"col": "aic",      "init_col": "aic_init",    "lower_is_better": True},
    "BIC":   {"col": "bic",      "init_col": "bic_init",    "lower_is_better": True},
    "Loss":  {"col": "loss",     "init_col": "loss_init",   "lower_is_better": False},
}

ok_mask = df["status"].eq("ok")
total_ok = ok_mask.sum()
total_failed = len(df) - total_ok

print(f"Total rows        : {len(df)}")
print(f"Status = ok       : {total_ok}")
print(f"Status = failed   : {total_failed}")
print("=" * 75)

results = {}

for name, cfg in metric_configs.items():
    col, init_col = cfg["col"], cfg["init_col"]
    lower_is_better = cfg["lower_is_better"]

    sub = df.loc[ok_mask, [col, init_col]].copy()

    both_present = sub[col].notna() & sub[init_col].notna()
    calib_only   = sub[col].notna() & sub[init_col].isna()
    init_only    = sub[col].isna() & sub[init_col].notna()
    both_null    = sub[col].isna() & sub[init_col].isna()

    # Direction comparison
    diff = sub.loc[both_present, col] - sub.loc[both_present, init_col]

    if lower_is_better:
        improved  = (diff < 0).sum()    # calibrated < init
        regressed = (diff > 0).sum()   # calibrated > init
    else:
        improved  = (diff > 0).sum()    # calibrated > init (if higher is better)
        regressed = (diff < 0).sum()   # calibrated < init

    same = (diff == 0).sum()

    results[name] = {
        "both_present": int(both_present.sum()),
        "calib_only":   int(calib_only.sum()),
        "init_only":    int(init_only.sum()),
        "both_null":    int(both_null.sum()),
        "improved":     int(improved),
        "regressed":    int(regressed),
        "same":         int(same),
    }

    improvement_symbol = "<" if lower_is_better else ">"
    print(f"\n{'─' * 75}")
    print(f"  Metric: {name}  (calibrated {improvement_symbol} init = improvement)")
    print(f"{'─' * 75}")
    print(f"  Both present (comparable) : {both_present.sum():>6}")
    print(f"    ↳ Calibrated {improvement_symbol} init     : {improved:>6}  (improved)")
    print(f"    ↳ Calibrated {'<' if not lower_is_better else '>'} init     : {regressed:>6}  (regressed)")
    print(f"    ↳ Calibrated == init    : {same:>6}  (unchanged)")
    print(f"  Calib-only (init null)    : {calib_only.sum():>6}")
    print(f"  Init-only (calib null)    : {init_only.sum():>6}")
    print(f"  Both null                 : {both_null.sum():>6}")

results = pd.DataFrame(results)
results.to_csv("results.csv", index=False)
print("\n" + "=" * 75)
print("SUMMARY TABLE")
print("=" * 75)
print(results.T.to_string())
