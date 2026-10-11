#!/usr/bin/env python3
"""Compare IV calibration reports. Edit SETTINGS below, then run in your editor.
Requires pandas and numpy. Exports four CSV reports to OUTPUT_DIR.
Positive reduction means improvement. MRE is displayed as a percentage.
Company summaries use option-count weighting over paired valid periods.
Filtered/unfiltered comparisons evaluate different contract samples.
"""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd

# ========================= SETTINGS =========================
UNFILTERED_FILE = Path("finance2_stats/Finance2_HN.csv")
FILTERED_FILE = Path("finance2_stats_filter/Finance_HN_filter.csv")
OUTPUT_DIR = Path("calibration_analysis")

# None means all companies / periods. Examples: ["JPM", "MA"], ["Period3"]
COMPANIES = None
PERIODS = None

IMPROVEMENT_THRESHOLD = 20.0  # Required percentage reduction in BOTH MAE/MSE
CHANGE_TOLERANCE = 0.1       # Changes within +/- this percent are negligible
MIN_OPTIONS = 100           # Flag smaller samples; do not exclude them
# ============================================================

KEYS = ['ticker', 'period']
METRICS = ['mae', 'mse']


def reduction(initial, final):
    if not np.isfinite(initial) or not np.isfinite(final) or initial <= 0:
        return np.nan
    return 100 * (initial - final) / initial


def classify(a, b, threshold, tolerance):
    if not np.isfinite(a) or not np.isfinite(b):
        return 'unavailable'
    if min(a, b) >= threshold and min(a, b) > tolerance:
        return 'improved'
    if max(a, b) < -tolerance:
        return 'deteriorated'
    if min(a, b) < -tolerance:
        return 'mixed'
    if max(abs(a), abs(b)) <= tolerance:
        return 'no_change'
    return 'limited_improvement'


def load(path, dataset, args):
    d = pd.read_csv(path)
    required = KEYS + ['n_options'] + [f'mean_iv_{m}{suffix}' for m in METRICS for suffix in ['', '_init']]
    missing = set(required) - set(d.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    if d.duplicated(KEYS).any():
        raise ValueError(f'{path}: duplicate ticker/period rows; select one strategy first.')
    for c in d.columns:
        if c not in KEYS + ['strategy', 'status']:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    d['ticker'] = d.ticker.astype(str).str.upper()
    if d.duplicated(KEYS).any():
        raise ValueError(f'{path}: duplicate keys after normalizing ticker case.')
    d['dataset'] = dataset
    records, buckets = [], []
    for _, r in d.iterrows():
        record = {'dataset': dataset, 'ticker': r.ticker, 'period': r.period,
                  'n_options': r.n_options, 'status': r.get('status', np.nan)}
        for m in METRICS:
            initial, final = r[f'mean_iv_{m}_init'], r[f'mean_iv_{m}']
            if (pd.notna(initial) and initial < 0) or (pd.notna(final) and final < 0):
                raise ValueError(f'{path}: negative {m} for {r.ticker}/{r.period}')
            record[f'{m}_init'] = initial
            record[f'{m}_final'] = final
            record[f'{m}_reduction_pct'] = reduction(initial, final)
        record['classification'] = classify(record['mae_reduction_pct'], record['mse_reduction_pct'], args.threshold, args.tolerance)
        record['small_sample'] = bool(pd.notna(r.n_options) and r.n_options < args.min_options)
        for c in [c for c in d if c.startswith('mre_')]:
            bucket = c[4:]
            count, init_count = r.get('n_' + bucket, np.nan), r.get('init_n_' + bucket, np.nan)
            initial, final = r.get('init_' + c, np.nan), r[c]
            comparable = pd.notna(count) and count > 0 and count == init_count and np.isfinite(initial) and np.isfinite(final)
            buckets.append({'dataset': dataset, 'ticker': r.ticker, 'period': r.period,
                            'dimension': 'maturity' if bucket.startswith('T') else 'moneyness',
                            'bucket': bucket, 'n_init': init_count, 'n_final': count,
                            'mre_init_pct': initial * 100, 'mre_final_pct': final * 100,
                            'mre_reduction_pct': reduction(initial, final) if comparable else np.nan,
                            'change_pp': (final - initial) * 100 if comparable else np.nan,
                            'comparable': comparable,
                            'small_sample': bool(pd.notna(count) and count < args.min_options)})
        records.append(record)
    return pd.DataFrame(records), pd.DataFrame(buckets)


def company_summary(rows, args):
    result = []
    for (dataset, ticker), g in rows.groupby(['dataset', 'ticker']):
        cols = ['mae_init', 'mae_final', 'mse_init', 'mse_final', 'n_options']
        valid = g[cols].apply(lambda s: np.isfinite(s)).all(axis=1) & g.n_options.gt(0)
        v = g[valid]
        item = {'dataset': dataset, 'ticker': ticker, 'periods_available': len(g),
                'periods_used': len(v), 'periods_used_names': ','.join(v.period),
                'n_options_used': v.n_options.sum(), 'n_options_reported': g.n_options.sum(min_count=1),
                'improved_periods': int((g.classification == 'improved').sum()),
                'deteriorated_periods': int((g.classification == 'deteriorated').sum())}
        for m in METRICS:
            for suffix in ['init', 'final']:
                item[f'{m}_{suffix}'] = np.average(v[f'{m}_{suffix}'], weights=v.n_options) if len(v) else np.nan
            item[f'{m}_reduction_pct'] = reduction(item[f'{m}_init'], item[f'{m}_final'])
        item['classification'] = classify(item['mae_reduction_pct'], item['mse_reduction_pct'], args.threshold, args.tolerance)
        item['small_sample'] = item['n_options_used'] < args.min_options
        result.append(item)
    return pd.DataFrame(result)


def comparison(rows):
    a = rows[rows.dataset == 'unfiltered'].drop(columns='dataset')
    b = rows[rows.dataset == 'filtered'].drop(columns='dataset')
    joined = a.merge(b, on=KEYS, how='outer', suffixes=('_unfiltered', '_filtered'), indicator='coverage')
    joined['retained_options_pct'] = 100 * joined.n_options_filtered / joined.n_options_unfiltered.replace(0, np.nan)
    for m in METRICS:
        joined[f'{m}_filtered_minus_unfiltered'] = joined[f'{m}_final_filtered'] - joined[f'{m}_final_unfiltered']
        joined[f'{m}_filter_reduction_pct'] = [reduction(a, b) for a, b in zip(joined[f'{m}_final_unfiltered'], joined[f'{m}_final_filtered'])]
    return joined


def display(rows, buckets, args, companies=None, periods=None):
    selected = rows.copy()
    if companies:
        selected = selected[selected.ticker.isin([t.upper() for t in companies])]
    if periods:
        selected = selected[selected.period.isin(periods)]
    if selected.empty:
        print('No matching companies/periods.'); return
    cols = ['dataset', 'ticker', 'period', 'n_options', 'mae_init', 'mae_final', 'mae_reduction_pct', 'mse_init', 'mse_final', 'mse_reduction_pct', 'classification', 'small_sample']
    print('\nCOMPANY / PERIOD ERRORS\n' + selected[cols].round(4).to_string(index=False))
    pooled = company_summary(selected, args)
    print('\nPOOLED COMPANY ERRORS (selected periods; paired valid rows)\n' + pooled.round(4).to_string(index=False))
    if companies:
        keys = selected[['dataset'] + KEYS]
        details = buckets.merge(keys, on=['dataset'] + KEYS)
        print('\nBUCKET MRE (percent; positive change_pp means worsening)\n' + details.round(3).to_string(index=False))


def analyze(unfiltered_file=UNFILTERED_FILE, filtered_file=FILTERED_FILE,
            companies=COMPANIES, periods=PERIODS, output_dir=OUTPUT_DIR,
            threshold=IMPROVEMENT_THRESHOLD, tolerance=CHANGE_TOLERANCE,
            min_options=MIN_OPTIONS):
    """Return pandas tables for further exploration; also print and export results.

    Example in a notebook or Python console:
        results = analyze(companies=["JPM", "MA"], periods=["Period3"])
        results["row_summary"]

    Exports always contain all companies/periods. companies/periods select the
    printed views only, so you can inspect other rows in the returned tables.
    """
    if threshold < 0 or tolerance < 0 or min_options < 1:
        raise ValueError("Threshold/tolerance must be nonnegative; min_options must be positive.")
    args = SimpleNamespace(threshold=threshold, tolerance=tolerance, min_options=min_options)
    loaded = [load(Path(unfiltered_file), 'unfiltered', args),
              load(Path(filtered_file), 'filtered', args)]
    rows = pd.concat([a for a, _ in loaded], ignore_index=True)
    buckets = pd.concat([b for _, b in loaded], ignore_index=True)
    tables = {'row_summary': rows, 'company_summary': company_summary(rows, args),
              'bucket_mre': buckets, 'filtered_vs_unfiltered': comparison(rows)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / (name + '.csv'), index=False)
    display(rows, buckets, args, companies, periods)
    print('\nClassification counts:\n' + pd.crosstab(rows.dataset, rows.classification).to_string())
    print(f'\nSaved four CSV reports to {output_dir.resolve()}')
    print('Filtering comparisons use different option samples and cannot isolate a causal filtering effect.')
    return tables


if __name__ == '__main__':
    results = analyze()
