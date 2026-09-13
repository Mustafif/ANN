import csv
import glob
import os

ROOT_DIR = "."  # change to your root directory


def count_sigma_stats(path):
    neg_count = 0
    gt1_count = 0
    total = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "sigma" not in (reader.fieldnames or []):
            return None, None, None
        for row in reader:
            try:
                val = float(row["sigma"])
            except (ValueError, TypeError):
                continue
            total += 1
            if val < 0:
                neg_count += 1
            if val > 1:
                gt1_count += 1
    return neg_count, gt1_count, total


def is_empty(path):
    if os.path.getsize(path) == 0:
        return True
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return len(rows) <= 1  # header only or no rows


def main():
    results = []
    period_dirs = glob.glob(os.path.join(ROOT_DIR, "*", "Period[123]"))
    for period_dir in sorted(period_dirs):
        folder = os.path.basename(os.path.dirname(period_dir))
        period = os.path.basename(period_dir)
        dataset_path = os.path.join(period_dir, "dataset.csv")
        prices_path = os.path.join(period_dir, "asset_prices.csv")

        neg_sigma = None
        gt1_sigma = None
        sigma_total = None
        if os.path.isfile(dataset_path):
            neg_sigma, gt1_sigma, sigma_total = count_sigma_stats(dataset_path)

        prices_empty = None
        if os.path.isfile(prices_path):
            prices_empty = is_empty(prices_path)

        results.append({
            "folder": folder,
            "period": period,
            "dataset_exists": os.path.isfile(dataset_path),
            "neg_sigma_count": neg_sigma,
            "gt1_sigma_count": gt1_sigma,
            "sigma_total": sigma_total,
            "asset_prices_exists": os.path.isfile(prices_path),
            "asset_prices_empty": prices_empty,
        })

    print(f"{'Folder':<25}{'Period':<10}{'sigma<0':<10}{'pct':<10}{'sigma>1':<10}{'pct':<10}{'asset_prices empty':<20}")
    for r in results:
        neg = r["neg_sigma_count"] if r["neg_sigma_count"] is not None else "N/A"
        gt1 = r["gt1_sigma_count"] if r["gt1_sigma_count"] is not None else "N/A"

        if r["neg_sigma_count"] is not None and r["sigma_total"]:
            neg_pct = f"{(r['neg_sigma_count'] / r['sigma_total']) * 100:.2f}%"
        else:
            neg_pct = "N/A"

        if r["gt1_sigma_count"] is not None and r["sigma_total"]:
            gt1_pct = f"{(r['gt1_sigma_count'] / r['sigma_total']) * 100:.2f}%"
        else:
            gt1_pct = "N/A"

        empty = r["asset_prices_empty"] if r["asset_prices_empty"] is not None else "N/A"
        print(f"{r['folder']:<25}{r['period']:<10}{str(neg):<10}{neg_pct:<10}{str(gt1):<10}{gt1_pct:<10}{str(empty):<20}")

    total_neg = sum(r["neg_sigma_count"] or 0 for r in results if r["neg_sigma_count"] is not None)
    total_gt1 = sum(r["gt1_sigma_count"] or 0 for r in results if r["gt1_sigma_count"] is not None)
    total_rows = sum(r["sigma_total"] or 0 for r in results if r["sigma_total"] is not None)
    total_neg_pct = f"{(total_neg / total_rows) * 100:.2f}%" if total_rows else "N/A"
    total_gt1_pct = f"{(total_gt1 / total_rows) * 100:.2f}%" if total_rows else "N/A"
    total_empty = sum(1 for r in results if r["asset_prices_empty"])

    print()
    print(f"Total (Period, folder) entries scanned: {len(results)}")
    print(f"Total sigma<0 rows across all datasets: {total_neg} ({total_neg_pct} of {total_rows} valid sigma rows)")
    print(f"Total sigma>1 rows across all datasets: {total_gt1} ({total_gt1_pct} of {total_rows} valid sigma rows)")
    print(f"Total empty asset_prices.csv files: {total_empty}")


if __name__ == "__main__":
    main()
