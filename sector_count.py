from pathlib import Path
import pandas as pd

ROOTS = {
    "CD": Path("src2_CD"),
    "Finance": Path("src2_Finance"),
    "Health": Path("src2_Health"),
    "Industrial": Path("src2_Industrial"),
    "IT": Path("src2_IT"),
}

CHUNK_SIZE = 100_000


def count_options():
    records = []

    for sector, root in ROOTS.items():
        total = 0
        successful = 0
        failed = 0

        if not root.is_dir():
            print(f"Missing sector folder: {root}")
            records.append({
                "sector": sector,
                "total_options": None,
                "files_counted": 0,
                "files_failed": 0,
                "status": "missing_root",
            })
            continue

        files = sorted(root.glob("*/Period*/dataset.csv"))

        for path in files:
            file_total = 0

            try:
                # Read only one column to reduce memory usage.
                with pd.read_csv(
                    path,
                    usecols=[0],
                    chunksize=CHUNK_SIZE,
                ) as reader:
                    for chunk in reader:
                        file_total += len(chunk)

            except (OSError, ValueError, pd.errors.ParserError) as exc:
                print(f"Failed {path}: {exc}")
                failed += 1
                continue  # Discard partial counts.

            total += file_total
            successful += 1

        records.append({
            "sector": sector,
            "total_options": total,
            "files_counted": successful,
            "files_failed": failed,
            "status": (
                "no_dataset_files" if not files
                else "partial" if failed
                else "ok"
            ),
        })

    summary = pd.DataFrame(records)
    summary["total_options"] = summary["total_options"].astype("Int64")

    print("\nOPTIONS BY SECTOR")
    print(summary.to_string(index=False))

    grand_total = summary["total_options"].sum()
    print(f"\nTotal options across successfully read files: {grand_total:,}")

    summary.to_csv("sector_option_counts.csv", index=False)
    return summary


if __name__ == "__main__":
    results = count_options()
