#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

# -------------------------------------------------
# PROJECT FOLDERS
# -------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent

PROCESSED_DIR = BASE_DIR / "2_Parsed_Store" / "3_Processed"
OUTPUT_DIR = BASE_DIR / "2_Parsed_Store" / "4_Daily_Stats" / "4_Signal"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main() -> None:
    processed_files = sorted(PROCESSED_DIR.glob("*_processed.csv"))

    print("\n----------------------------------------")
    print(f"Processed files found : {len(processed_files)}")
    print(f"Output directory      : {OUTPUT_DIR}")
    print("----------------------------------------\n")

    if not processed_files:
        print(f"No processed files found in: {PROCESSED_DIR}")
        return

    required_cols = [
        "date",
        "location_name",
        "device_location",
        "ipei",
        "min_signal_level",
        "mean_signal_level",
        "max_signal_level",
    ]

    for file_number, processed_file in enumerate(processed_files, start=1):
        print(f"[{file_number}/{len(processed_files)}] Processing: {processed_file.name}")

        df = pd.read_csv(processed_file)

        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            print(f" -> Skipping; missing columns: {missing}")
            continue

        df = df[required_cols].copy()

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["ipei"] = df["ipei"].astype(str).str.strip()
        df["location_name"] = df["location_name"].astype(str).str.strip()
        df["device_location"] = df["device_location"].astype(str).str.strip()

        for col in ["min_signal_level", "mean_signal_level", "max_signal_level"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["date", "ipei", "mean_signal_level"])

        if df.empty:
            print(" -> No valid signal rows, skipping")
            continue

        stats_df = (
            df.groupby(
                ["date", "location_name", "device_location", "ipei"],
                dropna=False,
            )
            .agg(
                min_signal_level=("min_signal_level", "min"),
                mean_signal_level=("mean_signal_level", "mean"),
                max_signal_level=("max_signal_level", "max"),
                signal_sample_count=("mean_signal_level", "count"),
            )
            .reset_index()
        )

        numeric_cols = stats_df.select_dtypes(include="number").columns
        stats_df[numeric_cols] = stats_df[numeric_cols].round(2)

        stats_df = stats_df.sort_values(
            by=["date", "location_name", "device_location", "ipei"]
        )

        output_name = processed_file.name.replace("_processed.csv", "_signal_stats.csv")
        output_path = OUTPUT_DIR / output_name

        stats_df.to_csv(output_path, index=False)

        print(f" -> Output written: {output_path.name}")
        print(f" -> Signal rows written: {len(stats_df)}")

    print("\nDone.")
    print(f"Daily signal stats saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()