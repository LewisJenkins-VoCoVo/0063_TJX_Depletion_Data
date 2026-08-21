#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

# -------------------------------------------------
# PROJECT FOLDERS
# -------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent

PROCESSED_DIR = BASE_DIR / "2_Parsed_Store" / "3_Processed"
OUTPUT_DIR = BASE_DIR / "2_Parsed_Store" / "4_Daily_Stats" / "1_Depletion"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main() -> None:
    processed_files = sorted(PROCESSED_DIR.glob("*_processed.csv"))

    print("\n----------------------------------------")
    print(f"Processed store files found : {len(processed_files)}")
    print(f"Output directory            : {OUTPUT_DIR}")
    print("----------------------------------------\n")

    if not processed_files:
        print(f"No processed files found in: {PROCESSED_DIR}")
        return

    for file_number, processed_file in enumerate(processed_files, start=1):
        print(f"[{file_number}/{len(processed_files)}] Processing: {processed_file.name}")

        df = pd.read_csv(processed_file)

        if df.empty:
            print(" -> Empty file, skipping")
            continue

        required_cols = [
            "date",
            "location_name",
            "device_location",
            "ipei",
            "battery_drop_percent",
            "duration_hours",
            "discharge_rate_percent_per_day",
        ]

        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            print(f" -> Skipping; missing columns: {missing}")
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["battery_drop_percent"] = pd.to_numeric(df["battery_drop_percent"], errors="coerce")
        df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce")
        df["discharge_rate_percent_per_day"] = pd.to_numeric(
            df["discharge_rate_percent_per_day"],
            errors="coerce",
        )

        df = df.dropna(subset=[
            "date",
            "battery_drop_percent",
            "duration_hours",
            "discharge_rate_percent_per_day",
        ])

        df = df[df["duration_hours"] > 0]

        if df.empty:
            print(" -> No valid depletion rows, skipping")
            continue

        stats_df = (
            df.groupby(
                [
                    "date",
                    "location_name",
                    "device_location",
                    "ipei",
                ],
                dropna=False,
            )
            .agg(
                total_battery_drop_percent=(
                    "battery_drop_percent",
                    "sum",
                ),
                total_discharge_duration_hours=(
                    "duration_hours",
                    "sum",
                ),
                min_discharge_rate_percent_per_day=(
                    "discharge_rate_percent_per_day",
                    "min",
                ),
                mean_discharge_rate_percent_per_day=(
                    "discharge_rate_percent_per_day",
                    "mean",
                ),
                max_discharge_rate_percent_per_day=(
                    "discharge_rate_percent_per_day",
                    "max",
                ),
                cycle_count=(
                    "discharge_rate_percent_per_day",
                    "count",
                ),
            )
            .reset_index()
        )

        stats_df["weighted_discharge_rate_percent_per_day"] = (
            stats_df["total_battery_drop_percent"] /
            (stats_df["total_discharge_duration_hours"] / 24)
        )

        numeric_cols = stats_df.select_dtypes(include="number").columns
        stats_df[numeric_cols] = stats_df[numeric_cols].round(2)

        stats_df = stats_df.sort_values(
            by=[
                "date",
                "location_name",
                "device_location",
                "ipei",
            ]
        )

        output_name = processed_file.name.replace("_processed.csv", "_daily_stats.csv")
        output_path = OUTPUT_DIR / output_name

        stats_df.to_csv(output_path, index=False)

        print(f" -> Output written: {output_path.name}")
        print(f" -> Daily stats rows: {len(stats_df)}")

    print("\nDone.")
    print(f"Daily depletion stats saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()