#!/usr/bin/env python3

import re
import pandas as pd
from pathlib import Path

# -------------------------------------------------
# PROJECT FOLDERS
# -------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent.parent

DEPLETION_DIR = BASE_DIR / "2_Parsed_Store" / "4_Daily_Stats" / "1_Depletion"
SIGNAL_DIR = BASE_DIR / "2_Parsed_Store" / "4_Daily_Stats" / "4_Signal"

COMBINED_SIGNAL_DIR = BASE_DIR / "2_Parsed_Store" / "4_Daily_Stats" / "5_Signal_Combined"
ANALYSIS_DIR = BASE_DIR / "3_Analysis_Datasets"

COMBINED_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MASTER_ANALYSIS_FILE = ANALYSIS_DIR / "all_stores_signal_analysis.csv"


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def get_store_key(path: Path) -> str:
    name = path.stem
    match = re.search(r"(.+?)_\d{4}-\d{2}-\d{2}", name)

    if match:
        return match.group(1)

    return name


def normalise_key(value: str) -> str:
    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["ipei"] = df["ipei"].astype(str).str.strip()
    df["location_name"] = df["location_name"].astype(str).str.strip()
    df["device_location"] = df["device_location"].astype(str).str.strip()

    return df


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main() -> None:
    depletion_files = sorted(DEPLETION_DIR.glob("*.csv"))
    signal_files = sorted(SIGNAL_DIR.glob("*.csv"))

    depletion_map = {
        normalise_key(get_store_key(path)): path
        for path in depletion_files
    }

    signal_map = {
        normalise_key(get_store_key(path)): path
        for path in signal_files
    }

    common_keys = sorted(set(depletion_map) & set(signal_map))

    print("\n----------------------------------------")
    print(f"Depletion files found : {len(depletion_files)}")
    print(f"Signal files found    : {len(signal_files)}")
    print(f"Matched stores        : {len(common_keys)}")
    print(f"Per-store output dir  : {COMBINED_SIGNAL_DIR}")
    print(f"Master analysis file  : {MASTER_ANALYSIS_FILE}")
    print("----------------------------------------\n")

    if not common_keys:
        print("No matching depletion/signal store files found.")
        return

    all_merged_rows = []

    required_depletion = [
        "date",
        "location_name",
        "device_location",
        "ipei",
        "min_discharge_rate_percent_per_day",
        "mean_discharge_rate_percent_per_day",
        "max_discharge_rate_percent_per_day",
        "weighted_discharge_rate_percent_per_day",
        "total_battery_drop_percent",
        "total_discharge_duration_hours",
        "cycle_count",
    ]

    required_signal = [
        "date",
        "location_name",
        "device_location",
        "ipei",
        "min_signal_level",
        "mean_signal_level",
        "max_signal_level",
        "signal_sample_count",
    ]

    for file_number, key in enumerate(common_keys, start=1):
        depletion_file = depletion_map[key]
        signal_file = signal_map[key]

        print(f"[{file_number}/{len(common_keys)}] Processing: {depletion_file.name}")

        depletion_df = load_csv(depletion_file)
        signal_df = load_csv(signal_file)

        missing_depletion = [col for col in required_depletion if col not in depletion_df.columns]
        missing_signal = [col for col in required_signal if col not in signal_df.columns]

        if missing_depletion:
            print(f" -> Skipping; depletion missing columns: {missing_depletion}")
            continue

        if missing_signal:
            print(f" -> Skipping; signal missing columns: {missing_signal}")
            continue

        depletion_df = depletion_df[required_depletion]
        signal_df = signal_df[required_signal]

        merged_df = depletion_df.merge(
            signal_df,
            on=["date", "location_name", "device_location", "ipei"],
            how="inner",
        )

        if merged_df.empty:
            print(" -> No matching rows after merge, skipping")
            continue

        numeric_cols = merged_df.select_dtypes(include="number").columns
        merged_df[numeric_cols] = merged_df[numeric_cols].round(2)

        merged_df = merged_df.sort_values(
            by=["date", "location_name", "device_location", "ipei"]
        )

        output_name = depletion_file.name.replace("_daily_stats.csv", "_signal_analysis.csv")
        output_path = COMBINED_SIGNAL_DIR / output_name

        merged_df.to_csv(output_path, index=False)
        all_merged_rows.append(merged_df)

        print(f" -> Output written: {output_path.name}")
        print(f" -> Rows written: {len(merged_df)}")

    if all_merged_rows:
        master_df = pd.concat(all_merged_rows, ignore_index=True)

        master_df = master_df.sort_values(
            by=["date", "location_name", "device_location", "ipei"]
        )

        numeric_cols = master_df.select_dtypes(include="number").columns
        master_df[numeric_cols] = master_df[numeric_cols].round(2)

        master_df.to_csv(MASTER_ANALYSIS_FILE, index=False)

        print(f"\nMaster signal analysis file written: {MASTER_ANALYSIS_FILE}")
        print(f"Total rows written                 : {len(master_df)}")

    print("\nDone.")
    print(f"Per-store signal combined files saved to: {COMBINED_SIGNAL_DIR}")
    print(f"Master signal analysis file saved to: {MASTER_ANALYSIS_FILE}")


if __name__ == "__main__":
    main()