### Pull per-IPEI data into a per-store database.
#
#####################################

#!/usr/bin/env python3

import re
import pandas as pd
from pathlib import Path

# -------------------------------------------------
# PROJECT FOLDERS
# -------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent

PARSED_DIR = BASE_DIR / "1_Parsed_IPEI"
INDEX_DIR = BASE_DIR / "0_Index"
OUTPUT_DIR = BASE_DIR / "2_Parsed_Store" / "1_Raw"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = INDEX_DIR / "index.csv"

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def safe_filename(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name if name else "Unknown_Store"


def safe_time_for_filename(value) -> str:
    dt = pd.to_datetime(value, errors="coerce")

    if pd.isna(dt):
        return "unknown_time"

    return dt.strftime("%Y-%m-%d_%H-%M-%S")


def load_index() -> pd.DataFrame:
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"0_Index file not found: {INDEX_FILE}")

    index_df = pd.read_csv(INDEX_FILE)

    required = [
        "device_id",
        "ipei",
        "device_location",
        "location_name",
    ]

    missing = [col for col in required if col not in index_df.columns]

    if missing:
        raise ValueError(f"0_Index file missing columns: {missing}")

    index_df["ipei"] = index_df["ipei"].astype(str).str.strip()

    return index_df


def extract_datetime(col_name: str):
    # Match leading timestamp in column name
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", col_name)
    if match:
        return pd.to_datetime(match.group(1), errors="coerce")
    return pd.NaT

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main() -> None:
    index_df = load_index()

    store_folders = sorted([
        folder for folder in PARSED_DIR.iterdir()
        if folder.is_dir()
    ])

    print("\n----------------------------------------")
    print(f"Stores found: {len(store_folders)}")
    print("----------------------------------------\n")

    if not store_folders:
        print(f"No store folders found in: {PARSED_DIR}")
        return

    for store_number, store_folder in enumerate(store_folders, start=1):
        print(f"[{store_number}/{len(store_folders)}] Processing store: {store_folder.name}")

        ipei_files = sorted(store_folder.glob("*.csv"))

        if not ipei_files:
            print(" -> No IPEI files found, skipping")
            continue

        store_rows = []
        all_snapshot_times = []

        for ipei_file in ipei_files:
            ipei = ipei_file.stem.strip()

            df = pd.read_csv(ipei_file)

            required = [
                "snapshot_time",
                "timestamp",
                "status",
                "last_active",
                "battery_level",
                "signal_level",
            ]
            missing = [col for col in required if col not in df.columns]

            if missing:
                print(f" -> Skipping {ipei_file.name}; missing columns: {missing}")
                continue

            df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce")
            df = df.dropna(subset=["snapshot_time"])
            df = df.sort_values(by="snapshot_time")

            if df.empty:
                continue

            all_snapshot_times.extend(df["snapshot_time"].tolist())

            index_match = index_df[index_df["ipei"] == ipei]

            if not index_match.empty:
                index_row = index_match.iloc[0]
                location_name = index_row.get("location_name", store_folder.name)
                device_id = index_row.get("device_id", "")
                device_location = index_row.get("device_location", "")
            else:
                location_name = store_folder.name
                device_id = ""
                device_location = ""

            output_row = {
                "location_name": location_name,
                "device_id": device_id,
                "ipei": ipei,
                "device_location": device_location,
            }

            for _, row in df.iterrows():
                snapshot_label = row["snapshot_time"].strftime("%Y-%m-%d %H:%M:%S")

                #output_row[f"{snapshot_label}_snapshot_time"] = snapshot_label
                output_row[f"{snapshot_label}_timestamp"] = row.get("timestamp")
                output_row[f"{snapshot_label}_status"] = row.get("status")
                output_row[f"{snapshot_label}_last_active"] = row.get("last_active")
                output_row[f"{snapshot_label}_battery_level"] = row.get("battery_level")
                output_row[f"{snapshot_label}_signal_level"] = row.get("signal_level")

            store_rows.append(output_row)

        if not store_rows:
            print(" -> No valid IPEI data found, skipping")
            continue

        store_df = pd.DataFrame(store_rows)

        fixed_cols = [
            "location_name",
            "device_id",
            "ipei",
            "device_location",
        ]

        data_cols = [
            col for col in store_df.columns
            if col not in fixed_cols
        ]

        # Sort by extracted datetime, then by column name for stability
        data_cols = sorted(
            data_cols,
            key=lambda col: (extract_datetime(col), col)
        )

        store_df = store_df[fixed_cols + data_cols]

        start_time = min(all_snapshot_times)
        end_time = max(all_snapshot_times)

        output_name = (
            f"{safe_filename(store_folder.name)}_"
            f"{safe_time_for_filename(start_time)}_"
            f"{safe_time_for_filename(end_time)}_raw.csv"
        )

        output_path = OUTPUT_DIR / output_name
        store_df.to_csv(output_path, index=False)

        print(f" -> Output written: {output_path.name}")
        print(f" -> Devices included: {len(store_df)}")

    print("\nDone.")
    print(f"1_Raw store files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()