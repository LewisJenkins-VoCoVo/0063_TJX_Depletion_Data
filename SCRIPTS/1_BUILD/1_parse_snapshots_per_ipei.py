### Parse each snapshot into inividual datasets per IPEI, structured within each respective store.
#
# Outputs:
# 0_Index -> CSV index with Device ID, IPEI, device_location, location_name.
#
# Folder Structure
# | # Location_Name
# |
# |- IPEI 1 (.csv)
# |- IPEI 2 (.csv)
# | ...
# |- IPEI n (.csv)
#
# Each IPEI will contain the snapshot time, battery_level, signal_level, timestamp and last_active time.
#
#####################################

#!/usr/bin/env python3

import re
import shutil
import pandas as pd
from pathlib import Path

# -------------------------------------------------
# PROJECT FOLDERS
# -------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent

SNAPSHOT_DIR = BASE_DIR / "SOURCE_DATA" / "Snapshots"
PARSED_DIR = BASE_DIR / "1_Parsed_IPEI"
INDEX_DIR = BASE_DIR / "0_Index"

INDEX_FILE = INDEX_DIR / "index.csv"
PROCESSED_FILE = INDEX_DIR / "processed_snapshots.csv"

FORCE_REBUILD = False  # Set True only when intentionally rebuilding everything
PURGE_PARSED_ON_REBUILD = False  # Set True if you want FORCE_REBUILD to wipe 1_Parsed_IPEI first

# -------------------------------------------------
# LOCATION NAME CORRECTIONS
# -------------------------------------------------
LOCATION_NAME_FIXES = {
    "STUKTK_408(47/02/05)_Covent Garden": "ST_UKTK_408(47_02_05)_Covent Garden",
    "STUKTK_408(47_02_05)_Covent Garden": "ST_UKTK_408(47_02_05)_Covent Garden",
    "ST_UKTKM_190_(25_3_4)Charlton": "ST_UKTK_190(25_03_04)_Charlton",
}

# -------------------------------------------------
# EXPECTED COLUMNS
# -------------------------------------------------
INDEX_COLUMNS = [
    "device_id",
    "ipei",
    "device_location",
    "location_name",
]

IPEI_COLUMNS = [
    "snapshot_time",
    "battery_level",
    "signal_level",
    "status",
    "timestamp",
    "last_active",
]

REQUIRED_COLUMNS = INDEX_COLUMNS + [
    "battery_level",
    "signal_level",
    "status",
    "timestamp",
    "last_active",
]


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def normalise_location_name(name: str) -> str:
    name = str(name).strip()

    if name in LOCATION_NAME_FIXES:
        return LOCATION_NAME_FIXES[name]

    return name

def safe_folder_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", " ", name)
    return name if name else "Unknown_Location"


def get_snapshot_time(csv_path: Path) -> str:
    match = re.search(
        r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})",
        csv_path.stem,
    )

    if match:
        date_part = match.group(1)
        time_part = match.group(2).replace("-", ":")
        return f"{date_part} {time_part}"

    return csv_path.stem


def load_existing_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_processed_snapshots() -> set:
    if PROCESSED_FILE.exists():
        df = pd.read_csv(PROCESSED_FILE)
        if "snapshot_file" in df.columns:
            return set(df["snapshot_file"].astype(str))
    return set()


def save_processed_snapshots(snapshot_files: list[Path]) -> None:
    existing = load_existing_csv(PROCESSED_FILE)

    now = pd.Timestamp.now().isoformat()

    new_rows = pd.DataFrame([
        {
            "snapshot_file": snapshot_file.name,
            "processed_time": now,
        }
        for snapshot_file in snapshot_files
    ])

    updated = pd.concat([existing, new_rows], ignore_index=True)
    updated = updated.drop_duplicates(subset=["snapshot_file"], keep="last")
    updated = updated.sort_values(by="snapshot_file")
    updated.to_csv(PROCESSED_FILE, index=False)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main() -> None:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    all_snapshot_files = sorted(SNAPSHOT_DIR.glob("*.csv"))
    processed_snapshots = load_processed_snapshots()

    if FORCE_REBUILD:
        print("\nWARNING: Force rebuild will clear the processed snapshot log.")
        print("This will cause ALL snapshots to be reprocessed.\n")

        if PURGE_PARSED_ON_REBUILD:
            print("PURGE_PARSED_ON_REBUILD is also enabled.")
            print("This will delete the existing 1_Parsed_IPEI folder.\n")

        confirm = input("Type 'y' to continue: ").strip().lower()

        if confirm != "y":
            print("Aborting rebuild. No changes made.\n")
            return

        print("\nConfirmed. Clearing processed snapshot log...\n")
        PROCESSED_FILE.unlink(missing_ok=True)
        processed_snapshots = set()

        if PURGE_PARSED_ON_REBUILD and PARSED_DIR.exists():
            print("Deleting existing 1_Parsed_IPEI folder...\n")
            shutil.rmtree(PARSED_DIR)
            PARSED_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_files = [
        f for f in all_snapshot_files
        if f.name not in processed_snapshots
    ]

    print("\n----------------------------------------")
    print(f"Total snapshot files found : {len(all_snapshot_files)}")
    print(f"Already processed          : {len(all_snapshot_files) - len(snapshot_files)}")
    print(f"To process                 : {len(snapshot_files)}")
    print("----------------------------------------\n")

    if not snapshot_files:
        print("No new snapshot CSV files to parse.")
        return

    parsed_rows = []
    index_rows = []
    successfully_processed_files = []

    # -------------------------------------------------
    # LOAD ALL NEW SNAPSHOTS INTO MEMORY
    # -------------------------------------------------
    for file_number, snapshot_file in enumerate(snapshot_files, start=1):
        print(f"[{file_number}/{len(snapshot_files)}] Reading: {snapshot_file.name}")

        df = pd.read_csv(snapshot_file)

        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            print(f"Skipping {snapshot_file.name}; missing columns: {missing}")
            continue

        snapshot_time = get_snapshot_time(snapshot_file)

        df["snapshot_time"] = snapshot_time
        df["location_name"] = df["location_name"].apply(normalise_location_name)
        df["safe_location_name"] = df["location_name"].apply(safe_folder_name)

        parsed_rows.append(df[
            [
                "safe_location_name",
                "location_name",
                "ipei",
                "snapshot_time",
                "battery_level",
                "signal_level",
                "status",
                "timestamp",
                "last_active",
            ]
        ])

        index_rows.append(df[INDEX_COLUMNS])

        successfully_processed_files.append(snapshot_file)

        print(f" -> {len(df)} records staged")

    if not parsed_rows:
        print("No valid snapshot data was found.")
        return

    parsed_df = pd.concat(parsed_rows, ignore_index=True)
    index_df = pd.concat(index_rows, ignore_index=True)

    parsed_df["ipei"] = parsed_df["ipei"].astype(str).str.strip()
    parsed_df["timestamp"] = pd.to_datetime(parsed_df["timestamp"], errors="coerce")

    # -------------------------------------------------
    # WRITE EACH IPEI FILE ONCE
    # -------------------------------------------------
    grouped = parsed_df.groupby(["safe_location_name", "ipei"], dropna=False)

    print("\nWriting per-IPEI files...")

    for file_number, ((safe_location_name, ipei), group) in enumerate(grouped, start=1):
        location_folder = PARSED_DIR / safe_location_name
        location_folder.mkdir(parents=True, exist_ok=True)

        ipei_file = location_folder / f"{ipei}.csv"

        new_data = group[IPEI_COLUMNS].copy()

        existing = load_existing_csv(ipei_file)

        updated = pd.concat([existing, new_data], ignore_index=True)

        updated["timestamp"] = pd.to_datetime(updated["timestamp"], errors="coerce")
        updated["snapshot_time"] = pd.to_datetime(updated["snapshot_time"], errors="coerce")

        updated = updated.drop_duplicates(
            subset=["snapshot_time"],
            keep="last",
        )

        updated = updated.sort_values(by="snapshot_time")

        updated.to_csv(ipei_file, index=False)

        if file_number % 50 == 0:
            print(f" -> {file_number}/{len(grouped)} IPEI files written")

    # -------------------------------------------------
    # WRITE INDEX
    # -------------------------------------------------
    existing_index = load_existing_csv(INDEX_FILE)

    final_index = pd.concat(
        [existing_index, index_df],
        ignore_index=True,
    )

    final_index = final_index.drop_duplicates(
        subset=["device_id", "ipei", "device_location", "location_name"],
        keep="last",
    )

    final_index = final_index.sort_values(
        by=["location_name", "device_location", "ipei"]
    )

    final_index.to_csv(INDEX_FILE, index=False)

    # -------------------------------------------------
    # UPDATE PROCESSED SNAPSHOT LOG
    # -------------------------------------------------
    save_processed_snapshots(successfully_processed_files)

    print("\n----------------------------------------")
    print("Processing complete")
    print(f"Snapshots processed : {len(successfully_processed_files)}")
    print(f"IPEI files updated  : {len(grouped)}")
    print(f"Total index entries : {len(final_index)}")
    print("----------------------------------------")

    print(f"\nParsed IPEI files saved to: {PARSED_DIR}")
    print(f"0_Index saved to: {INDEX_FILE}")
    print(f"Processed snapshot log saved to: {PROCESSED_FILE}")


if __name__ == "__main__":
    main()