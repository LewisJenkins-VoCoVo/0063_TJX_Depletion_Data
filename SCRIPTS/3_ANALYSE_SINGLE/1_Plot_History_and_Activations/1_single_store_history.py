#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import pandas as pd

# -------------------------------------------------
# USER SETTINGS
# -------------------------------------------------
STORE_SEARCH = "ST_UKTK_0786"  # partial match is OK, e.g. "Oxford", "0786", "Covent"

# -------------------------------------------------
# PROJECT FOLDERS
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]

RAW_DIR = BASE_DIR / "2_Parsed_Store" / "1_Raw"
PORTAL_REPORT_DIR = BASE_DIR / "SOURCE_DATA" / "Portal_Data" / "Activation_Reports"
INDEX_FILE = BASE_DIR / "0_Index" / "index.csv"

OUTPUT_DIR = BASE_DIR / "3_Analysis_Datasets" / "1_Single_Store_Analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def normalise_text(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("/", "_")
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def safe_filename(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r'[<>:"/\\|?*]', "_", value)
    value = re.sub(r"\s+", "_", value)
    return value if value else "Unknown_Store"


def extract_store_key_from_raw_filename(path: Path) -> str:
    match = re.search(r"(.+?)_\d{4}-\d{2}-\d{2}", path.stem)
    return match.group(1) if match else path.stem


def extract_store_key_from_portal_filename(path: Path) -> str:
    stem = path.stem

    if " - call point -" in stem:
        return stem.split(" - call point -")[0].strip()

    return stem


def find_matching_file(folder: Path, pattern: str, search: str) -> Path:
    search_key = normalise_text(search)

    matches = []

    for file in sorted(folder.glob(pattern)):
        file_key = normalise_text(file.stem)

        if search_key in file_key:
            matches.append(file)

    if not matches:
        raise FileNotFoundError(f"No matching file found in {folder} for search: {search}")

    if len(matches) > 1:
        print("\nMultiple matches found:")
        for idx, match in enumerate(matches, start=1):
            print(f"{idx}: {match.name}")

        raise ValueError(
            "Search is ambiguous. Make STORE_SEARCH more specific."
        )

    return matches[0]


def load_index() -> pd.DataFrame:
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Index file not found: {INDEX_FILE}")

    index_df = pd.read_csv(INDEX_FILE)

    required = [
        "device_id",
        "ipei",
        "device_location",
        "location_name",
    ]

    missing = [col for col in required if col not in index_df.columns]

    if missing:
        raise ValueError(f"Index file missing columns: {missing}")

    index_df["device_id"] = index_df["device_id"].astype(str).str.strip()
    index_df["ipei"] = index_df["ipei"].astype(str).str.strip()
    index_df["device_location"] = index_df["device_location"].astype(str).str.strip()
    index_df["location_name"] = index_df["location_name"].astype(str).str.strip()
    index_df["location_key"] = index_df["location_name"].apply(normalise_text)

    return index_df


def parse_wide_raw_store(raw_file: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(raw_file)

    fixed_cols = {
        "location_name",
        "device_id",
        "ipei",
        "device_location",
    }

    output_rows = []

    for _, row in raw_df.iterrows():
        records = {}

        for col, value in row.items():
            if col in fixed_cols:
                continue

            match = re.match(
                r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})_(.+)",
                col,
            )

            if not match:
                continue

            snapshot_time = match.group(1)
            field_name = match.group(2)

            records.setdefault(snapshot_time, {})
            records[snapshot_time][field_name] = value

        for snapshot_time, fields in records.items():
            output_rows.append({
                "event_type": "SOC",
                "event_time": snapshot_time,

                "location_name": row.get("location_name"),
                "device_id": row.get("device_id"),
                "ipei": row.get("ipei"),
                "device_location": row.get("device_location"),

                "snapshot_time": snapshot_time,
                "battery_level": fields.get("battery_level"),
                "signal_level": fields.get("signal_level"),
                "status": fields.get("status"),
                "timestamp": fields.get("timestamp"),
                "last_active": fields.get("last_active"),

                "activation_description": pd.NA,
                "activation_response_time_s": pd.NA,
                "activation_endpoint_id": pd.NA,
                "activation_portal_location": pd.NA,
            })

    df = pd.DataFrame(output_rows)

    if df.empty:
        return df

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["last_active"] = pd.to_datetime(df["last_active"], errors="coerce")
    df["battery_level"] = pd.to_numeric(df["battery_level"], errors="coerce")
    df["signal_level"] = pd.to_numeric(df["signal_level"], errors="coerce")

    df["ipei"] = df["ipei"].astype(str).str.strip()
    df["device_id"] = df["device_id"].astype(str).str.strip()
    df["device_location"] = df["device_location"].astype(str).str.strip()
    df["location_name"] = df["location_name"].astype(str).str.strip()

    df = df.dropna(subset=["event_time"])
    return df


def load_portal_report(portal_file: Path) -> pd.DataFrame:
    df = pd.read_csv(portal_file)

    required = [
        "Call initiation time",
        "Endpoint ID",
        "Location",
        "Description",
        "Response time (s)",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Portal report missing columns: {missing}")

    df["Call initiation time"] = pd.to_datetime(
        df["Call initiation time"],
        dayfirst=True,
        errors="coerce",
    )

    df["Endpoint ID"] = df["Endpoint ID"].astype(str).str.strip()
    df["Location"] = df["Location"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()
    df["Response time (s)"] = pd.to_numeric(df["Response time (s)"], errors="coerce")

    df = df.dropna(subset=["Call initiation time"])

    return df


def map_activations_to_ipei(
    portal_df: pd.DataFrame,
    index_df: pd.DataFrame,
    raw_store_df: pd.DataFrame,
) -> pd.DataFrame:
    location_names = raw_store_df["location_name"].dropna().astype(str).unique()

    if len(location_names) == 0:
        raise ValueError("Could not infer location_name from raw store data.")

    location_name = location_names[0]

    location_index = index_df[
        index_df["location_key"] == normalise_text(location_name)
    ].copy()

    if location_index.empty:
        # Fallback: only use devices already present in raw file
        raw_devices = raw_store_df[[
            "device_id",
            "ipei",
            "device_location",
            "location_name",
        ]].drop_duplicates()

        location_index = raw_devices.copy()

    mapped = portal_df.merge(
        location_index[[
            "device_id",
            "ipei",
            "device_location",
            "location_name",
        ]].drop_duplicates(),
        left_on="Endpoint ID",
        right_on="device_id",
        how="left",
    )

    mapped["ipei"] = mapped["ipei"].fillna("UNKNOWN_IPEI")
    mapped["device_location"] = mapped["device_location"].fillna(mapped["Location"])
    mapped["location_name"] = mapped["location_name"].fillna(location_name)

    return mapped


def build_activation_history(mapped_portal_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in mapped_portal_df.iterrows():
        rows.append({
            "event_type": "ACTIVATION",
            "event_time": row.get("Call initiation time"),

            "location_name": row.get("location_name"),
            "device_id": row.get("device_id"),
            "ipei": row.get("ipei"),
            "device_location": row.get("device_location"),

            "snapshot_time": pd.NA,
            "battery_level": pd.NA,
            "signal_level": pd.NA,
            "status": pd.NA,
            "timestamp": pd.NA,
            "last_active": pd.NA,

            "activation_description": row.get("Description"),
            "activation_response_time_s": row.get("Response time (s)"),
            "activation_endpoint_id": row.get("Endpoint ID"),
            "activation_portal_location": row.get("Location"),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["activation_response_time_s"] = pd.to_numeric(
        df["activation_response_time_s"],
        errors="coerce",
    )

    return df.dropna(subset=["event_time"])


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main(store_search: str | None = None) -> Path:
    store_search = store_search or STORE_SEARCH

    raw_file = find_matching_file(RAW_DIR, "*_raw.csv", store_search)
    portal_file = find_matching_file(PORTAL_REPORT_DIR, "*.csv", store_search)

    print("\n----------------------------------------")
    print(f"Store search       : {store_search}")
    print(f"Raw file matched   : {raw_file.name}")
    print(f"Portal file matched: {portal_file.name}")
    print(f"Output directory   : {OUTPUT_DIR}")
    print("----------------------------------------\n")

    index_df = load_index()

    soc_df = parse_wide_raw_store(raw_file)
    portal_df = load_portal_report(portal_file)

    mapped_portal_df = map_activations_to_ipei(
        portal_df=portal_df,
        index_df=index_df,
        raw_store_df=soc_df,
    )

    activation_df = build_activation_history(mapped_portal_df)

    combined_df = pd.concat(
        [soc_df, activation_df],
        ignore_index=True,
    )

    combined_df = combined_df.sort_values(
        by=[
            "ipei",
            "event_time",
            "event_type",
        ]
    )

    numeric_cols = combined_df.select_dtypes(include="number").columns
    combined_df[numeric_cols] = combined_df[numeric_cols].round(2)

    store_key = extract_store_key_from_raw_filename(raw_file)

    output_file = OUTPUT_DIR / f"{safe_filename(store_key)}_single_store_history.csv"

    combined_df.to_csv(output_file, index=False)

    print("Done.")
    print(f"Rows written      : {len(combined_df)}")
    print(f"SOC rows          : {(combined_df['event_type'] == 'SOC').sum()}")
    print(f"Activation rows   : {(combined_df['event_type'] == 'ACTIVATION').sum()}")
    print(f"Unique IPEIs      : {combined_df['ipei'].nunique()}")
    print(f"Output file       : {output_file}")

    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single-store combined SoC and activation history CSV.")
    parser.add_argument(
        "--store-search",
        default=STORE_SEARCH,
        help="Store search string. Partial match is OK, e.g. Oxford, 0786, Covent.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(store_search=args.store_search)