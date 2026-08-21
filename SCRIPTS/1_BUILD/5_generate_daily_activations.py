#!/usr/bin/env python3

import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

# -------------------------------------------------
# PROJECT FOLDERS
# -------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent

PORTAL_REPORT_DIR = BASE_DIR / "SOURCE_DATA" / "Portal_Data" / "Activation_Reports"
INDEX_DIR = BASE_DIR / "0_Index"

OUTPUT_DIR = BASE_DIR / "2_Parsed_Store" / "4_Daily_Stats" / "2_Activations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = INDEX_DIR / "index.csv"


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


def safe_filename(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name if name else "Unknown"


def extract_location_from_report_filename(path: Path) -> str:
    stem = path.stem

    if " - call point -" in stem:
        return stem.split(" - call point -")[0].strip()

    return stem.strip()


def extract_store_codes(value: str) -> set[str]:
    """
    Extract likely store code(s) from location/report names.

    Examples:
    ST_UKTK_0786(...)     -> {"786"}
    STUKTK_408(...)       -> {"408"}
    ST_UK_COMBO_266_661   -> {"266", "661"}
    """
    value = str(value).strip()

    # Only inspect the prefix before the bracketed department code
    prefix = value.split("(")[0]

    # Normalise known typos / variants
    prefix = prefix.replace("STUKTK", "ST_UKTK")
    prefix = prefix.replace("ST_UKTKM", "ST_UKTK")

    numbers = re.findall(r"\d+", prefix)

    return {str(int(n)) for n in numbers if n.strip("0") != ""}


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalise_text(a), normalise_text(b)).ratio()


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
    index_df["store_codes"] = index_df["location_name"].apply(extract_store_codes)

    return index_df


def find_location_index(report_file: Path, index_df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    report_location = extract_location_from_report_filename(report_file)
    report_codes = extract_store_codes(report_location)
    report_key = normalise_text(report_location)

    # 1) Best method: match by extracted store code(s)
    if report_codes:
        matched = index_df[
            index_df["store_codes"].apply(lambda codes: bool(codes & report_codes))
        ].copy()

        if not matched.empty:
            location_name = matched["location_name"].mode().iloc[0]
            return location_name, matched

    # 2) Fallback: exact normalised location name
    exact = index_df[index_df["location_key"] == report_key].copy()

    if not exact.empty:
        location_name = exact["location_name"].mode().iloc[0]
        return location_name, exact

    # 3) Fallback: fuzzy match against unique index locations
    unique_locations = index_df["location_name"].dropna().unique()

    best_location = None
    best_score = 0

    for location in unique_locations:
        score = similarity(report_location, location)

        if score > best_score:
            best_score = score
            best_location = location

    if best_location is not None and best_score >= 0.70:
        matched = index_df[index_df["location_name"] == best_location].copy()
        return best_location, matched

    # 4) No match
    return report_location, index_df.iloc[0:0].copy()


def load_portal_report(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [
        "Call initiation time",
        "Endpoint ID",
        "Location",
        "Description",
        "Response time (s)",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df["Call initiation time"] = pd.to_datetime(
        df["Call initiation time"],
        dayfirst=True,
        errors="coerce",
    )

    df["Response time (s)"] = pd.to_numeric(
        df["Response time (s)"],
        errors="coerce",
    )

    df["Endpoint ID"] = df["Endpoint ID"].astype(str).str.strip()
    df["Location"] = df["Location"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()

    df = df.dropna(subset=["Call initiation time"])

    return df


def map_portal_row_to_index(row: pd.Series, location_index: pd.DataFrame) -> pd.Series:
    endpoint_id = str(row["Endpoint ID"]).strip()
    portal_location = str(row["Location"]).strip()

    candidates = location_index[
        location_index["device_id"] == endpoint_id
    ].copy()

    if candidates.empty:
        return pd.Series({
            "ipei": "UNKNOWN_IPEI",
            "device_location": portal_location,
            "mapping_status": "unmatched_endpoint",
        })

    if len(candidates) == 1:
        match = candidates.iloc[0]
        return pd.Series({
            "ipei": match["ipei"],
            "device_location": match["device_location"],
            "mapping_status": "matched_endpoint",
        })

    # If duplicate endpoint IDs exist, choose best device_location similarity
    candidates["location_similarity"] = candidates["device_location"].apply(
        lambda x: similarity(portal_location, x)
    )

    match = candidates.sort_values(
        by="location_similarity",
        ascending=False,
    ).iloc[0]

    return pd.Series({
        "ipei": match["ipei"],
        "device_location": match["device_location"],
        "mapping_status": "matched_endpoint_fuzzy_location",
    })


def process_portal_report(report_file: Path, index_df: pd.DataFrame) -> pd.DataFrame:
    df = load_portal_report(report_file)

    matched_location_name, location_index = find_location_index(report_file, index_df)
    df["location_name"] = matched_location_name

    if location_index.empty:
        print(f" -> WARNING: Could not match report to index location: {report_file.name}")

    mapping_df = df.apply(
        lambda row: map_portal_row_to_index(row, location_index),
        axis=1,
    )

    df = pd.concat([df, mapping_df], axis=1)

    unknown_count = (df["ipei"] == "UNKNOWN_IPEI").sum()

    if unknown_count:
        print(f" -> WARNING: UNKNOWN_IPEI rows: {unknown_count} / {len(df)}")

    df["date"] = df["Call initiation time"].dt.date

    df["is_timeout"] = df["Description"] == "TIMEOUT"
    df["is_cleardown"] = df["Description"] == "CLEARDOWN"
    df["is_answered"] = df["Description"] == "ANSWERED"

    df["timeout_response_time_s"] = df["Response time (s)"].where(df["is_timeout"], 0)
    df["cleardown_response_time_s"] = df["Response time (s)"].where(df["is_cleardown"], 0)
    df["answered_response_time_s"] = df["Response time (s)"].where(df["is_answered"], 0)

    grouped = (
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
            total_activations=(
                "Description",
                "count",
            ),
            total_activation_time_s=(
                "Response time (s)",
                "sum",
            ),
            timeout_count=(
                "is_timeout",
                "sum",
            ),
            timeout_response_time_s=(
                "timeout_response_time_s",
                "sum",
            ),
            cleardown_count=(
                "is_cleardown",
                "sum",
            ),
            cleardown_response_time_s=(
                "cleardown_response_time_s",
                "sum",
            ),
            answered_count=(
                "is_answered",
                "sum",
            ),
            answered_response_time_s=(
                "answered_response_time_s",
                "sum",
            ),
        )
        .reset_index()
    )

    numeric_cols = grouped.select_dtypes(include="number").columns
    grouped[numeric_cols] = grouped[numeric_cols].round(2)

    grouped = grouped.sort_values(
        by=[
            "date",
            "location_name",
            "device_location",
            "ipei",
        ]
    )

    return grouped


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main() -> None:
    index_df = load_index()

    report_files = sorted(
        file for file in PORTAL_REPORT_DIR.glob("*.csv")
        if not file.name.startswith("._")
    )

    print("\n----------------------------------------")
    print(f"Portal report files found : {len(report_files)}")
    print(f"Output directory          : {OUTPUT_DIR}")
    print("----------------------------------------\n")

    if not report_files:
        print(f"No portal report CSV files found in: {PORTAL_REPORT_DIR}")
        return

    for file_number, report_file in enumerate(report_files, start=1):
        print(f"[{file_number}/{len(report_files)}] Processing: {report_file.name}")

        try:
            output_df = process_portal_report(report_file, index_df)
        except Exception as exc:
            print(f" -> ERROR: {exc}")
            continue

        if output_df.empty:
            print(" -> No activation rows produced, skipping")
            continue

        report_location = extract_location_from_report_filename(report_file)

        start_date = pd.to_datetime(output_df["date"]).min().strftime("%Y-%m-%d")
        end_date = pd.to_datetime(output_df["date"]).max().strftime("%Y-%m-%d")

        output_name = (
            f"{safe_filename(report_location)}_"
            f"{start_date}_"
            f"{end_date}_activations.csv"
        )

        output_path = OUTPUT_DIR / output_name
        output_df.to_csv(output_path, index=False)

        print(f" -> Output written: {output_path.name}")
        print(f" -> Daily activation rows: {len(output_df)}")

    print("\nDone.")
    print(f"Daily activation files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()