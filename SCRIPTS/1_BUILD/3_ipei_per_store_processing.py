#!/usr/bin/env python3

import re
import pandas as pd
from pathlib import Path

# -------------------------------------------------
# PROCESSING SETTINGS
# -------------------------------------------------
MIN_BATTERY_DROP_PERCENT = 0

IGNORE_LOW_VARIANCE_CYCLES = True
MIN_CYCLE_VARIANCE_PERCENT = 1.0

FILTER_DISCONTINUITY_DROPS = True
DISCONTINUITY_DROP_PERCENT = 40
DISCONTINUITY_MAX_HOURS = 1.25

# -------------------------------------------------
# DAY SELECTION / FULL-DAY RULES
# -------------------------------------------------
USE_SELECTED_DATE_RANGE = True

SELECTED_START_DATE = "2026-04-25"
SELECTED_END_DATE = "2026-05-06"

REQUIRE_FULL_SOC_DAYS = False  ## Broken - don't use this.
MIN_FULL_DAY_SOC_HOURS = 22.5

# -------------------------------------------------
# PROJECT FOLDERS
# -------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent

RAW_DIR = BASE_DIR / "2_Parsed_Store" / "1_Raw"
PREPROCESS_DIR = BASE_DIR / "2_Parsed_Store" / "2_Preprocess"
OUTPUT_DIR = BASE_DIR / "2_Parsed_Store" / "3_Processed"

PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def is_selected_date(day) -> bool:
    if not USE_SELECTED_DATE_RANGE:
        return True

    day = pd.to_datetime(day).date()
    start = pd.to_datetime(SELECTED_START_DATE).date()
    end = pd.to_datetime(SELECTED_END_DATE).date()

    return start <= day <= end


def is_full_soc_day(day_df: pd.DataFrame, time_col: str) -> bool:
    if not REQUIRE_FULL_SOC_DAYS:
        return True

    if day_df.empty or len(day_df) < 2:
        return False

    first_time = day_df[time_col].min()
    last_time = day_df[time_col].max()

    if pd.isna(first_time) or pd.isna(last_time):
        return False

    coverage_hours = (last_time - first_time).total_seconds() / 3600

    return coverage_hours >= MIN_FULL_DAY_SOC_HOURS

def parse_wide_store_row(row: pd.Series) -> pd.DataFrame:
    fixed_cols = {
        "location_name",
        "device_id",
        "ipei",
        "device_location",
    }

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

    output_rows = []

    for snapshot_time, fields in records.items():
        output_rows.append({
            "location_name": row.get("location_name"),
            "device_id": row.get("device_id"),
            "ipei": row.get("ipei"),
            "device_location": row.get("device_location"),
            "snapshot_time": snapshot_time,
            "timestamp": fields.get("timestamp"),
            "status": fields.get("status"),
            "last_active": fields.get("last_active"),
            "battery_level": fields.get("battery_level"),
            "signal_level": fields.get("signal_level"),
        })

    df = pd.DataFrame(output_rows)

    if df.empty:
        return df

    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["last_active"] = pd.to_datetime(df["last_active"], errors="coerce")
    df["battery_level"] = pd.to_numeric(df["battery_level"], errors="coerce")
    df["signal_level"] = pd.to_numeric(df["signal_level"], errors="coerce")

    df = df.dropna(subset=["snapshot_time", "battery_level"])
    df = df.sort_values(by="snapshot_time")

    return df


def is_discontinuity_drop(prev_row: dict, curr_row: dict) -> bool:
    if not FILTER_DISCONTINUITY_DROPS:
        return False

    prev_batt = prev_row["battery_level"]
    curr_batt = curr_row["battery_level"]

    if pd.isna(prev_batt) or pd.isna(curr_batt):
        return False

    battery_drop = prev_batt - curr_batt

    time_delta_hours = (
        curr_row["snapshot_time"] - prev_row["snapshot_time"]
    ).total_seconds() / 3600

    return (
        battery_drop >= DISCONTINUITY_DROP_PERCENT
        and time_delta_hours <= DISCONTINUITY_MAX_HOURS
    )


def make_cycle_output(
    source_row: pd.Series,
    day,
    cycle_number: int,
    cycle_rows: list[dict],
) -> dict | None:
    if len(cycle_rows) < 2:
        return None

    cycle_df = pd.DataFrame(cycle_rows).sort_values(by="snapshot_time")

    first_row = cycle_df.iloc[0]
    last_row = cycle_df.iloc[-1]

    first_battery = first_row["battery_level"]
    last_battery = last_row["battery_level"]

    battery_drop = first_battery - last_battery

    if battery_drop <= MIN_BATTERY_DROP_PERCENT:
        return None

    if IGNORE_LOW_VARIANCE_CYCLES and battery_drop <= MIN_CYCLE_VARIANCE_PERCENT:
        return None

    duration_hours = (
        last_row["snapshot_time"] - first_row["snapshot_time"]
    ).total_seconds() / 3600

    discharge_rate_per_day = pd.NA

    if duration_hours > 0:
        discharge_rate_per_day = battery_drop / (duration_hours / 24)

    return {
        "location_name": source_row.get("location_name"),
        "device_id": source_row.get("device_id"),
        "ipei": source_row.get("ipei"),
        "device_location": source_row.get("device_location"),

        "date": day,
        "cycle_number": cycle_number,

        "cycle_start_snapshot_time": first_row["snapshot_time"],
        "cycle_end_snapshot_time": last_row["snapshot_time"],
        "cycle_start_timestamp": first_row["timestamp"],
        "cycle_end_timestamp": last_row["timestamp"],
        "cycle_start_last_active": first_row["last_active"],
        "cycle_end_last_active": last_row["last_active"],

        "cycle_start_status": first_row["status"],
        "cycle_end_status": last_row["status"],
        "status_values_seen": ", ".join(
            sorted(cycle_df["status"].dropna().astype(str).unique())
        ),

        "cycle_start_battery_level": first_battery,
        "cycle_end_battery_level": last_battery,
        "battery_drop_percent": battery_drop,

        "cycle_start_signal_level": first_row["signal_level"],
        "cycle_end_signal_level": last_row["signal_level"],
        "min_signal_level": cycle_df["signal_level"].min(),
        "max_signal_level": cycle_df["signal_level"].max(),
        "mean_signal_level": cycle_df["signal_level"].mean(),

        "duration_hours": duration_hours,
        "discharge_rate_percent_per_day": discharge_rate_per_day,
        "samples_used": len(cycle_df),
    }


def add_preprocess_cycle_rows(
    preprocess_rows: list[dict],
    source_row: pd.Series,
    day,
    cycle_number: int,
    cycle_rows: list[dict],
) -> None:
    if len(cycle_rows) < 2:
        return

    for sample_number, sample in enumerate(cycle_rows, start=1):
        preprocess_rows.append({
            "location_name": source_row.get("location_name"),
            "device_id": source_row.get("device_id"),
            "ipei": source_row.get("ipei"),
            "device_location": source_row.get("device_location"),

            "date": day,
            "cycle_number": cycle_number,
            "sample_number": sample_number,

            "snapshot_time": sample.get("snapshot_time"),
            "battery_level": sample.get("battery_level"),
            "signal_level": sample.get("signal_level"),
            "status": sample.get("status"),
            "timestamp": sample.get("timestamp"),
            "last_active": sample.get("last_active"),
        })


def process_device_daily_cycles(row: pd.Series) -> tuple[list[dict], list[dict]]:
    df = parse_wide_store_row(row)

    if df.empty:
        return [], []

    df = df.copy()
    df["date"] = df["snapshot_time"].dt.date

    output_rows = []
    preprocess_rows = []

    for day, day_df in df.groupby("date"):
        day_df = day_df.sort_values(by="snapshot_time")

        if not is_selected_date(day):
            continue

        if not is_full_soc_day(day_df, "snapshot_time"):
            continue

        if day_df.empty:
            continue

        day_df = day_df[
            (day_df["status"].astype(str).str.lower() != "offline") &
            ~(
                (day_df["status"].astype(str).str.lower() == "online") &
                (day_df["battery_level"] == 0)
            )
        ]

        if len(day_df) < 2:
            continue

        # Remove flat battery runs only at the charge/discharge extremes.
        # This prevents long 100% charger plateaus being treated as the
        # start of a discharge cycle, but preserves valid mid-range repeated
        # SoC samples during normal discharge.
        next_battery = day_df["battery_level"].shift(-1)

        extreme_flat_duplicate = (
                (day_df["battery_level"] == next_battery) &
                (
                        (day_df["battery_level"] <= 2) |
                        (day_df["battery_level"] >= 100)
                )
        )

        day_df = day_df[~extreme_flat_duplicate]

        if len(day_df) < 2:
            continue

        current_cycle = []
        cycle_number = 1

        rows = list(day_df.to_dict("records"))

        for idx in range(1, len(rows)):
            prev_row = rows[idx - 1]
            curr_row = rows[idx]

            prev_batt = prev_row["battery_level"]
            curr_batt = curr_row["battery_level"]

            if is_discontinuity_drop(prev_row, curr_row):
                current_cycle = []
                continue

            if curr_batt < prev_batt:
                # Discharge
                if not current_cycle:
                    current_cycle = [prev_row, curr_row]
                else:
                    current_cycle.append(curr_row)

            elif curr_batt == prev_batt:
                # Keep repeated mid-range SoC values as valid duration samples.
                # Do not keep flat 0/100 charger/dead plateaus.
                if 2 < curr_batt < 100:
                    if current_cycle:
                        current_cycle.append(curr_row)
                continue

            elif curr_batt > prev_batt:
                cycle_output = make_cycle_output(
                    row,
                    day,
                    cycle_number,
                    current_cycle,
                )

                if cycle_output is not None:
                    output_rows.append(cycle_output)

                    add_preprocess_cycle_rows(
                        preprocess_rows,
                        row,
                        day,
                        cycle_number,
                        current_cycle,
                    )

                    cycle_number += 1

                current_cycle = []

            else:
                continue

        cycle_output = make_cycle_output(
            row,
            day,
            cycle_number,
            current_cycle,
        )

        if cycle_output is not None:
            output_rows.append(cycle_output)

            add_preprocess_cycle_rows(
                preprocess_rows,
                row,
                day,
                cycle_number,
                current_cycle,
            )

    return output_rows, preprocess_rows


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main() -> None:
    raw_files = sorted(RAW_DIR.glob("*_raw.csv"))

    print("\n----------------------------------------")
    print(f"Raw store files found : {len(raw_files)}")
    print(f"Preprocess directory  : {PREPROCESS_DIR}")
    print(f"Processed directory   : {OUTPUT_DIR}")
    print(f"Min battery drop      : > {MIN_BATTERY_DROP_PERCENT}%")
    print(f"Low variance filter   : {'enabled' if IGNORE_LOW_VARIANCE_CYCLES else 'disabled'} <= {MIN_CYCLE_VARIANCE_PERCENT}%")
    print(f"Discontinuity filter  : {FILTER_DISCONTINUITY_DROPS}")
    print(f"Discontinuity drop    : >= {DISCONTINUITY_DROP_PERCENT}% within {DISCONTINUITY_MAX_HOURS}h")
    print("----------------------------------------\n")

    if not raw_files:
        print(f"No raw files found in: {RAW_DIR}")
        return

    for file_number, raw_file in enumerate(raw_files, start=1):
        print(f"[{file_number}/{len(raw_files)}] Processing: {raw_file.name}")

        store_df = pd.read_csv(raw_file)

        if store_df.empty:
            print(" -> Empty file, skipping")
            continue

        processed_rows = []
        preprocess_rows = []

        for _, row in store_df.iterrows():
            device_processed_rows, device_preprocess_rows = process_device_daily_cycles(row)
            processed_rows.extend(device_processed_rows)
            preprocess_rows.extend(device_preprocess_rows)

        if not processed_rows:
            print(" -> No valid discharge cycles produced, skipping")
            continue

        processed_df = pd.DataFrame(processed_rows)

        processed_df = processed_df.sort_values(
            by=[
                "location_name",
                "device_location",
                "ipei",
                "date",
                "cycle_number",
            ]
        )

        numeric_cols = processed_df.select_dtypes(include="number").columns
        processed_df[numeric_cols] = processed_df[numeric_cols].round(2)

        output_name = raw_file.name.replace("_raw.csv", "_processed.csv")
        output_path = OUTPUT_DIR / output_name

        processed_df.to_csv(output_path, index=False)

        print(f" -> Processed written: {output_path.name}")
        print(f" -> Discharge cycles written: {len(processed_df)}")

        if preprocess_rows:
            preprocess_df = pd.DataFrame(preprocess_rows)

            preprocess_df = preprocess_df.sort_values(
                by=[
                    "location_name",
                    "device_location",
                    "ipei",
                    "date",
                    "cycle_number",
                    "sample_number",
                ]
            )

            numeric_cols = preprocess_df.select_dtypes(include="number").columns
            preprocess_df[numeric_cols] = preprocess_df[numeric_cols].round(2)

            preprocess_output_name = raw_file.name.replace("_raw.csv", "_cycle_preprocess.csv")
            preprocess_output_path = PREPROCESS_DIR / preprocess_output_name

            preprocess_df.to_csv(preprocess_output_path, index=False)

            print(f" -> Preprocess written: {preprocess_output_path.name}")
            print(f" -> Preprocess rows: {len(preprocess_df)}")

    print("\nDone.")
    print(f"Preprocess files saved to: {PREPROCESS_DIR}")
    print(f"Processed cycle files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()