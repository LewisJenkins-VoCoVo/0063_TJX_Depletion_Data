#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent.parent

INPUT_DIR = BASE_DIR / "2_Parsed_Store" / "2_Preprocess"

OUTPUT_DIR = (
    BASE_DIR
    / "3_Analysis_Datasets"
    / "2_Hourly_Discharge_by_Time"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_EVENTS_CSV = OUTPUT_DIR / "rapid_discharge_events.csv"
OUTPUT_ALL_INTERVALS_CSV = OUTPUT_DIR / "classified_discharge_intervals.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "rapid_discharge_summary.csv"
OUTPUT_PLOT = OUTPUT_DIR / "rapid_discharge_distribution.png"

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
BIN_HOURS = 2

MAX_INTERVAL_HOURS = 2
MIN_INTERVAL_MINUTES = 1

NOMINAL_DROP_MIN_PERCENT = 0
NOMINAL_DROP_MAX_PERCENT = 1

RAPID_DROP_MIN_PERCENT = 2
RAPID_DROP_MAX_PERCENT = 12

SHOW_PLOT = True
SAVE_PLOT = True

# -------------------------------------------------
# LOAD
# -------------------------------------------------
files = sorted(INPUT_DIR.glob("*_cycle_preprocess.csv"))

if not files:
    raise FileNotFoundError(f"No preprocess files found in: {INPUT_DIR}")

df_list = []

for file in files:
    tmp = pd.read_csv(file)
    tmp["source_file"] = file.name
    df_list.append(tmp)

df = pd.concat(df_list, ignore_index=True)

required_cols = [
    "location_name",
    "device_location",
    "ipei",
    "date",
    "cycle_number",
    "sample_number",
    "snapshot_time",
    "battery_level",
]

missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -------------------------------------------------
# CLEAN
# -------------------------------------------------
df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce")
df["battery_level"] = pd.to_numeric(df["battery_level"], errors="coerce")
df["sample_number"] = pd.to_numeric(df["sample_number"], errors="coerce")
df["cycle_number"] = pd.to_numeric(df["cycle_number"], errors="coerce")

df = df.dropna(
    subset=[
        "snapshot_time",
        "battery_level",
        "sample_number",
        "cycle_number",
    ]
)

df = df.sort_values(
    by=[
        "location_name",
        "device_location",
        "ipei",
        "date",
        "cycle_number",
        "sample_number",
        "snapshot_time",
    ]
)

# -------------------------------------------------
# BUILD INTERVALS
# -------------------------------------------------
interval_rows = []

group_cols = [
    "location_name",
    "device_location",
    "ipei",
    "date",
    "cycle_number",
]

for _, group in df.groupby(group_cols, dropna=False):

    group = group.sort_values(["sample_number", "snapshot_time"]).copy()

    group["prev_snapshot_time"] = group["snapshot_time"].shift(1)
    group["prev_battery_level"] = group["battery_level"].shift(1)

    group = group.dropna(
        subset=[
            "prev_snapshot_time",
            "prev_battery_level",
        ]
    )

    if group.empty:
        continue

    group["interval_hours"] = (
        group["snapshot_time"] - group["prev_snapshot_time"]
    ).dt.total_seconds() / 3600

    group["battery_drop_percent"] = (
        group["prev_battery_level"] - group["battery_level"]
    )

    group["drop_rate_percent_per_hour"] = (
        group["battery_drop_percent"] / group["interval_hours"]
    )

    group["midpoint_time"] = (
        group["prev_snapshot_time"]
        + (group["snapshot_time"] - group["prev_snapshot_time"]) / 2
    )

    # -------------------------------------------------
    # STRICT CLASSIFICATION
    # -------------------------------------------------
    group["interval_class"] = "excluded"

    valid_interval_mask = (
        (group["interval_hours"] >= MIN_INTERVAL_MINUTES / 60) &
        (group["interval_hours"] <= MAX_INTERVAL_HOURS)
    )

    nominal_mask = (
        valid_interval_mask &
        (group["battery_drop_percent"] >= NOMINAL_DROP_MIN_PERCENT) &
        (group["battery_drop_percent"] <= NOMINAL_DROP_MAX_PERCENT)
    )

    rapid_mask = (
        valid_interval_mask &
        (group["battery_drop_percent"] >= RAPID_DROP_MIN_PERCENT) &
        (group["battery_drop_percent"] <= RAPID_DROP_MAX_PERCENT)
    )

    group.loc[nominal_mask, "interval_class"] = "nominal"
    group.loc[rapid_mask, "interval_class"] = "rapid_discharge"

    interval_rows.append(
        group[
            [
                "location_name",
                "device_location",
                "ipei",
                "date",
                "cycle_number",
                "prev_snapshot_time",
                "snapshot_time",
                "midpoint_time",
                "prev_battery_level",
                "battery_level",
                "battery_drop_percent",
                "interval_hours",
                "drop_rate_percent_per_hour",
                "interval_class",
                "source_file",
            ]
        ]
    )

if not interval_rows:
    raise ValueError("No intervals produced.")

all_intervals_df = pd.concat(interval_rows, ignore_index=True)

events_df = all_intervals_df[
    all_intervals_df["interval_class"] == "rapid_discharge"
].copy()

if events_df.empty:
    raise ValueError("No rapid discharge events detected.")

# -------------------------------------------------
# CHRONOLOGICAL 2-HOUR BINS
# -------------------------------------------------
events_df["time_bin"] = events_df["midpoint_time"].dt.floor(f"{BIN_HOURS}h")

# -------------------------------------------------
# SUMMARY - CHRONOLOGICAL
# -------------------------------------------------
summary = (
    events_df.groupby("time_bin", dropna=False)
    .agg(
        rapid_event_count=("ipei", "count"),
        unique_ipeis=("ipei", "nunique"),
        unique_stores=("location_name", "nunique"),
        mean_drop_percent=("battery_drop_percent", "mean"),
        mean_drop_rate_percent_per_hour=("drop_rate_percent_per_hour", "mean"),
    )
    .reset_index()
    .sort_values("time_bin")
)

# -------------------------------------------------
# ENSURE ALL 2-HOUR BINS EXIST ACROSS SAMPLE
# -------------------------------------------------
start_time = all_intervals_df["midpoint_time"].min().floor(f"{BIN_HOURS}h")
end_time = all_intervals_df["midpoint_time"].max().ceil(f"{BIN_HOURS}h")

all_bins = pd.DataFrame({
    "time_bin": pd.date_range(
        start=start_time,
        end=end_time,
        freq=f"{BIN_HOURS}h",
    )
})

summary = all_bins.merge(
    summary,
    on="time_bin",
    how="left",
)

summary = summary.sort_values("time_bin").reset_index(drop=True)

summary["rapid_event_count"] = summary["rapid_event_count"].fillna(0)
summary["unique_ipeis"] = summary["unique_ipeis"].fillna(0)
summary["unique_stores"] = summary["unique_stores"].fillna(0)

# -------------------------------------------------
# ROUNDING
# -------------------------------------------------
numeric_cols = summary.select_dtypes(include="number").columns
summary[numeric_cols] = summary[numeric_cols].round(3)

# -------------------------------------------------
# SAVE OUTPUTS
# -------------------------------------------------
all_intervals_df.to_csv(OUTPUT_ALL_INTERVALS_CSV, index=False)
events_df.to_csv(OUTPUT_EVENTS_CSV, index=False)
summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

# -------------------------------------------------
# PLOT - CHRONOLOGICAL
# -------------------------------------------------
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(14, 5))

ax.bar(
    summary["time_bin"],
    summary["rapid_event_count"],
    width=0.07,
)

ax.set_xlabel("Date / Time")
ax.set_ylabel("Rapid Discharge Event Count")

ax.set_title(
    "Rapid Discharge Events Across Full Sample Period\n"
    "Rapid = 2–12% SoC drop over ≤2 hours"
)

# -------------------------------------------------
# X AXIS FORMATTING
# -------------------------------------------------
import matplotlib.dates as mdates

# Major ticks = each day
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

# Minor ticks = 06:00, 12:00, 18:00
ax.xaxis.set_minor_locator(
    mdates.HourLocator(byhour=[6, 12, 18])
)

ax.xaxis.set_minor_formatter(
    mdates.DateFormatter("%H:%M")
)

# Major labels (dates)
for label in ax.get_xticklabels(which="major"):
    label.set_rotation(45)
    label.set_horizontalalignment("right")
    label.set_fontsize(12)

# Minor labels (times)
for label in ax.get_xticklabels(which="minor"):
    label.set_rotation(45)
    label.set_horizontalalignment("right")
    label.set_fontsize(8)

# Increase spacing
ax.tick_params(axis="x", which="major", pad=12)
ax.tick_params(axis="x", which="minor", pad=2)

# -------------------------------------------------
# GRID
# -------------------------------------------------
ax.grid(
    True,
    axis="y",
    alpha=0.3,
)

ax.grid(
    True,
    which="minor",
    axis="x",
    alpha=0.15,
)

plt.tight_layout()

if SAVE_PLOT:
    plt.savefig(
        OUTPUT_PLOT,
        dpi=200,
    )

if SHOW_PLOT:
    plt.show()
else:
    plt.close()