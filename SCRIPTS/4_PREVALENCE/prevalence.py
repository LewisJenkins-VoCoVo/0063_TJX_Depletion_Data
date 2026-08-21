#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent

CALLBACK_OUTCOMES_FILE = (
    BASE_DIR
    / "3_Analysis_Datasets"
    / "2_Rapid_Depletion"
    / "Output"
    / "callback_outcomes.csv"
)

PREPROCESS_DIR = (
    BASE_DIR
    / "2_Parsed_Store"
    / "2_Preprocess"
)

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
OBSERVATION_DAYS = 12

RAPID_CLASSES = [
    "suspected_rapid_depletion",
    "severe_rapid_depletion",
]

MIN_VALID_SAMPLES = 24
MIN_SOC_VARIATION = 5

HIGH_STATIC_MIN = 95
LOW_STATIC_MAX = 5

DISCONTINUITY_DROP_PERCENT = 40
DISCONTINUITY_MAX_HOURS = 1.0

# -------------------------------------------------
# LOAD callback_outcomes.csv
# -------------------------------------------------
if not CALLBACK_OUTCOMES_FILE.exists():
    raise FileNotFoundError(
        f"Could not find: {CALLBACK_OUTCOMES_FILE}"
    )

events_df = pd.read_csv(CALLBACK_OUTCOMES_FILE)

# -------------------------------------------------
# LOAD PREPROCESS DATA
# -------------------------------------------------
files = sorted(PREPROCESS_DIR.glob("*_cycle_preprocess.csv"))

if not files:
    raise FileNotFoundError(
        f"No preprocess files found in: {PREPROCESS_DIR}"
    )

df_list = []

for file in files:
    tmp = pd.read_csv(file)
    tmp["source_file"] = file.name
    df_list.append(tmp)

raw_df = pd.concat(df_list, ignore_index=True)

required_cols = [
    "location_name",
    "ipei",
    "snapshot_time",
    "battery_level",
]

missing = [c for c in required_cols if c not in raw_df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

raw_df["snapshot_time"] = pd.to_datetime(
    raw_df["snapshot_time"],
    errors="coerce",
)

raw_df["battery_level"] = pd.to_numeric(
    raw_df["battery_level"],
    errors="coerce",
)

raw_df = raw_df.dropna(
    subset=[
        "snapshot_time",
        "battery_level",
    ]
)

raw_df = raw_df.sort_values(
    ["ipei", "snapshot_time"]
)

# -------------------------------------------------
# PER-IPEI QUALITY ANALYSIS
# -------------------------------------------------
quality_rows = []

for ipei, group in raw_df.groupby("ipei"):
    group = group.sort_values("snapshot_time").copy()

    sample_count = len(group)

    min_soc = group["battery_level"].min()
    max_soc = group["battery_level"].max()
    soc_range = max_soc - min_soc

    always_high = (
        min_soc >= HIGH_STATIC_MIN and
        max_soc <= 100
    )

    always_low = (
        min_soc >= 0 and
        max_soc <= LOW_STATIC_MAX
    )

    static_extreme = always_high or always_low

    battery_powered = (
        soc_range >= MIN_SOC_VARIATION and
        not static_extreme
    )

    insufficient_data = sample_count < MIN_VALID_SAMPLES

    # Detect discontinuities
    has_discontinuity = False

    if sample_count >= 2:
        temp = group.copy()
        temp["prev_time"] = temp["snapshot_time"].shift(1)
        temp["prev_soc"] = temp["battery_level"].shift(1)

        temp = temp.dropna(
            subset=[
                "prev_time",
                "prev_soc",
            ]
        )

        temp["dt_hours"] = (
            temp["snapshot_time"] -
            temp["prev_time"]
        ).dt.total_seconds() / 3600

        temp["soc_change"] = (
            temp["battery_level"] -
            temp["prev_soc"]
        ).abs()

        discontinuities = temp[
            (temp["dt_hours"] <= DISCONTINUITY_MAX_HOURS) &
            (temp["soc_change"] >= DISCONTINUITY_DROP_PERCENT)
        ]

        has_discontinuity = not discontinuities.empty

    excluded = (
        static_extreme or
        insufficient_data
    )

    quality_rows.append({
        "ipei": ipei,
        "sample_count": sample_count,
        "min_soc": min_soc,
        "max_soc": max_soc,
        "soc_range": soc_range,
        "battery_powered": battery_powered,
        "static_extreme": static_extreme,
        "insufficient_data": insufficient_data,
        "has_discontinuity": has_discontinuity,
        "excluded": excluded,
    })

quality_df = pd.DataFrame(quality_rows)

# -------------------------------------------------
# RAPID DEPLETION ANALYSIS
# -------------------------------------------------
answered_df = events_df[
    events_df["callback_type"] == "ANSWERED"
].copy()

rapid_df = answered_df[
    answered_df["trigger_classification"].isin(
        RAPID_CLASSES
    )
].copy()

# -------------------------------------------------
# BASIC COUNTS
# -------------------------------------------------
total_stores = raw_df["location_name"].nunique()
total_callpoints = quality_df["ipei"].nunique()

battery_powered_count = int(
    quality_df["battery_powered"].sum()
)

usable_battery_powered = int(
    (
        quality_df["battery_powered"] &
        ~quality_df["excluded"]
    ).sum()
)

static_extreme_count = int(
    quality_df["static_extreme"].sum()
)

insufficient_count = int(
    quality_df["insufficient_data"].sum()
)

discontinuity_count = int(
    quality_df["has_discontinuity"].sum()
)

excluded_count = int(
    quality_df["excluded"].sum()
)

# -------------------------------------------------
# RAPID DEPLETION METRICS
# -------------------------------------------------
total_answered_callbacks = len(answered_df)
total_rapid_events = len(rapid_df)

affected_callpoints = rapid_df["ipei"].nunique()
affected_stores = rapid_df["location_name"].nunique()

rapid_prevalence = (
    total_rapid_events / total_answered_callbacks
    if total_answered_callbacks > 0
    else 0
)

total_callpoint_days = (
    usable_battery_powered *
    OBSERVATION_DAYS
)

days_between_events = (
    total_callpoint_days / total_rapid_events
    if total_rapid_events > 0
    else np.inf
)

daily_probability = (
    total_rapid_events / total_callpoint_days
    if total_callpoint_days > 0
    else 0
)

events_per_day = (
    total_rapid_events / OBSERVATION_DAYS
)

events_per_100_callpoints_per_day = (
    total_rapid_events /
    total_callpoint_days *
    100
    if total_callpoint_days > 0
    else 0
)

affected_callpoint_percent = (
    affected_callpoints /
    usable_battery_powered *
    100
    if usable_battery_powered > 0
    else 0
)

# -------------------------------------------------
# OUTPUT
# -------------------------------------------------
print("\n====================================================")
print("Rapid Depletion Operational Frequency Summary")
print("====================================================")

print("Dataset Quality")
print("----------------------------------------------------")
print(f"Total stores represented                 : {total_stores}")
print(f"Total unique IPEIs                       : {total_callpoints}")
print(f"Devices showing battery-powered behaviour: {battery_powered_count}")
print(f"Devices with usable battery traces       : {usable_battery_powered}")
print()
print(f"Devices excluded from analysis           : {excluded_count}")
print(f"  Static 95–100% or 0–5% SoC             : {static_extreme_count}")
print(f"  Insufficient data                      : {insufficient_count}")
print(f"Devices with >40% change in 1 hour       : {discontinuity_count}")
print()

print("Rapid Depletion Metrics")
print("----------------------------------------------------")
print(f"Observation period (days)                : {OBSERVATION_DAYS}")
print(f"Total Answered Callbacks analysed        : {total_answered_callbacks}")
print(f"Total rapid depletion events             : {total_rapid_events}")
print()
print(f"Rapid depletion prevalence               : {rapid_prevalence * 100:.2f}%")

if rapid_prevalence > 0:
    print(
        f"Equivalent                              : "
        f"1 in {1 / rapid_prevalence:.2f} Answered Callbacks"
    )

print()
print(f"Stores affected                         : {affected_stores}")
print(f"Callpoints affected                     : {affected_callpoints}")
print(f"Callpoints affected (%)                 : {affected_callpoint_percent:.2f}%")
print()
print(f"Rapid events per day (estate-wide)      : {events_per_day:.2f}")
print(f"Rapid events per 100 Callpoints/day     : {events_per_100_callpoints_per_day:.2f}")
print()
print(f"Average days between events/Callpoint   : {days_between_events:.2f} days")
print(f"Daily probability per Callpoint         : {daily_probability * 100:.2f}%")

print("====================================================")

print("\nRecommended 8D Statement:")
print(
    f"Based on {OBSERVATION_DAYS} days of monitoring across "
    f"{usable_battery_powered} battery-powered Callpoints, "
    f"the rapid depletion condition occurs approximately once every "
    f"{days_between_events:.1f} days per device on average. "
    f"The issue is triggered following approximately "
    f"{rapid_prevalence * 100:.1f}% of all Answered Callback events "
    f"(approximately 1 in {1 / rapid_prevalence:.1f} Answered Callbacks)."
)