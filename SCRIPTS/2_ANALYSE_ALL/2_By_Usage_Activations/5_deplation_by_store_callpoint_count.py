#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# PATHS
# -------------------------------------------------
# This script is located in:
# SCRIPTS/2_ANALYSE_ALL/2_By_Usage_Activations/
#
# Therefore:
# Path(__file__).parent          = 2_By_Usage_Activations
# .parent.parent                 = 2_ANALYSE_ALL
# .parent.parent.parent          = SCRIPTS
# .parent.parent.parent.parent   = TJX_Data_Analysis  <-- required
# -------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent.parent

EVENTS_FILE = (
    BASE_DIR
    / "3_Analysis_Datasets"
    / "2_Rapid_Depletion"
    / "Output"
    / "callback_outcomes.csv"
)

INDEX_FILE = (
    BASE_DIR
    / "0_Index"
    / "index.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "3_Analysis_Datasets"
    / "2_Rapid_Depletion"
    / "Output"
)

OUTPUT_CSV = OUTPUT_DIR / "rapid_depletion_vs_store_size.csv"
OUTPUT_PLOT = OUTPUT_DIR / "rapid_depletion_vs_store_size.png"

# -------------------------------------------------
# LOAD
# -------------------------------------------------
events_df = pd.read_csv(EVENTS_FILE)
index_df = pd.read_csv(INDEX_FILE)

# -------------------------------------------------
# FILTER RAPID EVENTS
# -------------------------------------------------
rapid_classes = [
    "suspected_rapid_depletion",
    "severe_rapid_depletion",
]

rapid_df = events_df[
    events_df["trigger_classification"].isin(rapid_classes)
].copy()

# -------------------------------------------------
# STORE SIZE
# -------------------------------------------------
store_sizes = (
    index_df.groupby("location_name")
    .agg(
        total_callpoints=("ipei", "nunique")
    )
    .reset_index()
)

# -------------------------------------------------
# RAPID EVENT COUNTS
# -------------------------------------------------
rapid_counts = (
    rapid_df.groupby("location_name")
    .agg(
        rapid_events=("ipei", "count"),
        affected_ipeis=("ipei", "nunique"),
    )
    .reset_index()
)

# -------------------------------------------------
# MERGE
# -------------------------------------------------
summary = store_sizes.merge(
    rapid_counts,
    on="location_name",
    how="left",
)

summary["rapid_events"] = summary["rapid_events"].fillna(0)
summary["affected_ipeis"] = summary["affected_ipeis"].fillna(0)

# -------------------------------------------------
# NORMALISED METRICS
# -------------------------------------------------
summary["rapid_events_per_callpoint"] = (
    summary["rapid_events"]
    / summary["total_callpoints"]
)

summary["affected_callpoint_percent"] = (
    summary["affected_ipeis"]
    / summary["total_callpoints"]
) * 100

# -------------------------------------------------
# ROUNDING
# -------------------------------------------------
numeric_cols = summary.select_dtypes(include="number").columns

summary[numeric_cols] = (
    summary[numeric_cols]
    .round(3)
)

# -------------------------------------------------
# SAVE
# -------------------------------------------------
summary = summary.sort_values(
    "rapid_events_per_callpoint",
    ascending=False,
)

summary.to_csv(
    OUTPUT_CSV,
    index=False,
)

# -------------------------------------------------
# SCATTER PLOT
# -------------------------------------------------
plot_df = summary.dropna(
    subset=[
        "total_callpoints",
        "rapid_events",
    ]
).copy()

x = plot_df["total_callpoints"]
y = plot_df["rapid_events"]

plt.figure(figsize=(8, 6))

plt.scatter(
    x,
    y,
    alpha=0.75,
)

# -------------------------------------------------
# LINEAR FIT
# -------------------------------------------------
if len(plot_df) >= 2:
    coeffs = np.polyfit(x, y, 1)
    trend = np.poly1d(coeffs)

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = trend(x_line)

    r_squared = np.corrcoef(x, y)[0, 1] ** 2

    plt.plot(
        x_line,
        y_line,
        linewidth=2,
        label=f"Linear fit, R² = {r_squared:.3f}",
    )

    plt.legend()

# -------------------------------------------------
# LABELS
# -------------------------------------------------
plt.xlabel("Store Callpoint Count")
plt.ylabel("Rapid Depletion Events")

plt.title(
    "Rapid Depletion Events vs Store Callpoint Count"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=200,
)

plt.close()

# -------------------------------------------------
# OUTPUT
# -------------------------------------------------
print("\n----------------------------------------")
print("Rapid Depletion vs Store Size")
print("----------------------------------------")
print(f"Stores analysed : {len(summary)}")
print(f"Total rapid events : {int(summary['rapid_events'].sum())}")
print("----------------------------------------")

print(
    summary[
        [
            "location_name",
            "total_callpoints",
            "rapid_events",
            "rapid_events_per_callpoint",
            "affected_callpoint_percent",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("----------------------------------------")
print(f"CSV  : {OUTPUT_CSV}")
print(f"Plot : {OUTPUT_PLOT}")
print("----------------------------------------")