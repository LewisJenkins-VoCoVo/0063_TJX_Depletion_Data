#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# -------------------------------------------------
# PATH
# -------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent.parent
INPUT_FILE = BASE_DIR / "3_Analysis_Datasets" / "all_stores_usage_analysis.csv"

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
RATE_COLUMN = "mean_discharge_rate_percent_per_day"
MAX_REASONABLE_RATE = 100
RATED_DEPLETION = 15

EVENT_COLUMNS = {
    "Answered Count": ("answered_count", "green"),
    "Cleardown Count": ("cleardown_count", "gold"),
    "Timeout Count": ("timeout_count", "red"),
}

# -------------------------------------------------
# LOAD
# -------------------------------------------------
df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

required_cols = [
    "date",
    "location_name",
    "device_location",
    "ipei",
    RATE_COLUMN,
    "answered_count",
    "cleardown_count",
    "timeout_count",
]

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df[RATE_COLUMN] = pd.to_numeric(df[RATE_COLUMN], errors="coerce")

for event_col, _colour in EVENT_COLUMNS.values():
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")

df = df.dropna(subset=[RATE_COLUMN] + [v[0] for v in EVENT_COLUMNS.values()])

df = df[
    (df[RATE_COLUMN] > 0) &
    (df[RATE_COLUMN] < MAX_REASONABLE_RATE)
]

# -------------------------------------------------
# AGGREGATE PER DEVICE
# -------------------------------------------------
agg = df.groupby("ipei").agg({
    RATE_COLUMN: "mean",
    "answered_count": "sum",
    "cleardown_count": "sum",
    "timeout_count": "sum",
}).reset_index()

agg = agg.dropna()

# -------------------------------------------------
# SUMMARY COUNTS
# -------------------------------------------------
dataset_count = df[["date", "location_name", "device_location", "ipei"]].drop_duplicates().shape[0]
unique_ipei_count = agg["ipei"].nunique()

# -------------------------------------------------
# PLOT
# -------------------------------------------------
fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(9, 10),
    sharex=True,
)

for ax, (title, (event_col, colour)) in zip(axes, EVENT_COLUMNS.items()):
    x = agg[RATE_COLUMN].values
    y = agg[event_col].values

    ax.scatter(x, y, color=colour, label=title)

    if len(x) >= 2 and len(set(x)) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = slope * x + intercept

        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        fit_x = np.linspace(x.min(), x.max(), 100)
        fit_y = slope * fit_x + intercept

        ax.plot(fit_x, fit_y, color="black", linestyle="--", label="Linear fit")

        ax.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
        )

    ax.axvline(
        RATED_DEPLETION,
        linestyle=":",
        linewidth=2,
        color="purple",
        label="Rated depletion (15%/day)",
    )

    ax.set_ylabel(title)
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Battery Depletion Rate (% per day)")

fig.suptitle("Battery Depletion vs Activation Event Counts (Per Device)")
plt.tight_layout()
plt.show()

print("\n----------------------------------------")
print(f"Daily datasets analysed : {dataset_count}")
print(f"Unique IPEIs plotted    : {unique_ipei_count}")
print("----------------------------------------")