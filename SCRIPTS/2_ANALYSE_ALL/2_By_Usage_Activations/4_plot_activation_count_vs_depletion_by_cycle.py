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

DOT_SIZE = 32          # Small markers for large datasets
DOT_ALPHA = 0.55      # Transparency helps reveal density

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
    "total_activations",
]

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

for col in [RATE_COLUMN, "total_activations"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[
    RATE_COLUMN,
    "total_activations",
])

# Remove unrealistic depletion values
df = df[
    (df[RATE_COLUMN] > 0) &
    (df[RATE_COLUMN] < MAX_REASONABLE_RATE)
].copy()

# -------------------------------------------------
# SUMMARY COUNTS
# -------------------------------------------------
cycle_count = len(df)
unique_ipei_count = df["ipei"].nunique()
store_count = df["location_name"].nunique()

# -------------------------------------------------
# DATA FOR FIT
# -------------------------------------------------
x = df[RATE_COLUMN].values          # Battery depletion rate (%/day)
y = df["total_activations"].values  # Number of activations

# Linear regression
if len(x) >= 2 and len(set(x)) > 1:
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept

    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
else:
    slope = intercept = r2 = np.nan

# -------------------------------------------------
# PLOT
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

# Scatter plot: one point per individual cycle
ax.scatter(
    x,
    y,
    color="tab:blue",
    alpha=DOT_ALPHA,
    s=DOT_SIZE,
    rasterized=True,
)

# Linear fit
if not np.isnan(slope):
    fit_x = np.linspace(x.min(), x.max(), 100)
    fit_y = slope * fit_x + intercept

    ax.plot(
        fit_x,
        fit_y,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Linear fit",
    )

    ax.text(
        0.05,
        0.95,
        f"R² = {r2:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
    )

# Rated depletion reference
ax.axvline(
    RATED_DEPLETION,
    linestyle=":",
    linewidth=2,
    color="purple",
    label="Rated depletion (15%/day)",
)

# Axis limits
ax.set_xlim(
    min(x.min(), RATED_DEPLETION),
    max(x.max(), RATED_DEPLETION),
)

# Labels
ax.set_xlabel("Battery Depletion Rate (% per day)")
ax.set_ylabel("Total Number of Activations")
ax.set_title("Battery Depletion vs Activation Count (Individual Cycles)")

ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()

# -------------------------------------------------
# SUMMARY
# -------------------------------------------------
print("\n----------------------------------------")
print(f"Individual cycles plotted : {cycle_count}")
print(f"Unique IPEIs              : {unique_ipei_count}")
print(f"Stores included           : {store_count}")

if not np.isnan(r2):
    print(f"R² correlation            : {r2:.4f}")
    print(f"Slope                     : {slope:.4f} activations per %/day")

print("----------------------------------------")