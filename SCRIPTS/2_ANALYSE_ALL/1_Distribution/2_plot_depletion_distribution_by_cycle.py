#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent.parent

DEPLETION_DIR = BASE_DIR / "2_Parsed_Store" / "4_Daily_Stats" / "1_Depletion"

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
RATE_COLUMN = "weighted_discharge_rate_percent_per_day"
MAX_REASONABLE_RATE = 100
BIN_COUNT = 25
RATED_DEPLETION = 15

# -------------------------------------------------
# LOAD ALL DEPLETION FILES
# -------------------------------------------------
files = sorted(DEPLETION_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError(f"No depletion files found in: {DEPLETION_DIR}")

df_list = []

for file in files:
    tmp = pd.read_csv(file)
    tmp["source_file"] = file.name
    df_list.append(tmp)

df = pd.concat(df_list, ignore_index=True)

if RATE_COLUMN not in df.columns:
    raise ValueError(f"Missing column: {RATE_COLUMN}")

df[RATE_COLUMN] = pd.to_numeric(df[RATE_COLUMN], errors="coerce")

df = df.dropna(subset=[RATE_COLUMN])

df = df[
    (df[RATE_COLUMN] > 0) &
    (df[RATE_COLUMN] < MAX_REASONABLE_RATE)
]

# -------------------------------------------------
# COUNTS
# -------------------------------------------------
unique_ipei_count = df["ipei"].nunique() if "ipei" in df.columns else 0
discharge_cycle_count = len(df)

# -------------------------------------------------
# SUMMARY
# -------------------------------------------------
mean_rate = df[RATE_COLUMN].mean()
median_rate = df[RATE_COLUMN].median()
std_rate = df[RATE_COLUMN].std()

data_min = df[RATE_COLUMN].min()
data_max = df[RATE_COLUMN].max()
bin_width = (data_max - data_min) / BIN_COUNT

pass_count = (df[RATE_COLUMN] < RATED_DEPLETION).sum()
fail_count = (df[RATE_COLUMN] >= RATED_DEPLETION).sum()
total_count = len(df)

pass_percent = (pass_count / total_count) * 100 if total_count else 0
fail_percent = (fail_count / total_count) * 100 if total_count else 0

print("\n----------------------------------------")
print(f"Files loaded            : {len(files)}")
print(f"Unique IPEIs found      : {unique_ipei_count}")
print(f"Discharge cycles plotted: {discharge_cycle_count}")
print(f"Mean depletion          : {mean_rate:.2f} %/day")
print(f"Median depletion        : {median_rate:.2f} %/day")
print(f"Std deviation           : {std_rate:.2f} %/day")
print(f"Bin width               : {bin_width:.2f} %/day")
print("----------------------------------------")
print(f"Passable (<{RATED_DEPLETION:.2f}%/day) : {pass_count} ({pass_percent:.2f}%)")
print(f"Not passable          : {fail_count} ({fail_percent:.2f}%)")
print("----------------------------------------")

# -------------------------------------------------
# PLOT
# -------------------------------------------------
plt.figure(figsize=(9, 6))

plt.hist(
    df[RATE_COLUMN],
    bins=BIN_COUNT,
    edgecolor="black",
)

plt.axvline(
    RATED_DEPLETION,
    linestyle=":",
    linewidth=2,
    color="purple",
    label=f"Rated depletion = {RATED_DEPLETION:.2f}%/day",
)

plt.xlabel("Battery Depletion Rate (% per day)")
plt.ylabel("Discharge Cycle Count")
plt.title("Distribution of Individual Discharge Cycle Depletion Rates")

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()