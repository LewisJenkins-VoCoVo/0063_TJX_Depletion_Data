#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# USER SETTINGS
# -------------------------------------------------
STORE_SEARCH = "ST_UKTK_0786"
PLOT_ALL_IPEIS = True
TARGET_IPEI = ""  # used only if PLOT_ALL_IPEIS = False

# -------------------------------------------------
# PROJECT FOLDERS
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]

INPUT_DIR = BASE_DIR / "3_Analysis_Datasets" / "1_Single_Store_Analysis"
OUTPUT_DIR = BASE_DIR / "3_Analysis_Datasets" / "1_Single_Store_Analysis" / "Plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def normalise_text(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def safe_filename(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r'[<>:"/\\|?*]', "_", value)
    value = re.sub(r"\s+", "_", value)
    return value if value else "Unknown"


def find_matching_file(folder: Path, pattern: str, search: str) -> Path:
    search_key = normalise_text(search)

    matches = []

    for file in sorted(folder.glob(pattern)):
        if search_key in normalise_text(file.stem):
            matches.append(file)

    if not matches:
        raise FileNotFoundError(f"No matching file found in {folder} for search: {search}")

    if len(matches) > 1:
        print("\nMultiple matches found:")
        for idx, match in enumerate(matches, start=1):
            print(f"{idx}: {match.name}")

        raise ValueError("Search is ambiguous. Make STORE_SEARCH more specific.")

    return matches[0]


def plot_ipei(df: pd.DataFrame, ipei: str, output_file: Path) -> None:
    device_df = df[df["ipei"].astype(str) == str(ipei)].copy()

    soc_df = device_df[device_df["event_type"] == "SOC"].copy()
    act_df = device_df[device_df["event_type"] == "ACTIVATION"].copy()

    # -------------------------------------------------
    # HARD FILTER: no SoC → skip entirely
    # -------------------------------------------------
    soc_df = soc_df.dropna(subset=["battery_level"])

    if soc_df.empty:
        return  # nothing meaningful to plot

    soc_df = soc_df.sort_values(by="event_time")

    # -------------------------------------------------
    # TIME WINDOW FILTER
    # Only keep activations during SoC coverage
    # -------------------------------------------------
    t_min = soc_df["event_time"].min()
    t_max = soc_df["event_time"].max()

    act_df = act_df[
        (act_df["event_time"] >= t_min) &
        (act_df["event_time"] <= t_max)
    ].copy()

    # If you want to be stricter (optional):
    # Require at least 2 SoC points (real trend, not noise)
    if len(soc_df) < 2:
        return

    # -------------------------------------------------
    # PLOT
    # -------------------------------------------------
    device_location = soc_df["device_location"].dropna().iloc[0]
    location_name = soc_df["location_name"].dropna().iloc[0]

    plt.figure(figsize=(12, 6))

    # SoC line
    plt.plot(
        soc_df["event_time"],
        soc_df["battery_level"],
        marker="o",
        linewidth=1,
        label="Battery SoC",
    )

    # -------------------------------------------------
    # ACTIVATION MARKERS (only within SoC window)
    # -------------------------------------------------
    marker_styles = {
        "TIMEOUT": "x",
        "CLEARDOWN": "|",
        "ANSWERED": "^",
    }

    for description, marker in marker_styles.items():
        subset = act_df[
            act_df["activation_description"].astype(str).str.upper() == description
        ]

        if subset.empty:
            continue

        plt.scatter(
            subset["event_time"],
            [105] * len(subset),
            marker=marker,
            label=description,
        )

    plt.ylim(-5, 110)

    plt.xlabel("Time")
    plt.ylabel("Battery SoC (%)")
    plt.title(f"{location_name} | {device_location} | IPEI {ipei}")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_file, dpi=150)
    plt.close()


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main(store_search: str | None = None) -> int:
    store_search = store_search or STORE_SEARCH

    input_file = find_matching_file(
        INPUT_DIR,
        "*_single_store_history.csv",
        store_search,
    )

    df = pd.read_csv(input_file)

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["battery_level"] = pd.to_numeric(df["battery_level"], errors="coerce")

    df = df.dropna(subset=["event_time"])

    if PLOT_ALL_IPEIS:
        ipeis = sorted([
            ipei for ipei in df["ipei"].dropna().astype(str).unique()
            if ipei != "UNKNOWN_IPEI"
        ])
    else:
        ipeis = [TARGET_IPEI]

    print("\n----------------------------------------")
    print(f"Input file     : {input_file.name}")
    print(f"IPEIs to plot  : {len(ipeis)}")
    print(f"Output folder  : {OUTPUT_DIR}")
    print("----------------------------------------\n")

    for idx, ipei in enumerate(ipeis, start=1):
        output_file = OUTPUT_DIR / f"{safe_filename(input_file.stem)}_{safe_filename(ipei)}.png"

        print(f"[{idx}/{len(ipeis)}] Plotting IPEI: {ipei}")
        plot_ipei(df, ipei, output_file)

    print("\nDone.")
    print(f"Plots saved to: {OUTPUT_DIR}")

    return len(ipeis)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot basic single-store SoC history by IPEI.")
    parser.add_argument(
        "--store-search",
        default=STORE_SEARCH,
        help="Store search string used to find the single-store history CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(store_search=args.store_search)