#!/usr/bin/env python3

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

# -------------------------------------------------
# USER SETTINGS
# -------------------------------------------------
STORE_SEARCH = "ST_UKTK_0786"

PLOT_ALL_IPEIS = True
TARGET_IPEI = ""

FILTER_DISCONTINUITY_DROPS = True
DISCONTINUITY_DROP_PERCENT = 40
DISCONTINUITY_MAX_HOURS = 1.25

SHOW_SAMPLE_LABELS = False
SHOW_CYCLE_LABELS = True
SHOW_DAILY_WEIGHTED_RATE_LABELS = True

RATED_DEPLETION_PERCENT_PER_DAY = 15

SHOW_SIGNAL_SUBPLOT = True
SHOW_ACTIVATION_COUNT_SUBPLOT = True
SIGNAL_MIN_DBM = -90
SIGNAL_MAX_DBM = 0

OFFLINE_GAP_HOURS = 1.5

# -------------------------------------------------
# DAY SELECTION / FULL-DAY RULES
# -------------------------------------------------
USE_SELECTED_DATE_RANGE = True

SELECTED_START_DATE = "2026-04-25"
SELECTED_END_DATE = "2026-05-06"

REQUIRE_FULL_SOC_DAYS = False
MIN_FULL_DAY_SOC_HOURS = 22.5

# -------------------------------------------------
# PROJECT FOLDERS
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]

INPUT_DIR = BASE_DIR / "3_Analysis_Datasets" / "1_Single_Store_Analysis"
OUTPUT_BASE_DIR = INPUT_DIR / "Annotated_Plots"

OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)


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

    matches = [
        file for file in sorted(folder.glob(pattern))
        if search_key in normalise_text(file.stem)
    ]

    if not matches:
        raise FileNotFoundError(
            f"No matching file found in {folder} for search: {search}"
        )

    if len(matches) > 1:
        print("\nMultiple matches found:")
        for idx, match in enumerate(matches, start=1):
            print(f"{idx}: {match.name}")
        raise ValueError("Search is ambiguous. Make STORE_SEARCH more specific.")

    return matches[0]


def is_discontinuity_drop(prev_row: dict, curr_row: dict) -> bool:
    if not FILTER_DISCONTINUITY_DROPS:
        return False

    prev_batt = prev_row["battery_level"]
    curr_batt = curr_row["battery_level"]

    if pd.isna(prev_batt) or pd.isna(curr_batt):
        return False

    battery_drop = prev_batt - curr_batt

    time_delta_hours = (
        curr_row["event_time"] - prev_row["event_time"]
    ).total_seconds() / 3600

    return (
        battery_drop >= DISCONTINUITY_DROP_PERCENT
        and time_delta_hours <= DISCONTINUITY_MAX_HOURS
    )


def shade_offline_regions(
    ax,
    soc_df: pd.DataFrame,
    colour: str,
    alpha: float,
    label: str,
) -> None:
    offline_df = soc_df[
        soc_df["status"].astype(str).str.lower() == "offline"
    ].copy()

    if offline_df.empty:
        return

    offline_df = offline_df.sort_values("event_time")

    start_time = None
    prev_time = None
    label_added = False

    for _, row in offline_df.iterrows():
        current_time = row["event_time"]

        if start_time is None:
            start_time = current_time
            prev_time = current_time
            continue

        gap_hours = (current_time - prev_time).total_seconds() / 3600

        if gap_hours > OFFLINE_GAP_HOURS:
            ax.axvspan(
                start_time,
                prev_time,
                color=colour,
                alpha=alpha,
                label=label if not label_added else None,
                zorder=0,
            )
            label_added = True
            start_time = current_time

        prev_time = current_time

    if start_time is not None and prev_time is not None:
        ax.axvspan(
            start_time,
            prev_time,
            color=colour,
            alpha=alpha,
            label=label if not label_added else None,
            zorder=0,
        )


def classify_soc_rows(soc_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    soc_df = soc_df.copy()
    soc_df = soc_df.sort_values("event_time").reset_index(drop=True)

    soc_df["parse_state"] = "raw_unused"
    soc_df["discard_reason"] = ""
    soc_df["cycle_number"] = pd.NA
    soc_df["sample_number"] = pd.NA

    cycles = []
    events = []

    offline_mask = soc_df["status"].astype(str).str.lower() == "offline"
    online_zero_mask = (
        (soc_df["status"].astype(str).str.lower() == "online") &
        (soc_df["battery_level"] == 0)
    )

    soc_df.loc[offline_mask, "parse_state"] = "discarded"
    soc_df.loc[offline_mask, "discard_reason"] = "offline"

    soc_df.loc[online_zero_mask, "parse_state"] = "discarded"
    soc_df.loc[online_zero_mask, "discard_reason"] = "online_0_percent"

    candidate_df = soc_df[soc_df["discard_reason"] == ""].copy()

    if len(candidate_df) < 2:
        return soc_df, cycles, events

    candidate_df["next_battery"] = candidate_df["battery_level"].shift(-1)

    flat_discard_idx = candidate_df[
        (candidate_df["battery_level"] == candidate_df["next_battery"]) &
        (
            (candidate_df["battery_level"] <= 2) |
            (candidate_df["battery_level"] >= 100)
        )
    ].index

    soc_df.loc[flat_discard_idx, "parse_state"] = "discarded"
    soc_df.loc[flat_discard_idx, "discard_reason"] = "flat_duplicate_kept_last"

    candidate_df = candidate_df.drop(index=flat_discard_idx)
    candidate_df = candidate_df.drop(columns=["next_battery"], errors="ignore")

    if len(candidate_df) < 2:
        return soc_df, cycles, events

    candidate_df["date"] = candidate_df["event_time"].dt.date

    global_cycle_number = 1

    for day, day_df in candidate_df.groupby("date"):
        day_df = day_df.sort_values("event_time")

        if not is_selected_date(day):
            continue

        if not is_full_soc_day(day_df, "event_time"):
            continue

        rows = list(day_df.to_dict("records"))
        current_cycle = []

        for idx in range(1, len(rows)):
            prev_row = rows[idx - 1]
            curr_row = rows[idx]

            prev_batt = prev_row["battery_level"]
            curr_batt = curr_row["battery_level"]

            curr_idx = curr_row["_source_index"]

            if is_discontinuity_drop(prev_row, curr_row):
                soc_df.loc[curr_idx, "parse_state"] = "discarded"
                soc_df.loc[curr_idx, "discard_reason"] = "discontinuity_drop"

                events.append({
                    "event_time": curr_row["event_time"],
                    "event_type": "DISCONTINUITY",
                    "label": f"Discontinuity: {prev_batt:.0f}%→{curr_batt:.0f}%",
                })

                current_cycle = []
                continue

            if curr_batt < prev_batt:
                if not current_cycle:
                    current_cycle = [prev_row, curr_row]
                else:
                    current_cycle.append(curr_row)

            elif curr_batt == prev_batt:
                if 2 < curr_batt < 100:
                    if not current_cycle:
                        current_cycle = [prev_row, curr_row]
                    else:
                        current_cycle.append(curr_row)
                continue

            elif curr_batt > prev_batt:
                cycle = finalise_cycle(
                    current_cycle,
                    day,
                    global_cycle_number,
                    soc_df,
                )

                if cycle is not None:
                    cycles.append(cycle)
                    global_cycle_number += 1

                current_cycle = []

        cycle = finalise_cycle(
            current_cycle,
            day,
            global_cycle_number,
            soc_df,
        )

        if cycle is not None:
            cycles.append(cycle)
            global_cycle_number += 1

    return soc_df, cycles, events


def finalise_cycle(
    cycle_rows: list[dict],
    day,
    cycle_number: int,
    soc_df: pd.DataFrame,
) -> dict | None:
    if len(cycle_rows) < 2:
        return None

    cycle_df = pd.DataFrame(cycle_rows).sort_values("event_time")

    start_row = cycle_df.iloc[0]
    end_row = cycle_df.iloc[-1]

    start_batt = start_row["battery_level"]
    end_batt = end_row["battery_level"]

    battery_drop = start_batt - end_batt

    if battery_drop <= 0:
        return None

    duration_hours = (
        end_row["event_time"] - start_row["event_time"]
    ).total_seconds() / 3600

    if duration_hours <= 0:
        return None

    discharge_rate = battery_drop / (duration_hours / 24)

    for sample_number, source_idx in enumerate(cycle_df["_source_index"], start=1):
        soc_df.loc[source_idx, "parse_state"] = "valid_cycle"
        soc_df.loc[source_idx, "cycle_number"] = cycle_number
        soc_df.loc[source_idx, "sample_number"] = sample_number

    return {
        "day": day,
        "cycle_number": cycle_number,
        "start_time": start_row["event_time"],
        "end_time": end_row["event_time"],
        "start_battery": start_batt,
        "end_battery": end_batt,
        "battery_drop": battery_drop,
        "duration_hours": duration_hours,
        "discharge_rate_percent_per_day": discharge_rate,
        "rows": cycle_df,
    }


def get_daily_weighted_rates(cycles: list[dict]) -> dict:
    if not cycles:
        return {}

    cycle_df = pd.DataFrame([
        {
            "day": cycle["day"],
            "battery_drop": cycle["battery_drop"],
            "duration_hours": cycle["duration_hours"],
        }
        for cycle in cycles
    ])

    daily = (
        cycle_df.groupby("day")
        .agg(
            total_battery_drop=("battery_drop", "sum"),
            total_duration_hours=("duration_hours", "sum"),
        )
        .reset_index()
    )

    daily["weighted_rate"] = (
        daily["total_battery_drop"] /
        (daily["total_duration_hours"] / 24)
    )

    return {
        row["day"]: row["weighted_rate"]
        for _, row in daily.iterrows()
    }




def get_daily_activation_counts(act_df: pd.DataFrame) -> pd.DataFrame:
    if act_df.empty:
        return pd.DataFrame(columns=["day", "ANSWERED", "CLEARDOWN", "TIMEOUT", "total"])

    daily_df = act_df.copy()
    daily_df["activation_description"] = (
        daily_df["activation_description"]
        .astype(str)
        .str.upper()
        .str.strip()
    )
    daily_df["day"] = daily_df["event_time"].dt.floor("D")

    counts = (
        daily_df[daily_df["activation_description"].isin(["ANSWERED", "CLEARDOWN", "TIMEOUT"])]
        .groupby(["day", "activation_description"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["ANSWERED", "CLEARDOWN", "TIMEOUT"]:
        if col not in counts.columns:
            counts[col] = 0

    counts["total"] = counts[["ANSWERED", "CLEARDOWN", "TIMEOUT"]].sum(axis=1)
    return counts[["day", "ANSWERED", "CLEARDOWN", "TIMEOUT", "total"]].sort_values("day")


def plot_daily_activation_bars(ax, act_df: pd.DataFrame, daily_rates: dict | None = None) -> None:
    daily_counts = get_daily_activation_counts(act_df)

    ax.set_ylabel("Events / Day")

    if daily_counts.empty:
        ax.text(
            0.5,
            0.5,
            "No activations",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            alpha=0.7,
        )
        ax.set_ylim(0, 1)
        return

    x = mdates.date2num(daily_counts["day"] + pd.Timedelta(hours=12))
    bar_width_days = 0.72

    answered = daily_counts["ANSWERED"].to_numpy()
    cleardown = daily_counts["CLEARDOWN"].to_numpy()
    timeout = daily_counts["TIMEOUT"].to_numpy()

    ax.bar(x, answered, width=bar_width_days, color="green", alpha=0.70, label="ANSWERED")
    ax.bar(x, cleardown, width=bar_width_days, bottom=answered, color="orange", alpha=0.70, label="CLEARDOWN")
    ax.bar(x, timeout, width=bar_width_days, bottom=answered + cleardown, color="red", alpha=0.70, label="TIMEOUT")

    for xpos, total in zip(x, daily_counts["total"]):
        if total <= 0:
            continue
        ax.annotate(
            f"{int(total)}",
            xy=(xpos, total),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_ylim(0, max(1, daily_counts["total"].max() * 1.25))
    ax.grid(True, which="major", axis="y", alpha=0.35)
    ax.grid(True, which="major", axis="x", alpha=0.25)
    ax.legend(loc="upper left", ncols=3, fontsize=7)

    # Daily depletion-rate labels are drawn in the reserved lower area of the SoC plot.
    # Keeping them out of this subplot avoids crowding the event bars.


def configure_time_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%d %b"))

    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))

    ax.tick_params(axis="x", which="major", labelsize=9, pad=5)
    ax.tick_params(axis="x", which="minor", labelsize=7, pad=1)

    ax.grid(True, which="major", axis="x", alpha=0.5)
    ax.grid(True, which="minor", axis="x", color="lightgray", alpha=0.25, linewidth=0.5)

def plot_ipei(df: pd.DataFrame, ipei: str, output_file: Path) -> bool:
    device_df = df[df["ipei"].astype(str) == str(ipei)].copy()

    soc_df = device_df[device_df["event_type"] == "SOC"].copy()
    act_df = device_df[device_df["event_type"] == "ACTIVATION"].copy()

    soc_df = soc_df.dropna(subset=["event_time", "battery_level"])

    if len(soc_df) < 2:
        return False

    soc_df = soc_df.sort_values("event_time").reset_index(drop=True)
    soc_df["_source_index"] = soc_df.index

    parsed_soc_df, cycles, parser_events = classify_soc_rows(soc_df)

    valid_soc_df = parsed_soc_df[parsed_soc_df["parse_state"] == "valid_cycle"].copy()
    discarded_soc_df = parsed_soc_df[parsed_soc_df["parse_state"] == "discarded"].copy()

    if valid_soc_df.empty:
        return False

    t_min = valid_soc_df["event_time"].min()
    t_max = valid_soc_df["event_time"].max()

    act_df = act_df[
        (act_df["event_time"] >= t_min) &
        (act_df["event_time"] <= t_max)
    ].copy()

    device_location = soc_df["device_location"].dropna().iloc[0]
    location_name = soc_df["location_name"].dropna().iloc[0]

    if SHOW_SIGNAL_SUBPLOT and SHOW_ACTIVATION_COUNT_SUBPLOT:
        fig, (ax, ax_activations, ax_signal) = plt.subplots(
            3,
            1,
            figsize=(15, 8.6),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={
                "height_ratios": [4.8, 0.9, 1.15],
                "hspace": 0.04,
            },
        )
    elif SHOW_SIGNAL_SUBPLOT:
        fig, (ax, ax_signal) = plt.subplots(
            2,
            1,
            figsize=(15, 7.6),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={
                "height_ratios": [4.8, 1.15],
                "hspace": 0.04,
            },
        )
        ax_activations = None
    elif SHOW_ACTIVATION_COUNT_SUBPLOT:
        fig, (ax, ax_activations) = plt.subplots(
            2,
            1,
            figsize=(15, 7.8),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={
                "height_ratios": [4.8, 0.9],
                "hspace": 0.04,
            },
        )
        ax_signal = None
    else:
        fig, ax = plt.subplots(figsize=(15, 6.8), constrained_layout=True)
        ax_signal = None
        ax_activations = None

    shade_offline_regions(
        ax,
        soc_df,
        colour="pink",
        alpha=0.18,
        label="Offline region",
    )

    if ax_signal is not None:
        shade_offline_regions(
            ax_signal,
            soc_df,
            colour="pink",
            alpha=0.18,
            label=None,
        )

    if ax_activations is not None:
        shade_offline_regions(
            ax_activations,
            soc_df,
            colour="pink",
            alpha=0.18,
            label=None,
        )

    ax.plot(
        soc_df["event_time"],
        soc_df["battery_level"],
        linewidth=1,
        color="lightgray",
        label="Raw SoC trace",
        zorder=1,
    )

    if not discarded_soc_df.empty:
        ax.scatter(
            discarded_soc_df["event_time"],
            discarded_soc_df["battery_level"],
            color="gray",
            marker="x",
            s=45,
            label="Discarded samples",
            zorder=3,
        )

    for cycle in cycles:
        cycle_rows = cycle["rows"]

        ax.plot(
            cycle_rows["event_time"],
            cycle_rows["battery_level"],
            marker="o",
            linewidth=2,
            label="Valid discharge cycle" if cycle["cycle_number"] == 1 else None,
            zorder=4,
        )

        if SHOW_CYCLE_LABELS:
            mid_time = cycle["start_time"] + (
                cycle["end_time"] - cycle["start_time"]
            ) / 2

            duration_hours = max(float(cycle["duration_hours"]), 1e-9)
            cycle_rate_per_day = float(cycle["battery_drop"]) / duration_hours * 24.0

            # Keep the depletion label in the reserved lower band, away from the SoC data.
            ax.annotate(
                f"{cycle['battery_drop']:.1f}% / {cycle['duration_hours']:.1f}h\n"
                f"{cycle_rate_per_day:.1f}% / day",
                xy=(mid_time, -5.0),
                xytext=(0, 0),
                textcoords="offset points",
                fontsize=8,
                ha="center",
                va="center",
                linespacing=1.15,
                bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="0.35", alpha=0.96),
                zorder=60,
                clip_on=False,
            )

    for event in parser_events:
        if event["event_type"] != "DISCONTINUITY":
            continue

        ax.axvline(
            event["event_time"],
            color="red",
            linestyle=":",
            linewidth=1,
            alpha=0.75,
            label="Discontinuity",
        )

    daily_rates = get_daily_weighted_rates(cycles) if SHOW_DAILY_WEIGHTED_RATE_LABELS else {}

    # -------------------------------------------------
    # ACTIVATION EVENT BAR
    # -------------------------------------------------
    activation_styles = {
        "TIMEOUT": "red",
        "CLEARDOWN": "orange",
        "ANSWERED": "green",
    }

    event_bar_ymin = 108.0
    event_bar_ymax = 111.0

    # Background bar to make the event region visually clear
    ax.axhspan(
        event_bar_ymin,
        event_bar_ymax,
        color="gray",
        alpha=0.30,
        zorder=1,
    )

    for description, colour in activation_styles.items():
        subset = act_df[
            act_df["activation_description"].astype(str).str.upper() == description
            ].copy()

        if subset.empty:
            continue

        for event_time in subset["event_time"]:
            # Light dotted guide line through the SoC plot
            ax.axvline(
                event_time,
                color="dimgray",
                linestyle=":",
                linewidth=1.0,
                alpha=0.95,
                zorder=1,
            )

            # Coloured event marker line in the top event bar only
            ax.vlines(
                event_time,
                event_bar_ymin,
                event_bar_ymax,
                color=colour,
                linewidth=1.6,
                alpha=0.95,
                zorder=6,
                label=description,
            )

    # The previous purple dotted line used RATED_DEPLETION_PERCENT_PER_DAY on the SoC axis.
    # This is intentionally hidden because %/day is not the same unit as battery SoC %.

    ax.set_ylim(-10, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Reported Battery State-of-Charge (SoC) (%)")
    ax.set_title(f"Depletion Analysis | {location_name} | {device_location} | IPEI {ipei}")

    ax.grid(True, which="major", alpha=0.45)
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    configure_time_axis(ax)
    ax.grid(True, which="minor", color="lightgray", alpha=0.25, linewidth=0.5)

    if ax_activations is not None:
        plot_daily_activation_bars(ax_activations, act_df, daily_rates=daily_rates)
        configure_time_axis(ax_activations)
        ax_activations.set_xlabel("")

    ax.set_xlabel("")

    if ax_signal is not None:
        signal_df = soc_df.dropna(subset=["event_time", "signal_level"]).copy()

        if not signal_df.empty:
            ax_signal.plot(
                signal_df["event_time"],
                signal_df["signal_level"],
                linewidth=0.8,
                alpha=0.75,
                label="Signal strength",
            )

        ax_signal.set_ylabel("Signal")
        ax_signal.set_ylim(SIGNAL_MIN_DBM, SIGNAL_MAX_DBM)
        ax_signal.set_xlabel("")

        ax_signal.grid(True, which="major", alpha=0.35)
        ax_signal.yaxis.set_minor_locator(MultipleLocator(5))
        configure_time_axis(ax_signal)
        ax_signal.grid(True, which="minor", color="lightgray", alpha=0.20, linewidth=0.5)


    # Force y-axis labels to sit on a common vertical line across subplots.
    ax.yaxis.set_label_coords(-0.035, 0.5)
    if ax_activations is not None:
        ax_activations.yaxis.set_label_coords(-0.035, 0.5)
    if ax_signal is not None:
        ax_signal.yaxis.set_label_coords(-0.035, 0.5)
    fig.align_ylabels([axis for axis in [ax, ax_activations, ax_signal] if axis is not None])

    handles, labels = ax.get_legend_handles_labels()
    unique = {
        label: handle
        for label, handle in zip(labels, handles)
        if label and label != "Activation event band"
    }
    ax.legend(unique.values(), unique.keys(), loc="best")

    fig.savefig(output_file, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    return True


# -------------------------------------------------
# COMMAND LINE ARGUMENTS
# -------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate annotated depletion-cycle plots for one store.")
    parser.add_argument(
        "--store-search",
        default=STORE_SEARCH,
        help="Store search string. Partial match is OK, e.g. Oxford, 0786, Covent.",
    )
    return parser.parse_args()


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main() -> None:
    args = parse_args()
    store_search = str(args.store_search).strip()

    if not store_search:
        raise ValueError("STORE_SEARCH cannot be blank.")

    input_file = find_matching_file(
        INPUT_DIR,
        "*_single_store_history.csv",
        store_search,
    )

    store_name = safe_filename(input_file.stem.replace("_single_store_history", ""))
    store_output_dir = OUTPUT_BASE_DIR / store_name

    if store_output_dir.exists():
        shutil.rmtree(store_output_dir)

    store_output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
    df["battery_level"] = pd.to_numeric(df["battery_level"], errors="coerce")
    df["signal_level"] = pd.to_numeric(df["signal_level"], errors="coerce")

    df = df.dropna(subset=["event_time"])

    if PLOT_ALL_IPEIS:
        ipeis = sorted([
            ipei for ipei in df["ipei"].dropna().astype(str).unique()
            if ipei != "UNKNOWN_IPEI"
        ])
    else:
        ipeis = [TARGET_IPEI]

    print("\n----------------------------------------")
    print(f"Input file      : {input_file.name}")
    print(f"IPEIs requested : {len(ipeis)}")
    print(f"Output folder   : {store_output_dir}")
    print("----------------------------------------\n")

    plotted_count = 0

    for idx, ipei in enumerate(ipeis, start=1):
        output_file = store_output_dir / f"{safe_filename(ipei)}.png"

        print(f"[{idx}/{len(ipeis)}] Plotting IPEI: {ipei}")

        plotted = plot_ipei(df, ipei, output_file)

        if plotted:
            plotted_count += 1
        else:
            print(" -> Skipped; no valid depletion cycle available")

    print("\nDone.")
    print(f"Plots written : {plotted_count}")
    print(f"Plots saved to: {store_output_dir}")


if __name__ == "__main__":
    main()