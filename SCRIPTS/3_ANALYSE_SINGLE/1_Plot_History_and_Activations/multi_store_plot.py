#!/usr/bin/env python3
"""
Run the single-store plotting pipeline for every ST_UKTK_XXXX prefix found in index.csv.

This script should sit in the same folder as:
- single_store_plot.py
- 1_single_store_history.py
- 2_plot_single_store_history.py
- 3_plot_annotated_history.py

It reads:
- <project root>/0_Index/index.csv

The project root is assumed to be three folders above this script, matching the existing scripts:
    BASE_DIR = Path(__file__).resolve().parents[3]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


# -------------------------------------------------
# USER SETTINGS
# -------------------------------------------------
STORE_PREFIX_REGEX = r"ST_UKTK_\d{4}"

# Useful while testing. Set to None for all stores.
LIMIT_STORES: int | None = None

# Useful if a long run failed part-way through.
START_FROM_STORE = ""

# If True, failed stores are logged and the script continues.
# If False, the script stops on the first failure.
CONTINUE_ON_ERROR = True

RUN_BUILD_HISTORY = True
RUN_BASIC_PLOTS = False
RUN_ANNOTATED_PLOTS = True


# -------------------------------------------------
# PROJECT FILES
# -------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[3]
INDEX_FILE = BASE_DIR / "0_Index" / "index.csv"
SINGLE_STORE_RUNNER = SCRIPT_DIR / "single_store_plot.py"


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def extract_store_prefix(value: object, pattern: str) -> str | None:
    match = re.search(pattern, str(value))
    if not match:
        return None
    return match.group(0)


def load_store_prefixes(index_file: Path, pattern: str) -> list[str]:
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")

    index_df = pd.read_csv(index_file)

    if "location_name" not in index_df.columns:
        raise ValueError("index.csv must contain a 'location_name' column.")

    prefixes = (
        index_df["location_name"]
        .dropna()
        .map(lambda value: extract_store_prefix(value, pattern))
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not prefixes:
        raise ValueError(f"No store prefixes found using regex: {pattern}")

    return prefixes


def apply_start_and_limit(prefixes: list[str], start_from: str, limit: int | None) -> list[str]:
    if start_from:
        if start_from not in prefixes:
            raise ValueError(f"START_FROM_STORE not found in index prefixes: {start_from}")

        start_idx = prefixes.index(start_from)
        prefixes = prefixes[start_idx:]

    if limit is not None:
        prefixes = prefixes[:limit]

    return prefixes


def build_runner_command(store_prefix: str, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(SINGLE_STORE_RUNNER),
        "--store-search",
        store_prefix,
    ]

    if not args.run_history:
        command.append("--skip-history")

    if not args.run_basic_plots:
        command.append("--skip-basic-plots")

    if not args.run_annotated_plots:
        command.append("--skip-annotated-plots")

    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the single-store plotting pipeline for all ST_UKTK_XXXX stores in index.csv."
    )
    parser.add_argument(
        "--index-file",
        default=str(INDEX_FILE),
        help="Path to index.csv. Defaults to the project 0_Index/index.csv.",
    )
    parser.add_argument(
        "--store-prefix-regex",
        default=STORE_PREFIX_REGEX,
        help="Regex used to extract store prefixes from location_name.",
    )
    parser.add_argument(
        "--limit-stores",
        type=int,
        default=LIMIT_STORES,
        help="Optional maximum number of stores to process. Useful for testing.",
    )
    parser.add_argument(
        "--start-from-store",
        default=START_FROM_STORE,
        help="Optional store prefix to start from, e.g. ST_UKTK_0786.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one store fails. Default is to continue and report failures at the end.",
    )
    parser.add_argument(
        "--skip-history",
        dest="run_history",
        action="store_false",
        default=RUN_BUILD_HISTORY,
        help="Skip building single-store history CSV files.",
    )
    parser.add_argument(
        "--skip-basic-plots",
        dest="run_basic_plots",
        action="store_false",
        default=RUN_BASIC_PLOTS,
        help="Skip basic per-IPEI plots.",
    )
    parser.add_argument(
        "--skip-annotated-plots",
        dest="run_annotated_plots",
        action="store_false",
        default=RUN_ANNOTATED_PLOTS,
        help="Skip annotated depletion-cycle plots.",
    )
    return parser.parse_args()


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main() -> None:
    args = parse_args()

    if not SINGLE_STORE_RUNNER.exists():
        raise FileNotFoundError(f"single_store_plot.py not found: {SINGLE_STORE_RUNNER}")

    index_file = Path(args.index_file).expanduser().resolve()

    prefixes = load_store_prefixes(
        index_file=index_file,
        pattern=args.store_prefix_regex,
    )

    prefixes = apply_start_and_limit(
        prefixes=prefixes,
        start_from=str(args.start_from_store).strip(),
        limit=args.limit_stores,
    )

    print("\n----------------------------------------")
    print(f"Index file       : {index_file}")
    print(f"Store regex      : {args.store_prefix_regex}")
    print(f"Stores to process: {len(prefixes)}")
    print(f"Runner           : {SINGLE_STORE_RUNNER.name}")
    print("----------------------------------------\n")

    failed: list[tuple[str, int]] = []
    stop_on_error = args.stop_on_error or not CONTINUE_ON_ERROR

    for idx, store_prefix in enumerate(prefixes, start=1):
        print("\n########################################")
        print(f"[{idx}/{len(prefixes)}] Store: {store_prefix}")
        print("########################################\n")

        command = build_runner_command(store_prefix, args)

        try:
            subprocess.run(command, cwd=SCRIPT_DIR, check=True)
        except subprocess.CalledProcessError as exc:
            failed.append((store_prefix, exc.returncode))
            print(f"\nFAILED: {store_prefix} returned exit code {exc.returncode}\n")

            if stop_on_error:
                raise

    print("\n========================================")
    print("Multi-store run complete")
    print(f"Stores requested : {len(prefixes)}")
    print(f"Stores succeeded : {len(prefixes) - len(failed)}")
    print(f"Stores failed    : {len(failed)}")

    if failed:
        print("\nFailed stores:")
        for store_prefix, return_code in failed:
            print(f"- {store_prefix} / exit code {return_code}")

    print("========================================\n")


if __name__ == "__main__":
    main()
