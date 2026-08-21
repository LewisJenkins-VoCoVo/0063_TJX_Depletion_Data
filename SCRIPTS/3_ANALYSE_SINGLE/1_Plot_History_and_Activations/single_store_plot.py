#!/usr/bin/env python3
"""
Run the full single-store analysis flow from one STORE_SEARCH value.

Order:
1. Build the combined single-store history CSV.
2. Generate the basic per-IPEI SoC plots.
3. Generate the annotated depletion-cycle plots.

Keep this file in the same folder as:
- 1_single_store_history.py
- 2_plot_single_store_history.py
- 3_plot_annotated_history.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# -------------------------------------------------
# USER SETTINGS
# -------------------------------------------------
STORE_SEARCH = "ST_UKTK_0003"  # partial match is OK, e.g. "Oxford", "0786", "Covent"

RUN_BUILD_HISTORY = True
RUN_BASIC_PLOTS = True
RUN_ANNOTATED_PLOTS = True


# -------------------------------------------------
# PROJECT FILES
# -------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

STEPS = [
    (RUN_BUILD_HISTORY, "Build single-store history", SCRIPT_DIR / "1_single_store_history.py"),
    (RUN_BASIC_PLOTS, "Generate basic plots", SCRIPT_DIR / "2_plot_single_store_history.py"),
    (RUN_ANNOTATED_PLOTS, "Generate annotated plots", SCRIPT_DIR / "3_plot_annotated_history.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full single-store analysis flow.")
    parser.add_argument(
        "--store-search",
        default=STORE_SEARCH,
        help="Store search string passed into the single-store scripts.",
    )
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Skip building the combined single-store history CSV.",
    )
    parser.add_argument(
        "--skip-basic-plots",
        action="store_true",
        help="Skip the basic per-IPEI plots.",
    )
    parser.add_argument(
        "--skip-annotated-plots",
        action="store_true",
        help="Skip the annotated depletion-cycle plots.",
    )
    return parser.parse_args()


def run_step(label: str, script_path: Path, store_search: str) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Required script not found: {script_path}")

    print("\n========================================")
    print(label)
    print(f"Script      : {script_path.name}")
    print(f"STORE_SEARCH: {store_search}")
    print("========================================\n")

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--store-search",
            store_search,
        ],
        cwd=SCRIPT_DIR,
        check=True,
    )


def main() -> None:
    args = parse_args()
    store_search = str(args.store_search).strip()

    if not store_search:
        raise ValueError("STORE_SEARCH cannot be blank.")

    steps = [
        (RUN_BUILD_HISTORY and not args.skip_history, "Build single-store history", SCRIPT_DIR / "1_single_store_history.py"),
        (RUN_BASIC_PLOTS and not args.skip_basic_plots, "Generate basic plots", SCRIPT_DIR / "2_plot_single_store_history.py"),
        (RUN_ANNOTATED_PLOTS and not args.skip_annotated_plots, "Generate annotated plots", SCRIPT_DIR / "3_plot_annotated_history.py"),
    ]

    for enabled, label, script_path in steps:
        if not enabled:
            print(f"Skipping: {label}")
            continue

        run_step(label, script_path, store_search)

    print("\nDone. Single-store analysis flow complete.")


if __name__ == "__main__":
    main()
