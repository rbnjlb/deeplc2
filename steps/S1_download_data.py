# S1_download_data.py

"""
Step 1: Download or specify raw Binance Futures BTCUSDT LOB data.

This script defines a clear interface and expected file format for raw data.

You have two options:

1. Implement real download logic (e.g., using Binance Futures API or a data vendor).
2. Place your own pre-downloaded .csv / .parquet files into data/raw/ that follow
   the expected column format documented below.

Expected raw file format
------------------------

Each raw file should contain order book snapshots with at least the following columns:

- timestamp: Unix timestamp in milliseconds OR a string parsable by pandas.to_datetime

For each level i = 1..N (N = config.data_config.num_levels), the following columns:

- bid_price_i   (float)  OR  bid_px_i   (depending on column_naming config)
- bid_size_i   (float)  OR  bid_qty_i
- ask_price_i   (float)  OR  ask_px_i
- ask_size_i   (float)  OR  ask_qty_i

Files are expected to be:
    data/raw/btcusdt_depth_YYYY-MM-DD.parquet
or
    data/raw/BTCUSDT_lob_*.csv

You can adjust the naming pattern in config.DataConfig.raw_file_glob.
"""

from pathlib import Path
from typing import List

import pandas as pd

from config import paths, data_config
from utils import ensure_dir, simple_logger


def list_raw_files() -> List[Path]:
    """List all raw LOB files matching the configured glob pattern."""
    raw_dir = paths.raw_data_dir
    ensure_dir(raw_dir)
    
    # Try parquet files first
    files = sorted(raw_dir.glob(data_config.raw_file_glob))
    
    # Also try CSV files
    if not files:
        files = sorted(raw_dir.glob(data_config.csv_file_glob))
    
    return files


def example_fake_downloader() -> None:
    """
    Placeholder for real download logic.

    Currently, this function just prints the expected location and filenames.
    Implement actual logic here if you have Binance credentials or external data source.
    """
    simple_logger(
        f"Raw data directory is at: {paths.raw_data_dir}\n"
        f"Expected file pattern: {data_config.raw_file_glob} or {data_config.csv_file_glob}\n\n"
        "Please place your Binance BTCUSDT Futures LOB files (CSV/Parquet) here.\n"
        "Each file should contain timestamp + bid/ask prices and sizes for each level.\n"
        f"Column naming convention: {data_config.column_naming}\n",
        prefix="S1",
    )


def inspect_sample_file() -> None:
    """
    Inspect the first available raw file and print its columns and head.
    Helps you verify that your file format matches the expectations.
    """
    files = list_raw_files()
    if not files:
        simple_logger(
            "No raw files found. Put your LOB files into data/raw/ and rerun.",
            prefix="S1",
        )
        return

    sample_file = files[0]
    simple_logger(f"Inspecting sample file: {sample_file}", prefix="S1")

    try:
        if sample_file.suffix.lower() == ".parquet":
            df = pd.read_parquet(sample_file)
        elif sample_file.suffix.lower() == ".csv":
            # Try reading first few rows to check size
            df = pd.read_csv(sample_file, nrows=5)
        else:
            simple_logger(
                f"Unsupported file extension for {sample_file}. Use .csv or .parquet.",
                prefix="S1",
            )
            return

        simple_logger(f"Columns ({len(df.columns)}): {list(df.columns)}", prefix="S1")
        simple_logger(f"Shape: {df.shape}", prefix="S1")
        simple_logger(f"Head:\n{df.head()}", prefix="S1")
        
        # Check for expected columns
        if data_config.column_naming == "px_qty":
            expected_prefixes = ["bid_px_", "bid_qty_", "ask_px_", "ask_qty_"]
        else:
            expected_prefixes = ["bid_price_", "bid_size_", "ask_price_", "ask_size_"]
        
        found_cols = []
        for prefix in expected_prefixes:
            matching = [c for c in df.columns if c.startswith(prefix)]
            if matching:
                found_cols.extend(matching[:3])  # Show first 3
        
        if found_cols:
            simple_logger(f"Found expected LOB columns: {found_cols[:10]}...", prefix="S1")
        else:
            simple_logger("Warning: Could not find expected LOB column prefixes!", prefix="S1")
            
    except Exception as e:
        simple_logger(f"Error reading file: {e}", prefix="S1")


if __name__ == "__main__":
    ensure_dir(paths.raw_data_dir)
    example_fake_downloader()
    inspect_sample_file()

