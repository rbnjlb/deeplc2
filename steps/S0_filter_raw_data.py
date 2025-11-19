# S0_filter_raw_data.py

"""
Step 0: Filter raw LOB data to February 2024 only.

This script:
- Loads all existing raw files matching the current raw_file_glob pattern
- Concatenates them into a single DataFrame
- Filters rows to February 2024 only (year=2024, month=2)
- Saves the filtered data as a single parquet file: btcusdt_depth_2024-02_feb_only.parquet

After running this script, update config.py to set:
    raw_file_glob = "btcusdt_depth_2024-02_feb_only.parquet"

Then re-run S2, S3, S4, S5 to rebuild the pipeline on February-only data.

Run:
    python steps/S0_filter_raw_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import paths, data_config
from utils import ensure_dir, simple_logger


def main():
    """Filter raw LOB data to February 2024 only."""
    ensure_dir(paths.raw_data_dir)

    # 1. List all existing raw files using the ORIGINAL glob patterns
    # (before config.py is updated to use the February-only file)
    # Try both parquet and CSV patterns to find all raw files
    parquet_pattern = "btcusdt_depth_*.parquet"
    csv_pattern = "BTCUSDT_lob_*.csv"
    
    raw_files = sorted(paths.raw_data_dir.glob(parquet_pattern))
    if not raw_files:
        simple_logger(
            f"No files matching '{parquet_pattern}', trying CSV pattern '{csv_pattern}'...",
            prefix="S0",
        )
        raw_files = sorted(paths.raw_data_dir.glob(csv_pattern))
    
    # Also try the config patterns as fallback
    if not raw_files:
        raw_files = sorted(paths.raw_data_dir.glob(data_config.raw_file_glob))
    if not raw_files:
        raw_files = sorted(paths.raw_data_dir.glob(data_config.csv_file_glob))
    
    if not raw_files:
        simple_logger(
            f"No raw files found matching common patterns in {paths.raw_data_dir}",
            prefix="S0",
        )
        simple_logger(
            f"Tried: '{parquet_pattern}', '{csv_pattern}', "
            f"'{data_config.raw_file_glob}', '{data_config.csv_file_glob}'",
            prefix="S0",
        )
        simple_logger("Please add raw LOB data first.", prefix="S0")
        return

    simple_logger(f"Found {len(raw_files)} raw file(s) to process", prefix="S0")

    # 2. Load and concatenate all raw files
    dfs = []
    for f in raw_files:
        simple_logger(f"Loading {f.name}...", prefix="S0")
        try:
            if f.suffix.lower() == ".parquet":
                df = pd.read_parquet(f)
            elif f.suffix.lower() == ".csv":
                # For large CSV files, use chunked reading to save memory
                file_size_mb = f.stat().st_size / (1024 * 1024)
                if file_size_mb > 100:  # If file > 100MB, use chunked reading
                    simple_logger(
                        f"Large file ({file_size_mb:.1f} MB), reading in chunks...",
                        prefix="S0",
                    )
                    chunk_list = []
                    chunk_size = 100000  # Read 100k rows at a time
                    for chunk in pd.read_csv(f, chunksize=chunk_size):
                        chunk_list.append(chunk)
                    df = pd.concat(chunk_list, ignore_index=True)
                    del chunk_list  # Free memory
                else:
                    df = pd.read_csv(f)
            else:
                simple_logger(
                    f"Skipping unsupported file extension: {f.suffix}",
                    prefix="S0",
                )
                continue
            
            simple_logger(f"  Loaded {len(df):,} rows from {f.name}", prefix="S0")
            dfs.append(df)
        except Exception as e:
            simple_logger(f"Error loading {f}: {e}", prefix="S0")
            raise

    if not dfs:
        simple_logger("No valid .csv/.parquet files found.", prefix="S0")
        return

    # Concatenate all DataFrames
    simple_logger("Concatenating all raw files...", prefix="S0")
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    simple_logger(f"Combined raw shape: {df_all.shape[0]:,} rows × {df_all.shape[1]} columns", prefix="S0")
    
    # Free memory
    del dfs

    # 3. Parse timestamp column
    ts_col = data_config.timestamp_col
    if ts_col not in df_all.columns:
        raise KeyError(
            f"Timestamp column '{ts_col}' not found in raw data. "
            f"Available columns: {df_all.columns.tolist()}"
        )

    simple_logger(f"Parsing timestamp column '{ts_col}'...", prefix="S0")
    
    # Use the same logic as S2_preprocess_lob.py for consistency
    if np.issubdtype(df_all[ts_col].dtype, np.number):
        # Numeric timestamps: assume milliseconds
        dt = pd.to_datetime(df_all[ts_col], unit="ms", utc=True)
    else:
        # String timestamps: parse directly
        dt = pd.to_datetime(df_all[ts_col], utc=True)

    df_all[ts_col] = dt

    # Show timestamp range
    min_ts = df_all[ts_col].min()
    max_ts = df_all[ts_col].max()
    simple_logger(
        f"Timestamp range: {min_ts} to {max_ts}",
        prefix="S0",
    )

    # 4. Filter to February 2024
    simple_logger("Filtering to February 2024...", prefix="S0")
    mask = (df_all[ts_col].dt.year == 2024) & (df_all[ts_col].dt.month == 2)
    df_feb = df_all.loc[mask].copy()
    
    simple_logger(
        f"Filtered to February 2024: {len(df_feb):,} rows "
        f"({100.0 * len(df_feb) / len(df_all):.2f}% of original)",
        prefix="S0",
    )

    if df_feb.empty:
        simple_logger(
            "WARNING: February filter produced an empty DataFrame. "
            "Check that your data contains February 2024 timestamps.",
            prefix="S0",
        )
        return

    # Show February date range
    feb_min_ts = df_feb[ts_col].min()
    feb_max_ts = df_feb[ts_col].max()
    simple_logger(
        f"February timestamp range: {feb_min_ts} to {feb_max_ts}",
        prefix="S0",
    )

    # Free memory
    del df_all

    # 5. Save to new parquet file
    out_file = paths.raw_data_dir / "btcusdt_depth_2024-02_feb_only.parquet"
    simple_logger(f"Saving February-only data to {out_file}...", prefix="S0")
    
    df_feb.to_parquet(out_file, index=False)
    
    simple_logger(
        f"✓ Saved February-only raw data to {out_file}",
        prefix="S0",
    )
    simple_logger(
        f"  File size: {out_file.stat().st_size / (1024 * 1024):.2f} MB",
        prefix="S0",
    )
    simple_logger(
        f"\nNext steps:",
        prefix="S0",
    )
    simple_logger(
        f"1. Update config.py: set raw_file_glob = 'btcusdt_depth_2024-02_feb_only.parquet'",
        prefix="S0",
    )
    simple_logger(
        f"2. Run: python steps/S2_preprocess_lob.py",
        prefix="S0",
    )
    simple_logger(
        f"3. Run: python steps/S3_build_sequences_and_labels.py",
        prefix="S0",
    )
    simple_logger(
        f"4. Run: python steps/S4_train_model.py",
        prefix="S0",
    )


if __name__ == "__main__":
    main()
