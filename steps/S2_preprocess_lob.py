# S2_preprocess_lob.py

"""
Step 2: Preprocess raw Binance LOB data.

Tasks:
- Load raw LOB snapshots from data/raw.
- Clean and align to a strict 1-second time grid.
- Enforce a fixed number of levels per side (top N bid/ask levels).
- Output a standardized dataset (.npz) ready for sequence building.

Output format
-------------

We save a compressed .npz file at config.paths.preprocessed_lob_file containing:

- timestamps: 1D int64 array of Unix timestamps in seconds (aligned grid).
- lob: 2D float32 array of shape (T, F), where:

    F = 4 * num_levels

Feature order per row:

    [ask_price_1, ..., ask_price_N,
     ask_size_1,  ..., ask_size_N,
     bid_price_1, ..., bid_price_N,\
     bid_size_1,  ..., bid_size_N]

Or if using px_qty naming:
    [ask_px_1, ..., ask_px_N,
     ask_qty_1,  ..., ask_qty_N,
     bid_px_1, ..., bid_px_N,
     bid_qty_1,  ..., bid_qty_N]
"""

import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from config import paths, data_config, label_config
from utils import ensure_dir, simple_logger


def load_and_concat_raw_files(files: List[Path]) -> pd.DataFrame:
    """Load and concatenate raw LOB files into a single DataFrame."""
    dfs = []
    for f in files:
        simple_logger(f"Loading {f}", prefix="S2")
        try:
            if f.suffix.lower() == ".parquet":
                df = pd.read_parquet(f)
            elif f.suffix.lower() == ".csv":
                # For large CSV files, use chunked reading to save memory
                file_size_mb = f.stat().st_size / (1024 * 1024)
                if file_size_mb > 100:  # If file > 100MB, use chunked reading
                    simple_logger(f"Large file ({file_size_mb:.1f} MB), reading in chunks...", prefix="S2")
                    chunk_list = []
                    chunk_size = 100000  # Read 100k rows at a time
                    for chunk in pd.read_csv(f, chunksize=chunk_size):
                        chunk_list.append(chunk)
                    df = pd.concat(chunk_list, ignore_index=True)
                    del chunk_list  # Free memory
                else:
                    df = pd.read_csv(f)
            else:
                raise ValueError(f"Unsupported file extension: {f}")
            dfs.append(df)
        except Exception as e:
            simple_logger(f"Error loading {f}: {e}", prefix="S2")
            raise

    if not dfs:
        raise RuntimeError("No raw files found. Run S1 and add raw data first.")

    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    return df_all


def align_to_1s_grid_chunked(df: pd.DataFrame, chunk_size: int = 100000) -> pd.DataFrame:
    """
    Align snapshots to a strict 1-second grid using chunked processing for memory efficiency.
    
    This function processes the dataframe in chunks to avoid memory issues.
    """
    ts_col = data_config.timestamp_col

    if ts_col not in df.columns:
        raise KeyError(f"timestamp column '{ts_col}' not found in raw data.")

    # Check if timestamp is already datetime type (e.g., from S0_filter_raw_data.py)
    if pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        # Already datetime, ensure it's timezone-aware (UTC)
        try:
            # Check if timezone-aware by trying to access tz attribute
            tz = df[ts_col].dt.tz
            if tz is None:
                df[ts_col] = df[ts_col].dt.tz_localize("UTC")
            else:
                df[ts_col] = df[ts_col].dt.tz_convert("UTC")
        except (AttributeError, TypeError):
            # Fallback: try to localize if naive
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    # Try to detect if timestamp is in ms; if numeric, assume ms and convert
    elif np.issubdtype(df[ts_col].dtype, np.number):
        dt = pd.to_datetime(df[ts_col], unit="ms", utc=True)
        df[ts_col] = dt
    else:
        dt = pd.to_datetime(df[ts_col], utc=True)
        df[ts_col] = dt

    df = df.set_index(ts_col).sort_index()

    # Check alignment using a sample
    sample_size = min(10000, len(df))
    sample_indices = df.index[:sample_size]
    time_diffs = sample_indices.to_series().diff().dt.total_seconds()
    median_diff = time_diffs.median()
    
    if 0.9 <= median_diff <= 1.1:
        # Data is already ~1 second aligned
        simple_logger("Data appears to be already 1-second aligned, skipping resample", prefix="S2")
        # Use both backward and forward fill to handle gaps
        df_resampled = df.bfill().ffill().dropna(how='all')  # Only drop rows where ALL values are NaN
        del df
        return df_resampled
    
    # Need to resample - use chunked approach for large datasets
    simple_logger(f"Resampling data (median interval: {median_diff:.2f}s) in chunks...", prefix="S2")
    
    # Process in time-based chunks to avoid memory issues
    total_rows = len(df)
    chunk_results = []
    
    # Split into time-based chunks
    start_time = df.index[0]
    end_time = df.index[-1]
    chunk_duration = pd.Timedelta(days=1)  # Process 1 day at a time
    
    current_start = start_time
    chunk_num = 0
    
    while current_start < end_time:
        current_end = min(current_start + chunk_duration, end_time)
        chunk = df[(df.index >= current_start) & (df.index < current_end)]
        
        if len(chunk) > 0:
            try:
                # Resample to 1-second grid using 'last' to get the most recent value
                chunk_resampled = chunk.resample(data_config.resample_rule).last()
                
                # Use both forward and backward fill to handle sparse data
                # This ensures we fill gaps from both directions
                chunk_resampled = chunk_resampled.bfill().ffill()
                
                # Only drop rows where ALL values are still NaN (after fill)
                # This preserves rows that have at least some data
                chunk_resampled = chunk_resampled.dropna(how='all')
                
                if len(chunk_resampled) > 0:
                    chunk_results.append(chunk_resampled)
                else:
                    # If chunk is completely empty, it might be a gap in the data
                    # Log occasionally but don't fail
                    if chunk_num % 100 == 0:  # Only log every 100th empty chunk
                        simple_logger(f"Note: Chunk {chunk_num} had no data after resampling (likely gap in data)", prefix="S2")
            except Exception as e:
                simple_logger(f"Error processing chunk {chunk_num}: {e}", prefix="S2")
                # Continue with next chunk instead of failing
            
            chunk_num += 1
            if chunk_num % 10 == 0:
                simple_logger(f"Processed {chunk_num} chunks, {len(chunk_results)} chunks with data...", prefix="S2")
        
        current_start = current_end
    
    del df  # Free memory
    
    # Concatenate results efficiently
    if chunk_results:
        simple_logger(f"Concatenating {len(chunk_results)} chunks...", prefix="S2")
        
        # Concatenate in batches to avoid memory issues
        # Process in batches of 50 chunks at a time
        batch_size = 50
        batched_results = []
        total_batches = (len(chunk_results) + batch_size - 1) // batch_size
        
        for i in range(0, len(chunk_results), batch_size):
            batch = chunk_results[i:i + batch_size]
            if batch:
                batched_df = pd.concat(batch, axis=0)
                batched_results.append(batched_df)
                batch_num = i // batch_size + 1
                if batch_num % 5 == 0 or batch_num == total_batches:
                    simple_logger(f"Concatenated batch {batch_num}/{total_batches}...", prefix="S2")
        
        # Now concatenate the batches (much fewer objects)
        simple_logger(f"Final concatenation of {len(batched_results)} batches...", prefix="S2")
        df_resampled = pd.concat(batched_results, axis=0)
        del chunk_results, batched_results  # Free memory
        
        # Sort by index to ensure proper ordering
        # Check if already sorted to avoid unnecessary sort
        simple_logger("Checking if sorting is needed...", prefix="S2")
        if not df_resampled.index.is_monotonic_increasing:
            simple_logger("Sorting by index...", prefix="S2")
            df_resampled = df_resampled.sort_index()
        else:
            simple_logger("DataFrame already sorted, skipping sort step.", prefix="S2")
        
        # Final forward fill across chunk boundaries
        # For very large DataFrames, process forward fill in chunks to avoid memory issues
        simple_logger("Applying forward fill across chunk boundaries...", prefix="S2")
        n_rows_before = len(df_resampled)
        
        if len(df_resampled) > 5000000:  # > 5M rows
            # Process forward fill in time-based chunks to avoid memory issues
            simple_logger(f"Very large DataFrame ({len(df_resampled):,} rows), processing forward fill in chunks...", prefix="S2")
            
            # Process in weekly chunks
            chunk_duration = pd.Timedelta(days=7)
            start_time = df_resampled.index[0]
            end_time = df_resampled.index[-1]
            current_start = start_time
            
            filled_chunks = []
            chunk_num = 0
            
            while current_start < end_time:
                current_end = min(current_start + chunk_duration, end_time)
                chunk = df_resampled[(df_resampled.index >= current_start) & (df_resampled.index < current_end)]
                
                if len(chunk) > 0:
                    # Forward fill within chunk with limit
                    chunk_filled = chunk.ffill(limit=10)
                    filled_chunks.append(chunk_filled)
                    
                    chunk_num += 1
                    if chunk_num % 10 == 0:
                        simple_logger(f"Forward filled chunk {chunk_num}...", prefix="S2")
                
                current_start = current_end
            
            # Concatenate filled chunks using disk-based approach to avoid memory issues
            simple_logger(f"Concatenating {len(filled_chunks)} filled chunks using disk-based approach...", prefix="S2")
            
            # Write chunks to temporary parquet files, then read back incrementally
            # This avoids holding all data in memory at once
            temp_dir = Path(tempfile.mkdtemp(prefix="s2_lob_"))
            temp_files = []
            
            try:
                # Write each chunk to a temporary parquet file
                for i, chunk in enumerate(filled_chunks):
                    temp_file = temp_dir / f"chunk_{i:04d}.parquet"
                    chunk.to_parquet(temp_file, compression='snappy')
                    temp_files.append(temp_file)
                    del chunk  # Free memory immediately
                    
                    if (i + 1) % 10 == 0:
                        simple_logger(f"Written {i + 1}/{len(filled_chunks)} chunks to disk...", prefix="S2")
                
                del filled_chunks  # Free all chunks from memory
                
                # Drop NaN rows from each chunk BEFORE concatenating to reduce memory usage
                # Then concatenate in smaller batches
                simple_logger("Dropping NaN rows from chunks and concatenating in batches...", prefix="S2")
                
                # Process chunks in batches: drop NaN, concatenate batch, write to intermediate file
                batch_size = 5  # Process 5 chunks at a time
                batch_files = []
                
                for batch_start in range(0, len(temp_files), batch_size):
                    batch_end = min(batch_start + batch_size, len(temp_files))
                    batch = temp_files[batch_start:batch_end]
                    
                    # Read batch chunks, drop NaN, concatenate
                    batch_dfs = []
                    for temp_file in batch:
                        chunk = pd.read_parquet(temp_file)
                        chunk = chunk.dropna(how='all')  # Drop rows with all NaN
                        if len(chunk) > 0:
                            batch_dfs.append(chunk)
                        del chunk
                    
                    if batch_dfs:
                        batch_concat = pd.concat(batch_dfs, axis=0, ignore_index=False)
                        batch_file = temp_dir / f"batch_{batch_start // batch_size:04d}.parquet"
                        batch_concat.to_parquet(batch_file, compression='snappy')
                        batch_files.append(batch_file)
                        del batch_dfs, batch_concat
                    
                    batch_num = batch_start // batch_size + 1
                    total_batches = (len(temp_files) + batch_size - 1) // batch_size
                    if batch_num % 5 == 0 or batch_num == total_batches:
                        simple_logger(f"Processed batch {batch_num}/{total_batches}...", prefix="S2")
                
                # Now concatenate batches (much fewer files)
                simple_logger(f"Concatenating {len(batch_files)} batches...", prefix="S2")
                if len(batch_files) == 1:
                    df_resampled = pd.read_parquet(batch_files[0])
                else:
                    # Read first batch
                    df_resampled = pd.read_parquet(batch_files[0])
                    
                    # Append remaining batches
                    for i, batch_file in enumerate(batch_files[1:], start=2):
                        batch_df = pd.read_parquet(batch_file)
                        df_resampled = pd.concat([df_resampled, batch_df], axis=0, ignore_index=False)
                        del batch_df
                        
                        if i % 3 == 0 or i == len(batch_files):
                            simple_logger(f"Concatenated batch {i}/{len(batch_files)}...", prefix="S2")
                
                # Clean up batch files
                for batch_file in batch_files:
                    if batch_file.exists():
                        batch_file.unlink()
                
            finally:
                # Clean up temporary files
                simple_logger("Cleaning up temporary files...", prefix="S2")
                for temp_file in temp_files:
                    if temp_file.exists():
                        temp_file.unlink()
                try:
                    temp_dir.rmdir()
                except OSError:
                    pass  # Directory might not be empty, that's okay
            
            # Note: NaN rows already dropped during batch processing, so no need to drop again
        elif len(df_resampled) > 500000:
            # For moderately large DataFrames, use limit
            simple_logger(f"Large DataFrame ({len(df_resampled):,} rows), using limited forward fill (limit=5)...", prefix="S2")
            df_resampled = df_resampled.ffill(limit=5).dropna(how='all')
        else:
            df_resampled = df_resampled.ffill().dropna(how='all')
        
        n_rows_after = len(df_resampled)
        simple_logger(f"Forward fill complete: {n_rows_before:,} -> {n_rows_after:,} rows", prefix="S2")
        
        simple_logger(f"Final resampled shape: {df_resampled.shape}", prefix="S2")
    else:
        # More informative error message
        raise RuntimeError(
            f"No data after resampling. Processed {chunk_num} chunks but all were empty. "
            "This might indicate the data is too sparse for 1-second resampling, or there's an issue with the timestamp format."
        )
    
    simple_logger(
        f"Resampled to 1-second grid. New shape: {df_resampled.shape}",
        prefix="S2",
    )

    return df_resampled


def align_to_1s_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align snapshots to a strict 1-second grid.

    Rules:
    - Parse timestamp column to pandas datetime.
    - Set it as index.
    - Resample at 1-second frequency using "last" observation within each second.
    - Forward-fill LOB values to handle seconds with no updates.
    - Drop rows that still have NaNs (if any).
    """
    # Use chunked version for large datasets
    if len(df) > 500000:
        return align_to_1s_grid_chunked(df)
    
    ts_col = data_config.timestamp_col

    if ts_col not in df.columns:
        raise KeyError(f"timestamp column '{ts_col}' not found in raw data.")

    # Check if timestamp is already datetime type (e.g., from S0_filter_raw_data.py)
    if pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        # Already datetime, ensure it's timezone-aware (UTC)
        try:
            # Check if timezone-aware by trying to access tz attribute
            tz = df[ts_col].dt.tz
            if tz is None:
                df[ts_col] = df[ts_col].dt.tz_localize("UTC")
            else:
                df[ts_col] = df[ts_col].dt.tz_convert("UTC")
        except (AttributeError, TypeError):
            # Fallback: try to localize if naive
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    # Try to detect if timestamp is in ms; if numeric, assume ms and convert
    elif np.issubdtype(df[ts_col].dtype, np.number):
        dt = pd.to_datetime(df[ts_col], unit="ms", utc=True)
        df[ts_col] = dt
    else:
        dt = pd.to_datetime(df[ts_col], utc=True)
        df[ts_col] = dt

    df = df.set_index(ts_col).sort_index()

    # Check if data is already roughly 1-second aligned
    sample_size = min(10000, len(df))
    sample_indices = df.index[:sample_size]
    time_diffs = sample_indices.to_series().diff().dt.total_seconds()
    median_diff = time_diffs.median()
    
    if 0.9 <= median_diff <= 1.1:
        simple_logger("Data appears to be already 1-second aligned, skipping resample", prefix="S2")
        # Use both backward and forward fill to handle gaps
        df_resampled = df.bfill().ffill().dropna(how='all')  # Only drop rows where ALL values are NaN
    else:
        simple_logger(f"Resampling data (median interval: {median_diff:.2f}s)", prefix="S2")
        df_resampled = df.resample(data_config.resample_rule).last()
        # Use both backward and forward fill to handle gaps
        df_resampled = df_resampled.bfill().ffill().dropna(how='all')  # Only drop rows where ALL values are NaN
    
    del df

    simple_logger(
        f"Resampled to 1-second grid. New shape: {df_resampled.shape}",
        prefix="S2",
    )

    return df_resampled


def enforce_lob_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce that the DataFrame contains exactly num_levels levels per side.

    Supports both naming conventions:
    - bid_price_i, bid_size_i, ask_price_i, ask_size_i
    - bid_px_i, bid_qty_i, ask_px_i, ask_qty_i
    """
    n = data_config.num_levels
    expected_cols = []

    if data_config.column_naming == "px_qty":
        # Use px/qty naming
        for side in ["bid", "ask"]:
            for field in ["px", "qty"]:
                for level in range(1, n + 1):
                    col = f"{side}_{field}_{level}"
                    expected_cols.append(col)
    else:
        # Use price/size naming
        for side in ["bid", "ask"]:
            for field in ["price", "size"]:
                for level in range(1, n + 1):
                    col = f"{side}_{field}_{level}"
                    expected_cols.append(col)

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required LOB columns: {missing[:10]}... "
            f"Please ensure your raw files have these columns."
        )

    # Restrict to these columns (plus index)
    df_levels = df[expected_cols].copy()
    return df_levels


def build_lob_matrix(df_levels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert level-wise DataFrame into normalized, microstructure-aware features.

    Timestamps: seconds since epoch (UTC).
    Features: normalized feature matrix with:
        - Relative bid/ask prices (centered around mid-price)
        - Log-scaled bid/ask volumes
        - Spread (relative to mid)
        - LOB imbalance
    
    Output shape: (T, 4 * num_levels + 2 + 8) = (T, 4 * num_levels + 10)
    Base features: 4*n + 2 (relative prices, log sizes, spread, imbalance)
    Extra features: 8 (ret_1s, ret_5s, ret_10s, delta_bid_vol, delta_ask_vol, top3_mean, deep_mean, imb_slope)
    
    Returns:
        timestamps: (T,) int64 array of Unix timestamps
        features: (T, F) float32 array of normalized features
        mid_raw: (T,) float64 array of raw mid-prices (before normalization)
    """
    n = data_config.num_levels
    idx = df_levels.index

    # Convert to epoch seconds (int64)
    timestamps = idx.view("int64") // 10**9

    # Extract raw prices and sizes
    if data_config.column_naming == "px_qty":
        ask_price_cols = [f"ask_px_{i}" for i in range(1, n + 1)]
        ask_size_cols = [f"ask_qty_{i}" for i in range(1, n + 1)]
        bid_price_cols = [f"bid_px_{i}" for i in range(1, n + 1)]
        bid_size_cols = [f"bid_qty_{i}" for i in range(1, n + 1)]
    else:
        ask_price_cols = [f"ask_price_{i}" for i in range(1, n + 1)]
        ask_size_cols = [f"ask_size_{i}" for i in range(1, n + 1)]
        bid_price_cols = [f"bid_price_{i}" for i in range(1, n + 1)]
        bid_size_cols = [f"bid_size_{i}" for i in range(1, n + 1)]

    # Extract as numpy arrays
    ask_prices = df_levels[ask_price_cols].to_numpy(dtype=np.float32)  # (T, n)
    ask_sizes = df_levels[ask_size_cols].to_numpy(dtype=np.float32)     # (T, n)
    bid_prices = df_levels[bid_price_cols].to_numpy(dtype=np.float32)   # (T, n)
    bid_sizes = df_levels[bid_size_cols].to_numpy(dtype=np.float32)     # (T, n)

    # Compute raw mid-price BEFORE normalization
    best_bid = bid_prices[:, 0]  # (T,)
    best_ask = ask_prices[:, 0]  # (T,)
    mid_raw = (best_bid + best_ask) / 2.0  # (T,) - raw mid-price before normalization
    
    # Compute normalized mid-price for feature computation
    mid = mid_raw.copy()  # (T,)
    
    # Avoid division by zero
    mid_safe = np.where(mid == 0, 1.0, mid)  # (T,)
    
    # Compute relative prices (centered and scale-free)
    rel_bid_prices = (bid_prices - mid[:, None]) / mid_safe[:, None]  # (T, n)
    rel_ask_prices = (ask_prices - mid[:, None]) / mid_safe[:, None]  # (T, n)
    
    # Compute log-scaled volumes
    log_bid_sizes = np.log1p(bid_sizes)  # log(1 + v), shape (T, n)
    log_ask_sizes = np.log1p(ask_sizes)  # log(1 + v), shape (T, n)
    
    # Spread (relative to mid)
    spread = (best_ask - best_bid) / mid_safe  # (T,)
    spread = spread[:, None]  # (T, 1)
    
    # LOB imbalance over all N levels
    total_bid_vol = bid_sizes.sum(axis=1)  # (T,)
    total_ask_vol = ask_sizes.sum(axis=1)   # (T,)
    denom = total_bid_vol + total_ask_vol
    denom_safe = np.where(denom == 0, 1.0, denom)
    imbalance = total_bid_vol / denom_safe  # (T,)
    imbalance = imbalance[:, None]  # (T, 1)
    
    # Stack all normalized features
    features = np.concatenate(
        [
            rel_bid_prices,   # (T, n)
            rel_ask_prices,   # (T, n)
            log_bid_sizes,    # (T, n)
            log_ask_sizes,    # (T, n)
            spread,           # (T, 1)
            imbalance,        # (T, 1)
        ],
        axis=1,
    )  # shape (T, 4*n + 2)
    
    # ===== ADD RICHER TEMPORAL / MICROSTRUCTURE FEATURES =====
    T = len(mid_raw)
    
    # 1A. Short-term mid-price returns
    ret_1s = np.zeros(T, dtype=np.float32)
    ret_5s = np.zeros(T, dtype=np.float32)
    ret_10s = np.zeros(T, dtype=np.float32)
    
    # 1s return
    denom_1s = mid_raw[:-1]
    denom_1s_safe = np.where(denom_1s == 0, 1.0, denom_1s)
    ret_1s[1:] = (mid_raw[1:] - mid_raw[:-1]) / denom_1s_safe
    
    # 5s return
    if T >= 5:
        denom_5s = mid_raw[:-5]
        denom_5s_safe = np.where(denom_5s == 0, 1.0, denom_5s)
        ret_5s[5:] = (mid_raw[5:] - mid_raw[:-5]) / denom_5s_safe
    
    # 10s return
    if T >= 10:
        denom_10s = mid_raw[:-10]
        denom_10s_safe = np.where(denom_10s == 0, 1.0, denom_10s)
        ret_10s[10:] = (mid_raw[10:] - mid_raw[:-10]) / denom_10s_safe
    
    # Clean NaN/Inf
    ret_1s = np.nan_to_num(ret_1s, nan=0.0, posinf=0.0, neginf=0.0)
    ret_5s = np.nan_to_num(ret_5s, nan=0.0, posinf=0.0, neginf=0.0)
    ret_10s = np.nan_to_num(ret_10s, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 1B. Volume deltas
    total_bid_vol = bid_sizes.sum(axis=1)  # shape (T,)
    total_ask_vol = ask_sizes.sum(axis=1)  # shape (T,)
    
    delta_bid_vol = np.zeros(T, dtype=np.float32)
    delta_ask_vol = np.zeros(T, dtype=np.float32)
    
    delta_bid_vol[1:] = total_bid_vol[1:] - total_bid_vol[:-1]
    delta_ask_vol[1:] = total_ask_vol[1:] - total_ask_vol[:-1]
    
    # 1C. Imbalance slope across levels
    eps = 1e-6
    denom_imb = bid_sizes + ask_sizes + eps
    imb_level = (bid_sizes - ask_sizes) / denom_imb  # shape (T, N)
    
    N = bid_sizes.shape[1]
    if N >= 3:
        top3_mean = imb_level[:, :3].mean(axis=1)
    else:
        top3_mean = imb_level.mean(axis=1)
    
    if N > 3:
        deep_mean = imb_level[:, 3:].mean(axis=1)
    else:
        deep_mean = np.zeros_like(top3_mean)
    
    imb_slope = top3_mean - deep_mean
    
    # 1D. Append all extra features
    extra_features = np.stack(
        [
            ret_1s,
            ret_5s,
            ret_10s,
            delta_bid_vol,
            delta_ask_vol,
            top3_mean.astype(np.float32),
            deep_mean.astype(np.float32),
            imb_slope.astype(np.float32),
        ],
        axis=1,  # shape (T, 8)
    )
    
    features = np.concatenate([features, extra_features], axis=1)  # shape (T, 4*n + 2 + 8)
    
    # Clean non-finite values before normalization
    n_nan = np.isnan(features).sum()
    n_inf = np.isinf(features).sum()
    if n_nan > 0 or n_inf > 0:
        simple_logger(
            f"Found {n_nan} NaNs and {n_inf} Infs in features before normalization; replacing with 0.0",
            prefix="S2",
        )
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        simple_logger("No NaNs/Infs found in features before normalization", prefix="S2")
    
    simple_logger(
        f"Feature matrix shape: {features.shape} (base: {4*n + 2}, extra: 8, total: {features.shape[1]})",
        prefix="S2",
    )
    
    # Apply normalization based on normalization_mode
    # Note: Relative prices, spread, and imbalance are already normalized (relative to mid)
    # We apply z-score normalization to log volumes and extra features
    # For simplicity, we normalize all features together, but could separate LOB vs extra if needed
    normalization_mode = data_config.normalization_mode
    if normalization_mode != "none":  # Allow disabling normalization
        features_normalized, scaler_stats = normalize_features(
            features=features,
            normalization_mode=normalization_mode,
            train_fraction=data_config.normalization_train_fraction,
            scaler_stats=None,  # Will be computed
        )
        features = features_normalized
        
        # Save scaler stats for train_only mode (so we can reuse them)
        if normalization_mode == "train_only" and scaler_stats is not None:
            scaler_stats_path = paths.preprocessed_dir / "scaler_stats.npz"
            np.savez_compressed(
                scaler_stats_path,
                mean=scaler_stats["mean"],
                std=scaler_stats["std"],
            )
            simple_logger(
                f"Saved scaler statistics to {scaler_stats_path}",
                prefix="S2",
            )
    else:
        simple_logger(
            "[S2] Normalization disabled (normalization_mode='none')",
            prefix="S2",
        )
    
    return timestamps.astype(np.int64), features.astype(np.float32), mid_raw.astype(np.float64)


def normalize_features(
    features: np.ndarray,
    normalization_mode: str,
    train_fraction: float = 0.7,
    scaler_stats: dict = None,
) -> tuple[np.ndarray, dict]:
    """
    Normalize features based on normalization_mode.
    
    Args:
        features: Feature matrix (T, F)
        normalization_mode: "static", "train_only", or "rolling"
        train_fraction: Fraction of data to use for fitting scaler (train_only mode)
        scaler_stats: Pre-computed scaler statistics (mean, std) if available
    
    Returns:
        features_normalized: Normalized feature matrix (T, F)
        scaler_stats: Dictionary with 'mean' and 'std' arrays for saving/reuse
    """
    T, F = features.shape
    
    if normalization_mode == "static":
        # Compute mean/std on all data (backward compatible)
        feature_mean = np.mean(features, axis=0, keepdims=True)  # (1, F)
        feature_std = np.std(features, axis=0, keepdims=True)  # (1, F)
        
        # Avoid division by zero
        feature_std = np.where(feature_std == 0, 1.0, feature_std)
        
        features_normalized = (features - feature_mean) / feature_std
        
        scaler_stats = {
            "mean": feature_mean[0, :].astype(np.float32),  # (F,)
            "std": feature_std[0, :].astype(np.float32),  # (F,)
        }
        
        simple_logger(
            f"[S2] Static normalization: computed mean/std on all {T:,} samples",
            prefix="S2",
        )
        
    elif normalization_mode == "train_only":
        # Compute mean/std only on training period, reuse for val/test
        train_end_idx = int(train_fraction * T)
        
        if scaler_stats is None:
            # Fit scaler on training period
            train_features = features[:train_end_idx]  # (T_train, F)
            feature_mean = np.mean(train_features, axis=0, keepdims=True)  # (1, F)
            feature_std = np.std(train_features, axis=0, keepdims=True)  # (1, F)
            
            # Avoid division by zero
            feature_std = np.where(feature_std == 0, 1.0, feature_std)
            
            scaler_stats = {
                "mean": feature_mean[0, :].astype(np.float32),  # (F,)
                "std": feature_std[0, :].astype(np.float32),  # (F,)
            }
            
            simple_logger(
                f"[S2] Train-only normalization: fitted on {train_end_idx:,} samples ({train_fraction:.1%} of data)",
                prefix="S2",
            )
        else:
            # Use provided scaler stats
            feature_mean = scaler_stats["mean"][None, :]  # (1, F)
            feature_std = scaler_stats["std"][None, :]  # (1, F)
            
            simple_logger(
                "[S2] Train-only normalization: using provided scaler statistics",
                prefix="S2",
            )
        
        # Apply normalization to all data using training statistics
        features_normalized = (features - feature_mean) / feature_std
        
    elif normalization_mode == "rolling":
        # TODO: Implement rolling z-score normalization
        # For now, use a simple placeholder: rolling window z-score
        # This is a basic implementation; full rolling normalization may require more sophisticated handling
        window_size = data_config.normalization_rolling_window
        
        simple_logger(
            f"[S2] Rolling normalization (placeholder): using window_size={window_size}",
            prefix="S2",
        )
        
        features_normalized = np.zeros_like(features)
        
        # Simple rolling z-score: compute mean/std in rolling windows
        for i in range(T):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(T, i + window_size // 2 + 1)
            window_features = features[start_idx:end_idx]
            
            window_mean = np.mean(window_features, axis=0, keepdims=True)
            window_std = np.std(window_features, axis=0, keepdims=True)
            window_std = np.where(window_std == 0, 1.0, window_std)
            
            features_normalized[i:i+1] = (features[i:i+1] - window_mean) / window_std
        
        # For rolling mode, we don't save scaler stats (they're time-dependent)
        scaler_stats = None
        
        simple_logger(
            "[S2] WARNING: Rolling normalization is a placeholder implementation. "
            "Consider using 'static' or 'train_only' for production use.",
            prefix="S2",
        )
    else:
        raise ValueError(
            f"Unknown normalization_mode: {normalization_mode}. "
            "Allowed: 'static', 'train_only', 'rolling'"
        )
    
    # Clean non-finite values
    n_nan = np.isnan(features_normalized).sum()
    n_inf = np.isinf(features_normalized).sum()
    if n_nan > 0 or n_inf > 0:
        simple_logger(
            f"[S2] Found {n_nan} NaNs and {n_inf} Infs after normalization; replacing with 0.0",
            prefix="S2",
        )
        features_normalized = np.nan_to_num(features_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features_normalized.astype(np.float32), scaler_stats


def main():
    ensure_dir(paths.preprocessed_dir)

    # Load raw files
    from S1_download_data import list_raw_files

    files = list_raw_files()
    if not files:
        raise RuntimeError(
            "No raw files found in data/raw. "
            "Please add raw BTCUSDT Futures LOB files and rerun."
        )

    df_raw = load_and_concat_raw_files(files)
    simple_logger(f"Combined raw shape: {df_raw.shape}", prefix="S2")

    # Align to 1-second grid
    df_1s = align_to_1s_grid(df_raw)
    del df_raw  # Free memory

    # Enforce fixed levels per side
    df_levels = enforce_lob_levels(df_1s)
    del df_1s  # Free memory

    # Build standardized LOB matrix
    timestamps, lob, mid_raw = build_lob_matrix(df_levels)
    del df_levels  # Free memory

    simple_logger(
        f"Final preprocessed LOB shape: {lob.shape}, timestamps shape: {timestamps.shape}, mid_raw shape: {mid_raw.shape}",
        prefix="S2",
    )

    # Save to .npz
    np.savez_compressed(
        paths.preprocessed_lob_file,
        lob=lob.astype(np.float32),
        mid=mid_raw.astype(np.float64),
        timestamps=timestamps,
    )
    simple_logger(
        f"Saved preprocessed LOB to {paths.preprocessed_lob_file}",
        prefix="S2",
    )


if __name__ == "__main__":
    main()
