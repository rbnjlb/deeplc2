# S2b_check_data_quality.py

"""
Quick data check script to verify preprocessed LOB data size.

Run this after S2_preprocess_lob.py to confirm the dataset size.
For February-only data, expect ~2-3 million rows, NOT 31 million.

Run:
    python steps/S2b_check_data_quality.py
"""

from pathlib import Path

import numpy as np

from config import paths
from utils import simple_logger


def main():
    """Check preprocessed LOB data size."""
    pre_file = paths.preprocessed_lob_file

    if not pre_file.exists():
        simple_logger("Preprocessed file not found.", prefix="CHK")
        simple_logger(f"Expected at: {pre_file}", prefix="CHK")
        return

    simple_logger(f"Loading preprocessed file: {pre_file}", prefix="CHK")
    data = np.load(pre_file, mmap_mode='r')
    
    lob = data["lob"]
    timestamps = data["timestamps"]
    
    # Check for mid_raw (raw mid-price)
    if "mid" in data:
        mid_raw = data["mid"]
        simple_logger(f"Raw mid-price shape: {mid_raw.shape}", prefix="CHK")
        simple_logger(f"Raw mid-price dtype: {mid_raw.dtype}", prefix="CHK")
        if len(mid_raw) > 0:
            simple_logger(f"Raw mid-price range: {mid_raw.min():.2f} - {mid_raw.max():.2f}", prefix="CHK")
    else:
        simple_logger("WARNING: 'mid' key not found in preprocessed file!", prefix="CHK")

    simple_logger(f"Preprocessed LOB shape: {lob.shape}", prefix="CHK")
    simple_logger(f"Timestamps shape: {timestamps.shape}", prefix="CHK")
    
    if len(timestamps) > 0:
        # Convert Unix timestamps to readable dates
        from datetime import datetime
        first_ts = datetime.fromtimestamp(timestamps[0])
        last_ts = datetime.fromtimestamp(timestamps[-1])
        simple_logger(f"Timestamp range: {first_ts} → {last_ts}", prefix="CHK")
        
        # Calculate duration
        duration_hours = (timestamps[-1] - timestamps[0]) / 3600
        simple_logger(f"Duration: {duration_hours:.1f} hours", prefix="CHK")
    
    # Check if size is reasonable for February data
    num_rows = lob.shape[0]
    if num_rows == 0:
        simple_logger("WARNING: Empty dataset!", prefix="CHK")
    elif num_rows < 100_000:
        simple_logger(f"WARNING: Very small dataset ({num_rows:,} rows). Expected ~2-3M for February.", prefix="CHK")
    elif num_rows > 10_000_000:
        simple_logger(f"WARNING: Very large dataset ({num_rows:,} rows). Expected ~2-3M for February.", prefix="CHK")
        simple_logger("This might be full-year data, not February-only.", prefix="CHK")
    else:
        simple_logger(f"✓ Dataset size looks reasonable for February data: {num_rows:,} rows", prefix="CHK")


if __name__ == "__main__":
    main()
