# S3_build_sequences_and_labels.py

"""
Step 3: Build sequences and labels for DeepLOB training (SHARDED VERSION, BINARY LABELS).

Tasks:
- Load preprocessed LOB (timestamps, lob, mid) from .npz.
- Load raw mid-price from S2 output (not normalized).
- Compute future mid-price returns r_t = (m_{t+k} - m_t) / m_t.
- Determine return threshold α automatically using quantile.
- Keep only "large movement" samples (|r_t| > α).
- Build sequences of length L (label_config.seq_len).
- Write sequences to SHARDED files.

Labeling rule (Binary Up/Down)
-------------------------------

Let:
    m_t      = raw mid-price at time t
    m_{t+k}  = raw mid-price at time t + k (k = LABEL_HORIZON)

Define relative return:
    r_t = (m_{t+k} - m_t) / m_t

Determine threshold:
    α = quantile(|returns|, LABEL_ALPHA_QUANTILE)

Keep only samples with |r_t| > α:
    Up   = r_t >  α  -> label 1
    Down = r_t < -α  -> label 0

We map labels to {0, 1} for PyTorch CrossEntropyLoss (binary classification).

Output
------

We save SHARDED .npz files in paths.sequences_dir:
- train_shard_000.npz, train_shard_001.npz, ...
- val_shard_000.npz, ...
- test_shard_000.npz, ...

Each shard contains:
- X: (N_shard, L, F)
- y: (N_shard,) with values in {0, 1}

And a shard_index.json file listing all shards and metadata.
"""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from config import paths, data_config, label_config, model_config
from utils import ensure_dir, simple_logger

# Print progress every N shards to reduce console noise
PRINT_SHARD_EVERY = 30  # print progress every 30 shards

def compute_returns_binary_alpha(
    mid: np.ndarray,
    k: int,
    alpha_quantile: float,
    min_samples_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute future mid-price returns and filter to keep only large movements (binary_alpha mode).
    
    Args:
        mid: Raw mid-price array (T,)
        k: Future horizon in seconds
        alpha_quantile: Quantile for threshold (e.g., 0.90 for top 10%)
        min_samples_per_class: Minimum samples required per class
    
    Returns:
        returns_filtered: Filtered returns array
        indices_filtered: Original time indices corresponding to filtered returns
        labels: Binary labels (0=Down, 1=Up)
    """
    T = len(mid)
    
    # Compute returns: r_t = (m_{t+k} - m_t) / m_t
    # for all t where t + k < T
    mid_t = mid[:-k]  # (T-k,)
    mid_tk = mid[k:]  # (T-k,)
    
    # Avoid division by zero
    mid_t_safe = np.where(mid_t == 0, 1.0, mid_t)
    returns = (mid_tk - mid_t) / mid_t_safe  # (T-k,)
    
    # Original time indices (sequence end positions)
    indices = np.arange(0, T - k)  # (T-k,)
    
    # Determine return threshold α automatically
    abs_r = np.abs(returns)
    
    # 2A. More diagnostics on returns and alpha
    simple_logger(
        f"[S3] returns stats: mean={returns.mean():.2e}, std={returns.std():.2e}, "
        f"min={returns.min():.2e}, max={returns.max():.2e}",
        prefix="S3",
    )
    
    for q in [0.5, 0.9, 0.95, 0.99]:
        simple_logger(
            f"[S3] |returns| quantile {q:.2f}: {np.quantile(abs_r, q):.2e}",
            prefix="S3",
        )
    
    alpha = np.quantile(abs_r, alpha_quantile)
    simple_logger(
        f"[S3] Using LABEL_ALPHA_QUANTILE={alpha_quantile}, alpha={alpha:.2e}",
        prefix="S3",
    )
    simple_logger(
        f"[S3] Fraction of points with |r_t| > alpha: {float((abs_r > alpha).mean()):.2%}",
        prefix="S3",
    )
    
    # Keep only "large movement" samples: |r_t| > α
    mask = abs_r > alpha
    returns_filtered = returns[mask]
    indices_filtered = indices[mask]
    
    # Generate binary labels: Up=1 if r_t > α, Down=0 if r_t < -α
    labels = (returns_filtered > 0).astype(np.int64)  # Up=1, Down=0
    
    # Sanity-check label counts
    down = int(np.sum(labels == 0))
    up = int(np.sum(labels == 1))
    
    simple_logger(
        f"[S3] Filtered samples: {len(labels):,} total (Down={down:,}, Up={up:,})",
        prefix="S3",
    )
    
    if down < min_samples_per_class or up < min_samples_per_class:
        raise ValueError(
            f"Not enough samples: down={down}, up={up}. "
            "Consider lowering LABEL_ALPHA_QUANTILE or adjusting LABEL_HORIZON."
        )
    
    # 2B. Optional label balancing before split (binary_alpha mode only)
    from config import label_config
    
    if label_config.label_mode == "binary_alpha" and label_config.label_balance_enabled:
        down_idx = np.where(labels == 0)[0]
        up_idx = np.where(labels == 1)[0]
        
        down_count = len(down_idx)
        up_count = len(up_idx)
        
        simple_logger(
            f"[S3] BEFORE balancing: Down={down_count}, Up={up_count}",
            prefix="S3",
        )
        
        majority_label = 0 if down_count > up_count else 1
        minority_label = 1 - majority_label
        
        majority_idx_all = down_idx if majority_label == 0 else up_idx
        minority_idx_all = up_idx if majority_label == 1 else down_idx
        
        majority_count = len(majority_idx_all)
        minority_count = len(minority_idx_all)
        
        if (
            majority_count > label_config.label_balance_max_ratio * minority_count
            and minority_count > 0
        ):
            target_majority = int(
                label_config.label_balance_max_ratio * minority_count
            )
            np.random.seed(42)  # For reproducibility
            keep_majority_idx = np.random.choice(
                majority_idx_all, size=target_majority, replace=False
            )
            keep_indices = np.concatenate([keep_majority_idx, minority_idx_all])
            keep_indices.sort()
            
            indices_filtered = indices_filtered[keep_indices]
            labels = labels[keep_indices]
            returns_filtered = returns_filtered[keep_indices]
            
            simple_logger(
                f"[S3] AFTER balancing: Down={np.sum(labels==0)}, Up={np.sum(labels==1)}",
                prefix="S3",
            )
        else:
            simple_logger(
                "[S3] Balancing not applied; ratio within threshold.",
                prefix="S3",
            )
    
    return returns_filtered, indices_filtered, labels


def compute_returns_three_class(
    mid: np.ndarray,
    k: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute future mid-price returns and assign three-class labels (three_class mode).
    
    Args:
        mid: Raw mid-price array (T,)
        k: Future horizon in seconds
        threshold: Symmetric threshold for three-class classification
    
    Returns:
        returns: All returns array (no filtering)
        indices: Original time indices corresponding to returns
        labels: Three-class labels (0=Down, 1=Stationary, 2=Up)
    """
    T = len(mid)
    
    # Compute returns: r_t = (m_{t+k} - m_t) / m_t
    # for all t where t + k < T
    mid_t = mid[:-k]  # (T-k,)
    mid_tk = mid[k:]  # (T-k,)
    
    # Avoid division by zero
    mid_t_safe = np.where(mid_t == 0, 1.0, mid_t)
    returns = (mid_tk - mid_t) / mid_t_safe  # (T-k,)
    
    # Original time indices (sequence end positions)
    indices = np.arange(0, T - k)  # (T-k,)
    
    # Generate three-class labels using symmetric threshold
    # Up = 2 if r_t > +threshold
    # Down = 0 if r_t < -threshold
    # Stationary = 1 otherwise
    labels = np.ones(len(returns), dtype=np.int64)  # Default: Stationary (1)
    labels[returns > threshold] = 2  # Up
    labels[returns < -threshold] = 0  # Down
    
    # Log label distribution
    down_count = int(np.sum(labels == 0))
    stationary_count = int(np.sum(labels == 1))
    up_count = int(np.sum(labels == 2))
    
    simple_logger(
        f"[S3] Three-class labels: Down={down_count:,}, Stationary={stationary_count:,}, Up={up_count:,}",
        prefix="S3",
    )
    simple_logger(
        f"[S3] Using threshold={threshold:.2e} for three-class classification",
        prefix="S3",
    )
    
    # Log return statistics
    simple_logger(
        f"[S3] Returns stats: mean={returns.mean():.2e}, std={returns.std():.2e}, "
        f"min={returns.min():.2e}, max={returns.max():.2e}",
        prefix="S3",
    )
    
    return returns, indices, labels


def compute_returns_and_filter(
    mid: np.ndarray,
    k: int,
    label_mode: str,
    alpha_quantile: float = None,
    min_samples_per_class: int = None,
    three_class_threshold: float = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute future mid-price returns and generate labels based on label_mode.
    
    Args:
        mid: Raw mid-price array (T,)
        k: Future horizon in seconds
        label_mode: "binary_alpha" or "three_class"
        alpha_quantile: Quantile for threshold (binary_alpha mode only)
        min_samples_per_class: Minimum samples required per class (binary_alpha mode only)
        three_class_threshold: Symmetric threshold for three_class mode
    
    Returns:
        returns: Returns array (filtered for binary_alpha, all for three_class)
        indices: Original time indices corresponding to returns
        labels: Labels (binary: 0=Down, 1=Up) or (three_class: 0=Down, 1=Stationary, 2=Up)
    """
    if label_mode == "binary_alpha":
        if alpha_quantile is None or min_samples_per_class is None:
            raise ValueError("alpha_quantile and min_samples_per_class required for binary_alpha mode")
        return compute_returns_binary_alpha(mid, k, alpha_quantile, min_samples_per_class)
    elif label_mode == "three_class":
        if three_class_threshold is None:
            raise ValueError("three_class_threshold required for three_class mode")
        return compute_returns_three_class(mid, k, three_class_threshold)
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}. Allowed: 'binary_alpha', 'three_class'")


def count_valid_sequences(
    indices_filtered: np.ndarray,
    seq_len: int,
) -> int:
    """
    Count how many valid sequences we can build from filtered indices.
    
    A sequence is valid if t >= seq_len - 1 (so we can extract [t-seq_len+1, t+1]).
    """
    valid_count = 0
    for t in indices_filtered:
        if t >= seq_len - 1:
            valid_count += 1
    
    return valid_count


def build_sequences_sharded(
    lob: np.ndarray,
    indices_filtered: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
    max_seqs_per_shard: int,
    train_frac: float,
    val_frac: float,
):
    """
    Build sequences and write them to sharded files.
    
    This function:
    1. Iterates over filtered indices (large movements only)
    2. Builds sequences and labels
    3. Assigns each sequence to train/val/test based on global index
    4. Writes shards when they reach max_seqs_per_shard
    5. Returns shard index information
    
    Args:
        lob: Normalized LOB features (T, F)
        indices_filtered: Time indices for filtered samples (large movements)
        labels: Binary labels (0=Down, 1=Up) corresponding to indices_filtered
        seq_len: Sequence length L
        max_seqs_per_shard: Maximum sequences per shard file
        train_frac: Training fraction
        val_frac: Validation fraction
    
    Returns:
        shard_index: dict with keys "train", "val", "test", "seq_len", "num_features"
    """
    T, F = lob.shape
    
    # Count valid sequences (require t >= seq_len - 1)
    simple_logger("Counting valid sequences...", prefix="S3")
    total_valid = count_valid_sequences(indices_filtered, seq_len)
    simple_logger(f"Total valid sequences: {total_valid:,}", prefix="S3")
    
    if total_valid == 0:
        raise ValueError("No valid sequences found!")
    
    # Calculate split boundaries
    train_end_idx = int(math.floor(train_frac * total_valid))
    val_end_idx = train_end_idx + int(math.floor(val_frac * total_valid))
    
    simple_logger(
        f"Split boundaries: train [0, {train_end_idx}), "
        f"val [{train_end_idx}, {val_end_idx}), "
        f"test [{val_end_idx}, {total_valid})",
        prefix="S3",
    )
    
    # Initialize shard buffers and counters
    shard_X = []
    shard_y = []
    global_seq_idx = 0
    
    # Shard counters per split
    train_shard_idx = 0
    val_shard_idx = 0
    test_shard_idx = 0
    
    # Lists to track shard filenames and sizes
    train_shards = []
    val_shards = []
    test_shards = []
    train_shard_sizes = []
    val_shard_sizes = []
    test_shard_sizes = []
    
    current_split = None  # "train", "val", or "test"
    
    # Track label distribution
    label_counter = Counter()
    
    # Track total shards written for progress logging
    total_shards_written = 0
    
    def flush_shard(split_name: str, shard_idx: int):
        """Write current shard buffer to disk and reset."""
        nonlocal total_shards_written
        if not shard_X:
            return
        
        X_arr = np.array(shard_X, dtype=np.float32)
        y_arr = np.array(shard_y, dtype=np.int64)
        
        # Clean non-finite values in sequences
        n_nan = np.isnan(X_arr).sum()
        n_inf = np.isinf(X_arr).sum()
        if n_nan > 0 or n_inf > 0:
            simple_logger(
                f"Cleaning {n_nan} NaNs and {n_inf} Infs in shard {split_name}_{shard_idx:03d}",
                prefix="S3",
            )
            X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure labels are valid integers
        # For binary_alpha: {0, 1}
        # For three_class: {0, 1, 2}
        if label_config.label_mode == "binary_alpha":
            y_arr = np.clip(y_arr, 0, 1).astype(np.int64)
        elif label_config.label_mode == "three_class":
            y_arr = np.clip(y_arr, 0, 2).astype(np.int64)
        else:
            raise ValueError(f"Unknown label_mode: {label_config.label_mode}")
        
        shard_size = len(X_arr)
        
        shard_filename = f"{split_name}_shard_{shard_idx:03d}.npz"
        shard_path = paths.sequences_dir / shard_filename
        
        np.savez_compressed(shard_path, X=X_arr, y=y_arr)
        
        if split_name == "train":
            train_shards.append(shard_filename)
            train_shard_sizes.append(shard_size)
        elif split_name == "val":
            val_shards.append(shard_filename)
            val_shard_sizes.append(shard_size)
        else:
            test_shards.append(shard_filename)
            test_shard_sizes.append(shard_size)
        
        # Clear buffers
        shard_X.clear()
        shard_y.clear()
        
        # Increment total shards written
        total_shards_written += 1
        
        # Per-shard write messages removed to reduce console noise
        # Progress is printed every PRINT_SHARD_EVERY shards instead
    
    # Build sequences only for large-move indices
    simple_logger(
        f"Building sequences and writing shards (max_seqs_per_shard={max_seqs_per_shard:,})...",
        prefix="S3",
    )
    
    for idx, t in enumerate(indices_filtered):
        # Sequence covers t-L+1 to t, so require t >= L-1
        if t < seq_len - 1:
            continue
        
        start = t - seq_len + 1
        end = t + 1
        
        if end > T:
            continue
        
        X_seq = lob[start:end]  # (seq_len, num_features)
        y = labels[idx]
        
        # Determine which split this sequence belongs to
        if global_seq_idx < train_end_idx:
            split_name = "train"
            shard_idx = train_shard_idx
        elif global_seq_idx < val_end_idx:
            split_name = "val"
            shard_idx = val_shard_idx
        else:
            split_name = "test"
            shard_idx = test_shard_idx
        
        # If we switched splits, flush the previous shard
        if current_split is not None and current_split != split_name:
            if current_split == "train":
                flush_shard("train", train_shard_idx)
                train_shard_idx += 1
            elif current_split == "val":
                flush_shard("val", val_shard_idx)
                val_shard_idx += 1
            else:
                flush_shard("test", test_shard_idx)
                test_shard_idx += 1
            
            # Progress logging: print every PRINT_SHARD_EVERY shards
            if total_shards_written % PRINT_SHARD_EVERY == 0:
                simple_logger(
                    f"Processed {idx + 1:,}/{len(indices_filtered):,} filtered indices, "
                    f"{global_seq_idx:,} sequences, "
                    f"{len(train_shards)} train shards, {len(val_shards)} val shards, {len(test_shards)} test shards...",
                    prefix="S3",
                )
        
        # Add to current shard buffer
        shard_X.append(X_seq)
        shard_y.append(y)
        
        # Track label distribution
        label_counter[y] += 1
        
        # If shard is full, flush it
        if len(shard_X) >= max_seqs_per_shard:
            flush_shard(split_name, shard_idx)
            if split_name == "train":
                train_shard_idx += 1
            elif split_name == "val":
                val_shard_idx += 1
            else:
                test_shard_idx += 1
            
            # Progress logging: print every PRINT_SHARD_EVERY shards
            if total_shards_written % PRINT_SHARD_EVERY == 0:
                simple_logger(
                    f"Processed {idx + 1:,}/{len(indices_filtered):,} filtered indices, "
                    f"{global_seq_idx:,} sequences, "
                    f"{len(train_shards)} train shards, {len(val_shards)} val shards, {len(test_shards)} test shards...",
                    prefix="S3",
                )
        
        current_split = split_name
        global_seq_idx += 1
    
    # Flush final shard
    if shard_X:
        if current_split == "train":
            flush_shard("train", train_shard_idx)
        elif current_split == "val":
            flush_shard("val", val_shard_idx)
        else:
            flush_shard("test", test_shard_idx)
        
        # Print progress if we haven't printed recently (for final shard)
        total_expected_shards = math.ceil(total_valid / max_seqs_per_shard) if max_seqs_per_shard > 0 else 0
        if total_shards_written % PRINT_SHARD_EVERY != 0:
            simple_logger(
                f"Processed {len(indices_filtered):,}/{len(indices_filtered):,} filtered indices, "
                f"{global_seq_idx:,} sequences, "
                f"{len(train_shards)} train shards, {len(val_shards)} val shards, {len(test_shards)} test shards...",
                prefix="S3",
            )
    
    # Build shard index (sizes already tracked during shard writing)
    shard_index = {
        "train": train_shards,
        "val": val_shards,
        "test": test_shards,
        "train_sizes": train_shard_sizes,
        "val_sizes": val_shard_sizes,
        "test_sizes": test_shard_sizes,
        "seq_len": seq_len,
        "num_features": F,
        "total_sequences": global_seq_idx,
    }
    
    simple_logger(
        f"Built {global_seq_idx:,} sequences in "
        f"{len(train_shards)} train, {len(val_shards)} val, {len(test_shards)} test shards",
        prefix="S3",
    )
    
    # Log label histogram
    label_hist = dict(sorted(label_counter.items()))
    simple_logger(
        f"Label histogram: {label_hist} (Down=0, Up=1)",
        prefix="S3",
    )
    
    # Final summary
    num_train_shards = len(train_shards)
    num_val_shards = len(val_shards)
    num_test_shards = len(test_shards)
    simple_logger(
        f"[S3 DONE] Total sequences: {global_seq_idx:,} | "
        f"Train shards: {num_train_shards}, Val shards: {num_val_shards}, Test shards: {num_test_shards}",
        prefix="S3",
    )
    
    return shard_index


def main():
    ensure_dir(paths.sequences_dir)
    
    # Load preprocessed LOB using memory-mapped mode
    if not paths.preprocessed_lob_file.exists():
        raise RuntimeError(
            f"Preprocessed LOB file not found: {paths.preprocessed_lob_file}. "
            "Run S2_preprocess_lob.py first."
        )
    
    simple_logger("Loading preprocessed LOB (memory-mapped mode)...", prefix="S3")
    data = np.load(paths.preprocessed_lob_file, mmap_mode='r')
    
    # Check for required keys
    if "lob" not in data:
        raise KeyError("'lob' key not found in preprocessed file. Run S2_preprocess_lob.py first.")
    if "mid" not in data:
        raise KeyError("'mid' key not found in preprocessed file. Run S2_preprocess_lob.py first.")
    
    lob_shape = data["lob"].shape
    simple_logger(f"Preprocessed LOB shape: {lob_shape}", prefix="S3")
    
    # Load raw mid-price from S2 output (not normalized)
    simple_logger("Loading raw mid-price...", prefix="S3")
    mid = np.array(data["mid"])  # Raw mid-price
    simple_logger(f"Raw mid-price shape: {mid.shape}, dtype: {mid.dtype}", prefix="S3")
    
    # For very large datasets, load LOB in chunks
    if lob_shape[0] > 10_000_000:
        simple_logger(
            f"Very large dataset ({lob_shape[0]:,} rows), using chunked processing...",
            prefix="S3",
        )
        lob_mmap = data["lob"]
        
        chunk_size = 1_000_000
        lob_chunks = []
        
        for i in range(0, lob_shape[0], chunk_size):
            end_idx = min(i + chunk_size, lob_shape[0])
            chunk = np.array(lob_mmap[i:end_idx])
            lob_chunks.append(chunk)
            if (i // chunk_size + 1) % 5 == 0:
                simple_logger(f"Loaded chunk {i // chunk_size + 1}...", prefix="S3")
        
        simple_logger("Concatenating chunks...", prefix="S3")
        lob = np.concatenate(lob_chunks, axis=0)
        del lob_chunks, lob_mmap
    else:
        lob = np.array(data["lob"])
    
    del data
    
    # Compute returns and generate labels based on label_mode
    simple_logger(f"Computing returns and labels (mode: {label_config.label_mode})...", prefix="S3")
    returns_filtered, indices_filtered, labels = compute_returns_and_filter(
        mid=mid,
        k=label_config.label_horizon,
        label_mode=label_config.label_mode,
        alpha_quantile=label_config.label_alpha_quantile if label_config.label_mode == "binary_alpha" else None,
        min_samples_per_class=label_config.label_min_samples_per_class if label_config.label_mode == "binary_alpha" else None,
        three_class_threshold=label_config.three_class_threshold if label_config.label_mode == "three_class" else None,
    )
    
    # Log label distribution
    from collections import Counter
    label_counts = Counter(labels)
    simple_logger(
        f"[S3] Label distribution: {dict(sorted(label_counts.items()))}",
        prefix="S3",
    )
    
    # For binary_alpha mode, check class balance
    if label_config.label_mode == "binary_alpha":
        down_count = label_counts.get(0, 0)
        up_count = label_counts.get(1, 0)
        if down_count > 0 and up_count > 0:
            imbalance_ratio = max(down_count, up_count) / min(down_count, up_count)
            simple_logger(
                f"[S3] Class imbalance ratio: {imbalance_ratio:.2f} (max/min)",
                prefix="S3",
            )
            if imbalance_ratio > 5.0:
                simple_logger(
                    "[S3] WARNING: High class imbalance detected. Consider adjusting LABEL_ALPHA_QUANTILE or enabling label balancing.",
                    prefix="S3",
                )
    
    # Build sequences and write to shards
    shard_index = build_sequences_sharded(
        lob=lob,
        indices_filtered=indices_filtered,
        labels=labels,
        seq_len=label_config.seq_len,
        max_seqs_per_shard=label_config.max_sequences_per_shard,
        train_frac=label_config.train_frac,
        val_frac=label_config.val_frac,
    )
    
    # Free data
    del lob, mid, returns_filtered, indices_filtered, labels
    import gc
    gc.collect()
    
    # Write shard index JSON file
    simple_logger("Writing shard index...", prefix="S3")
    with open(paths.shard_index_file, "w") as f:
        json.dump(shard_index, f, indent=2)
    simple_logger(
        f"Saved shard index to {paths.shard_index_file}",
        prefix="S3",
    )
    simple_logger(
        f"Shard summary: {len(shard_index['train'])} train, "
        f"{len(shard_index['val'])} val, {len(shard_index['test'])} test shards",
        prefix="S3",
    )


if __name__ == "__main__":
    main()
