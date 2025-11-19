# dataset_sharded.py

"""
Sharded dataset implementation for DeepLOB sequences.

This module provides ShardedSequenceDataset which can load sequences
from multiple shard files without loading everything into memory at once.
"""

import bisect
import gc
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from config import paths, data_config


class ShardedSequenceDataset(Dataset):
    """
    Dataset that loads sequences from multiple shard files.
    
    This dataset:
    - Stores only file paths, not data
    - Loads shards on-demand in __getitem__
    - Caches the currently loaded shard to avoid repeated file I/O
    - Maps global indices to (shard_id, local_index) pairs
    """
    
    def __init__(
        self,
        shard_files: List[str],
        shard_sizes: List[int] = None,
        sequences_dir: Path = None,
        split_features: bool = True,
    ):
        """
        Initialize dataset with list of shard filenames.
        
        Args:
            shard_files: List of shard filenames (e.g., ["train_shard_000.npz", ...])
            shard_sizes: Optional list of shard sizes. If None, will load from files.
            sequences_dir: Directory containing shard files (defaults to paths.sequences_dir)
            split_features: If True, return separate (lob_features, extra_features, y) tuples.
                           If False, return combined (X, y) for backward compatibility.
        """
        if sequences_dir is None:
            sequences_dir = paths.sequences_dir
        
        self.shard_paths = [sequences_dir / fname for fname in shard_files]
        self.sequences_dir = sequences_dir
        self.split_features = split_features
        
        # Feature dimensions
        num_levels = data_config.num_levels
        self.num_lob_features = 4 * num_levels + 2  # Raw LOB features
        self.num_extra_features = 8  # Engineered features
        
        # Use provided shard sizes or load from files
        if shard_sizes is not None:
            self.shard_sizes = shard_sizes
        else:
            # Fallback: load shard sizes from files (slower)
            # Use mmap_mode='r' to avoid loading full data, just read shape metadata
            self.shard_sizes = []
            for shard_path in self.shard_paths:
                if not shard_path.exists():
                    raise FileNotFoundError(f"Shard file not found: {shard_path}")
                # Try to read shape without loading full array
                try:
                    shard_data = np.load(shard_path, mmap_mode='r')
                    self.shard_sizes.append(shard_data["X"].shape[0])
                    del shard_data
                except (ValueError, TypeError):
                    # Fallback if mmap_mode doesn't work with NPZ
                    shard_data = np.load(shard_path)
                    self.shard_sizes.append(shard_data["X"].shape[0])
                    del shard_data
                gc.collect()
        
        # Build cumulative sizes for fast lookup
        self.cumulative_sizes = [0]
        for size in self.shard_sizes:
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)
        
        self.total_size = self.cumulative_sizes[-1]
        
        # Cache for currently loaded shard
        self.cached_shard_idx = None
        self.cached_X = None
        self.cached_y = None
    
    def __len__(self) -> int:
        return self.total_size
    
    def __getitem__(self, idx: int):
        """
        Get sequence and label at global index.
        
        Loads the appropriate shard if not already cached.
        """
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} out of range [0, {self.total_size})")
        
        # Find which shard contains this index using binary search on cumulative sizes
        # Since cumulative_sizes is sorted, we can use bisect
        shard_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        local_idx = idx - self.cumulative_sizes[shard_idx]
        
        # Load shard if not cached
        if self.cached_shard_idx != shard_idx:
            # Explicitly drop previous shard to free memory
            self.cached_X = None
            self.cached_y = None
            gc.collect()
            
            # Load new shard
            shard_path = self.shard_paths[shard_idx]
            shard_data = np.load(shard_path)
            self.cached_X = shard_data["X"]
            self.cached_y = shard_data["y"]
            self.cached_shard_idx = shard_idx
            del shard_data
            gc.collect()
        
        # Return sequence and label
        X_seq = torch.from_numpy(self.cached_X[local_idx]).float()  # (L, F_total)
        y_label = torch.tensor(self.cached_y[local_idx], dtype=torch.long)
        
        if self.split_features:
            # Split into LOB features and extra features
            lob_features = X_seq[:, :self.num_lob_features]  # (L, F_lob)
            extra_features = X_seq[:, self.num_lob_features:]  # (L, F_extra)
            return lob_features, extra_features, y_label
        else:
            # Backward compatibility: return combined features
            return X_seq, y_label


def load_shard_index(shard_index_file: Path = None) -> dict:
    """
    Load shard index JSON file.
    
    Returns:
        dict with keys: "train", "val", "test", "seq_len", "num_features", "total_sequences"
    """
    if shard_index_file is None:
        shard_index_file = paths.shard_index_file
    
    if not shard_index_file.exists():
        raise FileNotFoundError(
            f"Shard index file not found: {shard_index_file}. "
            "Run S3_build_sequences_and_labels.py first."
        )
    
    with open(shard_index_file, "r") as f:
        shard_index = json.load(f)
    
    return shard_index

