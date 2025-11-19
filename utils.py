# utils.py

"""
Common helper utilities: seeding, directory creation, logging, etc.
"""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device if available else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def simple_logger(msg: str, *, prefix: Optional[str] = None) -> None:
    """Very simple logger that prints messages with an optional prefix."""
    if prefix:
        print(f"[{prefix}] {msg}")
    else:
        print(msg)

