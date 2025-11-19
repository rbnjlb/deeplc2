# config.py

"""
Central configuration for the DeepLOB Binance BTCUSDT project.

Edit this file to change paths, hyperparameters, sequence length, etc.

MODE SWITCHING:
==============

To switch between binary and ternary modes, edit these two variables at the top:

    state = "binary"    # or "ternary"
    alpha = 0.85        # Alpha threshold for binary mode (only used when state="binary")

Modes:
- "binary": 2-class classification (Down=0, Up=1) with α-quantile filtering
- "ternary": 3-class classification (Down=0, Stationary=1, Up=2) DeepLOB baseline

The configuration system will automatically apply the correct settings based on the selected mode.
"""

from dataclasses import dataclass
from pathlib import Path

# ===== USER-SELECTABLE MODE =====
# Change these two variables to switch between binary and ternary modes
state = "binary"    # options: "binary", "ternary"
alpha = 0.85        # Alpha threshold for binary extreme-move filtering (only used in binary mode)

# --------- PATH CONFIG ----------

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class Paths:
    # Raw LOB snapshots (CSV/Parquet) from Binance or external source
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"

    # Preprocessed, 1-second aligned LOB snapshots
    preprocessed_dir: Path = PROJECT_ROOT / "data" / "preprocessed"
    preprocessed_lob_file: Path = preprocessed_dir / "lob_preprocessed_btcusdt.npz"

    # Final sequences and labels (sharded format)
    sequences_dir: Path = PROJECT_ROOT / "data" / "sequences"
    sequences_file: Path = sequences_dir / "btc_usdt_sequences_labels.npz"  # Deprecated: kept for backward compatibility
    shard_index_file: Path = sequences_dir / "shard_index.json"  # Index file listing all shards

    # Models, logs, results
    models_dir: Path = PROJECT_ROOT / "models"
    logs_dir: Path = PROJECT_ROOT / "logs"
    results_dir: Path = PROJECT_ROOT / "results"


@dataclass
class DataConfig:
    symbol: str = "btcusdt"            # Binance symbol (lowercase for file naming)
    num_levels: int = 10              # Number of bid/ask levels to enforce
    timestamp_col: str = "timestamp"  # Name of timestamp column in raw data

    # Pattern for raw files. You can adapt this to your actual naming scheme.
    # Example expected file names:
    #   data/raw/btcusdt_depth_2024-02-01.parquet
    #   data/raw/btcusdt_depth_2024-02-02.parquet
    # 
    # NOTE: After running steps/S0_filter_raw_data.py, this is set to the February-only file:
    raw_file_glob: str = "btcusdt_depth_2024-02_feb_only.parquet"
    
    # Alternative: support CSV files with existing format
    csv_file_glob: str = "BTCUSDT_lob_*.csv"

    # Frequency for alignment
    resample_rule: str = "1s"         # 1-second grid (use lowercase 's')
    
    # Column naming convention: 'px_qty' (bid_px_1, bid_qty_1) or 'price_size' (bid_price_1, bid_size_1)
    column_naming: str = "px_qty"    # Use 'px_qty' for existing Binance data
    
    # --- Normalization configuration ---
    normalization_mode: str = "static"  # Allowed: "static", "train_only", "rolling", "none"
    # "static": Compute mean/std on all data (backward compatible)
    # "train_only": Compute mean/std only on training period, reuse for val/test (avoids leakage)
    # "rolling": Rolling z-score normalization (placeholder for future implementation)
    # "none": Disable normalization (features already normalized: relative prices, log volumes, etc.)
    
    # For train_only mode: fraction of data to use for fitting scaler (from start)
    normalization_train_fraction: float = 0.7  # Use first 70% of data for fitting scaler
    
    # For rolling mode (future): window size for rolling statistics
    normalization_rolling_window: int = 1000  # Number of samples for rolling window (placeholder)


@dataclass
class LabelConfig:
    seq_len: int = 100          # L: sequence length in seconds
    pred_horizon: int = 10      # k: horizon in seconds ahead for labels (deprecated, use label_horizon)
    threshold: float = 0.0001   # Relative mid-price change threshold (deprecated, kept for backward compatibility)

    # Train/val/test split configuration (time-ordered sequences)
    # Sequences are assigned to splits based on their temporal order (no leakage)
    train_frac: float = 0.7     # Fraction of sequences for training (earliest time period)
    val_frac: float = 0.15      # Fraction of sequences for validation (middle time period)
    # test_frac = 1 - train_frac - val_frac (latest time period)
    
    # Alternative: explicit split boundaries (if provided, override train_frac/val_frac)
    # These can be shard indices or sequence indices
    train_start_idx: int = None  # Start index for training (None = use train_frac)
    train_end_idx: int = None    # End index for training
    val_start_idx: int = None     # Start index for validation
    val_end_idx: int = None       # End index for validation
    test_start_idx: int = None    # Start index for test
    test_end_idx: int = None      # End index for test
    
    # Sharding configuration
    max_sequences_per_shard: int = 10_000  # Maximum sequences per shard file (reduced for memory efficiency)
    
    # --- Label mode configuration ---
    label_mode: str = "binary_alpha"  # Allowed: "binary_alpha", "three_class"
    label_horizon: int = 10            # Future horizon in seconds for label computation
    
    # --- Binary alpha mode settings ---
    label_alpha_quantile: float = 0.90     # Quantile threshold for binary_alpha mode (keep only top X% largest |returns|)
    label_min_samples_per_class: int = 1000  # Minimum samples required per class (binary_alpha mode only)
    
    # --- Three-class mode settings ---
    three_class_threshold: float = 0.0002  # Symmetric threshold for three_class mode (DeepLOB-style)
    # Labels: Up=2 if r_t > +threshold, Down=0 if r_t < -threshold, Stationary=1 otherwise
    
    # --- Label balancing (optional, binary_alpha mode only) ---
    label_balance_enabled: bool = True      # Enable label balancing before train/val/test split (binary_alpha only)
    label_balance_max_ratio: float = 1.5   # majority / minority capped at this ratio


@dataclass
class TrainingConfig:
    batch_size: int = 128  # Balanced batch size for CPU training
    num_epochs: int = 25  # Increased default (was 5 for quick testing)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    random_seed: int = 42
    
    # Early stopping configuration
    early_stopping_patience: int = 7  # Patience for early stopping (epochs without improvement)
    early_stopping_min_delta: float = 0.0  # Minimum change to qualify as improvement
    early_stopping_metric: str = "val_loss"  # Metric to monitor: "val_loss" or "val_acc"
    
    # Class weighting and downsampling (for binary_alpha mode)
    use_class_weights: bool = False  # Use inverse frequency class weights (default: False for binary_alpha)
    enable_downsampling: bool = False  # Enable downsampling majority class (default: False)
    
    # Shard usage per epoch
    # Set to None to use all available shards per epoch
    max_train_shards_per_epoch: int = None  # None = use all train shards, or set limit
    max_val_shards_per_epoch: int = None    # None = use all val shards, or set limit
    
    # Shard sampling strategy
    shuffle_train_shards: bool = False  # Shuffle train shards each epoch (False = preserve temporal order)
    shuffle_val_shards: bool = False    # Shuffle val shards (should be False for proper validation)
    
    # Learning rate scheduler
    scheduler_type: str = "ReduceLROnPlateau"  # "ReduceLROnPlateau" or "StepLR" or "CosineAnnealingLR"
    scheduler_patience: int = 3  # Patience for ReduceLROnPlateau
    scheduler_factor: float = 0.5  # LR reduction factor when plateau
    scheduler_min_lr: float = 1e-6  # Minimum learning rate
    scheduler_step_size: int = 10  # Step size for StepLR
    scheduler_gamma: float = 0.1  # Gamma for StepLR
    scheduler_T_max: int = 25  # T_max for CosineAnnealingLR
    
    # Gradient clipping
    max_grad_norm: float = 5.0  # Maximum gradient norm for clipping
    
    # Logging
    log_val_accuracy: bool = True  # Log validation accuracy each epoch
    log_train_accuracy: bool = True  # Log training accuracy each epoch


@dataclass
class ModelConfig:
    # num_classes is set dynamically based on label_config.label_mode
    # Will be 3 if label_mode == "three_class", 2 if label_mode == "binary_alpha"
    num_classes: int = None  # Set automatically based on label_mode (see get_num_classes() below)
    # Features are computed dynamically from lob.shape[1] at runtime
    # Base features: rel_bid_prices (n) + rel_ask_prices (n) + log_bid_sizes (n) + log_ask_sizes (n) + spread (1) + imbalance (1)
    # Extra features: ret_1s (1) + ret_5s (1) + ret_10s (1) + delta_bid_vol (1) + delta_ask_vol (1) + top3_mean (1) + deep_mean (1) + imb_slope (1)
    # Total: 4*n + 2 + 8 = 4*n + 10 (for n=10 levels: 42 + 8 = 50 features)
    num_features: int = None  # Will be inferred from data at runtime
    
    # Architecture options for dual-branch model
    use_extra_feature_branch: bool = True  # Enable extra-feature MLP branch (set False for pure DeepLOB)
    lstm_hidden_size: int = 64  # LSTM hidden size (increased from 32 for better capacity)
    conv_channel_base: int = 32  # Base number of channels in conv layers (increased from 16)
    extra_feature_embedding_size: int = 16  # Output size of extra-feature MLP branch
    extra_feature_use_last_timestep: bool = True  # Use last timestep (True) or mean pooling (False) for extra features


paths = Paths()
data_config = DataConfig()
label_config = LabelConfig()
training_config = TrainingConfig()
model_config = ModelConfig()


def get_num_classes() -> int:
    """
    Get number of classes based on label_mode.
    
    Returns:
        3 if label_mode == "three_class"
        2 if label_mode == "binary_alpha"
    """
    if label_config.label_mode == "three_class":
        return 3
    elif label_config.label_mode == "binary_alpha":
        return 2
    else:
        raise ValueError(
            f"Unknown label_mode: {label_config.label_mode}. "
            "Allowed values: 'three_class', 'binary_alpha'"
        )


# Set num_classes automatically based on label_mode
model_config.num_classes = get_num_classes()


# ===== EXPERIMENT PRESETS =====
# These functions can be called to configure the system for specific experiments
# Import logger for preset functions
try:
    from utils import simple_logger
except ImportError:
    # Fallback if utils not available
    def simple_logger(msg, prefix=""):
        print(f"[{prefix}] {msg}")

def configure_deeplob_baseline():
    """
    Configure for DeepLOB baseline experiment (3-class mode, pure LOB branch).
    
    This preset:
    - Uses 3-class labels (Down=0, Stationary=1, Up=2)
    - Disables extra-feature branch (pure DeepLOB)
    - Uses train-only normalization
    - Sets reasonable training parameters
    
    NOTE: After calling this, you need to rebuild sequences:
        python S3_build_sequences_and_labels.py
    """
    # Enforce ternary (3-class) mode
    label_config.label_mode = "three_class"
    label_config.three_class_threshold = 0.0002
    
    # Disable binary-alpha specific fields
    label_config.label_alpha_quantile = None  # Not used in ternary mode
    label_config.label_min_samples_per_class = None  # Not used in ternary mode
    label_config.label_balance_enabled = False  # Not used in ternary mode
    
    model_config.num_classes = get_num_classes()  # Will be 3
    
    model_config.use_extra_feature_branch = False
    data_config.normalization_mode = "train_only"
    
    training_config.num_epochs = 30
    training_config.use_class_weights = False
    training_config.shuffle_train_shards = False
    training_config.max_train_shards_per_epoch = None  # Use all shards
    training_config.max_val_shards_per_epoch = None
    training_config.early_stopping_patience = 7
    training_config.early_stopping_metric = "val_loss"
    
    simple_logger("Configured for DeepLOB baseline experiment (3-class, pure LOB)", prefix="CONFIG")
    simple_logger("IMPORTANT: Rebuild sequences with: python S3_build_sequences_and_labels.py", prefix="CONFIG")


def configure_extreme_binary_alpha(alpha: float):
    """
    Configure for extreme binary α-quantile experiment.
    
    This preset:
    - Uses binary_alpha labels with specified quantile (extreme moves only)
    - Enables extra-feature branch
    - Uses train-only normalization
    - Sets training parameters optimized for binary classification
    
    Args:
        alpha: Alpha quantile threshold (e.g., 0.85 for top 15% largest moves)
    """
    # Enforce binary (2-class) mode
    label_config.label_mode = "binary_alpha"
    label_config.label_alpha_quantile = alpha
    label_config.label_min_samples_per_class = 1000  # Minimum samples per class
    label_config.label_balance_enabled = True  # Enable label balancing
    
    # Disable ternary-specific fields
    label_config.three_class_threshold = None  # Not used in binary mode
    
    model_config.num_classes = get_num_classes()  # Will be 2
    
    model_config.use_extra_feature_branch = True
    data_config.normalization_mode = "train_only"
    
    training_config.num_epochs = 25
    training_config.use_class_weights = False  # No class weights by default
    training_config.shuffle_train_shards = False
    training_config.max_train_shards_per_epoch = None  # Use all shards
    training_config.max_val_shards_per_epoch = None
    training_config.early_stopping_patience = 7
    training_config.early_stopping_metric = "val_acc"  # Monitor accuracy for binary
    
    simple_logger(f"Configured for extreme binary α experiment (binary_alpha, alpha={alpha}, extra features)", prefix="CONFIG")


# ===== AUTO DISPATCH BASED ON USER MODE =====
print(f"[CONFIG] Loading mode: {state}")

if state == "ternary":
    configure_deeplob_baseline()
elif state == "binary":
    configure_extreme_binary_alpha(alpha)
else:
    raise ValueError(f"Invalid config state: {state}. Allowed values: 'binary', 'ternary'")
