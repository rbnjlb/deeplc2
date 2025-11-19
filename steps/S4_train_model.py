# S4_train_model.py

"""
Step 4: Train the DeepLOB model (SHARD-WISE VERSION, BINARY CLASSIFICATION).

Tasks:
- Load training and validation data from sharded sequence files.
- Initialize DeepLOB model (lob.model_deeplob) for binary classification (Down=0, Up=1).
- Train with configurable hyperparameters (config.TrainingConfig).
- Use CrossEntropyLoss with optional class weights (inverse frequency).
- Early stopping based on validation loss.
- Save best model and simple training logs.

Training is done shard-by-shard to minimize memory usage:
- Only one shard (~160MB) is loaded at a time
- Process shard → train → delete → next shard
- This ensures bounded memory usage even with many shards

Run:
    python steps/S4_train_model.py
"""

import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

from config import (
    paths,
    label_config,
    model_config,
    training_config,
    data_config,
)
from lob.model_deeplob import build_deeplob_model
from utils import ensure_dir, get_device, set_seed, simple_logger

# Print train shard progress every N shards to reduce console noise
PRINT_TRAIN_SHARD_EVERY = 30  # only print 1 line every 30 train shards

class EarlyStopping:
    """Early stopping on validation loss."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def format_progress_bar(current: int, total: int, width: int = 30) -> str:
    """
    Create a simple text-based progress bar.
    
    Args:
        current: Current progress value
        total: Total value
        width: Width of the progress bar in characters
    
    Returns:
        Formatted progress bar string
    """
    if total == 0:
        return "[" + "=" * width + "]"
    
    filled = int(width * current / total)
    bar = "=" * filled + "-" * (width - filled)
    percent = 100.0 * current / total
    return f"[{bar}] {percent:.1f}%"


def log_batch_progress(
    epoch: int,
    shard_idx: int,
    total_shards: int,
    batch_idx: int,
    total_batches: int,
    batch_loss: float,
    running_loss: float,
    running_acc: float,
    samples_processed: int,
    elapsed_time: float,
    prefix: str = "S4",
):
    """
    Log detailed batch-level progress during training.
    
    Args:
        epoch: Current epoch number
        shard_idx: Current shard index
        total_shards: Total number of shards in this epoch
        batch_idx: Current batch index within shard
        total_batches: Total batches in current shard
        batch_loss: Loss for current batch
        running_loss: Running average loss
        running_acc: Running average accuracy
        samples_processed: Total samples processed so far
        elapsed_time: Time elapsed in seconds
        prefix: Log prefix
    """
    # Log every 10 batches or on last batch
    if batch_idx % 10 == 0 or batch_idx == total_batches:
        progress_bar = format_progress_bar(batch_idx, total_batches, width=20)
        samples_per_sec = samples_processed / elapsed_time if elapsed_time > 0 else 0
        
        simple_logger(
            f"Epoch {epoch:03d} | Shard {shard_idx}/{total_shards} | "
            f"Batch {batch_idx}/{total_batches} {progress_bar} | "
            f"Loss: {batch_loss:.4f} (avg: {running_loss:.4f}) | "
            f"Acc: {running_acc:.2%} | "
            f"Speed: {samples_per_sec:.0f} samples/s",
            prefix=prefix,
        )


def compute_class_weights_from_shards(
    sequences_dir: Path,
    shard_index: dict,
    num_classes: int,
) -> torch.Tensor:
    """
    Efficiently compute class weights by scanning only the y arrays
    from all train shards, without going through DataLoader.

    Args:
        sequences_dir: base directory where shard .npz files live.
        shard_index: dict loaded from shard_index.json.
        num_classes: number of classes (2 for Down/Up binary classification).

    Returns:
        torch.Tensor of shape (2,), dtype float32.
    """
    train_shards = shard_index["train"]
    counts = np.zeros(num_classes, dtype=np.int64)
    
    for i, fname in enumerate(train_shards, start=1):
        shard_path = sequences_dir / fname
        data = np.load(shard_path)
        y = data["y"]
        counts += np.bincount(y, minlength=num_classes)
        del data
        
        if i % 5 == 0 or i == len(train_shards):
            simple_logger(
                f"Class weight pass: processed {i}/{len(train_shards)} train shards, "
                f"current counts = {counts.tolist()} (Down={counts[0]}, Up={counts[1]})",
                prefix="S4",
            )
    
    # Expect exactly 2 classes
    if counts[0] == 0 or counts[1] == 0:
        raise RuntimeError(
            f"Training data missing one of the classes. "
            f"Counts: Down={counts[0]}, Up={counts[1]}"
        )
    
    # Class weights = inverse frequency
    weights = 1.0 / counts.astype(np.float32)
    # Normalize so mean weight ≈ 1
    weights = weights * (num_classes / weights.sum())
    
    # Ensure weight length = 2
    assert len(weights) == 2, f"Expected 2 class weights, got {len(weights)}"
    
    return torch.tensor(weights, dtype=torch.float32)


def train():
    set_seed(training_config.random_seed)
    device = get_device()
    simple_logger(f"Using device: {device}", prefix="S4")

    ensure_dir(paths.models_dir)
    ensure_dir(paths.logs_dir)

    # ===== COMPREHENSIVE CONFIGURATION LOGGING =====
    simple_logger("=" * 80, prefix="S4")
    simple_logger("TRAINING CONFIGURATION", prefix="S4")
    simple_logger("=" * 80, prefix="S4")
    
    # Label and dataset config
    simple_logger(f"Label Mode: {label_config.label_mode}", prefix="S4")
    simple_logger(f"Number of Classes: {model_config.num_classes}", prefix="S4")
    simple_logger(f"Label Horizon: {label_config.label_horizon} seconds", prefix="S4")
    if label_config.label_mode == "binary_alpha":
        simple_logger(f"Alpha Quantile: {label_config.label_alpha_quantile}", prefix="S4")
    elif label_config.label_mode == "three_class":
        simple_logger(f"Three-Class Threshold: {label_config.three_class_threshold:.2e}", prefix="S4")
    
    # Normalization config
    simple_logger(f"Normalization Mode: {data_config.normalization_mode}", prefix="S4")
    if data_config.normalization_mode == "train_only":
        simple_logger(f"Normalization Train Fraction: {data_config.normalization_train_fraction:.1%}", prefix="S4")
    
    # Architecture config
    simple_logger(f"LSTM Hidden Size: {model_config.lstm_hidden_size}", prefix="S4")
    simple_logger(f"Conv Channel Base: {model_config.conv_channel_base}", prefix="S4")
    simple_logger(f"Extra Feature Branch: {model_config.use_extra_feature_branch}", prefix="S4")
    if model_config.use_extra_feature_branch:
        simple_logger(f"Extra Feature Embedding Size: {model_config.extra_feature_embedding_size}", prefix="S4")
    
    # Training config
    simple_logger(f"Batch Size: {training_config.batch_size}", prefix="S4")
    simple_logger(f"Number of Epochs: {training_config.num_epochs}", prefix="S4")
    simple_logger(f"Learning Rate: {training_config.learning_rate:.2e}", prefix="S4")
    simple_logger(f"Weight Decay: {training_config.weight_decay:.2e}", prefix="S4")
    simple_logger(f"Early Stopping Patience: {training_config.early_stopping_patience}", prefix="S4")
    simple_logger(f"Early Stopping Metric: {training_config.early_stopping_metric}", prefix="S4")
    simple_logger(f"Use Class Weights: {training_config.use_class_weights}", prefix="S4")
    simple_logger(f"Shuffle Train Shards: {training_config.shuffle_train_shards}", prefix="S4")
    simple_logger(f"Scheduler Type: {training_config.scheduler_type}", prefix="S4")
    simple_logger("=" * 80, prefix="S4")

    # Load shard index
    with open(paths.shard_index_file, "r") as f:
        shard_index = json.load(f)
    
    train_shards = shard_index["train"]
    val_shards = shard_index["val"]
    test_shards = shard_index.get("test", [])
    seq_len = shard_index["seq_len"]
    num_features = shard_index["num_features"]
    num_classes = model_config.num_classes
    
    # Feature dimensions
    num_levels = data_config.num_levels
    num_lob_features = 4 * num_levels + 2  # Raw LOB features
    num_extra_features = 8  # Engineered features
    
    simple_logger(
        f"Loaded shard index: {len(train_shards)} train shards, {len(val_shards)} val shards, {len(test_shards)} test shards",
        prefix="S4",
    )
    simple_logger(
        f"Dataset config: seq_len={seq_len}, num_features={num_features}",
        prefix="S4",
    )
    simple_logger(
        f"Feature split: LOB features={num_lob_features}, extra features={num_extra_features}",
        prefix="S4",
    )
    
    # Ensure shards are used in temporal order (no shuffling unless explicitly enabled)
    if not training_config.shuffle_train_shards:
        simple_logger(
            "[S4] Using train shards in temporal order (no shuffling) to preserve time-based split",
            prefix="S4",
        )
    if not training_config.shuffle_val_shards:
        simple_logger(
            "[S4] Using val shards in temporal order (no shuffling) for proper validation",
            prefix="S4",
        )

    # Model
    model = build_deeplob_model(
        seq_len=seq_len,
        num_features=num_features,
        num_classes=num_classes,
        num_levels=num_levels,
    ).to(device)

    # Loss function (optionally with class weights)
    if training_config.use_class_weights:
        simple_logger("Computing class weights from training shards...", prefix="S4")
        class_weights = compute_class_weights_from_shards(
            sequences_dir=paths.sequences_dir,
            shard_index=shard_index,
            num_classes=num_classes,
        ).to(device)
        simple_logger(
            f"Using class weights: {class_weights.cpu().numpy()}",
            prefix="S4",
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        simple_logger("Not using class weights (use_class_weights=False)", prefix="S4")
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    
    # Learning rate scheduler based on config
    scheduler = None
    if training_config.scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=training_config.scheduler_factor,
            patience=training_config.scheduler_patience,
            min_lr=training_config.scheduler_min_lr,
        )
        simple_logger(f"Using ReduceLROnPlateau scheduler (patience={training_config.scheduler_patience})", prefix="S4")
    elif training_config.scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_config.scheduler_step_size,
            gamma=training_config.scheduler_gamma,
        )
        simple_logger(f"Using StepLR scheduler (step_size={training_config.scheduler_step_size})", prefix="S4")
    elif training_config.scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config.scheduler_T_max,
        )
        simple_logger(f"Using CosineAnnealingLR scheduler (T_max={training_config.scheduler_T_max})", prefix="S4")
    else:
        simple_logger(f"Unknown scheduler_type: {training_config.scheduler_type}, not using scheduler", prefix="S4")

    early_stopper = EarlyStopping(
        patience=training_config.early_stopping_patience,
        min_delta=training_config.early_stopping_min_delta,
    )

    # Initialize best metric value based on metric type
    if training_config.early_stopping_metric == "val_loss":
        best_metric_value = float("inf")
    else:  # val_acc
        best_metric_value = 0.0
    best_model_path = paths.models_dir / "deeplob_btcusdt.pt"
    log_path = paths.logs_dir / "training_log.json"

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    # Get shard usage limits from config
    max_train_shards = training_config.max_train_shards_per_epoch
    max_val_shards = training_config.max_val_shards_per_epoch
    
    # Use all shards if limits are None
    if max_train_shards is None:
        max_train_shards = len(train_shards)
    if max_val_shards is None:
        max_val_shards = len(val_shards)
    
    # Create lists for shard selection each epoch
    all_train_shards = list(train_shards)  # Preserve temporal order
    all_val_shards = list(val_shards)      # Preserve temporal order
    
    simple_logger(
        f"Starting training for {training_config.num_epochs} epochs",
        prefix="S4",
    )
    simple_logger(
        f"Train shards: {len(train_shards)} total, using up to {max_train_shards} per epoch",
        prefix="S4",
    )
    simple_logger(
        f"Val shards: {len(val_shards)} total, using up to {max_val_shards} per epoch",
        prefix="S4",
    )
    if training_config.shuffle_train_shards:
        simple_logger(
            "[S4] WARNING: shuffle_train_shards=True - temporal order will be broken!",
            prefix="S4",
        )

    for epoch in range(1, training_config.num_epochs + 1):
        # ----- TRAIN -----
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        
        # 3D. Per-class metrics during training
        train_preds = []
        train_labels = []

        # Select train shards for this epoch (preserve temporal order unless shuffling enabled)
        epoch_train_shards = all_train_shards.copy()
        if training_config.shuffle_train_shards:
            random.shuffle(epoch_train_shards)
        epoch_train_shards = epoch_train_shards[:min(max_train_shards, len(epoch_train_shards))]

        # Process train shards one at a time
        epoch_start_time = time.time()
        for shard_idx, fname in enumerate(epoch_train_shards, start=1):
            shard_path = paths.sequences_dir / fname
            
            # Load single shard
            data = np.load(shard_path)
            X_shard = torch.from_numpy(data["X"]).float().to(device)  # (N_shard, L, F)
            y_shard = torch.from_numpy(data["y"]).long().to(device)   # (N_shard,)
            shard_size = len(X_shard)
            del data
            
            # Split features into LOB and extra features
            lob_shard = X_shard[:, :, :num_lob_features]  # (N_shard, L, F_lob)
            extra_shard = X_shard[:, :, num_lob_features:]  # (N_shard, L, F_extra)
            
            # Create dataset and loader for this shard
            dataset = TensorDataset(lob_shard, extra_shard, y_shard)
            loader = DataLoader(
                dataset,
                batch_size=training_config.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,      # IMPORTANT: keep 0 to avoid multiple shard copies
                pin_memory=False,   # CPU only; no need for pinned memory
            )

            # Track running metrics for this shard
            shard_start_time = time.time()
            shard_batch_losses = []
            shard_batch_accs = []
            
            # Train on this shard
            total_batches = len(loader)
            for batch_idx, (lob_batch, extra_batch, y_batch) in enumerate(loader, start=1):
                # Runtime safety checks
                if not torch.isfinite(lob_batch).all():
                    simple_logger("Non-finite values detected in lob_batch", prefix="S4")
                    raise ValueError("Non-finite lob_batch")
                if not torch.isfinite(extra_batch).all():
                    simple_logger("Non-finite values detected in extra_batch", prefix="S4")
                    raise ValueError("Non-finite extra_batch")
                
                optimizer.zero_grad()
                # Forward pass with separate feature inputs
                if model_config.use_extra_feature_branch:
                    logits = model(lob_batch, extra_batch)  # (B, num_classes)
                else:
                    logits = model(lob_batch, None)  # (B, num_classes)
                
                # Check logits
                if not torch.isfinite(logits).all():
                    simple_logger("Non-finite logits detected", prefix="S4")
                    raise ValueError("Non-finite logits")
                
                loss = criterion(logits, y_batch)
                
                # Check loss
                if not torch.isfinite(loss):
                    simple_logger(f"Non-finite loss detected: {loss.item()}", prefix="S4")
                    raise ValueError("Non-finite loss during training")

                loss.backward()
                
                # 3C. Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=training_config.max_grad_norm
                )
                
                optimizer.step()

                # Track metrics
                batch_loss = loss.item()
                batch_size = lob_batch.size(0)
                train_loss_sum += batch_loss * batch_size
                preds = logits.argmax(dim=1)
                batch_correct = (preds == y_batch).sum().item()
                train_correct += batch_correct
                train_total += batch_size
                
                # Collect predictions for per-class metrics
                train_preds.append(preds.cpu().numpy())
                train_labels.append(y_batch.cpu().numpy())
                
                # Store for running averages
                shard_batch_losses.append(batch_loss)
                batch_acc = batch_correct / batch_size
                shard_batch_accs.append(batch_acc)
                
                # Calculate running averages
                running_loss = np.mean(shard_batch_losses)
                running_acc = np.mean(shard_batch_accs)
                
                # Per-batch logging disabled to reduce console noise
                # Progress is printed every PRINT_TRAIN_SHARD_EVERY shards instead
                # Uncomment below to enable per-batch logging:
                # elapsed_time = time.time() - epoch_start_time
                # log_batch_progress(
                #     epoch=epoch,
                #     shard_idx=shard_idx,
                #     total_shards=len(epoch_train_shards),
                #     batch_idx=batch_idx,
                #     total_batches=total_batches,
                #     batch_loss=batch_loss,
                #     running_loss=running_loss,
                #     running_acc=running_acc,
                #     samples_processed=train_total,
                #     elapsed_time=elapsed_time,
                # )

            # Explicitly drop shard tensors and trigger GC before next shard
            del X_shard, lob_shard, extra_shard, y_shard, dataset, loader
            gc.collect()

            # Progress logging: print every PRINT_TRAIN_SHARD_EVERY shards or on final shard
            shard_time = time.time() - shard_start_time
            shard_avg_loss = np.mean(shard_batch_losses) if shard_batch_losses else 0.0
            shard_avg_acc = np.mean(shard_batch_accs) if shard_batch_accs else 0.0
            
            shard_number = shard_idx  # shard_idx is already 1-based from enumerate(start=1)
            num_train_shards = len(epoch_train_shards)
            
            if (shard_number % PRINT_TRAIN_SHARD_EVERY == 0) or (shard_number == num_train_shards):
                simple_logger(
                    f"[S4] Epoch {epoch:03d} - ✓ Completed train shard {shard_number}/{num_train_shards} "
                    f"({shard_size:,} samples, {shard_time:.1f}s) | "
                    f"Shard avg loss: {shard_avg_loss:.4f}, acc: {shard_avg_acc:.2%} | "
                    f"Total samples: {train_total:,}",
                    prefix="S4",
                )

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        
        # Compute per-class metrics for training
        train_preds_all = np.concatenate(train_preds)
        train_labels_all = np.concatenate(train_labels)
        
        if num_classes == 2:
            cm_train = confusion_matrix(train_labels_all, train_preds_all, labels=[0, 1])
            train_down_acc = cm_train[0, 0] / max(cm_train[0].sum(), 1)
            train_up_acc = cm_train[1, 1] / max(cm_train[1].sum(), 1)
            train_per_class_str = f"Down_acc={train_down_acc:.2%}, Up_acc={train_up_acc:.2%}"
        elif num_classes == 3:
            cm_train = confusion_matrix(train_labels_all, train_preds_all, labels=[0, 1, 2])
            train_down_acc = cm_train[0, 0] / max(cm_train[0].sum(), 1) if cm_train[0].sum() > 0 else 0.0
            train_stationary_acc = cm_train[1, 1] / max(cm_train[1].sum(), 1) if cm_train[1].sum() > 0 else 0.0
            train_up_acc = cm_train[2, 2] / max(cm_train[2].sum(), 1) if cm_train[2].sum() > 0 else 0.0
            train_per_class_str = f"Down_acc={train_down_acc:.2%}, Stationary_acc={train_stationary_acc:.2%}, Up_acc={train_up_acc:.2%}"
        else:
            train_per_class_str = "Per-class metrics not computed"
        
        # Log epoch training summary
        epoch_train_time = time.time() - epoch_start_time
        simple_logger(
            f"Epoch {epoch:03d} TRAIN COMPLETE | "
            f"Time: {epoch_train_time:.1f}s | "
            f"Loss: {train_loss:.4f} | Acc: {train_acc:.2%} | "
            f"Samples: {train_total:,}",
            prefix="S4",
        )
        if training_config.log_train_accuracy:
            simple_logger(
                f"Epoch {epoch:03d} TRAIN PER-CLASS: {train_per_class_str}",
                prefix="S4",
            )

        # ----- VALIDATION -----
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        
        # 3D. Per-class metrics during validation
        val_preds = []
        val_labels = []
        
        val_start_time = time.time()

        with torch.no_grad():
            # Select val shards for this epoch (preserve temporal order unless shuffling enabled)
            epoch_val_shards = all_val_shards.copy()
            if training_config.shuffle_val_shards:
                random.shuffle(epoch_val_shards)
            epoch_val_shards = epoch_val_shards[:min(max_val_shards, len(epoch_val_shards))]

            # Process val shards one at a time
            for shard_idx, fname in enumerate(epoch_val_shards, start=1):
                shard_path = paths.sequences_dir / fname
                
                # Load single shard
                data = np.load(shard_path)
                X_shard = torch.from_numpy(data["X"]).float().to(device)
                y_shard = torch.from_numpy(data["y"]).long().to(device)
                shard_size = len(X_shard)
                del data

                # Split features into LOB and extra features
                lob_shard = X_shard[:, :, :num_lob_features]  # (N_shard, L, F_lob)
                extra_shard = X_shard[:, :, num_lob_features:]  # (N_shard, L, F_extra)

                # Create dataset and loader for this shard
                dataset = TensorDataset(lob_shard, extra_shard, y_shard)
                loader = DataLoader(
                    dataset,
                    batch_size=training_config.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=0,      # IMPORTANT: keep 0 to avoid multiple shard copies
                    pin_memory=False,   # CPU only; no need for pinned memory
                )

                # Validate on this shard
                shard_val_loss = 0.0
                shard_val_correct = 0
                shard_val_total = 0
                
                for batch_idx, (lob_batch, extra_batch, y_batch) in enumerate(loader, start=1):
                    # Runtime safety checks
                    if not torch.isfinite(lob_batch).all():
                        simple_logger("Non-finite values detected in lob_batch (validation)", prefix="S4")
                        raise ValueError("Non-finite lob_batch in validation")
                    if not torch.isfinite(extra_batch).all():
                        simple_logger("Non-finite values detected in extra_batch (validation)", prefix="S4")
                        raise ValueError("Non-finite extra_batch in validation")
                    
                    # Forward pass with separate feature inputs
                    if model_config.use_extra_feature_branch:
                        logits = model(lob_batch, extra_batch)
                    else:
                        logits = model(lob_batch, None)
                    
                    # Check logits
                    if not torch.isfinite(logits).all():
                        simple_logger("Non-finite logits detected (validation)", prefix="S4")
                        raise ValueError("Non-finite logits in validation")
                    
                    loss = criterion(logits, y_batch)
                    
                    # Check loss
                    if not torch.isfinite(loss):
                        simple_logger(f"Non-finite loss detected (validation): {loss.item()}", prefix="S4")
                        raise ValueError("Non-finite loss in validation")

                    batch_loss = loss.item()
                    batch_size = lob_batch.size(0)
                    val_loss_sum += batch_loss * batch_size
                    preds = logits.argmax(dim=1)
                    batch_correct = (preds == y_batch).sum().item()
                    val_correct += batch_correct
                    val_total += batch_size
                    
                    # Collect predictions for per-class metrics
                    val_preds.append(preds.cpu().numpy())
                    val_labels.append(y_batch.cpu().numpy())
                    
                    # Track shard metrics
                    shard_val_loss += batch_loss * batch_size
                    shard_val_correct += batch_correct
                    shard_val_total += batch_size

                # Explicitly drop shard tensors and trigger GC before next shard
                del X_shard, lob_shard, extra_shard, y_shard, dataset, loader
                gc.collect()
                
                # Progress logging after each shard
                shard_val_avg_loss = shard_val_loss / max(shard_val_total, 1)
                shard_val_avg_acc = shard_val_correct / max(shard_val_total, 1)
                
                simple_logger(
                    f"Epoch {epoch:03d} - ✓ Completed val shard {shard_idx}/{len(epoch_val_shards)} "
                    f"({shard_size:,} samples) | "
                    f"Shard avg loss: {shard_val_avg_loss:.4f}, acc: {shard_val_avg_acc:.2%} | "
                    f"Total samples: {val_total:,}",
                    prefix="S4",
                )

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        
        # Compute per-class metrics for validation
        val_preds_all = np.concatenate(val_preds)
        val_labels_all = np.concatenate(val_labels)
        
        if num_classes == 2:
            cm_val = confusion_matrix(val_labels_all, val_preds_all, labels=[0, 1])
            val_down_acc = cm_val[0, 0] / max(cm_val[0].sum(), 1)
            val_up_acc = cm_val[1, 1] / max(cm_val[1].sum(), 1)
            val_per_class_str = f"Down_acc={val_down_acc:.2%}, Up_acc={val_up_acc:.2%}"
        elif num_classes == 3:
            cm_val = confusion_matrix(val_labels_all, val_preds_all, labels=[0, 1, 2])
            val_down_acc = cm_val[0, 0] / max(cm_val[0].sum(), 1) if cm_val[0].sum() > 0 else 0.0
            val_stationary_acc = cm_val[1, 1] / max(cm_val[1].sum(), 1) if cm_val[1].sum() > 0 else 0.0
            val_up_acc = cm_val[2, 2] / max(cm_val[2].sum(), 1) if cm_val[2].sum() > 0 else 0.0
            val_per_class_str = f"Down_acc={val_down_acc:.2%}, Stationary_acc={val_stationary_acc:.2%}, Up_acc={val_up_acc:.2%}"
        else:
            val_per_class_str = "Per-class metrics not computed"
        
        val_time = time.time() - val_start_time
        total_epoch_time = time.time() - epoch_start_time
        
        # Step scheduler
        if scheduler is not None:
            if training_config.scheduler_type == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            simple_logger(
                f"Epoch {epoch:03d} | LR now = {current_lr:.2e}",
                prefix="S4",
            )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Log epoch summary
        simple_logger(
            f"{'='*80}",
            prefix="S4",
        )
        simple_logger(
            f"Epoch {epoch:03d} SUMMARY | "
            f"Total time: {total_epoch_time:.1f}s (train: {epoch_train_time:.1f}s, val: {val_time:.1f}s)",
            prefix="S4",
        )
        simple_logger(
            f"  Train: loss={train_loss:.4f}, acc={train_acc:.2%} ({train_total:,} samples)",
            prefix="S4",
        )
        if training_config.log_train_accuracy:
            simple_logger(
                f"  Train per-class: {train_per_class_str}",
                prefix="S4",
            )
        simple_logger(
            f"  Val:   loss={val_loss:.4f}, acc={val_acc:.2%} ({val_total:,} samples)",
            prefix="S4",
        )
        if training_config.log_val_accuracy:
            simple_logger(
                f"  Val per-class: {val_per_class_str}",
                prefix="S4",
            )
        simple_logger(
            f"{'='*80}",
            prefix="S4",
        )

        # Early stopping based on configured metric
        if training_config.early_stopping_metric == "val_loss":
            metric_value = val_loss
            is_best = val_loss < best_metric_value
            if is_best:
                best_metric_value = val_loss
        elif training_config.early_stopping_metric == "val_acc":
            metric_value = -val_acc  # Negate because EarlyStopping expects lower is better
            is_best = val_acc > best_metric_value
            if is_best:
                best_metric_value = val_acc
        else:
            metric_value = val_loss  # Default to val_loss
            is_best = val_loss < best_metric_value
            if is_best:
                best_metric_value = val_loss
        
        if is_best:
            torch.save(model.state_dict(), best_model_path)
            simple_logger(
                f"Saved new best model to {best_model_path} (best {training_config.early_stopping_metric}: {best_metric_value:.4f})",
                prefix="S4",
            )

        early_stopper.step(metric_value)
        if early_stopper.should_stop:
            simple_logger(
                f"Early stopping triggered after {early_stopper.counter} epochs without improvement.",
                prefix="S4",
            )
            break

    # Save training log
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    # Final training summary
    total_epochs = len(history["train_loss"])
    if training_config.early_stopping_metric == "val_loss":
        best_epoch = np.argmin(history["val_loss"]) + 1  # +1 because epochs start at 1
        best_metric_final = min(history["val_loss"])
    else:  # val_acc
        best_epoch = np.argmax(history["val_acc"]) + 1
        best_metric_final = max(history["val_acc"])
    best_train_loss = history["train_loss"][best_epoch - 1]
    best_train_acc = history["train_acc"][best_epoch - 1]
    best_val_loss = history["val_loss"][best_epoch - 1]
    best_val_acc = history["val_acc"][best_epoch - 1]
    
    simple_logger("", prefix="S4")  # Empty line for spacing
    simple_logger("=" * 80, prefix="S4")
    simple_logger("TRAINING COMPLETE", prefix="S4")
    simple_logger("=" * 80, prefix="S4")
    simple_logger(f"Total epochs completed: {total_epochs}", prefix="S4")
    simple_logger(f"Best epoch: {best_epoch} (best {training_config.early_stopping_metric})", prefix="S4")
    simple_logger(f"Best {training_config.early_stopping_metric}: {best_metric_final:.4f}", prefix="S4")
    simple_logger(f"Best validation loss: {best_val_loss:.4f}", prefix="S4")
    simple_logger(f"Best validation accuracy: {best_val_acc:.2%}", prefix="S4")
    simple_logger(f"Training loss at best epoch: {best_train_loss:.4f}", prefix="S4")
    simple_logger(f"Training accuracy at best epoch: {best_train_acc:.2%}", prefix="S4")
    simple_logger("", prefix="S4")
    simple_logger(f"Training log saved to: {log_path}", prefix="S4")
    simple_logger(f"Best model saved to: {best_model_path}", prefix="S4")
    simple_logger("=" * 80, prefix="S4")


if __name__ == "__main__":
    train()
