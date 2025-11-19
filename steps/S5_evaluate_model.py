# S5_evaluate_model.py

"""
Step 5: Evaluate the trained DeepLOB model on the test set (binary classification).

Tasks:
- Load test data (X_test, y_test) from sharded sequences.
- Load trained model weights.
- Compute:
    - Overall accuracy.
    - Per-class precision, recall, F1 (Down(0), Up(1)).
    - Confusion matrix with labels=[0, 1].
- Print and save results to JSON.

Run:
    python steps/S5_evaluate_model.py
"""

import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import paths, model_config, label_config, data_config
from lob.model_deeplob import build_deeplob_model
from lob.dataset_sharded import ShardedSequenceDataset, load_shard_index
from torch.utils.data import DataLoader
from utils import ensure_dir, get_device, simple_logger


def evaluate():
    device = get_device()
    simple_logger(f"Using device: {device}", prefix="S5")

    ensure_dir(paths.results_dir)

    # Load shard index
    shard_index = load_shard_index()
    
    test_shards = shard_index["test"]
    seq_len = shard_index["seq_len"]
    num_features = shard_index["num_features"]
    num_classes = model_config.num_classes
    
    # Feature dimensions
    num_levels = data_config.num_levels
    num_lob_features = 4 * num_levels + 2  # Raw LOB features
    num_extra_features = 8  # Engineered features
    
    simple_logger(
        f"Loaded shard index: {len(test_shards)} test shards",
        prefix="S5",
    )
    simple_logger(
        f"Dataset config: seq_len={seq_len}, num_features={num_features}",
        prefix="S5",
    )
    simple_logger(
        f"Feature split: LOB features={num_lob_features}, extra features={num_extra_features}",
        prefix="S5",
    )
    
    # Create sharded test dataset with pre-computed sizes from index
    # Use split_features=True to get separate feature groups
    test_shard_sizes = shard_index.get("test_sizes")
    test_dataset = ShardedSequenceDataset(
        test_shards, shard_sizes=test_shard_sizes, split_features=True
    )
    simple_logger(
        f"Test dataset size: {len(test_dataset):,}",
        prefix="S5",
    )

    # Load model
    best_model_path = paths.models_dir / "deeplob_btcusdt.pt"
    if not best_model_path.exists():
        raise RuntimeError(
            f"Model file not found: {best_model_path}. "
            "Run S4_train_model.py first."
        )

    model = build_deeplob_model(
        seq_len=seq_len,
        num_features=num_features,
        num_classes=num_classes,
        num_levels=num_levels,
    ).to(device)

    # Try to load model checkpoint
    try:
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint)
    except RuntimeError as e:
        if "size mismatch" in str(e):
            simple_logger(
                f"ERROR: Model checkpoint was trained with different features than current data.",
                prefix="S5",
            )
            simple_logger(
                f"Current sequences have {num_features} features, but checkpoint expects different architecture.",
                prefix="S5",
            )
            simple_logger(
                "SOLUTION: Retrain the model with the current feature set by running:",
                prefix="S5",
            )
            simple_logger("  python S4_train_model.py", prefix="S5")
            raise RuntimeError(
                "Model checkpoint incompatible with current feature set. "
                "Please retrain the model with the updated features."
            ) from e
        else:
            raise
    model.eval()

    # Evaluate using DataLoader to process in batches
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,  # Batch size for evaluation
        shuffle=False,
        drop_last=False,
        num_workers=0,      # IMPORTANT: keep 0 to avoid multiple shard copies in memory
        pin_memory=False,   # CPU only; no need for pinned memory
    )
    
    simple_logger("Evaluating model on test set...", prefix="S5")
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for batch_idx, (lob_batch, extra_batch, y_batch) in enumerate(test_loader):
            lob_batch = lob_batch.to(device)
            extra_batch = extra_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Runtime safety checks
            if not torch.isfinite(lob_batch).all():
                simple_logger("Non-finite values detected in lob_batch (evaluation)", prefix="S5")
                raise ValueError("Non-finite lob_batch in evaluation")
            if not torch.isfinite(extra_batch).all():
                simple_logger("Non-finite values detected in extra_batch (evaluation)", prefix="S5")
                raise ValueError("Non-finite extra_batch in evaluation")
            
            # Forward pass with separate feature inputs
            if model_config.use_extra_feature_branch:
                logits = model(lob_batch, extra_batch)
            else:
                logits = model(lob_batch, None)
            
            # Check logits
            if not torch.isfinite(logits).all():
                simple_logger("Non-finite logits detected (evaluation)", prefix="S5")
                raise ValueError("Non-finite logits in evaluation")
            
            # Predict logits → argmax (binary classification: labels ∈ {0,1})
            preds = logits.argmax(dim=1)
            
            y_true_list.append(y_batch.cpu().numpy())
            y_pred_list.append(preds.cpu().numpy())
            
            if (batch_idx + 1) % 100 == 0:
                simple_logger(
                    f"Processed {batch_idx + 1}/{len(test_loader)} batches...",
                    prefix="S5",
                )
    
    # Concatenate all predictions
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    
    # Warning if missing class
    unique_labels = np.unique(y_true)
    expected_classes = model_config.num_classes
    if len(unique_labels) < expected_classes:
        missing_classes = set(range(expected_classes)) - set(unique_labels)
        simple_logger(
            f"[S5] WARNING: test set missing classes: {missing_classes}",
            prefix="S5",
        )

    # Set target names based on label mode
    if model_config.num_classes == 2:
        target_names = ["Down(0)", "Up(1)"]
        labels_cm = [0, 1]
    elif model_config.num_classes == 3:
        target_names = ["Down(0)", "Stationary(1)", "Up(2)"]
        labels_cm = [0, 1, 2]
    else:
        target_names = [f"Class_{i}" for i in range(model_config.num_classes)]
        labels_cm = list(range(model_config.num_classes))
    
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels_cm)
    
    # Compute per-class precision, recall, F1
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_cm, zero_division=0
    )
    
    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
    )

    # Log comprehensive metrics
    simple_logger("=" * 80, prefix="S5")
    simple_logger("EVALUATION RESULTS", prefix="S5")
    simple_logger("=" * 80, prefix="S5")
    simple_logger(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)", prefix="S5")
    simple_logger(f"Total Samples: {len(y_true):,}", prefix="S5")
    simple_logger("", prefix="S5")
    
    # Per-class metrics
    simple_logger("Per-Class Metrics:", prefix="S5")
    simple_logger("-" * 80, prefix="S5")
    simple_logger(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}", prefix="S5")
    simple_logger("-" * 80, prefix="S5")
    for i, class_name in enumerate(target_names):
        simple_logger(
            f"{class_name:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<12}",
            prefix="S5",
        )
    simple_logger("-" * 80, prefix="S5")
    simple_logger("", prefix="S5")
    
    # Confusion matrix
    simple_logger("Confusion Matrix:", prefix="S5")
    simple_logger(f"{cm}", prefix="S5")
    simple_logger("", prefix="S5")
    # Confusion matrix interpretation based on label mode
    if model_config.num_classes == 2:
        cm_interpretation = (
            "Confusion matrix interpretation:\n"
            "  cm[0,0] = Down→Down\n"
            "  cm[0,1] = Down→Up\n"
            "  cm[1,0] = Up→Down\n"
            "  cm[1,1] = Up→Up"
        )
    elif model_config.num_classes == 3:
        cm_interpretation = (
            "Confusion matrix interpretation:\n"
            "  cm[0,0] = Down→Down, cm[0,1] = Down→Stationary, cm[0,2] = Down→Up\n"
            "  cm[1,0] = Stationary→Down, cm[1,1] = Stationary→Stationary, cm[1,2] = Stationary→Up\n"
            "  cm[2,0] = Up→Down, cm[2,1] = Up→Stationary, cm[2,2] = Up→Up"
        )
    else:
        cm_interpretation = f"Confusion matrix: {model_config.num_classes}x{model_config.num_classes} (rows=true, cols=predicted)"
    
    simple_logger(cm_interpretation, prefix="S5")

    # Save comprehensive results
    results_path = paths.results_dir / "evaluation_results.json"
    results = {
        "overall_accuracy": float(acc),
        "total_samples": int(len(y_true)),
        "per_class_metrics": {
            target_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(target_names))
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels_cm,
        "classification_report": cls_report,
        "label_mode": label_config.label_mode,
        "num_classes": model_config.num_classes,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    simple_logger(f"Saved evaluation results to {results_path}", prefix="S5")
    simple_logger("=" * 80, prefix="S5")


if __name__ == "__main__":
    evaluate()
