# Quick Start Guide

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data:**
   - Copy your Binance BTCUSDT LOB CSV files to `data/raw/`
   - Example: `BTCUSDT_lob_1s_10levels_year.csv`
   - The script expects columns: `timestamp`, `bid_px_1`, `bid_qty_1`, ..., `ask_px_10`, `ask_qty_10`

## Run the Pipeline

All commands should be run from the project root (`my_deeplob_project/`).

### Step 0 (optional): Filter raw data to February only
```bash
python steps/S0_filter_raw_data.py
```

### Step 1: Verify data format / download hooks
```bash
python steps/S1_download_data.py
```

### Step 2: Preprocess LOB data
```bash
python steps/S2_preprocess_lob.py
```
Outputs `data/preprocessed/lob_preprocessed_btcusdt.npz`

### (Optional) Quick quality check on preprocessing
```bash
python steps/S2b_check_data_quality.py
```

### Step 3: Build sequences and labels
```bash
python steps/S3_build_sequences_and_labels.py
```
Outputs sharded sequence files under `data/sequences/` (train/val/test shards + shard_index.json)

### Step 4: Train model
```bash
python steps/S4_train_model.py
```
Saves best model to `models/deeplob_btcusdt.pt` and logs to `logs/`

### Step 5: Evaluate model
```bash
python steps/S5_evaluate_model.py
```
Saves metrics to `results/evaluation_results.json`

## Configuration

Edit `config.py` to adjust:
- `num_levels`: Number of LOB levels (default: 10)
- `seq_len`: Sequence length in seconds (default: 100)
- `pred_horizon`: Prediction horizon in seconds (default: 10)
- `threshold`: Price change threshold for labels (default: 0.0001)
- `batch_size`, `num_epochs`, `learning_rate`: Training hyperparameters

## Using Existing Data

If you have CSV files in `lune/CNN-LSTM/`, you can:

1. Create a symlink or copy:
   ```bash
   cp lune/CNN-LSTM/BTCUSDT_lob_1s_10levels_year.csv data/raw/
   ```

2. Or modify `config.py` to point to a different location:
   ```python
   raw_data_dir: Path = Path("../lune/CNN-LSTM")
   ```

## Troubleshooting

- **Column naming mismatch**: Check `config.py` → `data_config.column_naming`
  - Use `"px_qty"` for `bid_px_1`, `bid_qty_1` format
  - Use `"price_size"` for `bid_price_1`, `bid_size_1` format

- **Out of memory**: Reduce `batch_size` in `config.py`

- **Training too slow**: Reduce `num_epochs` or use GPU if available
