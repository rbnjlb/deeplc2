# model_deeplob.py

"""
DeepLOB-style CNN + LSTM model in PyTorch with dual-branch architecture.

This module defines the DeepLOB model architecture. It is imported by training
and evaluation scripts (steps/S4_train_model.py, steps/S5_evaluate_model.py).

FEATURE LAYOUT (from S2_preprocess_lob.py):
=============================================
Total features: 4 * num_levels + 2 + 8 = 4*n + 10

RAW LOB FEATURES (indices 0 to 4*n+1, total: 4*n+2 features):
- rel_bid_prices: indices [0, n-1]           # n features: relative bid prices (centered around mid)
- rel_ask_prices: indices [n, 2*n-1]         # n features: relative ask prices (centered around mid)
- log_bid_sizes: indices [2*n, 3*n-1]        # n features: log(1 + bid sizes)
- log_ask_sizes: indices [3*n, 4*n-1]        # n features: log(1 + ask sizes)
- spread: index [4*n]                         # 1 feature: (ask - bid) / mid
- imbalance: index [4*n+1]                    # 1 feature: total_bid_vol / (total_bid_vol + total_ask_vol)

ENGINEERED FEATURES (indices 4*n+2 to 4*n+9, total: 8 features):
- ret_1s: index [4*n+2]                       # 1s mid-price return
- ret_5s: index [4*n+3]                       # 5s mid-price return
- ret_10s: index [4*n+4]                      # 10s mid-price return
- delta_bid_vol: index [4*n+5]                # Change in total bid volume
- delta_ask_vol: index [4*n+6]                # Change in total ask volume
- top3_mean: index [4*n+7]                    # Mean imbalance over top 3 levels
- deep_mean: index [4*n+8]                    # Mean imbalance over deep levels (4+)
- imb_slope: index [4*n+9]                    # top3_mean - deep_mean

For n=10 levels: raw LOB = indices 0-41 (42 features), engineered = indices 42-49 (8 features)

ARCHITECTURE:
=============
Two-branch design:
1. DeepLOB-pure branch: CNN+LSTM on raw LOB features only
   - Input: (batch, seq_len, F_lob) where F_lob = 4*n+2
   - Reshape to (batch, 1, seq_len, F_lob) for 2D convolutions
   - Three conv blocks + Inception module
   - LSTM over time dimension
   - Output: (batch, lstm_hidden_size)

2. Extra-feature branch: MLP on engineered features
   - Input: (batch, seq_len, F_extra) where F_extra = 8
   - Use last time step or pooled representation
   - Small MLP (2 Linear layers)
   - Output: (batch, extra_embedding_size)

3. Merge: Concatenate both branches → final classification head
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn

from config import model_config, data_config


class DeepLOB(nn.Module):
    """
    Dual-branch DeepLOB model:
    - Branch 1: CNN+LSTM on raw LOB features (DeepLOB-pure)
    - Branch 2: MLP on engineered features
    - Merge: Concatenate → classification head
    """
    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int = model_config.num_classes,
        num_levels: int = None,
        lstm_hidden_size: int = None,
        conv_channel_base: int = None,
        use_extra_feature_branch: bool = None,
        extra_feature_embedding_size: int = None,
        extra_feature_use_last_timestep: bool = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Get num_levels from config if not provided
        if num_levels is None:
            num_levels = data_config.num_levels
        
        # Use config defaults if not provided
        if lstm_hidden_size is None:
            lstm_hidden_size = model_config.lstm_hidden_size
        if conv_channel_base is None:
            conv_channel_base = model_config.conv_channel_base
        if use_extra_feature_branch is None:
            use_extra_feature_branch = model_config.use_extra_feature_branch
        if extra_feature_embedding_size is None:
            extra_feature_embedding_size = model_config.extra_feature_embedding_size
        if extra_feature_use_last_timestep is None:
            extra_feature_use_last_timestep = model_config.extra_feature_use_last_timestep
        
        self.use_extra_feature_branch = use_extra_feature_branch
        self.lstm_hidden_size = lstm_hidden_size
        self.extra_feature_use_last_timestep = extra_feature_use_last_timestep
        
        # Feature dimensions
        # Raw LOB: 4*n + 2 (rel_bid_prices, rel_ask_prices, log_bid_sizes, log_ask_sizes, spread, imbalance)
        self.num_lob_features = 4 * num_levels + 2
        # Engineered features: 8 (ret_1s, ret_5s, ret_10s, delta_bid_vol, delta_ask_vol, top3_mean, deep_mean, imb_slope)
        self.num_extra_features = 8
        
        # Verify total features match
        assert self.num_lob_features + self.num_extra_features == num_features, \
            f"Feature mismatch: lob_features={self.num_lob_features}, extra_features={self.num_extra_features}, total={num_features}"
        
        # ===== BRANCH 1: DeepLOB-pure CNN+LSTM on raw LOB features =====
        # Input: (B, L, F_lob) -> unsqueeze -> (B, 1, L, F_lob)
        
        # Conv1: 1 x 2, stride 1 x 2 (collapse neighbor feature pairs)
        # Temporal convolutions use padding='same' to preserve sequence length
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=conv_channel_base,
                kernel_size=(1, 2),
                stride=(1, 2),
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_channel_base),
            nn.Conv2d(in_channels=conv_channel_base, out_channels=conv_channel_base, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_channel_base),
            nn.Conv2d(in_channels=conv_channel_base, out_channels=conv_channel_base, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_channel_base),
        )

        # Conv2: 1 x 2, further compress feature dimension
        # Temporal convolutions use padding='same' to preserve sequence length
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_channel_base,
                out_channels=conv_channel_base,
                kernel_size=(1, 2),
                stride=(1, 2),
            ),
            nn.Tanh(),
            nn.BatchNorm2d(conv_channel_base),
            nn.Conv2d(in_channels=conv_channel_base, out_channels=conv_channel_base, kernel_size=(4, 1), padding='same'),
            nn.Tanh(),
            nn.BatchNorm2d(conv_channel_base),
            nn.Conv2d(in_channels=conv_channel_base, out_channels=conv_channel_base, kernel_size=(4, 1), padding='same'),
            nn.Tanh(),
            nn.BatchNorm2d(conv_channel_base),
        )

        # Conv3: 1 x 10 aggregates across levels (or use remaining feature dim)
        # Use adaptive kernel size based on remaining feature dimension
        # Temporal convolutions use padding='same' to preserve sequence length
        conv3_kernel_w = min(10, self.num_lob_features // 4)  # Approximate remaining width after conv1/conv2
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_channel_base,
                out_channels=conv_channel_base,
                kernel_size=(1, conv3_kernel_w),
                stride=(1, 1),
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_channel_base),
            nn.Conv2d(in_channels=conv_channel_base, out_channels=conv_channel_base, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_channel_base),
            nn.Conv2d(in_channels=conv_channel_base, out_channels=conv_channel_base, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_channel_base),
        )

        # Inception-style block (parallel convs with multi-scale temporal patterns)
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=conv_channel_base, out_channels=conv_channel_base, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_channel_base),
            nn.Conv2d(in_channels=conv_channel_base, out_channels=conv_channel_base, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_channel_base),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=conv_channel_base, out_channels=conv_channel_base, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_channel_base),
            nn.Conv2d(in_channels=conv_channel_base, out_channels=conv_channel_base, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_channel_base),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=conv_channel_base, out_channels=conv_channel_base, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_channel_base),
        )

        # LSTM over time (per time-step embedding from conv stack)
        # Input size will be computed dynamically: (conv_channel_base * 3) * feature_width_after_convs
        self.lstm = None  # Will be initialized in forward pass after computing input size
        
        # ===== BRANCH 2: Extra-feature MLP branch =====
        if self.use_extra_feature_branch:
            # Small MLP: input -> hidden -> output
            self.extra_branch = nn.Sequential(
                nn.Linear(self.num_extra_features, extra_feature_embedding_size * 2),
                nn.LayerNorm(extra_feature_embedding_size * 2),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(p=0.2),
                nn.Linear(extra_feature_embedding_size * 2, extra_feature_embedding_size),
                nn.LayerNorm(extra_feature_embedding_size),
                nn.LeakyReLU(negative_slope=0.01),
            )
            final_input_size = lstm_hidden_size + extra_feature_embedding_size
        else:
            self.extra_branch = None
            final_input_size = lstm_hidden_size

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)
        
        # Final classification head
        self.fc_out = nn.Linear(final_input_size, num_classes)

    def forward(
        self,
        lob_features: torch.Tensor,
        extra_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with separate feature inputs.
        
        Args:
            lob_features: (B, L, F_lob) tensor of raw LOB features
            extra_features: (B, L, F_extra) tensor of engineered features, or None if use_extra_feature_branch=False
        
        Returns:
            logits: (B, num_classes) raw logits for classification
        """
        B, L, F_lob = lob_features.shape
        assert F_lob == self.num_lob_features, \
            f"Expected {self.num_lob_features} LOB features, got {F_lob}"
        
        # ===== BRANCH 1: DeepLOB-pure CNN+LSTM =====
        # Input shape: (B, L, F_lob) where F_lob = 4*n+2
        # Reshape: (B, L, F_lob) -> (B, 1, L, F_lob) for 2D convolutions
        x_lob = lob_features.unsqueeze(1)  # (B, 1, L, F_lob)
        assert x_lob.shape == (B, 1, L, self.num_lob_features), \
            f"Expected shape (B, 1, L, {self.num_lob_features}), got {x_lob.shape}"
        
        # Conv blocks: progressively reduce feature dimension while preserving temporal dimension
        x_lob = self.conv1(x_lob)  # (B, C, L, F1) where F1 ≈ F_lob/2
        x_lob = self.conv2(x_lob)  # (B, C, L, F2) where F2 ≈ F1/2
        x_lob = self.conv3(x_lob)  # (B, C, L, F3) where F3 ≈ F2 (or aggregated)
        
        # Inception block: multi-scale temporal convolutions (parallel branches)
        x_inp1 = self.inp1(x_lob)  # (B, C, L, F3) - 3x1 temporal kernel
        x_inp2 = self.inp2(x_lob)  # (B, C, L, F3) - 5x1 temporal kernel
        x_inp3 = self.inp3(x_lob)  # (B, C, L, F3) - maxpool + 1x1
        x_lob = torch.cat([x_inp1, x_inp2, x_inp3], dim=1)  # (B, C*3, L, F_final)
        
        # Reshape for LSTM: flatten feature dimension, keep temporal dimension
        B, C_total, L_new, W_new = x_lob.shape
        assert L_new == L, f"Temporal dimension should remain {L}, got {L_new}"
        x_lob = x_lob.permute(0, 2, 1, 3).contiguous()  # (B, L_new, C_total, W_new)
        x_lob = x_lob.view(B, L_new, C_total * W_new)  # (B, L_new, lstm_input_size)
        # Shape assertion: lstm_input_size = C_total * W_new = (conv_channel_base * 3) * F_final
        
        # Initialize LSTM if needed (first forward pass or input size changed)
        lstm_input_size = x_lob.size(-1)
        if self.lstm is None or self.lstm.input_size != lstm_input_size:
            self.lstm = nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=self.lstm_hidden_size,
                num_layers=1,
                batch_first=True,
            ).to(x_lob.device)
        
        # LSTM forward: process temporal sequence
        lstm_out, _ = self.lstm(x_lob)  # (B, L_new, lstm_hidden_size)
        assert lstm_out.shape == (B, L_new, self.lstm_hidden_size), \
            f"LSTM output shape mismatch: expected (B, {L_new}, {self.lstm_hidden_size}), got {lstm_out.shape}"
        
        # Use last time step output (most recent representation)
        lob_embedding = lstm_out[:, -1, :]  # (B, lstm_hidden_size)
        assert lob_embedding.shape == (B, self.lstm_hidden_size), \
            f"LOB embedding shape mismatch: expected (B, {self.lstm_hidden_size}), got {lob_embedding.shape}"
        
        # ===== BRANCH 2: Extra-feature MLP =====
        if self.use_extra_feature_branch:
            if extra_features is None:
                raise ValueError("use_extra_feature_branch=True but extra_features not provided")
            
            B_extra, L_extra, F_extra = extra_features.shape
            assert B_extra == B, f"Batch size mismatch: lob_features={B}, extra_features={B_extra}"
            assert F_extra == self.num_extra_features, \
                f"Expected {self.num_extra_features} extra features, got {F_extra}"
            
            # Use last timestep or mean pooling
            if self.extra_feature_use_last_timestep:
                extra_input = extra_features[:, -1, :]  # (B, F_extra)
            else:
                extra_input = extra_features.mean(dim=1)  # (B, F_extra)
            
            # MLP forward: transform engineered features
            extra_embedding = self.extra_branch(extra_input)  # (B, extra_feature_embedding_size)
            assert extra_embedding.shape == (B, model_config.extra_feature_embedding_size), \
                f"Extra embedding shape mismatch: expected (B, {model_config.extra_feature_embedding_size}), got {extra_embedding.shape}"
            
            # Concatenate branches: combine LOB and extra feature representations
            combined = torch.cat([lob_embedding, extra_embedding], dim=1)  # (B, lstm_hidden_size + extra_feature_embedding_size)
            expected_combined_size = self.lstm_hidden_size + model_config.extra_feature_embedding_size
            assert combined.shape == (B, expected_combined_size), \
                f"Combined embedding shape mismatch: expected (B, {expected_combined_size}), got {combined.shape}"
        else:
            # Pure DeepLOB mode: only use LOB branch
            combined = lob_embedding  # (B, lstm_hidden_size)
            assert combined.shape == (B, self.lstm_hidden_size), \
                f"Combined embedding shape mismatch: expected (B, {self.lstm_hidden_size}), got {combined.shape}"
        
        # Final classification head: map combined representation to class logits
        combined = self.dropout(combined)
        logits = self.fc_out(combined)  # (B, num_classes)
        assert logits.shape == (B, self.num_classes), \
            f"Logits shape mismatch: expected (B, {self.num_classes}), got {logits.shape}"
        
        return logits


def build_deeplob_model(
    seq_len: int,
    num_features: int,
    num_classes: int,
    num_levels: int = None,
) -> DeepLOB:
    """
    Factory function to build the DeepLOB model.
    
    Returns a dual-branch model:
    - Branch 1 (DeepLOB-pure): CNN+LSTM processing raw LOB features (4*n+2 features)
    - Branch 2 (Extra-feature): MLP processing engineered features (8 features)
    - Merge: Concatenate both branches → classification head
    
    Architecture capacity:
    - Conv channels: config.conv_channel_base (default: 32, increased from 16)
    - LSTM hidden size: config.lstm_hidden_size (default: 64, increased from 32)
    - Extra feature embedding: config.extra_feature_embedding_size (default: 16)
    
    The model can run in pure DeepLOB mode by setting config.use_extra_feature_branch=False.
    """
    return DeepLOB(
        seq_len=seq_len,
        num_features=num_features,
        num_classes=num_classes,
        num_levels=num_levels,
    )


if __name__ == "__main__":
    """
    Test/verification script for the DeepLOB model.
    
    Run this to verify the model architecture is correct:
        python -m lob.model_deeplob
    """
    import sys
    from pathlib import Path
    
    # Add parent directory to path to import config
    sys.path.insert(0, str(Path(__file__).parent))
    
    from config import model_config, data_config, label_config
    
    print("=" * 80)
    print("DEEPLOB MODEL VERIFICATION")
    print("=" * 80)
    
    # Get configuration
    seq_len = label_config.seq_len
    num_levels = data_config.num_levels
    num_lob_features = 4 * num_levels + 2
    num_extra_features = 8
    num_features = num_lob_features + num_extra_features
    num_classes = model_config.num_classes
    
    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of levels: {num_levels}")
    print(f"  LOB features: {num_lob_features}")
    print(f"  Extra features: {num_extra_features}")
    print(f"  Total features: {num_features}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Extra feature branch: {model_config.use_extra_feature_branch}")
    print(f"  LSTM hidden size: {model_config.lstm_hidden_size}")
    print(f"  Conv channel base: {model_config.conv_channel_base}")
    
    # Build model
    print(f"\nBuilding model...")
    model = build_deeplob_model(
        seq_len=seq_len,
        num_features=num_features,
        num_classes=num_classes,
        num_levels=num_levels,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = 4
    
    # Create dummy inputs
    lob_features = torch.randn(batch_size, seq_len, num_lob_features)
    if model_config.use_extra_feature_branch:
        extra_features = torch.randn(batch_size, seq_len, num_extra_features)
        print(f"  Input shapes: lob_features={lob_features.shape}, extra_features={extra_features.shape}")
    else:
        extra_features = None
        print(f"  Input shapes: lob_features={lob_features.shape}, extra_features=None")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(lob_features, extra_features)
    
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Expected shape: ({batch_size}, {num_classes})")
    
    if logits.shape == (batch_size, num_classes):
        print(f"\n✓ Model verification successful!")
        print(f"  Model is ready for training.")
    else:
        print(f"\n✗ Model verification failed!")
        print(f"  Expected output shape ({batch_size}, {num_classes}), got {logits.shape}")
        sys.exit(1)
    
    print("=" * 80)
