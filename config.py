"""
Central configuration for the volatility / price prediction engine.

Defaults are chosen so `python train.py` and `python evaluate.py` run
without CLI flags. Every value can be overridden via argparse in those scripts.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
SCALER_DIR = ARTIFACTS_DIR / "scalers"

# ---------------------------------------------------------------------------
# Data / tickers
# ---------------------------------------------------------------------------
# Default universe — a small, liquid set that downloads quickly.
DEFAULT_TICKERS: list[str] = ["AAPL", "MSFT", "GOOGL"]

# yfinance history window. Use period OR (start, end); period wins if both set.
DATA_PERIOD: str = "5y"
DATA_START: str | None = None
DATA_END: str | None = None
DATA_INTERVAL: str = "1d"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
LOOKBACK: int = 60  # sliding-window sequence length (trading days)
VOL_WINDOW: int = 21  # realized / rolling volatility window
MA_WINDOWS: tuple[int, ...] = (10, 20, 50)
RSI_WINDOW: int = 14
TRAIN_RATIO: float = 0.80  # chronological split — no shuffle

# Feature columns fed to the model (order matters for scaler / windows).
FEATURE_COLUMNS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return_1d",
    "log_return",
    "rolling_vol",
    "ma_10",
    "ma_20",
    "ma_50",
    "rsi",
]

# Dual regression targets predicted by the LSTM heads.
TARGET_RETURN: str = "target_return"  # next-day simple return (momentum proxy)
TARGET_VOL: str = "target_vol"  # next-day realized / rolling volatility

# Scaler: "minmax" | "standard"
SCALER_TYPE: str = "minmax"

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
LSTM_UNITS: tuple[int, ...] = (64, 32)
DROPOUT_RATE: float = 0.2
DENSE_UNITS: int = 16
LEARNING_RATE: float = 1e-3

# Loss weights for the dual-output heads (return, volatility).
LOSS_WEIGHT_RETURN: float = 1.0
LOSS_WEIGHT_VOL: float = 1.0

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPOCHS: int = 30
BATCH_SIZE: int = 32
VALIDATION_SPLIT: float = 0.1  # fraction of *train* used for Keras validation
EARLY_STOPPING_PATIENCE: int = 8
RANDOM_SEED: int = 42

# Checkpoint filename (SavedModel / Keras 3 native format preferred).
CHECKPOINT_NAME: str = "lstm_dual_head.keras"

# ---------------------------------------------------------------------------
# Evaluation / baselines
# ---------------------------------------------------------------------------
MA_BASELINE_WINDOW: int = 20  # moving-average strategy lookback
ANNUALIZATION_FACTOR: float = 252.0  # trading days for Sharpe proxy
