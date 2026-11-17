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
# Default universe — a small, liquid multi-ticker set that downloads quickly.
# Always treated as a list end-to-end (train / evaluate / walk-forward).
DEFAULT_TICKERS: list[str] = ["AAPL", "MSFT", "GOOGL"]

# yfinance history window. Use period OR (start, end); period wins if both set.
DATA_PERIOD: str = "5y"
DATA_START: str | None = None
DATA_END: str | None = None
DATA_INTERVAL: str = "1d"


def parse_tickers(values: list[str] | str | None = None) -> list[str]:
    """
    Normalize CLI / config tickers to a de-duplicated uppercase list.

    Accepts ``None`` (→ defaults), a single string (``\"AAPL,MSFT\"`` or
    ``\"AAPL MSFT\"``), or an argparse ``nargs=\"+\"`` list that may itself
    contain comma-separated tokens.
    """
    if values is None:
        return list(DEFAULT_TICKERS)
    if isinstance(values, str):
        raw = [values]
    else:
        raw = list(values)
    if not raw:
        return list(DEFAULT_TICKERS)

    out: list[str] = []
    seen: set[str] = set()
    for token in raw:
        # Allow "AAPL,MSFT" or "AAPL;MSFT" alongside space-separated nargs.
        for part in token.replace(";", ",").split(","):
            ticker = part.strip().upper()
            if ticker and ticker not in seen:
                seen.add(ticker)
                out.append(ticker)
    return out or list(DEFAULT_TICKERS)

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

# ---------------------------------------------------------------------------
# Walk-forward validation (time-ordered; no shuffle across the split)
# ---------------------------------------------------------------------------
# Expanding: train = [start .. test_start); Rolling: fixed-length train window
# ending at test_start. Test folds partition the post-min-train calendar.
WF_N_SPLITS: int = 4
WF_MODE: str = "expanding"  # "expanding" | "rolling"
WF_MIN_TRAIN_FRACTION: float = 0.50  # earliest fold's train ends here (by date count)
WF_ROLLING_TRAIN_FRACTION: float = 0.50  # rolling mode: train window length / n_dates
WF_EPOCHS: int = 15  # epochs per fold when evaluate --walk-forward retrains
