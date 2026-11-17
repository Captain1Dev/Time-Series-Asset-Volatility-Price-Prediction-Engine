#!/usr/bin/env python3
"""
Train the dual-head LSTM on historical OHLCV features.

Usage
-----
    python train.py
    python train.py --tickers AAPL --epochs 5 --lookback 40
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run as a script.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import config
from data.loader import build_dataset, save_scaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train dual-head LSTM volatility / return model.")
    p.add_argument(
        "--tickers",
        nargs="+",
        default=config.DEFAULT_TICKERS,
        help="Ticker symbols (default: %(default)s)",
    )
    p.add_argument("--lookback", type=int, default=config.LOOKBACK)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--train-ratio", type=float, default=config.TRAIN_RATIO)
    p.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    p.add_argument("--period", type=str, default=config.DATA_PERIOD, help="yfinance period, e.g. 2y, 5y")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=config.CHECKPOINT_DIR / config.CHECKPOINT_NAME,
        help="Path for the .keras checkpoint",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Honour CLI period override via config mutation for this process.
    config.DATA_PERIOD = args.period

    # Fetch/build data *before* importing TensorFlow — Yahoo crumb / curl_cffi
    # impersonation can flake once TF has initialised native runtimes.
    logger.info("Building dataset for tickers=%s lookback=%d …", args.tickers, args.lookback)
    bundle = build_dataset(
        tickers=args.tickers,
        lookback=args.lookback,
        train_ratio=args.train_ratio,
    )

    from tensorflow import keras  # noqa: PLC0415 — intentional lazy import
    from models.lstm_model import build_lstm_model, set_global_seeds  # noqa: PLC0415

    set_global_seeds(args.seed)

    model = build_lstm_model(
        lookback=bundle.lookback,
        n_features=len(bundle.feature_columns),
        learning_rate=args.learning_rate,
    )
    model.summary(print_fn=logger.info)

    y_train = {
        "return_output": bundle.y_return_train,
        "vol_output": bundle.y_vol_train,
    }

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(args.checkpoint),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(config.EARLY_STOPPING_PATIENCE // 2, 2),
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        bundle.X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    # Ensure final (best-restored) weights are on disk even if checkpoint
    # callback already wrote an earlier best.
    model.save(args.checkpoint)
    scaler_path = save_scaler(bundle.scaler)
    logger.info("Saved model → %s", args.checkpoint)
    logger.info("Saved scaler → %s", scaler_path)

    # Persist lightweight training metadata for evaluate.py / reproducibility.
    meta = {
        "tickers": args.tickers,
        "lookback": bundle.lookback,
        "feature_columns": bundle.feature_columns,
        "train_samples": int(len(bundle.X_train)),
        "test_samples": int(len(bundle.X_test)),
        "epochs_ran": len(history.history.get("loss", [])),
        "final_loss": float(history.history["loss"][-1]) if history.history.get("loss") else None,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "scaler": str(scaler_path),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }
    meta_path = config.ARTIFACTS_DIR / "train_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Wrote training metadata → %s", meta_path)

    # Quick in-process smoke metrics on the held-out test split.
    preds = model.predict(bundle.X_test, verbose=0)
    if isinstance(preds, dict):
        r_pred = np.asarray(preds["return_output"]).ravel()
        v_pred = np.asarray(preds["vol_output"]).ravel()
    else:
        r_pred, v_pred = np.asarray(preds[0]).ravel(), np.asarray(preds[1]).ravel()

    from utils.metrics import rmse_numpy, sharpe_proxy

    logger.info(
        "Test RMSE return=%.6f  vol=%.6f  |  pred-return Sharpe proxy=%.3f",
        rmse_numpy(bundle.y_return_test, r_pred),
        rmse_numpy(bundle.y_vol_test, v_pred),
        sharpe_proxy(r_pred),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
