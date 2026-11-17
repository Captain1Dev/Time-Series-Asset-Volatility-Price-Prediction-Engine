#!/usr/bin/env python3
"""
Evaluate a trained dual-head LSTM against simple baselines and save plots.

Baselines
---------
* Buy-and-hold — hold the asset; realised next-day returns.
* Moving-average — long when close > MA(N), else flat (0 return).

Usage
-----
    python evaluate.py
    python evaluate.py --tickers AAPL --checkpoint artifacts/checkpoints/lstm_dual_head.keras
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from data.loader import build_dataset
from utils.metrics import directional_accuracy, mse, rmse_numpy, sharpe_proxy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("evaluate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate dual-head LSTM and plot results.")
    p.add_argument("--tickers", nargs="+", default=None, help="Override tickers (default: from train_meta or config)")
    p.add_argument("--lookback", type=int, default=None)
    p.add_argument("--period", type=str, default=config.DATA_PERIOD)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=config.CHECKPOINT_DIR / config.CHECKPOINT_NAME,
    )
    p.add_argument(
        "--meta",
        type=Path,
        default=config.ARTIFACTS_DIR / "train_meta.json",
        help="Training metadata written by train.py",
    )
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUTS_DIR,
        help="Directory for figures and metrics JSON",
    )
    return p.parse_args(argv)


def _load_meta(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _strategy_returns_from_signal(signal: np.ndarray, realised_returns: np.ndarray) -> np.ndarray:
    """Apply a long/flat signal (1 / 0) to realised next-day returns."""
    return np.asarray(signal, dtype=np.float64).ravel() * np.asarray(realised_returns, dtype=np.float64).ravel()


def moving_average_baseline(closes: np.ndarray, window: int, realised_returns: np.ndarray) -> np.ndarray:
    """
    Long when price > trailing MA, else flat.

    Because windows are per-sample (not a continuous series across tickers),
    we approximate the MA from the close series in evaluation order grouped
    externally; here `closes` is assumed contiguous for one ticker.
    """
    closes = np.asarray(closes, dtype=np.float64)
    signal = np.zeros(len(closes), dtype=np.float64)
    if len(closes) < window:
        return signal * realised_returns
    ma = pd.Series(closes).rolling(window).mean().values
    signal = (closes > ma).astype(np.float64)
    signal = np.nan_to_num(signal, nan=0.0)
    return _strategy_returns_from_signal(signal, realised_returns)


def model_strategy_returns(pred_returns: np.ndarray, realised_returns: np.ndarray) -> np.ndarray:
    """Long when predicted next-day return > 0, else flat."""
    signal = (np.asarray(pred_returns).ravel() > 0).astype(np.float64)
    return _strategy_returns_from_signal(signal, realised_returns)


def evaluate(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.DATA_PERIOD = args.period

    meta = _load_meta(args.meta)
    tickers = args.tickers or meta.get("tickers") or config.DEFAULT_TICKERS
    lookback = args.lookback or meta.get("lookback") or config.LOOKBACK

    if not args.checkpoint.exists():
        logger.error(
            "Checkpoint not found at %s — run `python train.py` first.",
            args.checkpoint,
        )
        return 1

    # Rebuild dataset before importing TensorFlow (Yahoo / curl_cffi reliability).
    logger.info("Rebuilding dataset for evaluation (tickers=%s) …", tickers)
    bundle = build_dataset(tickers=tickers, lookback=lookback)

    from tensorflow import keras  # noqa: PLC0415 — intentional lazy import
    from models.lstm_model import set_global_seeds  # noqa: PLC0415
    from utils.metrics import combined_dual_loss, rmse, sharpe_proxy_metric  # noqa: PLC0415

    set_global_seeds(args.seed)

    logger.info("Loading model from %s", args.checkpoint)
    # Custom objects registered via decorator; load_model still needs them listed
    # for some TF versions.
    model = keras.models.load_model(
        args.checkpoint,
        custom_objects={
            "combined_dual_loss": combined_dual_loss,
            "rmse": rmse,
            "sharpe_proxy_metric": sharpe_proxy_metric,
        },
        compile=False,
    )

    raw_preds = model.predict(bundle.X_test, verbose=0)
    if isinstance(raw_preds, dict):
        r_pred = np.asarray(raw_preds["return_output"]).ravel()
        v_pred = np.asarray(raw_preds["vol_output"]).ravel()
    else:
        # Fallback for list/tuple output order matching model construction.
        r_pred = np.asarray(raw_preds[0]).ravel()
        v_pred = np.asarray(raw_preds[1]).ravel()

    y_r = bundle.y_return_test.ravel()
    y_v = bundle.y_vol_test.ravel()

    # ---- Metrics ----
    metrics = {
        "return_rmse": rmse_numpy(y_r, r_pred),
        "return_mse": mse(y_r, r_pred),
        "vol_rmse": rmse_numpy(y_v, v_pred),
        "vol_mse": mse(y_v, v_pred),
        "directional_accuracy": directional_accuracy(y_r, r_pred),
        "sharpe_predicted_returns": sharpe_proxy(r_pred),
        "sharpe_actual_returns": sharpe_proxy(y_r),
        "n_test": int(len(y_r)),
        "tickers": list(tickers),
        "lookback": int(lookback),
    }

    # Per-ticker backtests so MA baseline is well-defined on contiguous prices.
    index = bundle.test_index
    closes = bundle.test_close
    model_strat_parts: list[np.ndarray] = []
    bh_parts: list[np.ndarray] = []
    ma_parts: list[np.ndarray] = []

    for ticker in pd.Index(index.get_level_values("ticker")).unique():
        mask = index.get_level_values("ticker") == ticker
        c = closes[mask]
        yr = y_r[mask]
        rp = r_pred[mask]
        model_strat_parts.append(model_strategy_returns(rp, yr))
        bh_parts.append(yr.copy())
        ma_parts.append(moving_average_baseline(c, config.MA_BASELINE_WINDOW, yr))

    model_strat = np.concatenate(model_strat_parts) if model_strat_parts else np.array([])
    buy_hold = np.concatenate(bh_parts) if bh_parts else np.array([])
    ma_strat = np.concatenate(ma_parts) if ma_parts else np.array([])

    metrics["sharpe_model_strategy"] = sharpe_proxy(model_strat)
    metrics["sharpe_buy_and_hold"] = sharpe_proxy(buy_hold)
    metrics["sharpe_ma_baseline"] = sharpe_proxy(ma_strat)
    metrics["cum_return_model_strategy"] = float(np.nansum(model_strat))
    metrics["cum_return_buy_and_hold"] = float(np.nansum(buy_hold))
    metrics["cum_return_ma_baseline"] = float(np.nansum(ma_strat))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics → %s\n%s", metrics_path, json.dumps(metrics, indent=2))

    # ---- Plots ----
    sns.set_theme(style="whitegrid", context="talk")
    _plot_predicted_vs_actual(y_r, r_pred, args.output_dir / "predicted_vs_actual_returns.png")
    _plot_volatility_curves(y_v, v_pred, args.output_dir / "volatility_curves.png")
    _plot_training_loss(meta, args.output_dir / "training_loss.png")
    _plot_equity_curves(model_strat, buy_hold, ma_strat, args.output_dir / "equity_curves.png")

    logger.info("Figures saved under %s", args.output_dir)
    return 0


def _plot_predicted_vs_actual(y_true, y_pred, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n = min(len(y_true), 300)
    axes[0].plot(y_true[:n], label="Actual", alpha=0.85, linewidth=1.2)
    axes[0].plot(y_pred[:n], label="Predicted", alpha=0.85, linewidth=1.2)
    axes[0].set_title("Next-day return — predicted vs actual")
    axes[0].set_xlabel("Test sample")
    axes[0].set_ylabel("Return")
    axes[0].legend()

    axes[1].scatter(y_true, y_pred, s=12, alpha=0.4, edgecolors="none")
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    axes[1].plot(lims, lims, "--", color="gray", linewidth=1)
    axes[1].set_title("Return scatter")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", path)


def _plot_volatility_curves(y_true, y_pred, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    n = min(len(y_true), 300)
    ax.plot(y_true[:n], label="Actual rolling vol", linewidth=1.3)
    ax.plot(y_pred[:n], label="Predicted vol", linewidth=1.3)
    ax.set_title("Realized / rolling volatility — predicted vs actual")
    ax.set_xlabel("Test sample")
    ax.set_ylabel("Volatility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", path)


def _plot_training_loss(meta: dict, path: Path) -> None:
    history = meta.get("history") or {}
    if not history.get("loss"):
        logger.warning("No training history in meta — skipping loss plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history["loss"]) + 1)
    ax.plot(epochs, history["loss"], label="train loss")
    if history.get("val_loss"):
        ax.plot(epochs, history["val_loss"], label="val loss")
    # Optional per-head losses when present.
    for key, label in (
        ("return_output_loss", "train return loss"),
        ("val_return_output_loss", "val return loss"),
        ("vol_output_loss", "train vol loss"),
        ("val_vol_output_loss", "val vol loss"),
    ):
        if history.get(key):
            ax.plot(epochs, history[key], label=label, alpha=0.7, linestyle="--")

    ax.set_title("Training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", path)


def _plot_equity_curves(
    model_strat: np.ndarray,
    buy_hold: np.ndarray,
    ma_strat: np.ndarray,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.cumsum(model_strat), label="LSTM long/flat", linewidth=1.4)
    ax.plot(np.cumsum(buy_hold), label="Buy & hold", linewidth=1.4)
    ax.plot(np.cumsum(ma_strat), label=f"MA({config.MA_BASELINE_WINDOW}) long/flat", linewidth=1.4)
    ax.set_title("Cumulative strategy returns (test set)")
    ax.set_xlabel("Test sample")
    ax.set_ylabel("Cumulative return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    raise SystemExit(evaluate())
