#!/usr/bin/env python3
"""
Evaluate a trained dual-head LSTM against simple baselines and save plots.

Baselines
---------
* Buy-and-hold — hold the asset; realised next-day returns.
* Moving-average — long when close > MA(N), else flat (0 return).

Walk-forward
------------
``--walk-forward`` retrains on each time-ordered fold (expanding or rolling),
scores the fold test window, and aggregates metrics. No shuffle across the
train/test date boundary.

Usage
-----
    python evaluate.py
    python evaluate.py --tickers AAPL,MSFT --checkpoint artifacts/checkpoints/lstm_dual_head.keras
    python evaluate.py --walk-forward --tickers AAPL MSFT --wf-splits 3 --wf-mode expanding --epochs 5
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
from data.loader import build_dataset, iter_walk_forward_folds
from utils.metrics import directional_accuracy, mse, rmse_numpy, sharpe_proxy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("evaluate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate dual-head LSTM and plot results.")
    p.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help=(
            "One or more tickers (space- or comma-separated). "
            "Default: train_meta tickers, else config.DEFAULT_TICKERS"
        ),
    )
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
    # Walk-forward
    p.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run time-ordered walk-forward folds (retrain per fold; no shuffle leakage)",
    )
    p.add_argument("--wf-splits", type=int, default=config.WF_N_SPLITS)
    p.add_argument(
        "--wf-mode",
        choices=["expanding", "rolling"],
        default=config.WF_MODE,
        help="Expanding (growing train) or rolling (fixed train length)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=config.WF_EPOCHS,
        help="Epochs per fold when --walk-forward is set",
    )
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
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

    ``closes`` must be a contiguous single-ticker series (callers group by ticker).
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


def _predict_heads(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_preds = model.predict(X, verbose=0)
    if isinstance(raw_preds, dict):
        r_pred = np.asarray(raw_preds["return_output"]).ravel()
        v_pred = np.asarray(raw_preds["vol_output"]).ravel()
    else:
        r_pred = np.asarray(raw_preds[0]).ravel()
        v_pred = np.asarray(raw_preds[1]).ravel()
    return r_pred, v_pred


def compute_backtest_metrics(
    y_r: np.ndarray,
    y_v: np.ndarray,
    r_pred: np.ndarray,
    v_pred: np.ndarray,
    test_index: pd.MultiIndex,
    test_close: np.ndarray,
    tickers: list[str],
    lookback: int,
) -> dict:
    """RMSE / directional metrics plus per-ticker strategy Sharpes vs baselines."""
    metrics: dict = {
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

    model_strat_parts: list[np.ndarray] = []
    bh_parts: list[np.ndarray] = []
    ma_parts: list[np.ndarray] = []

    for ticker in pd.Index(test_index.get_level_values("ticker")).unique():
        mask = test_index.get_level_values("ticker") == ticker
        c = test_close[mask]
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
    metrics["_model_strat"] = model_strat
    metrics["_buy_hold"] = buy_hold
    metrics["_ma_strat"] = ma_strat
    return metrics


def _strip_private(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


def evaluate(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.DATA_PERIOD = args.period

    meta = _load_meta(args.meta)
    tickers = config.parse_tickers(args.tickers or meta.get("tickers"))
    lookback = args.lookback or meta.get("lookback") or config.LOOKBACK

    if args.walk_forward:
        return _run_walk_forward(args, tickers=tickers, lookback=lookback)

    if not args.checkpoint.exists():
        logger.error(
            "Checkpoint not found at %s — run `python train.py` first "
            "(or use --walk-forward to retrain per fold).",
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
    model = keras.models.load_model(
        args.checkpoint,
        custom_objects={
            "combined_dual_loss": combined_dual_loss,
            "rmse": rmse,
            "sharpe_proxy_metric": sharpe_proxy_metric,
        },
        compile=False,
    )

    r_pred, v_pred = _predict_heads(model, bundle.X_test)
    y_r = bundle.y_return_test.ravel()
    y_v = bundle.y_vol_test.ravel()

    metrics = compute_backtest_metrics(
        y_r,
        y_v,
        r_pred,
        v_pred,
        bundle.test_index,
        bundle.test_close,
        tickers=tickers,
        lookback=lookback,
    )
    model_strat = metrics.pop("_model_strat")
    buy_hold = metrics.pop("_buy_hold")
    ma_strat = metrics.pop("_ma_strat")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics → %s\n%s", metrics_path, json.dumps(metrics, indent=2))

    sns.set_theme(style="whitegrid", context="talk")
    _plot_predicted_vs_actual(y_r, r_pred, args.output_dir / "predicted_vs_actual_returns.png")
    _plot_volatility_curves(y_v, v_pred, args.output_dir / "volatility_curves.png")
    _plot_training_loss(meta, args.output_dir / "training_loss.png")
    _plot_equity_curves(model_strat, buy_hold, ma_strat, args.output_dir / "equity_curves.png")

    logger.info("Figures saved under %s", args.output_dir)
    return 0


def _run_walk_forward(args: argparse.Namespace, tickers: list[str], lookback: int) -> int:
    """Retrain + score each time-ordered fold; write aggregated metrics JSON."""
    logger.info(
        "Walk-forward: mode=%s splits=%d tickers=%s lookback=%d epochs=%d",
        args.wf_mode,
        args.wf_splits,
        tickers,
        lookback,
        args.epochs,
    )

    # Fetch features before TensorFlow import.
    from data.loader import fetch_ohlcv, prepare_features  # noqa: PLC0415

    ohlcv = fetch_ohlcv(tickers=tickers)
    features = prepare_features(ohlcv)

    folds = list(
        iter_walk_forward_folds(
            features=features,
            tickers=tickers,
            n_splits=args.wf_splits,
            mode=args.wf_mode,
            lookback=lookback,
        )
    )
    if not folds:
        logger.error("No walk-forward folds produced — check history length / --wf-splits.")
        return 1

    from tensorflow import keras  # noqa: PLC0415
    from models.lstm_model import build_lstm_model, set_global_seeds  # noqa: PLC0415

    set_global_seeds(args.seed)

    fold_rows: list[dict] = []
    for fold in folds:
        bundle = fold.bundle
        logger.info(
            "Training fold %d (train_n=%d test_n=%d) …",
            fold.fold_id,
            len(bundle.X_train),
            len(bundle.X_test),
        )
        model = build_lstm_model(
            lookback=bundle.lookback,
            n_features=len(bundle.feature_columns),
            learning_rate=args.learning_rate,
        )
        y_train = {
            "return_output": bundle.y_return_train,
            "vol_output": bundle.y_vol_train,
        }
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=max(config.EARLY_STOPPING_PATIENCE // 2, 3),
                restore_best_weights=True,
                verbose=0,
            ),
        ]
        model.fit(
            bundle.X_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=config.VALIDATION_SPLIT,
            callbacks=callbacks,
            shuffle=False,
            verbose=0,
        )

        r_pred, v_pred = _predict_heads(model, bundle.X_test)
        y_r = bundle.y_return_test.ravel()
        y_v = bundle.y_vol_test.ravel()
        metrics = compute_backtest_metrics(
            y_r,
            y_v,
            r_pred,
            v_pred,
            bundle.test_index,
            bundle.test_close,
            tickers=tickers,
            lookback=lookback,
        )
        row = _strip_private(metrics)
        row.update(
            {
                "fold_id": fold.fold_id,
                "mode": fold.mode,
                "train_start": str(fold.train_start.date()),
                "train_end": str(fold.train_end.date()),
                "test_start": str(fold.test_start.date()),
                "test_end": str(fold.test_end.date()),
            }
        )
        fold_rows.append(row)
        logger.info(
            "Fold %d — return_rmse=%.6f vol_rmse=%.6f dir_acc=%.3f "
            "sharpe_model=%.3f sharpe_bh=%.3f sharpe_ma=%.3f",
            fold.fold_id,
            row["return_rmse"],
            row["vol_rmse"],
            row["directional_accuracy"],
            row["sharpe_model_strategy"],
            row["sharpe_buy_and_hold"],
            row["sharpe_ma_baseline"],
        )

    aggregate_keys = [
        "return_rmse",
        "vol_rmse",
        "directional_accuracy",
        "sharpe_model_strategy",
        "sharpe_buy_and_hold",
        "sharpe_ma_baseline",
        "cum_return_model_strategy",
        "cum_return_buy_and_hold",
        "cum_return_ma_baseline",
    ]
    summary: dict = {}
    for key in aggregate_keys:
        vals = np.asarray([r[key] for r in fold_rows], dtype=np.float64)
        summary[f"mean_{key}"] = float(np.mean(vals))
        summary[f"std_{key}"] = float(np.std(vals))

    payload = {
        "mode": args.wf_mode,
        "n_splits": args.wf_splits,
        "n_folds_scored": len(fold_rows),
        "tickers": tickers,
        "lookback": lookback,
        "epochs": args.epochs,
        "period": args.period,
        "summary": summary,
        "folds": fold_rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "walk_forward_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Walk-forward metrics → %s\n%s", out_path, json.dumps(payload["summary"], indent=2))

    _plot_walk_forward_summary(fold_rows, args.output_dir / "walk_forward_sharpe.png")
    return 0


def _plot_walk_forward_summary(fold_rows: list[dict], path: Path) -> None:
    if not fold_rows:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ids = [r["fold_id"] for r in fold_rows]
    ax.plot(ids, [r["sharpe_model_strategy"] for r in fold_rows], marker="o", label="LSTM long/flat")
    ax.plot(ids, [r["sharpe_buy_and_hold"] for r in fold_rows], marker="o", label="Buy & hold")
    ax.plot(ids, [r["sharpe_ma_baseline"] for r in fold_rows], marker="o", label=f"MA({config.MA_BASELINE_WINDOW})")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Sharpe proxy")
    ax.set_title("Walk-forward Sharpe by fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", path)


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
