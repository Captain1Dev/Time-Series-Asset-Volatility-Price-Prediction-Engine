"""
Custom losses and evaluation helpers for the dual-target LSTM.

Targets
-------
* return head  — next-day simple return (momentum proxy)
* vol head     — next-day rolling / realized volatility

The Sharpe-ratio proxy rewards predicted-return series that have a favourable
mean / std ratio; it is a soft regulariser, not a portfolio optimiser.

NumPy helpers are importable without TensorFlow. Keras losses/metrics are
registered lazily on first access (or via ``ensure_keras_metrics()``).
"""

from __future__ import annotations

import numpy as np

import config

_KERAS_READY = False


# ---------------------------------------------------------------------------
# NumPy helpers (evaluation / reporting) — no TensorFlow required
# ---------------------------------------------------------------------------

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def rmse_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error (NumPy)."""
    return float(np.sqrt(mse(y_true, y_pred)))


def sharpe_proxy(
    returns: np.ndarray,
    annualization: float | None = None,
    eps: float = 1e-8,
) -> float:
    """
    Annualised Sharpe-style ratio of a return series (risk-free ≈ 0).

    Used as a backtest summary statistic, not a trading signal itself.
    """
    annualization = annualization if annualization is not None else config.ANNUALIZATION_FACTOR
    r = np.asarray(returns, dtype=np.float64).ravel()
    if r.size == 0:
        return 0.0
    mu = np.mean(r)
    sigma = np.std(r)
    return float((mu / (sigma + eps)) * np.sqrt(annualization))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of samples where sign(pred) matches sign(true)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0:
        return 0.0
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


# ---------------------------------------------------------------------------
# Keras metrics / losses (lazy — importing this module must not load TF)
# ---------------------------------------------------------------------------

def ensure_keras_metrics():
    """Register Keras serializable losses/metrics; safe to call multiple times."""
    global _KERAS_READY, rmse, sharpe_proxy_metric, combined_dual_loss
    if _KERAS_READY:
        return

    import tensorflow as tf
    from tensorflow import keras

    @keras.utils.register_keras_serializable(package="tsav")
    def _rmse(y_true, y_pred):
        """Keras RMSE metric compatible with dual-output training."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    @keras.utils.register_keras_serializable(package="tsav")
    def _sharpe_proxy_metric(y_true, y_pred):
        """
        Batch-wise Sharpe proxy of *predicted* returns (higher is better).

        Reported as a metric; the training loss uses a complementary penalty.
        """
        y_pred = tf.cast(y_pred, tf.float32)
        mu = tf.reduce_mean(y_pred)
        sigma = tf.math.reduce_std(y_pred)
        return (mu / (sigma + 1e-8)) * tf.sqrt(
            tf.constant(config.ANNUALIZATION_FACTOR, tf.float32)
        )

    @keras.utils.register_keras_serializable(package="tsav")
    def _combined_dual_loss(y_true, y_pred):
        """
        MSE on returns plus a mild penalty that discourages low Sharpe of predictions.

        L = MSE(y, ŷ) + λ * softplus( -Sharpe(ŷ) )
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        mse_term = tf.reduce_mean(tf.square(y_true - y_pred))

        mu = tf.reduce_mean(y_pred)
        sigma = tf.math.reduce_std(y_pred)
        sharpe = (mu / (sigma + 1e-8)) * tf.sqrt(
            tf.constant(config.ANNUALIZATION_FACTOR, tf.float32)
        )
        lambda_sharpe = tf.constant(0.01, tf.float32)
        sharpe_penalty = lambda_sharpe * tf.nn.softplus(-sharpe)

        return mse_term + sharpe_penalty

    rmse = _rmse
    sharpe_proxy_metric = _sharpe_proxy_metric
    combined_dual_loss = _combined_dual_loss
    _KERAS_READY = True


def __getattr__(name: str):
    """Lazy attribute access for Keras callables so `from utils.metrics import rmse` works."""
    if name in {"rmse", "sharpe_proxy_metric", "combined_dual_loss"}:
        ensure_keras_metrics()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
