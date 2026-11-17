"""
Multi-layer Keras LSTM with dual regression heads.

Architecture
------------
Input  : (batch, lookback, n_features)
Trunk  : stacked LSTM → Dropout → BatchNormalization (per layer)
Heads  :
  * return_output — next-day return / momentum (linear)
  * vol_output    — next-day realized volatility (softplus ≥ 0)
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config
from utils.metrics import ensure_keras_metrics

ensure_keras_metrics()
from utils.metrics import (  # noqa: E402 — after Keras registration
    combined_dual_loss,
    rmse,
    sharpe_proxy_metric,
)


def build_lstm_model(
    lookback: int | None = None,
    n_features: int | None = None,
    lstm_units: Sequence[int] | None = None,
    dropout_rate: float | None = None,
    dense_units: int | None = None,
    learning_rate: float | None = None,
    loss_weight_return: float | None = None,
    loss_weight_vol: float | None = None,
) -> keras.Model:
    """
    Construct and compile the dual-output LSTM.

    Softplus on the volatility head keeps predictions non-negative without
    hard-clipping gradients.
    """
    lookback = lookback or config.LOOKBACK
    n_features = n_features or len(config.FEATURE_COLUMNS)
    lstm_units = tuple(lstm_units or config.LSTM_UNITS)
    dropout_rate = dropout_rate if dropout_rate is not None else config.DROPOUT_RATE
    dense_units = dense_units if dense_units is not None else config.DENSE_UNITS
    learning_rate = learning_rate if learning_rate is not None else config.LEARNING_RATE
    loss_weight_return = (
        loss_weight_return if loss_weight_return is not None else config.LOSS_WEIGHT_RETURN
    )
    loss_weight_vol = (
        loss_weight_vol if loss_weight_vol is not None else config.LOSS_WEIGHT_VOL
    )

    inputs = keras.Input(shape=(lookback, n_features), name="sequence_input")
    x = inputs

    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        x = layers.LSTM(
            units,
            return_sequences=return_sequences,
            name=f"lstm_{i + 1}",
        )(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i + 1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i + 1}")(x)

    shared = layers.Dense(dense_units, activation="relu", name="shared_dense")(x)

    return_output = layers.Dense(1, activation="linear", name="return_output")(shared)
    # Softplus ≈ smooth ReLU; volatility cannot be negative.
    # Bias ≈ -3 → softplus(-3) ≈ 0.05 so early vol preds start near typical daily vol.
    vol_output = layers.Dense(
        1,
        activation="softplus",
        name="vol_output",
        bias_initializer=keras.initializers.Constant(-3.0),
    )(shared)

    model = keras.Model(
        inputs=inputs,
        outputs={"return_output": return_output, "vol_output": vol_output},
        name="lstm_dual_head",
    )

    # Named losses so Keras matches dict outputs / targets.
    losses = {
        "return_output": combined_dual_loss,  # MSE + mild Sharpe-aware term
        "vol_output": keras.losses.MeanSquaredError(name="vol_mse"),
    }
    loss_weights = {
        "return_output": loss_weight_return,
        "vol_output": loss_weight_vol,
    }
    metrics = {
        "return_output": [rmse, sharpe_proxy_metric, "mse"],
        "vol_output": [rmse, "mse"],
    }

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    return model


def set_global_seeds(seed: int | None = None) -> None:
    """Best-effort determinism for NumPy / Python / TensorFlow."""
    import os
    import random

    import numpy as np

    seed = seed if seed is not None else config.RANDOM_SEED
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Prefer deterministic ops when the backend supports them.
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # noqa: BLE001 — not available on all TF builds
        pass
