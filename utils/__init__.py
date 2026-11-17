"""Shared metrics and loss helpers.

NumPy helpers are re-exported eagerly. Keras losses/metrics are available via
``utils.metrics`` and load TensorFlow only when accessed.
"""

from utils.metrics import (
    directional_accuracy,
    mse,
    rmse_numpy,
    sharpe_proxy,
)

__all__ = [
    "combined_dual_loss",
    "directional_accuracy",
    "mse",
    "rmse",
    "rmse_numpy",
    "sharpe_proxy",
    "sharpe_proxy_metric",
]


def __getattr__(name: str):
    if name in {"combined_dual_loss", "rmse", "sharpe_proxy_metric"}:
        from utils import metrics as _metrics

        _metrics.ensure_keras_metrics()
        return getattr(_metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
