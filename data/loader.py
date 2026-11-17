"""
OHLCV ingestion, cleaning, technical indicators, scaling, and sliding windows.

Design notes
------------
* Features are scaled with statistics fit on the **training** split only
  to avoid leakage into validation / test.
* Targets (next-day return & volatility) are left in raw space so that
  evaluation metrics (RMSE, Sharpe proxy) stay interpretable.
* Non-stationarity is addressed via returns / log-returns and rolling
  volatility rather than raw price levels alone; prices are still kept
  as features after scaling.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config

logger = logging.getLogger(__name__)

# Prefer a concrete Chrome profile — alias "chrome" can resolve to a build
# (e.g. chrome152) that curl_cffi occasionally rejects as unsupported.
_CURL_IMPERSONATE_CANDIDATES = ("chrome131", "chrome124", "chrome120", "chrome")


def _make_yf_session():
    """Build a curl_cffi session with a known-good browser impersonation."""
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        return None

    last_err: Exception | None = None
    for profile in _CURL_IMPERSONATE_CANDIDATES:
        try:
            session = curl_requests.Session(impersonate=profile)
            # Probe that the profile is actually usable in this process.
            session.get("https://example.com", timeout=5)
            logger.debug("Using curl_cffi impersonate=%s", profile)
            return session
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.debug("impersonate=%s unavailable: %s", profile, exc)
    if last_err:
        logger.warning("Could not create curl_cffi session (%s); using yfinance default.", last_err)
    return None


# ---------------------------------------------------------------------------
# Fetch & clean
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    tickers: Iterable[str] | None = None,
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
    interval: str | None = None,
    max_retries: int = 4,
) -> pd.DataFrame:
    """
    Download daily OHLCV for one or more tickers via yfinance.

    Retries on empty responses (Yahoo / curl_cffi crumb flakes are common)
    and falls back to per-ticker ``Ticker.history`` when the bulk download
    fails.

    Returns a MultiIndex (date, ticker) DataFrame with lowercase columns:
    open, high, low, close, volume, adj_close (when available).
    """
    tickers = list(tickers or config.DEFAULT_TICKERS)
    period = period if period is not None else config.DATA_PERIOD
    start = start if start is not None else config.DATA_START
    end = end if end is not None else config.DATA_END
    interval = interval or config.DATA_INTERVAL

    logger.info("Downloading OHLCV for %s …", tickers)
    raw = _download_with_retries(
        tickers=tickers,
        period=period,
        start=start,
        end=end,
        interval=interval,
        max_retries=max_retries,
    )

    frames: list[pd.DataFrame] = []
    # yfinance returns MultiIndex columns for multi-ticker (and often single) downloads.
    if isinstance(raw.columns, pd.MultiIndex):
        # Columns are typically (field, ticker) with names Price / Ticker.
        for ticker in tickers:
            try:
                df = raw.xs(ticker, axis=1, level=-1).copy()
            except KeyError:
                # Fallback: (ticker, field) layout in some yfinance versions.
                try:
                    df = raw.xs(ticker, axis=1, level=0).copy()
                except KeyError:
                    logger.warning("Ticker %s missing from download; skipping.", ticker)
                    continue
            df = _normalize_ohlcv_columns(df)
            df["ticker"] = ticker
            frames.append(df)
    else:
        # Single-ticker flat columns.
        df = _normalize_ohlcv_columns(raw.copy())
        df["ticker"] = tickers[0]
        frames.append(df)

    if not frames:
        raise RuntimeError("No usable OHLCV frames after download.")

    data = pd.concat(frames, axis=0)
    data.index = pd.to_datetime(data.index)
    data.index.name = "date"
    data = data.reset_index().set_index(["date", "ticker"]).sort_index()
    data = _clean_ohlcv(data)
    logger.info(
        "Fetched %d rows across %d tickers.",
        len(data),
        data.index.get_level_values("ticker").nunique(),
    )
    return data


def _download_with_retries(
    tickers: list[str],
    period: str | None,
    start: str | None,
    end: str | None,
    interval: str,
    max_retries: int,
) -> pd.DataFrame:
    """Bulk ``yf.download`` with exponential backoff, then per-ticker fallback."""
    session = _make_yf_session()
    download_kwargs: dict = {
        "tickers": " ".join(tickers),
        "interval": interval,
        "auto_adjust": False,
        "progress": False,
        "threads": False,  # more reliable with crumb / session quirks
    }
    if session is not None:
        download_kwargs["session"] = session
    if start or end:
        download_kwargs["start"] = start
        download_kwargs["end"] = end
    else:
        download_kwargs["period"] = period

    for attempt in range(1, max_retries + 1):
        try:
            # Fresh session each retry in case the previous crumb/TLS state is bad.
            if attempt > 1:
                session = _make_yf_session()
                if session is not None:
                    download_kwargs["session"] = session
            raw = yf.download(**download_kwargs)
        except Exception as exc:  # noqa: BLE001 — network / impersonation flakes
            logger.warning("yfinance download attempt %d failed: %s", attempt, exc)
            raw = pd.DataFrame()

        if raw is not None and not raw.empty:
            return raw

        sleep_s = min(2 ** (attempt - 1), 8)
        logger.warning(
            "Empty yfinance response (attempt %d/%d); retrying in %ss …",
            attempt,
            max_retries,
            sleep_s,
        )
        time.sleep(sleep_s)

    logger.warning("Bulk download failed — falling back to per-ticker history().")
    return _download_per_ticker(tickers, period=period, start=start, end=end, interval=interval)


def _download_per_ticker(
    tickers: list[str],
    period: str | None,
    start: str | None,
    end: str | None,
    interval: str,
) -> pd.DataFrame:
    """Concatenate per-ticker history frames into a MultiIndex column layout."""
    pieces: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        hist_kwargs: dict = {"interval": interval, "auto_adjust": False}
        if start or end:
            hist_kwargs["start"] = start
            hist_kwargs["end"] = end
        else:
            hist_kwargs["period"] = period or "5y"

        hist = pd.DataFrame()
        for attempt in range(1, 4):
            try:
                session = _make_yf_session()
                hist = yf.Ticker(ticker, session=session).history(**hist_kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ticker.history(%s) attempt %d failed: %s", ticker, attempt, exc)
                hist = pd.DataFrame()
            if not hist.empty:
                break
            time.sleep(attempt)

        if hist.empty:
            logger.warning("No history for %s; skipping.", ticker)
            continue
        pieces[ticker] = _normalize_ohlcv_columns(hist)

    if not pieces:
        raise RuntimeError(f"yfinance returned no data for tickers={tickers}")

    # Build (field, ticker) MultiIndex columns to match yf.download layout.
    cols = {}
    for ticker, df in pieces.items():
        for col in df.columns:
            cols[(col, ticker)] = df[col]
    raw = pd.DataFrame(cols)
    raw.columns = pd.MultiIndex.from_tuples(raw.columns, names=["Price", "Ticker"])
    return raw


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase / rename yfinance columns to a stable schema."""
    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns={c: rename.get(c, c.lower().replace(" ", "_")) for c in df.columns})
    # Keep only expected fields when present.
    keep = [c for c in ("open", "high", "low", "close", "volume", "adj_close") if c in df.columns]
    return df[keep]


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all-NaN rows, forward-fill sparse gaps, then drop residual NaNs."""
    df = df.replace([np.inf, -np.inf], np.nan)
    # Per-ticker forward fill for isolated missing bars (holidays / halts).
    df = df.groupby(level="ticker", group_keys=False).ffill()
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    # Guard against non-positive prices / volume.
    price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    df = df[(df[price_cols] > 0).all(axis=1) & (df["volume"] >= 0)]
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def prepare_features(
    ohlcv: pd.DataFrame,
    vol_window: int | None = None,
    ma_windows: tuple[int, ...] | None = None,
    rsi_window: int | None = None,
) -> pd.DataFrame:
    """
    Add returns, rolling volatility, moving averages, RSI, and dual targets.

    Targets are shifted so that row *t* predicts the outcome realized at *t+1*.
    """
    vol_window = vol_window or config.VOL_WINDOW
    ma_windows = ma_windows or config.MA_WINDOWS
    rsi_window = rsi_window or config.RSI_WINDOW

    parts: list[pd.DataFrame] = []
    for ticker, group in ohlcv.groupby(level="ticker"):
        g = group.droplevel("ticker").copy()
        g = _add_indicators(g, vol_window=vol_window, ma_windows=ma_windows, rsi_window=rsi_window)
        g["ticker"] = ticker
        parts.append(g)

    features = pd.concat(parts)
    features.index = pd.MultiIndex.from_arrays(
        [features.index, features.pop("ticker")],
        names=["date", "ticker"],
    )
    features = features.sort_index()
    # Drop warm-up NaNs from rolling indicators and the final target-less row.
    features = features.dropna()
    return features


def _add_indicators(
    g: pd.DataFrame,
    vol_window: int,
    ma_windows: tuple[int, ...],
    rsi_window: int,
) -> pd.DataFrame:
    """Compute indicators and dual targets for a single-ticker frame."""
    close = g["close"]

    # Stationarity-oriented transforms.
    g["return_1d"] = close.pct_change()
    g["log_return"] = np.log(close / close.shift(1))

    # Realized / rolling volatility of returns (annualization optional later).
    g["rolling_vol"] = g["return_1d"].rolling(vol_window).std()

    for w in ma_windows:
        g[f"ma_{w}"] = close.rolling(w).mean()

    g["rsi"] = _rsi(close, window=rsi_window)

    # Dual targets: next-day return (momentum) and next-day rolling vol.
    g[config.TARGET_RETURN] = g["return_1d"].shift(-1)
    g[config.TARGET_VOL] = g["rolling_vol"].shift(-1)
    return g


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# Scaling & sliding windows
# ---------------------------------------------------------------------------

def _make_scaler(scaler_type: str):
    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown scaler_type={scaler_type!r}; use 'minmax' or 'standard'.")


@dataclass
class DatasetBundle:
    """Train / test arrays plus fitted scaler and metadata for evaluation."""

    X_train: np.ndarray
    y_return_train: np.ndarray
    y_vol_train: np.ndarray
    X_test: np.ndarray
    y_return_test: np.ndarray
    y_vol_test: np.ndarray
    scaler: object
    feature_columns: list[str]
    test_index: pd.MultiIndex  # (date, ticker) aligned with test samples
    test_close: np.ndarray  # close prices aligned with test samples (for backtests)
    lookback: int


def build_dataset(
    features: pd.DataFrame | None = None,
    tickers: Iterable[str] | None = None,
    lookback: int | None = None,
    train_ratio: float | None = None,
    scaler_type: str | None = None,
) -> DatasetBundle:
    """
    End-to-end: fetch → features → chronological split → fit scaler on train
    → sliding windows for train and test.
    """
    lookback = lookback or config.LOOKBACK
    train_ratio = train_ratio if train_ratio is not None else config.TRAIN_RATIO
    scaler_type = scaler_type or config.SCALER_TYPE
    feature_columns = list(config.FEATURE_COLUMNS)

    if features is None:
        ohlcv = fetch_ohlcv(tickers=tickers)
        features = prepare_features(ohlcv)

    # Chronological split on the global date axis (shared calendar).
    dates = features.index.get_level_values("date").unique().sort_values()
    split_idx = max(int(len(dates) * train_ratio), 1)
    split_date = dates[split_idx - 1]
    train_df = features.loc[features.index.get_level_values("date") <= split_date]
    test_df = features.loc[features.index.get_level_values("date") > split_date]

    if train_df.empty or test_df.empty:
        raise RuntimeError(
            "Train or test split is empty — widen the history window or lower train_ratio."
        )

    scaler = _make_scaler(scaler_type)
    scaler.fit(train_df[feature_columns].values)

    X_train, y_r_train, y_v_train, _, _ = _window_ticker_frames(
        train_df, feature_columns, scaler, lookback
    )
    X_test, y_r_test, y_v_test, test_index, test_close = _window_ticker_frames(
        test_df, feature_columns, scaler, lookback
    )

    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError(
            f"Not enough rows for lookback={lookback}. "
            "Reduce LOOKBACK or download a longer history."
        )

    logger.info(
        "Dataset ready — train windows: %d, test windows: %d, features: %d, lookback: %d",
        len(X_train),
        len(X_test),
        len(feature_columns),
        lookback,
    )

    return DatasetBundle(
        X_train=X_train,
        y_return_train=y_r_train,
        y_vol_train=y_v_train,
        X_test=X_test,
        y_return_test=y_r_test,
        y_vol_test=y_v_test,
        scaler=scaler,
        feature_columns=feature_columns,
        test_index=test_index,
        test_close=test_close,
        lookback=lookback,
    )


def _window_ticker_frames(
    df: pd.DataFrame,
    feature_columns: list[str],
    scaler,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.MultiIndex, np.ndarray]:
    """
    Build sliding windows independently per ticker so sequences never cross assets.
    """
    X_list: list[np.ndarray] = []
    y_r_list: list[np.ndarray] = []
    y_v_list: list[np.ndarray] = []
    index_tuples: list[tuple] = []
    close_list: list[float] = []

    for ticker, group in df.groupby(level="ticker"):
        g = group.droplevel("ticker").sort_index()
        if len(g) <= lookback:
            continue

        scaled = scaler.transform(g[feature_columns].values)
        returns = g[config.TARGET_RETURN].values
        vols = g[config.TARGET_VOL].values
        closes = g["close"].values
        dates = g.index

        for i in range(lookback, len(g)):
            X_list.append(scaled[i - lookback : i])
            y_r_list.append(returns[i])
            y_v_list.append(vols[i])
            # Align label with the last bar of the window (prediction time t).
            index_tuples.append((dates[i], ticker))
            close_list.append(closes[i])

    if not X_list:
        empty_X = np.empty((0, lookback, len(feature_columns)), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.float32)
        empty_idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
        return empty_X, empty_y, empty_y, empty_idx, np.empty((0,), dtype=np.float32)

    X = np.asarray(X_list, dtype=np.float32)
    y_r = np.asarray(y_r_list, dtype=np.float32)
    y_v = np.asarray(y_v_list, dtype=np.float32)
    idx = pd.MultiIndex.from_tuples(index_tuples, names=["date", "ticker"])
    closes_arr = np.asarray(close_list, dtype=np.float32)
    return X, y_r, y_v, idx, closes_arr


def save_scaler(scaler, path: Path | None = None) -> Path:
    """Persist the fitted feature scaler with joblib."""
    import joblib

    path = path or (config.SCALER_DIR / "feature_scaler.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    return path


def load_scaler(path: Path | None = None):
    import joblib

    path = path or (config.SCALER_DIR / "feature_scaler.joblib")
    return joblib.load(path)
