# Time-Series Asset Volatility & Price Prediction Engine

Modular Python pipeline that forecasts **next-day returns** and **realized volatility** from historical OHLCV using a dual-head LSTM.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Multi-ticker config

Defaults in `config.py` use a small liquid universe: `AAPL MSFT GOOGL`. Tickers are always a **list** end-to-end (fetch → features → windows → train/eval). Override via CLI with spaces and/or commas:

```bash
python3 train.py --tickers AAPL MSFT GOOGL
python3 train.py --tickers AAPL,MSFT --period 5y
python3 evaluate.py --tickers AAPL,MSFT,GOOGL
```

`config.parse_tickers()` uppercases, strips, and de-duplicates. Sliding windows are built **per ticker** so sequences never cross assets.

## Train

```bash
python3 train.py
python3 train.py --tickers AAPL MSFT --epochs 20 --lookback 60 --period 5y
```

Artifacts:

- `artifacts/checkpoints/lstm_dual_head.keras` — model weights
- `artifacts/scalers/feature_scaler.joblib` — feature scaler (fit on train only)
- `artifacts/train_meta.json` — hyperparameters + loss history

## Evaluate (held-out split)

```bash
python3 evaluate.py
python3 evaluate.py --tickers AAPL,MSFT --checkpoint artifacts/checkpoints/lstm_dual_head.keras
```

Compares the model long/flat strategy to **buy-and-hold** and a **moving-average** baseline. Outputs land in `outputs/`:

- `predicted_vs_actual_returns.png`
- `volatility_curves.png`
- `training_loss.png`
- `equity_curves.png`
- `eval_metrics.json`

## Walk-forward validation

Time-ordered folds with **no shuffle** across the train/test date boundary. Scaler is refit on each fold’s train rows only; the model is **retrained per fold**.

**Fold logic**

1. Sort the shared trading calendar across all tickers.
2. Keep the earliest `WF_MIN_TRAIN_FRACTION` of dates as minimum history; partition the remainder into `N` contiguous test blocks.
3. **Expanding** (`--wf-mode expanding`): train = all dates before the test block (history grows).
4. **Rolling** (`--wf-mode rolling`): train = fixed-length window (`WF_ROLLING_TRAIN_FRACTION` of the calendar) ending just before the test block.
5. Windows stay per-ticker; Keras `validation_split` uses `shuffle=False`.

```bash
# Smoke-friendly example
python3 evaluate.py --walk-forward --tickers AAPL --period 2y --lookback 30 \
  --wf-splits 3 --wf-mode expanding --epochs 3

# Default multi-ticker expanding walk-forward
python3 evaluate.py --walk-forward --tickers AAPL,MSFT,GOOGL --wf-splits 4 --wf-mode expanding
```

Writes `outputs/walk_forward_metrics.json` (per-fold + mean/std summary) and `outputs/walk_forward_sharpe.png`.

## Results

Reproduce metrics vs **MA** and **buy-and-hold**:

```bash
# 1) Single chronological holdout (train once, then evaluate)
python3 train.py --tickers AAPL,MSFT --period 5y --epochs 15
python3 evaluate.py --tickers AAPL,MSFT
# → outputs/eval_metrics.json

# 2) Walk-forward (retrain each fold)
python3 evaluate.py --walk-forward --tickers AAPL,MSFT --period 5y \
  --wf-splits 4 --wf-mode expanding --epochs 10
# → outputs/walk_forward_metrics.json
```

**`eval_metrics.json` shape** (keys of interest):

```json
{
  "return_rmse": 0.0,
  "vol_rmse": 0.0,
  "directional_accuracy": 0.0,
  "sharpe_model_strategy": 0.0,
  "sharpe_buy_and_hold": 0.0,
  "sharpe_ma_baseline": 0.0,
  "cum_return_model_strategy": 0.0,
  "cum_return_buy_and_hold": 0.0,
  "cum_return_ma_baseline": 0.0,
  "n_test": 0,
  "tickers": ["AAPL", "MSFT"]
}
```

**`walk_forward_metrics.json` shape**:

```json
{
  "mode": "expanding",
  "n_splits": 3,
  "tickers": ["AAPL"],
  "summary": {
    "mean_return_rmse": 0.0,
    "mean_sharpe_model_strategy": 0.0,
    "mean_sharpe_buy_and_hold": 0.0,
    "mean_sharpe_ma_baseline": 0.0
  },
  "folds": [{ "fold_id": 0, "train_start": "…", "test_end": "…", "return_rmse": 0.0 }]
}
```

Strategies: LSTM / MA are **long/flat** on next-day returns (signal from predicted return sign or price vs MA); buy-and-hold uses realised next-day returns on the same test windows.

### Smoke numbers (example run)

```bash
python3 evaluate.py --walk-forward --tickers AAPL --period 2y --lookback 30 \
  --wf-splits 3 --wf-mode expanding --epochs 3 --batch-size 64
```

Observed summary (AAPL, expanding, 3 folds; market data can revise slightly on re-run):

| Metric | Mean across folds |
|--------|-------------------|
| return RMSE | 0.047 |
| vol RMSE | 0.034 |
| directional accuracy | 0.504 |
| Sharpe LSTM long/flat | −0.49 |
| Sharpe buy-and-hold | 0.05 |
| Sharpe MA(20) | 0.23 |

This smoke uses only 3 epochs — treat it as a wiring check, not a performance claim. Use longer `--period` / `--epochs` for serious comparison.

## Layout

| Path | Role |
|------|------|
| `config.py` | Tickers, windows, hyperparameters, walk-forward defaults |
| `data/loader.py` | yfinance fetch, indicators, scaling, sliding windows, walk-forward folds |
| `models/lstm_model.py` | Multi-layer LSTM + Dropout + BatchNorm, dual outputs |
| `utils/metrics.py` | Custom losses, RMSE, Sharpe proxy |
| `train.py` | Training entrypoint |
| `evaluate.py` | Holdout backtest, walk-forward, plots |

## Notes

- Features are scaled with statistics fit on the **training** split (or fold train) only — no leakage.
- Dual targets: next-day return (momentum) and next-day rolling volatility.
- Deterministic seeds are set where practical (`config.RANDOM_SEED`).
