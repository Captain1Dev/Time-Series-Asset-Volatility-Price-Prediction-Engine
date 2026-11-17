# Time-Series Asset Volatility & Price Prediction Engine

Modular Python pipeline that forecasts **next-day returns** and **realized volatility** from historical OHLCV using a dual-head LSTM.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

Defaults (tickers=`AAPL MSFT GOOGL`, lookback=60, epochs=30, etc.) live in `config.py`. Override via CLI as needed:

```bash
python3 train.py
python3 train.py --tickers AAPL MSFT --epochs 20 --lookback 60 --period 5y
```

Artifacts written to:

- `artifacts/checkpoints/lstm_dual_head.keras` — model weights
- `artifacts/scalers/feature_scaler.joblib` — feature scaler (fit on train only)
- `artifacts/train_meta.json` — hyperparameters + loss history

## Evaluate

```bash
python3 evaluate.py
python3 evaluate.py --tickers AAPL MSFT --checkpoint artifacts/checkpoints/lstm_dual_head.keras
```

Compares the model long/flat strategy to **buy-and-hold** and a **moving-average** baseline. Figures and metrics land in `outputs/`:

- `predicted_vs_actual_returns.png`
- `volatility_curves.png`
- `training_loss.png`
- `equity_curves.png`
- `eval_metrics.json`

## Layout

| Path | Role |
|------|------|
| `config.py` | Tickers, windows, hyperparameters |
| `data/loader.py` | yfinance fetch, indicators, scaling, sliding windows |
| `models/lstm_model.py` | Multi-layer LSTM + Dropout + BatchNorm, dual outputs |
| `utils/metrics.py` | Custom losses, RMSE, Sharpe proxy |
| `train.py` | Training entrypoint |
| `evaluate.py` | Backtest + plots |

## Notes

- Features are scaled with statistics fit on the **training** split only (no leakage).
- Dual targets: next-day return (momentum) and next-day rolling volatility.
- Deterministic seeds are set where practical (`config.RANDOM_SEED`).
