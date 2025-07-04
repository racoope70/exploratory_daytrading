
# Day Trading With Machine Learning – Project Overview

This project explores the design, implementation, and deployment of machine learning strategies for day trading U.S. equities using high-frequency (hourly) data. The focus is on combining supervised learning, reinforcement learning, and unsupervised methods, with walkforward validation and live paper trading execution through Alpaca.

---

## ✅ Project Goals

- Implement a full end-to-end machine learning pipeline.
- Support both backtesting (QuantConnect) and live trading (Alpaca).
- Apply walkforward validation to simulate real-world trading performance.
- Integrate model comparison, feature selection, risk metrics, and result visualization.

---

## 📂 Directory Structure

```
lightgbm/
├── models/
│   ├── model_AAPL.txt
│   ├── model_MSFT.txt
│   └── ...
├── scalers/
│   ├── scaler_AAPL.pkl
│   ├── scaler_MSFT.pkl
│   └── ...
├── features/
│   ├── features_AAPL.txt
│   ├── features_MSFT.txt
│   └── ...
├── metrics/
│   └── lightgbm_walkforward_summary.csv
├── plots/
│   └── AAPL_portfolio_value.png
```

This structure allows storing trained models, scalers, and feature sets per stock ticker for later retrieval during live trading or analysis.

---

## 📊 LightGBM Model Performance (Top 5)

These are the top-performing stocks by final portfolio value using the walkforward backtesting pipeline:

| Ticker | Final_Portfolio | Return_% | Sharpe | Accuracy | F1_Score | Drawdown |
|--------|----------------:|---------:|--------:|----------:|----------:|----------:|
| MSFT   | 113902.79       | 13.90    | 0.2136 | 0.489     | 0.2919    | 20108.83  |
| AAPL   | 106302.43       | 6.30     | 0.1187 | 0.4907    | 0.2495    | 20349.34  |
| NVDA   | 104822.77       | 4.82     | 0.1132 | 0.4813    | 0.2514    | 21230.21  |
| GOOGL  | 103981.14       | 3.98     | 0.1068 | 0.4721    | 0.2432    | 22594.12  |
| AMZN   | 102764.55       | 2.76     | 0.0983 | 0.4682    | 0.2378    | 23985.00  |

Each model was trained using walkforward splits on 720 days of hourly data and tested on out-of-sample segments.

---

## 🔄 Backtesting Setup (QuantConnect)

LightGBM models were trained directly in QuantConnect using `MinMaxScaler`, 8+ technical indicators, and binary labels for next-hour returns. All models were validated using realistic trading simulations and stored using:

```python
# Save model
model.booster_.save_model("model_AAPL.txt")

# Save scaler
joblib.dump(scaler, "scaler_AAPL.pkl")

# Save features
with open("features_AAPL.txt", "w") as f:
    f.write(",".join(features))
```

These files were then downloaded and stored in the `lightgbm/` GitHub directory for reproducibility.

---

## ⚡ Live Paper Trading (Alpaca)

Alpaca’s API is used to execute the model predictions in a live paper trading environment. The script:

- Loads model + scaler from Google Drive
- Downloads recent 30-day hourly data
- Computes the same technical features
- Submits a market buy order if predicted probability > 0.6 and not already holding the stock
- Skips execution if the market is closed or the signal is weak

```python
# Pseudocode Logic
if market_open:
    if not holding and prob > 0.6:
        buy()
    elif holding and prob <= 0.6:
        sell()
```

---

## 🔧 Enhancements in Progress

- Execution & Slippage Simulation
- Noise Filtering (e.g., wavelet transforms, denoising autoencoders)
- Market Regime Detection (clustering)
- Latency Simulation & Broker Hooks (Alpaca + Interactive Brokers)
- Online Learning Support
- Modular Risk Management Layer (stop-loss, drawdown controls)

---

## 📌 Next Steps

- Finalize paper trading validation for LightGBM top 5 tickers.
- Extend pipeline to PPO, DQN, SAC, and XGBoost with consistent evaluation.
- Integrate unified model selector and publish full leaderboard.
- Continue preparing public release, documentation, and reproducibility.

---

## 🧠 References

- [Alpaca API Docs](https://alpaca.markets/docs/)
- [QuantConnect Lean](https://www.quantconnect.com/docs/)
- [LightGBM Docs](https://lightgbm.readthedocs.io/)
- [Project GitHub Repo](https://github.com/racoope70/daytrading-with-ml)
