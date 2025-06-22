# Day Trading with Machine Learning: Research-Centric Overview

This repository presents a robust and research-driven machine learning framework for algorithmic day trading using high-frequency stock data. It integrates supervised and reinforcement learning models with advanced backtesting and signal labeling strategies across a pipeline exceeding 70 Python scripts. The goal is not only high performance but also methodological integrity, reproducibility, and readiness for real-world trading.

## Project Scope
- **Time Frame**: 720 days of historical 1-hour OHLCV data from Yahoo Finance
- **Assets**: Multi-asset support across 53+ stocks
- **Features**: Over 30 engineered technical indicators
- **Models**:
  - Supervised: XGBoost, LightGBM, Random Forest, LSTM, GRU
  - Reinforcement Learning: SAC, PPO, TD3, Deep Q, Deep SARSA
  - Unsupervised: Autoencoders, Isolation Forest, DBSCAN, KMeans, UMAP

---

## Core Methodologies

### Time-Aware Validation
To simulate realistic trading outcomes and reduce overfitting risk:
- **Walk-Forward Validation**: Models are retrained on rolling or expanding training windows, then evaluated on strictly forward test segments.
- **Multiple Folds**: Performance is aggregated over 10+ rolling steps (e.g., train 660d → test 60d), improving confidence.
- **Gap Between Windows**: A temporal gap is introduced to avoid trailing indicators contaminating test data.
- **Nested TimeSeriesSplit**: Inner folds tune hyperparameters; outer folds evaluate generalization.

### Signal Labeling
- **Dynamic Quantiles**: Buy/Sell thresholds are computed from training data only to prevent look-ahead bias.
- **Triple Barrier Labeling** (Planned): Models will soon incorporate event-based outcomes using stop-loss and take-profit logic for more realistic trade simulation.

### Feature Engineering
- **Redundancy Elimination**: Correlation analysis and feature importance pruning reduce multicollinearity.
- **Rolling Feature Recalculation**: All technical indicators and scalers are recalculated fresh per walk-forward fold to prevent leakage.

### Data Leakage Mitigation
- **Purged Cross-Validation**: Ensures no overlap between label horizons and training data.
- **Strict Temporal Ordering**: No shuffling or random splits. Only scikit-learn’s `TimeSeriesSplit` or manual slicing used.
- **Isolated Feature Scaling**: Standardization is performed only on training data, then applied to test data.

### Unsupervised Learning Validation
- **Anomaly Detection**: Autoencoders and Isolation Forests are trained per walk window; their performance is gauged via return impact of anomaly-filtered strategies.
- **Clustering**: KMeans/DBSCAN clustering is done only on training folds. Clusters are interpreted and tested for regime identification.

### Model Evaluation
- **Metrics**: Sharpe Ratio, Max Drawdown, Accuracy, Final Portfolio Value
- **Variance Reporting**: Metrics include mean/std across multiple seeds to show robustness
- **Baselines**: Compared against simple strategies (e.g., moving average crossover) for context

---

## Project Highlights
- **70+ Modular Scripts**: Each model has dedicated training/evaluation scripts (e.g. `ppo_walkforward.py`, `ae_anomaly.py`, `xgboost_train.py`).
- **Colab-Ready & GPU Optimized**: Built to run on Google Colab with GPU acceleration, automatic memory cleanup, and save/load support to Google Drive.
- **Scalable for Deployment**: Walk-forward architecture is ready for real-time trading logic and model retraining.

---

## Visualization Outputs
- Feature importance bar plots for XGBoost, LightGBM, and Random Forest
- Portfolio growth plots for each model and stock
- Anomaly detection overlays on price data

---

## Future Directions
- Implement Triple Barrier Labeling (profit/loss/timeout exit logic)
- Add strategy-specific stop-loss/take-profit rules during backtests
- Expand asset universe to ETFs, crypto, and FX
- Create a Streamlit-based model selector dashboard
- Deploy live testing on paper trading platforms to validate real-time performance
---

## References
- ForecastEgy: [Time Series CV](https://forecastegy.com/posts/time-series-cross-validation-python/)
- QuantInsti: [Purging & Embargo](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/)
- Medium (Y. Oz): [Triple Barrier Method](https://medium.com/@yairoz/the-triple-barrier-method-labeling-financial-time-series-for-ml-in-elixir-e539301b90d6)
- dotData: [Feature Engineering Leakage](https://dotdata.com/blog/preventing-data-leakage-in-feature-engineering-strategies-and-solutions/)

---

For full implementation, see: [daytrading-with-ml GitHub repo](https://github.com/racoope70/daytrading-with-ml)

