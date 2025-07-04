# Algorithmic Trading with Machine Learning (WIP)

**Table of Contents**  
1. [Overview](#overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Project Structure](#project-structure)  
5. [Usage](#usage)  
6. [Visualizations & Results](#visualizations--results)  
7. [Model Performance](#model-performance)  
8. [Real-World Potential & Tangible Benefits](#real-world-potential--tangible-benefits)  
9. [Further Innovations & Expansion Plans](#further-innovations--expansion-plans)  
10. [License](#license)  
11. [Acknowledgments](#acknowledgments)

---

## Overview  
This project explores the use of **machine learning**, **reinforcement learning**, and **anomaly detection** to generate and evaluate algorithmic trading strategies on hourly stock data.

The pipeline supports data download, feature engineering, multi-window walkforward validation, model saving, and performance visualization across 50+ tickers.

> ⚠️ **Work in Progress**: This repository is under active development and continues to be updated with additional models, cleaner integration, and improved signal logic.

---

## Features  
- **Stock Data Collection**  
  - Pulls 1-hour interval stock data using `yfinance` (720-day limit).
- **Feature Engineering**  
  - Adds indicators like RSI, MACD, Bollinger Bands, SMA, EMA, and custom log-return + momentum lags.
- **Labeling Framework**  
  - Supports classification (up/down/flat via quantiles) and regression (log-return prediction).
- **Walkforward Training & Testing**  
  - Uses true date-based rolling or expanding windows for realistic performance evaluation.
- **Model Coverage**  
  - Includes XGBoost, LightGBM, Random Forest, Isolation Forest, AutoEncoder, KMeans, and deep RL agents (PPO, SAC, DQN, TD3, A2C).
- **Evaluation Metrics**  
  - Reports Sharpe, Accuracy, Max Drawdown, Final Portfolio Value, and Cumulative Return.

---

## Installation  
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/racoope70/daytrading-with-ml.git
   ```

2. **Navigate to the Project Directory**  
   ```bash
   cd daytrading-with-ml
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

Ensure Python 3.8+ is installed. GPU recommended for reinforcement learning training.

---

## Project Structure  

```
daytrading-with-ml/
|
├── data/                          # Local storage (optional)
├── download/                      # Data download scripts (yfinance)
│   ├── download_hourly_data.py
|
├── features/                      # Technical indicators & custom features
│   ├── feature_engineering.py
│   ├── calculate_rsi.py
│   ├── generate_trade_labels.py
|
├── models/                        # Model wrappers and helpers
│   ├── xgboost_model.py
│   ├── lightgbm_model.py
│   ├── rl_envs/                   # Custom RL environments
|
├── walkforward/                   # Walkforward scripts
│   ├── train_xgboost_walkforward.py
│   ├── train_ppo_walkforward.py
│   ├── train_td3_walkforward.py
|
├── notebooks/                     # EDA and prototyping
│   ├── anomaly_detection_visuals.ipynb
│   ├── portfolio_comparison.ipynb
|
├── results/                       # Saved plots, metrics, and models
│   ├── metrics/
│   ├── plots/
│   ├── models/
|
├── requirements.txt               # Dependency list
├── LICENSE                        # License file
├── README.md                      # Project documentation (this file)
```

---

## Usage  

**Option 1: Notebook Mode**  
Explore individual models or strategies inside `notebooks/`, e.g., PPO walkforward or KMeans anomaly detection.

**Option 2: Script Mode**

1. **Download Data**
   ```bash
   python download/download_hourly_data.py
   ```

2. **Generate Features**
   ```bash
   python features/feature_engineering.py
   ```

3. **Train Models**
   ```bash
   python walkforward/train_xgboost_walkforward.py
   ```

4. **Visualize Results**
   ```bash
   python utils/plot_portfolio.py
   ```

---

## Visualizations & Results  

**Portfolio Growth (Example: AAPL + PPO)**  
![Portfolio Growth](results/plots/aapl_ppo_portfolio.png)

**Confusion Matrix**  
Illustrates prediction performance on directional signals across walkforward splits.

**Signal Overlay Plot**  
Shows buy/sell signal placement on price chart, aligned with true and predicted labels.

---

## Model Performance  

**Training Accuracy (Example: XGBoost ~87%)**  
Model correctly learned patterns on in-sample data.

**Test Accuracy (Walkforward Windows ~74–83%)**  
Out-of-sample accuracy for directional signals.

**Sharpe Ratio (~1.2 - 2.5)**  
Reflects risk-adjusted return based on hourly returns.

**Max Drawdown (~8-15%)**  
Measures the largest equity dip during the evaluation.

**Final Portfolio Value**  
Total capital after executing trading signals across the test window.

---

## Real-World Potential & Tangible Benefits  

This project has applications across:

- **Retail Algorithmic Trading**  
  Run systematic strategies using public stock data and custom features.

- **Quantitative Research**  
  Rapidly experiment with ML, anomaly detection, and RL to generate trade signals.

- **Portfolio Analysis Tools**  
  Evaluate strategies using risk-return metrics, overlays, and walkforward simulation.

- **Educational Use Case**  
  Learn and teach ML + RL in financial market contexts.

---

## Further Innovations & Expansion Plans  
- **Live Testing Support**  
  Integrate with backtest engines like **RealTest** or platforms like **QuantConnect**.

- **Multi-Agent Reinforcement Learning**  
  Introduce cooperative agents per asset cluster or sector.

- **Execution Layer**  
  Add slippage, commission, and market constraints to refine signal realism.

- **AutoML for Strategy Selection**  
  Compare model types per ticker using dynamic selector logic.

- **Cloud Optimization**  
  Improve memory usage and GPU utilization for Colab + Drive workflows.

---

## License  
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments  
- **Yahoo Finance** via `yfinance`  
- **TA-Lib**, `pandas_ta`, `scikit-learn`, `xgboost`, `lightgbm`, `stable-baselines3`  
- **OpenAI Gym** for reinforcement learning environments  
- **Google Colab** and **GitHub** for dev environment support  
- Thanks to all community contributors!
