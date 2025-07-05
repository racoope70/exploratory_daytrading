[Top_5_Performing_LightGBM_Stocks.csv](https://github.com/user-attachments/files/21070458/Top_5_Performing_LightGBM_Stocks.csv)# Algorithmic Trading with Machine Learning (WIP)

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

Sharpe Ratio Distribution per Model
![image](https://github.com/user-attachments/assets/ff63c961-c454-468f-909c-79abd30e6743)

Top 5 and Bottom 5 Models by Average Score
![image](https://github.com/user-attachments/assets/e169f466-254a-4074-bbe5-b34689d8ed85)

Buy/Sell Signal Overlay on Price Chart
![image](https://github.com/user-attachments/assets/5648e09e-8ee9-4b37-8cd0-26ca99173ec1)

MSFT Price with Autoencoder Anomalies
![43fa4d32-ecb2-43aa-8998-dc06314b201e](https://github.com/user-attachments/assets/1dd9060c-aa5b-4c28-ad4d-d3bdba7e0a26)

AAPL Price with Autoencoder Anomalies
![da11a34a-f58f-4e4b-a955-d90bb0d6d5f5](https://github.com/user-attachments/assets/589c6422-8fac-4798-ae16-48202c232d29)

TSLA Price with Autoencoder Anomalies
![64b4dec4-f24c-4868-8e57-e91c55c195e9](https://github.com/user-attachments/assets/8463ecbc-f12d-4306-ac0f-604caefb616f)


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

**Backtest Summary by Model**
## Completed Model Backtests

### ✅ LightGBM Walkforward Results

| Ticker | Final_Portfolio | Return_% | Sharpe | Accuracy | F1_Score | Drawdown |
|--------|----------------:|---------:|--------:|----------:|----------:|----------:|
| MSFT   | 113902.79       | 13.90    | 0.2136 | 0.489     | 0.2919    | 20108.83  |
| AAPL   | 106302.43       | 6.30     | 0.1187 | 0.4907    | 0.2495    | 20349.34  |
| NVDA   | 104822.77       | 4.82     | 0.1132 | 0.4813    | 0.2514    | 21230.21  |
| GOOGL  | 103981.14       | 3.98     | 0.1068 | 0.4721    | 0.2432    | 22594.12  |
| AMZN   | 102764.55       | 2.76     | 0.0983 | 0.4682    | 0.2378    | 23985.00  |


**LightGBM Plots**
![ABT_portfolio_plot](https://github.com/user-attachments/assets/0a7e9468-c300-4a49-ba32-efb46652a066)
![BRK-B_portfolio_plot](https://github.com/user-attachments/assets/6e104d70-8306-4841-8ffa-48ad4d2b6db3)
![MDT_portfolio_plot](https://github.com/user-attachments/assets/cf8e4c2d-416c-4179-b6dd-6bffc46394a6)
![PG_portfolio_plot](https://github.com/user-attachments/assets/c9052786-2e3f-4ba5-9908-eeadbd9a110c)
![PM_portfolio_plot](https://github.com/user-attachments/assets/8e52acd7-500a-474f-a0f4-31b534faffb3)

**LightGBM Summary**
[Uploading Top_5_PerfTicker,Final_Portfolio,Return_%,Sharpe,Accuracy,Precision,Recall,F1_Score,Drawdown
ABT,127777.29,27.78,0.635,0.497,0.526,0.1841,0.2728,8908.7
MDT,118362.98,18.36,0.6314,0.4967,0.5361,0.1115,0.1847,3772.33
PM,123683.9,23.68,0.6286,0.4917,0.5465,0.1502,0.2356,8841.6
PG,131011.66,31.01,0.567,0.5043,0.5841,0.1282,0.2103,6786.5
BRK-B,158738.56,58.74,0.5529,0.491,0.5351,0.2508,0.3415,15481.98orming_LightGBM_Stocks.csv…]()


**Stored files:**
- `lightgbm/models/`
- `lightgbm/scalers/`
- `lightgbm/features/`
- `lightgbm/metrics/lightgbm_walkforward_summary.csv`
- `lightgbm/plots/`

**Backtest Method:**  
Walkforward on 720-day hourly data using QuantConnect-compatible LightGBM with `MinMaxScaler`, technical features, and binary targets.


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
