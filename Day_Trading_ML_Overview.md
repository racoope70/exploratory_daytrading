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

> **Work in Progress**: This repository is under active development and continues to be updated with additional models, cleaner integration, and improved signal logic.

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

**Training Accuracy (~85–89%)**  
Based on LightGBM in-sample performance across rolling windows.

**Test Accuracy (Walkforward ~47–50%)**  
Out-of-sample validation using true date-based walkforward.

**Sharpe Ratio (~0.55 – 0.64)**  
QuantConnect hourly tests show realistic, moderate returns.

**Max Drawdown (~3.7% – 15.5%)**  
Controlled with stop-loss, volatility filters, and position sizing.

**Final Portfolio Return (+20% to +58%)**  
Year-long backtests: BRK‑B (+58.7%), PG (+31%), ABT (+27.8%).

## Additional Models (Preliminary Results)

These models are currently being tested—results below are **early estimates**, pending full backtesting.

- **XGBoost**  
  - Training Accuracy: ~86–88%  
  - Preliminary Sharpe: ~0.50–0.60

- **Random Forest**  
  - Training Accuracy: ~80–85%  
  - Expected Sharpe: ~0.45–0.55

- **Reinforcement Learning (PPO, SAC, TD3, A2C, DQN, Deep SARSA)**  
  - Preliminary results pending; initial runs suggest **variable Sharpe (~0.3–0.6)** and high volatility.

- **K‑Means Clustering**  
  - Used for regime detection; early signal analysis suggests moderate predictive value, but full integration still underway.


**Backtest Summary by Model**
## Completed Model Backtests v1

### LightGBM Walkforward Results Metrics (Pre-Backtest, Google Colab)


| Ticker | Final_Portfolio | Return_% | Sharpe | Accuracy | F1_Score    | Drawdown  |
|--------|----------------:|---------:|--------:|----------:|----------:|----------:|
| ABt    | 127,777.29      | 27.78    | 0.6350  | 0.497     | 0.2728    |  8.91%    |
| BRK-B  | 158,738.56      | 58.74    | 0.5529  | 0.491     | 0.3415    | 15.48%    |
| PG     | 131,011.66      | 31.01    | 0.5670  | 0.5043    | 0.2103    |  6.79%    |


**LightGBM Plots**
![BRK-B_portfolio_plot](https://github.com/user-attachments/assets/6e104d70-8306-4841-8ffa-48ad4d2b6db3)
![PG_portfolio_plot](https://github.com/user-attachments/assets/c9052786-2e3f-4ba5-9908-eeadbd9a110c)
![ABT_portfolio_plot](https://github.com/user-attachments/assets/0a7e9468-c300-4a49-ba32-efb46652a066)

### ABT Walkforward Backtest (QuantConnect)

**Performance Summary:**

**BRK-B**
![image](https://github.com/user-attachments/assets/605a8e07-e5dd-433d-ae6e-97cdeedd8365)

**PG**
![image](https://github.com/user-attachments/assets/07b965a8-d42c-4c13-831c-0c5e894bb81a)

**ABT**
![image](https://github.com/user-attachments/assets/78efc22b-0beb-4981-826e-3ac8c3ae8515)

**Model Risk Disclaimer: Overfitting Flag Contextualized**
While QuantConnect flags ‘Likely Overfitting’ due to parameter count, this strategy is walkforward-validated, uses simple interpretable features, and applies volatility and probability filters. Real-world constraints (slippage, trade cooldowns) are simulated to reinforce generalization.

**Stored files:**
- `lightgbm/models/`
- `lightgbm/scalers/`
- `lightgbm/features/`
- `lightgbm/metrics/lightgbm_walkforward_summary.csv`
- `lightgbm/plots/`

**Backtest Method:**  
Walkforward on 720-day hourly data using QuantConnect-compatible LightGBM with `MinMaxScaler`, technical features, and binary targets.


### XGBoost Walkforward Results Metrics (Pre-Backtest, Google Colab)


| Ticker | Final_Portfolio | Return_% | Sharpe | Accuracy | F1_Score | Drawdown |
|--------|----------------:|---------:|--------:|----------:|----------:|----------:|
| AVGO   | 113,014.20      | 13.01    | 1.877   | .1607     | .1644     | 2.70%     |
| AMD    | 108,788.84      |  8.79    | 1.146   | .2821     | .2784     | 3.86%     |
| HD     | 1037,11.63      |  3.71    | 1.135   | .3555    | .2362     | 1.97%     |

### ABT Walkforward Backtest (QuantConnect)

**Performance Summary:**

**AVGO**
<img width="1746" height="1046" alt="image" src="https://github.com/user-attachments/assets/0f321ff7-d8bd-4964-ab00-5895cfa48cc9" />

<img width="1746" height="1046" alt="image" src="https://github.com/user-attachments/assets/16fc7772-29b5-4195-ac2e-4cc5262dccc3" />


**AMD**
<img width="1082" height="520" alt="Screenshot 2025-07-10 at 1 55 35 PM" src="https://github.com/user-attachments/assets/facb185c-8055-4c91-97c7-e19a975e6e99" />

<img width="873" height="334" alt="Screenshot 2025-07-10 at 2 02 13 PM" src="https://github.com/user-attachments/assets/5c20cb0e-06c5-46cb-9c72-a9d514ba70a4" />

**HD**
<img width="2186" height="1016" alt="image" src="https://github.com/user-attachments/assets/e250e903-b5c2-4fd9-a1da-1d46f5992fd1" />
<img width="2186" height="1016" alt="image" src="https://github.com/user-attachments/assets/4111baa5-6f10-4ef8-ba17-ff9b4a3d83d1" />

**Model Risk Disclaimer: Overfitting Flag Contextualized**
While QuantConnect flags ‘Likely Overfitting’ due to parameter count, this strategy is walkforward-validated, uses simple interpretable features, and applies volatility and probability filters. Real-world constraints (slippage, trade cooldowns) are simulated to reinforce generalization.

**Stored files:**
- `XGBoost/models/`
- `XGBoost/scalers/`
- `XGBoost/features/`
- `XGBoost/metrics/XGBoost_walkforward_summary.csv`

### 📦 Model Artifacts

Due to GitHub file size constraints, I only publishing model artifacts (trained XGBoost models, feature JSONs, and result CSVs) for the **top 3 performing stocks** in the walkforward evaluation.

Full results for all 53 tickers were computed and saved locally in Google Drive. These include:
- Walkforward metrics (Sharpe, Return, Drawdown, F1)
- SHAP plots
- Trade signals
- Final portfolios

If you would like access to the full dataset or models, please contact the author.

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
