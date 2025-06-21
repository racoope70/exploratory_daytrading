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
This repository contains an ongoing pipeline for **day trading with machine learning**, integrating supervised learning, deep reinforcement learning, and unsupervised anomaly detection models. Using hourly stock data, the system generates, tests, and optimizes algorithmic trading strategies through walkforward evaluation, signal labeling, model selection, and performance visualization.

>  **Work in Progress**: The pipeline is under active development and expanding to support live testing, deployment automation, and full white paper documentation.

---

## Features  

- **Data Handling**
  - Download and cache 720-day hourly stock data using `yfinance`.
- **Feature Engineering**
  - Apply technical indicators (RSI, MACD, OBV, Stochastic, Bollinger Bands, etc.).
- **Trade Signal Labeling**
  - Quantile-based classification (Buy/Hold/Sell) or regression with log-return targets.
- **Walkforward Backtesting**
  - Date-based rolling and expanding window evaluation for realistic strategy validation.
- **Model Performance Tracking**
  - Save metrics (Sharpe, Accuracy, Max Drawdown, Final Portfolio) and signal charts.

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

---

## Project Structure

```
daytrading-with-ml/
|
 data/                          
 download/                      # Hourly data download
    download_hourly_data.py
|
 features/                      # Feature generation scripts
    feature_engineering.py
    calculate_rsi.py
    generate_trade_labels.py
|
 models/                        
    xgboost_model.py
    lightgbm_model.py
    rl_envs/
|
 walkforward/                   # Main walkforward evaluation logic
    xgboost_walkforward.py
    lightgbm_walkforward.py
    random_forest_train.py
    deep_q_walkforward.py
    deep_sarsa_walkforward.py
    sac_walkforward.py
    ppo_walkforward.py
    td3_walkforward.py
|
 unsupervised/                  # Anomaly detection and clustering
    anomaly_pipeline.py
    ae_anomaly.py
    autoencoder_pipeline.py
    anomaly_viz_cluster.py
|
 notebooks/                     
 results/                       
 requirements.txt               
 LICENSE                        
 README.md                      
```

---

## Usage  

**Option 1: Notebooks**  
- Run interactive notebooks in `/notebooks` to explore specific models or strategies.

**Option 2: End-to-End Pipeline**
```bash
python download/download_hourly_data.py
python features/feature_engineering.py
python walkforward/xgboost_walkforward.py
```

---


---

## Visualizations & Results

### Portfolio Performance (ORCL Strategy)
![download (1)](https://github.com/user-attachments/assets/23af4f86-1b1e-49d9-91a8-fe7d0281b94c)

### Buy/Sell Signal Overlay
![download (2)](https://github.com/user-attachments/assets/81ca0cd4-ca46-4e72-99cb-1cd9ef43dc42)

### Top and Bottom models by Average Score
![output](https://github.com/user-attachments/assets/0b4b53db-5008-488d-a577-cfb2259d73b8)

### Random Forest Feature Importance (Top 15 Features)
![RANDOM_FOREST_FEATURE](https://github.com/user-attachments/assets/45fdeadd-7195-4c96-89b3-a26bef596ee2)

### XGBoost Feature Importance (Top 15 Features)
![XGBOOST_FEATURE](https://github.com/user-attachments/assets/0d60690c-cc39-457b-99fc-26635344b1cf)

### LightGBM Feature Importance (Top 15 Features)
![LightGBM_Feature](https://github.com/user-attachments/assets/94fab601-d853-4a6c-b649-866f8b5d4bb7)


## Model Performance  

** Supervised Learning**
-  Random Forest (`random_forest_train.py`)
-  XGBoost (`xgboost_walkforward.py`)
-  LightGBM (`lightgbm_walkforward.py`)
-  Deep Q-Learning (`deep_q_walkforward.py`)
-  Deep SARSA (`deep_sarsa_walkforward.py`)
-  LSTM / GRU (`lstm_trade_model.py`, `gru_train.py`)

** Reinforcement Learning**
-  SAC - Soft Actor-Critic (`sac_walkforward.py`)
-  PPO - Proximal Policy Optimization (`ppo_walkforward.py`, `ppo_multi.py`)
-  TD3 - Twin Delayed DDPG (`td3_walkforward.py`)

** Unsupervised Learning / Anomaly Detection**
-  Isolation Forest, Clustering (`anomaly_pipeline.py`)
-  Autoencoders (`ae_anomaly.py`, `autoencoder_pipeline.py`)
-  DBSCAN / UMAP / t-SNE (`anomaly_viz_cluster.py`)

---

## Real-World Potential & Tangible Benefits  

- Build live-ready models using historical signals and ML predictions  
- Combine technical analysis with deep learning to reduce noise  
- Use anomaly detection to isolate rare but profitable trading events  
- Reinforce human rules with AI probability filters to improve accuracy  
- Perform realistic walkforward evaluation for backtest confidence

---

## Further Innovations & Expansion Plans  

-  **Step 4: Live Testing on New Market Data**  
-  **Step 5: Deploy as a Live Trading System with Execution Automation**  
-  **Write and Publish a White Paper Documenting Strategy & Results**  
-  Integrate RealTest/QuantConnect for deployment validation  
-  Build GUI for real-time trade monitoring  
-  Optimize deep learning models for multi-stock multi-agent learning  
-  Add macroeconomic & sentiment data for multi-source signal stacking

---

## License  
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments  
- `yfinance`, `ta`, `pandas_ta`, `scikit-learn`, `xgboost`, `lightgbm`, `stable-baselines3`  
- **OpenAI Gym** for RL environment structure  
- **Google Colab** for enabling GPU-based model training and prototyping  
- Personal rule-based trading experience as the foundational logic  
