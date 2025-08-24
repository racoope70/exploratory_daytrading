# inference_bundle.py
# Contains the exact featurizer + env + loader needed to run predict() with the exported artifacts.

import os, json, numpy as np, pandas as pd


def compute_enhanced_features(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]

    # --- Technicals ---
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['STD_20'] = df['Close'].rolling(20).std()
    df['Upper_Band'] = df['SMA_20'] + 2 * df['STD_20']
    df['Lower_Band'] = df['SMA_20'] - 2 * df['STD_20']

    df['Lowest_Low']   = df['Low'].rolling(14).min()
    df['Highest_High'] = df['High'].rolling(14).max()
    denom = (df['Highest_High'] - df['Lowest_Low']).replace(0, np.nan)
    df['Stoch'] = ((df['Close'] - df['Lowest_Low']) / denom) * 100

    df['ROC'] = df['Close'].pct_change(10)
    df['OBV'] = (np.sign(df['Close'].diff()).fillna(0) * df['Volume'].fillna(0)).cumsum()

    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(20).mean()
    md = (tp - sma_tp).abs().rolling(20).mean()
    df['CCI'] = (tp - sma_tp) / (0.015 * md)

    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line']   = ema12 - ema26
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()

    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df['RSI'] = 100 - (100 / (1 + rs))

    tr = pd.concat([
        (df['High'] - df['Low']),
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    df['Volatility'] = df['Close'].pct_change().rolling(20).std()

    # --- Denoised Close (wavelet smooth) ---
    df['Denoised_Close'] = denoise_wavelet(df['Close'].ffill())

    # --- Market regime (optional) ---
    if USE_REGIME:
        df = add_regime(df)

    # --- Sentiment (optional, placeholder headline) ---
    if USE_SENTIMENT and len(df):
        headline = f"{df['Symbol'].iloc[0]} is expected to perform well in the market."
        try:
            score = score_sentiment([headline])[0]
        except Exception as e:
            logging.warning(f"Sentiment scoring failed for {df['Symbol'].iloc[0]}: {e}")
            score = 0.0
        df['SentimentScore'] = float(score)
    else:
        df['SentimentScore'] = 0.0

    # --- Greeks-ish (simple proxies) ---
    df['Delta'] = df['Close'].pct_change(1).fillna(0)
    df['Gamma'] = df['Delta'].diff().fillna(0)

    # Cleanup
    df = df.dropna().reset_index(drop=True)

    cols = [c for c in df.columns if c not in ['Symbol']] + ['Symbol']
    return df[cols]


# TODO: Paste your exact ContinuousPositionEnv used for training.
class ContinuousPositionEnv:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Paste ContinuousPositionEnv definition here.")


# Minimal loader wrapper (adjust to your env if needed)
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

FINAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "ppo_models_master")

def load_model_and_env(prefix):
    model_path = os.path.join(FINAL_MODEL_DIR, f"{prefix}_model.zip")
    vec_path   = os.path.join(FINAL_MODEL_DIR, f"{prefix}_vecnorm.pkl")
    model = PPO.load(model_path, device="cpu")

    def make_env(df_window):
        from gymnasium.spaces import Box as GBox  # if needed by your env
        frame_bound = (50, len(df_window) - 3)
        e = DummyVecEnv([lambda: ContinuousPositionEnv(
            df=df_window, frame_bound=frame_bound, window_size=10,
            cost_rate=0.0002, slip_rate=0.0003,
            k_alpha=0.20, k_mom=0.05, k_sent=0.0,
            mom_source="denoised", mom_lookback=20,
            min_trade_delta=0.01, cooldown=5, reward_clip=1.0
        )])
        if os.path.exists(vec_path):
            e = VecNormalize.load(vec_path, e)
        e.training = False
        e.norm_reward = False
        return e

    return model, make_env


# --- Simple prediction helper (uses the loader above) ---
def predict_from_features(df_window, prefix, deterministic=True):
    """
    df_window must already have columns exactly matching <prefix>_features.json.
    """
    model, make_env = load_model_and_env(prefix)
    env = make_env(df_window)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs
    action, _ = model.predict(obs, deterministic=deterministic)
    # Example mapping: continuous action -> side
    a = float(action.squeeze())
    signal = "BUY" if a > 0.1 else ("SELL" if a < -0.3 else "HOLD")
    return dict(action=a, signal=signal)
