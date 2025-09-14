# PPO Walk-Forward (External Signals) — Summary

This repo hosts a walk-forward PPO pipeline that produces external trading signals consumed by a QuantConnect (LEAN) client.  
The model runs per-ticker walk-forward training/evaluation, saves artifacts per fold, and exports a clean signal feed for execution.

## What’s new / improvements

1) **Execution & Slippage Simulation**
   - Constant and impact-style slippage profiles; fee model hooks.
   - QC integration validated (IB Margin + ConstantSlippageModel).

2) **Noise Filtering**
   - **Wavelet denoising** on price/feature series to reduce microstructure noise.

3) **Market Regime Detection**
   - Volatility & trend state machine; regime used for feature gating and reward shaping.

4) **Latency & Broker Integration**
   - JSON signal emitter (RAW Gist alias) with `valid_until_utc` freshness guard.
   - Confidence-weighted targets (mapped to 10–50% caps) with cash buffer normalization.

5) **Risk Management Module**
   - Per-name caps, portfolio gross cap (~95% exposure), optional trade-cooldowns.

6) **Feature Additions**
   - **FinBERT** sentiment scores (headline/text ingestion optional).
   - **Mock option Greeks** (synthetic IV surface heuristics) for skew/convexity proxies.
   - Technical suite + wavelet-filtered variants.

7) **PPO Walk-Forward Enhancements**
   - Confidence-based rewards, **whipsaw penalty**, regime filtering.
   - Expanding/rolling windows; per-fold early-stopping and checkpointing.

8) **Evaluation & Logging**
   - Extended metrics (return, Sharpe, Sortino, PSR, drawdown, turnover, hit-rate).
   - QC end-of-run `RUN_SUMMARY` log (orders, fills, closed trades, win-rate).

---

## Minimal pipeline overview

- **Data → Features**: raw OHLCV (+ sentiment, Greeks proxies) → wavelet denoise → scale/clip.
- **Split**: walk-forward folds (expanding or rolling).
- **Train**: PPO per fold with reward shaping (confidence & whipsaw).
- **Validate**: per fold; keep best checkpoint.
- **Emit**: merge fold predictions → `live_signals.json` (per symbol).
- **Execute**: QC consumer polls alias URL; maps confidence → holdings; enforces risk caps.

---

## Key metrics we track

- Net Return, CAGR, Max Drawdown, **Sharpe**, **PSR (Sharpe>0)**, Sortino, Information Ratio, Turnover, Trades/Day.
- Per-trade and daily PnL distributions; regime-aware attribution (optional).

---

## Saved artifacts (consistent naming)

All artifacts are grouped by **ticker** and **fold** under `artifacts/`.  
Use these patterns so everything lines up with the report and orders CSV:

---

## Installation & Usage

### Prerequisites
- Python 3.9+
- pip or conda
- Git

### Setup
```bash
git clone https://github.com/racoope70/quant-trading-model-zoo.git
cd quant-trading-model-zoo/ppo-model-2024-09
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
