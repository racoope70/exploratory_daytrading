HOW TO USE (QC/Alpaca)

1) Inputs / Features
   - Use compute_enhanced_features(...) from code/inference_bundle.py to recreate EXACT feature columns
     listed in each <prefix>_features.json (same order!) before calling predict.

2) Normalization
   - Wrap your env/obs with VecNormalize using <prefix>_vecnorm.pkl as shown in load_model_and_env().

3) Predict
   - Load PPO with <prefix>_model.zip and call model.predict(obs, deterministic=True).
   - The helper predict_from_features(df_window, prefix) shows a minimal example.

4) Threshold / Confidence (optional)
   - See <prefix>_probability_config.json for any gating configuration.

5) Dependencies
   - See requirements.txt for pinned versions that match training.

Included prefixes:
  ppo_GE_window1, ppo_UNH_window3, ppo_CVX_window1
