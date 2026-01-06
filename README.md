## Momentum in Tennis — Baseline Flow Model

This repository provides a baseline point-level model that captures the flow of play during a tennis match using only pre-point “context” (server, score state, set/game counts, break-point flags, etc.). It estimates the probability that Player 1 will win the next point, then measures performance relative to that expectation and visualizes the evolving flow (“momentum”) within a match.

### Data
- Input CSV (already included): `data/2024_Wimbledon_featured_matches.csv`
- Data dictionary: `data/2024_data_dictionary.csv`

### Dependencies
- Python 3.10+
- Packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `scipy`

Install (example):
```bash
pip install pandas numpy scikit-learn xgboost matplotlib scipy
```

### How it works
1. Baseline model (XGBoost) predicts `P(Player1 wins current point)` using pre-point context:
   - Server
   - Score state (`p1_score`, `p2_score`), set/game counts
   - Given flags: `p1_break_pt`, `p2_break_pt`
   - Derived flags: game point, set point, match point (non-tiebreak approximation)
   - Tiebreak indicator
2. Performance signal: residual = `(actual outcome ∈ {0,1}) − (predicted probability)`.
3. Flow curve: exponentially weighted moving average (EWMA) of residuals.
   - Positive flow ⇒ Player 1 outperforming baseline expectation
   - Negative flow ⇒ Player 2 outperforming baseline expectation

### Running
```bash
python baseline_model.py
```
- The script will train a baseline model with a match-level split (no leakage across the same match).
- It will then visualize the flow for the 2023 Wimbledon Gentlemen’s Final (`match_id = 2023-wimbledon-1701`) if present; otherwise, it picks the first match in the file.

To force a different match, edit the preferred match id in `baseline_model.py` function `_pick_match_id`.

### Interpreting the plots
- Flow plot: area above zero (blue) means Player 1 is performing better than expected given pre-point context; below zero (red) favors Player 2. Vertical thin lines mark game boundaries, darker lines mark set boundaries.
- Baseline probability plot: the model’s pre-point expectation for `P(Player 1 wins)`, showing serve advantage and leverage dynamics across the match.

### Notes and assumptions
- Only pre-point context is used to avoid leakage from point outcomes (fair baseline for “momentum”).
- Tiebreak set-point detection is approximated (since point counters for tiebreaks are not encoded in `p1_score/p2_score`).
- The approach directly accounts for server advantage and critical points via both given and derived flags.
