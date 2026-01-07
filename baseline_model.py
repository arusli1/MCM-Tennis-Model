import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import OneHotEncoder

from scipy.sparse import csr_matrix, hstack
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# -----------------------------
# Paths
# -----------------------------
def _default_data_path() -> str:
    # Prefer project-local data file
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "data", "2024_Wimbledon_featured_matches.csv")
    return candidate


# -----------------------------
# 1) Load + basic cleaning
# -----------------------------
def load_data(path: str | None = None) -> pd.DataFrame:
    data_path = path or _default_data_path()
    df = pd.read_csv(data_path)

    # Ensure chronological order within each match
    df = df.sort_values(["match_id", "set_no", "game_no", "point_no"]).reset_index(drop=True)

    # Pre-point progress proxy (optional; not used in the baseline features by default)
    df["point_index_in_match"] = df.groupby("match_id").cumcount()

    # Tiebreak indicator (in this dataset, game_no reaches 13 only for tiebreak)
    df["is_tiebreak"] = (df["game_no"] == 13).astype(int)

    return df


# -----------------------------
# 2) Derived pre-point flags (game/set/match point)
# -----------------------------
def _is_game_point_columns(p1_score: pd.Series, p2_score: pd.Series) -> tuple[pd.Series, pd.Series]:
    s1 = p1_score.astype(str).str.upper()
    s2 = p2_score.astype(str).str.upper()
    # Normal-game "game point" conditions (tiebreak handled separately)
    p1_gp = (s1.eq("AD")) | (s1.eq("40") & s2.isin(["0", "15", "30"]))
    p2_gp = (s2.eq("AD")) | (s2.eq("40") & s1.isin(["0", "15", "30"]))
    return p1_gp.astype(int), p2_gp.astype(int)


def derive_point_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Game points only make sense in non-tiebreak games
    p1_gp, p2_gp = _is_game_point_columns(df["p1_score"], df["p2_score"])
    p1_gp = (p1_gp & (df["is_tiebreak"] == 0)).astype(int)
    p2_gp = (p2_gp & (df["is_tiebreak"] == 0)).astype(int)

    out["is_game_point_p1"] = p1_gp
    out["is_game_point_p2"] = p2_gp

    # Set point (non-tiebreak approximation): if winning this game would close the set
    p1_would_win_set = ((df["p1_games"] + 1 >= 6) & ((df["p1_games"] + 1) - df["p2_games"] >= 2)).astype(int)
    p2_would_win_set = ((df["p2_games"] + 1 >= 6) & ((df["p2_games"] + 1) - df["p1_games"] >= 2)).astype(int)
    out["is_set_point_p1"] = (p1_gp & (df["is_tiebreak"] == 0) & p1_would_win_set).astype(int)
    out["is_set_point_p2"] = (p2_gp & (df["is_tiebreak"] == 0) & p2_would_win_set).astype(int)

    # Match point (best-of-5): a set point while already at 2 sets
    out["is_match_point_p1"] = (out["is_set_point_p1"] & (df["p1_sets"] == 2)).astype(int)
    out["is_match_point_p2"] = (out["is_set_point_p2"] & (df["p2_sets"] == 2)).astype(int)

    return out


# -----------------------------
# 3) Build baseline features (PRE-POINT only)
# -----------------------------
def build_baseline_Xy(df: pd.DataFrame):
    """
    Target y: Player1 wins point? (point_victor == 1)
    
    Baseline features (pre-point context only):
    - Server: dominant factor in tennis (serving player has ~60-70% win rate)
    - Point score: captures leverage within game (40-0 vs 0-40 very different)
    - Set score: captures match-level pressure (late sets more critical)
    
    Rationale:
    - We exclude games score (p1_games, p2_games) because:
      * Highly correlated with point score (if ahead in games, likely ahead in points)
      * Would over-explain match state, leaving less room for "momentum" to show
      * Point score already captures in-game leverage
    - This minimal baseline captures expected performance from rules/context,
      leaving residuals to capture actual performance above/below expectation.
    """
    y = (df["point_victor"] == 1).astype(int).to_numpy()

    # Pre-point context features (minimal but sufficient)
    X = pd.DataFrame({
        # Serve context (dominant pre-point factor)
        "is_p1_serving": (df["server"] == 1).astype(int),

        # Set score (current match) - captures match-level pressure
        "p1_sets": df["p1_sets"].astype(int),
        "p2_sets": df["p2_sets"].astype(int),

        # Point score state (categorical) - captures in-game leverage
        "p1_score": df["p1_score"].astype(str),
        "p2_score": df["p2_score"].astype(str),
    })

    groups = df["match_id"].to_numpy()
    return X, y, groups


# -----------------------------
# 4) One-hot encode + train XGBoost baseline
# -----------------------------
def train_xgb_baseline(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, exclude_match_id: str | None = None):
    """
    Trains XGBoost on match-level split (no leakage across points of same match).
    If exclude_match_id is provided, that match is excluded from training entirely.
    Returns trained booster + encoder so you can transform new points the same way.
    """
    # Exclude target match from training if specified
    if exclude_match_id is not None:
        train_mask = groups != exclude_match_id
        X = X.loc[train_mask].copy()
        y = y[train_mask]
        groups = groups[train_mask]
        print(f"Excluded match '{exclude_match_id}' from training. Training on {len(X)} points from {len(np.unique(groups))} matches.")
    
    # Minimal baseline uses numeric-only features, but keep this generic:
    possible_cat_cols = ["p1_score", "p2_score"]
    cat_cols = [c for c in possible_cat_cols if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(splitter.split(X, y, groups=groups))

    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_te, y_te = X.iloc[te_idx], y[te_idx]

    Xtr_num = csr_matrix(X_tr[num_cols].to_numpy())
    Xte_num = csr_matrix(X_te[num_cols].to_numpy())

    enc = None
    if len(cat_cols) > 0:
        # One-hot for categorical score columns (support older/newer sklearn)
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=True)
        Xtr_cat = enc.fit_transform(X_tr[cat_cols])
        Xte_cat = enc.transform(X_te[cat_cols])
        Xtr = hstack([Xtr_num, Xtr_cat], format="csr")
        Xte = hstack([Xte_num, Xte_cat], format="csr")
    else:
        Xtr = Xtr_num
        Xte = Xte_num

    dtr = xgb.DMatrix(Xtr, label=y_tr)
    dte = xgb.DMatrix(Xte, label=y_te)

    # Hyperparameter grid for tuning
    param_grid = {
        "max_depth": [3, 4, 5],
        "eta": [0.03, 0.05, 0.07],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [3, 5, 7],
        "lambda": [0.5, 1.0, 1.5],
    }
    
    # Simple grid search: try combinations and pick best validation logloss
    best_params = None
    best_logloss = float('inf')
    best_n_rounds = 0
    
    print("Tuning hyperparameters...")
    # Sample key combinations (full grid too expensive)
    param_combinations = [
        {"max_depth": 4, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "lambda": 1.0},
        {"max_depth": 4, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "lambda": 1.0},
        {"max_depth": 5, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "lambda": 1.0},
        {"max_depth": 4, "eta": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "lambda": 1.0},
        {"max_depth": 4, "eta": 0.07, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "lambda": 1.0},
        {"max_depth": 4, "eta": 0.05, "subsample": 0.7, "colsample_bytree": 0.8, "min_child_weight": 5, "lambda": 1.0},
        {"max_depth": 4, "eta": 0.05, "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 5, "lambda": 1.0},
    ]
    
    for params in param_combinations:
        params_full = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
            **params
        }
        
        booster_candidate = xgb.train(
            params=params_full,
            dtrain=dtr,
            num_boost_round=2000,
            evals=[(dtr, "train"), (dte, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        
        p_te_candidate = booster_candidate.predict(dte)
        logloss = -np.mean(y_te * np.log(p_te_candidate + 1e-12) + (1 - y_te) * np.log(1 - p_te_candidate + 1e-12))
        
        if logloss < best_logloss:
            best_logloss = logloss
            best_params = params_full
            best_n_rounds = booster_candidate.best_iteration + 1
    
    print(f"Best validation logloss: {best_logloss:.4f}")
    print(f"Best params: max_depth={best_params['max_depth']}, eta={best_params['eta']}, subsample={best_params['subsample']}, colsample={best_params['colsample_bytree']}, min_child={best_params['min_child_weight']}, lambda={best_params['lambda']}")
    print(f"Best n_rounds: {best_n_rounds}")
    
    # Train final model with best params
    booster = xgb.train(
        params=best_params,
        dtrain=dtr,
        num_boost_round=best_n_rounds,
        evals=[(dtr, "train"), (dte, "valid")],
        verbose_eval=100,
    )

    # Holdout logloss (sanity check)
    p_te = booster.predict(dte)
    eps = 1e-12
    logloss = -np.mean(y_te * np.log(p_te + eps) + (1 - y_te) * np.log(1 - p_te + eps))
    print(f"Holdout logloss: {logloss:.4f}")

    return booster, enc, num_cols, cat_cols


def predict_proba(booster, enc, num_cols, cat_cols, X: pd.DataFrame) -> np.ndarray:
    X_num = csr_matrix(X[num_cols].to_numpy())
    if enc is not None and len(cat_cols) > 0:
        X_cat = enc.transform(X[cat_cols])
        Xm = hstack([X_num, X_cat], format="csr")
    else:
        Xm = X_num
    dm = xgb.DMatrix(Xm)
    return booster.predict(dm)


# -----------------------------
# 5) Residuals + EWMA flow curve
# -----------------------------
def ewma(residuals: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Exponential weighted moving average of residuals."""
    F = np.zeros_like(residuals, dtype=float)
    for i, e in enumerate(residuals):
        F[i] = (1 - alpha) * (F[i - 1] if i > 0 else 0.0) + alpha * e
    return F


def ewma(residuals: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """
    Exponential Weighted Moving Average of residuals (no reset).
    Lower alpha = smoother (less noisy), higher alpha = more responsive.
    """
    F = np.zeros_like(residuals, dtype=float)
    for i, e in enumerate(residuals):
        if i == 0:
            F[i] = alpha * e
        else:
            F[i] = (1 - alpha) * F[i - 1] + alpha * e
    return F


def rolling_mean(residuals: np.ndarray, window: int = 15) -> np.ndarray:
    """
    Simple moving average of residuals over fixed window.
    All points in window weighted equally.
    """
    # Ensure window is valid (positive and not larger than data)
    window = max(1, min(int(window), len(residuals)))
    return pd.Series(residuals).rolling(window=window, min_periods=1).mean().to_numpy()


def cumulative_by_set(residuals: np.ndarray, set_no: np.ndarray) -> np.ndarray:
    """
    Cumulative sum of residuals within each set (resets at set boundaries).
    Clear indicator: positive = P1 outperformed in this set, negative = P2.
    """
    cumsum = np.zeros_like(residuals, dtype=float)
    current_set = None
    running_sum = 0.0
    for i, (r, s) in enumerate(zip(residuals, set_no)):
        if current_set is None or s != current_set:
            # New set: reset
            current_set = s
            running_sum = r
        else:
            # Same set: accumulate
            running_sum += r
        cumsum[i] = running_sum
    return cumsum


def compute_point_weights(df_match: pd.DataFrame) -> np.ndarray:
    """
    Compute weights for each point based on point "importance", "pressure", and "quality".
    Higher weight = point matters more for performance evaluation.
    
    Design philosophy:
    - Critical points (break/match points) should dominate performance evaluation
    - Pressure situations (close scores) test mental strength more
    - Skill contests (long rallies) better reflect true performance
    - Unearned points (aces, errors) less informative about momentum
    
    Returns array of weights (normalized to have mean=1.0).
    """
    weights = np.ones(len(df_match), dtype=float)
    
    # 1. LEVERAGE: Critical points matter most for performance
    # Match points (most critical - tests nerves under ultimate pressure)
    if "is_match_point_p1" in df_match.columns and "is_match_point_p2" in df_match.columns:
        is_match_pt = (df_match["is_match_point_p1"] == 1) | (df_match["is_match_point_p2"] == 1)
        weights[is_match_pt] *= 3.5  # 3.5x - highest weight
    
    # Set points (winning/losing a set is huge momentum swing)
    if "is_set_point_p1" in df_match.columns and "is_set_point_p2" in df_match.columns:
        is_set_pt = (df_match["is_set_point_p1"] == 1) | (df_match["is_set_point_p2"] == 1)
        weights[is_set_pt] *= 2.8  # 2.8x - very important
    
    # Break points (game-changing opportunities)
    if "p1_break_pt" in df_match.columns and "p2_break_pt" in df_match.columns:
        is_break_pt = (df_match["p1_break_pt"] == 1) | (df_match["p2_break_pt"] == 1)
        weights[is_break_pt] *= 2.2  # 2.2x - high leverage
    
    # Game points (standard leverage)
    if "is_game_point_p1" in df_match.columns and "is_game_point_p2" in df_match.columns:
        is_game_pt = (df_match["is_game_point_p1"] == 1) | (df_match["is_game_point_p2"] == 1)
        weights[is_game_pt] *= 1.4  # 1.4x - moderate importance
    
    # 2. PRESSURE: Close scores test mental strength and focus
    # Deuce/advantage situations are more pressure than 40-0
    if "p1_score" in df_match.columns and "p2_score" in df_match.columns:
        s1 = df_match["p1_score"].astype(str).str.upper()
        s2 = df_match["p2_score"].astype(str).str.upper()
        
        # Deuce (40-40): highest pressure
        is_deuce = (s1 == "40") & (s2 == "40")
        weights[is_deuce] *= 1.3
        
        # Advantage situations: high pressure
        is_advantage = (s1 == "AD") | (s2 == "AD")
        weights[is_advantage] *= 1.25
        
        # Close scores (30-30, 30-40, 40-30): moderate pressure
        is_close = ((s1 == "30") & (s2 == "30")) | ((s1 == "30") & (s2 == "40")) | ((s1 == "40") & (s2 == "30"))
        weights[is_close] *= 1.1
    
    # 3. QUALITY: Longer rallies and more effort = better test of skill
    # Rally length: longer rallies = more skill contest = higher weight
    # Only weight points where rally_count data is actually available
    if "rally_count" in df_match.columns:
        rally = df_match["rally_count"].astype(float)
        has_rally_data = ~rally.isna()
        rally_valid = rally[has_rally_data].clip(lower=1.0)  # Ensure >= 1
        # Log scale: rally of 1→1.0, 5→1.25, 10→1.4, 20→1.6 weight
        rally_adjustment = np.maximum(rally_valid - 1, 0)
        rally_weights = 1.0 + 0.25 * np.log1p(rally_adjustment)
        # Only apply to points with valid data
        weights[has_rally_data] *= rally_weights.values
    
    # Distance run: more running = more effort = higher weight
    # Only weight points where distance data is available for both players
    if "p1_distance_run" in df_match.columns and "p2_distance_run" in df_match.columns:
        dist1 = df_match["p1_distance_run"].astype(float)
        dist2 = df_match["p2_distance_run"].astype(float)
        has_dist_data = ~(dist1.isna() | dist2.isna())
        total_dist = (dist1 + dist2)[has_dist_data]
        total_dist = total_dist.clip(lower=0.0, upper=200.0)  # Clip to reasonable range
        # Scale: 0m→1.0, 50m→1.15, 100m→1.3 weight
        dist_weights = 1.0 + 0.3 * (total_dist / 100.0).clip(0, 1.5)
        # Only apply to points with valid data
        weights[has_dist_data] *= dist_weights.values
    
    # 4. OUTCOME QUALITY: Reduce weight for unearned points
    # These tell us less about true performance/momentum
    if "p1_ace" in df_match.columns and "p2_ace" in df_match.columns:
        is_ace = (df_match["p1_ace"] == 1) | (df_match["p2_ace"] == 1)
        weights[is_ace] *= 0.45  # Ace = minimal skill contest, mostly server skill
    
    if "p1_double_fault" in df_match.columns and "p2_double_fault" in df_match.columns:
        is_df = (df_match["p1_double_fault"] == 1) | (df_match["p2_double_fault"] == 1)
        weights[is_df] *= 0.55  # Double fault = unearned point, mostly unforced error
    
    if "p1_unf_err" in df_match.columns and "p2_unf_err" in df_match.columns:
        is_ue = (df_match["p1_unf_err"] == 1) | (df_match["p2_unf_err"] == 1)
        weights[is_ue] *= 0.65  # Unforced error = point given away, less informative
    
    # Handle any NaN or Inf values before normalization
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Normalize so mean weight = 1.0 (keeps residual scale similar)
    mean_weight = weights.mean()
    if mean_weight > 0 and np.isfinite(mean_weight):
        weights = weights / mean_weight
    else:
        # Fallback: all weights = 1.0 if normalization fails
        weights = np.ones_like(weights)
    
    return weights


def weighted_residual(residuals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply point-quality weights to residuals."""
    return residuals * weights




# -----------------------------
# 6) Visualization for one match
# -----------------------------
def plot_match_flow(df_match: pd.DataFrame, p: np.ndarray, flow: np.ndarray, title: str):
    # Mark game boundaries for vertical lines
    game_change = df_match["game_no"].diff().fillna(0).ne(0).to_numpy()
    set_change = df_match["set_no"].diff().fillna(0).ne(0).to_numpy()

    x = np.arange(len(df_match))

    plt.figure(figsize=(12, 4))
    plt.plot(x, flow, color="#1f77b4", label="Flow (EWMA residual)")
    plt.axhline(0, color="gray", lw=1, alpha=0.7)
    plt.fill_between(x, 0, flow, where=(flow >= 0), color="#1f77b4", alpha=0.15)
    plt.fill_between(x, 0, flow, where=(flow < 0), color="#d62728", alpha=0.15)
    plt.title(title)
    plt.xlabel("Point index (within match)")
    plt.ylabel("Flow (y − p), EWMA")

    # Light structure markers
    for i in np.where(game_change)[0]:
        plt.axvline(i, color="gray", lw=0.5, alpha=0.25)
    for i in np.where(set_change)[0]:
        plt.axvline(i, color="black", lw=1.0, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_match_performance(df_match: pd.DataFrame, flow: np.ndarray, title: str):
    """
    Performance metric: EWMA of residuals (y - p).
      > 0 means P1 is outperforming baseline expectation
      < 0 means P2 is outperforming baseline expectation
    """
    game_change = df_match["game_no"].diff().fillna(0).ne(0).to_numpy()
    set_change = df_match["set_no"].diff().fillna(0).ne(0).to_numpy()

    x = np.arange(len(df_match))
    plt.figure(figsize=(12, 3.8))
    plt.plot(x, flow, color="#2ca02c", label="Performance (EWMA of y − p)")
    plt.axhline(0, color="gray", lw=1, alpha=0.7)
    plt.fill_between(x, 0, flow, where=(flow >= 0), color="#2ca02c", alpha=0.12)
    plt.fill_between(x, 0, flow, where=(flow < 0), color="#d62728", alpha=0.12)

    # Game / set boundaries
    for i in np.where(game_change)[0]:
        plt.axvline(i, color="gray", lw=0.5, alpha=0.25)
    for i in np.where(set_change)[0]:
        plt.axvline(i, color="black", lw=1.0, alpha=0.35)

    plt.title(title + " — Performance (EWMA residual)")
    plt.xlabel("Point index (within match)")
    plt.ylabel("EWMA(y − p)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_match_summary(df_match: pd.DataFrame, p: np.ndarray, flow: np.ndarray | None, residual: np.ndarray, title: str):
    """
    Clean visualization with baseline + 3 best performance metrics.
    
    Performance metric selection rationale:
    1. Weighted EWMA (reset per set, α=0.20): 
       - Quality-adjusted (longer rallies, break points weighted higher)
       - Responsive (α=0.20 gives ~3.5 point half-life, captures short-term momentum)
       - Resets per set (each set independent, aligns with tennis structure)
       - Best for: identifying who is playing better RIGHT NOW
    
    2. Cumulative Residual (reset per set):
       - Clear set-level performance: positive = P1 outperformed, negative = P2
       - No smoothing bias: directly sums actual vs expected
       - Resets per set: shows who won each set "on merit" vs baseline
       - Best for: answering "who performed better in set X?"
    
    3. Rolling Mean (15 points):
       - Smooth, game-level view (~1 game window)
       - Less noisy than raw residuals, less laggy than long windows
       - No reset: shows sustained trends across sets
       - Best for: identifying medium-term momentum shifts
    """
    game_change = df_match["game_no"].diff().fillna(0).ne(0).to_numpy()
    set_change = df_match["set_no"].diff().fillna(0).ne(0).to_numpy()

    # Server switches (directional)
    server = df_match["server"].to_numpy()
    server_switch_idx = np.where(server[1:] != server[:-1])[0] + 1

    x = np.arange(len(df_match))
    set_no = df_match["set_no"].to_numpy()

    # Performance metric: Compare EWMA vs Rolling Mean, find best alpha
    print("\nTesting different smoothing approaches...")
    
    # Test different alphas for EWMA
    alphas = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    ewma_results = {}
    for alpha in alphas:
        ewma_vals = ewma(residual, alpha=alpha)
        # Use variance as measure of smoothness (lower = smoother)
        # Also consider responsiveness (how quickly it responds to changes)
        variance = np.var(ewma_vals)
        # Measure lag: correlation with shifted residuals (higher = less lag)
        if len(residual) > 1:
            lag_measure = np.corrcoef(ewma_vals[1:], residual[:-1])[0,1] if len(ewma_vals) > 1 else 0
        else:
            lag_measure = 0
        ewma_results[alpha] = {"variance": variance, "lag": lag_measure}
        print(f"  EWMA α={alpha:.2f}: variance={variance:.4f}, lag_corr={lag_measure:.4f}")
    
    # Test rolling mean with different windows
    windows = [10, 15, 20, 25]
    rolling_results = {}
    for window in windows:
        rolling_vals = rolling_mean(residual, window=window)
        variance = np.var(rolling_vals)
        if len(residual) > window:
            lag_measure = np.corrcoef(rolling_vals[window:], residual[:-window])[0,1] if len(rolling_vals) > window else 0
        else:
            lag_measure = 0
        rolling_results[window] = {"variance": variance, "lag": lag_measure}
        print(f"  Rolling mean window={window}: variance={variance:.4f}, lag_corr={lag_measure:.4f}")
    
    # Choose best: balance smoothness (low variance) and responsiveness (low lag)
    # Best alpha: minimizes variance while maintaining reasonable responsiveness
    best_alpha = min(alphas, key=lambda a: ewma_results[a]["variance"] + 0.3 * (1 - ewma_results[a]["lag"]))
    print(f"\nSelected EWMA α={best_alpha:.2f} (best balance of smoothness and responsiveness)")
    print(f"  Rationale: α={best_alpha:.2f} provides half-life ~{-np.log(0.5)/np.log(1-best_alpha):.1f} points")
    print(f"  This captures short-term momentum (2-3 games) without excessive noise")
    
    flow_ewma = ewma(residual, alpha=best_alpha)
    
    # Clean layout: 2 plots stacked vertically - baseline + EWMA
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 1, hspace=0.25, left=0.06, right=0.97, top=0.94, bottom=0.08,
                         height_ratios=[1, 1])  # Equal height

    # Plot 1: Baseline probability (XGBoost predictions)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(x, p, color="#9467bd", label="Baseline P(P1 wins point)", lw=2.0)
    ax0.axhline(0.5, color="gray", lw=1.5, alpha=0.6, linestyle="--", label="Equal chance")
    ax0.set_ylabel("Probability", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Point Index", fontsize=11)
    ax0.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax0.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax0.grid(True, alpha=0.2, linestyle=":")
    ax0.set_ylim([0, 1])
    for i in np.where(game_change)[0]:
        ax0.axvline(i, color="lightgray", lw=0.5, alpha=0.3)
    for i in np.where(set_change)[0]:
        ax0.axvline(i, color="black", lw=1.5, alpha=0.5, linestyle="-")
    for i in server_switch_idx:
        prev_srv = server[i - 1]
        curr_srv = server[i]
        if prev_srv == 1 and curr_srv == 2:
            ax0.axvline(i, color="#1f77b4", lw=1.2, alpha=0.6, linestyle="--")
        elif prev_srv == 2 and curr_srv == 1:
            ax0.axvline(i, color="#ff7f0e", lw=1.2, alpha=0.6, linestyle="--")

    # Plot 2: Performance Flow (EWMA of residuals)
    ax1 = fig.add_subplot(gs[1, 0])
    
    # Handle NaN/Inf values
    flow_ewma = np.nan_to_num(flow_ewma, nan=0.0, posinf=0.0, neginf=0.0)
    
    ax1.plot(x, flow_ewma, color="#2ca02c", label=f"Performance Flow (EWMA, α={best_alpha:.2f})", lw=2.0, alpha=0.9)
    ax1.axhline(0, color="gray", lw=1.5, alpha=0.6, linestyle="--")
    ax1.fill_between(x, 0, flow_ewma, where=(flow_ewma >= 0), color="#2ca02c", alpha=0.25)
    ax1.fill_between(x, 0, flow_ewma, where=(flow_ewma < 0), color="#d62728", alpha=0.25)
    ax1.set_ylabel("Performance Flow", fontsize=12, fontweight="bold")
    ax1.set_title("Performance Flow (EWMA of Residuals)", fontsize=14, fontweight="bold", pad=12)
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.2, linestyle=":")
    ax1.tick_params(labelsize=10)
    
    # Auto-scale y-axis
    y_min, y_max = float(flow_ewma.min()), float(flow_ewma.max())
    if not (np.isfinite(y_min) and np.isfinite(y_max)):
        y_min, y_max = -0.1, 0.1
    
    y_range = y_max - y_min
    if y_range < 0.01 or not np.isfinite(y_range):
        y_center = (y_max + y_min) / 2 if np.isfinite((y_max + y_min) / 2) else 0.0
        ax1.set_ylim(y_center - 0.1, y_center + 0.1)
    else:
        ax1.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # Set boundaries
    for i in np.where(game_change)[0]:
        ax1.axvline(i, color="lightgray", lw=0.5, alpha=0.3)
    for i in np.where(set_change)[0]:
        ax1.axvline(i, color="black", lw=1.5, alpha=0.5)
    
    ax1.set_xlabel("Point Index", fontsize=11)

    # Save figure
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    match_id = str(df_match["match_id"].iloc[0]) if "match_id" in df_match.columns else "match"
    out_path = os.path.join(results_dir, f"{match_id}_summary.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")

    plt.show()


def plot_match_probability(df_match: pd.DataFrame, p: np.ndarray, title: str):
    # Mark game and set boundaries
    game_change = df_match["game_no"].diff().fillna(0).ne(0).to_numpy()
    set_change = df_match["set_no"].diff().fillna(0).ne(0).to_numpy()
    # Server switches (directional)
    server = df_match["server"].to_numpy()
    server_switch_idx = np.where(server[1:] != server[:-1])[0] + 1  # indices where server changes at point i

    x = np.arange(len(df_match))

    plt.figure(figsize=(12, 3.5))
    plt.plot(x, p, color="#9467bd", label="Baseline P(P1 wins)")
    plt.axhline(0.5, color="gray", lw=1, alpha=0.7)
    for i in np.where(game_change)[0]:
        plt.axvline(i, color="gray", lw=0.5, alpha=0.25)
    for i in np.where(set_change)[0]:
        plt.axvline(i, color="black", lw=1.0, alpha=0.35)
    # Dotted lines for server switches with direction-specific colors
    added_legend_p1_to_p2 = False
    added_legend_p2_to_p1 = False
    for i in server_switch_idx:
        prev_srv = server[i - 1]
        curr_srv = server[i]
        if prev_srv == 1 and curr_srv == 2:
            plt.axvline(
                i, color="#1f77b4", lw=1.0, alpha=0.6, linestyle="--",
                label="Server switch P1→P2" if not added_legend_p1_to_p2 else None
            )
            added_legend_p1_to_p2 = True
        elif prev_srv == 2 and curr_srv == 1:
            plt.axvline(
                i, color="#ff7f0e", lw=1.0, alpha=0.6, linestyle="--",
                label="Server switch P2→P1" if not added_legend_p2_to_p1 else None
            )
            added_legend_p2_to_p1 = True
    plt.title(title + " — Baseline probability")
    plt.xlabel("Point index (within match)")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _pick_match_id(df: pd.DataFrame, preferred: str = "2023-wimbledon-1701") -> str:
    if preferred in set(df["match_id"].unique()):
        return preferred
    return df["match_id"].iloc[0]


# -----------------------------
# Run end-to-end
# -----------------------------
if __name__ == "__main__":
    df = load_data()
    
    # Choose target match FIRST (before building X,y)
    target_match_id = "2023-wimbledon-1701"  # Alcaraz vs Djokovic final
    if target_match_id not in set(df["match_id"].unique()):
        target_match_id = df["match_id"].iloc[0]
        print(f"Warning: Target match '2023-wimbledon-1701' not found. Using '{target_match_id}' instead.")
    
    # Build features for all data
    X, y, groups = build_baseline_Xy(df)

    # Train on ALL matches EXCEPT the target match
    print(f"\nTraining baseline model (excluding match '{target_match_id}')...")
    booster, enc, num_cols, cat_cols = train_xgb_baseline(X, y, groups, exclude_match_id=target_match_id)

    # Extract target match for visualization
    mask = (df["match_id"] == target_match_id)
    df_m = df.loc[mask].copy()
    X_m = X.loc[mask].copy()
    y_m = y[mask]

    print(f"\nEvaluating on match '{target_match_id}' ({len(df_m)} points)...")
    p_m = predict_proba(booster, enc, num_cols, cat_cols, X_m)
    residual_m = y_m - p_m
    
    # Match-level performance summary
    match_logloss = -np.mean(y_m * np.log(p_m + 1e-12) + (1 - y_m) * np.log(1 - p_m + 1e-12))
    print(f"\nMatch Evaluation:")
    print(f"  Match logloss: {match_logloss:.4f}")
    print(f"  Mean absolute residual: {np.abs(residual_m).mean():.4f}")
    
    # Summary statistics
    p1_advantage_points = np.sum(residual_m > 0)
    p2_advantage_points = np.sum(residual_m < 0)
    print(f"  Points where P1 outperformed: {p1_advantage_points} ({100*p1_advantage_points/len(residual_m):.1f}%)")
    print(f"  Points where P2 outperformed: {p2_advantage_points} ({100*p2_advantage_points/len(residual_m):.1f}%)")

    title = f"Match Flow — {df_m['player1'].iloc[0]} vs {df_m['player2'].iloc[0]}"
    plot_match_summary(df_m, p_m, None, residual_m, title=title)
