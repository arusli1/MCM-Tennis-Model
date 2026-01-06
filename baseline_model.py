import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder

from scipy.sparse import csr_matrix, hstack
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    Features: server, score state, leverage flags, derived flags, progress proxy
    """
    y = (df["point_victor"] == 1).astype(int).to_numpy()

    derived = derive_point_flags(df)

    # Pre-point context features (safe)
    X = pd.DataFrame({
        # Serve context
        "is_p1_serving": (df["server"] == 1).astype(int),

        # Match state
        "set_no": df["set_no"].astype(int),
        "p1_sets": df["p1_sets"].astype(int),
        "p2_sets": df["p2_sets"].astype(int),

        "game_no": df["game_no"].astype(int),
        "p1_games": df["p1_games"].astype(int),
        "p2_games": df["p2_games"].astype(int),

        # Point score state (categorical as strings)
        "p1_score": df["p1_score"].astype(str),
        "p2_score": df["p2_score"].astype(str),

        # Given leverage flags (pre-point)
        "p1_break_pt": df["p1_break_pt"].astype(int),
        "p2_break_pt": df["p2_break_pt"].astype(int),

        # Derived leverage flags
        "is_game_point_p1": derived["is_game_point_p1"].astype(int),
        "is_game_point_p2": derived["is_game_point_p2"].astype(int),
        "is_set_point_p1": derived["is_set_point_p1"].astype(int),
        "is_set_point_p2": derived["is_set_point_p2"].astype(int),
        "is_match_point_p1": derived["is_match_point_p1"].astype(int),
        "is_match_point_p2": derived["is_match_point_p2"].astype(int),

        # Tiebreak
        "is_tiebreak": df["is_tiebreak"].astype(int),
    })

    groups = df["match_id"].to_numpy()
    return X, y, groups


# -----------------------------
# 4) One-hot encode + train XGBoost baseline
# -----------------------------
def train_xgb_baseline(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
    """
    Trains XGBoost on match-level split (no leakage across points of same match).
    Returns trained booster + encoder so you can transform new points the same way.
    """
    cat_cols = ["p1_score", "p2_score"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(splitter.split(X, y, groups=groups))

    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_te, y_te = X.iloc[te_idx], y[te_idx]

    # One-hot for categorical score columns (support older/newer sklearn)
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=True)

    Xtr_cat = enc.fit_transform(X_tr[cat_cols])
    Xte_cat = enc.transform(X_te[cat_cols])

    Xtr_num = csr_matrix(X_tr[num_cols].to_numpy())
    Xte_num = csr_matrix(X_te[num_cols].to_numpy())

    Xtr = hstack([Xtr_num, Xtr_cat], format="csr")
    Xte = hstack([Xte_num, Xte_cat], format="csr")

    dtr = xgb.DMatrix(Xtr, label=y_tr)
    dte = xgb.DMatrix(Xte, label=y_te)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "lambda": 1.0,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=2000,
        evals=[(dtr, "train"), (dte, "valid")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    # Holdout logloss (sanity check)
    p_te = booster.predict(dte)
    eps = 1e-12
    logloss = -np.mean(y_te * np.log(p_te + eps) + (1 - y_te) * np.log(1 - p_te + eps))
    print(f"Holdout logloss: {logloss:.4f}")

    return booster, enc, num_cols, cat_cols


def predict_proba(booster, enc, num_cols, cat_cols, X: pd.DataFrame) -> np.ndarray:
    X_cat = enc.transform(X[cat_cols])
    X_num = csr_matrix(X[num_cols].to_numpy())
    Xm = hstack([X_num, X_cat], format="csr")
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


def rolling_mean(residuals: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling mean of residuals over last N points."""
    return pd.Series(residuals).rolling(window=window, min_periods=1).mean().to_numpy()


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


def plot_match_summary(df_match: pd.DataFrame, p: np.ndarray, flow: np.ndarray, residual: np.ndarray, title: str):
    """
    One output figure with two stacked subplots:
      1) Baseline P(P1 wins point) vs point index
      2) Performance = EWMA(y - p) vs point index
    """
    game_change = df_match["game_no"].diff().fillna(0).ne(0).to_numpy()
    set_change = df_match["set_no"].diff().fillna(0).ne(0).to_numpy()

    # Server switches (directional)
    server = df_match["server"].to_numpy()
    server_switch_idx = np.where(server[1:] != server[:-1])[0] + 1

    x = np.arange(len(df_match))

    fig, (ax0, ax1) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(12, 7),
        gridspec_kw={"height_ratios": [1, 1]},
    )

    # --- Top: baseline probability ---
    ax0.plot(x, p, color="#9467bd", label="Baseline P(P1 wins)")
    ax0.axhline(0.5, color="gray", lw=1, alpha=0.7)
    ax0.set_ylabel("Probability")
    ax0.set_title(title)

    # --- Bottom: performance (multiple momentum metrics) ---
    # Fast EWMA (reacts quickly to changes)
    flow_fast = ewma(residual, alpha=0.25)
    # Slow EWMA (smoother, captures longer trends)
    flow_slow = ewma(residual, alpha=0.05)
    # Rolling mean (game-level window ~20 points)
    flow_rolling = rolling_mean(residual, window=20)
    
    ax1.plot(x, flow_fast, color="#2ca02c", label="Fast momentum (α=0.25)", alpha=0.7, lw=1.5)
    ax1.plot(x, flow_slow, color="#1f77b4", label="Slow momentum (α=0.05)", alpha=0.7, lw=1.5)
    ax1.plot(x, flow_rolling, color="#9467bd", label="Rolling mean (20 pts)", alpha=0.7, lw=1.5, linestyle="--")
    ax1.axhline(0, color="gray", lw=1, alpha=0.7)
    ax1.fill_between(x, 0, flow_slow, where=(flow_slow >= 0), color="#1f77b4", alpha=0.08)
    ax1.fill_between(x, 0, flow_slow, where=(flow_slow < 0), color="#d62728", alpha=0.08)
    ax1.set_ylabel("Performance (y − p, smoothed)")
    ax1.set_xlabel("Point index (within match)")
    ax1.legend(loc="upper right", fontsize=9)

    # --- Shared structure markers (game/set) on both axes ---
    for ax in (ax0, ax1):
        for i in np.where(game_change)[0]:
            ax.axvline(i, color="gray", lw=0.5, alpha=0.25)
        for i in np.where(set_change)[0]:
            ax.axvline(i, color="black", lw=1.0, alpha=0.35)

    # --- Server switch markers ONLY on top (probability) axis ---
    for i in server_switch_idx:
        prev_srv = server[i - 1]
        curr_srv = server[i]
        if prev_srv == 1 and curr_srv == 2:
            ax0.axvline(i, color="#1f77b4", lw=1.0, alpha=0.55, linestyle="--")
        elif prev_srv == 2 and curr_srv == 1:
            ax0.axvline(i, color="#ff7f0e", lw=1.0, alpha=0.55, linestyle="--")

    # Figure-level legend for baseline + server-switch markers (performance legend is on ax1)
    handles = [
        ax0.lines[0],  # baseline probability line
        Line2D([0], [0], color="#1f77b4", lw=1.0, linestyle="--", label="Server switch P1→P2"),
        Line2D([0], [0], color="#ff7f0e", lw=1.0, linestyle="--", label="Server switch P2→P1"),
    ]
    fig.legend(handles=handles, loc="upper right")
    plt.tight_layout()

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
    X, y, groups = build_baseline_Xy(df)

    booster, enc, num_cols, cat_cols = train_xgb_baseline(X, y, groups)

    # Choose a match to visualize (default to Alcaraz vs Djokovic final if present)
    match_id = _pick_match_id(df, preferred="2023-wimbledon-1701")
    mask = (df["match_id"] == match_id)
    df_m = df.loc[mask].copy()
    X_m = X.loc[mask].copy()
    y_m = y[mask]

    p_m = predict_proba(booster, enc, num_cols, cat_cols, X_m)
    residual_m = y_m - p_m
    # Use slow EWMA as primary flow metric (can be changed)
    flow_m = ewma(residual_m, alpha=0.05)

    title = f"Match flow (XGBoost baseline) — {df_m['player1'].iloc[0]} vs {df_m['player2'].iloc[0]}"
    plot_match_summary(df_m, p_m, flow_m, residual_m, title=title)
