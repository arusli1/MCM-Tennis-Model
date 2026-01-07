import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Load data
# -----------------------------
def load_data(path: str | None = None) -> pd.DataFrame:
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "data", "2024_Wimbledon_featured_matches.csv")
    df = pd.read_csv(path)
    df = df.sort_values(["match_id", "set_no", "game_no", "point_no"]).reset_index(drop=True)
    return df


# -----------------------------
# EWMA smoothing
# -----------------------------
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


# -----------------------------
# Plot performance flow
# -----------------------------
def plot_performance_flow(df_match: pd.DataFrame, residual: np.ndarray, title: str, alpha: float = 0.15):
    """Plot performance flow using EWMA of residuals."""
    game_change = df_match["game_no"].diff().fillna(0).ne(0).to_numpy()
    set_change = df_match["set_no"].diff().fillna(0).ne(0).to_numpy()
    
    x = np.arange(len(df_match))
    
    # Compute EWMA
    flow_ewma = ewma(residual, alpha=alpha)
    flow_ewma = np.nan_to_num(flow_ewma, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(18, 5))
    
    ax.plot(x, flow_ewma, color="#2ca02c", label=f"Performance Flow (EWMA, α={alpha:.2f})", lw=2.0, alpha=0.9)
    ax.axhline(0, color="gray", lw=1.5, alpha=0.6, linestyle="--")
    ax.fill_between(x, 0, flow_ewma, where=(flow_ewma >= 0), color="#2ca02c", alpha=0.25)
    ax.fill_between(x, 0, flow_ewma, where=(flow_ewma < 0), color="#d62728", alpha=0.25)
    
    ax.set_ylabel("Performance Flow", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Point Index", fontsize=11)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.tick_params(labelsize=10)
    
    # Set boundaries
    for i in np.where(game_change)[0]:
        ax.axvline(i, color="lightgray", lw=0.5, alpha=0.3)
    for i in np.where(set_change)[0]:
        ax.axvline(i, color="black", lw=1.5, alpha=0.5)
    
    # Auto-scale y-axis
    y_min, y_max = float(flow_ewma.min()), float(flow_ewma.max())
    if not (np.isfinite(y_min) and np.isfinite(y_max)):
        y_min, y_max = -0.1, 0.1
    
    y_range = y_max - y_min
    if y_range < 0.01 or not np.isfinite(y_range):
        y_center = (y_max + y_min) / 2 if np.isfinite((y_max + y_min) / 2) else 0.0
        ax.set_ylim(y_center - 0.1, y_center + 0.1)
    else:
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    plt.tight_layout()
    
    # Save figure
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    match_id = str(df_match["match_id"].iloc[0]) if "match_id" in df_match.columns else "match"
    out_path = os.path.join(results_dir, f"{match_id}_test.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")
    
    plt.show()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    df = load_data()
    
    # Choose target match
    target_match_id = "2023-wimbledon-1701"  # Alcaraz vs Djokovic
    if target_match_id not in set(df["match_id"].unique()):
        target_match_id = df["match_id"].iloc[0]
        print(f"Warning: Target match '2023-wimbledon-1701' not found. Using '{target_match_id}' instead.")
    
    # Extract target match
    df_match = df.loc[df["match_id"] == target_match_id].copy()
    
    # Simple baseline: constant 0.5 (accounts for nothing)
    y = (df_match["point_victor"] == 1).astype(int).to_numpy()
    p = np.full(len(df_match), 0.5)  # Constant 0.5 baseline
    
    # Compute residuals
    residual = y - p
    
    print(f"\nTest Model - Constant Baseline (p=0.5)")
    print(f"Match: {df_match['player1'].iloc[0]} vs {df_match['player2'].iloc[0]}")
    print(f"Points: {len(df_match)}")
    print(f"Mean residual: {residual.mean():.4f}")
    print(f"Mean absolute residual: {np.abs(residual).mean():.4f}")
    
    # Plot performance flow
    title = f"Performance Flow (Constant Baseline p=0.5) — {df_match['player1'].iloc[0]} vs {df_match['player2'].iloc[0]}"
    plot_performance_flow(df_match, residual, title, alpha=0.15)

