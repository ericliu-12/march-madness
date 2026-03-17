"""Matplotlib visualizations for prediction results.

Generates charts showing championship probabilities, advancement
paths, and upset analysis.
"""

import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from config import OUTPUT_DIR


def plot_championship_odds(mc_results: dict, output_dir: str = OUTPUT_DIR, top_n: int = 15):
    """Bar chart of championship probabilities for top teams."""
    os.makedirs(output_dir, exist_ok=True)

    top_teams = mc_results["top_10_champions"][:top_n]
    if not top_teams:
        return

    # Extend to top_n if available
    champ_counts = mc_results["champion_counts"]
    n_sims = mc_results["n_simulations"]
    sorted_teams = champ_counts.most_common(top_n)
    names = [t[0] for t in sorted_teams]
    probs = [t[1] / n_sims for t in sorted_teams]

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))[::-1]
    bars = ax.barh(range(len(names)), probs, color=colors)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Championship Probability", fontsize=12)
    ax.set_title("2026 NCAA Tournament — Championship Odds", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    # Add percentage labels
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, "championship_odds.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved championship odds chart to {path}")


def plot_advancement_heatmap(mc_results: dict, output_dir: str = OUTPUT_DIR, top_n: int = 25):
    """Heatmap showing advancement probabilities for top teams."""
    os.makedirs(output_dir, exist_ok=True)

    # Get top teams by championship probability
    adv = mc_results["advancement"]
    champ_sorted = sorted(adv.items(), key=lambda x: x[1].get("Champion", 0), reverse=True)
    top = champ_sorted[:top_n]

    if not top:
        return

    rounds = ["Second Round", "Sweet 16", "Elite 8", "Final Four", "Championship", "Champion"]
    round_labels = ["R32", "S16", "E8", "F4", "Finals", "Champ"]

    names = [t[0] for t in top]
    data = []
    for name, probs in top:
        row = [probs.get(r, 0) for r in rounds]
        data.append(row)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(round_labels)))
    ax.set_xticklabels(round_labels, fontsize=11)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)

    # Add text annotations
    for i in range(len(names)):
        for j in range(len(rounds)):
            val = data[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title("2026 NCAA Tournament — Advancement Probabilities", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Probability", shrink=0.8)
    plt.tight_layout()

    path = os.path.join(output_dir, "advancement_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved advancement heatmap to {path}")


def plot_upset_analysis(upsets: list[dict], output_dir: str = OUTPUT_DIR):
    """Scatter plot of upset probabilities vs seed differential."""
    os.makedirs(output_dir, exist_ok=True)

    if not upsets:
        return

    seed_diffs = [u["underdog_seed"] - u["favorite_seed"] for u in upsets]
    model_probs = [u["upset_probability"] for u in upsets]
    hist_rates = [u["historical_upset_rate"] for u in upsets]
    labels = [f"#{u['underdog_seed']} {u['underdog'][:12]}" for u in upsets]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot model predictions
    scatter = ax.scatter(seed_diffs, model_probs, s=100, c="red", alpha=0.7,
                         label="Model prediction", zorder=3)

    # Plot historical baselines
    ax.scatter(seed_diffs, hist_rates, s=60, c="blue", alpha=0.4,
               marker="x", label="Historical rate", zorder=2)

    # Label points
    for i, label in enumerate(labels):
        ax.annotate(label, (seed_diffs[i], model_probs[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Seed Differential (underdog - favorite)", fontsize=12)
    ax.set_ylabel("Upset Probability", fontsize=12)
    ax.set_title("2026 NCAA Tournament — Upset Analysis", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "upset_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved upset analysis chart to {path}")


def generate_all_visualizations(mc_results: dict, upsets: list[dict], output_dir: str = OUTPUT_DIR):
    """Generate all visualization charts."""
    print("\nGenerating visualizations...")
    plot_championship_odds(mc_results, output_dir)
    plot_advancement_heatmap(mc_results, output_dir)
    plot_upset_analysis(upsets, output_dir)
