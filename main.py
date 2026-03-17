#!/usr/bin/env python3
"""March Madness 2026 Bracket Prediction Model.

Collects team statistics from multiple sources, trains an ensemble
of ML models on historical tournament data, and simulates the
full bracket with Monte Carlo analysis.

Usage:
    python main.py                    # Run with cached data
    python main.py --scrape           # Force re-scrape all data
    python main.py --simulations 5000 # Custom simulation count
    python main.py --no-monte-carlo   # Skip Monte Carlo (faster)
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd

from config import MONTE_CARLO_SIMULATIONS, OUTPUT_DIR, RANDOM_SEED
from data.scraper_ncaa import get_bracket_from_fallback
from features.builder import collect_data, build_team_profiles, build_training_data
from features.matchup_features import compute_matchup_features
from bracket.structure import build_bracket_from_teams
from bracket.simulator import get_most_likely_bracket
from bracket.monte_carlo import run_monte_carlo, print_monte_carlo_results
from bracket.upset_detector import detect_upsets, print_upset_alerts
from output.bracket_display import display_bracket, display_compact_bracket
from output.csv_export import export_game_predictions, export_advancement_probs, export_upset_alerts
from output.visualizations import generate_all_visualizations
from models.ensemble import EnsembleModel


def main():
    parser = argparse.ArgumentParser(description="March Madness 2026 Bracket Predictor")
    parser.add_argument("--scrape", action="store_true", help="Force re-scrape all data")
    parser.add_argument("--simulations", type=int, default=MONTE_CARLO_SIMULATIONS,
                        help=f"Number of Monte Carlo simulations (default: {MONTE_CARLO_SIMULATIONS})")
    parser.add_argument("--no-monte-carlo", action="store_true", help="Skip Monte Carlo simulation")
    parser.add_argument("--compact", action="store_true", help="Use compact bracket display")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory for results")
    args = parser.parse_args()

    start_time = time.time()
    np.random.seed(RANDOM_SEED)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     MARCH MADNESS 2026 — BRACKET PREDICTION MODEL      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ─── Step 1: Collect Data ───
    data = collect_data(force_scrape=args.scrape)

    # ─── Step 2: Build Team Profiles ───
    team_features = build_team_profiles(data)
    print(f"\nTeam features shape: {team_features.shape}")

    # ─── Step 3: Train Model ───
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    ensemble = EnsembleModel()

    # Try historical training data first
    training_data = build_training_data(data)

    if training_data is not None:
        X_train, y_train = training_data
        print(f"\nTraining on {len(X_train)} historical games...")

        # Extract season labels if available
        seasons = None
        if "season" in X_train.columns:
            seasons = X_train.pop("season")

        ensemble.fit(X_train, y_train, seasons=seasons)
    else:
        print("\nNo historical training data available.")
        print("Training on synthetic data derived from seed-based priors...")
        X_train, y_train = _generate_synthetic_training_data(team_features)
        ensemble.fit(X_train, y_train)

    # Show feature importance
    importance = ensemble.get_feature_importance()
    print("\nTop 10 Most Important Features:")
    for i, (feat, val) in enumerate(list(importance.items())[:10], 1):
        print(f"  {i:2d}. {feat:<30s} {val:.4f}")

    # ─── Step 4: Build Bracket ───
    print("\n" + "=" * 60)
    print("BUILDING BRACKET")
    print("=" * 60)

    tournament_teams = data["tournament_teams"]
    bracket = build_bracket_from_teams(tournament_teams, team_features)

    n_teams = sum(len(t) for t in bracket.regions.values())
    print(f"  {n_teams} teams in bracket across {len(bracket.regions)} regions")
    for region, teams in bracket.regions.items():
        print(f"  {region}: {len(teams)} teams")

    # ─── Step 5: Predict Most Likely Bracket ───
    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)

    predicted = get_most_likely_bracket(bracket, ensemble)

    if args.compact:
        display_compact_bracket(predicted)
    else:
        display_bracket(predicted)

    # ─── Step 6: Upset Detection ───
    upsets = detect_upsets(bracket, ensemble, threshold=0.30)
    print_upset_alerts(upsets)

    # ─── Step 7: Monte Carlo Simulation ───
    mc_results = None
    if not args.no_monte_carlo:
        mc_results = run_monte_carlo(bracket, ensemble, n_simulations=args.simulations)
        print_monte_carlo_results(mc_results)

    # ─── Step 8: Export Results ───
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)

    export_game_predictions(predicted, args.output_dir)
    export_upset_alerts(upsets, args.output_dir)

    if mc_results:
        export_advancement_probs(mc_results, args.output_dir)
        generate_all_visualizations(mc_results, upsets, args.output_dir)

    # ─── Summary ───
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f} seconds")
    print(f"Results saved to {args.output_dir}/")

    # Print champion
    champ_game = predicted.rounds.get(6, [None])[0] if predicted.rounds.get(6) else None
    if champ_game and champ_game.winner:
        print(f"\nPREDICTED CHAMPION: {champ_game.winner}")
        if mc_results:
            champ_name = champ_game.winner.name
            champ_prob = mc_results["advancement"].get(champ_name, {}).get("Champion", 0)
            print(f"Championship probability: {champ_prob:.1%}")


def _generate_synthetic_training_data(
    team_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic training data when no Kaggle data is available.

    Creates matchups between all pairs of tournament teams and uses
    seed-based priors + efficiency margins to generate labels.
    """
    from data.kaggle_loader import SEED_WIN_RATES

    np.random.seed(RANDOM_SEED)
    rows = []
    labels = []

    teams = team_features.dropna(subset=["seed"])
    n = len(teams)

    for i in range(n):
        for j in range(i + 1, n):
            team_a = teams.iloc[i]
            team_b = teams.iloc[j]

            features = compute_matchup_features(team_a, team_b)

            # Generate label based on seed priors + efficiency
            seed_a = int(team_a["seed"])
            seed_b = int(team_b["seed"])
            s_low = min(seed_a, seed_b)
            s_high = max(seed_a, seed_b)
            base_rate = SEED_WIN_RATES.get((s_low, s_high), 0.5)

            if seed_a > seed_b:
                base_rate = 1 - base_rate

            # Adjust for efficiency margin if available
            em_diff = features.get("efficiency_margin_diff") or features.get("adjem_diff", 0)
            if em_diff:
                # Logistic adjustment: each point of efficiency margin ~ 3% win prob
                adjustment = 1 / (1 + np.exp(-em_diff / 10)) - 0.5
                prob_a = np.clip(base_rate + adjustment * 0.3, 0.05, 0.95)
            else:
                prob_a = base_rate

            label = 1 if np.random.random() < prob_a else 0
            rows.append(features)
            labels.append(label)

    X = pd.DataFrame(rows)
    y = pd.Series(labels)

    # Fill NaN
    X = X.fillna(0.0)

    print(f"  Generated {len(X)} synthetic training samples")
    return X, y


if __name__ == "__main__":
    main()
