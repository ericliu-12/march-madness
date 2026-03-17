"""CSV export for predictions.

Exports game-by-game predictions, advancement probabilities,
and upset alerts to CSV files.
"""

import os

import pandas as pd

from bracket.structure import Bracket
from config import OUTPUT_DIR, ROUNDS


def export_game_predictions(bracket: Bracket, output_dir: str = OUTPUT_DIR):
    """Export game-by-game predictions to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for round_num in range(1, 7):
        games = bracket.rounds.get(round_num, [])
        for game in games:
            prob_a = game.win_probability
            row = {
                "round": ROUNDS.get(round_num, f"Round {round_num}"),
                "round_num": round_num,
                "region": game.region,
                "team_a": game.team_a.name,
                "seed_a": game.team_a.seed,
                "team_b": game.team_b.name,
                "seed_b": game.team_b.seed,
                "prob_team_a_wins": f"{prob_a:.4f}" if prob_a is not None else "",
                "predicted_winner": game.winner.name if game.winner else "",
                "is_upset": (
                    game.winner.seed > game.higher_seed.seed
                    if game.winner else ""
                ),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "predictions_game_by_game.csv")
    df.to_csv(path, index=False)
    print(f"Saved game predictions to {path}")
    return path


def export_advancement_probs(mc_results: dict, output_dir: str = OUTPUT_DIR):
    """Export Monte Carlo advancement probabilities to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for team_name, probs in mc_results["advancement"].items():
        row = {"team": team_name}
        row.update(probs)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by championship probability
    if "Champion" in df.columns:
        df = df.sort_values("Champion", ascending=False)

    path = os.path.join(output_dir, "predictions_advancement.csv")
    df.to_csv(path, index=False)
    print(f"Saved advancement probabilities to {path}")
    return path


def export_upset_alerts(upsets: list[dict], output_dir: str = OUTPUT_DIR):
    """Export upset alerts to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for upset in upsets:
        rows.append({
            "round": upset["round"],
            "matchup": upset["matchup"],
            "underdog": upset["underdog"],
            "favorite": upset["favorite"],
            "underdog_seed": upset["underdog_seed"],
            "favorite_seed": upset["favorite_seed"],
            "upset_probability": f"{upset['upset_probability']:.4f}",
            "historical_rate": f"{upset['historical_upset_rate']:.4f}",
            "upset_score": f"{upset['upset_score']:.4f}",
            "region": upset["region"],
            "key_factors": "; ".join(upset["key_factors"]),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "upset_alerts.csv")
    df.to_csv(path, index=False)
    print(f"Saved upset alerts to {path}")
    return path
