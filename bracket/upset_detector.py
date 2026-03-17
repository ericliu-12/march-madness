"""Upset detection and analysis.

Identifies likely upsets where the lower-seeded team has a significant
chance of winning, and explains the key factors driving the upset potential.
"""

from features.matchup_features import compute_matchup_features
from features.historical_features import compute_upset_score, get_seed_matchup_prob
from bracket.structure import Bracket, Game
from models.calibration import clip_probability


def detect_upsets(
    bracket: Bracket,
    model,
    threshold: float = 0.30,
) -> list[dict]:
    """Identify potential upsets in the first two rounds.

    An "upset" is when the higher-seeded team (underdog) has at least
    `threshold` probability of winning.

    Returns a sorted list of upset alerts.
    """
    upsets = []

    # Check first two rounds
    for round_num in [1, 2]:
        games = bracket.rounds.get(round_num, [])
        for game in games:
            seed_a = game.team_a.seed
            seed_b = game.team_b.seed

            if seed_a == seed_b:
                continue

            # Determine who's the favorite and underdog
            if seed_a < seed_b:
                favorite = game.team_a
                underdog = game.team_b
            else:
                favorite = game.team_b
                underdog = game.team_a

            # Get model probability
            features = compute_matchup_features(game.team_a.stats, game.team_b.stats)
            prob_a = model.predict_single(features)
            prob_a = clip_probability(prob_a)

            # P(underdog wins)
            if underdog == game.team_a:
                p_upset = prob_a
            else:
                p_upset = 1 - prob_a

            if p_upset >= threshold:
                # Compute upset score
                upset_score = compute_upset_score(
                    underdog.seed, favorite.seed, p_upset
                )

                # Analyze why
                factors = _explain_upset(favorite, underdog, features)

                historical_rate = 1 - get_seed_matchup_prob(
                    favorite.seed, underdog.seed
                )

                upsets.append({
                    "round": round_num,
                    "matchup": f"#{underdog.seed} {underdog.name} over #{favorite.seed} {favorite.name}",
                    "underdog": underdog.name,
                    "favorite": favorite.name,
                    "underdog_seed": underdog.seed,
                    "favorite_seed": favorite.seed,
                    "upset_probability": p_upset,
                    "upset_score": upset_score,
                    "historical_upset_rate": historical_rate,
                    "region": game.region,
                    "key_factors": factors,
                })

    # Sort by upset probability (highest first)
    upsets.sort(key=lambda x: x["upset_probability"], reverse=True)
    return upsets


def _explain_upset(favorite, underdog, features: dict) -> list[str]:
    """Generate human-readable explanations for an upset prediction."""
    factors = []

    # Check efficiency matchup
    adjem_diff = features.get("adjem_diff") or features.get("efficiency_margin_diff")
    if adjem_diff is not None:
        # If the underdog is team_b and diff < 0, underdog has better efficiency
        # This is approximate - sign depends on team ordering
        if abs(adjem_diff) < 3:
            factors.append("Efficiency margins are very close")

    # Check four factors advantages
    for factor_name, label in [
        ("a_efg_off_vs_b_def", "shooting efficiency"),
        ("a_tov_off_vs_b_def", "turnover forcing"),
        ("a_orb_off_vs_b_def", "offensive rebounding"),
        ("a_ftr_off_vs_b_def", "free throw rate"),
    ]:
        val = features.get(factor_name)
        if val is not None and abs(val) > 3:
            factors.append(f"Significant {label} advantage in matchup")

    # Check tempo mismatch
    tempo_mismatch = features.get("tempo_mismatch")
    if tempo_mismatch is not None and tempo_mismatch > 5:
        factors.append(f"Large tempo mismatch ({tempo_mismatch:.1f} possessions)")

    # Check SOS difference
    sos_diff = features.get("sos_diff")
    if sos_diff is not None and abs(sos_diff) > 2:
        factors.append("Strength of schedule discrepancy")

    if not factors:
        factors.append("Model sees overall statistical edge for underdog")

    return factors


def print_upset_alerts(upsets: list[dict]):
    """Print upset alerts in a readable format."""
    if not upsets:
        print("\nNo significant upset alerts detected.")
        return

    print("\n" + "=" * 70)
    print("UPSET ALERTS")
    print("=" * 70)

    for i, upset in enumerate(upsets, 1):
        hist_rate = upset["historical_upset_rate"]
        model_prob = upset["upset_probability"]

        # Flag if model rate significantly exceeds historical rate
        hot = " 🔥" if model_prob > hist_rate * 1.5 else ""

        print(f"\n  {i}. {upset['matchup']}{hot}")
        print(f"     Region: {upset['region']} | Round {upset['round']}")
        print(f"     Model upset prob: {model_prob:.1%} "
              f"(historical: {hist_rate:.1%})")
        print(f"     Upset score: {upset['upset_score']:.2f}")
        print(f"     Key factors:")
        for factor in upset["key_factors"]:
            print(f"       - {factor}")
