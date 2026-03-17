"""Monte Carlo bracket simulation.

Runs thousands of bracket simulations to produce probability
distributions for how far each team advances.

Optimized: pre-computes all pairwise win probabilities once,
then simulations are pure random draws (~1000 sims/second).
"""

from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

from bracket.structure import Bracket, Team
from features.matchup_features import compute_matchup_features
from models.calibration import clip_probability
from config import MONTE_CARLO_SIMULATIONS, RANDOM_SEED, ROUNDS, SEED_MATCHUPS, REGIONS


def _precompute_win_probs(teams: list[Team], model) -> dict[tuple[str, str], float]:
    """Pre-compute P(A wins) for all possible team pairings.

    This is the key optimization: instead of calling the model
    63 times per simulation, we call it once for all O(n^2) pairs
    and cache the results.
    """
    probs = {}
    team_list = list(teams)
    n = len(team_list)

    for i in range(n):
        for j in range(i + 1, n):
            a = team_list[i]
            b = team_list[j]
            features = compute_matchup_features(a.stats, b.stats)
            p = model.predict_single(features)
            p = clip_probability(p)
            probs[(a.name, b.name)] = p
            probs[(b.name, a.name)] = 1 - p

    return probs


def _get_prob(probs: dict, name_a: str, name_b: str) -> float:
    """Get P(A wins) from precomputed probabilities."""
    key = (name_a, name_b)
    if key in probs:
        return probs[key]
    return 0.5


def _simulate_bracket_fast(
    region_teams: dict[str, list[tuple[int, str]]],
    probs: dict,
    rng: np.random.Generator,
) -> dict:
    """Simulate one full tournament using precomputed probabilities.

    Returns dict mapping round -> list of (winner_name, winner_seed).
    Also returns the champion name.
    """
    results = {}

    # Simulate each region
    region_winners = {}
    for region in REGIONS:
        teams = region_teams.get(region, [])
        if len(teams) < 2:
            continue

        # Build matchup order by seed pairing
        teams_by_seed = {seed: name for seed, name in teams}
        current_round = []

        for s1, s2 in SEED_MATCHUPS:
            a = teams_by_seed.get(s1)
            b = teams_by_seed.get(s2)
            if a and b:
                current_round.append((s1, a, s2, b))

        round_num = 1
        while len(current_round) >= 1:
            winners = []
            for matchup in current_round:
                s_a, name_a, s_b, name_b = matchup
                p = _get_prob(probs, name_a, name_b)
                if rng.random() < p:
                    winners.append((s_a, name_a))
                else:
                    winners.append((s_b, name_b))

                # Track both participants reached this round
                key = (round_num, region)
                if key not in results:
                    results[key] = []
                results[key].append(name_a)
                results[key].append(name_b)

            if len(winners) == 1:
                region_winners[region] = winners[0]
                break

            # Pair up winners for next round
            current_round = []
            for i in range(0, len(winners) - 1, 2):
                s_a, name_a = winners[i]
                s_b, name_b = winners[i + 1]
                current_round.append((s_a, name_a, s_b, name_b))

            round_num += 1

    # Final Four
    ff_pairings = [
        (REGIONS[0], REGIONS[1]),  # East vs South
        (REGIONS[2], REGIONS[3]),  # Midwest vs West
    ]

    ff_winners = []
    for r1, r2 in ff_pairings:
        if r1 in region_winners and r2 in region_winners:
            s_a, name_a = region_winners[r1]
            s_b, name_b = region_winners[r2]
            p = _get_prob(probs, name_a, name_b)

            results.setdefault((5, "FF"), []).extend([name_a, name_b])

            if rng.random() < p:
                ff_winners.append((s_a, name_a))
            else:
                ff_winners.append((s_b, name_b))

    # Championship
    champion = None
    if len(ff_winners) >= 2:
        s_a, name_a = ff_winners[0]
        s_b, name_b = ff_winners[1]
        p = _get_prob(probs, name_a, name_b)

        results.setdefault((6, "Champ"), []).extend([name_a, name_b])

        if rng.random() < p:
            champion = name_a
        else:
            champion = name_b

    return results, champion


def run_monte_carlo(
    bracket: Bracket,
    model,
    n_simulations: int = MONTE_CARLO_SIMULATIONS,
) -> dict:
    """Run Monte Carlo simulation of the tournament.

    Pre-computes all pairwise win probabilities, then runs fast
    random simulations.
    """
    all_teams = bracket.get_all_teams()

    print(f"\nPre-computing {len(all_teams)} x {len(all_teams)} win probabilities...")
    probs = _precompute_win_probs(all_teams, model)
    print(f"  Cached {len(probs)} pairwise probabilities")

    # Build region teams structure for fast simulation
    region_teams = {}
    for region, teams in bracket.regions.items():
        region_teams[region] = [(t.seed, t.name) for t in teams]

    # Track results
    team_names = [t.name for t in all_teams]
    round_counts = defaultdict(lambda: Counter())
    champion_counts = Counter()

    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Running {n_simulations:,} Monte Carlo simulations...")
    for _ in tqdm(range(n_simulations), desc="Simulating"):
        results, champion = _simulate_bracket_fast(region_teams, probs, rng)

        # Count round appearances
        for (round_num, _), participants in results.items():
            for name in participants:
                round_counts[round_num][name] += 1

        if champion:
            champion_counts[champion] += 1

    # Convert to advancement probabilities
    advancement = {}
    round_names = {1: "First Round", 2: "Second Round", 3: "Sweet 16",
                   4: "Elite 8", 5: "Final Four", 6: "Championship"}

    for team_name in team_names:
        team_adv = {}
        for round_num, round_name in round_names.items():
            count = round_counts[round_num].get(team_name, 0)
            team_adv[round_name] = count / n_simulations
        team_adv["Champion"] = champion_counts.get(team_name, 0) / n_simulations
        advancement[team_name] = team_adv

    most_common = champion_counts.most_common(1)
    most_common_champ = most_common[0][0] if most_common else "Unknown"
    champ_prob = most_common[0][1] / n_simulations if most_common else 0

    return {
        "advancement": advancement,
        "champion_counts": champion_counts,
        "n_simulations": n_simulations,
        "most_common_champion": most_common_champ,
        "champion_probability": champ_prob,
        "top_10_champions": [
            (name, count / n_simulations)
            for name, count in champion_counts.most_common(10)
        ],
    }


def print_monte_carlo_results(mc_results: dict):
    """Print a summary of Monte Carlo simulation results."""
    print("\n" + "=" * 70)
    print("MONTE CARLO SIMULATION RESULTS")
    print(f"({mc_results['n_simulations']:,} simulations)")
    print("=" * 70)

    print(f"\nMost likely champion: {mc_results['most_common_champion']} "
          f"({mc_results['champion_probability']:.1%})")

    print("\nTop 10 Championship Contenders:")
    print(f"  {'Team':<25s} {'Champion %':>10s}")
    print(f"  {'-'*25} {'-'*10}")
    for name, prob in mc_results["top_10_champions"]:
        bar = "█" * int(prob * 50)
        print(f"  {name:<25s} {prob:>9.1%} {bar}")

    print("\nAdvancement Probabilities (Top 20 teams):")
    print(f"  {'Team':<22s} {'R32':>6s} {'S16':>6s} {'E8':>6s} {'F4':>6s} {'Final':>6s} {'Champ':>6s}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    sorted_teams = sorted(
        mc_results["advancement"].items(),
        key=lambda x: x[1].get("Champion", 0),
        reverse=True,
    )

    for name, adv in sorted_teams[:20]:
        print(
            f"  {name:<22s} "
            f"{adv.get('Second Round', 0):>5.0%} "
            f"{adv.get('Sweet 16', 0):>5.0%} "
            f"{adv.get('Elite 8', 0):>5.0%} "
            f"{adv.get('Final Four', 0):>5.0%} "
            f"{adv.get('Championship', 0):>5.0%} "
            f"{adv.get('Champion', 0):>5.0%}"
        )
