"""Monte Carlo bracket simulation.

Runs thousands of bracket simulations to produce probability
distributions for how far each team advances.
"""

from collections import Counter, defaultdict

from tqdm import tqdm

from bracket.structure import Bracket
from bracket.simulator import simulate_bracket
from config import MONTE_CARLO_SIMULATIONS, RANDOM_SEED, ROUNDS


def run_monte_carlo(
    bracket: Bracket,
    model,
    n_simulations: int = MONTE_CARLO_SIMULATIONS,
) -> dict:
    """Run Monte Carlo simulation of the tournament.

    Returns a dict with:
        - 'advancement': {team_name: {round: probability}}
        - 'champion_counts': Counter of championship wins
        - 'final_four_counts': Counter of Final Four appearances
        - 'most_common_champion': Most likely champion
    """
    all_teams = bracket.get_all_teams()
    team_names = [t.name for t in all_teams]

    # Track how far each team goes in each simulation
    # advancement[team_name][round_num] = count of times reached
    advancement = defaultdict(lambda: Counter())
    champion_counts = Counter()

    print(f"\nRunning {n_simulations:,} Monte Carlo simulations...")
    for i in tqdm(range(n_simulations), desc="Simulating"):
        sim = simulate_bracket(bracket, model, use_random=True, seed=RANDOM_SEED + i)

        # Track advancement for each round
        for round_num, games in sim.rounds.items():
            for game in games:
                # Both teams in this game reached this round
                advancement[game.team_a.name][round_num] += 1
                advancement[game.team_b.name][round_num] += 1

                # Winner advances to next round
                if game.winner and round_num == 6:
                    champion_counts[game.winner.name] += 1

    # Convert to probabilities
    results = {}
    for team_name in team_names:
        team_adv = {}
        for round_num in range(1, 7):
            count = advancement[team_name][round_num]
            team_adv[ROUNDS[round_num]] = count / n_simulations

        # Championship probability
        team_adv["Champion"] = champion_counts.get(team_name, 0) / n_simulations
        results[team_name] = team_adv

    # Find most common champion
    most_common = champion_counts.most_common(1)
    most_common_champ = most_common[0][0] if most_common else "Unknown"
    champ_prob = most_common[0][1] / n_simulations if most_common else 0

    return {
        "advancement": results,
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

    # Print advancement table for top teams
    print("\nAdvancement Probabilities (Top 20 teams):")
    print(f"  {'Team':<22s} {'R32':>6s} {'S16':>6s} {'E8':>6s} {'F4':>6s} {'Final':>6s} {'Champ':>6s}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    # Sort by championship probability
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
