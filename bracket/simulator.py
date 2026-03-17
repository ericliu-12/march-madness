"""Single bracket simulation engine.

Simulates the entire tournament game-by-game using the ensemble
model to generate win probabilities and stochastic outcomes.
"""

import random

import numpy as np

from bracket.structure import Bracket, Game, Team
from features.matchup_features import compute_matchup_features
from models.calibration import clip_probability


def simulate_game(
    game: Game,
    model,
    use_random: bool = True,
) -> Team:
    """Simulate a single game.

    Args:
        game: The game to simulate
        model: Ensemble model with predict_single() method
        use_random: If True, draw stochastically. If False, pick the favorite.

    Returns:
        The winning team.
    """
    features = compute_matchup_features(game.team_a.stats, game.team_b.stats)
    prob_a = model.predict_single(features)
    prob_a = clip_probability(prob_a)

    game.win_probability = prob_a

    if use_random:
        winner = game.team_a if random.random() < prob_a else game.team_b
    else:
        winner = game.team_a if prob_a >= 0.5 else game.team_b

    game.winner = winner
    return winner


def simulate_bracket(
    bracket: Bracket,
    model,
    use_random: bool = True,
    seed: int | None = None,
) -> Bracket:
    """Simulate the entire tournament.

    Args:
        bracket: The bracket to simulate (will be copied if use_random=True)
        model: Ensemble model
        use_random: Stochastic (True) or deterministic (False)
        seed: Random seed for reproducibility

    Returns:
        The simulated bracket with winners filled in.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    b = bracket.copy() if use_random else bracket

    # Simulate each round
    for round_num in range(1, 7):
        games = b.rounds.get(round_num, [])
        for game in games:
            simulate_game(game, model, use_random=use_random)

        # Generate next round matchups
        if round_num < 6:
            b.advance_winners(round_num)

    return b


def get_most_likely_bracket(bracket: Bracket, model) -> Bracket:
    """Generate the single most likely bracket (deterministic).

    Always picks the team with the higher win probability.
    """
    return simulate_bracket(bracket, model, use_random=False)
