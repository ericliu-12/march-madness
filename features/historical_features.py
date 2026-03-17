"""Historical tournament features.

Provides seed-based priors and historical tournament performance
metrics that supplement the team-level and matchup-level features.
"""

import numpy as np

from data.kaggle_loader import SEED_WIN_RATES


# Probability of reaching each round by seed (based on 1985-2024 data)
SEED_ROUND_PROBS = {
    # seed: (R64, R32, S16, E8, F4, Finals, Champ)
    1:  (1.00, 0.99, 0.88, 0.72, 0.51, 0.32, 0.20),
    2:  (1.00, 0.94, 0.70, 0.50, 0.32, 0.18, 0.10),
    3:  (1.00, 0.85, 0.54, 0.33, 0.17, 0.09, 0.04),
    4:  (1.00, 0.79, 0.47, 0.28, 0.14, 0.07, 0.03),
    5:  (1.00, 0.65, 0.36, 0.20, 0.09, 0.04, 0.02),
    6:  (1.00, 0.62, 0.33, 0.17, 0.08, 0.04, 0.02),
    7:  (1.00, 0.61, 0.27, 0.14, 0.07, 0.03, 0.01),
    8:  (1.00, 0.52, 0.20, 0.09, 0.04, 0.02, 0.01),
    9:  (1.00, 0.48, 0.17, 0.07, 0.03, 0.01, 0.005),
    10: (1.00, 0.39, 0.16, 0.07, 0.03, 0.01, 0.005),
    11: (1.00, 0.38, 0.15, 0.08, 0.04, 0.02, 0.01),
    12: (1.00, 0.35, 0.12, 0.04, 0.02, 0.01, 0.003),
    13: (1.00, 0.21, 0.05, 0.01, 0.005, 0.002, 0.001),
    14: (1.00, 0.15, 0.03, 0.01, 0.003, 0.001, 0.0005),
    15: (1.00, 0.06, 0.02, 0.005, 0.002, 0.001, 0.0003),
    16: (1.00, 0.01, 0.005, 0.001, 0.0005, 0.0002, 0.0001),
}


def get_seed_advancement_probs(seed: int) -> dict[str, float]:
    """Get historical advancement probabilities for a seed."""
    probs = SEED_ROUND_PROBS.get(seed, SEED_ROUND_PROBS[8])
    return {
        "R64": probs[0],
        "R32": probs[1],
        "S16": probs[2],
        "E8": probs[3],
        "F4": probs[4],
        "Finals": probs[5],
        "Champion": probs[6],
    }


def get_seed_matchup_prob(seed_a: int, seed_b: int) -> float:
    """Get the historical win probability for seed_a against seed_b.

    Returns the probability that seed_a wins.
    """
    if seed_a == seed_b:
        return 0.5

    s_low = min(seed_a, seed_b)
    s_high = max(seed_a, seed_b)

    # Lower seed wins at rate from lookup
    lower_wins = SEED_WIN_RATES.get((s_low, s_high), 0.5)

    if seed_a == s_low:
        return lower_wins
    else:
        return 1 - lower_wins


def compute_upset_score(seed_higher: int, seed_lower: int, model_prob: float) -> float:
    """Compute an upset likelihood score.

    Higher score = more likely/notable upset.

    Args:
        seed_higher: The numerically higher seed (underdog)
        seed_lower: The numerically lower seed (favorite)
        model_prob: Model's probability that the underdog wins

    Returns:
        Upset score (0-1 range, higher = more likely upset)
    """
    seed_gap = seed_higher - seed_lower
    historical_baseline = 1 - get_seed_matchup_prob(seed_lower, seed_higher)

    # Weight model probability heavily, but factor in historical rarity
    # A 12-over-5 upset is less surprising than a 15-over-2
    rarity_factor = seed_gap / 15.0  # Normalize to 0-1
    model_signal = model_prob

    # Blend: if the model thinks the upset is likely AND it's rare, score higher
    score = 0.7 * model_signal + 0.3 * (model_signal * rarity_factor)

    return np.clip(score, 0, 1)
