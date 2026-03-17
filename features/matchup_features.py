"""Pairwise matchup feature computation.

Given two team feature profiles, computes ~25 features that capture
how each team's strengths and weaknesses interact.
"""

import numpy as np
import pandas as pd

from data.kaggle_loader import SEED_WIN_RATES


def compute_matchup_features(team_a: pd.Series, team_b: pd.Series) -> dict:
    """Compute matchup features between two teams.

    Args:
        team_a: Feature series for team A (from team_features.extract_features)
        team_b: Feature series for team B

    Returns:
        Dictionary of matchup features.
    """
    features = {}

    # --- Seed features ---
    seed_a = team_a.get("seed", 8)
    seed_b = team_b.get("seed", 8)
    seed_a = seed_a if pd.notna(seed_a) else 8
    seed_b = seed_b if pd.notna(seed_b) else 8

    features["seed_diff"] = seed_a - seed_b
    features["seed_a"] = seed_a
    features["seed_b"] = seed_b

    # --- Efficiency features ---
    for col in ["adjoe", "adjde", "adjem", "barthag",
                "off_efficiency", "def_efficiency", "efficiency_margin"]:
        a_val = _get(team_a, col)
        b_val = _get(team_b, col)
        if a_val is not None and b_val is not None:
            features[f"{col}_diff"] = a_val - b_val

    # --- Four factors matchup ---
    # Each offensive factor vs opponent's defensive factor
    four_factors = [("efg", "efg"), ("tov", "tov"), ("orb", "orb"), ("ftr", "ftr")]
    for off_name, def_name in four_factors:
        # A's offense vs B's defense
        a_off = _get(team_a, f"{off_name}_o")
        b_def = _get(team_b, f"{def_name}_d")
        if a_off is not None and b_def is not None:
            features[f"a_{off_name}_off_vs_b_def"] = a_off - b_def

        # B's offense vs A's defense
        b_off = _get(team_b, f"{off_name}_o")
        a_def = _get(team_a, f"{def_name}_d")
        if b_off is not None and a_def is not None:
            features[f"b_{off_name}_off_vs_a_def"] = b_off - a_def

    # --- Cross-efficiency matchup ---
    # How A's offense fares against B's defense strength
    adjoe_a = _get(team_a, "adjoe")
    adjde_b = _get(team_b, "adjde")
    adjoe_b = _get(team_b, "adjoe")
    adjde_a = _get(team_a, "adjde")

    if all(v is not None for v in [adjoe_a, adjde_b]):
        features["a_off_vs_b_def"] = adjoe_a - adjde_b
    if all(v is not None for v in [adjoe_b, adjde_a]):
        features["b_off_vs_a_def"] = adjoe_b - adjde_a

    # --- Tempo mismatch ---
    tempo_a = _get(team_a, "pace") or _get(team_a, "adj_tempo")
    tempo_b = _get(team_b, "pace") or _get(team_b, "adj_tempo")
    if tempo_a is not None and tempo_b is not None:
        features["tempo_mismatch"] = abs(tempo_a - tempo_b)
        features["tempo_diff"] = tempo_a - tempo_b

    # --- Strength features ---
    for col in ["srs", "sos", "win_pct", "point_diff", "ppg", "opp_ppg"]:
        a_val = _get(team_a, col)
        b_val = _get(team_b, col)
        if a_val is not None and b_val is not None:
            features[f"{col}_diff"] = a_val - b_val

    # --- Three-point and shooting ---
    for col in ["three_rate_o", "ts_pct"]:
        a_val = _get(team_a, col)
        b_val = _get(team_b, col)
        if a_val is not None and b_val is not None:
            features[f"{col}_diff"] = a_val - b_val

    # --- Historical seed matchup ---
    s_low = int(min(seed_a, seed_b))
    s_high = int(max(seed_a, seed_b))
    base_rate = SEED_WIN_RATES.get((s_low, s_high), 0.5)
    # Flip if team_a is the higher seed
    if seed_a > seed_b:
        base_rate = 1 - base_rate
    features["hist_seed_win_rate"] = base_rate

    return features


def compute_matchup_dataframe(
    team_a: pd.Series,
    team_b: pd.Series,
) -> pd.DataFrame:
    """Compute matchup features and return as a single-row DataFrame."""
    feats = compute_matchup_features(team_a, team_b)
    return pd.DataFrame([feats])


def _get(series: pd.Series, key: str) -> float | None:
    """Get a numeric value from a series, returning None if missing."""
    val = series.get(key)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
