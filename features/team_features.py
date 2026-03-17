"""Per-team feature extraction from multiple data sources.

Merges ESPN, Sports-Reference, and Torvik data into a unified
feature set for each team.
"""

import numpy as np
import pandas as pd

from data.team_mapping import normalize_team_name


# Standard feature columns we want for every team
CORE_FEATURES = [
    "seed", "win_pct",
    "adjoe", "adjde", "adjem", "barthag",
    "off_efficiency", "def_efficiency", "efficiency_margin",
    "efg_o", "efg_d", "tov_o", "tov_d",
    "orb_o", "orb_d", "ftr_o", "ftr_d",
    "adj_tempo", "pace",
    "srs", "sos",
    "ppg", "opp_ppg", "point_diff",
    "three_rate_o", "ts_pct",
]


def merge_team_data(
    espn_df: pd.DataFrame | None,
    sportsref_df: pd.DataFrame | None,
    torvik_df: pd.DataFrame | None,
    tournament_teams: list[dict],
) -> pd.DataFrame:
    """Merge all data sources into a unified team feature DataFrame.

    Priority: Torvik > Sports-Ref > ESPN for overlapping stats.
    Falls back to available sources if some are missing.
    """
    # Start with tournament teams as the base
    teams = pd.DataFrame(tournament_teams)
    teams["name_norm"] = teams["name"].apply(normalize_team_name)

    result = teams.copy()

    # Merge ESPN data
    if espn_df is not None and not espn_df.empty:
        espn = _prepare_espn(espn_df)
        result = result.merge(espn, on="name_norm", how="left", suffixes=("", "_espn"))

    # Merge Sports-Reference data
    if sportsref_df is not None and not sportsref_df.empty:
        sref = _prepare_sportsref(sportsref_df)
        result = result.merge(sref, on="name_norm", how="left", suffixes=("", "_sref"))

    # Merge Torvik data
    if torvik_df is not None and not torvik_df.empty:
        torvik = _prepare_torvik(torvik_df)
        result = result.merge(torvik, on="name_norm", how="left", suffixes=("", "_torvik"))

    # Consolidate overlapping columns (prefer Torvik > SportsRef > ESPN)
    result = _consolidate(result)

    return result


def _prepare_espn(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ESPN data for merging."""
    out = pd.DataFrame()
    out["name_norm"] = df["name"].apply(normalize_team_name)

    # Actual ESPN API column names -> our standard names
    gp = pd.to_numeric(df.get("General_GP", pd.Series(dtype=float)), errors="coerce").replace(0, np.nan)

    # Per-game stats
    if "Offensive_PTS" in df.columns and gp is not None:
        out["ppg_espn"] = pd.to_numeric(df["Offensive_PTS"], errors="coerce") / gp

    if "General_REB" in df.columns and gp is not None:
        out["rpg_espn"] = pd.to_numeric(df["General_REB"], errors="coerce") / gp

    if "Offensive_AST" in df.columns and gp is not None:
        out["apg_espn"] = pd.to_numeric(df["Offensive_AST"], errors="coerce") / gp

    if "Offensive_TO" in df.columns and gp is not None:
        out["topg_espn"] = pd.to_numeric(df["Offensive_TO"], errors="coerce") / gp

    if "Defensive_STL" in df.columns and gp is not None:
        out["spg_espn"] = pd.to_numeric(df["Defensive_STL"], errors="coerce") / gp

    if "Defensive_BLK" in df.columns and gp is not None:
        out["bpg_espn"] = pd.to_numeric(df["Defensive_BLK"], errors="coerce") / gp

    # Shooting percentages
    for espn_col, our_col in [
        ("Offensive_FG%", "fg_pct_espn"),
        ("Offensive_3P%", "three_pct_espn"),
        ("Offensive_FT%", "ft_pct_espn"),
        ("Offensive_SC-EFF", "scoring_eff_espn"),
        ("General_AST/TO", "ast_to_ratio_espn"),
    ]:
        if espn_col in df.columns:
            out[our_col] = pd.to_numeric(df[espn_col], errors="coerce")

    # Offensive rebound total for later calculations
    if "Offensive_OR" in df.columns:
        out["oreb_espn"] = pd.to_numeric(df["Offensive_OR"], errors="coerce")
    if "Defensive_DR" in df.columns:
        out["dreb_espn"] = pd.to_numeric(df["Defensive_DR"], errors="coerce")

    return out.drop_duplicates(subset="name_norm")


def _prepare_sportsref(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Sports-Reference data for merging."""
    out = pd.DataFrame()

    name_col = "school_name" if "school_name" in df.columns else df.columns[0]
    out["name_norm"] = df[name_col].apply(normalize_team_name)

    # Actual Sports-Reference column names -> our standard names
    col_map = {
        "srs": "srs",
        "sos": "sos",
        "pace": "pace",
        "off_rtg": "off_efficiency_sref",
        "fta_per_fga_pct": "ftr_o_sref",
        "fg3a_per_fga_pct": "three_rate_o_sref",
        "ts_pct": "ts_pct_sref",
        "trb_pct": "trb_pct",
        "ast_pct": "ast_pct",
        "stl_pct": "stl_pct",
        "blk_pct": "blk_pct",
        "efg_pct": "efg_o_sref",
        "tov_pct": "tov_o_sref",
        "orb_pct": "orb_o_sref",
        "ft_rate": "ftr_o_sref2",
        "wins": "wins_sref",
        "losses": "losses_sref",
        "pts": "pts_sref",
        "opp_pts": "opp_pts_sref",
        "win_loss_pct": "win_pct_sref",
        "g": "games_sref",
    }

    for old_col, new_col in col_map.items():
        if old_col in df.columns:
            out[new_col] = pd.to_numeric(df[old_col], errors="coerce")

    return out.drop_duplicates(subset="name_norm")


def _prepare_torvik(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Bart Torvik data for merging."""
    out = pd.DataFrame()

    name_col = "team" if "team" in df.columns else df.columns[1]
    out["name_norm"] = df[name_col].apply(normalize_team_name)

    col_map = {
        "adjoe": "adjoe",
        "adjde": "adjde",
        "barthag": "barthag",
        "adj_tempo": "adj_tempo",
        "efg_o": "efg_o_torvik",
        "efg_d": "efg_d_torvik",
        "tov_o": "tov_o_torvik",
        "tov_d": "tov_d_torvik",
        "orb_o": "orb_o_torvik",
        "orb_d": "orb_d_torvik",
        "ftr_o": "ftr_o_torvik",
        "ftr_d": "ftr_d_torvik",
        "two_pt_o": "two_pt_o",
        "two_pt_d": "two_pt_d",
        "three_pt_o": "three_pt_o",
        "three_pt_d": "three_pt_d",
        "three_rate_o": "three_rate_o_torvik",
        "three_rate_d": "three_rate_d_torvik",
    }

    for old_col, new_col in col_map.items():
        if old_col in df.columns:
            out[new_col] = pd.to_numeric(df[old_col], errors="coerce")

    return out.drop_duplicates(subset="name_norm")


def _consolidate(df: pd.DataFrame) -> pd.DataFrame:
    """Consolidate overlapping columns across sources.

    Priority: Torvik > Sports-Ref > ESPN.
    """
    # Efficiency metrics
    df["adjoe"] = _coalesce(df, "adjoe", "off_efficiency_sref")
    df["adjde"] = _coalesce(df, "adjde", "def_efficiency_sref")

    # Estimate defensive efficiency if missing: DRtg ≈ opp_ppg / pace * 100
    # Or more directly: ORtg - SRS ≈ DRtg (since SRS ≈ point_diff scaled)
    if df["adjde"].isna().all():
        opp_ppg_est = _coalesce(df, "opp_pts_sref")
        games = _coalesce(df, "games_sref")
        pace_est = _coalesce(df, "pace")
        if opp_ppg_est.notna().any() and games.notna().any() and pace_est.notna().any():
            # DRtg = (opp_pts_per_game / pace) * 100
            # This is approximate but captures relative defensive strength
            opp_ppg_calc = opp_ppg_est / games
            df["adjde"] = (opp_ppg_calc / pace_est) * 100
        elif df["adjoe"].notna().any() and _coalesce(df, "srs").notna().any():
            # Fallback: ORtg - SRS ≈ DRtg
            df["adjde"] = df["adjoe"] - _coalesce(df, "srs")

    df["adjem"] = df["adjoe"] - df["adjde"]

    # Barthag: if not from Torvik, estimate from efficiency margin
    if "barthag" not in df.columns or df["barthag"].isna().all():
        # Rough approximation: barthag ≈ sigmoid(adjem / 10)
        df["barthag"] = 1 / (1 + np.exp(-df["adjem"] / 10))

    # Four factors
    df["efg_o"] = _coalesce(df, "efg_o_torvik", "efg_o_sref")
    df["efg_d"] = _coalesce(df, "efg_d_torvik", "efg_d_sref")
    df["tov_o"] = _coalesce(df, "tov_o_torvik", "tov_o_sref")
    df["tov_d"] = _coalesce(df, "tov_d_torvik", "tov_d_sref")
    df["orb_o"] = _coalesce(df, "orb_o_torvik", "orb_o_sref")
    df["orb_d"] = _coalesce(df, "orb_d_torvik", "orb_d_sref")
    df["ftr_o"] = _coalesce(df, "ftr_o_torvik", "ftr_o_sref", "ftr_o_sref2")
    df["ftr_d"] = _coalesce(df, "ftr_d_torvik", "ftr_d_sref")

    # Other stats
    df["pace"] = _coalesce(df, "adj_tempo", "pace")
    df["srs"] = _coalesce(df, "srs")
    df["sos"] = _coalesce(df, "sos")

    # PPG: prefer SportsRef (total pts / games), fall back to ESPN
    if "pts_sref" in df.columns and "games_sref" in df.columns:
        games = df["games_sref"].replace(0, np.nan)
        ppg_sref = df["pts_sref"] / games
        opp_ppg_sref = df["opp_pts_sref"] / games if "opp_pts_sref" in df.columns else pd.Series(np.nan, index=df.index)
    else:
        ppg_sref = pd.Series(np.nan, index=df.index)
        opp_ppg_sref = pd.Series(np.nan, index=df.index)

    df["ppg"] = ppg_sref.fillna(_coalesce(df, "ppg_espn"))
    df["opp_ppg"] = opp_ppg_sref
    df["point_diff"] = df["ppg"] - df["opp_ppg"]
    # Fall back to efficiency margin for point_diff if opp_ppg missing
    df["point_diff"] = df["point_diff"].fillna(df["adjem"])

    # Win percentage
    df["win_pct"] = _coalesce(df, "win_pct_sref")
    if df["win_pct"].isna().all() and "wins_sref" in df.columns:
        total = df["wins_sref"] + df["losses_sref"]
        df["win_pct"] = df["wins_sref"] / total.replace(0, np.nan)

    # Three-point rate
    df["three_rate_o"] = _coalesce(df, "three_rate_o_torvik", "three_rate_o_sref")
    df["ts_pct"] = _coalesce(df, "ts_pct_sref")

    # Efficiency metrics from SportsRef if Torvik unavailable
    df["off_efficiency"] = df["adjoe"]
    df["def_efficiency"] = df["adjde"]
    df["efficiency_margin"] = df["adjem"]

    return df


def _coalesce(df: pd.DataFrame, *cols: str) -> pd.Series:
    """Return the first non-null value across columns."""
    result = pd.Series(np.nan, index=df.index)
    for col in cols:
        if col in df.columns:
            result = result.fillna(df[col])
    return result


def extract_features(team_df: pd.DataFrame) -> pd.DataFrame:
    """Extract the standardized feature set from merged team data.

    Returns a DataFrame with only the columns needed for prediction.
    """
    features = pd.DataFrame(index=team_df.index)

    for col in CORE_FEATURES:
        if col in team_df.columns:
            features[col] = team_df[col]
        else:
            features[col] = np.nan

    # Keep identifiers
    features["name"] = team_df["name"]
    features["name_norm"] = team_df["name_norm"]
    if "region" in team_df.columns:
        features["region"] = team_df["region"]

    return features
