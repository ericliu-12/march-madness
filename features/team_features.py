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

    # Map ESPN stat names to our standard names
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "ppg" in cl or (col.endswith("_PTS") and "opp" not in cl.lower()):
            col_map[col] = "ppg_espn"
        elif "fg%" in cl or "fgpct" in cl.lower():
            col_map[col] = "fg_pct_espn"
        elif "3p%" in cl or "3ptpct" in cl.lower():
            col_map[col] = "three_pct_espn"
        elif "ft%" in cl or "ftpct" in cl.lower():
            col_map[col] = "ft_pct_espn"
        elif "rpg" in cl or "reb" in cl.lower():
            col_map[col] = "rpg_espn"
        elif "apg" in cl or "ast" in cl.lower():
            col_map[col] = "apg_espn"
        elif "spg" in cl or "stl" in cl.lower():
            col_map[col] = "spg_espn"
        elif "bpg" in cl or "blk" in cl.lower():
            col_map[col] = "bpg_espn"
        elif "topg" in cl or "turnover" in cl.lower():
            col_map[col] = "topg_espn"

    for old_col, new_col in col_map.items():
        if old_col in df.columns and new_col not in out.columns:
            out[new_col] = pd.to_numeric(df[old_col], errors="coerce")

    return out.drop_duplicates(subset="name_norm")


def _prepare_sportsref(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Sports-Reference data for merging."""
    out = pd.DataFrame()

    name_col = "school_name" if "school_name" in df.columns else df.columns[0]
    out["name_norm"] = df[name_col].apply(normalize_team_name)

    col_map = {
        "srs": "srs",
        "sos": "sos",
        "pace": "pace",
        "o_rtg": "off_efficiency_sref",
        "d_rtg": "def_efficiency_sref",
        "ftr": "ftr_o_sref",
        "x3p_ar": "three_rate_o_sref",
        "ts_pct": "ts_pct_sref",
        "trb_pct": "trb_pct",
        "ast_pct": "ast_pct",
        "stl_pct": "stl_pct",
        "blk_pct": "blk_pct",
        "e_fg_pct": "efg_o_sref",
        "tov_pct": "tov_o_sref",
        "orb_pct": "orb_o_sref",
        "ft_fga": "ftr_o_sref2",
        "e_fg_pct_2": "efg_d_sref",
        "tov_pct_2": "tov_d_sref",
        "orb_pct_2": "orb_d_sref",
        "ft_fga_2": "ftr_d_sref",
        "w": "wins_sref",
        "l": "losses_sref",
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

    # PPG from ESPN
    df["ppg"] = _coalesce(df, "ppg_espn")
    df["opp_ppg"] = df.get("opp_ppg", pd.Series(dtype=float))
    df["point_diff"] = df["ppg"] - df["opp_ppg"] if "opp_ppg" in df.columns else df["adjem"]

    # Win percentage
    if "wins_sref" in df.columns and "losses_sref" in df.columns:
        total = df["wins_sref"] + df["losses_sref"]
        df["win_pct"] = df["wins_sref"] / total.replace(0, np.nan)
    elif "win_pct" not in df.columns:
        df["win_pct"] = np.nan

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
