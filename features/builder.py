"""Feature engineering pipeline orchestrator.

Coordinates data collection, team feature extraction, and matchup
feature computation into a unified pipeline.
"""

import pandas as pd

from data.scraper_espn import scrape_espn
from data.scraper_sportsref import scrape_sportsref
from data.scraper_torvik import scrape_torvik
from data.scraper_ncaa import get_bracket_from_fallback, scrape_bracket, get_tournament_teams
from data.kaggle_loader import load_kaggle_data, build_historical_training_data
from features.team_features import merge_team_data, extract_features
from features.matchup_features import compute_matchup_features


def collect_data(force_scrape: bool = False) -> dict:
    """Collect all data from external sources.

    Returns a dict with all DataFrames needed for prediction.
    """
    print("=" * 60)
    print("COLLECTING DATA")
    print("=" * 60)

    # Get tournament bracket and teams
    print("\n--- Tournament Bracket ---")
    bracket_df = scrape_bracket(force=force_scrape)
    tournament_teams = []
    if bracket_df is not None and not bracket_df.empty:
        tournament_teams = get_tournament_teams(bracket_df)
        print(f"Got {len(tournament_teams)} teams from NCAA API")

    if len(tournament_teams) < 32:
        print("Insufficient bracket data from API, using fallback bracket...")
        tournament_teams = get_bracket_from_fallback()
        print(f"Using fallback bracket: {len(tournament_teams)} teams")

    # Scrape current season stats
    print("\n--- ESPN Stats ---")
    espn_df = scrape_espn(force=force_scrape)

    print("\n--- Sports-Reference Stats ---")
    sportsref_df = scrape_sportsref(force=force_scrape)

    print("\n--- Bart Torvik Stats ---")
    torvik_df = scrape_torvik(force=force_scrape)

    # Load historical data
    print("\n--- Historical Data ---")
    kaggle_data = load_kaggle_data()

    return {
        "tournament_teams": tournament_teams,
        "bracket_df": bracket_df,
        "espn_df": espn_df,
        "sportsref_df": sportsref_df,
        "torvik_df": torvik_df,
        "kaggle_data": kaggle_data,
    }


def build_team_profiles(data: dict) -> pd.DataFrame:
    """Build unified team feature profiles from collected data.

    Returns a DataFrame with one row per tournament team and
    standardized feature columns.
    """
    print("\n--- Building Team Profiles ---")

    team_df = merge_team_data(
        espn_df=data["espn_df"],
        sportsref_df=data["sportsref_df"],
        torvik_df=data["torvik_df"],
        tournament_teams=data["tournament_teams"],
    )

    features = extract_features(team_df)

    # Report coverage
    n_teams = len(features)
    n_complete = features.dropna(subset=["adjoe", "adjde"]).shape[0]
    print(f"  {n_teams} tournament teams")
    print(f"  {n_complete} with efficiency data")

    # Fill remaining NaN with column median (for teams with missing data)
    numeric_cols = features.select_dtypes(include="number").columns
    for col in numeric_cols:
        median_val = features[col].median()
        features[col] = features[col].fillna(median_val)

    return features


def build_training_data(data: dict) -> tuple[pd.DataFrame, pd.Series] | None:
    """Build historical training data for model fitting.

    Returns (X, y) tuple or None if insufficient data.
    """
    print("\n--- Building Training Data ---")
    return build_historical_training_data(data["kaggle_data"])


def compute_game_features(
    team_a: pd.Series,
    team_b: pd.Series,
) -> dict:
    """Compute features for a single game matchup."""
    return compute_matchup_features(team_a, team_b)
