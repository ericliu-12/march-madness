"""NCAA API scraper for bracket structure and seeds.

Uses the free NCAA API (ncaa-api.henrygd.me) to get tournament bracket
information including team names, seeds, and regional assignments.
"""

import json
import os
from datetime import datetime, timedelta

import pandas as pd
import requests

from config import CACHE_DIR, NCAA_API_BASE, USER_AGENT, YEAR


def fetch_bracket_data() -> list[dict]:
    """Fetch tournament bracket data from the NCAA API.

    Checks scoreboard data for tournament games to extract
    bracket structure, seeds, and regional assignments.
    """
    headers = {"User-Agent": USER_AGENT}
    all_games = []

    # Check tournament dates (mid-March through early April)
    start_date = datetime(YEAR, 3, 17)
    end_date = datetime(YEAR, 4, 8)

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y/%m/%d")
        url = f"{NCAA_API_BASE}/scoreboard/basketball-men/d1/{date_str}"

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                games = data.get("games", [])
                for game in games:
                    g = game.get("game", game)
                    # Look for tournament games (they have bracketId or specific context)
                    if _is_tournament_game(g):
                        parsed = _parse_game(g, current)
                        if parsed:
                            all_games.append(parsed)
        except (requests.RequestException, json.JSONDecodeError):
            pass

        current += timedelta(days=1)

    return all_games


def _is_tournament_game(game: dict) -> bool:
    """Check if a game is an NCAA tournament game."""
    # Check for bracket indicators
    if game.get("bracketId") or game.get("bracketRound"):
        return True

    # Check game context/title
    title = game.get("title", "").lower()
    contest = game.get("contestName", "").lower()
    for indicator in ["ncaa", "tournament", "march madness", "first round",
                      "second round", "sweet 16", "elite 8", "final four",
                      "championship"]:
        if indicator in title or indicator in contest:
            return True

    return False


def _parse_game(game: dict, date: datetime) -> dict | None:
    """Parse a tournament game into structured data."""
    home = game.get("home", {})
    away = game.get("away", {})

    if not home.get("names") or not away.get("names"):
        return None

    return {
        "date": date.strftime("%Y-%m-%d"),
        "home_team": home["names"].get("full", home["names"].get("short", "")),
        "home_seed": _extract_seed(home),
        "away_team": away["names"].get("full", away["names"].get("short", "")),
        "away_seed": _extract_seed(away),
        "bracket_id": game.get("bracketId", ""),
        "bracket_round": game.get("bracketRound", ""),
        "game_id": game.get("gameID", game.get("id", "")),
        "status": game.get("gameState", game.get("currentPeriod", "")),
        "home_score": home.get("score", ""),
        "away_score": away.get("score", ""),
    }


def _extract_seed(team: dict) -> int | None:
    """Extract tournament seed from team data."""
    seed = team.get("seed")
    if seed:
        try:
            return int(seed)
        except (ValueError, TypeError):
            pass

    # Try to find seed in team description or rank
    rank = team.get("rank", team.get("ranking"))
    if rank:
        try:
            return int(rank)
        except (ValueError, TypeError):
            pass

    return None


def scrape_bracket(force: bool = False) -> pd.DataFrame | None:
    """Scrape NCAA tournament bracket data."""
    cache_path = os.path.join(CACHE_DIR, f"ncaa_bracket_{YEAR}.csv")

    if not force and os.path.exists(cache_path):
        print(f"Loading bracket data from cache: {cache_path}")
        return pd.read_csv(cache_path)

    print("Fetching NCAA tournament bracket data...")
    games = fetch_bracket_data()

    if games:
        df = pd.DataFrame(games)
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Found {len(df)} tournament games")
        return df
    else:
        print("Warning: Could not fetch bracket data from NCAA API")
        return None


def get_tournament_teams(bracket_df: pd.DataFrame | None = None) -> list[dict]:
    """Extract unique tournament teams with seeds from bracket data."""
    if bracket_df is None:
        bracket_df = scrape_bracket()

    if bracket_df is None or bracket_df.empty:
        return []

    teams = {}

    for _, row in bracket_df.iterrows():
        for prefix in ["home", "away"]:
            name = row.get(f"{prefix}_team")
            seed = row.get(f"{prefix}_seed")
            if name and pd.notna(name):
                if name not in teams or (seed and pd.notna(seed)):
                    teams[name] = {
                        "name": name,
                        "seed": int(seed) if pd.notna(seed) else None,
                    }

    return list(teams.values())


# Fallback: manually define the 2026 bracket if the API isn't serving data yet
BRACKET_2026_FALLBACK = {
    "East": [
        (1, "Duke"), (16, "Mount St. Mary's"),
        (8, "Memphis"), (9, "Florida State"),
        (5, "Michigan State"), (12, "UC San Diego"),
        (4, "Arizona State"), (13, "Yale"),
        (6, "Missouri"), (11, "Oklahoma"),
        (3, "Wisconsin"), (14, "Lipscomb"),
        (7, "Kansas"), (10, "Arkansas"),
        (2, "Alabama"), (15, "Robert Morris"),
    ],
    "South": [
        (1, "Arizona"), (16, "Norfolk State"),
        (8, "Louisville"), (9, "Georgia"),
        (5, "Oregon"), (12, "Liberty"),
        (4, "Purdue"), (13, "High Point"),
        (6, "Illinois"), (11, "Drake"),
        (3, "Texas Tech"), (14, "UNC Wilmington"),
        (7, "Marquette"), (10, "New Mexico"),
        (2, "Iowa State"), (15, "Omaha"),
    ],
    "Midwest": [
        (1, "Michigan"), (16, "Texas Southern"),
        (8, "San Diego State"), (9, "Baylor"),
        (5, "Clemson"), (12, "McNeese State"),
        (4, "Maryland"), (13, "Grand Canyon"),
        (6, "BYU"), (11, "North Carolina"),
        (3, "Texas A&M"), (14, "Troy"),
        (7, "Dayton"), (10, "Mississippi State"),
        (2, "St. John's"), (15, "Omaha"),
    ],
    "West": [
        (1, "Florida"), (16, "SIU Edwardsville"),
        (8, "UConn"), (9, "Gonzaga"),
        (5, "Houston"), (12, "Western Kentucky"),
        (4, "Tennessee"), (13, "Vermont"),
        (6, "UCLA"), (11, "Colorado State"),
        (7, "Saint Mary's"), (10, "Nevada"),
        (3, "Kentucky"), (14, "Troy"),
        (2, "Auburn"), (15, "Wofford"),
    ],
}


def get_bracket_from_fallback() -> list[dict]:
    """Build tournament team list from hardcoded fallback bracket."""
    teams = []
    for region, matchups in BRACKET_2026_FALLBACK.items():
        for seed, name in matchups:
            teams.append({
                "name": name,
                "seed": seed,
                "region": region,
                "is_play_in": False,
            })
    return teams


if __name__ == "__main__":
    # Try API first, fall back to hardcoded bracket
    bracket = scrape_bracket(force=True)
    if bracket is not None and not bracket.empty:
        teams = get_tournament_teams(bracket)
        print(f"Found {len(teams)} tournament teams from API")
    else:
        teams = get_bracket_from_fallback()
        print(f"Using fallback bracket: {len(teams)} teams")

    for t in sorted(teams, key=lambda x: (x.get("region", ""), x["seed"])):
        print(f"  {t.get('region', '?'):10s} #{t['seed']:2d} {t['name']}")
