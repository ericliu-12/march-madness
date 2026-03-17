"""ESPN API scraper for college basketball team statistics.

Uses the public ESPN site API (no auth required) to fetch team-level
season statistics for all Division I teams.
"""

import json
import os
import time

import pandas as pd
import requests

from config import CACHE_DIR, ESPN_BASE, REQUEST_DELAY, USER_AGENT, YEAR


def get_all_teams() -> list[dict]:
    """Fetch all D-I basketball team IDs and names from ESPN."""
    teams = []
    page = 1
    while True:
        url = f"{ESPN_BASE}/teams?limit=100&page={page}&groups=50"
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        sports = data.get("sports", [])
        if not sports:
            break

        for sport in sports:
            for league in sport.get("leagues", []):
                for team in league.get("teams", []):
                    t = team.get("team", team)
                    teams.append({
                        "espn_id": t["id"],
                        "name": t.get("displayName", t.get("name", "")),
                        "abbreviation": t.get("abbreviation", ""),
                        "location": t.get("location", ""),
                    })

        # ESPN paginates; stop when we stop getting new teams
        if len(sports[0].get("leagues", [{}])[0].get("teams", [])) < 100:
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    return teams


def get_team_stats(team_id: str) -> dict | None:
    """Fetch season statistics for a single team."""
    url = f"{ESPN_BASE}/teams/{team_id}/statistics"
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError):
        return None

    stats = {}
    # Parse the splits/categories structure
    for split in data.get("results", data.get("splits", {}).get("categories", [])):
        if isinstance(split, dict):
            cat_name = split.get("displayName", split.get("name", ""))
            for stat in split.get("stats", []):
                key = stat.get("abbreviation", stat.get("name", ""))
                val = stat.get("value", stat.get("displayValue", ""))
                if key:
                    stats[f"{cat_name}_{key}" if cat_name else key] = _parse_num(val)

    # Also try the categories structure directly
    if not stats:
        categories = data.get("splits", {}).get("categories", [])
        if not categories:
            categories = data.get("results", {}).get("stats", {}).get("categories", [])
        for cat in categories:
            cat_name = cat.get("displayName", cat.get("name", ""))
            for stat in cat.get("stats", []):
                key = stat.get("abbreviation", stat.get("name", ""))
                val = stat.get("value", stat.get("displayValue", ""))
                if key:
                    full_key = f"{cat_name}_{key}" if cat_name else key
                    stats[full_key] = _parse_num(val)

    return stats if stats else None


def _parse_num(val):
    """Try to parse a value as a number."""
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return val
    return val


def scrape_espn(force: bool = False) -> pd.DataFrame:
    """Scrape all team stats from ESPN. Uses cache if available.

    Returns a DataFrame with one row per team and columns for each stat.
    """
    cache_path = os.path.join(CACHE_DIR, f"espn_stats_{YEAR}.csv")

    if not force and os.path.exists(cache_path):
        print(f"Loading ESPN data from cache: {cache_path}")
        return pd.read_csv(cache_path)

    print("Fetching team list from ESPN...")
    teams = get_all_teams()
    print(f"Found {len(teams)} teams. Fetching stats...")

    rows = []
    for i, team in enumerate(teams):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(teams)}")

        stats = get_team_stats(team["espn_id"])
        if stats:
            row = {
                "espn_id": team["espn_id"],
                "name": team["name"],
                "abbreviation": team["abbreviation"],
            }
            row.update(stats)
            rows.append(row)

        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(rows)
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Saved {len(df)} teams to {cache_path}")
    return df


if __name__ == "__main__":
    df = scrape_espn(force=True)
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample:\n{df.head()}")
