"""Bart Torvik scraper for T-Rank advanced statistics.

Torvik provides the highest-value predictive stats: AdjOE, AdjDE, Barthag,
adjusted tempo, and four-factor breakdowns. Tries JSON endpoint first,
falls back gracefully if blocked by Cloudflare.
"""

import json
import os

import pandas as pd
import requests

from config import CACHE_DIR, USER_AGENT, YEAR


# Column names for the Torvik T-Rank table
TORVIK_COLUMNS = [
    "rank", "team", "conf", "record", "adjoe", "adjoe_rank",
    "adjde", "adjde_rank", "barthag", "barthag_rank",
    "efg_o", "efg_d", "tov_o", "tov_d", "orb_o", "orb_d",
    "ftr_o", "ftr_d", "two_pt_o", "two_pt_d", "three_pt_o", "three_pt_d",
    "three_rate_o", "three_rate_d", "adj_tempo", "adj_tempo_rank",
    "fun_rank", "fun_rank_rk",
]


def _try_json_endpoint() -> pd.DataFrame | None:
    """Try Torvik's JSON data endpoint directly."""
    url = f"https://barttorvik.com/trank.php?year={YEAR}&json=1"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/html",
        "Referer": "https://barttorvik.com/",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return None

        data = resp.json()
        if not data:
            return None

        # The JSON endpoint returns a list of lists
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):
                # Each row is a list of values
                ncols = len(data[0])
                cols = TORVIK_COLUMNS[:ncols] if ncols <= len(TORVIK_COLUMNS) else [
                    f"col_{i}" for i in range(ncols)
                ]
                df = pd.DataFrame(data, columns=cols)
            elif isinstance(data[0], dict):
                df = pd.DataFrame(data)
            else:
                return None

            return df

    except (requests.RequestException, json.JSONDecodeError, ValueError):
        return None

    return None


def _try_html_scrape() -> pd.DataFrame | None:
    """Try scraping the Torvik HTML page directly."""
    url = f"https://barttorvik.com/trank.php?year={YEAR}&conlimit=All"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html",
        "Referer": "https://barttorvik.com/",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return None

        # Try to parse with pandas read_html
        tables = pd.read_html(resp.text)
        if tables:
            df = tables[0]
            # Clean up column names
            if len(df.columns) >= 5:
                return df

    except Exception:
        pass

    return None


def scrape_torvik(force: bool = False) -> pd.DataFrame | None:
    """Scrape Bart Torvik T-Rank data.

    Tries JSON endpoint first, then HTML scraping. Returns None
    if both fail (the model degrades gracefully without Torvik data).
    """
    cache_path = os.path.join(CACHE_DIR, f"torvik_trank_{YEAR}.csv")

    if not force and os.path.exists(cache_path):
        print(f"Loading Torvik data from cache: {cache_path}")
        return pd.read_csv(cache_path)

    print("Fetching T-Rank data from Bart Torvik...")

    # Strategy 1: JSON endpoint
    df = _try_json_endpoint()
    if df is not None and not df.empty:
        print(f"  Got {len(df)} teams from JSON endpoint")
        _save_torvik(df, cache_path)
        return df

    print("  JSON endpoint unavailable, trying HTML scrape...")

    # Strategy 2: HTML scrape
    df = _try_html_scrape()
    if df is not None and not df.empty:
        print(f"  Got {len(df)} teams from HTML scrape")
        _save_torvik(df, cache_path)
        return df

    print("  Warning: Could not fetch Torvik data (Cloudflare may be blocking).")
    print("  The model will use ESPN + Sports-Reference data only.")
    return None


def _save_torvik(df: pd.DataFrame, path: str):
    """Clean and save Torvik data."""
    # Ensure numeric columns
    numeric_cols = [
        "adjoe", "adjde", "barthag", "adj_tempo",
        "efg_o", "efg_d", "tov_o", "tov_d",
        "orb_o", "orb_d", "ftr_o", "ftr_d",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved to {path}")


if __name__ == "__main__":
    df = scrape_torvik(force=True)
    if df is not None:
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nTop 10:\n{df.head(10)}")
    else:
        print("Could not fetch Torvik data")
