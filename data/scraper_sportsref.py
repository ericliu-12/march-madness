"""Sports-Reference scraper for advanced college basketball statistics.

Scrapes the advanced school stats page which contains SRS, SOS, pace,
offensive/defensive ratings, and four-factor stats for all D-I teams.
"""

import os
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import (
    CACHE_DIR,
    SPORTSREF_ADVANCED_URL,
    SPORTSREF_BASIC_URL,
    REQUEST_DELAY,
    USER_AGENT,
    YEAR,
)


def _fetch_sportsref_table(url: str, table_id: str) -> pd.DataFrame | None:
    """Fetch and parse a Sports-Reference stats table."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.sports-reference.com/",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Sports-Reference sometimes wraps tables in comments
    table = soup.find("table", {"id": table_id})

    if table is None:
        # Try finding in comments
        from bs4 import Comment
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            if table_id in str(comment):
                comment_soup = BeautifulSoup(str(comment), "lxml")
                table = comment_soup.find("table", {"id": table_id})
                if table:
                    break

    if table is None:
        print(f"Could not find table '{table_id}' at {url}")
        return None

    # Parse header
    thead = table.find("thead")
    headers_row = thead.find_all("tr")[-1] if thead else None
    if not headers_row:
        return None

    columns = []
    for th in headers_row.find_all("th"):
        stat = th.get("data-stat", th.get_text(strip=True))
        columns.append(stat)

    # Parse body
    tbody = table.find("tbody")
    rows = []
    for tr in tbody.find_all("tr"):
        if "thead" in tr.get("class", []):
            continue  # Skip sub-header rows

        row = {}
        for td in tr.find_all(["td", "th"]):
            stat = td.get("data-stat", "")
            val = td.get_text(strip=True)
            if stat:
                row[stat] = val

        if row and row.get("school_name"):
            rows.append(row)

    df = pd.DataFrame(rows)

    # Convert numeric columns
    for col in df.columns:
        if col not in ("school_name", "school_id", "conf_abbr"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def scrape_advanced_stats(force: bool = False) -> pd.DataFrame | None:
    """Scrape advanced school stats from Sports-Reference."""
    cache_path = os.path.join(CACHE_DIR, f"sportsref_advanced_{YEAR}.csv")

    if not force and os.path.exists(cache_path):
        print(f"Loading Sports-Ref advanced data from cache: {cache_path}")
        return pd.read_csv(cache_path)

    print("Fetching advanced stats from Sports-Reference...")
    df = _fetch_sportsref_table(SPORTSREF_ADVANCED_URL, "adv_school_stats")

    if df is not None and not df.empty:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Saved {len(df)} teams to {cache_path}")
    else:
        print("Warning: Could not fetch Sports-Reference advanced stats")

    return df


def scrape_basic_stats(force: bool = False) -> pd.DataFrame | None:
    """Scrape basic school stats from Sports-Reference."""
    cache_path = os.path.join(CACHE_DIR, f"sportsref_basic_{YEAR}.csv")

    if not force and os.path.exists(cache_path):
        print(f"Loading Sports-Ref basic data from cache: {cache_path}")
        return pd.read_csv(cache_path)

    print("Fetching basic stats from Sports-Reference...")
    time.sleep(REQUEST_DELAY * 4)  # Sports-Ref rate limits aggressively

    df = _fetch_sportsref_table(SPORTSREF_BASIC_URL, "basic_school_stats")

    if df is not None and not df.empty:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Saved {len(df)} teams to {cache_path}")
    else:
        print("Warning: Could not fetch Sports-Reference basic stats")

    return df


def scrape_sportsref(force: bool = False) -> pd.DataFrame | None:
    """Scrape and merge both basic and advanced stats from Sports-Reference."""
    adv = scrape_advanced_stats(force=force)
    basic = scrape_basic_stats(force=force)

    if adv is None:
        return basic
    if basic is None:
        return adv

    # Merge on school name, keeping all advanced stats
    # Suffix basic columns to avoid collisions
    merged = adv.merge(
        basic,
        on="school_name",
        how="left",
        suffixes=("", "_basic"),
    )

    return merged


if __name__ == "__main__":
    df = scrape_sportsref(force=True)
    if df is not None:
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nSample:\n{df.head()}")
    else:
        print("Failed to scrape Sports-Reference data")
