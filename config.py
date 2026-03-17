"""Global configuration for the March Madness prediction model."""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
HISTORICAL_DIR = os.path.join(PROJECT_ROOT, "historical_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_results")

# Ensure directories exist
for d in [CACHE_DIR, HISTORICAL_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# Current season
YEAR = 2026

# ESPN API
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

# Sports Reference
SPORTSREF_ADVANCED_URL = (
    f"https://www.sports-reference.com/cbb/seasons/men/{YEAR}-advanced-school-stats.html"
)
SPORTSREF_BASIC_URL = (
    f"https://www.sports-reference.com/cbb/seasons/men/{YEAR}-school-stats.html"
)

# Bart Torvik
TORVIK_URL = f"https://barttorvik.com/trank.php?year={YEAR}&conlimit=All"

# NCAA API
NCAA_API_BASE = "https://ncaa-api.henrygd.me"

# Model parameters
MONTE_CARLO_SIMULATIONS = 10000
PROBABILITY_FLOOR = 0.02
PROBABILITY_CEILING = 0.98
RANDOM_SEED = 42

# Scraping
REQUEST_DELAY = 0.5  # seconds between requests
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Tournament structure
REGIONS = ["East", "South", "Midwest", "West"]
ROUNDS = {
    1: "First Round",
    2: "Second Round",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}
SEED_MATCHUPS = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]
