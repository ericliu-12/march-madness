"""Canonical team name mapping across data sources.

Maps between ESPN, Sports-Reference, Bart Torvik, and display names.
Uses fuzzy matching as fallback for unresolved names.
"""

import difflib
from functools import lru_cache


# Canonical mapping: display_name -> known aliases
# This covers the most common discrepancies across sources
TEAM_ALIASES = {
    "UConn": ["Connecticut", "Connecticut Huskies", "CONN"],
    "NC State": ["North Carolina State", "North Carolina St.", "N.C. State", "NC St."],
    "Miami (FL)": ["Miami", "Miami Florida", "Miami Hurricanes"],
    "Miami (OH)": ["Miami Ohio", "Miami (Ohio)", "Miami RedHawks"],
    "St. John's": ["Saint John's", "St John's", "St. John's (NY)"],
    "St. Mary's": ["Saint Mary's", "St Mary's", "Saint Mary's (CA)"],
    "St. Peter's": ["Saint Peter's", "St Peter's"],
    "St. Bonaventure": ["Saint Bonaventure", "St Bonaventure"],
    "St. Thomas": ["Saint Thomas", "St Thomas"],
    "USC": ["Southern California", "Southern Cal"],
    "UCF": ["Central Florida"],
    "SMU": ["Southern Methodist"],
    "UNLV": ["Nevada-Las Vegas"],
    "VCU": ["Virginia Commonwealth"],
    "UAB": ["Alabama-Birmingham"],
    "UTEP": ["Texas-El Paso"],
    "UNC": ["North Carolina", "N. Carolina"],
    "Ole Miss": ["Mississippi"],
    "Pitt": ["Pittsburgh"],
    "UMass": ["Massachusetts"],
    "LSU": ["Louisiana State"],
    "BYU": ["Brigham Young"],
    "TCU": ["Texas Christian"],
    "FDU": ["Fairleigh Dickinson"],
    "LIU": ["Long Island University"],
    "ETSU": ["East Tennessee State", "East Tennessee St."],
    "MTSU": ["Middle Tennessee", "Middle Tennessee State"],
    "SIU Edwardsville": ["SIUE", "SIU-Edwardsville"],
    "Loyola Chicago": ["Loyola (IL)", "Loyola-Chicago"],
    "Texas A&M": ["Texas A&M Aggies"],
    "Penn St.": ["Penn State", "Pennsylvania State"],
    "Iowa St.": ["Iowa State"],
    "Michigan St.": ["Michigan State"],
    "Ohio St.": ["Ohio State"],
    "Oklahoma St.": ["Oklahoma State"],
    "Oregon St.": ["Oregon State"],
    "Kansas St.": ["Kansas State"],
    "Boise St.": ["Boise State"],
    "San Diego St.": ["San Diego State", "SDSU"],
    "Colorado St.": ["Colorado State"],
    "Fresno St.": ["Fresno State"],
    "Utah St.": ["Utah State"],
    "Wichita St.": ["Wichita State"],
    "Norfolk St.": ["Norfolk State"],
    "Grambling St.": ["Grambling State", "Grambling"],
    "McNeese St.": ["McNeese State", "McNeese"],
    "Morehead St.": ["Morehead State"],
    "Murray St.": ["Murray State"],
    "Weber St.": ["Weber State"],
    "Wright St.": ["Wright State"],
    "Kennesaw St.": ["Kennesaw State"],
}

# Build reverse lookup: alias -> canonical name
_ALIAS_TO_CANONICAL = {}
for canonical, aliases in TEAM_ALIASES.items():
    _ALIAS_TO_CANONICAL[canonical.lower()] = canonical
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical


def normalize_team_name(name: str) -> str:
    """Normalize a team name to its canonical form.

    Tries exact match first, then fuzzy matching.
    """
    if not name:
        return name

    # Clean up common suffixes
    cleaned = name.strip()
    for suffix in [" Wildcats", " Blue Devils", " Wolverines", " Gators",
                   " Tigers", " Bulldogs", " Bears", " Eagles", " Hawks",
                   " Cardinals", " Cavaliers", " Hoosiers", " Jayhawks",
                   " Tar Heels", " Aggies", " Longhorns", " Huskies",
                   " Cougars", " Spartans", " Buckeyes", " Sooners",
                   " Volunteers", " Crimson Tide", " Hurricanes", " Seminoles",
                   " Yellow Jackets", " Demon Deacons", " Mountaineers",
                   " Red Raiders", " Cyclones", " Horned Frogs", " Panthers",
                   " Bruins", " Trojans", " Ducks", " Beavers", " Sun Devils",
                   " Buffaloes", " Utes"]:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
            break

    # Try exact match
    lower = cleaned.lower()
    if lower in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[lower]

    # Try fuzzy match
    return _fuzzy_match(cleaned)


@lru_cache(maxsize=512)
def _fuzzy_match(name: str) -> str:
    """Find the closest canonical name using fuzzy matching."""
    all_names = list(_ALIAS_TO_CANONICAL.keys())
    matches = difflib.get_close_matches(name.lower(), all_names, n=1, cutoff=0.7)
    if matches:
        return _ALIAS_TO_CANONICAL[matches[0]]
    # If no match, return as-is (many teams won't be in our alias list)
    return name


def create_team_lookup(teams_list: list[dict]) -> dict:
    """Create a lookup dictionary from a list of team dicts.

    Each team dict should have at least a 'name' key.
    Returns: {normalized_name: team_dict}
    """
    lookup = {}
    for team in teams_list:
        norm = normalize_team_name(team["name"])
        lookup[norm] = team
    return lookup
