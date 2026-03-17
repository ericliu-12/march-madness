"""Tournament bracket data structures.

Represents the NCAA tournament bracket with teams, games, regions,
and round-by-round progression.
"""

from copy import deepcopy
from dataclasses import dataclass, field

import pandas as pd

from config import REGIONS, SEED_MATCHUPS


@dataclass
class Team:
    """A tournament team with stats."""
    name: str
    seed: int
    region: str
    stats: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    is_play_in: bool = False

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Team):
            return self.name == other.name
        return False

    def __repr__(self):
        return f"({self.seed}) {self.name}"


@dataclass
class Game:
    """A single tournament game."""
    team_a: Team
    team_b: Team
    round_num: int  # 1=R64, 2=R32, 3=S16, 4=E8, 5=F4, 6=Championship
    region: str
    winner: Team | None = None
    win_probability: float | None = None  # P(team_a wins)

    @property
    def higher_seed(self) -> Team:
        """The favored team (lower seed number)."""
        return self.team_a if self.team_a.seed <= self.team_b.seed else self.team_b

    @property
    def lower_seed(self) -> Team:
        """The underdog (higher seed number)."""
        return self.team_b if self.team_a.seed <= self.team_b.seed else self.team_a

    def __repr__(self):
        w = f" -> {self.winner}" if self.winner else ""
        return f"R{self.round_num}: {self.team_a} vs {self.team_b}{w}"


class Bracket:
    """Full tournament bracket.

    Manages teams, games, and round-by-round advancement.
    """

    def __init__(self):
        self.regions: dict[str, list[Team]] = {r: [] for r in REGIONS}
        self.games: list[Game] = []
        self.rounds: dict[int, list[Game]] = {}

    def add_team(self, team: Team):
        """Add a team to its region."""
        if team.region in self.regions:
            self.regions[team.region].append(team)

    def build_first_round(self):
        """Generate first-round matchups based on standard seeding."""
        self.rounds[1] = []

        for region_name, teams in self.regions.items():
            # Sort by seed
            teams_by_seed = {t.seed: t for t in teams}

            for seed_a, seed_b in SEED_MATCHUPS:
                team_a = teams_by_seed.get(seed_a)
                team_b = teams_by_seed.get(seed_b)

                if team_a and team_b:
                    game = Game(
                        team_a=team_a,
                        team_b=team_b,
                        round_num=1,
                        region=region_name,
                    )
                    self.rounds[1].append(game)
                    self.games.append(game)

    def advance_winners(self, round_num: int) -> list[Game]:
        """Create next round matchups from current round winners.

        Returns the list of next-round games.
        """
        current_games = self.rounds.get(round_num, [])
        next_round = round_num + 1
        next_games = []

        if next_round <= 4:
            # Regional rounds: pair winners within each region
            for region in REGIONS:
                region_games = [g for g in current_games if g.region == region]
                winners = [g.winner for g in region_games if g.winner]

                for i in range(0, len(winners) - 1, 2):
                    game = Game(
                        team_a=winners[i],
                        team_b=winners[i + 1],
                        round_num=next_round,
                        region=region,
                    )
                    next_games.append(game)
                    self.games.append(game)

        elif next_round == 5:
            # Final Four: pair region winners
            # Standard pairing: East vs South, Midwest vs West
            region_winners = {}
            for game in current_games:
                if game.winner:
                    region_winners[game.region] = game.winner

            pairings = [
                (REGIONS[0], REGIONS[1]),  # East vs South
                (REGIONS[2], REGIONS[3]),  # Midwest vs West
            ]

            for r1, r2 in pairings:
                if r1 in region_winners and r2 in region_winners:
                    game = Game(
                        team_a=region_winners[r1],
                        team_b=region_winners[r2],
                        round_num=next_round,
                        region="Final Four",
                    )
                    next_games.append(game)
                    self.games.append(game)

        elif next_round == 6:
            # Championship
            ff_games = self.rounds.get(5, [])
            winners = [g.winner for g in ff_games if g.winner]

            if len(winners) >= 2:
                game = Game(
                    team_a=winners[0],
                    team_b=winners[1],
                    round_num=next_round,
                    region="Championship",
                )
                next_games.append(game)
                self.games.append(game)

        self.rounds[next_round] = next_games
        return next_games

    def get_all_teams(self) -> list[Team]:
        """Get all teams in the bracket."""
        all_teams = []
        for teams in self.regions.values():
            all_teams.extend(teams)
        return all_teams

    def copy(self) -> "Bracket":
        """Deep copy the bracket for simulation."""
        return deepcopy(self)


def build_bracket_from_teams(tournament_teams: list[dict], team_features: pd.DataFrame) -> Bracket:
    """Build a Bracket from team data and features.

    Args:
        tournament_teams: List of dicts with 'name', 'seed', 'region'
        team_features: DataFrame with team features (from features.builder)

    Returns:
        A populated Bracket ready for simulation.
    """
    bracket = Bracket()

    # Create lookup for features by normalized name
    feature_lookup = {}
    if team_features is not None and not team_features.empty:
        for _, row in team_features.iterrows():
            name = row.get("name_norm", row.get("name", ""))
            feature_lookup[name] = row

    for team_data in tournament_teams:
        name = team_data["name"]
        seed = team_data.get("seed", 8)
        region = team_data.get("region", "Unknown")

        # Skip play-in teams (we'll handle those separately)
        if team_data.get("is_play_in", False):
            continue

        # Find matching features
        from data.team_mapping import normalize_team_name
        norm_name = normalize_team_name(name)
        stats = feature_lookup.get(norm_name, pd.Series(dtype=float))

        team = Team(
            name=name,
            seed=int(seed) if pd.notna(seed) else 8,
            region=region,
            stats=stats,
        )
        bracket.add_team(team)

    bracket.build_first_round()
    return bracket
