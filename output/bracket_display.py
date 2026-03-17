"""ASCII bracket display for console output.

Renders the predicted bracket region-by-region with win probabilities
and highlights upsets.
"""

from bracket.structure import Bracket, Game
from config import ROUNDS


def display_bracket(bracket: Bracket):
    """Display the full bracket in the console."""
    print("\n" + "=" * 80)
    print("PREDICTED BRACKET — 2026 NCAA TOURNAMENT")
    print("=" * 80)

    # Display each region
    for region_name in bracket.regions:
        _display_region(bracket, region_name)

    # Display Final Four and Championship
    _display_final_rounds(bracket)


def _display_region(bracket: Bracket, region: str):
    """Display a single region's bracket."""
    print(f"\n{'─' * 70}")
    print(f"  {region.upper()} REGION")
    print(f"{'─' * 70}")

    # Round 1
    r1_games = [g for g in bracket.rounds.get(1, []) if g.region == region]
    if not r1_games:
        return

    print(f"\n  {'First Round':<20s} {'Second Round':<20s} {'Sweet 16':<20s} {'Elite 8':<20s}")
    print(f"  {'─' * 18}   {'─' * 18}   {'─' * 18}   {'─' * 18}")

    # Get games for each round in this region
    r2_games = [g for g in bracket.rounds.get(2, []) if g.region == region]
    r3_games = [g for g in bracket.rounds.get(3, []) if g.region == region]
    r4_games = [g for g in bracket.rounds.get(4, []) if g.region == region]

    # Print first round matchups with results
    for i, game in enumerate(r1_games):
        _print_game_line(game)

        # After every 2 games, show R2 winner
        if i % 2 == 1:
            r2_idx = i // 2
            if r2_idx < len(r2_games):
                _print_advancement(r2_games[r2_idx])

            # After every 4 games, show R3 winner
            if i % 4 == 3:
                r3_idx = i // 4
                if r3_idx < len(r3_games):
                    _print_advancement(r3_games[r3_idx], indent=2)

                # After all 8 games, show R4 winner
                if i == 7 and r4_games:
                    _print_advancement(r4_games[0], indent=3)

            print()

    # Region winner
    if r4_games and r4_games[0].winner:
        print(f"\n  >>> {region} Champion: {r4_games[0].winner} <<<")


def _print_game_line(game: Game):
    """Print a single game matchup with result."""
    a_str = f"({game.team_a.seed:2d}) {game.team_a.name[:18]:<18s}"
    b_str = f"({game.team_b.seed:2d}) {game.team_b.name[:18]:<18s}"

    if game.winner:
        prob = game.win_probability
        if prob is not None:
            winner_prob = prob if game.winner == game.team_a else (1 - prob)
            upset = _is_upset(game)
            marker = " *UPSET*" if upset else ""
            print(f"  {a_str} vs {b_str} -> {game.winner.name[:15]:<15s} ({winner_prob:.0%}){marker}")
        else:
            print(f"  {a_str} vs {b_str} -> {game.winner.name[:15]}")
    else:
        print(f"  {a_str} vs {b_str}")


def _print_advancement(game: Game, indent: int = 1):
    """Print an advancement line."""
    prefix = "    " * indent + "└─> "
    if game.winner:
        prob = game.win_probability
        if prob is not None:
            winner_prob = prob if game.winner == game.team_a else (1 - prob)
            upset = _is_upset(game)
            marker = " *UPSET*" if upset else ""
            print(f"{prefix}{game.winner} ({winner_prob:.0%}){marker}")
        else:
            print(f"{prefix}{game.winner}")


def _display_final_rounds(bracket: Bracket):
    """Display Final Four and Championship."""
    print(f"\n{'=' * 70}")
    print("  FINAL FOUR")
    print(f"{'=' * 70}")

    ff_games = bracket.rounds.get(5, [])
    for game in ff_games:
        _print_game_line(game)

    champ_games = bracket.rounds.get(6, [])
    if champ_games:
        print(f"\n{'=' * 70}")
        print("  CHAMPIONSHIP")
        print(f"{'=' * 70}")
        for game in champ_games:
            _print_game_line(game)
            if game.winner:
                print(f"\n  🏆 PREDICTED CHAMPION: {game.winner} 🏆")


def _is_upset(game: Game) -> bool:
    """Check if the game result is an upset (higher seed won)."""
    if game.winner is None:
        return False
    return game.winner.seed > game.higher_seed.seed


def display_compact_bracket(bracket: Bracket):
    """Display a compact single-line-per-game bracket."""
    print("\n" + "=" * 60)
    print("COMPACT BRACKET PREDICTIONS")
    print("=" * 60)

    for round_num in range(1, 7):
        games = bracket.rounds.get(round_num, [])
        if not games:
            continue

        print(f"\n--- {ROUNDS[round_num]} ---")
        for game in games:
            if game.winner:
                prob = game.win_probability
                winner_prob = prob if game.winner == game.team_a else (1 - prob)
                upset = " [UPSET]" if _is_upset(game) else ""
                region = f" ({game.region})" if round_num <= 4 else ""
                print(
                    f"  ({game.team_a.seed}){game.team_a.name:<20s} vs "
                    f"({game.team_b.seed}){game.team_b.name:<20s} "
                    f"-> {game.winner.name} ({winner_prob:.0%}){upset}{region}"
                )
