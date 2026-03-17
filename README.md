# March Madness 2026 Bracket Prediction Model

A statistical model that predicts the 2026 NCAA Men's Basketball Tournament bracket using ensemble machine learning and Monte Carlo simulation.

Scrapes real team data from ESPN and Sports-Reference, trains a Random Forest + XGBoost ensemble on matchup features, then simulates 10,000 brackets to produce win probabilities, advancement odds, and upset alerts.

## 2026 Predictions

| Team | Champion % | Final Four % | Elite 8 % |
|------|-----------|-------------|----------|
| Duke (1, East) | 29.8% | 60% | 62% |
| Arizona (1, West) | 29.5% | 78% | 91% |
| Michigan (1, Midwest) | 23.5% | 60% | 62% |
| Florida (1, South) | 10.9% | 74% | 94% |
| Illinois (3, South) | 2.2% | 18% | 36% |
| Iowa State (2, Midwest) | 1.2% | 12% | 38% |
| Purdue (2, West) | 1.1% | 16% | 73% |

**Predicted Final Four:** Duke vs Florida, Michigan vs Arizona

**Predicted Champion:** Duke over Michigan

### Upset Alerts

| Matchup | Upset Prob | Historical Rate |
|---------|-----------|----------------|
| (5) Vanderbilt over (4) Nebraska | 63% | 45% |
| (13) Cal Baptist over (4) Kansas | 37% | 21% |
| (10) Texas A&M over (7) Saint Mary's | 37% | 39% |
| (12) Akron over (5) Texas Tech | 37% | 35% |

## How It Works

### Data Pipeline

The model collects data from three sources, with graceful fallback if any source is unavailable:

1. **ESPN API** (primary) — Team stats for all 362 D-I teams: FG%, 3P%, FT%, points, rebounds, assists, steals, blocks, turnovers. No authentication required.
2. **Sports-Reference** — Advanced stats: SRS (Simple Rating System), SOS (Strength of Schedule), pace, offensive rating, eFG%, turnover rate, offensive rebound rate, free throw rate, true shooting %.
3. **Bart Torvik** (optional) — Adjusted offensive/defensive efficiency, Barthag, adjusted tempo, four factors. Blocked by Cloudflare on some networks; model works without it.

All scraped data is cached in `cache/` to avoid redundant fetches.

### Feature Engineering

For each potential game matchup, the model computes ~25 features:

- **Seed features:** seed difference, individual seeds, historical seed-matchup win rates
- **Efficiency features:** adjusted efficiency margin diff, offensive/defensive rating differentials
- **Four factors matchup:** each team's offensive eFG%, turnover rate, ORB%, FT rate compared against the opponent's defensive equivalent (8 features)
- **Strength features:** SRS difference, SOS difference, win % difference, point differential difference
- **Tempo:** pace difference and absolute tempo mismatch
- **Shooting:** three-point rate difference, true shooting % difference

SRS is the primary quality metric when Torvik data is unavailable — unlike raw offensive/defensive ratings, SRS is schedule-adjusted, preventing mid-major teams with weak schedules from appearing artificially strong.

### Models

Three base models are trained and combined via optimized weighted averaging:

| Model | Role | Training Time |
|-------|------|--------------|
| **Logistic Regression** | Interpretable baseline with L2 regularization | <1 second |
| **Random Forest** | 300 trees, captures nonlinear interactions | ~2 seconds |
| **XGBoost** | Gradient boosted trees with early stopping | ~5 seconds |

**Ensemble weights** are optimized by minimizing log-loss on a validation split using Nelder-Mead optimization. Typical weights: Random Forest ~60%, XGBoost ~35%, Logistic ~5%.

**Probability calibration:** Isotonic regression ensures predicted probabilities are well-calibrated. All predictions are clipped to [0.02, 0.98] to avoid catastrophic log-loss from overconfident wrong predictions — upsets do happen (UMBC over Virginia, Fairleigh Dickinson over Purdue).

### Training Data

The model supports two training modes:

- **Historical (preferred):** If Kaggle March Machine Learning Mania CSVs are placed in `historical_data/`, the model trains on ~1,400 actual tournament games (2003-2025), computing matchup features from regular season box scores for each year.
- **Synthetic (fallback):** Without Kaggle data, generates training samples from all pairwise matchups of tournament teams, labeled using seed-based priors adjusted by efficiency margins.

### Bracket Simulation

1. **Deterministic bracket:** Picks the higher-probability team in every game to produce the single most likely bracket.
2. **Monte Carlo simulation:** Pre-computes all 64x64 pairwise win probabilities (2,016 unique pairs), then runs 10,000 stochastic simulations using random draws. Each simulation takes ~0.07ms; all 10,000 complete in <1 second.
3. **Upset detection:** Flags first/second-round games where the lower seed has 30%+ win probability, with explanations of the key statistical factors.

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/ericliu-12/march-madness.git
cd march-madness
pip install -r requirements.txt

# macOS only — required for XGBoost
brew install libomp
```

### Run

```bash
# Full run with Monte Carlo (recommended, ~40 seconds)
python main.py

# Quick run without Monte Carlo (~4 seconds)
python main.py --no-monte-carlo

# Force re-scrape fresh data from ESPN/Sports-Reference
python main.py --scrape

# Compact single-line-per-game bracket view
python main.py --compact

# Custom simulation count
python main.py --simulations 50000
```

### Optional: Add Historical Training Data

For better model accuracy, download the [March Machine Learning Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) dataset from Kaggle and place the CSVs in `historical_data/`:

```
historical_data/
  MNCAATourneyDetailedResults.csv
  MRegularSeasonDetailedResults.csv
  MNCAATourneySeeds.csv
  MTeams.csv
```

This gives the model ~1,400 real tournament games to train on instead of synthetic data.

## Output

All results are saved to `output_results/`:

| File | Description |
|------|-------------|
| `predictions_game_by_game.csv` | Every game with teams, seeds, win probability, predicted winner, upset flag |
| `predictions_advancement.csv` | Every team's probability of reaching each round (R32 through Champion) |
| `upset_alerts.csv` | Flagged upsets with probability, historical rate, upset score, key factors |
| `championship_odds.png` | Bar chart of title odds for top 15 teams |
| `advancement_heatmap.png` | Color-coded grid of advancement probabilities (teams x rounds) |
| `upset_analysis.png` | Scatter plot of model upset probability vs historical baseline |

## Project Structure

```
march-madness/
├── main.py                        # CLI entry point, orchestrates full pipeline
├── config.py                      # Global configuration and constants
├── requirements.txt
│
├── data/                          # Data collection
│   ├── scraper_espn.py            # ESPN JSON API (362 teams, no auth)
│   ├── scraper_sportsref.py       # Sports-Reference HTML scraper
│   ├── scraper_torvik.py          # Bart Torvik with Cloudflare fallback
│   ├── scraper_ncaa.py            # NCAA API + hardcoded 2026 bracket
│   ├── kaggle_loader.py           # Historical tournament data from Kaggle
│   └── team_mapping.py            # Team name normalization across sources
│
├── features/                      # Feature engineering
│   ├── builder.py                 # Pipeline orchestrator
│   ├── team_features.py           # Merge multi-source team profiles
│   ├── matchup_features.py        # ~25 pairwise game features
│   └── historical_features.py     # Seed priors and upset scoring
│
├── models/                        # Machine learning
│   ├── logistic.py                # L2 logistic regression
│   ├── random_forest.py           # Random forest classifier
│   ├── xgboost_model.py           # XGBoost with early stopping
│   ├── ensemble.py                # Weighted averaging + weight optimization
│   └── calibration.py             # Isotonic regression calibration
│
├── bracket/                       # Tournament simulation
│   ├── structure.py               # Team/Game/Bracket data structures
│   ├── simulator.py               # Single bracket simulation
│   ├── monte_carlo.py             # 10,000-sim Monte Carlo engine
│   └── upset_detector.py          # Upset identification + explanation
│
├── output/                        # Results
│   ├── bracket_display.py         # ASCII bracket renderer
│   ├── csv_export.py              # CSV exports
│   └── visualizations.py          # matplotlib charts
│
├── cache/                         # Scraped data cache (gitignored)
├── historical_data/               # Kaggle CSVs (gitignored)
└── output_results/                # Generated predictions (gitignored)
```

## Methodology Notes

**Why SRS over raw efficiency?** Sports-Reference's SRS is a schedule-adjusted point differential — it accounts for opponent strength. Without Torvik's adjusted metrics, raw offensive/defensive ratings make mid-majors with weak schedules look artificially elite. SRS fixes this.

**Why clip probabilities at [0.02, 0.98]?** Log-loss (the standard metric for bracket prediction) penalizes confident wrong predictions catastrophically. A model that says "99% chance the 1-seed wins" and is wrong once loses more than it gains from being right 99 times. Since 15-over-2 and 16-over-1 upsets do happen, we enforce humility.

**Why pre-compute pairwise probabilities?** The original Monte Carlo called the model ensemble 63 times per simulation (once per game), with feature computation and model inference each time. At 10,000 simulations, that's 630,000 model calls. Pre-computing all 2,016 unique pairwise probabilities once reduces this to 2,016 model calls + 630,000 dictionary lookups — a ~10,000x speedup.

## License

MIT
