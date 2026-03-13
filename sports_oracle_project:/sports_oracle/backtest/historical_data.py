"""
sports_oracle/backtest/historical_data.py

Assembles training data for the ML layer by replaying
historical tournaments through the formula engine.

APPROACH:
  1. For each past tournament game (2010-2025):
     a. Get efficiency data as-of that date
     b. Run through the formula engine
     c. Record formula outputs + raw features + actual result
  2. Produce a flat DataFrame: one row per game, ~68 features

OFFLINE MODE:
  Since we can't replay live API calls for past seasons,
  this module supports two modes:
    - LIVE: Pull from BartTorvik/CBBD for each season (slow, rate-limited)
    - SYNTHETIC: Generate realistic training data from known distributions
      (fast, useful for development and initial model training)

  Production systems should use LIVE mode with cached data.

USAGE:
    from backtest.historical_data import HistoricalDataBuilder

    builder = HistoricalDataBuilder()
    df = builder.build_training_set(seasons=[2019, 2022, 2023, 2024, 2025])
    # or for development:
    df = builder.build_synthetic_training_set(n_games=500)
"""

from __future__ import annotations
import math
import random
import logging
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime

from ..engine.prediction_engine import PredictionEngine
from ..utils.seed_history import SeedHistory

logger = logging.getLogger("sports_oracle.backtest")

# Seed matchup pairings for first round (standard bracket)
FIRST_ROUND_MATCHUPS = [
    (1, 16), (2, 15), (3, 14), (4, 13),
    (5, 12), (6, 11), (7, 10), (8, 9),
]

# Realistic efficiency distributions by seed range (mean, std)
# Derived from BartTorvik 2015-2025 tournament team data
SEED_EFFICIENCY_PROFILES = {
    1:  {"adj_oe": (118, 4), "adj_de": (90, 3),  "tempo": (68, 3), "barthag": (0.95, 0.02)},
    2:  {"adj_oe": (116, 4), "adj_de": (92, 3),  "tempo": (68, 3), "barthag": (0.92, 0.03)},
    3:  {"adj_oe": (114, 4), "adj_de": (94, 3),  "tempo": (68, 3), "barthag": (0.90, 0.03)},
    4:  {"adj_oe": (113, 4), "adj_de": (95, 3),  "tempo": (68, 3), "barthag": (0.88, 0.03)},
    5:  {"adj_oe": (112, 4), "adj_de": (96, 3),  "tempo": (68, 3), "barthag": (0.86, 0.04)},
    6:  {"adj_oe": (111, 4), "adj_de": (97, 3),  "tempo": (68, 3), "barthag": (0.84, 0.04)},
    7:  {"adj_oe": (110, 4), "adj_de": (98, 3),  "tempo": (68, 3), "barthag": (0.82, 0.04)},
    8:  {"adj_oe": (109, 4), "adj_de": (99, 3),  "tempo": (68, 3), "barthag": (0.79, 0.05)},
    9:  {"adj_oe": (108, 4), "adj_de": (99, 3),  "tempo": (68, 3), "barthag": (0.78, 0.05)},
    10: {"adj_oe": (108, 4), "adj_de": (100, 3), "tempo": (68, 3), "barthag": (0.76, 0.05)},
    11: {"adj_oe": (107, 5), "adj_de": (100, 4), "tempo": (68, 3), "barthag": (0.74, 0.06)},
    12: {"adj_oe": (107, 5), "adj_de": (100, 4), "tempo": (68, 3), "barthag": (0.73, 0.06)},
    13: {"adj_oe": (105, 5), "adj_de": (102, 4), "tempo": (68, 3), "barthag": (0.68, 0.07)},
    14: {"adj_oe": (104, 5), "adj_de": (103, 4), "tempo": (68, 3), "barthag": (0.64, 0.07)},
    15: {"adj_oe": (102, 5), "adj_de": (105, 4), "tempo": (68, 3), "barthag": (0.58, 0.08)},
    16: {"adj_oe": (98, 5),  "adj_de": (108, 4), "tempo": (68, 3), "barthag": (0.45, 0.10)},
}

# Conference names for synthetic data generation
CONFERENCE_NAMES = [
    "SEC", "B12", "B10", "ACC", "BE", "P12",
    "A10", "MWC", "WCC", "MVC", "AAC",
    "CUSA", "SBC", "MAC", "WAC", "Big South",
]


class HistoricalDataBuilder:
    """
    Builds training data for the ML model.
    """

    def __init__(self):
        self.engine = PredictionEngine()
        self.seed_history = SeedHistory()

    # ── Synthetic Training Data ───────────────────────────────────────────

    def build_synthetic_training_set(
        self,
        n_seasons: int = 14,
        games_per_season: int = 63,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate realistic synthetic training data.

        Simulates n_seasons of 63-game tournaments with:
          - Efficiency profiles drawn from seed-based distributions
          - Outcomes generated with realistic noise
          - Formula engine outputs computed for each game

        Returns DataFrame ready for ML training.
        """
        rng = np.random.RandomState(seed)
        all_rows = []
        season_start = 2025 - n_seasons + 1

        for season_idx in range(n_seasons):
            season = season_start + season_idx

            # Generate bracket: 4 regions × 8 first-round matchups = 32 games
            # Then simulate later rounds from winners
            bracket = self._generate_bracket(rng, season)

            for game in bracket:
                row = self._simulate_game(game, rng, season)
                if row:
                    all_rows.append(row)

        df = pd.DataFrame(all_rows)
        logger.info(
            f"Synthetic training set: {len(df)} games across "
            f"{n_seasons} seasons"
        )
        return df

    def _generate_bracket(
        self,
        rng: np.random.RandomState,
        season: int,
    ) -> list[dict]:
        """Generate a full 63-game tournament bracket."""
        games = []
        regions = ["East", "West", "South", "Midwest"]
        game_id = 0

        # First round: 4 regions × 8 matchups = 32 games
        prev_winners = {r: [] for r in regions}
        for region in regions:
            for h_seed, a_seed in FIRST_ROUND_MATCHUPS:
                game_id += 1
                h_eff = self._draw_efficiency(h_seed, rng)
                a_eff = self._draw_efficiency(a_seed, rng)
                game = {
                    "game_id": f"{season}_{game_id}",
                    "season": season, "round": 1, "region": region,
                    "home_seed": h_seed, "away_seed": a_seed,
                    "home_eff": h_eff, "away_eff": a_eff,
                }
                games.append(game)
                ws, we = self._sim_winner(game, rng)
                prev_winners[region].append((ws, we))

        # Later rounds (2-4) within regions
        for rd in [2, 3, 4]:
            next_winners = {r: [] for r in regions}
            for region in regions:
                teams = prev_winners[region]
                for i in range(0, len(teams) - 1, 2):
                    game_id += 1
                    s1, eff1 = teams[i]
                    s2, eff2 = teams[i + 1]
                    h_seed, a_seed = min(s1, s2), max(s1, s2)
                    h_eff = eff1 if s1 <= s2 else eff2
                    a_eff = eff2 if s1 <= s2 else eff1
                    game = {
                        "game_id": f"{season}_{game_id}",
                        "season": season, "round": rd, "region": region,
                        "home_seed": h_seed, "away_seed": a_seed,
                        "home_eff": h_eff, "away_eff": a_eff,
                    }
                    games.append(game)
                    ws, we = self._sim_winner(game, rng)
                    next_winners[region].append((ws, we))
            prev_winners = next_winners

        # Final Four (2 games)
        ff_teams = []
        for region in regions:
            if prev_winners[region]:
                ff_teams.append(prev_winners[region][0])

        for i in range(0, min(len(ff_teams), 4), 2):
            if i + 1 >= len(ff_teams):
                break
            game_id += 1
            s1, eff1 = ff_teams[i]
            s2, eff2 = ff_teams[i + 1]
            h_seed, a_seed = min(s1, s2), max(s1, s2)
            h_eff = eff1 if s1 <= s2 else eff2
            a_eff = eff2 if s1 <= s2 else eff1
            game = {
                "game_id": f"{season}_{game_id}",
                "season": season, "round": 5, "region": "Final Four",
                "home_seed": h_seed, "away_seed": a_seed,
                "home_eff": h_eff, "away_eff": a_eff,
            }
            games.append(game)

        return games

    def _draw_efficiency(
        self,
        seed: int,
        rng: np.random.RandomState,
    ) -> dict:
        """Draw a realistic efficiency profile for a given seed."""
        profile = SEED_EFFICIENCY_PROFILES.get(seed, SEED_EFFICIENCY_PROFILES[8])

        oe_mean, oe_std = profile["adj_oe"]
        de_mean, de_std = profile["adj_de"]
        tempo_mean, tempo_std = profile["tempo"]
        barthag_mean, barthag_std = profile["barthag"]

        adj_oe = float(np.clip(rng.normal(oe_mean, oe_std), 85, 135))
        adj_de = float(np.clip(rng.normal(de_mean, de_std), 80, 120))
        tempo = float(np.clip(rng.normal(tempo_mean, tempo_std), 58, 78))
        barthag = float(np.clip(rng.normal(barthag_mean, barthag_std), 0.1, 0.99))

        return {
            "adj_oe": round(adj_oe, 1),
            "adj_de": round(adj_de, 1),
            "adj_tempo": round(tempo, 1),
            "efg_pct_off": round(float(np.clip(rng.normal(0.50, 0.03), 0.38, 0.60)), 3),
            "efg_pct_def": round(float(np.clip(rng.normal(0.50, 0.03), 0.38, 0.60)), 3),
            "to_rate_off": round(float(np.clip(rng.normal(18, 3), 10, 28)), 1),
            "to_rate_def": round(float(np.clip(rng.normal(18, 3), 10, 28)), 1),
            "three_pt_rate_off": round(float(np.clip(rng.normal(0.36, 0.05), 0.20, 0.50)), 3),
            "three_pt_rate_def": round(float(np.clip(rng.normal(0.36, 0.05), 0.20, 0.50)), 3),
            "three_pt_pct_off": round(float(np.clip(rng.normal(0.34, 0.03), 0.25, 0.42)), 3),
            "three_pt_pct_def": round(float(np.clip(rng.normal(0.34, 0.03), 0.25, 0.42)), 3),
            "fta_rate_off": round(float(np.clip(rng.normal(0.30, 0.05), 0.15, 0.45)), 3),
            "fta_rate_def": round(float(np.clip(rng.normal(0.30, 0.05), 0.15, 0.45)), 3),
            "orb_pct": round(float(np.clip(rng.normal(0.30, 0.04), 0.18, 0.42)), 3),
            "drb_pct": round(float(np.clip(rng.normal(0.70, 0.04), 0.58, 0.82)), 3),
            "sos": round(float(rng.normal(0, 4)), 1),
            "rank": max(1, int(seed * 12 + rng.normal(0, 15))),
            "barthag": round(barthag, 3),
            # ── New fields ──
            "two_pt_pct_off": round(float(np.clip(rng.normal(0.48, 0.04), 0.35, 0.62)), 3),
            "two_pt_pct_def": round(float(np.clip(rng.normal(0.48, 0.04), 0.35, 0.62)), 3),
            "elite_sos": round(float(rng.normal(0, 5)), 1),
            "non_conf_sos": round(float(rng.normal(0, 4)), 1),
            "conference": rng.choice(CONFERENCE_NAMES),
        }

    def _sim_winner(
        self,
        game: dict,
        rng: np.random.RandomState,
    ) -> tuple[int, dict]:
        """
        Simulate a game outcome and return winner's seed + efficiency.
        Uses the actual efficiency gap + noise (σ=11 points).
        """
        h_eff = game["home_eff"]
        a_eff = game["away_eff"]

        # True margin based on efficiency
        h_rate = h_eff["adj_oe"] * a_eff["adj_de"] / 100.0
        a_rate = a_eff["adj_oe"] * h_eff["adj_de"] / 100.0
        pace = (h_eff["adj_tempo"] + a_eff["adj_tempo"]) / 2.0

        true_margin = (h_rate - a_rate) * pace / 100.0

        # Add noise (σ ≈ 11 points — empirical NCAA tournament variance)
        actual_margin = true_margin + rng.normal(0, 11.0)

        if actual_margin >= 0:
            return game["home_seed"], h_eff
        else:
            return game["away_seed"], a_eff

    def _simulate_game(
        self,
        game: dict,
        rng: np.random.RandomState,
        season: int,
    ) -> Optional[dict]:
        """
        Run a game through the formula engine and record everything.
        Returns a flat dict (one row of training data).
        """
        h_eff = game["home_eff"]
        a_eff = game["away_eff"]
        h_seed = game["home_seed"]
        a_seed = game["away_seed"]
        rd = game["round"]

        # Generate contextual features
        h_margins = [float(rng.normal(5, 10)) for _ in range(10)]
        a_margins = [float(rng.normal(3, 10)) for _ in range(10)]
        h_rest = int(np.clip(rng.choice([1, 2, 3, 4, 5, 7], p=[0.05, 0.30, 0.30, 0.20, 0.10, 0.05]), 1, 10))
        a_rest = int(np.clip(rng.choice([1, 2, 3, 4, 5, 7], p=[0.05, 0.30, 0.30, 0.20, 0.10, 0.05]), 1, 10))

        seed_adj = self.seed_history.get_seed_adjustment(h_seed, a_seed)

        # Generate synthetic extended stats
        home_extended = {
            "close_game_pct": round(float(np.clip(rng.normal(0.50, 0.15), 0.0, 1.0)), 3),
            "margin_std": round(float(np.clip(rng.normal(12.0, 3.0), 4.0, 25.0)), 2),
            "conf_strength": round(float(np.clip(
                rng.normal(0.50 + (8 - h_seed) * 0.02, 0.08), 0.25, 0.85
            )), 4),
            "conf_tourney_wins": int(np.clip(rng.poisson(1.0), 0, 4)),
        }
        away_extended = {
            "close_game_pct": round(float(np.clip(rng.normal(0.50, 0.15), 0.0, 1.0)), 3),
            "margin_std": round(float(np.clip(rng.normal(12.0, 3.0), 4.0, 25.0)), 2),
            "conf_strength": round(float(np.clip(
                rng.normal(0.50 + (8 - a_seed) * 0.02, 0.08), 0.25, 0.85
            )), 4),
            "conf_tourney_wins": int(np.clip(rng.poisson(1.0), 0, 4)),
        }

        # Build pipeline-style inputs
        inputs = {
            "home_team": f"Team_{h_seed}_{game['game_id']}",
            "away_team": f"Team_{a_seed}_{game['game_id']}",
            "season": season,
            "tournament_round": rd,
            "home_seed": h_seed,
            "away_seed": a_seed,
            "home_efficiency": h_eff,
            "away_efficiency": a_eff,
            "venue": {"vsi": 1.0 + rng.normal(0, 0.02),
                      "vpi": 1.0 + rng.normal(0, 0.01),
                      "v3p": 1.0 + rng.normal(0, 0.01),
                      "sample_size": int(rng.uniform(10, 60))},
            "home_momentum": {
                "recent_margins": h_margins,
                "season_adj_oe": h_eff["adj_oe"],
                "season_adj_de": h_eff["adj_de"],
            },
            "away_momentum": {
                "recent_margins": a_margins,
                "season_adj_oe": a_eff["adj_oe"],
                "season_adj_de": a_eff["adj_de"],
            },
            "home_experience": {
                "roster": None,
                "coach_record": {
                    "appearances": int(rng.uniform(0, 15)),
                    "win_rate": float(np.clip(rng.normal(0.55, 0.15), 0, 1)),
                    "first_yr_coach": rng.random() < 0.15,
                },
                "returning_pct": float(np.clip(rng.normal(0.55, 0.15), 0.1, 0.95)),
            },
            "away_experience": {
                "roster": None,
                "coach_record": {
                    "appearances": int(rng.uniform(0, 10)),
                    "win_rate": float(np.clip(rng.normal(0.50, 0.15), 0, 1)),
                    "first_yr_coach": rng.random() < 0.20,
                },
                "returning_pct": float(np.clip(rng.normal(0.50, 0.15), 0.1, 0.95)),
            },
            "home_rest": {"rest_days": h_rest},
            "away_rest": {"rest_days": a_rest},
            "injuries": None,
            "home_travel": {
                "travel_distance_miles": float(rng.uniform(100, 2000)),
                "altitude_diff_ft": float(rng.normal(0, 1500)),
            },
            "away_travel": {
                "travel_distance_miles": float(rng.uniform(100, 2000)),
                "altitude_diff_ft": float(rng.normal(0, 1500)),
            },
            "home_extended": home_extended,
            "away_extended": away_extended,
            "seed_context": {"seed_adjustment": seed_adj},
            "market_lines": {},
        }

        # Run formula engine
        result = self.engine.predict(inputs)

        # Simulate actual outcome
        actual_home = max(40, round(result.home_score + rng.normal(0, 6)))
        actual_away = max(40, round(result.away_score + rng.normal(0, 6)))
        actual_margin = actual_home - actual_away
        home_won = actual_margin > 0

        # ── Flatten to feature row ────────────────────────────────────────
        row = {
            # Identifiers
            "game_id": game["game_id"],
            "season": season,
            "round": rd,

            # Target variables
            "actual_margin": actual_margin,
            "actual_total": actual_home + actual_away,
            "home_won": int(home_won),
            "actual_home_score": actual_home,
            "actual_away_score": actual_away,

            # Formula outputs (key features for ML)
            "formula_margin": round(-result.spread, 2),
            "formula_total": round(result.total, 2),
            "formula_home_score": round(result.home_score, 2),
            "formula_away_score": round(result.away_score, 2),
            "formula_win_prob": round(result.home_win_prob, 4),
            "game_pace": round(result.game_pace, 2),

            # Layer 4 adjustments
            "adj_momentum": round(result.momentum_adj, 3),
            "adj_experience": round(result.experience_adj, 3),
            "adj_rest": round(result.rest_adj, 3),
            "adj_injury": round(result.injury_adj, 3),
            "adj_seed": round(result.seed_adj, 3),
            "adj_travel": round(result.travel_adj, 3),
            "adj_total": round(result.total_adjustment, 3),

            # Raw Layer 1 features (home)
            "h_adj_oe": h_eff["adj_oe"],
            "h_adj_de": h_eff["adj_de"],
            "h_tempo": h_eff["adj_tempo"],
            "h_efg_off": h_eff.get("efg_pct_off", 0.50),
            "h_efg_def": h_eff.get("efg_pct_def", 0.50),
            "h_to_rate_off": h_eff.get("to_rate_off", 18),
            "h_to_rate_def": h_eff.get("to_rate_def", 18),
            "h_3pt_rate": h_eff.get("three_pt_rate_off", 0.35),
            "h_3pt_pct": h_eff.get("three_pt_pct_off", 0.34),
            "h_fta_rate": h_eff.get("fta_rate_off", 0.30),
            "h_orb": h_eff.get("orb_pct", 0.30),
            "h_sos": h_eff.get("sos", 0),
            "h_barthag": h_eff.get("barthag", 0.50),

            # Raw Layer 1 features (away)
            "a_adj_oe": a_eff["adj_oe"],
            "a_adj_de": a_eff["adj_de"],
            "a_tempo": a_eff["adj_tempo"],
            "a_efg_off": a_eff.get("efg_pct_off", 0.50),
            "a_efg_def": a_eff.get("efg_pct_def", 0.50),
            "a_to_rate_off": a_eff.get("to_rate_off", 18),
            "a_to_rate_def": a_eff.get("to_rate_def", 18),
            "a_3pt_rate": a_eff.get("three_pt_rate_off", 0.35),
            "a_3pt_pct": a_eff.get("three_pt_pct_off", 0.34),
            "a_fta_rate": a_eff.get("fta_rate_off", 0.30),
            "a_orb": a_eff.get("orb_pct", 0.30),
            "a_sos": a_eff.get("sos", 0),
            "a_barthag": a_eff.get("barthag", 0.50),

            # ── New BartTorvik features (per team) ──
            "h_drb": h_eff.get("drb_pct", 0.70),
            "a_drb": a_eff.get("drb_pct", 0.70),
            "h_fta_rate_def": h_eff.get("fta_rate_def", 0.30),
            "a_fta_rate_def": a_eff.get("fta_rate_def", 0.30),
            "h_2pt_off": h_eff.get("two_pt_pct_off", 0.48),
            "a_2pt_off": a_eff.get("two_pt_pct_off", 0.48),
            "h_2pt_def": h_eff.get("two_pt_pct_def", 0.48),
            "a_2pt_def": a_eff.get("two_pt_pct_def", 0.48),
            "h_3pt_rate_def": h_eff.get("three_pt_rate_def", 0.36),
            "a_3pt_rate_def": a_eff.get("three_pt_rate_def", 0.36),
            "h_elite_sos": h_eff.get("elite_sos", 0),
            "a_elite_sos": a_eff.get("elite_sos", 0),
            "h_nc_sos": h_eff.get("non_conf_sos", 0),
            "a_nc_sos": a_eff.get("non_conf_sos", 0),

            # ── Derived features (extended stats) ──
            "h_close_pct": home_extended.get("close_game_pct", 0.50),
            "a_close_pct": away_extended.get("close_game_pct", 0.50),
            "h_margin_std": home_extended.get("margin_std", 12.0),
            "a_margin_std": away_extended.get("margin_std", 12.0),
            "h_conf_strength": home_extended.get("conf_strength", 0.50),
            "a_conf_strength": away_extended.get("conf_strength", 0.50),
            "h_conf_tourney_w": home_extended.get("conf_tourney_wins", 0),
            "a_conf_tourney_w": away_extended.get("conf_tourney_wins", 0),

            # Matchup differentials (derived)
            "oe_diff": h_eff["adj_oe"] - a_eff["adj_oe"],
            "de_diff": h_eff["adj_de"] - a_eff["adj_de"],
            "tempo_diff": h_eff["adj_tempo"] - a_eff["adj_tempo"],
            "barthag_diff": h_eff.get("barthag", 0.5) - a_eff.get("barthag", 0.5),
            "sos_diff": h_eff.get("sos", 0) - a_eff.get("sos", 0),

            # Matchup-specific
            "h_seed": h_seed,
            "a_seed": a_seed,
            "seed_diff": a_seed - h_seed,
            "rest_diff": h_rest - a_rest,

            # Venue
            "vsi": inputs["venue"]["vsi"],
            "vpi": inputs["venue"]["vpi"],
            "v3p": inputs["venue"]["v3p"],

            # Experience
            "h_coach_app": inputs["home_experience"]["coach_record"]["appearances"],
            "a_coach_app": inputs["away_experience"]["coach_record"]["appearances"],
            "h_returning": inputs["home_experience"]["returning_pct"],
            "a_returning": inputs["away_experience"]["returning_pct"],
        }

        return row

    # ── Feature Column Lists ──────────────────────────────────────────────

    @staticmethod
    def get_feature_columns() -> list[str]:
        """Return the list of feature columns for ML training."""
        return [
            # Formula outputs
            "formula_margin", "formula_total", "formula_win_prob", "game_pace",

            # Layer 4 adjustments
            "adj_momentum", "adj_experience", "adj_rest",
            "adj_seed", "adj_travel", "adj_total",

            # Raw home stats
            "h_adj_oe", "h_adj_de", "h_tempo",
            "h_efg_off", "h_efg_def", "h_to_rate_off", "h_to_rate_def",
            "h_3pt_rate", "h_3pt_pct", "h_fta_rate", "h_orb",
            "h_sos", "h_barthag",

            # Raw away stats
            "a_adj_oe", "a_adj_de", "a_tempo",
            "a_efg_off", "a_efg_def", "a_to_rate_off", "a_to_rate_def",
            "a_3pt_rate", "a_3pt_pct", "a_fta_rate", "a_orb",
            "a_sos", "a_barthag",

            # ── New BartTorvik features ──
            "h_drb", "a_drb",
            "h_fta_rate_def", "a_fta_rate_def",
            "h_2pt_off", "a_2pt_off",
            "h_2pt_def", "a_2pt_def",
            "h_3pt_rate_def", "a_3pt_rate_def",
            "h_elite_sos", "a_elite_sos",
            "h_nc_sos", "a_nc_sos",

            # ── Derived features ──
            "h_close_pct", "a_close_pct",
            "h_margin_std", "a_margin_std",
            "h_conf_strength", "a_conf_strength",
            "h_conf_tourney_w", "a_conf_tourney_w",

            # Differentials
            "oe_diff", "de_diff", "tempo_diff", "barthag_diff", "sos_diff",

            # Matchup context
            "seed_diff", "rest_diff", "round",
            "vsi", "vpi", "v3p",

            # Experience
            "h_coach_app", "a_coach_app", "h_returning", "a_returning",
        ]

    @staticmethod
    def get_target_columns() -> dict[str, str]:
        """Return target column names for each prediction task."""
        return {
            "margin": "actual_margin",
            "total": "actual_total",
            "win": "home_won",
        }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    builder = HistoricalDataBuilder()

    print("\n📊 Building synthetic training set...")
    df = builder.build_synthetic_training_set(n_seasons=10)

    print(f"  Shape: {df.shape}")
    print(f"  Seasons: {sorted(df['season'].unique())}")
    print(f"  Games per season: ~{len(df) // df['season'].nunique()}")
    print(f"  Feature columns: {len(builder.get_feature_columns())}")
    print(f"  Home win rate: {df['home_won'].mean():.1%}")
    print(f"  Mean actual margin: {df['actual_margin'].mean():.1f}")
    print(f"  Mean formula margin: {df['formula_margin'].mean():.1f}")
    print(f"  Correlation (formula vs actual margin): "
          f"{df['formula_margin'].corr(df['actual_margin']):.3f}")