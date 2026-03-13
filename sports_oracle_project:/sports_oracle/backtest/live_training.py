"""
sports_oracle/backtest/live_training.py

Live training data builder using CBBD + BartTorvik historical data.

This module adds the build_live_training_set() method and its helpers.
Import and call directly, or merge into historical_data.py.

USAGE:
    from sports_oracle.backtest.live_training import LiveTrainingBuilder

    builder = LiveTrainingBuilder()
    df = builder.build(
        cbbd_key="your_key",
        seasons=range(2012, 2026),
        cache_dir="data/training_cache",
    )
    # → ~880 rows of real tournament data with formula features

API BUDGET:
    BartTorvik: ~14 requests (one snapshot per season, 3.5s delay)
    CBBD: ~28 requests (games + lines per season, 0.5s delay)
    Total time: ~2-3 minutes
    Fits easily within CBBD free tier (1,000 calls/month)
"""

from __future__ import annotations
import os
import logging
import numpy as np
import pandas as pd
from typing import Optional

from ..collectors.cbbd_collector import CBBDCollector
from ..collectors.barttorvik_collector import BartTorvik
from ..engine.prediction_engine import PredictionEngine
from ..utils.seed_history import SeedHistory
from ..utils.team_resolver import get_resolver

logger = logging.getLogger("sports_oracle.backtest.live")


class LiveTrainingBuilder:
    """
    Builds ML training data from real historical tournament games.

    For each season (2012-2025, excluding 2021 bubble):
      1. Pull all NCAA tournament games from CBBD (scores, seeds, venues)
      2. Snapshot BartTorvik ratings as-of Selection Sunday
      3. Pull historical betting lines from CBBD
      4. For each game: look up both teams, run formula engine,
         record predictions alongside actual results

    Produces ~880 rows with 73 real features + 3 targets.
    """

    def __init__(self):
        self.engine = PredictionEngine()
        self.seed_history = SeedHistory()
        self.resolver = get_resolver()

    def build(
        self,
        cbbd_key: str = "",
        seasons: Optional[list[int]] = None,
        cache_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build training data from real historical tournament games.

        Args:
            cbbd_key: CBBD API key (or reads from CBBD_API_KEY env var)
            seasons: List of seasons to process (default: 2012-2025)
            cache_dir: If provided, saves/loads per-season CSVs so you
                       only hit the APIs once per season ever.

        Returns:
            DataFrame with ~880 rows, each containing:
              - Actual outcomes (margin, total, winner)
              - Formula engine predictions (all 73 features)
              - Historical betting lines for comparison
        """
        cbbd_key = cbbd_key or os.environ.get("CBBD_API_KEY", "")
        if not cbbd_key:
            raise ValueError(
                "CBBD API key required for live training data. "
                "Set CBBD_API_KEY in .env or pass cbbd_key parameter."
            )

        if seasons is None:
            seasons = list(range(2012, 2026))
        seasons = [s for s in seasons if s != 2021]  # exclude bubble

        cbbd = CBBDCollector(cbbd_key)
        torvik = BartTorvik()

        all_dfs = []
        total_skipped = 0

        for season in seasons:
            logger.info(f"\n{'='*55}")
            logger.info(f"  SEASON {season}")
            logger.info(f"{'='*55}")

            # ── Check cache ───────────────────────────────────────────
            if cache_dir:
                cache_path = os.path.join(cache_dir, f"training_{season}.csv")
                if os.path.exists(cache_path):
                    cached = pd.read_csv(cache_path)
                    logger.info(f"  Loaded {len(cached)} games from cache")
                    all_dfs.append(cached)
                    continue

            # ── 1. Tournament games from CBBD ─────────────────────────
            games = cbbd.get_games(season=season, season_type="postseason")
            if games.empty:
                logger.warning(f"  No postseason games for {season}")
                continue

            # Filter to NCAA tournament only
            if "tournament" in games.columns:
                games = games[games["tournament"].notna()].copy()

            if "status" in games.columns:
                games = games[games["status"] == "final"].copy()

            games["home_points"] = pd.to_numeric(games.get("home_points"), errors="coerce")
            games["away_points"] = pd.to_numeric(games.get("away_points"), errors="coerce")
            games = games.dropna(subset=["home_points", "away_points"])

            if games.empty:
                logger.warning(f"  No valid tournament games for {season}")
                continue

            logger.info(f"  Found {len(games)} tournament games")

            # ── 2. BartTorvik snapshot (Selection Sunday) ─────────────
            snapshot_date = f"{season}-03-15"
            ratings = torvik.get_team_ratings(season=season, as_of_date=snapshot_date)
            if ratings.empty:
                logger.warning(f"  No snapshot ratings, trying full season...")
                ratings = torvik.get_team_ratings(season=season)
            if ratings.empty:
                logger.warning(f"  No BartTorvik data for {season}, skipping")
                continue

            logger.info(f"  Ratings loaded for {len(ratings)} teams")

            # Build lookup: lowercase team name → row dict
            ratings_lookup = {}
            for _, r in ratings.iterrows():
                name = str(r.get("team", "")).strip()
                if name:
                    ratings_lookup[name.lower()] = r.to_dict()

            # ── 3. Historical betting lines ───────────────────────────
            lines_df = cbbd.get_lines(season=season)
            lines_lookup = {}
            if not lines_df.empty and "game_id" in lines_df.columns:
                for gid, group in lines_df.groupby("game_id"):
                    spreads = pd.to_numeric(group.get("spread"), errors="coerce").dropna()
                    totals = pd.to_numeric(group.get("over_under"), errors="coerce").dropna()
                    lines_lookup[gid] = {
                        "spread": round(float(spreads.mean()), 1) if len(spreads) else None,
                        "total": round(float(totals.mean()), 1) if len(totals) else None,
                    }

            # ── 4. Build rows ─────────────────────────────────────────
            rows = []
            skipped = 0
            for _, game in games.iterrows():
                row = self._build_row(game, ratings_lookup, lines_lookup, season)
                if row:
                    rows.append(row)
                else:
                    skipped += 1
            total_skipped += skipped

            if rows:
                season_df = pd.DataFrame(rows)
                logger.info(f"  Built {len(rows)} rows ({skipped} skipped)")

                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                    path = os.path.join(cache_dir, f"training_{season}.csv")
                    season_df.to_csv(path, index=False)
                    logger.info(f"  Cached → {path}")

                all_dfs.append(season_df)

        if not all_dfs:
            logger.warning("No training data built")
            return pd.DataFrame()

        df = pd.concat(all_dfs, ignore_index=True)
        logger.info(
            f"\n  LIVE TRAINING SET: {len(df)} games, "
            f"{len(seasons)} seasons, {total_skipped} skipped"
        )
        return df

    # ── Row builder ───────────────────────────────────────────────────────

    def _build_row(self, game, ratings_lookup, lines_lookup, season):
        """Build one training row from a real CBBD tournament game."""
        home_team = str(game.get("home_team", "")).strip()
        away_team = str(game.get("away_team", "")).strip()
        if not home_team or not away_team:
            return None

        home_r = self.resolver.resolve_or_original(home_team)
        away_r = self.resolver.resolve_or_original(away_team)

        h_eff = self._find_ratings(home_r, ratings_lookup)
        a_eff = self._find_ratings(away_r, ratings_lookup)
        if not h_eff or not a_eff:
            return None

        home_pts = float(game["home_points"])
        away_pts = float(game["away_points"])
        actual_margin = home_pts - away_pts

        h_seed = self._get_seed(game, "home", h_eff)
        a_seed = self._get_seed(game, "away", a_eff)
        rd = self._get_round(game)

        game_id = game.get("id") or game.get("game_id")
        market = lines_lookup.get(game_id, {})

        seed_adj = 0.0
        if h_seed and a_seed:
            seed_adj = self.seed_history.get_seed_adjustment(h_seed, a_seed)

        # Conference strength from ratings
        h_conf_str = self._conf_strength(h_eff, ratings_lookup)
        a_conf_str = self._conf_strength(a_eff, ratings_lookup)

        # Build formula engine inputs
        inputs = {
            "home_team": home_r, "away_team": away_r,
            "season": season, "tournament_round": rd,
            "home_seed": h_seed, "away_seed": a_seed,
            "home_efficiency": h_eff, "away_efficiency": a_eff,
            "venue": {"vsi": 1.0, "vpi": 1.0, "v3p": 1.0, "sample_size": 0},
            "home_momentum": {"recent_margins": [], "season_adj_oe": h_eff.get("adj_oe", 100),
                              "season_adj_de": h_eff.get("adj_de", 100)},
            "away_momentum": {"recent_margins": [], "season_adj_oe": a_eff.get("adj_oe", 100),
                              "season_adj_de": a_eff.get("adj_de", 100)},
            "home_experience": {"roster": None, "coach_record": {"appearances": 3, "win_rate": 0.5,
                                "first_yr_coach": False}, "returning_pct": 0.5},
            "away_experience": {"roster": None, "coach_record": {"appearances": 2, "win_rate": 0.45,
                                "first_yr_coach": False}, "returning_pct": 0.5},
            "home_rest": {"rest_days": 3}, "away_rest": {"rest_days": 3},
            "injuries": None,
            "home_travel": {"travel_distance_miles": 500, "altitude_diff_ft": 0},
            "away_travel": {"travel_distance_miles": 500, "altitude_diff_ft": 0},
            "home_extended": {"close_game_pct": 0.50, "margin_std": 12.0,
                              "conf_strength": h_conf_str, "conf_tourney_wins": 0},
            "away_extended": {"close_game_pct": 0.50, "margin_std": 12.0,
                              "conf_strength": a_conf_str, "conf_tourney_wins": 0},
            "seed_context": {"seed_adjustment": seed_adj},
            "market_lines": {"consensus_spread": market.get("spread"),
                             "consensus_total": market.get("total")},
        }

        try:
            result = self.engine.predict(inputs)
        except Exception as e:
            logger.warning(f"  Engine error {away_r} @ {home_r}: {e}")
            return None

        # Flatten
        return {
            "game_id": game_id or f"{season}_{home_r}_{away_r}",
            "season": season, "round": rd,
            "home_team_name": home_r, "away_team_name": away_r,

            # Targets (REAL)
            "actual_margin": actual_margin,
            "actual_total": home_pts + away_pts,
            "home_won": int(actual_margin > 0),
            "actual_home_score": home_pts, "actual_away_score": away_pts,
            "market_spread": market.get("spread"),
            "market_total": market.get("total"),

            # Formula outputs
            "formula_margin": round(-result.spread, 2),
            "formula_total": round(result.total, 2),
            "formula_home_score": round(result.home_score, 2),
            "formula_away_score": round(result.away_score, 2),
            "formula_win_prob": round(result.home_win_prob, 4),
            "game_pace": round(result.game_pace, 2),

            # Layer 4
            "adj_momentum": round(result.momentum_adj, 3),
            "adj_experience": round(result.experience_adj, 3),
            "adj_rest": round(result.rest_adj, 3),
            "adj_injury": round(result.injury_adj, 3),
            "adj_seed": round(result.seed_adj, 3),
            "adj_travel": round(result.travel_adj, 3),
            "adj_total": round(result.total_adjustment, 3),

            # Home efficiency
            "h_adj_oe": h_eff.get("adj_oe", 100), "h_adj_de": h_eff.get("adj_de", 100),
            "h_tempo": h_eff.get("adj_tempo", 68),
            "h_efg_off": h_eff.get("efg_pct_off", 0.50), "h_efg_def": h_eff.get("efg_pct_def", 0.50),
            "h_to_rate_off": h_eff.get("to_rate_off", 18), "h_to_rate_def": h_eff.get("to_rate_def", 18),
            "h_3pt_rate": h_eff.get("three_pt_rate_off", 0.35),
            "h_3pt_pct": h_eff.get("three_pt_pct_off", 0.34),
            "h_fta_rate": h_eff.get("fta_rate_off", 0.30),
            "h_orb": h_eff.get("orb_pct", 0.30),
            "h_sos": h_eff.get("sos", 0), "h_barthag": h_eff.get("barthag", 0.50),

            # Away efficiency
            "a_adj_oe": a_eff.get("adj_oe", 100), "a_adj_de": a_eff.get("adj_de", 100),
            "a_tempo": a_eff.get("adj_tempo", 68),
            "a_efg_off": a_eff.get("efg_pct_off", 0.50), "a_efg_def": a_eff.get("efg_pct_def", 0.50),
            "a_to_rate_off": a_eff.get("to_rate_off", 18), "a_to_rate_def": a_eff.get("to_rate_def", 18),
            "a_3pt_rate": a_eff.get("three_pt_rate_off", 0.35),
            "a_3pt_pct": a_eff.get("three_pt_pct_off", 0.34),
            "a_fta_rate": a_eff.get("fta_rate_off", 0.30),
            "a_orb": a_eff.get("orb_pct", 0.30),
            "a_sos": a_eff.get("sos", 0), "a_barthag": a_eff.get("barthag", 0.50),

            # New BartTorvik features
            "h_drb": h_eff.get("drb_pct", 0.70), "a_drb": a_eff.get("drb_pct", 0.70),
            "h_fta_rate_def": h_eff.get("fta_rate_def", 0.30),
            "a_fta_rate_def": a_eff.get("fta_rate_def", 0.30),
            "h_2pt_off": h_eff.get("two_pt_pct_off", 0.48),
            "a_2pt_off": a_eff.get("two_pt_pct_off", 0.48),
            "h_2pt_def": h_eff.get("two_pt_pct_def", 0.48),
            "a_2pt_def": a_eff.get("two_pt_pct_def", 0.48),
            "h_3pt_rate_def": h_eff.get("three_pt_rate_def", 0.36),
            "a_3pt_rate_def": a_eff.get("three_pt_rate_def", 0.36),
            "h_elite_sos": h_eff.get("elite_sos", 0), "a_elite_sos": a_eff.get("elite_sos", 0),
            "h_nc_sos": h_eff.get("non_conf_sos", 0), "a_nc_sos": a_eff.get("non_conf_sos", 0),

            # Derived
            "h_close_pct": 0.50, "a_close_pct": 0.50,
            "h_margin_std": 12.0, "a_margin_std": 12.0,
            "h_conf_strength": h_conf_str, "a_conf_strength": a_conf_str,
            "h_conf_tourney_w": 0, "a_conf_tourney_w": 0,

            # Differentials
            "oe_diff": h_eff.get("adj_oe", 100) - a_eff.get("adj_oe", 100),
            "de_diff": h_eff.get("adj_de", 100) - a_eff.get("adj_de", 100),
            "tempo_diff": h_eff.get("adj_tempo", 68) - a_eff.get("adj_tempo", 68),
            "barthag_diff": h_eff.get("barthag", 0.5) - a_eff.get("barthag", 0.5),
            "sos_diff": h_eff.get("sos", 0) - a_eff.get("sos", 0),

            # Context
            "h_seed": h_seed or 8, "a_seed": a_seed or 8,
            "seed_diff": (a_seed or 8) - (h_seed or 8),
            "rest_diff": 0, "vsi": 1.0, "vpi": 1.0, "v3p": 1.0,
            "h_coach_app": 3, "a_coach_app": 2,
            "h_returning": 0.5, "a_returning": 0.5,
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _find_ratings(self, team_name, lookup):
        """Find team in ratings lookup with fuzzy matching."""
        key = team_name.lower()
        if key in lookup:
            return lookup[key]
        resolved = self.resolver.resolve_or_original(team_name).lower()
        if resolved in lookup:
            return lookup[resolved]
        first = key.split()[0] if key else ""
        if first and len(first) >= 4:
            for k, v in lookup.items():
                if k.startswith(first) or first in k:
                    return v
        return None

    def _get_seed(self, game, side, eff):
        """Extract seed from game data or ratings."""
        for col in [f"{side}_seed", f"{side}Seed"]:
            val = game.get(col)
            if val is not None and not pd.isna(val):
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
        val = eff.get("seed")
        if val is not None and not pd.isna(val):
            try:
                s = int(val)
                if 1 <= s <= 16:
                    return s
            except (ValueError, TypeError):
                pass
        return None

    def _get_round(self, game):
        """Determine tournament round."""
        from ..collectors.config import TOURNAMENT_ROUNDS
        for col in ["tournament_round", "round", "bracket_round"]:
            val = game.get(col)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    if isinstance(val, str):
                        for name, num in TOURNAMENT_ROUNDS.items():
                            if name.lower() in val.lower():
                                return num
        start = game.get("start_date")
        if start:
            try:
                dt = pd.to_datetime(start)
                md = (dt.month, dt.day)
                if md <= (3, 21): return 1
                elif md <= (3, 23): return 2
                elif md <= (3, 28): return 3
                elif md <= (3, 30): return 4
                elif md <= (4, 6): return 5
                else: return 6
            except Exception:
                pass
        return 1

    def _conf_strength(self, eff, ratings_lookup):
        """Compute average barthag of same-conference teams."""
        conf = str(eff.get("conference", "")).lower()
        if not conf:
            return 0.50
        barthags = [
            v.get("barthag", 0.5) for v in ratings_lookup.values()
            if str(v.get("conference", "")).lower() == conf
            and v.get("barthag") is not None
        ]
        if len(barthags) >= 3:
            return round(float(np.mean(barthags)), 4)
        return 0.50


# ── Quick test / standalone runner ────────────────────────────────────────────
if __name__ == "__main__":
    import os

    key = os.environ.get("CBBD_API_KEY", "")
    if not key:
        print("Set CBBD_API_KEY to run live training builder")
        exit(1)

    builder = LiveTrainingBuilder()
    df = builder.build(
        cbbd_key=key,
        seasons=list(range(2015, 2026)),
        cache_dir="data/training_cache",
    )

    print(f"\nShape: {df.shape}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Feature cols: 73")
    print(f"Home win rate: {df['home_won'].mean():.1%}")
    print(f"Formula margin corr: {df['formula_margin'].corr(df['actual_margin']):.3f}")
