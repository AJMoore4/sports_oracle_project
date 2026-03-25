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
        seasons=range(2015, 2027),
        cache_dir="data/training_cache",
    )
    # → ~880 rows of real tournament data with formula features

API BUDGET:
    BartTorvik: ~1 request per uncached season (3.5s delay)
    CBBD: ~2 requests per uncached season (games + lines, 0.5s delay)
    Total time: ~2-3 minutes

CACHE:
    Historical seasons are stored in data/training_cache/live_training.sqlite.
    Finalized seasons are reused on later runs instead of re-pulling the APIs.
    Legacy training_YYYY.csv files are imported automatically if present.
"""

from __future__ import annotations
from collections import Counter
import os
import sqlite3
import logging
import numpy as np
import pandas as pd
from typing import Optional

from ..collectors.cbbd_collector import CBBDCollector
from ..collectors.barttorvik_collector import BartTorvik
from ..collectors.config import current_season
from ..engine.prediction_engine import PredictionEngine
from ..utils.seed_history import SeedHistory
from ..utils.team_resolver import get_resolver

logger = logging.getLogger("sports_oracle.backtest.live")

TRAINING_CACHE_SOURCE = "live_training_builder_v3"
TRAINING_START_SEASON = 2015
NCAA_TOURNAMENT_INCLUDE_TERMS = (
    "ncaa",
    "march madness",
    "national championship",
    "division i men s basketball championship",
    "division i men's basketball championship",
)


class LiveTrainingBuilder:
    """
    Builds ML training data from real historical tournament games.

    For each season (2015-current, excluding 2021 bubble when requested):
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
        cbbd_key: Optional[str] = None,
        seasons: Optional[list[int]] = None,
        cache_dir: Optional[str] = None,
        exclude_bubble_season: bool = False,
    ) -> pd.DataFrame:
        """
        Build training data from real historical tournament games.

        Args:
            cbbd_key: CBBD API key (or reads from CBBD_API_KEY env var)
            seasons: List of seasons to process (default: 2015 through the
                     current season)
            cache_dir: If provided, saves/loads historical seasons in a
                       SQLite cache so you only hit the APIs once per
                       finalized season. Legacy per-season CSVs are
                       imported automatically when present.
            exclude_bubble_season: Skip 2021 if you want to omit the
                       neutral-site bubble tournament from training.

        Returns:
            DataFrame with ~880 rows, each containing:
              - Actual outcomes (margin, total, winner)
              - Formula engine predictions (all 73 features)
              - Historical betting lines for comparison
        """
        if cbbd_key is None:
            cbbd_key = os.environ.get("CBBD_API_KEY", "")

        if seasons is None:
            seasons = list(range(TRAINING_START_SEASON, current_season() + 1))
        seasons = sorted({int(s) for s in seasons})
        if exclude_bubble_season:
            seasons = [s for s in seasons if s != 2021]

        cbbd = CBBDCollector(cbbd_key) if cbbd_key else None
        torvik = BartTorvik()
        cache_db_path = self._cache_db_path(cache_dir)

        all_dfs = []
        total_skipped = 0

        for season in seasons:
            logger.info(f"\n{'='*55}")
            logger.info(f"  SEASON {season}")
            logger.info(f"{'='*55}")

            # ── Check cache ───────────────────────────────────────────
            if cache_db_path and self._should_use_historical_cache(season):
                cached = self._load_cached_season(cache_db_path, season)
                if cached is not None and not cached.empty:
                    logger.info(
                        f"  Loaded {len(cached)} games from SQLite cache"
                    )
                    all_dfs.append(cached)
                    continue

                migrated = self._load_legacy_csv_cache(cache_dir, season)
                if migrated is not None and not migrated.empty:
                    self._save_cached_season(cache_db_path, season, migrated)
                    logger.info(
                        f"  Imported {len(migrated)} games from legacy CSV cache"
                    )
                    all_dfs.append(migrated)
                    continue

            if cbbd is None:
                logger.warning(
                    f"  No CBBD key available for uncached season {season}, skipping"
                )
                continue

            # ── 1. Tournament games from CBBD ─────────────────────────
            games = cbbd.get_games(season=season, season_type="postseason")
            if games.empty:
                logger.warning(f"  No postseason games for {season}")
                continue

            # Filter to NCAA tournament only
            games = self._filter_to_ncaa_tournament(games)

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

            ratings_lookup, ratings_entries = self._build_ratings_index(ratings)

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

            # ── 4. Compute national average AdjOE for Massey formula ──
            nat_avg_oe = 105.0
            if "adj_oe" in ratings.columns:
                avg = pd.to_numeric(ratings["adj_oe"], errors="coerce").mean()
                if not pd.isna(avg):
                    nat_avg_oe = float(avg)
                    logger.info(f"  National avg AdjOE: {nat_avg_oe:.1f}")

            # ── 5. Build rows ─────────────────────────────────────────
            rows = []
            skipped = 0
            skip_reasons: Counter[str] = Counter()
            for _, game in games.iterrows():
                row, skip_reason = self._build_row(
                    game,
                    ratings_lookup,
                    ratings_entries,
                    lines_lookup,
                    season,
                    nat_avg_oe,
                )
                if row is not None:
                    rows.append(row)
                else:
                    skipped += 1
                    skip_reasons[skip_reason or "unknown"] += 1
            total_skipped += skipped

            if rows:
                season_df = pd.DataFrame(rows)
                reason_str = ""
                if skip_reasons:
                    top_reasons = ", ".join(
                        f"{reason}={count}"
                        for reason, count in skip_reasons.most_common(3)
                    )
                    reason_str = f"; {top_reasons}"
                logger.info(
                    f"  Built {len(rows)} rows ({skipped} skipped{reason_str})"
                )

                if cache_db_path and self._should_use_historical_cache(season):
                    self._save_cached_season(cache_db_path, season, season_df)
                    logger.info(f"  Cached historical season → {cache_db_path}")

                all_dfs.append(season_df)

        if not all_dfs:
            logger.warning("No training data built")
            return pd.DataFrame()

        df = pd.concat(all_dfs, ignore_index=True)
        represented_seasons = sorted(
            int(s)
            for s in pd.to_numeric(df.get("season"), errors="coerce")
            .dropna()
            .unique()
        )
        season_counts = (
            df["season"].value_counts().sort_index().astype(int).to_dict()
            if "season" in df.columns
            else {}
        )
        logger.info(
            f"\n  LIVE TRAINING SET: {len(df)} games, "
            f"{len(represented_seasons)} represented seasons, {total_skipped} skipped"
        )
        if represented_seasons:
            coverage = ", ".join(
                f"{season}: {season_counts.get(season, 0)}"
                for season in represented_seasons
            )
            logger.info(f"  Season coverage: {coverage}")
        return df

    @staticmethod
    def _cache_db_path(cache_dir: Optional[str]) -> Optional[str]:
        if not cache_dir:
            return None
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "live_training.sqlite")

    @staticmethod
    def _should_use_historical_cache(season: int) -> bool:
        return season < current_season()

    def _load_cached_season(
        self,
        cache_db_path: str,
        season: int,
    ) -> Optional[pd.DataFrame]:
        if not cache_db_path or not os.path.exists(cache_db_path):
            return None

        try:
            with sqlite3.connect(cache_db_path) as conn:
                self._ensure_cache_schema(conn)
                if not self._table_exists(conn, "live_training_rows"):
                    return None
                meta = conn.execute(
                    """
                    SELECT source
                    FROM live_training_cache_meta
                    WHERE season = ?
                    """,
                    (season,),
                ).fetchone()
                if meta is None:
                    return None
                if meta[0] != TRAINING_CACHE_SOURCE:
                    logger.info(
                        f"  Ignoring stale cache for {season} ({meta[0]})"
                    )
                    return None
                cached = pd.read_sql_query(
                    "SELECT * FROM live_training_rows WHERE season = ?",
                    conn,
                    params=(season,),
                )
        except Exception as e:
            logger.warning(f"  SQLite cache read failed for {season}: {e}")
            return None

        return cached if not cached.empty else None

    def _save_cached_season(
        self,
        cache_db_path: str,
        season: int,
        season_df: pd.DataFrame,
    ) -> None:
        if season_df.empty:
            return

        try:
            with sqlite3.connect(cache_db_path) as conn:
                self._ensure_cache_schema(conn)
                if self._table_exists(conn, "live_training_rows"):
                    conn.execute(
                        "DELETE FROM live_training_rows WHERE season = ?",
                        (season,),
                    )
                season_df.to_sql(
                    "live_training_rows",
                    conn,
                    if_exists="append",
                    index=False,
                )
                conn.execute(
                    """
                    INSERT INTO live_training_cache_meta (
                        season,
                        row_count,
                        cached_at_utc,
                        source
                    ) VALUES (?, ?, CURRENT_TIMESTAMP, ?)
                    ON CONFLICT(season) DO UPDATE SET
                        row_count = excluded.row_count,
                        cached_at_utc = excluded.cached_at_utc,
                        source = excluded.source
                    """,
                    (season, len(season_df), TRAINING_CACHE_SOURCE),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"  SQLite cache write failed for {season}: {e}")

    @staticmethod
    def _load_legacy_csv_cache(
        cache_dir: Optional[str],
        season: int,
    ) -> Optional[pd.DataFrame]:
        if not cache_dir:
            return None
        cache_path = os.path.join(cache_dir, f"training_{season}.csv")
        if not os.path.exists(cache_path):
            return None
        try:
            cached = pd.read_csv(cache_path)
            if LiveTrainingBuilder._is_compatible_legacy_cache(cached):
                return cached
            logger.info(
                f"  Ignoring stale legacy CSV cache for {season}: {cache_path}"
            )
            return None
        except Exception as e:
            logger.warning(f"  Legacy CSV cache read failed for {season}: {e}")
            return None

    @staticmethod
    def _is_compatible_legacy_cache(df: pd.DataFrame) -> bool:
        """Reject broad postseason CSV caches from the pre-v2 builder."""
        if df.empty:
            return False
        if len(df) > 80:
            return False
        if {"h_seed", "a_seed"}.issubset(df.columns):
            home = pd.to_numeric(df["h_seed"], errors="coerce")
            away = pd.to_numeric(df["a_seed"], errors="coerce")
            valid_share = (
                home.between(1, 16, inclusive="both")
                & away.between(1, 16, inclusive="both")
            ).mean()
            return bool(valid_share >= 0.9)
        return True

    @staticmethod
    def _ensure_cache_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS live_training_cache_meta (
                season INTEGER PRIMARY KEY,
                row_count INTEGER NOT NULL,
                cached_at_utc TEXT NOT NULL,
                source TEXT NOT NULL
            )
            """
        )

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    # ── Row builder ───────────────────────────────────────────────────────

    def _build_row(
        self,
        game,
        ratings_lookup,
        ratings_entries,
        lines_lookup,
        season,
        national_avg_oe=105.0,
    ):
        """Build one training row from a real CBBD tournament game."""
        home_team = str(game.get("home_team", "")).strip()
        away_team = str(game.get("away_team", "")).strip()
        if not home_team or not away_team:
            return None, "missing_team_name"

        home_r = self.resolver.resolve_or_original(home_team)
        away_r = self.resolver.resolve_or_original(away_team)

        h_eff = self._find_ratings(home_r, ratings_lookup, ratings_entries)
        a_eff = self._find_ratings(away_r, ratings_lookup, ratings_entries)
        if not h_eff or not a_eff:
            return None, "ratings_not_found"

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
            seed_adj = self.seed_history.get_seed_adjustment(
                h_seed,
                a_seed,
                reference_year=season,
            )

        # Conference strength from ratings
        h_conf_str = self._conf_strength(h_eff, ratings_lookup)
        a_conf_str = self._conf_strength(a_eff, ratings_lookup)

        # Build formula engine inputs
        inputs = {
            "home_team": home_r, "away_team": away_r,
            "season": season, "tournament_round": rd,
            "home_seed": h_seed, "away_seed": a_seed,
            "national_avg_oe": national_avg_oe,
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
            "home_extended": {"close_game_pct": 0.50, "close_game_rate": 0.25,
                              "close_game_games": 0, "margin_std": 12.0,
                              "conf_strength": h_conf_str, "conf_tourney_wins": 0},
            "away_extended": {"close_game_pct": 0.50, "close_game_rate": 0.25,
                              "close_game_games": 0, "margin_std": 12.0,
                              "conf_strength": a_conf_str, "conf_tourney_wins": 0},
            "seed_context": {"seed_adjustment": seed_adj},
            "market_lines": {"consensus_spread": market.get("spread"),
                             "consensus_total": market.get("total")},
        }

        try:
            result = self.engine.predict(inputs)
        except Exception as e:
            logger.warning(f"  Engine error {away_r} @ {home_r}: {e}")
            return None, "engine_error"

        market_total_value = pd.to_numeric(market.get("total"), errors="coerce")
        market_total = (
            float(market_total_value)
            if pd.notna(market_total_value)
            else None
        )

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
            "market_total": market_total,

            # Formula outputs
            "formula_margin": round(-result.spread, 2),
            "formula_total": round(result.total, 2),
            "raw_total": round(result.raw_total, 2),
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
            "adj_total_points": round(result.total_points_adjustment, 3),

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
            "h_close_rate": 0.25, "a_close_rate": 0.25,
            "h_close_games": 0, "a_close_games": 0,
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
            "venue_sample_size": 0,
            "h_coach_app": 3, "a_coach_app": 2,
            "h_returning": 0.5, "a_returning": 0.5,
            "market_total_available": 1 if market_total is not None else 0,
            "market_total_delta": (
                round(result.total - market_total, 3)
                if market_total is not None
                else 0.0
            ),
        }, None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_ratings_index(self, ratings: pd.DataFrame):
        """Create a multi-key ratings index for historical team matching."""
        lookup = {}
        entries = []

        for _, row in ratings.iterrows():
            data = row.to_dict()
            team_name = str(row.get("team", "")).strip()
            if not team_name:
                continue

            for key in self._candidate_team_keys(team_name):
                lookup[key] = data

            entries.append(
                {
                    "row": data,
                    "tokens": self._team_tokens(team_name),
                }
            )

        return lookup, entries

    def _find_ratings(self, team_name, lookup, entries):
        """Find team in ratings lookup with normalized and token-overlap matching."""
        for key in self._candidate_team_keys(team_name):
            if key in lookup:
                return lookup[key]

        tokens = self._team_tokens(team_name)
        if not tokens:
            return None

        best_match = None
        best_score = 0.0
        for entry in entries:
            candidate_tokens = entry["tokens"]
            overlap = len(tokens & candidate_tokens)
            if overlap == 0:
                continue
            union = len(tokens | candidate_tokens)
            score = overlap / union if union else 0.0
            if overlap >= min(2, len(tokens)) and score > best_score:
                best_match = entry["row"]
                best_score = score

        if best_score >= 0.5:
            return best_match
        return None

    @staticmethod
    def _normalize_team_name(name: str) -> str:
        text = str(name or "").lower().strip()
        text = text.replace("&", " and ")
        text = text.replace("(", " ").replace(")", " ")
        text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
        tokens = []
        for token in text.split():
            if token == "saint":
                tokens.append("st")
            elif token == "mt":
                tokens.append("mount")
            else:
                tokens.append(token)
        return " ".join(tokens)

    def _candidate_team_keys(self, team_name: str) -> set[str]:
        keys = set()
        raw_name = str(team_name or "").strip()
        if not raw_name:
            return keys

        candidates = {raw_name}
        resolved = self.resolver.resolve(raw_name)
        if resolved:
            candidates.add(resolved)

        for candidate in list(candidates):
            normalized = self._normalize_team_name(candidate)
            if not normalized:
                continue
            keys.add(normalized)
            words = normalized.split()
            keys.add(" ".join("st" if word == "state" else word for word in words))
            keys.add(" ".join("saint" if word == "st" else word for word in words))
            keys.add(" ".join("state" if word == "st" else word for word in words))
            keys.add(" ".join(word for word in words if word != "and"))
            if len(words) > 1:
                keys.add("".join(words))

        return {key.strip() for key in keys if key.strip()}

    def _team_tokens(self, team_name: str) -> set[str]:
        normalized = self._normalize_team_name(team_name)
        stopwords = {"the", "of", "at", "and", "university"}
        return {
            token for token in normalized.split()
            if token and token not in stopwords
        }

    def _coerce_seed_series(self, games: pd.DataFrame, columns: list[str]) -> pd.Series:
        for col in columns:
            if col in games.columns:
                return pd.to_numeric(games[col], errors="coerce")
        return pd.Series(np.nan, index=games.index, dtype=float)

    def _filter_to_ncaa_tournament(self, games: pd.DataFrame) -> pd.DataFrame:
        """Keep only NCAA tournament games from the postseason feed."""
        if games.empty:
            return games

        filtered = games.copy()
        home_seeds = self._coerce_seed_series(filtered, ["home_seed", "homeSeed"])
        away_seeds = self._coerce_seed_series(filtered, ["away_seed", "awaySeed"])
        seeded_mask = (
            home_seeds.between(1, 16, inclusive="both")
            & away_seeds.between(1, 16, inclusive="both")
        )

        if "tournament" not in filtered.columns:
            if not seeded_mask.any():
                logger.warning(
                    "  Postseason feed missing tournament labels and NCAA-style seeds"
                )
            return filtered[seeded_mask].copy() if seeded_mask.any() else filtered

        tournament_name = (
            filtered["tournament"]
            .fillna("")
            .astype(str)
            .map(self._normalize_team_name)
        )
        explicit_ncaa = tournament_name.map(self._is_ncaa_tournament_label)

        if explicit_ncaa.any():
            mask = explicit_ncaa | (tournament_name.eq("") & seeded_mask)
        elif seeded_mask.any():
            logger.info(
                "  No explicit NCAA label found; falling back to seeded postseason games"
            )
            mask = seeded_mask
        else:
            logger.warning(
                "  Could not confidently isolate NCAA tournament rows from postseason feed"
            )
            mask = filtered["tournament"].notna()

        return filtered[mask].copy()

    @staticmethod
    def _is_ncaa_tournament_label(label: str) -> bool:
        normalized = " ".join(str(label or "").lower().split())
        return any(term in normalized for term in NCAA_TOURNAMENT_INCLUDE_TERMS)

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
