"""
sports_oracle/collectors/pipeline.py

Master data pipeline — orchestrates all collectors
and assembles model-ready input DataFrames for each
formula layer.
"""

from __future__ import annotations
import hashlib
import math
import os
import re
import pandas as pd
from typing import Any, Optional
from datetime import datetime, timedelta

from .config import (
    current_season, BUBBLE_SEASONS,
    SHOT_CLOCK_CHANGE_SEASON, logger,
)
from .cbbd_collector import CBBDCollector
from .barttorvik_collector import BartTorvik
from .espn_collector import ESPNCollector
from .odds_collector import OddsCollector
from .ncaa_collector import NCAACollector
from ..utils.team_resolver import get_resolver
from ..utils.data_validator import DataValidator
from ..utils.geo import GeoLookup
from ..utils.seed_history import SeedHistory

VENUE_INDEX_SHRINK_SAMPLES = 12.0
VENUE_HISTORY_CACHE_VERSION = "v2"
VENUE_HISTORY_CACHE_TTL_HOURS = 24.0
PIPELINE_DISK_CACHE_VERSION = "v1"
PIPELINE_LIVE_CACHE_TTL_HOURS = 8.0


class DataPipeline:
    """
    Orchestrates all data collectors and assembles
    model-ready inputs for the prediction engine.
    """

    def __init__(
        self,
        cbbd_key: str = "",
        odds_key: str = "",
        season: Optional[int] = None,
    ):
        self.season = season or current_season()

        # Initialize collectors
        self.espn = ESPNCollector()
        self.torvik = BartTorvik()
        self.odds = OddsCollector(api_key=odds_key)
        self.ncaa = NCAACollector()

        # Optional collectors
        self.cbbd = None
        if cbbd_key:
            try:
                self.cbbd = CBBDCollector(cbbd_key)
            except ValueError as e:
                logger.warning(f"CBBD not available: {e}")

        # Utilities
        self.resolver = get_resolver()
        self.validator = DataValidator()
        self.geo = GeoLookup()
        self.seed_history = SeedHistory()

        # Session cache
        self._cache = {}
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self._disk_cache_dir = os.path.join(
            project_root,
            "data",
            "pipeline_cache",
            PIPELINE_DISK_CACHE_VERSION,
        )
        self._venue_cache_dir = os.path.join(project_root, "data", "venue_cache")
        os.makedirs(self._disk_cache_dir, exist_ok=True)
        os.makedirs(self._venue_cache_dir, exist_ok=True)
        self._venue_history_cache_path = os.path.join(
            self._venue_cache_dir,
            f"venue_tournament_history_{VENUE_HISTORY_CACHE_VERSION}.csv",
        )

    # ── Scoreboard / Market Data ─────────────────────────────────────────────

    def get_scoreboard_with_historical_lines(
        self,
        date: Optional[str] = None,
        groups: str = "50",
    ) -> pd.DataFrame:
        """
        Fetch the ESPN scoreboard, then backfill final games with
        market lines while conserving paid-API usage.

        Priority:
          1. ESPN lines already on the scoreboard
          2. CBBD lines when available
          3. Odds API only for rows still missing a line
        """
        lookup_date = self._scoreboard_date_to_iso(date)
        scoreboard_cache_key = f"scoreboard_enriched_{lookup_date or 'today'}_{groups}"
        cache_ttl = self._cache_ttl_for_lookup_date(lookup_date)
        cached = self._read_disk_cache(
            category="scoreboards",
            key=scoreboard_cache_key,
            ttl=cache_ttl,
        )
        if isinstance(cached, pd.DataFrame):
            if self._should_refresh_cached_scoreboard(cached, lookup_date):
                logger.info(
                    "Refreshing stale scoreboard cache for %s",
                    lookup_date or "today",
                )
            else:
                logger.info(
                    "Loaded enriched scoreboard cache for %s",
                    lookup_date or "today",
                )
                return cached.copy()

        scoreboard = self.espn.get_scoreboard(date=date, groups=groups)
        if scoreboard.empty:
            return scoreboard
        enriched = self.enrich_scoreboard_with_market_lines(
            scoreboard,
            scoreboard_date=date,
        )
        self._write_disk_cache(
            category="scoreboards",
            key=scoreboard_cache_key,
            value=enriched,
            persist_empty=not enriched.empty,
        )
        return enriched

    def enrich_scoreboard_with_market_lines(
        self,
        scoreboard: pd.DataFrame,
        scoreboard_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Keep ESPN lines, then fill missing rows from CBBD first and
        use Odds API only as a final fallback.
        """
        if scoreboard.empty:
            return scoreboard

        enriched = scoreboard.copy()

        for source_col, backup_col in (
            ("betting_spread", "espn_betting_spread"),
            ("over_under", "espn_over_under"),
            ("odds_detail", "espn_odds_detail"),
        ):
            if source_col in enriched.columns and backup_col not in enriched.columns:
                enriched[backup_col] = enriched[source_col]

        enriched = self.enrich_scoreboard_with_cbbd_lines(
            enriched,
            scoreboard_date=scoreboard_date,
        )

        if self.odds and self.odds.is_configured:
            enriched = self._enrich_scoreboard_with_odds_api(
                enriched,
                scoreboard_date=scoreboard_date,
            )

        return enriched

    def _enrich_scoreboard_with_odds_api(
        self,
        scoreboard: pd.DataFrame,
        scoreboard_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply Odds API current lines for upcoming games and historical
        lines for completed games, but only for rows still missing a line.
        """
        if scoreboard.empty or "status" not in scoreboard.columns:
            return scoreboard

        enriched = scoreboard.copy()
        status_text = enriched["status"].fillna("").astype(str).str.lower()
        final_mask = status_text.str.contains("final|post", regex=True)
        missing_mask = ~enriched.apply(self._row_has_market_line, axis=1)
        pending_current_mask = (~final_mask) & missing_mask
        pending_final_mask = final_mask & missing_mask

        current_odds_feed = []
        if pending_current_mask.any():
            current_from, current_to = self._odds_feed_time_window(
                enriched.loc[pending_current_mask],
                scoreboard_date=scoreboard_date,
            )
            current_odds_feed = self.odds.get_current_odds_for_day(
                cache_label=self._odds_cache_label(scoreboard_date, enriched),
                commence_time_from=current_from,
                commence_time_to=current_to,
            )

        historical_odds_feed = []
        if pending_final_mask.any():
            historical_from, historical_to = self._odds_feed_time_window(
                enriched.loc[pending_final_mask],
                scoreboard_date=scoreboard_date,
            )
            snapshot_time = self._historical_snapshot_for_rows(
                enriched.loc[pending_final_mask],
                scoreboard_date=scoreboard_date,
            )
            if snapshot_time:
                historical_odds_feed = self.odds.get_historical_odds_for_day(
                    cache_label=self._odds_cache_label(scoreboard_date, enriched),
                    snapshot_time=snapshot_time,
                    commence_time_from=historical_from,
                    commence_time_to=historical_to,
                )

        for idx, row in enriched.iterrows():
            if self._row_has_market_line(row):
                continue

            home_team = str(row.get("home_team", "")).strip()
            away_team = str(row.get("away_team", "")).strip()
            if not home_team or not away_team:
                continue

            is_final = bool(final_mask.at[idx])
            if is_final:
                odds = self.odds.get_game_odds_from_feed(
                    all_odds=historical_odds_feed,
                    home_team=home_team,
                    away_team=away_team,
                )
                summary = self.odds._build_consensus_from_odds(odds)
                source = "odds_api_historical"
            else:
                odds = self.odds.get_game_odds_from_feed(
                    all_odds=current_odds_feed,
                    home_team=home_team,
                    away_team=away_team,
                )
                summary = self.odds._build_consensus_from_odds(odds)
                source = "odds_api_current"

            if not summary:
                continue

            spread = summary.get("consensus_spread")
            total = summary.get("consensus_total")
            if spread is None and total is None:
                continue

            self._apply_market_summary(
                enriched=enriched,
                idx=idx,
                spread=spread,
                total=total,
                provider_count=max(
                    summary.get("spread_bookmaker_count", 0),
                    summary.get("total_bookmaker_count", 0),
                ),
                source=source,
                event_id=summary.get("event_id"),
                snapshot_time=summary.get("snapshot_time"),
                historical=is_final,
            )

        return enriched

    def enrich_scoreboard_with_cbbd_lines(
        self,
        scoreboard: pd.DataFrame,
        scoreboard_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Merge CBBD spreads/totals onto scoreboard rows before using
        Odds API. This primarily helps completed games, but it also
        allows current-season CBBD lines to fill blanks for upcoming rows.
        """
        if scoreboard.empty or not self.cbbd:
            return scoreboard

        enriched = scoreboard.copy()

        enriched["cbbd_lookup_date"] = enriched["date"].apply(
            self._scoreboard_date_to_iso
        )
        requested_date = self._scoreboard_date_to_iso(scoreboard_date)
        season_lines = self._get_prepared_cbbd_season_lines()
        if season_lines.empty:
            return enriched

        for idx, row in enriched.iterrows():
            if self._row_has_market_line(enriched.loc[idx]):
                continue

            matched_lines = self._match_cbbd_lines_for_row(
                row,
                season_lines,
                requested_date,
            )
            if matched_lines.empty:
                continue

            summary = self.cbbd.summarize_lines(matched_lines)
            if summary["spread"] is None and summary["total"] is None:
                continue

            game_id = None
            if "game_id" in matched_lines.columns:
                game_ids = matched_lines["game_id"].dropna()
                if not game_ids.empty:
                    game_id = game_ids.iloc[0]

            is_final = self._row_is_final(row)

            self._apply_market_summary(
                enriched=enriched,
                idx=idx,
                spread=summary["spread"],
                total=summary["total"],
                provider_count=summary["provider_count"],
                source="cbbd_historical" if is_final else "cbbd_current",
                event_id=game_id,
                historical=is_final,
            )

        return enriched

    # ── Layer 1: Team Efficiency Profiles ────────────────────────────────────

    def get_team_efficiency(
        self,
        team: str,
        season: Optional[int] = None,
    ) -> dict:
        season = season or self.season
        cache_key = f"efficiency_{team}_{season}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        ratings = self._get_or_cache(
            f"torvik_ratings_{season}",
            lambda: self._get_torvik_ratings(season),
        )

        if ratings.empty:
            return {}

        mask = ratings["team"].str.lower() == team.lower()
        match = ratings[mask]

        if match.empty:
            mask = ratings["team"].str.lower().str.contains(
                team.lower().split()[0]
            )
            match = ratings[mask]

        if match.empty:
            return {}

        row = match.iloc[0].to_dict()

        result = {
            "team":             team,
            "season":           season,
            "adj_oe":           float(row.get("adj_oe") or 100),
            "adj_de":           float(row.get("adj_de") or 100),
            "adj_tempo":        float(row.get("adj_tempo") or 68),
            "efg_pct_off":      float(row.get("efg_pct_off") or 0.50),
            "efg_pct_def":      float(row.get("efg_pct_def") or 0.50),
            "to_rate_off":      float(row.get("to_rate_off") or 18),
            "to_rate_def":      float(row.get("to_rate_def") or 18),
            "three_pt_rate_off":float(row.get("three_pt_rate_off") or 0.35),
            "three_pt_rate_def":float(row.get("three_pt_rate_def") or 0.35),
            "three_pt_pct_off": float(row.get("three_pt_pct_off") or 0.33),
            "three_pt_pct_def": float(row.get("three_pt_pct_def") or 0.33),
            "fta_rate_off":     float(row.get("fta_rate_off") or 0.30),
            "fta_rate_def":     float(row.get("fta_rate_def") or 0.30),
            "orb_pct":          float(row.get("orb_pct") or 0.30),
            "drb_pct":          float(row.get("drb_pct") or 0.70),
            "sos":              float(row.get("sos") or 0),
            "rank":             int(row.get("rank") or 200),
            "barthag":          float(row.get("barthag") or 0.5),
            # ── New fields (already in BartTorvik CSV, now wired through) ──
            "two_pt_pct_off":   float(row.get("two_pt_pct_off") or 0.48),
            "two_pt_pct_def":   float(row.get("two_pt_pct_def") or 0.48),
            "three_pt_rate_def":float(row.get("three_pt_rate_def") or 0.35),
            "elite_sos":        float(row.get("elite_sos") or 0),
            "non_conf_sos":     float(row.get("non_conf_sos") or 0),
            "conference":       str(row.get("conference") or ""),
        }

        self._cache[cache_key] = result
        return result

    # ── Layer 2: Venue Profile ────────────────────────────────────────────────

    def get_venue_profile(
        self,
        venue_id: Optional[int] = None,
        venue_name: Optional[str] = None,
        seasons: Optional[list[int]] = None,
        is_ncaa_tournament: bool = False,
    ) -> dict:
        seasons = seasons or list(range(2010, self.season + 1))
        seasons = [s for s in seasons if s not in BUBBLE_SEASONS]

        nat_avgs = self._get_or_cache(
            "national_averages",
            lambda: self._get_national_averages_cached(seasons),
        )

        profile = {
            "venue_id":    venue_id,
            "venue_name":  venue_name or "Unknown",
            "vsi":         1.00,
            "vpi":         1.00,
            "v3p":         1.00,
            "sample_size": 0,
            "rounds": {
                1: 1.00, 2: 1.00, 3: 0.97,
                4: 0.94, 5: 0.96, 6: 0.95,
            },
        }

        if not is_ncaa_tournament or not self.cbbd or (not venue_id and not venue_name):
            return profile

        venue_history = self._get_full_venue_history()
        if not venue_history.empty and "season" in venue_history.columns:
            season_values = pd.to_numeric(venue_history["season"], errors="coerce")
            venue_history = venue_history[season_values.isin(seasons)].copy()

        if venue_history.empty:
            return profile

        if venue_id and "venue_id" in venue_history.columns:
            venue_games = venue_history[
                venue_history["venue_id"] == venue_id
            ]
        elif venue_name and "venue_name" in venue_history.columns:
            venue_games = venue_history[
                venue_history["venue_name"].str.contains(
                    venue_name, case=False, na=False
                )
            ]
        else:
            return profile

        if venue_games.empty:
            return profile

        profile = self._compute_venue_indices(
            venue_games, nat_avgs, profile, seasons
        )
        return profile

    def _compute_venue_indices(
        self,
        venue_games: pd.DataFrame,
        nat_avgs: pd.DataFrame,
        profile: dict,
        seasons: list[int],
    ) -> dict:
        LAMBDA = 0.15
        oldest = min(seasons)

        weighted_scoring_ratios = []
        weighted_pace_ratios = []
        weighted_3pt_ratios = []
        weights = []

        nat_lookup = {}
        if not nat_avgs.empty:
            for _, row in nat_avgs.iterrows():
                nat_lookup[int(row["season"])] = row.to_dict()

        for _, game in venue_games.iterrows():
            season = int(game.get("season", 0))
            if season == 0:
                continue

            nat = nat_lookup.get(season, {})
            nat_total = nat.get("nat_avg_total_pts", 140)
            nat_tempo = nat.get("nat_avg_tempo", 68)
            nat_3pt = nat.get("nat_avg_3pt_pct", 0.33)

            if not nat_total or nat_total == 0:
                continue

            decay_weight = math.exp(LAMBDA * (season - oldest))
            if season in BUBBLE_SEASONS:
                decay_weight *= 0.05

            total_pts = game.get("total_points")
            if not pd.isna(total_pts) and total_pts > 0:
                game_ratio = float(total_pts) / float(nat_total)
                weighted_scoring_ratios.append(game_ratio * decay_weight)

            game_tempo = game.get("tempo") or game.get("possessions")
            if game_tempo and not pd.isna(game_tempo) and nat_tempo > 0:
                pace_ratio = float(game_tempo) / float(nat_tempo)
                weighted_pace_ratios.append(pace_ratio * decay_weight)
            elif not pd.isna(total_pts) and total_pts > 0 and nat_total > 0:
                pace_proxy = 1.0 + (float(total_pts) / float(nat_total) - 1.0) * 0.6
                weighted_pace_ratios.append(pace_proxy * decay_weight)

            game_3pt = game.get("three_pt_pct") or game.get("fg3_pct")
            if game_3pt and not pd.isna(game_3pt) and nat_3pt > 0:
                three_ratio = float(game_3pt) / float(nat_3pt)
                weighted_3pt_ratios.append(three_ratio * decay_weight)

            weights.append(decay_weight)

        if weights:
            n = len(weights)
            if weighted_scoring_ratios:
                w_sum = sum(w for w, _ in zip(weights, weighted_scoring_ratios))
                if w_sum > 0:
                    raw_vsi = sum(weighted_scoring_ratios) / w_sum
                    profile["vsi"] = self._shrink_venue_index(raw_vsi, n)

            if weighted_pace_ratios:
                w_sum = sum(w for w, _ in zip(weights, weighted_pace_ratios))
                if w_sum > 0:
                    raw_vpi = sum(weighted_pace_ratios) / w_sum
                    profile["vpi"] = self._shrink_venue_index(raw_vpi, n)

            if weighted_3pt_ratios:
                w_sum = sum(w for w, _ in zip(weights, weighted_3pt_ratios))
                if w_sum > 0:
                    raw_v3p = sum(weighted_3pt_ratios) / w_sum
                    profile["v3p"] = self._shrink_venue_index(raw_v3p, n)

            profile["sample_size"] = n
        return profile

    def _get_full_venue_history(self) -> pd.DataFrame:
        cache_key = "venue_tournament_history_full"
        return self._get_or_cache(
            cache_key,
            self._load_or_build_venue_history_cache,
        )

    def _load_or_build_venue_history_cache(self) -> pd.DataFrame:
        cached = self._read_venue_history_cache()
        if cached is not None:
            return cached

        seasons = [s for s in range(2010, self.season + 1) if s not in BUBBLE_SEASONS]
        logger.info(
            "Building venue history cache from CBBD (%s seasons)...",
            len(seasons),
        )
        venue_history = self.cbbd.build_venue_game_history(seasons=seasons)
        if venue_history.empty:
            return venue_history

        self._write_venue_history_cache(venue_history)
        return venue_history

    def _read_venue_history_cache(self) -> Optional[pd.DataFrame]:
        path = self._venue_history_cache_path
        if not os.path.exists(path):
            return None
        if self._venue_history_cache_is_stale(path):
            logger.info("Venue history cache is stale, refreshing from CBBD...")
            return None

        try:
            cached = pd.read_csv(path)
        except Exception as exc:
            logger.warning(f"Could not read venue history cache {path}: {exc}")
            return None

        required_cols = {"venue_id", "season", "total_points"}
        if cached.empty or not required_cols.issubset(cached.columns):
            logger.info("Venue history cache missing required columns, rebuilding...")
            return None

        logger.info(f"Loaded venue history cache: {path}")
        return cached

    def _write_venue_history_cache(self, venue_history: pd.DataFrame) -> None:
        path = self._venue_history_cache_path
        try:
            venue_history.to_csv(path, index=False)
            logger.info(f"Saved venue history cache: {path}")
        except Exception as exc:
            logger.warning(f"Could not write venue history cache {path}: {exc}")

    def _venue_history_cache_is_stale(self, path: str) -> bool:
        current_season_in_cache = self.season == current_season()
        if not current_season_in_cache:
            return False

        try:
            modified = datetime.fromtimestamp(os.path.getmtime(path))
        except OSError:
            return True

        age = datetime.now() - modified
        return age > timedelta(hours=VENUE_HISTORY_CACHE_TTL_HOURS)

    @staticmethod
    def _shrink_venue_index(
        raw_value: float,
        sample_size: int,
        prior: float = 1.0,
    ) -> float:
        if raw_value is None or pd.isna(raw_value):
            return prior
        weight = float(sample_size) / (float(sample_size) + VENUE_INDEX_SHRINK_SAMPLES)
        return prior + (float(raw_value) - prior) * weight

    # ── Layer 4: Momentum ─────────────────────────────────────────────────────

    def get_momentum_data(
        self,
        team: str,
        season: Optional[int] = None,
        n_games: int = 10,
    ) -> dict:
        season = season or self.season
        cache_key = f"momentum_{team}_{season}_{n_games}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        full_game_log = self._get_team_game_log(team, season)
        if not full_game_log.empty and "date" in full_game_log.columns:
            game_log = full_game_log.tail(n_games).sort_values(
                "date",
                ascending=False,
            ).reset_index(drop=True)
        else:
            game_log = full_game_log.tail(n_games).copy() if isinstance(full_game_log, pd.DataFrame) else pd.DataFrame()

        season_eff = self.get_team_efficiency(team, season)
        conf_tourney = self._get_conf_tourney_games(team)

        margins = []
        if not game_log.empty:
            for col in ["margin", "pts_diff", "point_diff", "score_diff"]:
                if col in game_log.columns:
                    margins = pd.to_numeric(game_log[col], errors="coerce").dropna().tolist()
                    break

        result = {
            "team":            team,
            "season":          season,
            "recent_margins":  margins,
            "game_log":        game_log,
            "season_adj_oe":   season_eff.get("adj_oe", 100),
            "season_adj_de":   season_eff.get("adj_de", 100),
            "season_tempo":    season_eff.get("adj_tempo", 68),
            "season_efg":      season_eff.get("efg_pct_off", 0.50),
            "season_to_rate":  season_eff.get("to_rate_off", 18),
            "conf_tourney_games": conf_tourney,
        }
        self._cache[cache_key] = result
        return result

    def _get_conf_tourney_games(self, team: str) -> pd.DataFrame:
        cache_key = f"conf_tourney_{team}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        recent = self._get_or_cache(
            "conf_tourney_all_games",
            lambda: self._get_conf_tourney_games_feed(days_window=14),
        )
        if recent.empty:
            return pd.DataFrame()

        team_lower = team.lower()
        mask = (
            recent.get("home_team", pd.Series(dtype=str)).str.lower().str.contains(team_lower, na=False, regex=False)
            | recent.get("away_team", pd.Series(dtype=str)).str.lower().str.contains(team_lower, na=False, regex=False)
        )
        result = recent[mask].copy()
        self._cache[cache_key] = result
        return result

    # ── Extended Team Stats (close games, variance, conference) ───────────

    def get_extended_team_stats(
        self,
        team: str,
        season: Optional[int] = None,
    ) -> dict:
        """
        Compute derived stats not available directly from BartTorvik:
          - close_game_pct: win rate in games decided by ≤6 points
          - close_game_rate: share of games decided by ≤6 points
          - close_game_games: number of games decided by ≤6 points
          - margin_std: standard deviation of scoring margins (consistency)
          - conf_strength: average barthag of teams in the same conference
          - conf_tourney_wins: conference tournament wins this year

        These require the team's game log + full ratings table.
        All are cached per team per session.
        """
        season = season or self.season
        cache_key = f"extended_{team}_{season}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = {
            "close_game_pct": 0.50,    # default: coin flip
            "close_game_rate": 0.25,   # default: modest exposure to close games
            "close_game_games": 0,
            "margin_std": 12.0,        # national avg ~12 pts
            "conf_strength": 0.50,     # average barthag
            "conf_tourney_wins": 0,
        }

        # ── Close game record + margin variance ──
        # Use BartTorvik game log (already fetched for momentum)
        game_log = self._get_or_cache(
            f"game_log_{team}_{season}",
            lambda: self._get_team_game_log(team, season),
        )

        if not game_log.empty:
            # Find the margin column
            margin_col = None
            for col in ["margin", "pts_diff", "point_diff", "score_diff"]:
                if col in game_log.columns:
                    margin_col = col
                    break

            if margin_col:
                margins = pd.to_numeric(
                    game_log[margin_col], errors="coerce"
                ).dropna()

                if len(margins) >= 5:
                    # Scoring variance (consistency)
                    result["margin_std"] = round(float(margins.std()), 2)

                    # Close game pressure profile: games decided by ≤6 pts
                    close_mask = margins.abs() <= 6
                    close_games = margins[close_mask]
                    result["close_game_games"] = int(len(close_games))
                    result["close_game_rate"] = round(
                        float(len(close_games) / len(margins)),
                        3,
                    )
                    if len(close_games) >= 2:
                        close_wins = (close_games > 0).sum()
                        result["close_game_pct"] = round(
                            float(close_wins / len(close_games)), 3
                        )

        # ── Conference strength ──
        # Average barthag of all teams in the same conference
        team_eff = self.get_team_efficiency(team, season)
        conf = team_eff.get("conference", "")

        if conf:
            ratings = self._get_or_cache(
                f"torvik_ratings_{season}",
                lambda: self._get_torvik_ratings(season),
            )

            if not ratings.empty and "conference" in ratings.columns:
                conf_mask = ratings["conference"].str.lower() == conf.lower()
                conf_teams = ratings[conf_mask]

                if not conf_teams.empty and "barthag" in conf_teams.columns:
                    conf_barthags = pd.to_numeric(
                        conf_teams["barthag"], errors="coerce"
                    ).dropna()
                    if len(conf_barthags) >= 3:
                        result["conf_strength"] = round(
                            float(conf_barthags.mean()), 4
                        )

        # ── Conference tournament wins ──
        conf_tourney = self._get_conf_tourney_games(team)
        if isinstance(conf_tourney, pd.DataFrame) and not conf_tourney.empty:
            try:
                wins = 0
                for _, g in conf_tourney.iterrows():
                    home = str(g.get("home_team", "")).lower()
                    away = str(g.get("away_team", "")).lower()
                    team_lower = team.lower()
                    h_score = pd.to_numeric(g.get("home_score"), errors="coerce")
                    a_score = pd.to_numeric(g.get("away_score"), errors="coerce")
                    status = str(g.get("status", "")).lower()
                    if "final" not in status and "post" not in status:
                        continue
                    if pd.notna(h_score) and pd.notna(a_score):
                        if team_lower in home and h_score > a_score:
                            wins += 1
                        elif team_lower in away and a_score > h_score:
                            wins += 1
                result["conf_tourney_wins"] = wins
            except Exception:
                pass

        self._cache[cache_key] = result
        return result

    # ── Layer 4: Experience ───────────────────────────────────────────────────

    def get_experience_data(
        self,
        team: str,
        team_espn_id: Optional[str] = None,
        season: Optional[int] = None,
    ) -> dict:
        season = season or self.season
        cache_suffix = team_espn_id or "auto"
        cache_key = f"experience_{team}_{season}_{cache_suffix}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not team_espn_id:
            team_espn_id = self._lookup_espn_team_id(team)

        roster = pd.DataFrame()
        if team_espn_id:
            roster = self._get_or_cache(
                f"espn_roster_{team_espn_id}_{season}",
                lambda: self._get_disk_cached_value(
                    category="espn",
                    key=f"roster_{team_espn_id}_{season}",
                    fetch_fn=lambda: self.espn.get_roster(team_espn_id, season),
                    ttl=self._cache_ttl_for_season(season),
                ),
            )

        from ..utils.coach_data import get_coach_record
        coach_record = get_coach_record(team)

        rpm_data = self._get_or_cache(
            f"rpm_{season}",
            lambda: self._get_returning_production(season),
        )
        team_rpm = {}
        if not rpm_data.empty:
            mask = rpm_data.get("team", pd.Series(dtype=str)).str.lower() == team.lower()
            match = rpm_data[mask]
            if not match.empty:
                team_rpm = match.iloc[0].to_dict()

        returning_pct = pd.to_numeric(
            team_rpm.get("returning_pct", 0.5),
            errors="coerce",
        )
        if pd.isna(returning_pct):
            returning_pct = 0.5

        result = {
            "team":            team,
            "season":          season,
            "roster":          roster,
            "coach_record":    coach_record,
            "returning_pct":   float(returning_pct),
        }
        self._cache[cache_key] = result
        return result

    # ── Layer 4: Rest & Schedule ──────────────────────────────────────────────

    def get_rest_data(
        self,
        team: str,
        game_date: Optional[str] = None,
        season: Optional[int] = None,
    ) -> dict:
        season = season or self.season
        game_date = game_date or datetime.now().strftime("%Y-%m-%d")

        schedule = self._get_team_schedule(team, season)

        rest_days = None
        last_game_date = None

        if not schedule.empty and "start_date" in schedule.columns:
            completed = schedule[
                schedule.get("status", pd.Series()) == "final"
            ] if "status" in schedule.columns else schedule

            if not completed.empty:
                dates = self._normalize_datetime_series(completed["start_date"]).dropna()
                target = self._normalize_datetime_value(game_date)
                if pd.isna(target):
                    target = pd.Timestamp.now().normalize()
                dates = dates[dates < target]
                if not dates.empty:
                    last_date = dates.max()
                    rest_days = (target - last_date).days
                    last_game_date = last_date.strftime("%Y-%m-%d")

        return {
            "team":            team,
            "rest_days":       rest_days,
            "last_game_date":  last_game_date,
            "game_date":       game_date,
        }

    def _get_team_game_log(
        self,
        team: str,
        season: Optional[int] = None,
    ) -> pd.DataFrame:
        season = season or self.season
        resolved_team = self.resolver.resolve_or_original(team)
        cache_key = f"game_log_full_{resolved_team}_{season}"
        return self._get_or_cache(
            cache_key,
            lambda: self._get_disk_cached_value(
                category="barttorvik",
                key=f"game_log_{season}_{resolved_team}",
                fetch_fn=lambda: self.torvik.get_game_log(resolved_team, season=season),
                ttl=self._cache_ttl_for_season(season),
            ),
        )

    def _get_team_schedule(
        self,
        team: str,
        season: Optional[int] = None,
    ) -> pd.DataFrame:
        season = season or self.season
        resolved_team = self.resolver.resolve_or_original(team)
        source = "cbbd" if self.cbbd else "torvik"
        cache_key = f"team_schedule_{source}_{resolved_team}_{season}"
        return self._get_or_cache(
            cache_key,
            lambda: self._get_disk_cached_value(
                category="schedules",
                key=f"{source}_team_schedule_{season}_{resolved_team}",
                fetch_fn=lambda: self.cbbd.build_team_schedule(resolved_team, season)
                if self.cbbd else self.torvik.get_schedule(season, resolved_team),
                ttl=self._cache_ttl_for_season(season),
            ),
        )

    def _lookup_espn_team_id(self, team: str) -> Optional[str]:
        team_key = self._team_match_key(team)
        if not team_key:
            return None

        lookup = self._get_or_cache(
            "espn_team_lookup",
            self._build_espn_team_lookup,
        )
        return lookup.get(team_key)

    def _build_espn_team_lookup(self) -> dict[str, str]:
        teams = self._get_disk_cached_value(
            category="espn",
            key="teams_all_d1",
            fetch_fn=self.espn.get_teams,
            ttl=timedelta(hours=24),
        )
        if teams.empty:
            return {}

        lookup: dict[str, str] = {}
        for _, row in teams.iterrows():
            team_id = row.get("espn_team_id")
            if not team_id:
                continue

            for candidate in (
                row.get("name"),
                row.get("short_name"),
                row.get("location"),
                row.get("abbreviation"),
            ):
                key = self._team_match_key(candidate)
                if key:
                    lookup[key] = team_id

        return lookup

    def _get_returning_production(self, season: int) -> pd.DataFrame:
        return self._get_disk_cached_value(
            category="barttorvik",
            key=f"returning_production_{season}",
            fetch_fn=lambda: self.torvik.get_returning_production(season),
            ttl=self._cache_ttl_for_season(season),
        )

    def _get_torvik_ratings(self, season: int) -> pd.DataFrame:
        return self._get_disk_cached_value(
            category="barttorvik",
            key=f"team_ratings_{season}",
            fetch_fn=lambda: self.torvik.get_team_ratings(season),
            ttl=self._cache_ttl_for_season(season),
        )

    def _get_national_averages_cached(self, seasons: list[int]) -> pd.DataFrame:
        if not seasons:
            return pd.DataFrame()

        normalized = sorted(int(season) for season in seasons)
        key = "national_averages_" + "_".join(str(season) for season in normalized)
        ttl = None
        if any(int(season) >= current_season() for season in normalized):
            ttl = timedelta(hours=PIPELINE_LIVE_CACHE_TTL_HOURS)

        return self._get_disk_cached_value(
            category="barttorvik",
            key=key,
            fetch_fn=lambda: self.torvik.get_national_averages(normalized),
            ttl=ttl,
        )

    def _get_cbbd_lines_for_season(self, season: int) -> pd.DataFrame:
        if not self.cbbd:
            return pd.DataFrame()

        return self._get_disk_cached_value(
            category="cbbd",
            key=f"season_lines_{season}",
            fetch_fn=lambda: self.cbbd.get_lines(season=season),
            ttl=self._cache_ttl_for_season(season),
        )

    def _get_conf_tourney_games_feed(self, days_window: int = 14) -> pd.DataFrame:
        return self._get_disk_cached_value(
            category="espn",
            key=f"conf_tourney_games_{days_window}",
            fetch_fn=lambda: self.espn.get_conf_tournament_games(days_window=days_window),
            ttl=timedelta(hours=6),
        )

    def _get_injuries_feed(self) -> pd.DataFrame:
        return self._get_disk_cached_value(
            category="espn",
            key="injuries",
            fetch_fn=self.espn.get_injuries,
            ttl=timedelta(hours=4),
        )

    def _get_tournament_bracket(self, season: int) -> pd.DataFrame:
        return self._get_disk_cached_value(
            category="espn",
            key=f"tournament_bracket_{season}",
            fetch_fn=lambda: self.espn.get_tournament_bracket(season),
            ttl=self._cache_ttl_for_season(season),
        )

    # ── Full Game Input Assembly ──────────────────────────────────────────────

    def get_game_inputs(
        self,
        home_team: str,
        away_team: str,
        season: Optional[int] = None,
        tournament_round: int = 1,
        venue_id: Optional[int] = None,
        venue_name: Optional[str] = None,
        game_date: Optional[str] = None,
        home_seed: Optional[int] = None,
        away_seed: Optional[int] = None,
        espn_spread: Optional[float] = None,
        espn_total: Optional[float] = None,
        is_ncaa_tournament: bool = False,
    ) -> dict:
        season = season or self.season
        game_date = game_date or datetime.now().strftime("%Y-%m-%d")
        is_ncaa_tournament = bool(is_ncaa_tournament)

        home_team = self.resolver.resolve_or_original(home_team)
        away_team = self.resolver.resolve_or_original(away_team)
        cache_ttl = self._cache_ttl_for_lookup_date(self._scoreboard_date_to_iso(game_date))
        game_inputs_cache_key = self._game_inputs_cache_key(
            home_team=home_team,
            away_team=away_team,
            season=season,
            tournament_round=tournament_round,
            venue_id=venue_id,
            venue_name=venue_name,
            game_date=game_date,
            home_seed=home_seed,
            away_seed=away_seed,
            espn_spread=espn_spread,
            espn_total=espn_total,
            is_ncaa_tournament=is_ncaa_tournament,
        )
        cached_inputs = self._read_disk_cache(
            category="game_inputs",
            key=game_inputs_cache_key,
            ttl=cache_ttl,
        )
        if isinstance(cached_inputs, dict):
            logger.info(
                "Loaded cached game inputs: %s vs %s (%s)",
                away_team,
                home_team,
                game_date,
            )
            return cached_inputs

        logger.info(f"\n{'='*55}")
        logger.info(f"Assembling inputs: {away_team} @ {home_team}")
        logger.info(f"{'='*55}")

        home_eff = self.get_team_efficiency(home_team, season)
        away_eff = self.get_team_efficiency(away_team, season)

        # Compute national average AdjOE for the Massey scoring formula
        national_avg_oe = self._get_national_avg_oe(season)

        venue = self.get_venue_profile(
            venue_id=venue_id,
            venue_name=venue_name,
            is_ncaa_tournament=is_ncaa_tournament,
        )
        home_momentum = self.get_momentum_data(home_team, season)
        away_momentum = self.get_momentum_data(away_team, season)
        home_exp = self.get_experience_data(home_team, season=season)
        away_exp = self.get_experience_data(away_team, season=season)
        home_rest = self.get_rest_data(home_team, game_date, season)
        away_rest = self.get_rest_data(away_team, game_date, season)
        injuries = self._get_or_cache("injuries", self._get_injuries_feed)
        home_travel = self.geo.travel_context(home_team, venue_name)
        away_travel = self.geo.travel_context(away_team, venue_name)
        home_extended = self.get_extended_team_stats(home_team, season)
        away_extended = self.get_extended_team_stats(away_team, season)

        if is_ncaa_tournament and (home_seed is None or away_seed is None):
            inferred_home_seed, inferred_away_seed = self._lookup_bracket_seeds(
                home_team,
                away_team,
                season=season,
            )
            if home_seed is None:
                home_seed = inferred_home_seed
            if away_seed is None:
                away_seed = inferred_away_seed

        seed_context = {}
        if is_ncaa_tournament and home_seed and away_seed:
            seed_context = self.seed_history.get_matchup_context(
                home_seed,
                away_seed,
                reference_year=season,
            )

        # Explicitly bypassing Odds API for ESPN Scraping fallbacks
        market_lines = {
            "consensus_spread": espn_spread,
            "consensus_total": espn_total,
            "spread_bookmaker_count": 1 if espn_spread is not None else 0,
            "total_bookmaker_count": 1 if espn_total is not None else 0,
        }

        raw_inputs = {
            "home_team":        home_team,
            "away_team":        away_team,
            "season":           season,
            "game_date":        game_date,
            "is_ncaa_tournament": is_ncaa_tournament,
            "tournament_round": tournament_round,
            "home_seed":        home_seed,
            "away_seed":        away_seed,
            "national_avg_oe":  national_avg_oe,
            "home_efficiency":  home_eff,
            "away_efficiency":  away_eff,
            "venue":            venue,
            "home_momentum":    home_momentum,
            "away_momentum":    away_momentum,
            "home_experience":  home_exp,
            "away_experience":  away_exp,
            "home_rest":        home_rest,
            "away_rest":        away_rest,
            "injuries":         injuries,
            "home_travel":      home_travel,
            "away_travel":      away_travel,
            "home_extended":    home_extended,
            "away_extended":    away_extended,
            "seed_context":     seed_context,
            "market_lines":     market_lines,
        }

        validated, reports = self.validator.validate_game_inputs(raw_inputs)
        self._write_disk_cache(
            category="game_inputs",
            key=game_inputs_cache_key,
            value=validated,
            persist_empty=bool(validated),
        )
        return validated

    def _lookup_bracket_seeds(
        self,
        home_team: str,
        away_team: str,
        season: Optional[int] = None,
    ) -> tuple[Optional[int], Optional[int]]:
        season = season or self.season
        seed_map = self._get_tournament_seed_map(season)
        if not seed_map:
            return None, None

        home_seed = seed_map.get(self._team_match_key(home_team))
        away_seed = seed_map.get(self._team_match_key(away_team))
        return home_seed, away_seed

    def _get_tournament_seed_map(self, season: Optional[int] = None) -> dict[str, int]:
        season = season or self.season
        cache_key = f"espn_tournament_seed_map_{season}"
        return self._get_or_cache(
            cache_key,
            lambda: self._build_tournament_seed_map(season),
        )

    def _build_tournament_seed_map(self, season: int) -> dict[str, int]:
        bracket = self._get_or_cache(
            f"espn_tournament_bracket_{season}",
            lambda: self._get_tournament_bracket(season),
        )
        if bracket.empty:
            return {}

        seed_map: dict[str, int] = {}
        for _, row in bracket.iterrows():
            team_name = row.get("team")
            seed_value = self._parse_seed_number(row.get("seed"))
            if not team_name or seed_value is None:
                continue

            keys = {
                self._team_match_key(team_name),
                self._team_match_key(self.resolver.resolve_or_original(str(team_name))),
            }
            for key in keys:
                if key:
                    seed_map[key] = seed_value

        logger.info(
            "Loaded NCAA tournament seed map from ESPN bracket: %s teams for %s",
            len(seed_map),
            season,
        )
        return seed_map

    @staticmethod
    def _parse_seed_number(seed_value) -> Optional[int]:
        if seed_value is None or (isinstance(seed_value, float) and pd.isna(seed_value)):
            return None

        text = str(seed_value).strip()
        if not text or text.lower() == "nan":
            return None

        match = re.search(r"\d+", text)
        if not match:
            return None

        try:
            return int(match.group())
        except ValueError:
            return None

    def _get_national_avg_oe(self, season: Optional[int] = None) -> float:
        """
        Get the national average AdjOE for the current season.
        Used as the baseline in the Massey scoring formula:
          Score_per_100 = AdjOE + AdjDE - national_avg_oe

        Computed from the full BartTorvik ratings table (already cached).
        Falls back to 105.0 if data is unavailable.
        """
        season = season or self.season
        cache_key = f"nat_avg_oe_{season}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        ratings = self._get_or_cache(
            f"torvik_ratings_{season}",
            lambda: self._get_torvik_ratings(season),
        )

        if not ratings.empty and "adj_oe" in ratings.columns:
            avg = float(ratings["adj_oe"].mean())
            logger.info(f"  National avg AdjOE ({season}): {avg:.1f}")
        else:
            avg = 105.0
            logger.warning(
                f"  No ratings for national avg — using default {avg}"
            )

        self._cache[cache_key] = avg
        return avg

    def _build_cbbd_game_lookup(self, games: pd.DataFrame) -> dict:
        lookup = {}
        if games.empty:
            return lookup

        for _, game in games.iterrows():
            home_key = self._team_match_key(game.get("home_team"))
            away_key = self._team_match_key(game.get("away_team"))
            if home_key and away_key:
                lookup[(home_key, away_key)] = game.to_dict()
        return lookup

    def _match_cbbd_game(self, scoreboard_row, game_lookup: dict) -> Optional[dict]:
        home_key = self._team_match_key(scoreboard_row.get("home_team"))
        away_key = self._team_match_key(scoreboard_row.get("away_team"))
        if not home_key or not away_key:
            return None

        direct = game_lookup.get((home_key, away_key))
        if direct:
            return direct

        for (cand_home, cand_away), game in game_lookup.items():
            if self._team_keys_match(home_key, cand_home) and self._team_keys_match(away_key, cand_away):
                return game

        return None

    def _team_match_key(self, team_name) -> str:
        if team_name is None:
            return ""
        raw_name = str(team_name).strip()
        key = raw_name.lower()

        # Use the resolver's alias index directly here so bulk season-line
        # preparation does not emit warnings for obscure non-DI opponents.
        resolved = self.resolver._alias_index.get(key)
        if not resolved:
            normalized = self.resolver._normalize(key)
            resolved = self.resolver._alias_index.get(normalized)
        if not resolved:
            for variant in self.resolver._generate_variants(key):
                resolved = self.resolver._alias_index.get(variant)
                if resolved:
                    break
        if not resolved:
            resolved = self.resolver._fuzzy_match(key, max_distance=3)

        resolved = resolved or raw_name
        lowered = resolved.lower()
        cleaned = "".join(ch if ch.isalnum() else " " for ch in lowered)
        return " ".join(cleaned.split())

    @staticmethod
    def _team_keys_match(left: str, right: str) -> bool:
        if not left or not right:
            return False
        if left == right:
            return True
        if left in right or right in left:
            return True

        left_tokens = set(left.split())
        right_tokens = set(right.split())
        return bool(left_tokens and right_tokens and left_tokens == right_tokens)

    @staticmethod
    def _scoreboard_date_to_iso(value) -> Optional[str]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None

        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None

        for fmt in ("%Y-%m-%dT%H:%MZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%Y%m%d"):
            try:
                return pd.to_datetime(text, format=fmt, utc=False).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                continue

        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.strftime("%Y-%m-%d")

    def _odds_cache_label(
        self,
        scoreboard_date: Optional[str],
        rows: pd.DataFrame,
    ) -> str:
        lookup_date = self._scoreboard_date_to_iso(scoreboard_date)
        if lookup_date:
            return lookup_date.replace("-", "")

        if "date" in rows.columns:
            row_dates = self._normalize_datetime_series(rows["date"]).dropna()
            if not row_dates.empty:
                return row_dates.min().strftime("%Y%m%d")

        return datetime.now().strftime("%Y%m%d")

    def _odds_feed_time_window(
        self,
        rows: pd.DataFrame,
        scoreboard_date: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        if "date" in rows.columns:
            row_dates = self._normalize_datetime_series(rows["date"]).dropna()
            if not row_dates.empty:
                start = row_dates.min() - pd.Timedelta(hours=2)
                end = row_dates.max() + pd.Timedelta(hours=2)
                return (
                    start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )

        lookup_date = self._scoreboard_date_to_iso(scoreboard_date)
        base_date = pd.to_datetime(lookup_date, errors="coerce")
        if pd.isna(base_date):
            return None, None

        start = pd.Timestamp(base_date).normalize()
        end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        return (
            start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def _historical_snapshot_for_rows(
        self,
        rows: pd.DataFrame,
        scoreboard_date: Optional[str],
    ) -> Optional[str]:
        if "date" in rows.columns:
            row_dates = self._normalize_datetime_series(rows["date"]).dropna()
            if not row_dates.empty:
                snapshot = row_dates.min() - pd.Timedelta(minutes=5)
                return snapshot.strftime("%Y-%m-%dT%H:%M:%SZ")

        lookup_date = self._scoreboard_date_to_iso(scoreboard_date)
        base_date = pd.to_datetime(lookup_date, errors="coerce")
        if pd.isna(base_date):
            return None

        snapshot = pd.Timestamp(base_date).normalize() + pd.Timedelta(minutes=5)
        return snapshot.strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _normalize_datetime_series(values) -> pd.Series:
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
        if isinstance(parsed, pd.Series):
            return parsed.dt.tz_localize(None)
        return pd.Series(dtype="datetime64[ns]")

    @staticmethod
    def _normalize_datetime_value(value):
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(parsed):
            return parsed
        if hasattr(parsed, "tz_localize"):
            return parsed.tz_localize(None)
        return parsed

    @staticmethod
    def _row_has_market_line(row) -> bool:
        spread = row.get("betting_spread")
        total = row.get("over_under")
        has_spread = pd.notna(spread) and str(spread).strip() != ""
        has_total = pd.notna(total) and str(total).strip() != ""
        return has_spread or has_total

    @staticmethod
    def _row_is_final(row) -> bool:
        status = str(row.get("status", "")).strip().lower()
        return bool(status and ("final" in status or "post" in status))

    def _apply_market_summary(
        self,
        enriched: pd.DataFrame,
        idx,
        spread,
        total,
        provider_count: int,
        source: str,
        event_id=None,
        snapshot_time: Optional[str] = None,
        historical: bool = False,
    ) -> None:
        enriched.at[idx, "betting_spread"] = spread
        enriched.at[idx, "over_under"] = total
        enriched.at[idx, "line_source"] = source
        enriched.at[idx, "line_provider_count"] = provider_count

        if historical:
            enriched.at[idx, "historical_betting_spread"] = spread
            enriched.at[idx, "historical_over_under"] = total
            enriched.at[idx, "historical_line_provider_count"] = provider_count
            if event_id is not None:
                enriched.at[idx, "historical_line_game_id"] = event_id

        if snapshot_time:
            enriched.at[idx, "historical_line_snapshot_time"] = snapshot_time

        # Let downstream formatting render from numeric line values.
        if "odds_detail" in enriched.columns:
            enriched.at[idx, "odds_detail"] = ""

    def _get_prepared_cbbd_season_lines(self) -> pd.DataFrame:
        season_lines = self._get_or_cache(
            f"cbbd_lines_season_{self.season}",
            lambda: self._get_cbbd_lines_for_season(self.season),
        )
        if season_lines.empty:
            return season_lines

        prepared = season_lines.copy()
        prepared["home_team_key"] = prepared.get(
            "home_team", pd.Series(dtype=str)
        ).apply(self._team_match_key)
        prepared["away_team_key"] = prepared.get(
            "away_team", pd.Series(dtype=str)
        ).apply(self._team_match_key)
        prepared["line_date"] = prepared.get(
            "start_date", pd.Series(dtype=object)
        ).apply(self._scoreboard_date_to_iso)
        prepared["home_score_num"] = pd.to_numeric(
            prepared.get("home_score"), errors="coerce"
        )
        prepared["away_score_num"] = pd.to_numeric(
            prepared.get("away_score"), errors="coerce"
        )
        return prepared

    @staticmethod
    def _neighbor_dates(lookup_date: Optional[str]) -> list[str]:
        if not lookup_date:
            return []
        base = pd.to_datetime(lookup_date, errors="coerce")
        if pd.isna(base):
            return []
        return [
            (base + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            for offset in (0, -1, 1)
        ]

    def _candidate_lookup_dates_for_row(
        self,
        row,
        requested_date: Optional[str],
    ) -> list[str]:
        dates = []
        if requested_date:
            dates.extend(self._neighbor_dates(requested_date))

        row_date = row.get("cbbd_lookup_date")
        if row_date:
            dates.extend(self._neighbor_dates(row_date))

        return list(dict.fromkeys(d for d in dates if d))

    def _match_cbbd_lines_for_row(
        self,
        row,
        season_lines: pd.DataFrame,
        requested_date: Optional[str],
    ) -> pd.DataFrame:
        home_key = self._team_match_key(row.get("home_team"))
        away_key = self._team_match_key(row.get("away_team"))
        if not home_key or not away_key or season_lines.empty:
            return pd.DataFrame()

        matches = season_lines[
            (season_lines["home_team_key"] == home_key)
            & (season_lines["away_team_key"] == away_key)
        ].copy()

        if matches.empty:
            fuzzy_mask = season_lines["home_team_key"].apply(
                lambda cand: self._team_keys_match(home_key, cand)
            ) & season_lines["away_team_key"].apply(
                lambda cand: self._team_keys_match(away_key, cand)
            )
            matches = season_lines[fuzzy_mask].copy()

        if matches.empty:
            return matches

        candidate_dates = self._candidate_lookup_dates_for_row(
            row,
            requested_date,
        )
        if candidate_dates and "line_date" in matches.columns:
            dated = matches[matches["line_date"].isin(candidate_dates)].copy()
            if not dated.empty:
                matches = dated

        home_score = pd.to_numeric(row.get("home_score"), errors="coerce")
        away_score = pd.to_numeric(row.get("away_score"), errors="coerce")
        if pd.notna(home_score) and pd.notna(away_score):
            scored = matches[
                (matches["home_score_num"] == home_score)
                & (matches["away_score_num"] == away_score)
            ].copy()
            if not scored.empty:
                matches = scored

        if "game_id" in matches.columns:
            game_ids = matches["game_id"].dropna()
            if not game_ids.empty and game_ids.nunique() > 1:
                preferred_id = game_ids.iloc[0]
                matches = matches[matches["game_id"] == preferred_id].copy()

        return matches

    def _cache_ttl_for_season(self, season: int) -> Optional[timedelta]:
        current = current_season()
        if int(season) < current:
            return None
        return timedelta(hours=PIPELINE_LIVE_CACHE_TTL_HOURS)

    def _should_refresh_cached_scoreboard(
        self,
        cached: pd.DataFrame,
        lookup_date: Optional[str],
    ) -> bool:
        if not isinstance(cached, pd.DataFrame) or cached.empty:
            return False

        status_series = cached.get("status", pd.Series(dtype=object)).fillna("").astype(str).str.lower()
        if status_series.empty:
            return False

        all_final = status_series.str.contains("final|post", regex=True).all()
        if bool(all_final):
            return False

        if lookup_date:
            parsed_lookup = pd.to_datetime(lookup_date, errors="coerce")
        else:
            parsed_lookup = pd.Timestamp.now().normalize()
        if pd.isna(parsed_lookup):
            return True

        today = pd.Timestamp.now().normalize()
        return pd.Timestamp(parsed_lookup).normalize() <= today

    def _cache_ttl_for_lookup_date(
        self,
        lookup_date: Optional[str],
    ) -> Optional[timedelta]:
        if not lookup_date:
            return timedelta(hours=PIPELINE_LIVE_CACHE_TTL_HOURS)

        parsed = pd.to_datetime(lookup_date, errors="coerce")
        if pd.isna(parsed):
            return timedelta(hours=PIPELINE_LIVE_CACHE_TTL_HOURS)

        today = pd.Timestamp.now().normalize()
        if pd.Timestamp(parsed).normalize() < today:
            return None
        return timedelta(hours=PIPELINE_LIVE_CACHE_TTL_HOURS)

    def _game_inputs_cache_key(self, **kwargs) -> str:
        parts = []
        for key in sorted(kwargs):
            value = kwargs[key]
            if isinstance(value, float):
                value = round(value, 4)
            parts.append(f"{key}={value}")
        return "|".join(parts)

    def _get_disk_cached_value(
        self,
        category: str,
        key: str,
        fetch_fn,
        ttl: Optional[timedelta] = None,
        persist_empty: bool = False,
    ) -> Any:
        cached = self._read_disk_cache(category=category, key=key, ttl=ttl)
        if cached is not None:
            return cached

        value = fetch_fn()
        self._write_disk_cache(
            category=category,
            key=key,
            value=value,
            persist_empty=persist_empty,
        )
        return value

    def _read_disk_cache(
        self,
        category: str,
        key: str,
        ttl: Optional[timedelta] = None,
    ) -> Any:
        path = self._disk_cache_path(category, key)
        if not os.path.exists(path):
            return None

        if ttl is not None:
            try:
                modified = datetime.fromtimestamp(os.path.getmtime(path))
            except OSError:
                return None
            if datetime.now() - modified > ttl:
                return None

        try:
            return pd.read_pickle(path)
        except Exception as exc:
            logger.warning("Could not read disk cache %s: %s", path, exc)
            return None

    def _write_disk_cache(
        self,
        category: str,
        key: str,
        value: Any,
        persist_empty: bool = False,
    ) -> None:
        if value is None:
            return
        if isinstance(value, pd.DataFrame) and value.empty and not persist_empty:
            return
        if isinstance(value, dict) and not value and not persist_empty:
            return

        path = self._disk_cache_path(category, key)
        try:
            pd.to_pickle(value, path)
        except Exception as exc:
            logger.warning("Could not write disk cache %s: %s", path, exc)

    def _disk_cache_path(self, category: str, key: str) -> str:
        category_dir = os.path.join(self._disk_cache_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        safe_prefix = re.sub(r"[^a-zA-Z0-9]+", "_", key).strip("_")[:80] or "cache"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        filename = f"{safe_prefix}_{digest}.pkl"
        return os.path.join(category_dir, filename)

    def _get_or_cache(self, key: str, fetch_fn):
        if key not in self._cache:
            self._cache[key] = fetch_fn()
        return self._cache[key]

    def health_check(self) -> dict:
        """Test connectivity to all data sources."""
        status = {}
        try:
            df = self.torvik.get_team_ratings()
            status["barttorvik"] = "OK" if not df.empty else "EMPTY"
        except Exception as e:
            status["barttorvik"] = f"ERROR: {e}"

        try:
            df = self.espn.get_scoreboard()
            status["espn"] = "OK" if not df.empty else "EMPTY"
        except Exception as e:
            status["espn"] = f"ERROR: {e}"

        if self.cbbd:
            try:
                df = self.cbbd.get_teams()
                status["cbbd"] = "OK" if not df.empty else "EMPTY"
            except Exception as e:
                status["cbbd"] = f"ERROR: {e}"
        else:
            status["cbbd"] = "NOT CONFIGURED"

        return status
