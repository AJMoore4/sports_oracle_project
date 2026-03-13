"""
sports_oracle/collectors/pipeline.py

Master data pipeline — orchestrates all collectors
and assembles model-ready input DataFrames for each
formula layer.
"""

from __future__ import annotations
import math
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta

from .config import (
    current_season, BUBBLE_SEASONS,
    SHOT_CLOCK_CHANGE_SEASON, logger,
)
from .cbbd_collector import CBBDCollector
from .barttorvik_collector import BartTorvik
from .espn_collector import ESPNCollector
from .ncaa_collector import NCAACollector
from ..utils.team_resolver import get_resolver
from ..utils.data_validator import DataValidator
from ..utils.geo import GeoLookup
from ..utils.seed_history import SeedHistory


class DataPipeline:
    """
    Orchestrates all data collectors and assembles
    model-ready inputs for the prediction engine.
    """

    def __init__(
        self,
        cbbd_key: str = "",
        season: Optional[int] = None,
    ):
        self.season = season or current_season()

        # Initialize collectors
        self.espn = ESPNCollector()
        self.torvik = BartTorvik()
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
            lambda: self.torvik.get_team_ratings(season),
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
        }

        self._cache[cache_key] = result
        return result

    # ── Layer 2: Venue Profile ────────────────────────────────────────────────

    def get_venue_profile(
        self,
        venue_id: Optional[int] = None,
        venue_name: Optional[str] = None,
        seasons: Optional[list[int]] = None,
    ) -> dict:
        seasons = seasons or list(range(2010, self.season + 1))
        seasons = [s for s in seasons if s not in BUBBLE_SEASONS]

        nat_avgs = self._get_or_cache(
            "national_averages",
            lambda: self.torvik.get_national_averages(seasons),
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

        if not self.cbbd or (not venue_id and not venue_name):
            return profile

        venue_history = self._get_or_cache(
            "venue_tournament_history",
            lambda: self.cbbd.build_venue_game_history(seasons=seasons),
        )

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
                if w_sum > 0: profile["vsi"] = sum(weighted_scoring_ratios) / w_sum

            if weighted_pace_ratios:
                w_sum = sum(w for w, _ in zip(weights, weighted_pace_ratios))
                if w_sum > 0: profile["vpi"] = sum(weighted_pace_ratios) / w_sum

            if weighted_3pt_ratios:
                w_sum = sum(w for w, _ in zip(weights, weighted_3pt_ratios))
                if w_sum > 0: profile["v3p"] = sum(weighted_3pt_ratios) / w_sum

            profile["sample_size"] = n
        return profile

    # ── Layer 4: Momentum ─────────────────────────────────────────────────────

    def get_momentum_data(
        self,
        team: str,
        season: Optional[int] = None,
        n_games: int = 10,
    ) -> dict:
        season = season or self.season
        game_log = self.torvik.get_last_n_games(team, n=n_games, season=season)
        season_eff = self.get_team_efficiency(team, season)
        conf_tourney = self._get_conf_tourney_games(team)

        margins = []
        if not game_log.empty:
            for col in ["margin", "pts_diff", "point_diff", "score_diff"]:
                if col in game_log.columns:
                    margins = pd.to_numeric(game_log[col], errors="coerce").dropna().tolist()
                    break

        return {
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

    def _get_conf_tourney_games(self, team: str) -> pd.DataFrame:
        cache_key = f"conf_tourney_{team}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        recent = self._get_or_cache(
            "conf_tourney_all_games",
            lambda: self.espn.get_conf_tournament_games(days_window=14),
        )
        if recent.empty:
            return pd.DataFrame()

        team_lower = team.lower()
        mask = (
            recent.get("home_team", pd.Series(dtype=str)).str.lower().str.contains(team_lower, na=False)
            | recent.get("away_team", pd.Series(dtype=str)).str.lower().str.contains(team_lower, na=False)
        )
        result = recent[mask].copy()
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

        if not team_espn_id:
            team_espn_id = self.espn.get_team_id(team)

        roster = pd.DataFrame()
        if team_espn_id:
            roster = self.espn.get_roster(team_espn_id, season)

        from ..utils.coach_data import get_coach_record
        coach_record = get_coach_record(team)

        rpm_data = self._get_or_cache(
            f"rpm_{season}",
            lambda: self.torvik.get_returning_production(season),
        )
        team_rpm = {}
        if not rpm_data.empty:
            mask = rpm_data.get("team", pd.Series(dtype=str)).str.lower() == team.lower()
            match = rpm_data[mask]
            if not match.empty:
                team_rpm = match.iloc[0].to_dict()

        return {
            "team":            team,
            "season":          season,
            "roster":          roster,
            "coach_record":    coach_record,
            "returning_pct":   float(team_rpm.get("returning_pct", 0.5) or 0.5),
        }

    # ── Layer 4: Rest & Schedule ──────────────────────────────────────────────

    def get_rest_data(
        self,
        team: str,
        game_date: Optional[str] = None,
        season: Optional[int] = None,
    ) -> dict:
        season = season or self.season
        game_date = game_date or datetime.now().strftime("%Y-%m-%d")

        if self.cbbd: schedule = self.cbbd.build_team_schedule(team, season)
        else: schedule = self.torvik.get_schedule(season, team)

        rest_days = None
        last_game_date = None

        if not schedule.empty and "start_date" in schedule.columns:
            completed = schedule[
                schedule.get("status", pd.Series()) == "final"
            ] if "status" in schedule.columns else schedule

            if not completed.empty:
                dates = pd.to_datetime(completed["start_date"], errors="coerce").dropna()
                target = pd.to_datetime(game_date)
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
    ) -> dict:
        season = season or self.season
        game_date = game_date or datetime.now().strftime("%Y-%m-%d")

        home_team = self.resolver.resolve_or_original(home_team)
        away_team = self.resolver.resolve_or_original(away_team)

        logger.info(f"\n{'='*55}")
        logger.info(f"Assembling inputs: {away_team} @ {home_team}")
        logger.info(f"{'='*55}")

        home_eff = self.get_team_efficiency(home_team, season)
        away_eff = self.get_team_efficiency(away_team, season)
        venue = self.get_venue_profile(venue_id=venue_id, venue_name=venue_name)
        home_momentum = self.get_momentum_data(home_team, season)
        away_momentum = self.get_momentum_data(away_team, season)
        home_exp = self.get_experience_data(home_team, season=season)
        away_exp = self.get_experience_data(away_team, season=season)
        home_rest = self.get_rest_data(home_team, game_date, season)
        away_rest = self.get_rest_data(away_team, game_date, season)
        injuries = self._get_or_cache("injuries", lambda: self.espn.get_injuries())
        home_travel = self.geo.travel_context(home_team, venue_name)
        away_travel = self.geo.travel_context(away_team, venue_name)

        seed_context = {}
        if home_seed and away_seed:
            seed_context = self.seed_history.get_matchup_context(home_seed, away_seed)

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
            "tournament_round": tournament_round,
            "home_seed":        home_seed,
            "away_seed":        away_seed,
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
            "seed_context":     seed_context,
            "market_lines":     market_lines,
        }

        validated, reports = self.validator.validate_game_inputs(raw_inputs)
        return validated

    def _get_or_cache(self, key: str, fetch_fn):
        if key not in self._cache:
            self._cache[key] = fetch_fn()
        return self._cache[key]