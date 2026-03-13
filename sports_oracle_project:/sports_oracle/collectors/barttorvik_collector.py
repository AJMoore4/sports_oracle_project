"""
sports_oracle/collectors/barttorvik_collector.py

BartTorvik (T-Rank) data collector.

PRIMARY SOURCE FOR:
  Layer 1 — AdjOE, AdjDE, Pace, SOS, eFG%, TO rate,
             3PA rate, FTA rate, FT%, shooting splits
  Layer 4 — Momentum opponent quality ratings
  Layer 4 — Returning possession minutes (roster continuity proxy)

ACCESS:
  No API key needed. Bart makes bulk data publicly
  available as CSV/JSON files at predictable URLs.
  Format: barttorvik.com/YYYY_team_results.csv
  Updates every 15 minutes during the season.

RATE LIMITING:
  Be respectful — 3-4 second delay between requests.
  Bart has stated he blocks mass scrapers.
  We cache results locally to minimize hits.

RECENCY BIAS NOTE:
  BartTorvik already applies recency weighting:
    - Games > 40 days old: weight decreases 1%/day
    - Games > 80 days old: 60% weight vs recent games
  This aligns with our Momentum vs. Season Average design.
  Use no_bias=True endpoints when we want raw season totals
  for our own decay calculations.
"""

from __future__ import annotations
import io
import time
import pandas as pd
from typing import Optional
from .config import (
    BaseClient, BARTTORVIK_BASE,
    current_season, season_range, BUBBLE_SEASONS,
    SHOT_CLOCK_CHANGE_SEASON, logger,
)


# ── Column name mappings from BartTorvik CSV format ───────────────────────────
# BartTorvik column names vary slightly by season — we normalize them.
TORVIK_COL_MAP = {
    # Team info
    "team":         "team",
    "conf":         "conference",
    "g":            "games_played",
    "rec":          "record",

    # Efficiency ratings (T-Rank core)
    "adjoe":        "adj_oe",      # Adjusted Offensive Efficiency
    "adjde":        "adj_de",      # Adjusted Defensive Efficiency
    "barthag":      "barthag",     # Win probability vs average D1 team
    "efg_o":        "efg_pct_off", # Effective FG% offense
    "efg_d":        "efg_pct_def", # Effective FG% defense allowed
    "tov_o":        "to_rate_off", # Turnover rate offense (per 100 poss)
    "tov_d":        "to_rate_def", # Turnover rate defense (forced)
    "orb":          "orb_pct",     # Offensive rebound %
    "drb":          "drb_pct",     # Defensive rebound %
    "ftr_o":        "fta_rate_off",# FTA/FGA offense
    "ftr_d":        "fta_rate_def",# FTA/FGA defense allowed

    # Tempo
    "adjt":         "adj_tempo",   # Adjusted possessions per 40min
    "tempo_raw":    "raw_tempo",

    # Shooting splits
    "two_pt_o":     "two_pt_pct_off",
    "two_pt_d":     "two_pt_pct_def",
    "three_pt_o":   "three_pt_pct_off",
    "three_pt_d":   "three_pt_pct_def",
    "three_pr_o":   "three_pt_rate_off",  # 3PA / FGA
    "three_pr_d":   "three_pt_rate_def",

    # Schedule strength
    "sos":          "sos",
    "elite_sos":    "elite_sos",
    "ncsos":        "non_conf_sos",

    # Ranks
    "rk":           "rank",
    "seed":         "seed",
}


class BartTorvik(BaseClient):
    """
    Pulls T-Rank data from BartTorvik's public bulk files.
    All CSV/JSON files available without authentication.
    """

    def __init__(self):
        super().__init__("barttorvik", BARTTORVIK_BASE, delay=3.5)

    # ── Core: Team Season Ratings ────────────────────────────────────────────

    def get_team_ratings(
        self,
        season: Optional[int] = None,
        as_of_date: Optional[str] = None,   # 'YYYY-MM-DD' for snapshot
    ) -> pd.DataFrame:
        """
        Fetch full team efficiency ratings for a season.

        Returns DataFrame with AdjOE, AdjDE, Pace, SOS,
        shooting splits, turnover rates, and rebound rates
        for all Division I teams.

        season: NCAA season year (e.g. 2025 = 2024-25 season)
        as_of_date: optional snapshot date (BartTorvik supports this)
        """
        season = season or current_season()

        if as_of_date:
            # BartTorvik supports date-filtered T-Rank via URL params
            url = f"{self.base_url}/trank.php"
            params = {
                "year": season,
                "begindate": "20000101",
                "enddate": as_of_date.replace("-", ""),
                "csv": 1,
            }
            self.log.info(f"Fetching T-Rank ratings as of {as_of_date}...")
            csv_text = self._get_with_params(url, params)
        else:
            # Full season ratings CSV
            url = f"{self.base_url}/{season}_team_results.csv"
            self.log.info(f"Fetching team ratings for season {season}...")
            csv_text = self.get_csv(url)

        if not csv_text:
            self.log.warning(f"No data for season {season}")
            return pd.DataFrame()

        df = self._parse_csv(csv_text)
        df = self._normalize_columns(df)
        df["season"] = season

        self.log.info(f"  → {len(df)} teams loaded for season {season}")
        return df

    def get_ratings_multi_season(
        self,
        seasons: Optional[list[int]] = None,
        start_season: int = 2010,
    ) -> pd.DataFrame:
        """
        Load team ratings across multiple seasons.
        Used for historical baseline calculations in Layer 2.
        Automatically excludes bubble season (2021) from pace
        calculations but keeps it flagged.
        """
        seasons = seasons or season_range(start_season)
        all_dfs = []

        for season in seasons:
            df = self.get_team_ratings(season)
            if not df.empty:
                df["is_bubble_season"] = season in BUBBLE_SEASONS
                df["pre_shot_clock_change"] = season < SHOT_CLOCK_CHANGE_SEASON
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        self.log.info(
            f"Multi-season ratings loaded: {len(combined)} team-seasons "
            f"across {len(all_dfs)} seasons"
        )
        return combined

    # ── National Averages (for era adjustment) ────────────────────────────────

    def get_national_averages(
        self,
        seasons: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Calculate national average scoring and pace per season.
        Essential for Layer 2 era-adjusted Venue Scoring Index.

        Returns DataFrame:
          season | nat_avg_total_pts | nat_avg_adj_oe |
          nat_avg_adj_de | nat_avg_tempo | year_weight
        """
        seasons = seasons or season_range(2010)
        records = []

        for season in seasons:
            df = self.get_team_ratings(season)
            if df.empty:
                continue

            # Calculate season-wide averages
            record = {"season": season}

            if "adj_oe" in df.columns:
                record["nat_avg_adj_oe"] = df["adj_oe"].mean()
            if "adj_de" in df.columns:
                record["nat_avg_adj_de"] = df["adj_de"].mean()
            if "adj_tempo" in df.columns:
                record["nat_avg_tempo"] = df["adj_tempo"].mean()
            if "efg_pct_off" in df.columns:
                record["nat_avg_efg"] = df["efg_pct_off"].mean()
            if "three_pt_pct_off" in df.columns:
                record["nat_avg_3pt_pct"] = df["three_pt_pct_off"].mean()
            if "three_pt_rate_off" in df.columns:
                record["nat_avg_3pt_rate"] = df["three_pt_rate_off"].mean()
            if "to_rate_off" in df.columns:
                record["nat_avg_to_rate"] = df["to_rate_off"].mean()

            # Estimated avg total points per game
            # = (avg AdjOE + avg AdjDE) / 2 * (avg tempo / 100) * 2 teams
            if "adj_oe" in df.columns and "adj_tempo" in df.columns:
                avg_oe = df["adj_oe"].mean()
                avg_de = df["adj_de"].mean()
                avg_tempo = df["adj_tempo"].mean()
                # Rough total points estimate (both teams combined)
                record["nat_avg_total_pts"] = (
                    (avg_oe + (200 - avg_de)) / 2 * avg_tempo / 100 * 2
                )

            record["is_bubble_season"] = season in BUBBLE_SEASONS
            record["pre_shot_clock"] = season < SHOT_CLOCK_CHANGE_SEASON
            records.append(record)

        return pd.DataFrame(records)

    # ── Game Logs ─────────────────────────────────────────────────────────────

    def get_game_log(
        self,
        team: str,
        season: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch game-by-game results with efficiency metrics for a team.
        Used for Layer 4 Momentum calculation.

        Returns per-game: date, opponent, result, margin,
        raw_oe, raw_de, tempo, efg, to_rate
        """
        season = season or current_season()
        url = f"{self.base_url}/team_game_stats.php"
        params = {
            "year": season,
            "team": team,
            "csv": 1,
        }

        self.log.info(f"Fetching game log: {team} ({season})...")
        csv_text = self._get_with_params(url, params)

        if not csv_text:
            return pd.DataFrame()

        df = self._parse_csv(csv_text)
        if df.empty:
            return df

        # Normalize and sort chronologically
        df = self._normalize_columns(df)
        df["team"] = team
        df["season"] = season

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)

        self.log.info(f"  → {len(df)} games for {team}")
        return df

    def get_last_n_games(
        self,
        team: str,
        n: int = 10,
        season: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch the most recent N games for momentum calculation.
        Returns games sorted most-recent first.
        """
        df = self.get_game_log(team, season)
        if df.empty:
            return df
        # Return last N chronologically (most recent at top after reverse)
        return df.tail(n).sort_values(
            "date", ascending=False
        ).reset_index(drop=True) if "date" in df.columns else df.tail(n)

    # ── Returning Production / Roster Continuity ──────────────────────────────

    def get_returning_production(
        self,
        season: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch returning possession minutes (RPM) data.
        Proxy for roster continuity in Layer 4 Experience Factor.

        Files available at barttorvik.com/YYYY_rpm.json (back to 2017)
        Fields: team, total_poss_min, returning_poss_min,
                returning_pct, experienced_pct
        """
        season = season or current_season()
        url = f"{self.base_url}/{season}_rpm.json"

        self.log.info(f"Fetching returning production for season {season}...")
        data = self.get_csv(url)   # JSON returned as text

        if not data:
            return pd.DataFrame()

        try:
            import json
            records = json.loads(data)
            df = pd.DataFrame(records)
            df["season"] = season
            self.log.info(f"  → {len(df)} teams with RPM data")
            return df
        except Exception as e:
            self.log.warning(f"Could not parse RPM data: {e}")
            return pd.DataFrame()

    # ── Opponent Quality for Momentum ─────────────────────────────────────────

    def get_opponent_ratings(
        self,
        opponents: list[str],
        season: Optional[int] = None,
    ) -> dict[str, dict]:
        """
        Look up efficiency ratings for a list of opponents.
        Used in Layer 4 Momentum to weight wins/losses
        by opponent quality.

        Returns: {team_name: {adj_oe, adj_de, rank, ...}}
        """
        season = season or current_season()
        all_ratings = self.get_team_ratings(season)

        if all_ratings.empty:
            return {}

        result = {}
        for opp in opponents:
            # Case-insensitive match
            mask = all_ratings["team"].str.lower() == opp.lower()
            match = all_ratings[mask]
            if not match.empty:
                result[opp] = match.iloc[0].to_dict()
            else:
                self.log.warning(f"No ratings found for opponent: {opp}")
        return result

    # ── Schedule for Rest Factor ──────────────────────────────────────────────

    def get_schedule(
        self,
        season: Optional[int] = None,
        team: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch full schedule CSV.
        Used for Layer 4 Rest & Schedule (days since last game).
        """
        season = season or current_season()
        url = f"{self.base_url}/{season}_super_sked.csv"

        self.log.info(f"Fetching schedule data for season {season}...")
        csv_text = self.get_csv(url)
        if not csv_text:
            return pd.DataFrame()

        df = self._parse_csv(csv_text)
        df["season"] = season

        if team:
            mask = (
                df.get("team", pd.Series(dtype=str)).str.lower() == team.lower()
            ) | (
                df.get("opp", pd.Series(dtype=str)).str.lower() == team.lower()
            )
            df = df[mask]

        return df

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_with_params(self, url: str, params: dict) -> Optional[str]:
        """GET request with query params, returns raw text."""
        self._throttle()
        try:
            resp = self.session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            self.log.warning(f"Request failed {url}: {e}")
            return None

    def _parse_csv(self, csv_text: str) -> pd.DataFrame:
        """Parse CSV text into DataFrame, handle encoding issues."""
        try:
            return pd.read_csv(
                io.StringIO(csv_text),
                dtype=str,        # read all as str first, cast later
                na_values=["", "N/A", "nan", "-"],
            )
        except Exception as e:
            self.log.warning(f"CSV parse error: {e}")
            return pd.DataFrame()

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to our standard schema.
        BartTorvik columns vary across seasons and endpoints.
        """
        # Lowercase and strip all column names
        df.columns = [c.lower().strip() for c in df.columns]

        # Apply our mapping
        df = df.rename(columns=TORVIK_COL_MAP)

        # Numeric cast for key efficiency columns
        numeric_cols = [
            "adj_oe", "adj_de", "adj_tempo", "barthag",
            "efg_pct_off", "efg_pct_def",
            "to_rate_off", "to_rate_def",
            "three_pt_rate_off", "three_pt_rate_def",
            "three_pt_pct_off", "three_pt_pct_def",
            "two_pt_pct_off", "two_pt_pct_def",
            "fta_rate_off", "fta_rate_def",
            "orb_pct", "drb_pct",
            "sos", "elite_sos", "non_conf_sos",
            "rank", "seed",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bt = BartTorvik()

    print("\n📊 T-Rank Top 10 (2025 season):")
    ratings = bt.get_team_ratings(2025)
    if not ratings.empty:
        display_cols = [c for c in ["rank","team","adj_oe","adj_de","adj_tempo","sos"]
                        if c in ratings.columns]
        # Sort by rank if available
        if "rank" in ratings.columns:
            ratings["rank"] = pd.to_numeric(ratings["rank"], errors="coerce")
            ratings = ratings.sort_values("rank")
        print(ratings[display_cols].head(10).to_string(index=False))

    print("\n🌎 National Averages (2020-2025):")
    avgs = bt.get_national_averages(seasons=list(range(2020, 2026)))
    if not avgs.empty:
        print(avgs[["season","nat_avg_adj_oe","nat_avg_tempo"]].to_string(index=False))