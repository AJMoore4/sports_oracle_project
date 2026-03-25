"""
sports_oracle/collectors/cbbd_collector.py

CollegeBasketballData.com (CBBD) API collector.

PRIMARY SOURCE FOR:
  - Game results (regular season + tournament)
  - Venue data (979 venues, clean IDs)
  - Betting lines (historical spreads + totals)
  - Play-by-play and shot location data
  - Player and team metadata

SETUP:
  1. Sign up at collegebasketballdata.com/key (free)
  2. Add to .env: CBBD_API_KEY=your_key_here
  3. Free tier: 1,000 calls/month

OFFICIAL PYTHON PACKAGE: pip install cbbd
We use both the official package AND direct REST calls
depending on what's cleaner for each endpoint.
"""

from __future__ import annotations
import re
import pandas as pd
from typing import Optional
from .config import (
    BaseClient, CBBD_BASE, CBBD_API_KEY,
    current_season, season_range, BUBBLE_SEASONS,
    logger, TOURNAMENT_ROUNDS,
)


class CBBDCollector(BaseClient):
    """
    Wraps the CollegeBasketballData.com REST API.
    All endpoints require a free API key in the
    Authorization header.
    """

    def __init__(self, api_key: str = CBBD_API_KEY):
        super().__init__("cbbd", CBBD_BASE, delay=0.1)
        if not api_key:
            raise ValueError(
                "CBBD API key required. Sign up free at "
                "collegebasketballdata.com/key and add "
                "CBBD_API_KEY to your .env file."
            )
        self.session.headers["Authorization"] = f"Bearer {api_key}"

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        text = str(name).replace("-", "_")
        text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
        return text.lower()

    @classmethod
    def _normalize_dataframe_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df.rename(columns={col: cls._camel_to_snake(col) for col in df.columns})

    @classmethod
    def _normalize_record_keys(cls, record: dict) -> dict:
        return {cls._camel_to_snake(key): value for key, value in record.items()}

    # ── Venues ──────────────────────────────────────────────────────────────

    def get_venues(self) -> pd.DataFrame:
        """
        Fetch all known venues.
        Returns DataFrame with columns:
          id, source_id, name, city, state, country

        Key for Layer 2 — cross-reference tournament games
        to specific arenas by venue_id.
        """
        self.log.info("Fetching venues...")
        data = self.get("/venues")
        if not data:
            return pd.DataFrame()

        df = self._normalize_dataframe_columns(pd.DataFrame(data))
        self.log.info(f"  → {len(df)} venues loaded")
        return df

    # ── Games ────────────────────────────────────────────────────────────────

    def get_games(
        self,
        season: Optional[int] = None,
        team: Optional[str] = None,
        season_type: Optional[str] = None,   # 'regular', 'postseason'
        start_date: Optional[str] = None,    # 'YYYY-MM-DD'
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch game results with full metadata.

        Key fields returned:
          id, season, season_type, tournament
          neutral_site, conference_game
          start_date, status
          home_team, home_conference, home_points
          away_team, away_conference, away_points
          venue_id

        season_type options: 'regular', 'postseason'
        """
        params = {}
        if season:
            params["season"] = season
        if team:
            params["team"] = team
        if season_type:
            params["seasonType"] = season_type
        if start_date:
            params["startDateRange"] = start_date
        if end_date:
            params["endDateRange"] = end_date

        self.log.info(f"Fetching games {params}...")
        data = self.get("/games", params=params)
        if not data:
            return pd.DataFrame()

        df = self._normalize_dataframe_columns(pd.DataFrame(data))
        self.log.info(f"  → {len(df)} games loaded")
        return df

    def get_tournament_games(
        self,
        seasons: Optional[list[int]] = None,
        exclude_bubble: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch all NCAA tournament games across multiple seasons.
        Excludes 2021 bubble season by default.

        Used for:
          Layer 2 — Venue Scoring Index calculation
          Layer 4 — Experience Factor (prior tournament games)
        """
        if seasons is None:
            seasons = season_range(2010)

        if exclude_bubble:
            seasons = [s for s in seasons if s not in BUBBLE_SEASONS]

        all_games = []
        for season in seasons:
            self.log.info(f"  Fetching tournament games — season {season}...")
            df = self.get_games(season=season, season_type="postseason")
            if not df.empty:
                # Filter to NCAA tournament only (not NIT, CBI, etc.)
                if "tournament" in df.columns:
                    tourney = df[df["tournament"].notna()]
                else:
                    tourney = df[df.get("season_type", "") == "postseason"]
                all_games.append(tourney)

        if not all_games:
            return pd.DataFrame()

        combined = pd.concat(all_games, ignore_index=True)
        self.log.info(
            f"Tournament games loaded: {len(combined)} games "
            f"across {len(seasons)} seasons"
        )
        return combined

    def get_recent_games(
        self,
        team: str,
        season: Optional[int] = None,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Fetch the last N games for a team this season.
        Used for Layer 4 Momentum Factor.
        """
        season = season or current_season()
        df = self.get_games(season=season, team=team)

        if df.empty:
            return df

        # Filter to completed games only
        if "status" in df.columns:
            df = df[df["status"] == "final"]

        # Sort by date descending, take last N
        if "start_date" in df.columns:
            df["start_date"] = pd.to_datetime(df["start_date"])
            df = df.sort_values("start_date", ascending=False)

        return df.head(n).reset_index(drop=True)

    # ── Lines (Betting) ──────────────────────────────────────────────────────

    def get_lines(
        self,
        season: Optional[int] = None,
        team: Optional[str] = None,
        game_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical betting lines.
        Fields: game_id, provider, spread, over_under,
                home_moneyline, away_moneyline

        Used for:
          Layer 5 — comparing our predictions vs market
          Model accuracy backtesting
        """
        params = {}
        if season:
            params["season"] = season
        if team:
            params["team"] = team
        if game_id:
            params["gameId"] = game_id

        self.log.info(f"Fetching lines {params}...")
        data = self.get("/lines", params=params)
        if not data:
            return pd.DataFrame()

        rows = []
        for game in data:
            game_row = self._normalize_record_keys(game)
            book_lines = game_row.pop("lines", []) or []

            if not book_lines:
                rows.append({
                    **game_row,
                    "provider": None,
                    "spread": None,
                    "over_under": None,
                    "home_moneyline": None,
                    "away_moneyline": None,
                    "spread_open": None,
                    "over_under_open": None,
                })
                continue

            for book in book_lines:
                book_row = self._normalize_record_keys(book)
                rows.append({**game_row, **book_row})

        return self._normalize_dataframe_columns(pd.DataFrame(rows))

    @staticmethod
    def summarize_lines(lines: pd.DataFrame) -> dict:
        """
        Collapse provider-level line data into one consensus snapshot.

        Returns:
          spread: average home-team spread (negative = home favored)
          total: average over/under
          provider_count: number of books contributing data
        """
        if lines.empty:
            return {
                "spread": None,
                "total": None,
                "provider_count": 0,
            }

        spreads = pd.to_numeric(lines.get("spread"), errors="coerce").dropna()
        totals = pd.to_numeric(lines.get("over_under"), errors="coerce").dropna()
        providers = (
            lines.get("provider", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .str.strip()
        )

        return {
            "spread": round(float(spreads.mean()), 1) if len(spreads) else None,
            "total": round(float(totals.mean()), 1) if len(totals) else None,
            "provider_count": int(providers.nunique()) if len(providers) else 0,
        }

    # ── Play-by-Play ─────────────────────────────────────────────────────────

    def get_plays(
        self,
        game_id: int,
    ) -> pd.DataFrame:
        """
        Fetch play-by-play for a specific game.
        Fields: game_id, period, clock, team, play_type,
                score_value, shooting (bool), shot_location data

        Used for:
          Layer 4 Style Clash — interior vs perimeter analysis
          Layer 4 Style Clash — defensive scheme tagging
        """
        self.log.info(f"Fetching play-by-play for game {game_id}...")
        data = self.get("/plays", params={"gameId": game_id})
        if not data:
            return pd.DataFrame()

        return self._normalize_dataframe_columns(pd.DataFrame(data))

    def get_shot_locations(self, game_id: int) -> pd.DataFrame:
        """
        Extract shot attempts with locations from play-by-play.
        Filters to shooting plays only.
        Used for Style Clash interior/perimeter split.
        """
        plays = self.get_plays(game_id)
        if plays.empty:
            return plays

        # Filter to shot attempts
        if "shooting" in plays.columns:
            shots = plays[plays["shooting"] == True].copy()
        elif "play_type" in plays.columns:
            shot_types = {"FGM", "FGA", "3FGM", "3FGA"}
            shots = plays[plays["play_type"].isin(shot_types)].copy()
        else:
            shots = plays.copy()

        return shots.reset_index(drop=True)

    # ── Teams ────────────────────────────────────────────────────────────────

    def get_teams(
        self,
        season: Optional[int] = None,
        conference: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch team metadata."""
        params = {}
        if season:
            params["season"] = season
        if conference:
            params["conference"] = conference

        data = self.get("/teams", params=params)
        if not data:
            return pd.DataFrame()

        return self._normalize_dataframe_columns(pd.DataFrame(data))

    # ── Players ──────────────────────────────────────────────────────────────

    def get_players(
        self,
        team: Optional[str] = None,
        season: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch player data.
        Fields: id, name, team, position, hometown, height, weight

        Used for:
          Layer 4 Experience — class year (if available)
          Layer 4 Style Clash — size mismatch flag
        """
        params = {}
        if team:
            params["team"] = team
        if season:
            params["season"] = season

        data = self.get("/players", params=params)
        if not data:
            return pd.DataFrame()

        return self._normalize_dataframe_columns(pd.DataFrame(data))

    # ── Derived: Venue Tournament History ────────────────────────────────────

    def build_venue_game_history(
        self,
        seasons: Optional[list[int]] = None,
        exclude_bubble: bool = True,
    ) -> pd.DataFrame:
        """
        Builds the core dataset for Layer 2 Venue Scoring Index.

        Joins tournament games → venues to produce:
          venue_id, venue_name, season, round
          home_points, away_points, total_points
          neutral_site flag

        This is the foundation for VSI, VPI, V3P calculations.
        """
        self.log.info("Building venue tournament history...")

        # Get tournament games
        tourney_games = self.get_tournament_games(
            seasons=seasons,
            exclude_bubble=exclude_bubble,
        )
        if tourney_games.empty:
            self.log.warning("No tournament games found.")
            return pd.DataFrame()

        # Get venue lookup
        venues = self.get_venues()
        if venues.empty:
            self.log.warning("No venue data found.")
            return tourney_games

        # Merge venue names onto games
        venue_lookup = venues[["id", "name", "city", "state"]].rename(
            columns={"id": "venue_id", "name": "venue_name"}
        )

        # CBBD games have venue_id — merge on that
        if "venue_id" in tourney_games.columns:
            merged = tourney_games.merge(
                venue_lookup,
                on="venue_id",
                how="left",
            )
        else:
            self.log.warning("No venue_id in games data — skipping venue join")
            merged = tourney_games.copy()

        # Calculate total points per game
        home_pts = merged.get("home_points", pd.Series(dtype=float))
        away_pts = merged.get("away_points", pd.Series(dtype=float))
        merged["total_points"] = home_pts + away_pts
        merged["margin"] = home_pts - away_pts

        self.log.info(
            f"Venue history built: {len(merged)} tournament games "
            f"at {merged.get('venue_id', pd.Series()).nunique()} venues"
        )
        return merged

    # ── Derived: Team Schedule with Opponent Context ─────────────────────────

    def build_team_schedule(
        self,
        team: str,
        season: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Builds a team's full schedule with:
          - Game result (W/L)
          - Margin
          - Neutral site flag
          - Conference game flag
          - Opponent name

        Used for Layer 4 Momentum (last 10 games)
        and Layer 4 Rest & Schedule.
        """
        season = season or current_season()
        games = self.get_games(season=season, team=team)

        if games.empty:
            return games

        # Standardize: add team perspective columns
        if "home_team" in games.columns and "away_team" in games.columns:
            # Case-insensitive match — CBBD returns full names
            # (e.g. "Duke Blue Devils") while users pass short names ("Duke")
            team_lower = team.lower()
            is_home = games["home_team"].str.lower().str.contains(
                team_lower, na=False
            )
            games["team_points"] = games.apply(
                lambda r: r.get("home_points")
                          if team_lower in str(r.get("home_team", "")).lower()
                          else r.get("away_points"),
                axis=1,
            )
            games["opp_points"] = games.apply(
                lambda r: r.get("away_points")
                          if team_lower in str(r.get("home_team", "")).lower()
                          else r.get("home_points"),
                axis=1,
            )
            games["opponent"] = games.apply(
                lambda r: r["away_team"]
                          if team_lower in str(r.get("home_team", "")).lower()
                          else r["home_team"],
                axis=1,
            )
            games["is_home"] = is_home

        # Filter to completed games
        if "status" in games.columns:
            games = games[games["status"] == "final"].copy()

        # Calculate margin and result
        if "team_points" in games.columns and "opp_points" in games.columns:
            games["margin"] = (
                pd.to_numeric(games["team_points"], errors="coerce")
                - pd.to_numeric(games["opp_points"], errors="coerce")
            )
            games["result"] = games["margin"].apply(
                lambda m: "W" if m > 0 else "L"
            )

        # Sort chronologically
        if "start_date" in games.columns:
            games["start_date"] = pd.to_datetime(games["start_date"])
            games = games.sort_values("start_date").reset_index(drop=True)

        return games


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    api_key = os.environ.get("CBBD_API_KEY", "")
    if not api_key:
        print("\n⚠️  No CBBD_API_KEY found.")
        print("   Sign up free at: collegebasketballdata.com/key")
        print("   Then add to .env: CBBD_API_KEY=your_key\n")
    else:
        c = CBBDCollector(api_key)

        print("\n📍 Venues (first 5):")
        venues = c.get_venues()
        if not venues.empty:
            print(venues[["id", "name", "city", "state"]].head().to_string(index=False))

        print("\n🏀 Recent games (Duke, 2025):")
        games = c.get_recent_games("Duke", season=2025, n=5)
        if not games.empty:
            cols = [c for c in ["start_date", "home_team", "away_team",
                                "home_points", "away_points"] if c in games.columns]
            print(games[cols].to_string(index=False))
