"""
sports_oracle/collectors/ncaa_collector.py

NCAA API collector using ncaa-api.henrygd.me.

PRIMARY SOURCE FOR:
  - Official tournament bracket data (seeds, regions)
  - Game results with official scores
  - Seed matchup data for dynamic seed history
  - Backup source for scoreboard data

ACCESS:
  No API key required. Free proxy for NCAA data.
  Base: https://ncaa-api.henrygd.me

RATE LIMITING:
  Be respectful — 2 second delay between requests.
  This is a community project, not a commercial API.
"""

from __future__ import annotations
import pandas as pd
from typing import Optional
from .config import (
    BaseClient, NCAA_API_BASE,
    current_season, logger,
)


class NCAACollector(BaseClient):
    """
    Wraps the unofficial NCAA API proxy.
    Provides bracket, score, and seed data.
    """

    def __init__(self):
        super().__init__("ncaa", NCAA_API_BASE, delay=2.0)
        # NCAA API returns JSON, keep default headers
        self.session.headers["Accept"] = "application/json"

    # ── Scoreboard ────────────────────────────────────────────────────────

    def get_scoreboard(
        self,
        sport: str = "basketball-men",
        division: str = "d1",
        date: Optional[str] = None,  # 'YYYY/MM/DD'
    ) -> pd.DataFrame:
        """
        Fetch scoreboard for a given date.
        Returns game results with scores.

        date format: 'YYYY/MM/DD'
        """
        if date:
            endpoint = f"/scoreboard/{sport}/{division}/{date}"
        else:
            endpoint = f"/scoreboard/{sport}/{division}"

        data = self.get(endpoint)
        if not data:
            return pd.DataFrame()

        games = data.get("games", [])
        if not games:
            return pd.DataFrame()

        rows = []
        for game in games:
            home = game.get("home", {})
            away = game.get("away", {})

            rows.append({
                "game_id": game.get("gameID"),
                "date": game.get("startDate"),
                "status": game.get("gameState"),
                "home_team": home.get("names", {}).get("full"),
                "home_short": home.get("names", {}).get("short"),
                "home_score": home.get("score"),
                "home_seed": home.get("seed"),
                "away_team": away.get("names", {}).get("full"),
                "away_short": away.get("names", {}).get("short"),
                "away_score": away.get("score"),
                "away_seed": away.get("seed"),
                "neutral_site": game.get("neutralSite"),
                "conference_game": game.get("conferenceGame"),
                "current_period": game.get("currentPeriod"),
            })

        df = pd.DataFrame(rows)
        self.log.info(f"NCAA scoreboard: {len(df)} games")
        return df

    # ── Game Details ──────────────────────────────────────────────────────

    def get_game_details(
        self,
        game_id: str,
    ) -> dict:
        """
        Fetch detailed game data including box score.
        """
        data = self.get(f"/game/{game_id}")
        if not data:
            return {}
        return data

    # ── Tournament Bracket Seed Data ──────────────────────────────────────

    def get_tournament_scoreboard(
        self,
        season: Optional[int] = None,
        month: int = 3,
    ) -> pd.DataFrame:
        """
        Fetch tournament games by scanning March/April scoreboards.
        Tournament games have seed data populated.

        Used for:
          - Building dynamic seed history
          - Tracking bracket progress
          - Getting official seeds
        """
        season = season or current_season()
        all_games = []

        # Tournament spans mid-March through early April
        # Scan March 14-31 and April 1-10
        import datetime as dt

        year = season if month >= 8 else season  # Academic year logic
        dates_to_check = []

        # March tournament window
        for day in range(14, 32):
            try:
                d = dt.date(year, 3, day)
                dates_to_check.append(d)
            except ValueError:
                pass

        # April (Final Four typically first week)
        for day in range(1, 11):
            try:
                d = dt.date(year, 4, day)
                dates_to_check.append(d)
            except ValueError:
                pass

        for date in dates_to_check:
            date_str = date.strftime("%Y/%m/%d")
            df = self.get_scoreboard(date=date_str)
            if not df.empty:
                # Filter to games with seeds (tournament games)
                has_seed = (
                    df["home_seed"].notna() | df["away_seed"].notna()
                )
                tourney = df[has_seed]
                if not tourney.empty:
                    tourney = tourney.copy()
                    tourney["season"] = season
                    all_games.append(tourney)

        if not all_games:
            return pd.DataFrame()

        combined = pd.concat(all_games, ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_id"])

        self.log.info(
            f"Tournament games for {season}: {len(combined)} games found"
        )
        return combined

    def get_tournament_seeds(
        self,
        season: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Extract seed assignments from tournament games.
        Returns DataFrame: team, seed, season

        Used for Layer 4 Seed context and bracket predictor.
        """
        tourney = self.get_tournament_scoreboard(season)
        if tourney.empty:
            return pd.DataFrame()

        # Collect unique team-seed pairs
        seeds = {}
        for _, game in tourney.iterrows():
            if game.get("home_seed"):
                seeds[game["home_team"]] = {
                    "team": game["home_team"],
                    "seed": int(game["home_seed"]),
                    "season": game.get("season", season),
                }
            if game.get("away_seed"):
                seeds[game["away_team"]] = {
                    "team": game["away_team"],
                    "seed": int(game["away_seed"]),
                    "season": game.get("season", season),
                }

        if not seeds:
            return pd.DataFrame()

        df = pd.DataFrame(seeds.values())
        df = df.sort_values("seed").reset_index(drop=True)
        self.log.info(f"Tournament seeds: {len(df)} teams for {season}")
        return df

    # ── Seed Matchup History (for SeedHistory dynamic computation) ────────

    def build_seed_matchup_history(
        self,
        seasons: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Build a complete dataset of seed-vs-seed tournament results
        across multiple seasons. Used to feed SeedHistory with
        real data instead of hardcoded fallbacks.

        Returns DataFrame:
          season, round, higher_seed, lower_seed, higher_seed_won,
          higher_seed_team, lower_seed_team, margin

        NOTE: This is expensive — one API call per day per season.
        Cache results aggressively.
        """
        seasons = seasons or list(range(2015, current_season() + 1))
        all_matchups = []

        for season in seasons:
            self.log.info(f"  Building seed matchups for {season}...")
            tourney = self.get_tournament_scoreboard(season)
            if tourney.empty:
                continue

            for _, game in tourney.iterrows():
                home_seed = game.get("home_seed")
                away_seed = game.get("away_seed")
                home_score = game.get("home_score")
                away_score = game.get("away_score")
                status = game.get("status", "")

                # Only use completed games with both seeds
                if not all([home_seed, away_seed, home_score, away_score]):
                    continue
                if status and "final" not in str(status).lower():
                    continue

                try:
                    h_seed = int(home_seed)
                    a_seed = int(away_seed)
                    h_score = int(home_score)
                    a_score = int(away_score)
                except (ValueError, TypeError):
                    continue

                # Determine higher (better) seed
                if h_seed <= a_seed:
                    higher_seed = h_seed
                    lower_seed = a_seed
                    higher_seed_won = h_score > a_score
                    higher_team = game.get("home_team", "")
                    lower_team = game.get("away_team", "")
                    margin = h_score - a_score
                else:
                    higher_seed = a_seed
                    lower_seed = h_seed
                    higher_seed_won = a_score > h_score
                    higher_team = game.get("away_team", "")
                    lower_team = game.get("home_team", "")
                    margin = a_score - h_score

                all_matchups.append({
                    "season": season,
                    "higher_seed": higher_seed,
                    "lower_seed": lower_seed,
                    "higher_seed_team": higher_team,
                    "lower_seed_team": lower_team,
                    "higher_seed_won": higher_seed_won,
                    "margin": margin,  # positive = higher seed won by X
                })

        if not all_matchups:
            return pd.DataFrame()

        df = pd.DataFrame(all_matchups)
        self.log.info(
            f"Seed matchup history: {len(df)} games across "
            f"{df['season'].nunique()} seasons"
        )
        return df


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ncaa = NCAACollector()

    print("\n🏀 NCAA API — Scoreboard Test")
    print("=" * 50)

    # Try today's scoreboard
    board = ncaa.get_scoreboard()
    if not board.empty:
        print(f"  Today: {len(board)} games")
        cols = [c for c in ["home_team", "away_team", "home_score",
                            "away_score", "status"] if c in board.columns]
        print(board[cols].head(5).to_string(index=False))
    else:
        print("  No games today or API unreachable")

    print("\n  Seeds endpoint available: will scan tournament dates")
    print("  (skipping full scan in test — expensive)")
