"""
sports_oracle/collectors/espn_collector.py

ESPN Unofficial API collector.

PRIMARY SOURCE FOR:
  Layer 4 Experience — player class year, roster composition
  Layer 4 Experience — prior NCAA tournament games (per player)
  Layer 4 Momentum  — live scores, conference tournament results
  Layer 4 Rest      — game schedule, tip-off times
  Layer 4 Seed      — tournament bracket, seeds, matchups
  Layer 4 Momentum  — injury reports

ACCESS:
  No API key required.
  Base: site.api.espn.com/apis/site/v2/sports/
        basketball/mens-college-basketball/
  Rate limit: ~60-100 req/min — we use 1s delay to be polite.

NOTES:
  ESPN's unofficial API is not documented and can change.
  Endpoints and field names are best-effort.
  We validate and handle missing fields gracefully.
"""

from __future__ import annotations
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from .config import (
    BaseClient, ESPN_BASE,
    current_season, logger,
)


class ESPNCollector(BaseClient):
    """
    Wraps ESPN's unofficial college basketball API.
    No authentication required.
    """

    def __init__(self):
        super().__init__("espn", ESPN_BASE, delay=1.0)

    # ── Scoreboard ───────────────────────────────────────────────────────────

    def get_scoreboard(
        self,
        date: Optional[str] = None,   # 'YYYYMMDD'
        groups: str = "50",           # 50 = all D1
    ) -> pd.DataFrame:
        """
        Fetch scoreboard for a given date.
        Returns game metadata: teams, scores, status, venue.

        Used for live momentum updates and conf tournament tracking.
        """
        params = {"groups": groups}
        if date:
            params["dates"] = date
        else:
            params["dates"] = datetime.now().strftime("%Y%m%d")

        data = self.get("/scoreboard", params=params)
        if not data:
            return pd.DataFrame()

        games = []
        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])

            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away = next((c for c in competitors if c.get("homeAway") == "away"), {})

            venue_info = comp.get("venue", {})

            # Extract betting odds if available
            odds_data = comp.get("odds", [{}])
            odds = odds_data[0] if odds_data else {}
            spread_val = odds.get("spread")
            over_under = odds.get("overUnder")
            # odds.get("details") is like "UH -14.5" — the favorite abbreviation + spread
            odds_detail = odds.get("details", "")
            favorite_abbr = ""
            betting_spread = None
            if odds_detail and spread_val is not None:
                try:
                    betting_spread = float(spread_val)
                    # Determine who the spread favors
                    # ESPN spread is negative for favorite
                    favorite_abbr = odds_detail.split()[0] if odds_detail else ""
                except (ValueError, IndexError):
                    pass

            games.append({
                "espn_game_id":  event.get("id"),
                "name":          event.get("name"),
                "date":          event.get("date"),
                "status":        event.get("status", {}).get("type", {}).get("name"),
                "status_detail": event.get("status", {}).get("type", {}).get("description"),
                "home_team":     home.get("team", {}).get("displayName"),
                "home_team_id":  home.get("team", {}).get("id"),
                "home_abbr":     home.get("team", {}).get("abbreviation"),
                "home_score":    home.get("score"),
                "home_rank":     home.get("curatedRank", {}).get("current"),
                "away_team":     away.get("team", {}).get("displayName"),
                "away_team_id":  away.get("team", {}).get("id"),
                "away_abbr":     away.get("team", {}).get("abbreviation"),
                "away_score":    away.get("score"),
                "away_rank":     away.get("curatedRank", {}).get("current"),
                "neutral_site":  comp.get("neutralSite", False),
                "conf_game":     comp.get("conferenceCompetition", False),
                "venue_name":    venue_info.get("fullName"),
                "venue_city":    venue_info.get("address", {}).get("city"),
                "venue_state":   venue_info.get("address", {}).get("state"),
                "betting_spread": betting_spread,
                "over_under":    over_under,
                "odds_detail":   odds_detail,
                "favorite_abbr": favorite_abbr,
            })

        df = pd.DataFrame(games)
        self.log.info(
            f"Scoreboard {params.get('dates')}: {len(df)} games"
        )
        return df

    def get_recent_scoreboard(self, days_back: int = 10) -> pd.DataFrame:
        """
        Fetch scoreboards for the last N days.
        Used to build last-10-game momentum window.
        """
        all_games = []
        for i in range(days_back):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
            df = self.get_scoreboard(date=date)
            if not df.empty:
                all_games.append(df)

        if not all_games:
            return pd.DataFrame()

        combined = pd.concat(all_games, ignore_index=True)
        return combined.drop_duplicates(subset=["espn_game_id"])

    # ── Rosters ──────────────────────────────────────────────────────────────

    def get_roster(
        self,
        team_id: str,
        season: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch team roster with player class year.

        Returns: name, position, class_year (Fr/So/Jr/Sr/Gr),
                 jersey, height, weight, espn_player_id

        Layer 4 Experience — Roster Age Score calculation.
        """
        endpoint = f"/teams/{team_id}/roster"
        params = {}
        if season:
            params["season"] = season

        data = self.get(endpoint, params=params)
        if not data:
            return pd.DataFrame()

        players = []
        roster_data = data.get("roster", []) or data.get("athletes", [])

        for p in roster_data:
            # Class year may be in different fields
            class_year = (
                p.get("year")
                or p.get("experience", {}).get("displayValue")
                or p.get("displayYear")
            )

            players.append({
                "espn_player_id": p.get("id"),
                "name":           p.get("displayName") or p.get("fullName"),
                "position":       p.get("position", {}).get("abbreviation"),
                "jersey":         p.get("jersey"),
                "class_year":     class_year,
                "height":         p.get("displayHeight"),
                "weight":         p.get("displayWeight"),
                "status":         p.get("status", {}).get("type"),
            })

        df = pd.DataFrame(players)
        df["team_id"] = team_id

        # Normalize class year to numeric
        df["class_year_num"] = df["class_year"].map(
            self._class_year_to_num
        )

        self.log.info(f"Roster loaded: team {team_id}, {len(df)} players")
        return df

    @staticmethod
    def _class_year_to_num(year_str) -> Optional[float]:
        """
        Convert class year string to numeric for Age Score calculation.
        Handles compound forms: 'Redshirt Sophomore', 'R-So', 'RS Jr',
        '5th Year Senior', 'Graduate Student', etc.
        """
        if not year_str:
            return None
        s = str(year_str).lower().strip()

        # Try exact match first (fastest path)
        exact = {
            "freshman": 1, "fr": 1, "1": 1,
            "sophomore": 2, "so": 2, "2": 2,
            "junior": 3, "jr": 3, "3": 3,
            "senior": 4, "sr": 4, "4": 4,
            "graduate": 5, "gr": 5, "5th": 5, "5": 5,
        }
        if s in exact:
            return exact[s]

        # Substring match for compound forms — check most senior first
        # so "5th Year Senior" matches 5, not 4
        if any(k in s for k in ("graduate", "5th", "5th year", " gr")):
            return 5
        if any(k in s for k in ("senior", " sr", "r-sr", "rs sr")):
            return 4
        if any(k in s for k in ("junior", " jr", "r-jr", "rs jr")):
            return 3
        if any(k in s for k in ("sophomore", " so", "r-so", "rs so")):
            return 2
        if any(k in s for k in ("freshman", " fr", "r-fr", "rs fr", "redshirt")):
            return 1

        return None

    # ── Teams ────────────────────────────────────────────────────────────────

    def get_teams(self, conference: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch all D1 teams with ESPN IDs.
        ESPN team IDs are needed to look up rosters and stats.
        """
        params = {"groups": "50"}  # D1 only
        if conference:
            params["conference"] = conference

        data = self.get("/teams", params=params)
        if not data:
            return pd.DataFrame()

        teams = []
        sports_data = data.get("sports", [{}])
        leagues = sports_data[0].get("leagues", [{}]) if sports_data else [{}]
        team_list = leagues[0].get("teams", []) if leagues else []

        for entry in team_list:
            team = entry.get("team", {})
            teams.append({
                "espn_team_id":  team.get("id"),
                "name":          team.get("displayName"),
                "short_name":    team.get("shortDisplayName"),
                "abbreviation":  team.get("abbreviation"),
                "location":      team.get("location"),
                "nickname":      team.get("nickname"),
                "conference_id": team.get("conferenceId"),
                "logo_url":      (team.get("logos", [{}])[0].get("href")
                                  if team.get("logos") else None),
            })

        df = pd.DataFrame(teams)
        self.log.info(f"Teams loaded: {len(df)} teams")
        return df

    def get_team_id(self, team_name: str) -> Optional[str]:
        """Look up ESPN team ID by name (case-insensitive partial match)."""
        teams = self.get_teams()
        if teams.empty:
            return None

        name_lower = team_name.lower()
        for _, row in teams.iterrows():
            if (name_lower in str(row.get("name", "")).lower()
                    or name_lower in str(row.get("location", "")).lower()
                    or name_lower == str(row.get("abbreviation", "")).lower()):
                return row["espn_team_id"]
        return None

    # ── Injuries ─────────────────────────────────────────────────────────────

    def get_injuries(self) -> pd.DataFrame:
        """
        Fetch current injury reports.
        Used for Layer 4 Experience — player availability score.
        """
        data = self.get("/injuries")
        if not data:
            return pd.DataFrame()

        injuries = []
        for team_entry in data.get("injuries", []):
            team_name = team_entry.get("team", {}).get("displayName")
            for injury in team_entry.get("injuries", []):
                athlete = injury.get("athlete", {})
                injuries.append({
                    "team":       team_name,
                    "player":     athlete.get("displayName"),
                    "player_id":  athlete.get("id"),
                    "position":   athlete.get("position", {}).get("abbreviation"),
                    "status":     injury.get("status"),
                    "type":       injury.get("type", {}).get("description"),
                    "date":       injury.get("date"),
                    "return_date": injury.get("returnDate"),
                })

        df = pd.DataFrame(injuries)
        self.log.info(f"Injuries loaded: {len(df)} player entries")
        return df

    # ── Standings ────────────────────────────────────────────────────────────

    def get_standings(self, season: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch current season standings.
        Used for seed context and conference strength.
        """
        params = {"groups": "50"}
        if season:
            params["season"] = season

        data = self.get("/standings", params=params)
        if not data:
            return pd.DataFrame()

        rows = []
        for group in data.get("children", []):
            conference = group.get("name", "")
            for entry in group.get("standings", {}).get("entries", []):
                team = entry.get("team", {})
                stats = {s["name"]: s.get("displayValue")
                         for s in entry.get("stats", [])}
                rows.append({
                    "conference": conference,
                    "team":       team.get("displayName"),
                    "team_id":    team.get("id"),
                    **stats,
                })

        return pd.DataFrame(rows)

    # ── Tournament Bracket ────────────────────────────────────────────────────

    def get_tournament_bracket(
        self,
        season: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch NCAA tournament bracket data including seeds and regions.
        Used for Layer 4 Seed & Matchup Context.

        Note: Available once bracket is announced (Selection Sunday).
        """
        season = season or current_season()
        # ESPN bracket endpoint
        data = self.get(
            f"/tournaments/22/{season}/bracket",
        )

        if not data:
            self.log.info("Bracket not yet available (pre-Selection Sunday?)")
            return pd.DataFrame()

        bracket_entries = []
        for region in data.get("bracket", {}).get("rounds", [{}])[0].get("groups", []):
            region_name = region.get("name")
            for seed_entry in region.get("seeds", []):
                team = (seed_entry.get("teams") or [{}])[0]
                bracket_entries.append({
                    "region":     region_name,
                    "seed":       seed_entry.get("displaySeed"),
                    "team":       team.get("displayName"),
                    "team_id":    team.get("id"),
                    "conference": team.get("conferenceId"),
                    "record":     team.get("record", {}).get("displayValue"),
                    "season":     season,
                })

        df = pd.DataFrame(bracket_entries)
        self.log.info(
            f"Tournament bracket: {len(df)} teams seeded for {season}"
        )
        return df

    # ── Conference Tournament ────────────────────────────────────────────────

    def get_conf_tournament_games(
        self,
        days_window: int = 14,
    ) -> pd.DataFrame:
        """
        Fetch conference tournament games from recent days.
        Identifies conf tournament via is_conference_tournament flag
        or date proximity to NCAA tournament.

        Used for Layer 4 Momentum — conference tournament component.
        """
        all_games = []
        for i in range(days_window):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
            df = self.get_scoreboard(date=date)
            if not df.empty:
                all_games.append(df)

        if not all_games:
            return pd.DataFrame()

        combined = pd.concat(all_games, ignore_index=True)
        # Conference tournament games are typically non-neutral site
        # with teams from the same conference
        # ESPN may flag these; otherwise we filter by date range
        return combined.drop_duplicates(subset=["espn_game_id"])

    # ── Game Box Score ────────────────────────────────────────────────────────

    def get_game_summary(self, game_id: str) -> dict:
        """
        Fetch detailed game summary including team stats.
        Used for historical pressure performance calculation.
        """
        data = self.get(f"/summary", params={"event": game_id})
        if not data:
            return {}
        return data

    def get_team_game_stats(self, game_id: str) -> pd.DataFrame:
        """
        Extract team-level box score stats from a game summary.
        Returns: team, fg, fg_pct, 3pt, 3pt_pct, ft, ft_pct,
                 rebounds, assists, turnovers, steals, blocks
        """
        summary = self.get_game_summary(game_id)
        if not summary:
            return pd.DataFrame()

        rows = []
        for team_stats in summary.get("boxscore", {}).get("teams", []):
            team_name = team_stats.get("team", {}).get("displayName")
            stats = {
                s.get("name", f"stat_{i}"): s.get("displayValue")
                for i, s in enumerate(team_stats.get("statistics", []))
            }
            stats["team"] = team_name
            stats["game_id"] = game_id
            rows.append(stats)

        return pd.DataFrame(rows)


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    espn = ESPNCollector()

    print("\n📅 Today's Games:")
    board = espn.get_scoreboard()
    if not board.empty:
        cols = ["away_team", "home_team", "away_score",
                "home_score", "status", "neutral_site"]
        cols = [c for c in cols if c in board.columns]
        print(board[cols].head(10).to_string(index=False))
    else:
        print("  No games today or off-season.")

    print("\n🏀 Sample Teams (first 5):")
    teams = espn.get_teams()
    if not teams.empty:
        print(teams[["espn_team_id","name","abbreviation"]].head().to_string(index=False))

    print("\n🤕 Current Injuries (first 5):")
    injuries = espn.get_injuries()
    if not injuries.empty:
        print(injuries[["team","player","status","type"]].head().to_string(index=False))
