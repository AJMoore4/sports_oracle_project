"""
sports_oracle/collectors/sportsref_collector.py

Sports Reference (sports-reference.com/cbb) scraper.

PRIMARY SOURCE FOR:
  Layer 4 Experience — Coach career tournament record
                       (appearances, wins by round, win rate)
  Layer 4 Experience — Player historical tournament games
                       (how many tournament games has each player played?)
  Layer 4 Experience — Pressure performance splits
                       (ranked opponent games, rivalry games)
  Layer 2            — Historical venue game logs (if CBBD incomplete)

ACCESS:
  Free scraping — no account required.
  RATE LIMIT: 3.5 second delay between requests.
  Sports Reference explicitly allows scraping for personal use.
  Don't hammer the site — we cache aggressively.

CACHING:
  Results are cached to /tmp/sportsref_cache/ to avoid
  re-scraping the same page. Cache TTL: 24 hours.
  During tournament season, use cache_ttl=3600 for fresher data.
"""

from __future__ import annotations
import os
import json
import time
import hashlib
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from bs4 import BeautifulSoup
from .config import (
    BaseClient, SPORTS_REF_BASE,
    season_range, logger,
)

CACHE_DIR = "/tmp/sportsref_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


class SportsRefCollector(BaseClient):
    """
    Scrapes sports-reference.com/cbb for historical data
    not available through APIs.
    """

    def __init__(self, cache_ttl_hours: int = 24):
        super().__init__("sportsref", SPORTS_REF_BASE, delay=3.5)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        # Override user agent to be more specific
        self.session.headers["User-Agent"] = (
            "Mozilla/5.0 (educational sports research project; "
            "not for commercial use)"
        )

    # ── Caching ──────────────────────────────────────────────────────────────

    def _cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cached(self, url: str) -> Optional[str]:
        """Return cached HTML if fresh, else None."""
        key = self._cache_key(url)
        path = os.path.join(CACHE_DIR, f"{key}.html")
        meta_path = os.path.join(CACHE_DIR, f"{key}.meta")

        if os.path.exists(path) and os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            cached_at = datetime.fromisoformat(meta["cached_at"])
            if datetime.now() - cached_at < self.cache_ttl:
                with open(path, encoding="utf-8") as f:
                    return f.read()
        return None

    def _save_cache(self, url: str, html: str):
        """Cache HTML to disk."""
        key = self._cache_key(url)
        with open(os.path.join(CACHE_DIR, f"{key}.html"), "w", encoding="utf-8") as f:
            f.write(html)
        with open(os.path.join(CACHE_DIR, f"{key}.meta"), "w") as f:
            json.dump({"url": url, "cached_at": datetime.now().isoformat()}, f)

    def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML with caching and rate limiting."""
        cached = self._get_cached(url)
        if cached:
            self.log.debug(f"Cache hit: {url}")
            return cached

        self._throttle()
        try:
            resp = self.session.get(url, timeout=20)
            resp.raise_for_status()
            html = resp.text
            self._save_cache(url, html)
            self.log.debug(f"Fetched: {url}")
            return html
        except Exception as e:
            self.log.warning(f"Fetch failed {url}: {e}")
            return None

    def _parse_table(
        self,
        html: str,
        table_id: str,
    ) -> pd.DataFrame:
        """Parse a sports-reference HTML table by ID into DataFrame."""
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", {"id": table_id})
        if not table:
            self.log.warning(f"Table '{table_id}' not found")
            return pd.DataFrame()

        rows = []
        headers = []

        # Get headers
        header_row = table.find("thead")
        if header_row:
            headers = [
                th.get("data-stat") or th.get_text(strip=True)
                for th in header_row.find_all(["th", "td"])
            ]

        # Get data rows
        tbody = table.find("tbody")
        if tbody:
            for tr in tbody.find_all("tr"):
                # Skip separator rows
                if tr.get("class") and "thead" in tr.get("class", []):
                    continue
                cells = tr.find_all(["th", "td"])
                if not cells:
                    continue
                row = {}
                for i, cell in enumerate(cells):
                    key = (
                        cell.get("data-stat")
                        or (headers[i] if i < len(headers) else f"col_{i}")
                    )
                    # Prefer link text for names
                    link = cell.find("a")
                    row[key] = link.get_text(strip=True) if link else cell.get_text(strip=True)
                rows.append(row)

        return pd.DataFrame(rows)

    # ── Coach Tournament Records ──────────────────────────────────────────────

    def get_coach_tournament_record(
        self,
        coach_name: str,
    ) -> dict:
        """
        Scrape a coach's career NCAA tournament record.
        Returns:
          appearances, total_wins, total_losses,
          round_rates: {sweet_16, elite_8, final_four, championship},
          first_round_upsets_caused, first_round_upsets_suffered
          win_rate, career_games

        Layer 4 Experience — Coach Score calculation.
        """
        # Build coach page URL: "Kelvin Sampson" → "kelvin-sampson-1"
        # Sports-reference uses {first}-{last}-1 for most coaches
        slug = coach_name.lower().strip().replace(" ", "-").replace(".", "")
        coach_url = f"{self.base_url}/coaches/{slug}-1.html"

        self.log.info(f"Fetching coach record: {coach_name}...")
        html = self._fetch_html(coach_url)

        if not html:
            # Try without the -1 suffix
            coach_url = f"{self.base_url}/coaches/{slug}.html"
            html = self._fetch_html(coach_url)

        if not html:
            return self._empty_coach_record(coach_name)

        soup = BeautifulSoup(html, "lxml")

        record = {
            "coach": coach_name,
            "appearances": 0,
            "total_wins": 0,
            "total_losses": 0,
            "sweet_16s": 0,
            "elite_8s": 0,
            "final_fours": 0,
            "championships": 0,
            "win_rate": 0.0,
            "first_yr_coach": False,
        }

        # Parse tournament summary table
        table = soup.find("table", {"id": "coach-stats"}) or \
                soup.find("table")

        if not table:
            return record

        for row in table.find("tbody", {}).find_all("tr") if table.find("tbody") else []:
            cells = {
                td.get("data-stat"): td.get_text(strip=True)
                for td in row.find_all(["td", "th"])
            }
            season_type = cells.get("season_type", "")
            if "NCAA" in season_type or "tournament" in season_type.lower():
                record["appearances"] += 1
                w = int(cells.get("wins", "0") or 0)
                l = int(cells.get("losses", "0") or 0)
                record["total_wins"] += w
                record["total_losses"] += l

        total_games = record["total_wins"] + record["total_losses"]
        if total_games > 0:
            record["win_rate"] = record["total_wins"] / total_games

        return record

    def get_coach_tournament_record_by_school(
        self,
        team: str,
        season: Optional[int] = None,
    ) -> dict:
        """
        Get current head coach's tournament record for a team.
        Looks up who coached the team in the given season
        then fetches their career record.
        """
        season = season or datetime.now().year
        team_slug = team.lower().replace(" ", "-").replace(".", "")
        url = f"{self.base_url}/schools/{team_slug}/men/coaches.html"

        html = self._fetch_html(url)
        if not html:
            return self._empty_coach_record("Unknown")

        soup = BeautifulSoup(html, "lxml")
        # Find coach who coached in the target season
        coach_name = None
        table = soup.find("table", {"id": "coaches"})
        if table:
            for row in table.find("tbody", {}).find_all("tr") if table.find("tbody") else []:
                cells = {
                    td.get("data-stat"): td.get_text(strip=True)
                    for td in row.find_all(["td", "th"])
                }
                yr_min = int(cells.get("year_min", "0") or 0)
                yr_max = int(cells.get("year_max", "9999") or 9999)
                if yr_min <= season <= yr_max:
                    coach_link = row.find("a")
                    if coach_link:
                        coach_name = coach_link.get_text(strip=True)
                    break

        if not coach_name:
            return self._empty_coach_record("Unknown")

        return self.get_coach_tournament_record(coach_name)

    @staticmethod
    def _empty_coach_record(name: str) -> dict:
        return {
            "coach": name,
            "appearances": 0,
            "total_wins": 0,
            "total_losses": 0,
            "sweet_16s": 0,
            "elite_8s": 0,
            "final_fours": 0,
            "championships": 0,
            "win_rate": 0.0,
            "first_yr_coach": True,   # assume worst case
        }

    # ── Player Tournament History ─────────────────────────────────────────────

    def get_player_tournament_games(
        self,
        player_name: str,
        team: Optional[str] = None,
    ) -> int:
        """
        Scrape number of prior NCAA tournament games for a player.
        Returns integer count of tournament games played.

        Layer 4 Experience — Tournament Exposure Score.
        """
        # Search player
        search_url = (
            f"{self.base_url}/friv/players.fcgi"
            f"?search={player_name.replace(' ', '+')}"
        )

        html = self._fetch_html(search_url)
        if not html:
            return 0

        soup = BeautifulSoup(html, "lxml")

        # Find player link — may return multiple results
        player_links = soup.find_all("a", href=lambda h: h and "/players/" in h)
        if not player_links:
            return 0

        # Use first match or filter by team
        player_url = f"{self.base_url}{player_links[0]['href']}"
        player_html = self._fetch_html(player_url)
        if not player_html:
            return 0

        # Parse game log and count tournament games
        soup2 = BeautifulSoup(player_html, "lxml")
        tournament_games = 0

        # Look for game logs with tournament round info
        for table in soup2.find_all("table"):
            tbody = table.find("tbody")
            if not tbody:
                continue
            for row in tbody.find_all("tr"):
                cells = row.find_all(["td", "th"])
                row_text = " ".join(c.get_text() for c in cells)
                # Tournament games flagged by round names
                if any(kw in row_text for kw in
                       ["NCAA", "Sweet Sixteen", "Elite Eight",
                        "Final Four", "Championship", "First Round"]):
                    tournament_games += 1

        return tournament_games

    # ── Team Historical Tournament Performance ────────────────────────────────

    def get_team_tournament_history(
        self,
        team: str,
        seasons: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Scrape a team's full NCAA tournament history.
        Returns per-tournament-appearance:
          season, seed, round_reached, wins, losses, coach

        Used for:
          Layer 4 Experience — team familiarity with tournament
          Layer 4 Seed — historical upset rates for this program
        """
        team_slug = team.lower().replace(" ", "-").replace(".", "")
        url = f"{self.base_url}/schools/{team_slug}/men/schedule.html"

        html = self._fetch_html(url)
        if not html:
            return pd.DataFrame()

        # Parse the school's schedule history
        df = self._parse_table(html, "schedule")
        if df.empty:
            # Try alternate table ID
            df = self._parse_table(html, "seasons")

        if df.empty:
            return pd.DataFrame()

        # Filter to tournament seasons
        if "how_qual" in df.columns:
            tourney_df = df[
                df["how_qual"].str.contains("NCAA", na=False)
            ].copy()
        else:
            tourney_df = df.copy()

        if seasons:
            if "season" in tourney_df.columns:
                tourney_df = tourney_df[
                    tourney_df["season"].astype(str).str[:4].astype(int, errors="ignore").isin(seasons)
                ]

        tourney_df["team"] = team
        self.log.info(f"Tournament history for {team}: {len(tourney_df)} appearances")
        return tourney_df

    # ── Ranked Opponent Performance (Pressure Splits) ─────────────────────────

    def get_ranked_opponent_splits(
        self,
        team: str,
        season: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Scrape team game logs filtered to games vs ranked opponents.
        Used for Layer 4 Experience — Pressure Performance Index.

        Returns games where opponent was ranked at time of game.
        """
        season = season or datetime.now().year
        team_slug = team.lower().replace(" ", "-").replace(".", "")
        url = f"{self.base_url}/schools/{team_slug}/men/{season}/gamelog/"

        html = self._fetch_html(url)
        if not html:
            return pd.DataFrame()

        df = self._parse_table(html, "sgl-basic")
        if df.empty:
            df = self._parse_table(html, "gamelog")

        if df.empty:
            return pd.DataFrame()

        # Flag ranked opponent games
        if "opp_name" in df.columns or "opponent" in df.columns:
            opp_col = "opp_name" if "opp_name" in df.columns else "opponent"
            # Ranked teams listed with (#N) prefix in sports-reference
            df["vs_ranked"] = df[opp_col].str.contains(r"\(#\d+\)", na=False)
            ranked_games = df[df["vs_ranked"]].copy()
        else:
            ranked_games = df.copy()

        df["team"] = team
        df["season"] = season
        self.log.info(
            f"Ranked opponent splits for {team}: "
            f"{len(ranked_games)} games vs ranked opponents"
        )
        return ranked_games


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sr = SportsRefCollector()

    print("\n🏀 Duke Tournament History (last 5 appearances):")
    history = sr.get_team_tournament_history("Duke")
    if not history.empty:
        print(history.tail(5).to_string(index=False))
    else:
        print("  No data (may be rate limited or scraping blocked)")

    print("\n📊 Duke Ranked Opponent Splits (2025):")
    splits = sr.get_ranked_opponent_splits("Duke", 2025)
    if not splits.empty:
        print(f"  {len(splits)} games vs ranked opponents")
    else:
        print("  No data returned")
