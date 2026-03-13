"""
sports_oracle/collectors/odds_collector.py

The Odds API collector — betting lines and market data.

PRIMARY SOURCE FOR:
  Layer 5 — Market comparison (our spread vs Vegas spread)
  Edge detection — find mispriced games
  Backtesting — historical line accuracy

SETUP:
  1. Sign up at the-odds-api.com (free tier: 500 requests/month)
  2. Add to .env: ODDS_API_KEY=your_key_here
  3. Free tier supports: spreads, totals, moneyline, head2head

API DOCS:
  https://the-odds-api.com/liveapi/guides/v4/

RATE LIMITING:
  Free tier: 500 requests/month
  Each request returns multiple games.
  We cache aggressively — one call gets all lines for a sport.
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Optional
from ..utils.team_resolver import resolve_team

logger = logging.getLogger("sports_oracle.odds")

# The Odds API base URL
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Sport key for NCAAB
NCAAB_SPORT = "basketball_ncaab"

# Bookmakers to prioritize (in order of reliability for NCAAB)
PREFERRED_BOOKS = [
    "draftkings",
    "fanduel",
    "betmgm",
    "pointsbetus",
    "bovada",
    "betonlineag",
    "pinnacle",
    "williamhill_us",
]


class OddsCollector:
    """
    Fetches betting lines from The Odds API.
    Provides spreads, totals, and moneylines for NCAAB games.
    """

    def __init__(self, api_key: str = ""):
        from ..collectors.config import ODDS_API_KEY
        self.api_key = api_key or ODDS_API_KEY
        if not self.api_key:
            logger.warning(
                "No Odds API key configured. "
                "Sign up free at the-odds-api.com and add "
                "ODDS_API_KEY to your .env file."
            )
        self._cache: dict[str, dict] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=15)

    # ── Core: Get Current Lines ───────────────────────────────────────────

    def get_current_odds(
        self,
        markets: str = "spreads,totals,h2h",
        bookmakers: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Fetch current odds for all upcoming NCAAB games.

        markets: comma-separated list of market types
          - 'spreads': point spread
          - 'totals': over/under
          - 'h2h': moneyline / head-to-head
        bookmakers: optional list of bookmaker keys to include

        Returns list of game dicts with nested bookmaker odds.
        """
        if not self.api_key:
            logger.warning("No API key — returning empty odds")
            return []

        # Check cache
        if self._is_cache_fresh():
            return self._cache.get("current_odds", [])

        import requests

        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": markets,
            "oddsFormat": "american",
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        url = f"{ODDS_API_BASE}/sports/{NCAAB_SPORT}/odds"

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()

            # Track API usage from headers
            remaining = resp.headers.get("x-requests-remaining", "?")
            used = resp.headers.get("x-requests-used", "?")
            logger.info(
                f"Odds API: {len(resp.json())} games fetched "
                f"(API usage: {used} used, {remaining} remaining)"
            )

            games = resp.json()
            self._cache["current_odds"] = games
            self._cache_time = datetime.now()
            return games

        except Exception as e:
            logger.error(f"Odds API error: {e}")
            return self._cache.get("current_odds", [])

    def get_game_odds(
        self,
        home_team: str,
        away_team: str,
    ) -> Optional[dict]:
        """
        Get odds for a specific game by team names.
        Returns parsed odds dict or None if not found.
        """
        all_odds = self.get_current_odds()
        if not all_odds:
            return None

        # Resolve team names
        home_canonical = resolve_team(home_team) or home_team
        away_canonical = resolve_team(away_team) or away_team

        for game in all_odds:
            game_home = game.get("home_team", "")
            game_away = game.get("away_team", "")

            # Try matching (Odds API uses full names)
            home_resolved = resolve_team(game_home) or game_home
            away_resolved = resolve_team(game_away) or game_away

            if (home_resolved == home_canonical and
                    away_resolved == away_canonical):
                return self._parse_game_odds(game)

            # Try reversed (API might flip home/away)
            if (home_resolved == away_canonical and
                    away_resolved == home_canonical):
                return self._parse_game_odds(game, flip=True)

        logger.info(
            f"No odds found for {home_team} vs {away_team}"
        )
        return None

    def get_spread(
        self,
        home_team: str,
        away_team: str,
    ) -> Optional[dict]:
        """
        Get the consensus spread for a game.
        Returns: {
            'home_spread': float,
            'away_spread': float,
            'bookmaker': str,
            'home_team': str,
            'away_team': str,
        }
        """
        odds = self.get_game_odds(home_team, away_team)
        if not odds:
            return None

        spreads = odds.get("spreads", [])
        if not spreads:
            return None

        # Get spread from preferred bookmaker
        for book_key in PREFERRED_BOOKS:
            for s in spreads:
                if s.get("bookmaker") == book_key:
                    return {
                        "home_spread": s.get("home_spread"),
                        "away_spread": s.get("away_spread"),
                        "bookmaker": book_key,
                        "home_team": odds.get("home_team"),
                        "away_team": odds.get("away_team"),
                    }

        # Fallback: first available spread
        s = spreads[0]
        return {
            "home_spread": s.get("home_spread"),
            "away_spread": s.get("away_spread"),
            "bookmaker": s.get("bookmaker", "unknown"),
            "home_team": odds.get("home_team"),
            "away_team": odds.get("away_team"),
        }

    def get_total(
        self,
        home_team: str,
        away_team: str,
    ) -> Optional[dict]:
        """
        Get the consensus over/under total for a game.
        """
        odds = self.get_game_odds(home_team, away_team)
        if not odds:
            return None

        totals = odds.get("totals", [])
        if not totals:
            return None

        for book_key in PREFERRED_BOOKS:
            for t in totals:
                if t.get("bookmaker") == book_key:
                    return {
                        "total": t.get("total"),
                        "bookmaker": book_key,
                    }

        t = totals[0]
        return {"total": t.get("total"), "bookmaker": t.get("bookmaker")}

    # ── Consensus / Average Lines ─────────────────────────────────────────

    def get_consensus_lines(
        self,
        home_team: str,
        away_team: str,
    ) -> Optional[dict]:
        """
        Compute consensus (average) spread and total across all bookmakers.
        More robust than using a single bookmaker's line.
        """
        odds = self.get_game_odds(home_team, away_team)
        if not odds:
            return None

        spreads = [s["home_spread"] for s in odds.get("spreads", [])
                   if s.get("home_spread") is not None]
        totals = [t["total"] for t in odds.get("totals", [])
                  if t.get("total") is not None]

        result = {
            "home_team": odds.get("home_team"),
            "away_team": odds.get("away_team"),
            "consensus_spread": None,
            "consensus_total": None,
            "spread_bookmaker_count": len(spreads),
            "total_bookmaker_count": len(totals),
        }

        if spreads:
            result["consensus_spread"] = round(sum(spreads) / len(spreads), 1)
        if totals:
            result["consensus_total"] = round(sum(totals) / len(totals), 1)

        return result

    # ── Market Edge Detection ─────────────────────────────────────────────

    def find_edges(
        self,
        our_spread: float,
        our_total: float,
        home_team: str,
        away_team: str,
        threshold: float = 2.0,
    ) -> dict:
        """
        Compare our predictions against market lines.
        Returns edge analysis if our line differs significantly
        from the market consensus.

        threshold: minimum point difference to flag as an edge
        """
        consensus = self.get_consensus_lines(home_team, away_team)
        if not consensus:
            return {"has_edge": False, "reason": "no market data"}

        result = {
            "home_team": home_team,
            "away_team": away_team,
            "our_spread": our_spread,
            "our_total": our_total,
            "market_spread": consensus["consensus_spread"],
            "market_total": consensus["consensus_total"],
            "spread_edge": None,
            "total_edge": None,
            "has_edge": False,
            "edges": [],
        }

        # Spread edge
        if consensus["consensus_spread"] is not None:
            diff = our_spread - consensus["consensus_spread"]
            result["spread_edge"] = round(diff, 1)
            if abs(diff) >= threshold:
                result["has_edge"] = True
                if diff > 0:
                    result["edges"].append(
                        f"Model favors {away_team} by {abs(diff):.1f} more "
                        f"than market (bet {away_team} +{consensus['consensus_spread']})"
                    )
                else:
                    result["edges"].append(
                        f"Model favors {home_team} by {abs(diff):.1f} more "
                        f"than market (bet {home_team} {consensus['consensus_spread']})"
                    )

        # Total edge
        if consensus["consensus_total"] is not None:
            diff = our_total - consensus["consensus_total"]
            result["total_edge"] = round(diff, 1)
            if abs(diff) >= threshold:
                result["has_edge"] = True
                direction = "Over" if diff > 0 else "Under"
                result["edges"].append(
                    f"Model projects {abs(diff):.1f} pts "
                    f"{'higher' if diff > 0 else 'lower'} than market "
                    f"({direction} {consensus['consensus_total']})"
                )

        return result

    # ── Internal ──────────────────────────────────────────────────────────

    def _parse_game_odds(self, game: dict, flip: bool = False) -> dict:
        """Parse a raw Odds API game object into our format."""
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")

        if flip:
            home_team, away_team = away_team, home_team

        result = {
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": game.get("commence_time"),
            "sport": game.get("sport_key"),
            "spreads": [],
            "totals": [],
            "moneylines": [],
        }

        for bookmaker in game.get("bookmakers", []):
            book_key = bookmaker.get("key", "")

            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                outcomes = market.get("outcomes", [])

                if market_key == "spreads":
                    home_out = next(
                        (o for o in outcomes if o.get("name") == home_team),
                        {}
                    )
                    away_out = next(
                        (o for o in outcomes if o.get("name") == away_team),
                        {}
                    )
                    if flip:
                        home_out, away_out = away_out, home_out

                    result["spreads"].append({
                        "bookmaker": book_key,
                        "home_spread": home_out.get("point"),
                        "away_spread": away_out.get("point"),
                        "home_price": home_out.get("price"),
                        "away_price": away_out.get("price"),
                    })

                elif market_key == "totals":
                    over = next(
                        (o for o in outcomes if o.get("name") == "Over"), {}
                    )
                    under = next(
                        (o for o in outcomes if o.get("name") == "Under"), {}
                    )
                    result["totals"].append({
                        "bookmaker": book_key,
                        "total": over.get("point"),
                        "over_price": over.get("price"),
                        "under_price": under.get("price"),
                    })

                elif market_key == "h2h":
                    home_out = next(
                        (o for o in outcomes if o.get("name") == home_team),
                        {}
                    )
                    away_out = next(
                        (o for o in outcomes if o.get("name") == away_team),
                        {}
                    )
                    if flip:
                        home_out, away_out = away_out, home_out

                    result["moneylines"].append({
                        "bookmaker": book_key,
                        "home_ml": home_out.get("price"),
                        "away_ml": away_out.get("price"),
                    })

        return result

    def _is_cache_fresh(self) -> bool:
        """Check if cached odds data is still fresh."""
        if not self._cache_time:
            return False
        return datetime.now() - self._cache_time < self._cache_ttl

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    key = os.environ.get("ODDS_API_KEY", "")
    collector = OddsCollector(api_key=key)

    if collector.is_configured:
        print("\n📊 Odds API — Current NCAAB Lines")
        print("=" * 50)
        odds = collector.get_current_odds()
        print(f"  {len(odds)} games with odds")
        for game in odds[:5]:
            print(f"  {game.get('away_team')} @ {game.get('home_team')}")
    else:
        print("\n⚠️  No ODDS_API_KEY configured.")
        print("   Sign up free at: the-odds-api.com")
        print("   Then add to .env: ODDS_API_KEY=your_key")
        print("\n   OddsCollector will return empty results without a key.")
        print(f"   is_configured: {collector.is_configured}")
