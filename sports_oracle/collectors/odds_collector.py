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
import json
import logging
import os
import requests
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from ..utils.team_resolver import get_resolver

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
        self.resolver = get_resolver()
        self._cache: dict[str, object] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=15)
        self._last_request = 0.0
        self._min_delay = 0.35
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self._file_cache_dir = os.path.join(project_root, "data", "odds_cache")
        os.makedirs(self._file_cache_dir, exist_ok=True)

    # ── Core: Get Current Lines ───────────────────────────────────────────

    def get_current_odds(
        self,
        markets: str = "spreads,totals,h2h",
        bookmakers: Optional[list[str]] = None,
        regions: str = "us",
        commence_time_from: Optional[str] = None,
        commence_time_to: Optional[str] = None,
        cache_label: Optional[str] = None,
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

        if (
            commence_time_from is None
            and commence_time_to is None
            and bookmakers is None
            and regions == "us"
            and self._is_cache_fresh()
        ):
            return self._cache.get("current_odds", [])

        url = f"{ODDS_API_BASE}/sports/{NCAAB_SPORT}/odds"
        games = self._fetch_sport_feed(
            url=url,
            params=self._build_feed_params(
                markets=markets,
                regions=regions,
                bookmakers=bookmakers,
                commence_time_from=commence_time_from,
                commence_time_to=commence_time_to,
            ),
            session_cache_key=self._build_session_cache_key(
                prefix="current_odds",
                markets=markets,
                regions=regions,
                bookmakers=bookmakers,
                commence_time_from=commence_time_from,
                commence_time_to=commence_time_to,
            ),
            cache_category="current",
            cache_label=cache_label,
            cache_ttl=self._cache_ttl,
        )
        if (
            commence_time_from is None
            and commence_time_to is None
            and bookmakers is None
            and regions == "us"
        ):
            self._cache["current_odds"] = games
            self._cache_time = datetime.now()
        return games

    def get_current_odds_for_day(
        self,
        cache_label: str,
        commence_time_from: Optional[str] = None,
        commence_time_to: Optional[str] = None,
        markets: str = "spreads,totals,h2h",
        bookmakers: Optional[list[str]] = None,
        regions: str = "us",
    ) -> list[dict]:
        """Fetch and cache one current Odds API feed for a scoreboard day."""
        return self.get_current_odds(
            markets=markets,
            bookmakers=bookmakers,
            regions=regions,
            commence_time_from=commence_time_from,
            commence_time_to=commence_time_to,
            cache_label=cache_label,
        )

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

        return self.get_game_odds_from_feed(
            all_odds=all_odds,
            home_team=home_team,
            away_team=away_team,
        )

    def get_game_odds_from_feed(
        self,
        all_odds: list[dict],
        home_team: str,
        away_team: str,
    ) -> Optional[dict]:
        """
        Match a game against a previously fetched Odds API feed.
        """
        if not all_odds:
            return None

        for game in all_odds:
            if self._matchup_matches(
                home_team,
                away_team,
                game.get("home_team", ""),
                game.get("away_team", ""),
            ):
                return self._parse_game_odds(game)

            # Try reversed (API might flip home/away)
            if self._matchup_matches(
                away_team,
                home_team,
                game.get("home_team", ""),
                game.get("away_team", ""),
            ):
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
        return self._build_consensus_from_odds(odds)

    def get_historical_events(
        self,
        snapshot_time: str,
    ) -> list[dict]:
        """
        Fetch the Odds API historical event list at a specific timestamp.
        """
        if not self.api_key:
            return []

        cache_key = f"historical_events_{snapshot_time}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{ODDS_API_BASE}/historical/sports/{NCAAB_SPORT}/events"
        params = {
            "apiKey": self.api_key,
            "date": snapshot_time,
        }

        try:
            self._throttle()
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            events = payload.get("data", []) if isinstance(payload, dict) else []
            self._cache[cache_key] = events
            return events
        except Exception as e:
            logger.error(f"Historical events lookup failed: {e}")
            return self._cache.get(cache_key, [])

    def get_historical_event_odds(
        self,
        event_id: str,
        snapshot_time: str,
        markets: str = "spreads,totals,h2h",
        regions: str = "us",
    ) -> Optional[dict]:
        """
        Fetch historical odds for one event at a specific timestamp.
        """
        if not self.api_key:
            return None

        cache_key = f"historical_event_odds_{event_id}_{snapshot_time}_{markets}_{regions}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{ODDS_API_BASE}/historical/sports/{NCAAB_SPORT}/events/{event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "date": snapshot_time,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
        }

        try:
            self._throttle()
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data", {}) if isinstance(payload, dict) else {}
            if not data:
                return None

            parsed = self._parse_game_odds(data)
            parsed["event_id"] = event_id
            parsed["snapshot_time"] = snapshot_time
            self._cache[cache_key] = parsed
            return parsed
        except Exception as e:
            logger.error(f"Historical event odds lookup failed: {e}")
            return self._cache.get(cache_key)

    def get_historical_game_odds(
        self,
        home_team: str,
        away_team: str,
        commence_time: Optional[str] = None,
        snapshot_time: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Get historical odds for a specific game by team names.
        """
        for candidate_snapshot in self._candidate_historical_snapshots(
            commence_time=commence_time,
            snapshot_time=snapshot_time,
        ):
            events = self.get_historical_events(candidate_snapshot)
            if not events:
                continue

            for event in events:
                if self._matchup_matches(
                    home_team,
                    away_team,
                    event.get("home_team", ""),
                    event.get("away_team", ""),
                ):
                    odds = self.get_historical_event_odds(
                        event_id=event.get("id", ""),
                        snapshot_time=candidate_snapshot,
                    )
                    if odds:
                        return odds

        return None

    def get_historical_consensus_lines(
        self,
        home_team: str,
        away_team: str,
        commence_time: Optional[str] = None,
        snapshot_time: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Compute consensus historical spread and total across archived books.
        """
        odds = self.get_historical_game_odds(
            home_team=home_team,
            away_team=away_team,
            commence_time=commence_time,
            snapshot_time=snapshot_time,
        )
        return self._build_consensus_from_odds(odds)

    def get_historical_odds_for_day(
        self,
        cache_label: str,
        snapshot_time: str,
        commence_time_from: Optional[str] = None,
        commence_time_to: Optional[str] = None,
        markets: str = "spreads,totals,h2h",
        regions: str = "us",
        bookmakers: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Fetch the historical sport-level feed once for a scoreboard day and
        cache it on disk for reuse.
        """
        if not self.api_key:
            return []

        url = f"{ODDS_API_BASE}/historical/sports/{NCAAB_SPORT}/odds"
        return self._fetch_sport_feed(
            url=url,
            params=self._build_feed_params(
                markets=markets,
                regions=regions,
                bookmakers=bookmakers,
                commence_time_from=commence_time_from,
                commence_time_to=commence_time_to,
                snapshot_time=snapshot_time,
            ),
            session_cache_key=self._build_session_cache_key(
                prefix="historical_odds",
                markets=markets,
                regions=regions,
                bookmakers=bookmakers,
                commence_time_from=commence_time_from,
                commence_time_to=commence_time_to,
                snapshot_time=snapshot_time,
            ),
            cache_category="historical",
            cache_label=cache_label,
            cache_ttl=None,
            payload_data_key="data",
            snapshot_time=snapshot_time,
        )

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
        if game.get("id"):
            result["event_id"] = game.get("id")
        if game.get("snapshot_time"):
            result["snapshot_time"] = game.get("snapshot_time")

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

    def _build_consensus_from_odds(self, odds: Optional[dict]) -> Optional[dict]:
        if not odds:
            return None

        spreads = [
            s["home_spread"] for s in odds.get("spreads", [])
            if s.get("home_spread") is not None
        ]
        totals = [
            t["total"] for t in odds.get("totals", [])
            if t.get("total") is not None
        ]

        result = {
            "home_team": odds.get("home_team"),
            "away_team": odds.get("away_team"),
            "consensus_spread": None,
            "consensus_total": None,
            "spread_bookmaker_count": len(spreads),
            "total_bookmaker_count": len(totals),
        }

        if "event_id" in odds:
            result["event_id"] = odds.get("event_id")
        if "snapshot_time" in odds:
            result["snapshot_time"] = odds.get("snapshot_time")

        if spreads:
            result["consensus_spread"] = round(sum(spreads) / len(spreads), 1)
        if totals:
            result["consensus_total"] = round(sum(totals) / len(totals), 1)

        return result

    def build_snapshot_frames(
        self,
        all_odds: list[dict],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize a day-level odds feed into:
          1. one row per matchup/bookmaker
          2. one summary row per matchup
        """
        detail_rows: list[dict] = []
        summary_rows: list[dict] = []

        for game in all_odds:
            if not isinstance(game, dict):
                continue

            parsed = self._parse_game_odds(game)
            if not parsed.get("home_team") or not parsed.get("away_team"):
                continue

            matchup = f"{parsed.get('away_team')} @ {parsed.get('home_team')}"
            snapshot_time = parsed.get("snapshot_time")
            commence_time = parsed.get("commence_time")

            bookmaker_rows: dict[str, dict] = {}
            for spread in parsed.get("spreads", []):
                book = spread.get("bookmaker", "unknown")
                row = bookmaker_rows.setdefault(
                    book,
                    self._base_snapshot_row(parsed, matchup, snapshot_time, commence_time, book),
                )
                row.update({
                    "Home Spread": spread.get("home_spread"),
                    "Away Spread": spread.get("away_spread"),
                    "Home Spread Price": spread.get("home_price"),
                    "Away Spread Price": spread.get("away_price"),
                })

            for total in parsed.get("totals", []):
                book = total.get("bookmaker", "unknown")
                row = bookmaker_rows.setdefault(
                    book,
                    self._base_snapshot_row(parsed, matchup, snapshot_time, commence_time, book),
                )
                row.update({
                    "Total": total.get("total"),
                    "Over Price": total.get("over_price"),
                    "Under Price": total.get("under_price"),
                })

            for moneyline in parsed.get("moneylines", []):
                book = moneyline.get("bookmaker", "unknown")
                row = bookmaker_rows.setdefault(
                    book,
                    self._base_snapshot_row(parsed, matchup, snapshot_time, commence_time, book),
                )
                row.update({
                    "Home ML": moneyline.get("home_ml"),
                    "Away ML": moneyline.get("away_ml"),
                })

            detail_rows.extend(bookmaker_rows.values())

            consensus = self._build_consensus_from_odds(parsed) or {}
            spread_values = [
                float(s["home_spread"]) for s in parsed.get("spreads", [])
                if s.get("home_spread") is not None
            ]
            total_values = [
                float(t["total"]) for t in parsed.get("totals", [])
                if t.get("total") is not None
            ]
            summary_rows.append({
                "Snapshot Time": snapshot_time,
                "Commence Time": commence_time,
                "Event ID": parsed.get("event_id", ""),
                "Matchup": matchup,
                "Away Team": parsed.get("away_team"),
                "Home Team": parsed.get("home_team"),
                "Spread Books": consensus.get("spread_bookmaker_count", 0),
                "Total Books": consensus.get("total_bookmaker_count", 0),
                "Moneyline Books": len(parsed.get("moneylines", [])),
                "Consensus Home Spread": consensus.get("consensus_spread"),
                "Consensus Total": consensus.get("consensus_total"),
                "Min Home Spread": min(spread_values) if spread_values else None,
                "Max Home Spread": max(spread_values) if spread_values else None,
                "Min Total": min(total_values) if total_values else None,
                "Max Total": max(total_values) if total_values else None,
            })

        details_df = pd.DataFrame(detail_rows)
        summary_df = pd.DataFrame(summary_rows)

        if not details_df.empty:
            details_df = details_df.sort_values(
                by=["Commence Time", "Matchup", "Bookmaker"],
                na_position="last",
            ).reset_index(drop=True)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(
                by=["Commence Time", "Matchup"],
                na_position="last",
            ).reset_index(drop=True)

        return details_df, summary_df

    @staticmethod
    def _base_snapshot_row(
        parsed: dict,
        matchup: str,
        snapshot_time: Optional[str],
        commence_time: Optional[str],
        bookmaker: str,
    ) -> dict:
        return {
            "Snapshot Time": snapshot_time,
            "Commence Time": commence_time,
            "Event ID": parsed.get("event_id", ""),
            "Matchup": matchup,
            "Away Team": parsed.get("away_team"),
            "Home Team": parsed.get("home_team"),
            "Bookmaker": bookmaker,
            "Home Spread": None,
            "Away Spread": None,
            "Home Spread Price": None,
            "Away Spread Price": None,
            "Total": None,
            "Over Price": None,
            "Under Price": None,
            "Home ML": None,
            "Away ML": None,
        }

    def _candidate_historical_snapshots(
        self,
        commence_time: Optional[str] = None,
        snapshot_time: Optional[str] = None,
    ) -> list[str]:
        if snapshot_time:
            return [snapshot_time]

        if not commence_time:
            return []

        parsed = self._parse_iso_datetime(commence_time)
        if not parsed:
            return []

        snap = parsed - timedelta(minutes=5)
        return [snap.strftime("%Y-%m-%dT%H:%M:%SZ")]

    def _matchup_matches(
        self,
        expected_home: str,
        expected_away: str,
        candidate_home: str,
        candidate_away: str,
    ) -> bool:
        home_key = self._team_key(expected_home)
        away_key = self._team_key(expected_away)
        candidate_home_key = self._team_key(candidate_home)
        candidate_away_key = self._team_key(candidate_away)

        return (
            self._team_keys_match(home_key, candidate_home_key)
            and self._team_keys_match(away_key, candidate_away_key)
        )

    def _team_key(self, team_name: str) -> str:
        if not team_name:
            return ""

        raw_name = str(team_name).strip()
        key = raw_name.lower()

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
        cleaned = "".join(ch if ch.isalnum() else " " for ch in resolved.lower())
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
    def _parse_iso_datetime(value: str) -> Optional[datetime]:
        if not value:
            return None
        text = str(value).strip()
        for fmt in (
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%MZ",
        ):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None

    @staticmethod
    def _normalize_cache_label(value: Optional[str]) -> str:
        if value is None:
            return "all"
        cleaned = "".join(
            ch.lower() if ch.isalnum() else "_"
            for ch in str(value).strip()
        )
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")
        return cleaned.strip("_") or "all"

    def _cache_file_path(
        self,
        category: str,
        cache_label: Optional[str],
        markets: str,
        regions: str,
        bookmakers: Optional[list[str]] = None,
        commence_time_from: Optional[str] = None,
        commence_time_to: Optional[str] = None,
        snapshot_time: Optional[str] = None,
    ) -> str:
        file_bits = [
            category,
            cache_label,
            markets,
            regions,
            ",".join(bookmakers or []) or "all_books",
            commence_time_from,
            commence_time_to,
            snapshot_time,
        ]
        slug = "__".join(
            self._normalize_cache_label(bit)
            for bit in file_bits
            if bit is not None
        )
        return os.path.join(self._file_cache_dir, f"{slug}.json")

    def _load_file_cache(
        self,
        path: str,
        max_age: Optional[timedelta] = None,
        allow_stale: bool = False,
    ) -> Optional[list[dict]]:
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            logger.warning(f"Could not read odds cache {path}: {exc}")
            return None

        saved_at_raw = payload.get("saved_at")
        games = payload.get("games")
        if not isinstance(games, list):
            return None

        if max_age is None or allow_stale:
            return games

        saved_at = self._parse_iso_datetime(saved_at_raw) if saved_at_raw else None
        if saved_at and datetime.utcnow() - saved_at <= max_age:
            return games
        return None

    def _save_file_cache(self, path: str, games: list[dict]) -> None:
        payload = {
            "saved_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "games": games,
        }
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
        except OSError as exc:
            logger.warning(f"Could not write odds cache {path}: {exc}")

    @staticmethod
    def _annotate_feed_games(
        games: list[dict],
        snapshot_time: Optional[str] = None,
    ) -> list[dict]:
        if not snapshot_time:
            return games

        annotated = []
        for game in games:
            if isinstance(game, dict):
                enriched = dict(game)
                enriched["snapshot_time"] = snapshot_time
                annotated.append(enriched)
            else:
                annotated.append(game)
        return annotated

    def _build_feed_params(
        self,
        markets: str,
        regions: str,
        bookmakers: Optional[list[str]] = None,
        commence_time_from: Optional[str] = None,
        commence_time_to: Optional[str] = None,
        snapshot_time: Optional[str] = None,
    ) -> dict:
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from
        if commence_time_to:
            params["commenceTimeTo"] = commence_time_to
        if snapshot_time:
            params["date"] = snapshot_time
        return params

    @staticmethod
    def _build_session_cache_key(
        prefix: str,
        markets: str,
        regions: str,
        bookmakers: Optional[list[str]] = None,
        commence_time_from: Optional[str] = None,
        commence_time_to: Optional[str] = None,
        snapshot_time: Optional[str] = None,
    ) -> str:
        parts = [
            prefix,
            markets,
            regions,
            ",".join(bookmakers or []) or "all_books",
            commence_time_from or "",
            commence_time_to or "",
            snapshot_time or "",
        ]
        return "__".join(parts)

    def _fetch_sport_feed(
        self,
        url: str,
        params: dict,
        session_cache_key: str,
        cache_category: str,
        cache_label: Optional[str],
        cache_ttl: Optional[timedelta],
        payload_data_key: Optional[str] = None,
        snapshot_time: Optional[str] = None,
    ) -> list[dict]:
        if session_cache_key in self._cache:
            return self._cache[session_cache_key]

        cache_path = self._cache_file_path(
            category=cache_category,
            cache_label=cache_label,
            markets=params.get("markets", ""),
            regions=params.get("regions", ""),
            bookmakers=params.get("bookmakers", "").split(",") if params.get("bookmakers") else None,
            commence_time_from=params.get("commenceTimeFrom"),
            commence_time_to=params.get("commenceTimeTo"),
            snapshot_time=params.get("date"),
        )

        cached_games = self._load_file_cache(cache_path, max_age=cache_ttl)
        if cached_games is not None:
            self._cache[session_cache_key] = cached_games
            return cached_games

        try:
            self._throttle()
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()

            payload = resp.json()
            if payload_data_key:
                games = payload.get(payload_data_key, []) if isinstance(payload, dict) else []
            else:
                games = payload if isinstance(payload, list) else []
            if not isinstance(games, list):
                games = []
            games = self._annotate_feed_games(games, snapshot_time=snapshot_time)

            remaining = resp.headers.get("x-requests-remaining", "?")
            used = resp.headers.get("x-requests-used", "?")
            logger.info(
                f"Odds API: {len(games)} games fetched "
                f"(API usage: {used} used, {remaining} remaining)"
            )

            self._cache[session_cache_key] = games
            self._save_file_cache(cache_path, games)
            return games

        except Exception as exc:
            logger.error(f"Odds API error: {exc}")
            stale_games = self._load_file_cache(
                cache_path,
                max_age=cache_ttl,
                allow_stale=True,
            )
            if stale_games is not None:
                self._cache[session_cache_key] = stale_games
                return stale_games
            return self._cache.get(session_cache_key, [])

    def _is_cache_fresh(self) -> bool:
        """Check if cached odds data is still fresh."""
        if not self._cache_time:
            return False
        return datetime.now() - self._cache_time < self._cache_ttl

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)
        self._last_request = time.time()

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
