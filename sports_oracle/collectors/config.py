"""
sports_oracle/collectors/config.py

Shared configuration, constants, rate limiting,
and base HTTP utilities used by all collectors.
"""

import os
import time
import logging
import requests
from datetime import datetime
from functools import wraps
from typing import Optional, Union

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sports_oracle")


# ── API Keys (load from environment or .env file) ─────────────────────────────
def load_env(path: str = ".env"):
    """Simple .env loader — avoids python-dotenv dependency."""
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

load_env()

CBBD_API_KEY        = os.environ.get("CBBD_API_KEY", "")
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
ODDS_API_KEY        = os.environ.get("ODDS_API_KEY", "")


# ── API Base URLs ──────────────────────────────────────────────────────────────
CBBD_BASE           = "https://api.collegebasketballdata.com"
ESPN_BASE           = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ESPN_CDN_BASE       = "https://cdn.espn.com/core/mens-college-basketball"
SPORTS_REF_BASE     = "https://www.sports-reference.com/cbb"
BARTTORVIK_BASE     = "https://barttorvik.com"
NCAA_API_BASE       = "https://ncaa-api.henrygd.me"


# ── Common HTTP headers ────────────────────────────────────────────────────────
DEFAULT_HEADERS = {
    "User-Agent": (
        "SportsOracle/1.0 (NCAA Tournament Prediction Research; "
        "educational project)"
    ),
    "Accept": "application/json",
}


# ── Rate limiting decorator ────────────────────────────────────────────────────
def rate_limited(min_delay: float = 1.0):
    """
    Decorator that enforces a minimum delay between calls
    to the same function. Polite scraping / API usage.
    """
    last_called = {}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = func.__name__
            elapsed = time.time() - last_called.get(key, 0)
            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)
            result = func(*args, **kwargs)
            last_called[key] = time.time()
            return result
        return wrapper
    return decorator


# ── Base HTTP session ──────────────────────────────────────────────────────────
class BaseClient:
    """
    Shared HTTP client with retry logic, error handling,
    and request logging. All collectors inherit from this.
    """

    def __init__(self, name: str, base_url: str, delay: float = 1.0):
        self.name     = name
        self.base_url = base_url.rstrip("/")
        self.delay    = delay
        self.session  = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self._last_request = 0.0
        self.log = logging.getLogger(f"sports_oracle.{name}")

    def _throttle(self):
        """Enforce delay between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()

    def get(
        self,
        endpoint: str,
        params: dict = None,
        retries: int = 3,
        timeout: int = 15,
    ) -> Optional[Union[dict, list]]:
        """
        GET request with retry logic.
        Returns parsed JSON or None on failure.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        self._throttle()

        for attempt in range(1, retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
                resp.raise_for_status()
                self.log.debug(f"GET {url} → {resp.status_code}")
                return resp.json()

            except (ValueError, requests.exceptions.JSONDecodeError) as e:
                # Non-JSON response (HTML error page, empty body, etc.)
                self.log.warning(
                    f"Invalid JSON from {url} "
                    f"(attempt {attempt}/{retries}): {e}"
                )
                if attempt < retries:
                    time.sleep(2 ** attempt)

            except requests.exceptions.HTTPError as e:
                self.log.warning(
                    f"HTTP {resp.status_code} on {url} "
                    f"(attempt {attempt}/{retries}): {e}"
                )
                if resp.status_code in (400, 401, 403, 404):
                    break   # don't retry client errors
                if attempt < retries:
                    time.sleep(2 ** attempt)   # exponential backoff

            except requests.exceptions.ConnectionError as e:
                self.log.warning(f"Connection error {url} (attempt {attempt}): {e}")
                if attempt < retries:
                    time.sleep(2 ** attempt)

            except requests.exceptions.Timeout:
                self.log.warning(f"Timeout on {url} (attempt {attempt})")
                if attempt < retries:
                    time.sleep(2)

            except Exception as e:
                self.log.error(f"Unexpected error on {url}: {e}")
                break

        return None

    def get_csv(self, url: str, retries: int = 3) -> Optional[str]:
        """Fetch raw CSV text from a full URL."""
        self._throttle()
        for attempt in range(1, retries + 1):
            try:
                resp = self.session.get(url, timeout=20)
                resp.raise_for_status()
                return resp.text
            except Exception as e:
                self.log.warning(f"CSV fetch error {url} (attempt {attempt}): {e}")
                if attempt < retries:
                    time.sleep(2 ** attempt)
        return None


# ── Season utilities ───────────────────────────────────────────────────────────
def current_season() -> int:
    """
    Returns the current NCAA basketball season year.
    NCAA season 2025 = 2024-25 academic year.
    Season flips in November.
    """
    now = datetime.now()
    return now.year + 1 if now.month >= 11 else now.year


def season_range(start: int, end: Optional[int] = None) -> list[int]:
    """Returns list of season years from start to end (inclusive)."""
    end = end or current_season()
    return list(range(start, end + 1))


# ── Tournament round constants ─────────────────────────────────────────────────
TOURNAMENT_ROUNDS = {
    "First Four":       0,
    "First Round":      1,
    "Second Round":     2,
    "Sweet Sixteen":    3,
    "Elite Eight":      4,
    "Final Four":       5,
    "Championship":     6,
}

ROUND_MODIFIERS = {
    1: 1.00,   # First/Second round — baseline
    2: 1.00,
    3: 0.97,   # Sweet 16 — defenses tighten
    4: 0.94,   # Elite 8 — lowest scoring round
    5: 0.96,   # Final Four — elite offenses survive
    6: 0.95,   # Championship
}

# Exclude bubble year — no fans, single site
BUBBLE_SEASONS = {2021}

# Shot clock change — weight pre-change data less in pace calculations
SHOT_CLOCK_CHANGE_SEASON = 2016   # 35s → 30s rule change
