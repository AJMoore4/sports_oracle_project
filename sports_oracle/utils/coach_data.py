"""
sports_oracle/utils/coach_data.py

Hardcoded NCAA tournament coaching records for active coaches.
Falls back to redistributed weights if coach not found.

LAST UPDATED: March 2026 (includes 2025 tournament results)
"""

from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger("sports_oracle.coach_data")

COACH_RECORDS: dict[str, dict] = {
    "Duke": {"coach": "Jon Scheyer", "appearances": 3, "wins": 10, "losses": 2, "final_fours": 1, "titles": 1, "first_yr_coach": False},
    "North Carolina": {"coach": "Hubert Davis", "appearances": 4, "wins": 8, "losses": 4, "final_fours": 1, "titles": 0, "first_yr_coach": False},
    "Virginia": {"coach": "Tony Bennett", "appearances": 8, "wins": 16, "losses": 7, "final_fours": 1, "titles": 1, "first_yr_coach": False},
    "Clemson": {"coach": "Brad Brownell", "appearances": 4, "wins": 5, "losses": 4, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Miami (FL)": {"coach": "Jim Larranaga", "appearances": 7, "wins": 12, "losses": 7, "final_fours": 1, "titles": 0, "first_yr_coach": False},
    "Louisville": {"coach": "Pat Kelsey", "appearances": 1, "wins": 0, "losses": 1, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "NC State": {"coach": "Kevin Keatts", "appearances": 3, "wins": 5, "losses": 3, "final_fours": 1, "titles": 0, "first_yr_coach": False},
    "Florida St.": {"coach": "Leonard Hamilton", "appearances": 5, "wins": 7, "losses": 5, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "SMU": {"coach": "Andy Enfield", "appearances": 4, "wins": 5, "losses": 4, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Michigan": {"coach": "Dusty May", "appearances": 2, "wins": 3, "losses": 2, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Michigan St.": {"coach": "Tom Izzo", "appearances": 26, "wins": 59, "losses": 24, "final_fours": 8, "titles": 1, "first_yr_coach": False},
    "Purdue": {"coach": "Matt Painter", "appearances": 10, "wins": 15, "losses": 10, "final_fours": 1, "titles": 0, "first_yr_coach": False},
    "Illinois": {"coach": "Brad Underwood", "appearances": 4, "wins": 4, "losses": 4, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Wisconsin": {"coach": "Greg Gard", "appearances": 5, "wins": 5, "losses": 5, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Nebraska": {"coach": "Fred Hoiberg", "appearances": 1, "wins": 0, "losses": 1, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Iowa": {"coach": "Fran McCaffery", "appearances": 5, "wins": 3, "losses": 5, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "UCLA": {"coach": "Mick Cronin", "appearances": 9, "wins": 12, "losses": 9, "final_fours": 1, "titles": 0, "first_yr_coach": False},
    "Oregon": {"coach": "Dana Altman", "appearances": 8, "wins": 11, "losses": 8, "final_fours": 1, "titles": 0, "first_yr_coach": False},
    "USC": {"coach": "Eric Musselman", "appearances": 5, "wins": 10, "losses": 5, "final_fours": 1, "titles": 0, "first_yr_coach": False},
    "Ohio St.": {"coach": "Jake Diebler", "appearances": 1, "wins": 0, "losses": 1, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Northwestern": {"coach": "Chris Collins", "appearances": 2, "wins": 1, "losses": 2, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Maryland": {"coach": "Kevin Willard", "appearances": 3, "wins": 2, "losses": 3, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Houston": {"coach": "Kelvin Sampson", "appearances": 20, "wins": 38, "losses": 18, "final_fours": 3, "titles": 1, "first_yr_coach": False},
    "Kansas": {"coach": "Bill Self", "appearances": 20, "wins": 56, "losses": 18, "final_fours": 4, "titles": 2, "first_yr_coach": False},
    "Iowa St.": {"coach": "T.J. Otzelberger", "appearances": 4, "wins": 5, "losses": 4, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Arizona": {"coach": "Tommy Lloyd", "appearances": 3, "wins": 4, "losses": 3, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Baylor": {"coach": "Scott Drew", "appearances": 11, "wins": 19, "losses": 10, "final_fours": 1, "titles": 1, "first_yr_coach": False},
    "Texas Tech": {"coach": "Grant McCasland", "appearances": 2, "wins": 3, "losses": 2, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "BYU": {"coach": "Kevin Young", "appearances": 0, "wins": 0, "losses": 0, "final_fours": 0, "titles": 0, "first_yr_coach": True},
    "TCU": {"coach": "Jamie Dixon", "appearances": 8, "wins": 6, "losses": 8, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "UCF": {"coach": "Johnny Dawkins", "appearances": 2, "wins": 2, "losses": 2, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Florida": {"coach": "Todd Golden", "appearances": 2, "wins": 6, "losses": 1, "final_fours": 1, "titles": 0, "first_yr_coach": False},
    "Alabama": {"coach": "Nate Oats", "appearances": 4, "wins": 8, "losses": 4, "final_fours": 1, "titles": 0, "first_yr_coach": False},
    "Auburn": {"coach": "Bruce Pearl", "appearances": 7, "wins": 12, "losses": 6, "final_fours": 2, "titles": 0, "first_yr_coach": False},
    "Tennessee": {"coach": "Rick Barnes", "appearances": 15, "wins": 20, "losses": 15, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Kentucky": {"coach": "Mark Pope", "appearances": 1, "wins": 1, "losses": 1, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Texas A&M": {"coach": "Buzz Williams", "appearances": 5, "wins": 7, "losses": 5, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Arkansas": {"coach": "John Calipari", "appearances": 23, "wins": 59, "losses": 22, "final_fours": 4, "titles": 1, "first_yr_coach": False},
    "Missouri": {"coach": "Dennis Gates", "appearances": 1, "wins": 0, "losses": 1, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Ole Miss": {"coach": "Chris Beard", "appearances": 5, "wins": 5, "losses": 5, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Georgia": {"coach": "Mike White", "appearances": 5, "wins": 4, "losses": 5, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Texas": {"coach": "Sean Miller", "appearances": 13, "wins": 21, "losses": 12, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Oklahoma": {"coach": "Porter Moser", "appearances": 3, "wins": 6, "losses": 3, "final_fours": 1, "titles": 0, "first_yr_coach": False},
    "Connecticut": {"coach": "Dan Hurley", "appearances": 5, "wins": 17, "losses": 3, "final_fours": 2, "titles": 2, "first_yr_coach": False},
    "Marquette": {"coach": "Shaka Smart", "appearances": 7, "wins": 9, "losses": 7, "final_fours": 1, "titles": 0, "first_yr_coach": False},
    "Creighton": {"coach": "Greg McDermott", "appearances": 6, "wins": 7, "losses": 6, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "St. John's": {"coach": "Rick Pitino", "appearances": 18, "wins": 54, "losses": 14, "final_fours": 7, "titles": 2, "first_yr_coach": False},
    "Gonzaga": {"coach": "Mark Few", "appearances": 25, "wins": 47, "losses": 22, "final_fours": 2, "titles": 0, "first_yr_coach": False},
    "Saint Mary's": {"coach": "Randy Bennett", "appearances": 7, "wins": 5, "losses": 7, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Dayton": {"coach": "Anthony Grant", "appearances": 3, "wins": 3, "losses": 3, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "San Diego St.": {"coach": "Brian Dutcher", "appearances": 4, "wins": 7, "losses": 3, "final_fours": 1, "titles": 0, "first_yr_coach": False},
    "Memphis": {"coach": "Penny Hardaway", "appearances": 2, "wins": 1, "losses": 2, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "New Mexico": {"coach": "Richard Pitino", "appearances": 2, "wins": 1, "losses": 2, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "McNeese": {"coach": "Will Wade", "appearances": 3, "wins": 0, "losses": 3, "final_fours": 0, "titles": 0, "first_yr_coach": False},
    "Furman": {"coach": "Bob Richey", "appearances": 2, "wins": 1, "losses": 2, "final_fours": 0, "titles": 0, "first_yr_coach": False},
}

_COACH_NAME_INDEX: dict[str, dict] = {}
for _school, _rec in COACH_RECORDS.items():
    _COACH_NAME_INDEX[_rec["coach"].lower()] = {**_rec, "school": _school}


def get_coach_record(team_or_name: str, by_name: bool = False) -> Optional[dict]:
    if by_name:
        rec = _COACH_NAME_INDEX.get(team_or_name.lower().strip())
    else:
        rec = COACH_RECORDS.get(team_or_name)
        if not rec:
            for school, r in COACH_RECORDS.items():
                if school.lower() == team_or_name.lower():
                    rec = r
                    break
    if not rec:
        return None
    total_games = rec.get("wins", 0) + rec.get("losses", 0)
    win_rate = rec["wins"] / total_games if total_games > 0 else 0.0
    return {
        "coach": rec["coach"],
        "appearances": rec.get("appearances", 0),
        "total_wins": rec.get("wins", 0),
        "total_losses": rec.get("losses", 0),
        "final_fours": rec.get("final_fours", 0),
        "championships": rec.get("titles", 0),
        "win_rate": round(win_rate, 3),
        "first_yr_coach": rec.get("first_yr_coach", False),
    }
