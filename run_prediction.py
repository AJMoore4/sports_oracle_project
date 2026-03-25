#!/usr/bin/env python3
"""
run_prediction.py

Pull all D1 men's basketball games for any date from ESPN,
predict every matchup, and display clean results.

Also supports interactive startup mode for matchup sessions.

The Model Spread column IS the pick:
  - Negative spread = model favors the home team (listed second)
  - Positive spread = model favors the away team (listed first)
  - Example: "HOU -8.5" means the model picks Houston by 8.5

USAGE:
  python run_prediction.py                    # prompts for matchup session vs full scoreboard
  python run_prediction.py 03/12/2026         # prompts with a preset date
  python run_prediction.py 03/10/2026 --end-date 03/14/2026
  python run_prediction.py 03/18/2026 --export-odds-snapshot
  python run_prediction.py 03/13/2026 --export-odds-snapshot --snapshot-time 2026-03-13T08:00:00Z
  python run_prediction.py --home Duke --away Vermont
  python run_prediction.py 03/13/2026 --home "Saint Louis" --away "George Washington"
  python run_prediction.py 03/11/2026 --debug # with full error tracebacks
"""

import os
import sys
import re
import math
import json
import argparse
import shutil
import traceback
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Union
from pathlib import Path

SEASON = 2026
DEBUG = "--debug" in sys.argv
SCOREBOARD_CACHE: dict[str, list[dict]] = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Sports Oracle for a matchup session, a full date slate, a one-off single game, or an odds snapshot export."
    )
    parser.add_argument(
        "date",
        nargs="?",
        help="Slate date in MM/DD/YYYY, YYYY-MM-DD, or YYYYMMDD format",
    )
    parser.add_argument(
        "--end-date",
        help="Inclusive end date for multi-day scoreboard mode",
    )
    parser.add_argument("--home", help="Home team for single-game mode")
    parser.add_argument("--away", help="Away team for single-game mode")
    parser.add_argument("--venue", help="Venue name override for single-game mode")
    parser.add_argument(
        "--export-odds-snapshot",
        dest="odds_snapshot",
        action="store_true",
        help="Export a sportsbook-by-sportsbook odds snapshot for the date to XLSX.",
    )
    parser.add_argument(
        "--snapshot-time",
        help="Historical snapshot time in ISO UTC, e.g. 2026-03-18T08:00:00Z",
    )
    parser.add_argument(
        "--refresh-odds",
        action="store_true",
        help="Force a fresh day-level Odds API pull before running the model.",
    )
    parser.add_argument("--home-seed", type=int, help="Home seed for single-game mode")
    parser.add_argument("--away-seed", type=int, help="Away seed for single-game mode")
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Tournament round for single-game mode (default: 1)",
    )
    ncaa_group = parser.add_mutually_exclusive_group()
    ncaa_group.add_argument(
        "--ncaa-tournament",
        dest="ncaa_tournament",
        action="store_true",
        help="Use NCAA Tournament-specific training and adjustments.",
    )
    ncaa_group.add_argument(
        "--not-ncaa-tournament",
        dest="ncaa_tournament",
        action="store_false",
        help="Disable NCAA Tournament-specific fitting and adjustments.",
    )
    parser.set_defaults(ncaa_tournament=None)
    parser.add_argument("--debug", action="store_true", help="Show full tracebacks")
    return parser.parse_args()


def parse_date_arg(date_value: Optional[str]) -> str:
    if date_value:
        cleaned = str(date_value).strip()
        for fmt in (
            "%m/%d/%Y",
            "%m-%d-%Y",
            "%m/%d/%y",
            "%m-%d-%y",
            "%Y%m%d",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(cleaned, fmt).strftime("%Y%m%d")
            except ValueError:
                continue
        raise SystemExit(
            f"Invalid date format: {date_value}. "
            "Use MM/DD/YYYY, MM/DD/YY, YYYY-MM-DD, or YYYYMMDD."
        )
    return datetime.now().strftime("%Y%m%d")


def display_date(d: str) -> str:
    return f"{d[4:6]}/{d[6:]}/{d[:4]}"


def iter_date_range(start_date: str, end_date: str) -> list[str]:
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def prompt_text(label: str, default: Optional[str] = None, required: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            value = input(f"  {label}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\nCancelled.")

        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""
        print("  Please enter a value.")


def prompt_yes_no(label: str, default: bool = False) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        try:
            value = input(f"  {label} [{default_text}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\nCancelled.")

        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("  Enter y or n.")


def prompt_run_mode(args):
    args.matchup_session = False

    if getattr(args, "odds_snapshot", False):
        if not args.date and sys.stdin.isatty():
            default_date = display_date(parse_date_arg(args.date))
            args.date = prompt_text("Date (MM/DD/YYYY)", default=default_date)
            raw_snapshot = prompt_text(
                "Historical snapshot time UTC (optional)",
                default=args.snapshot_time or "",
            ).strip()
            args.snapshot_time = raw_snapshot or None
        if args.ncaa_tournament is None:
            args.ncaa_tournament = False
        return args

    if args.home or args.away:
        if args.ncaa_tournament is None and sys.stdin.isatty():
            args.ncaa_tournament = prompt_yes_no(
                "Are these games for the NCAA Tournament?",
                default=False,
            )
        if not args.refresh_odds and sys.stdin.isatty():
            args.refresh_odds = prompt_yes_no(
                "Refresh sportsbook odds first?",
                default=False,
            )
        return args

    if not sys.stdin.isatty():
        if args.ncaa_tournament is None:
            args.ncaa_tournament = False
        return args

    default_date = display_date(parse_date_arg(args.date))

    print("  Choose lookup mode:")
    print("    1) Matchup session")
    print("    2) Full scoreboard")
    print("    3) Date range")
    print("    4) Odds snapshot export")

    while True:
        choice = prompt_text("Selection", default="2").strip().lower()
        if choice in {"1", "single", "game", "match", "matchup", "session"}:
            mode = "session"
            break
        if choice in {"2", "scoreboard", "slate", "full", "date"}:
            mode = "scoreboard"
            break
        if choice in {"3", "range", "dates", "multi", "multiday"}:
            mode = "range"
            break
        if choice in {"4", "odds", "snapshot", "books", "export"}:
            mode = "odds"
            break
        print("  Enter 1, 2, 3, or 4.")

    if mode == "odds":
        args.odds_snapshot = True
        args.date = prompt_text("Date (MM/DD/YYYY)", default=default_date)
        raw_snapshot = prompt_text(
            "Historical snapshot time UTC (optional)",
            default=args.snapshot_time or "",
        ).strip()
        args.snapshot_time = raw_snapshot or None
        if args.ncaa_tournament is None:
            args.ncaa_tournament = False
        return args

    if mode == "range":
        args.date = prompt_text("Start date (MM/DD/YYYY)", default=default_date)
        args.end_date = prompt_text("End date (MM/DD/YYYY)", default=args.date)
        if args.ncaa_tournament is None:
            args.ncaa_tournament = prompt_yes_no(
                "Are these games for the NCAA Tournament?",
                default=False,
            )
        if not args.refresh_odds:
            args.refresh_odds = prompt_yes_no(
                "Refresh sportsbook odds first?",
                default=False,
            )
        return args

    args.date = prompt_text("Date (MM/DD/YYYY)", default=default_date)

    if args.ncaa_tournament is None:
        args.ncaa_tournament = prompt_yes_no(
            "Are these games for the NCAA Tournament?",
            default=False,
        )

    if mode == "session":
        args.matchup_session = True

    if not args.refresh_odds:
        args.refresh_odds = prompt_yes_no(
            "Refresh sportsbook odds first?",
            default=False,
        )

    return args


def prompt_int(label: str, default: Optional[int] = None, minimum: Optional[int] = None) -> int:
    while True:
        raw_value = prompt_text(label, default=str(default) if default is not None else None, required=True)
        try:
            value = int(raw_value)
        except ValueError:
            print("  Please enter a whole number.")
            continue
        if minimum is not None and value < minimum:
            print(f"  Enter a value greater than or equal to {minimum}.")
            continue
        return value


def prompt_matchup_session_settings(
    date_str: str,
    tournament_round: int,
    venue: Optional[str],
    is_ncaa_tournament: bool,
) -> tuple[str, int, Optional[str]]:
    session_date = parse_date_arg(
        prompt_text("Session date (MM/DD/YYYY)", default=display_date(date_str))
    )
    session_round = tournament_round
    if is_ncaa_tournament:
        session_round = prompt_int(
            "Default tournament round",
            default=tournament_round,
            minimum=0,
        )
    session_venue = prompt_text(
        "Default venue override",
        default=venue or "",
    ).strip() or None
    return session_date, session_round, session_venue


def prompt_excel_filename(label: str) -> str:
    default_name = f"sports_oracle_{label}.xlsx"
    if not sys.stdin.isatty():
        return default_name

    filename = prompt_text("XLSX filename", default=default_name, required=True).strip()
    if not filename:
        filename = default_name
    if not filename.lower().endswith(".xlsx"):
        filename += ".xlsx"
    return filename


def export_frames_to_xlsx(sheets: dict[str, pd.DataFrame], label: str) -> str:
    filename = prompt_excel_filename(label)
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
    )
    os.makedirs(results_dir, exist_ok=True)
    safe_filename = os.path.basename(filename)
    path = os.path.join(results_dir, safe_filename)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            safe_sheet = str(sheet_name)[:31] or "Sheet1"
            export_df = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
            export_df.to_excel(writer, sheet_name=safe_sheet, index=False)
    return path


def _normalize_datetime_series(values) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(parsed, pd.Series):
        return parsed.dt.tz_localize(None)
    return pd.Series(dtype="datetime64[ns]")


def odds_feed_time_window(scoreboard: pd.DataFrame, date_str: str) -> tuple[Optional[str], Optional[str]]:
    if isinstance(scoreboard, pd.DataFrame) and not scoreboard.empty and "date" in scoreboard.columns:
        row_dates = _normalize_datetime_series(scoreboard["date"]).dropna()
        if not row_dates.empty:
            start = row_dates.min() - pd.Timedelta(hours=2)
            end = row_dates.max() + pd.Timedelta(hours=2)
            return (
                start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )

    base_date = pd.to_datetime(parse_date_arg(date_str), format="%Y%m%d", errors="coerce")
    if pd.isna(base_date):
        return None, None

    start = pd.Timestamp(base_date).normalize()
    end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return (
        start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def default_historical_snapshot_time(scoreboard: pd.DataFrame, date_str: str) -> str:
    if isinstance(scoreboard, pd.DataFrame) and not scoreboard.empty and "date" in scoreboard.columns:
        row_dates = _normalize_datetime_series(scoreboard["date"]).dropna()
        if not row_dates.empty:
            snapshot = row_dates.min() - pd.Timedelta(hours=12)
            return snapshot.strftime("%Y-%m-%dT%H:%M:%SZ")

    base_date = pd.to_datetime(parse_date_arg(date_str), format="%Y%m%d", errors="coerce")
    if pd.isna(base_date):
        base_date = pd.Timestamp.utcnow().normalize()
    snapshot = pd.Timestamp(base_date).normalize() + pd.Timedelta(hours=8)
    return snapshot.strftime("%Y-%m-%dT%H:%M:%SZ")


def filter_feed_to_scoreboard(collector, feed: list[dict], scoreboard: pd.DataFrame) -> list[dict]:
    if not feed or scoreboard.empty:
        return feed

    scoreboard_pairs = []
    for _, row in scoreboard.iterrows():
        home_team = str(row.get("home_team", "")).strip()
        away_team = str(row.get("away_team", "")).strip()
        if home_team and away_team:
            scoreboard_pairs.append((home_team, away_team))

    filtered = []
    for game in feed:
        game_home = str(game.get("home_team", "")).strip()
        game_away = str(game.get("away_team", "")).strip()
        if not game_home or not game_away:
            continue
        for home_team, away_team in scoreboard_pairs:
            if collector._matchup_matches(home_team, away_team, game_home, game_away):
                filtered.append(game)
                break
            if collector._matchup_matches(away_team, home_team, game_home, game_away):
                filtered.append(game)
                break
    return filtered


def should_reuse_scoreboard_cache(date_str: str, games: list[dict]) -> bool:
    if not games:
        return False
    if all(bool(game.get("is_final")) for game in games):
        return True

    try:
        requested_dt = datetime.strptime(parse_date_arg(date_str), "%Y%m%d").date()
    except ValueError:
        return False

    return requested_dt > datetime.now().date()


def export_odds_snapshot(date_str: str, snapshot_time: Optional[str] = None) -> None:
    from sports_oracle.collectors.espn_collector import ESPNCollector
    from sports_oracle.collectors.odds_collector import OddsCollector

    date_str = parse_date_arg(date_str)
    display = display_date(date_str)
    print(f"  📚  Exporting odds snapshot for {display}...")

    odds = OddsCollector(api_key=os.environ.get("ODDS_API_KEY", ""))
    if not odds.is_configured:
        print("  ⚠️  No Odds API key configured.\n")
        return

    espn = ESPNCollector()
    scoreboard = espn.get_scoreboard(date=date_str)
    if scoreboard.empty:
        print("  No games found for that date.\n")
        return

    commence_from, commence_to = odds_feed_time_window(scoreboard, date_str)
    requested_dt = datetime.strptime(date_str, "%Y%m%d").date()
    today_dt = datetime.now().date()
    export_mode = "current"
    effective_snapshot = snapshot_time

    if effective_snapshot:
        export_mode = "historical"
    elif requested_dt < today_dt:
        export_mode = "historical"
        effective_snapshot = default_historical_snapshot_time(scoreboard, date_str)
        print(f"  🕰️  Using default historical snapshot time: {effective_snapshot}")

    if export_mode == "historical":
        feed = odds.get_historical_odds_for_day(
            cache_label=f"{date_str}_{effective_snapshot}",
            snapshot_time=effective_snapshot,
            commence_time_from=commence_from,
            commence_time_to=commence_to,
        )
    else:
        feed = odds.get_current_odds_for_day(
            cache_label=date_str,
            commence_time_from=commence_from,
            commence_time_to=commence_to,
        )

    filtered_feed = filter_feed_to_scoreboard(odds, feed, scoreboard)
    details_df, summary_df = odds.build_snapshot_frames(filtered_feed)

    metadata_df = pd.DataFrame([
        {"Field": "Date", "Value": display},
        {"Field": "Date Key", "Value": date_str},
        {"Field": "Mode", "Value": export_mode},
        {"Field": "Snapshot Time", "Value": effective_snapshot or ""},
        {"Field": "Commence Time From", "Value": commence_from or ""},
        {"Field": "Commence Time To", "Value": commence_to or ""},
        {"Field": "Scoreboard Games", "Value": len(scoreboard)},
        {"Field": "Odds Feed Games", "Value": len(feed)},
        {"Field": "Matched Games", "Value": len(summary_df)},
    ])

    label_bits = ["odds_snapshot", date_str]
    if effective_snapshot:
        snapshot_slug = re.sub(r"[^0-9A-Za-z]+", "_", effective_snapshot).strip("_")
        if snapshot_slug:
            label_bits.append(snapshot_slug)
    else:
        label_bits.append("current")
    export_label = "_".join(label_bits)

    path = export_frames_to_xlsx(
        {
            "Summary": summary_df,
            "All Odds": details_df,
            "Metadata": metadata_df,
        },
        export_label,
    )
    print(f"  💾  Odds snapshot saved to {path}\n")


def _remove_file_if_exists(path: Union[Path, str]) -> bool:
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def _clear_standard_odds_day_cache(
    odds,
    date_str: str,
    commence_from: Optional[str],
    commence_to: Optional[str],
    snapshot_time: Optional[str] = None,
) -> None:
    markets = "spreads,totals,h2h"
    regions = "us"
    prefix = "historical_odds" if snapshot_time else "current_odds"
    category = "historical" if snapshot_time else "current"

    session_cache_key = odds._build_session_cache_key(
        prefix=prefix,
        markets=markets,
        regions=regions,
        bookmakers=None,
        commence_time_from=commence_from,
        commence_time_to=commence_to,
        snapshot_time=snapshot_time,
    )
    odds._cache.pop(session_cache_key, None)
    if not snapshot_time:
        odds._cache.pop("current_odds", None)
        odds._cache_time = None

    cache_path = odds._cache_file_path(
        category=category,
        cache_label=date_str,
        markets=markets,
        regions=regions,
        bookmakers=None,
        commence_time_from=commence_from,
        commence_time_to=commence_to,
        snapshot_time=snapshot_time,
    )
    _remove_file_if_exists(cache_path)


def _archive_current_odds_snapshot(
    odds,
    date_str: str,
    games: list[dict],
    commence_from: Optional[str],
    commence_to: Optional[str],
) -> None:
    if not games:
        return

    archive_tag = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    archive_path = odds._cache_file_path(
        category="current",
        cache_label=date_str,
        markets="spreads,totals,h2h",
        regions="us",
        bookmakers=None,
        commence_time_from=commence_from,
        commence_time_to=commence_to,
        snapshot_time=archive_tag,
    )
    odds._save_file_cache(archive_path, games)


def _clear_scoreboard_cache_for_date(pipeline, date_str: str, groups: str = "50") -> None:
    normalized_date = parse_date_arg(date_str)
    SCOREBOARD_CACHE.pop(normalized_date, None)

    lookup_date = pipeline._scoreboard_date_to_iso(normalized_date)
    scoreboard_cache_key = f"scoreboard_enriched_{lookup_date or 'today'}_{groups}"
    cache_path = pipeline._disk_cache_path("scoreboards", scoreboard_cache_key)
    _remove_file_if_exists(cache_path)


def refresh_odds_cache_for_date(pipeline, date_str: str) -> bool:
    normalized_date = parse_date_arg(date_str)
    display = display_date(normalized_date)
    odds = getattr(pipeline, "odds", None)
    if not odds or not getattr(odds, "is_configured", False):
        print(f"  ⚠️  Skipping odds refresh for {display}: no Odds API key configured.")
        return False

    scoreboard = pipeline.espn.get_scoreboard(date=normalized_date)
    if scoreboard.empty:
        _clear_scoreboard_cache_for_date(pipeline, normalized_date)
        print(f"  ⚠️  No games found for {display}. Skipping odds refresh.")
        return False

    commence_from, commence_to = odds_feed_time_window(scoreboard, normalized_date)
    requested_dt = datetime.strptime(normalized_date, "%Y%m%d").date()
    today_dt = datetime.now().date()
    is_historical = requested_dt < today_dt
    effective_snapshot = (
        default_historical_snapshot_time(scoreboard, normalized_date)
        if is_historical
        else None
    )

    print(
        f"  🔄  Refreshing sportsbook odds for {display}"
        f"{f' ({effective_snapshot})' if effective_snapshot else ''}..."
    )

    _clear_standard_odds_day_cache(
        odds,
        normalized_date,
        commence_from,
        commence_to,
        snapshot_time=effective_snapshot,
    )
    _clear_scoreboard_cache_for_date(pipeline, normalized_date)

    if is_historical:
        feed = odds.get_historical_odds_for_day(
            cache_label=normalized_date,
            snapshot_time=effective_snapshot,
            commence_time_from=commence_from,
            commence_time_to=commence_to,
        )
    else:
        feed = odds.get_current_odds_for_day(
            cache_label=normalized_date,
            commence_time_from=commence_from,
            commence_time_to=commence_to,
        )
        _archive_current_odds_snapshot(
            odds,
            normalized_date,
            feed,
            commence_from,
            commence_to,
        )

    matched_feed = filter_feed_to_scoreboard(odds, feed, scoreboard)
    print(
        f"  📘  Odds refresh complete: {len(matched_feed)} matched / "
        f"{len(scoreboard)} scoreboard games."
    )
    return True


def refresh_odds_cache_for_dates(pipeline, date_strings: list[str]) -> None:
    seen = set()
    for date_value in date_strings:
        normalized_date = parse_date_arg(date_value)
        if normalized_date in seen:
            continue
        seen.add(normalized_date)
        refresh_odds_cache_for_date(pipeline, normalized_date)


def export_table_to_xlsx(rows, label: str, has_finals: bool) -> str:
    show_date_column = any(row.get("date_display") for row in rows)

    export_rows = []
    for row in rows:
        export_row = {
            "Date": row.get("date_display", "") if show_date_column else None,
            "Matchup": row.get("matchup", ""),
            "Proj Score": row.get("proj_score", ""),
            "Proj Total": row.get("proj_total", ""),
            "Vegas O/U": row.get("vegas_ou", ""),
            "Model Spread": row.get("model_spread", ""),
            "Vegas Spread": row.get("vegas_spread", ""),
            "ATS Pick": row.get("ats_pick", ""),
            "ATS Conf": row.get("ats_confidence", ""),
            "Play": row.get("play_pass", ""),
            "Books": row.get("books_display", ""),
            "Spread Range": row.get("spread_range_display", ""),
            "Line Move": row.get("line_move_display", ""),
            "Prob": row.get("prob", ""),
        }
        if not show_date_column:
            export_row.pop("Date", None)

        if has_finals:
            export_row.update({
                "Actual Total": row.get("actual_total", ""),
                "Total Diff": row.get("tot_diff", ""),
                "Winner": row.get("winner", ""),
                "SU": row.get("su", ""),
                "ATS": row.get("ats", ""),
                "O/U": row.get("ou", ""),
                "Score": row.get("score", ""),
            })

        if row.get("error"):
            export_row["Error"] = row["error"]

        export_rows.append(export_row)

    summary_row = {
        "Games": len(rows),
        "Completed Games": sum(1 for row in rows if row.get("score")),
        "Errors": sum(1 for row in rows if row.get("error")),
        "Has Finals": bool(has_finals),
    }
    if has_finals:
        summary_row.update({
            "SU Record": f"{sum(1 for row in rows if row.get('su') == 'W')}-{sum(1 for row in rows if row.get('su') == 'L')}",
            "ATS Record": f"{sum(1 for row in rows if row.get('ats') == 'W')}-{sum(1 for row in rows if row.get('ats') == 'L')}-{sum(1 for row in rows if row.get('ats') == 'P')}",
            "O/U Record": f"{sum(1 for row in rows if row.get('ou_grade') == 'W')}-{sum(1 for row in rows if row.get('ou_grade') == 'L')}-{sum(1 for row in rows if row.get('ou_grade') == 'P')}",
            "ATS Play Record": f"{sum(1 for row in rows if row.get('play_pass') == 'PLAY' and row.get('ats') == 'W')}-{sum(1 for row in rows if row.get('play_pass') == 'PLAY' and row.get('ats') == 'L')}-{sum(1 for row in rows if row.get('play_pass') == 'PLAY' and row.get('ats') == 'P')}",
        })
    summary_rows = [summary_row]
    return export_frames_to_xlsx(
        {
            "Predictions": pd.DataFrame(export_rows),
            "Summary": pd.DataFrame(summary_rows),
        },
        label,
    )


def init(is_ncaa_tournament: bool = False):
    from sports_oracle.collectors.pipeline import DataPipeline
    from sports_oracle.engine.prediction_engine import PredictionEngine
    from sports_oracle.backtest.training_bootstrap import (
        build_runtime_ml_predictor,
    )

    print("  📡  Initializing pipeline (season 2026)...")
    pipeline = DataPipeline(
        cbbd_key=os.environ.get("CBBD_API_KEY", ""),
        odds_key=os.environ.get("ODDS_API_KEY", ""),
        season=SEASON,
    )
    engine = PredictionEngine()

    print("  🤖  Training ML model...")
    bootstrap = build_runtime_ml_predictor(
        season=SEASON,
        cbbd_key=os.environ.get("CBBD_API_KEY", ""),
        cache_dir="data/training_cache",
        use_live_tournament_fit=is_ncaa_tournament,
        status=print,
    )
    engine.set_calibration_profile(
        bootstrap.predictor.build_engine_calibration_profile()
    )
    return pipeline, engine, bootstrap.predictor


def fetch_games(pipeline, date_str):
    if date_str in SCOREBOARD_CACHE:
        cached_games = SCOREBOARD_CACHE[date_str]
        if should_reuse_scoreboard_cache(date_str, cached_games):
            print(f"  🗂️  Using cached games for {display_date(date_str)}...")
            games = [game.copy() for game in cached_games]
            return attach_cached_market_context(pipeline, date_str, games)
        print(f"  🔄  Refreshing cached games for {display_date(date_str)}...")
        SCOREBOARD_CACHE.pop(date_str, None)

    print(f"  📅  Fetching games for {display_date(date_str)}...")
    scoreboard = pipeline.get_scoreboard_with_historical_lines(date=date_str)
    if scoreboard.empty:
        return []

    games = []
    for _, row in scoreboard.iterrows():
        home = str(row.get("home_team", "")).strip()
        away = str(row.get("away_team", "")).strip()
        if not home or not away:
            continue

        status = str(row.get("status", "")).lower()
        is_final = "final" in status or "post" in status

        h_score, a_score = None, None
        if is_final:
            try:
                h_score = int(row.get("home_score"))
                a_score = int(row.get("away_score"))
            except (TypeError, ValueError):
                is_final = False

        h_abbr = str(row.get("home_abbr", ""))
        a_abbr = str(row.get("away_abbr", ""))
        if not h_abbr or h_abbr == "None":
            h_abbr = home[:3].upper()
        if not a_abbr or a_abbr == "None":
            a_abbr = away[:3].upper()

        raw_odds_detail = str(row.get("odds_detail", "")).strip()
        if raw_odds_detail.lower() in ["nan", "none", ""]:
            raw_odds_detail = ""

        betting_spread = _coerce_float(row.get("betting_spread"))
        if betting_spread is None:
            betting_spread = _parse_market_spread_from_odds_detail(
                raw_odds_detail,
                home,
                away,
                h_abbr,
                a_abbr,
            )

        odds_detail = (
            format_spread(betting_spread, h_abbr, a_abbr)
            if betting_spread is not None
            else (raw_odds_detail or "—")
        )

        games.append({
            "home_team": home, "away_team": away,
            "home_abbr": h_abbr, "away_abbr": a_abbr,
            "home_score": h_score, "away_score": a_score,
            "is_final": is_final,
            "venue": row.get("venue_name"),
            "betting_spread": betting_spread,
            "over_under": row.get("over_under"),
            "odds_detail": odds_detail,
            "line_source": row.get("line_source", "espn"),
            "line_provider_count": int(_coerce_float(row.get("line_provider_count")) or 0),
        })

    games = attach_cached_market_context(pipeline, date_str, games)
    SCOREBOARD_CACHE[date_str] = [game.copy() for game in games]
    return [game.copy() for game in games]


def format_spread(spread, h_abbr, a_abbr):
    """Format spread as 'TEAM -X.X' where the team shown is the favorite."""
    if spread is None or (isinstance(spread, float) and pd.isna(spread)):
        return "—"
    try:
        val = float(spread)
        if val == 0:
            return "PK"
        elif val < 0:
            return f"{h_abbr} {val:+.1f}"
        else:
            return f"{a_abbr} {-val:+.1f}"
    except (TypeError, ValueError):
        return "—"


def _coerce_float(value):
    try:
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _parse_market_spread_from_odds_detail(odds_detail, home_team, away_team, h_abbr, a_abbr):
    if not odds_detail:
        return None

    detail = str(odds_detail).strip()
    if not detail or detail.upper() == "EVEN":
        return None

    match = re.search(r"[-+]?\d*\.?\d+", detail)
    if not match:
        return None

    try:
        value = abs(float(match.group()))
    except (TypeError, ValueError):
        return None

    favorite_text = detail.split()[0].upper()
    home_name = str(home_team or "").upper()
    away_name = str(away_team or "").upper()
    home_tag = str(h_abbr or "").upper()
    away_tag = str(a_abbr or "").upper()

    if (
        favorite_text in home_name
        or favorite_text == home_tag
        or home_name.startswith(favorite_text[:3])
    ):
        return -value
    if (
        favorite_text in away_name
        or favorite_text == away_tag
        or away_name.startswith(favorite_text[:3])
    ):
        return value
    return None


def _required_ats_edge(model_spread, market_spread, margin_uncertainty=None, confidence_score=None):
    spread_zone = max(abs(float(model_spread)), abs(float(market_spread)))
    if spread_zone <= 2.5:
        required = 2.5
    elif spread_zone <= 5.0:
        required = 2.0
    elif spread_zone >= 18.0:
        required = 1.8
    elif spread_zone >= 12.0:
        required = 1.3
    else:
        required = 1.0

    uncertainty = _coerce_float(margin_uncertainty)
    if uncertainty is not None:
        if uncertainty >= 10.0:
            required += 0.6
        elif uncertainty >= 9.0:
            required += 0.3

    confidence = _coerce_float(confidence_score)
    if confidence is not None:
        if confidence < 0.45:
            required += 0.6
        elif confidence < 0.60:
            required += 0.3

    return required


def _market_context_key(home_team, away_team):
    return (
        str(home_team or "").strip().lower(),
        str(away_team or "").strip().lower(),
    )


def _load_cached_odds_feeds(pipeline, date_str):
    odds = getattr(pipeline, "odds", None)
    if not odds or not getattr(odds, "is_configured", False):
        return []

    cache_dir = Path(getattr(odds, "_file_cache_dir", ""))
    if not cache_dir.exists():
        return []

    feeds = []
    for path in sorted(cache_dir.glob(f"*__{date_str}__*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        games = payload.get("games", [])
        if not isinstance(games, list) or not games:
            continue

        snapshot_time = None
        for game in games:
            if isinstance(game, dict) and game.get("snapshot_time"):
                snapshot_time = game.get("snapshot_time")
                break
        effective_time = snapshot_time or payload.get("saved_at")
        parsed_time = pd.to_datetime(effective_time, errors="coerce", utc=True)
        if pd.isna(parsed_time):
            continue

        feeds.append({
            "path": str(path),
            "timestamp": parsed_time,
            "games": games,
        })

    return sorted(feeds, key=lambda item: item["timestamp"])


def _summarize_cached_odds_for_game(pipeline, feed_games, home_team, away_team):
    odds = pipeline.odds.get_game_odds_from_feed(
        all_odds=feed_games,
        home_team=home_team,
        away_team=away_team,
    )
    if not odds:
        return None

    summary = pipeline.odds._build_consensus_from_odds(odds) or {}
    spreads = [
        float(item["home_spread"])
        for item in odds.get("spreads", [])
        if item.get("home_spread") is not None
    ]
    if spreads:
        summary["min_home_spread"] = min(spreads)
        summary["max_home_spread"] = max(spreads)
    else:
        summary["min_home_spread"] = None
        summary["max_home_spread"] = None
    return summary


def _format_spread_range(min_spread, max_spread):
    min_val = _coerce_float(min_spread)
    max_val = _coerce_float(max_spread)
    if min_val is None and max_val is None:
        return "—"
    if min_val is None:
        return f"{max_val:+.1f}"
    if max_val is None:
        return f"{min_val:+.1f}"
    if math.isclose(min_val, max_val, abs_tol=0.05):
        return f"{min_val:+.1f}"
    return f"{min_val:+.1f}/{max_val:+.1f}"


def _format_line_move(move_value, h_abbr, a_abbr):
    move = _coerce_float(move_value)
    if move is None:
        return "—"
    if math.isclose(move, 0.0, abs_tol=0.05):
        return "Flat"
    team = h_abbr if move < 0 else a_abbr
    return f"{team} {abs(move):.1f}"


def attach_cached_market_context(pipeline, date_str, games):
    feeds = _load_cached_odds_feeds(pipeline, date_str)
    if not feeds:
        for game in games:
            provider_count = _coerce_float(game.get("line_provider_count"))
            if provider_count is None:
                game["books_display"] = "—"
            else:
                provider_count = int(provider_count)
                game["line_provider_count"] = provider_count
                game["books_display"] = str(provider_count) if provider_count > 0 else "—"
            game.setdefault("spread_range_display", "—")
            game.setdefault("spread_range_width", None)
            game.setdefault("line_move_display", "—")
            game.setdefault("line_move_value", None)
        return games

    context_map = {}
    unique_matchups = {
        _market_context_key(game.get("home_team"), game.get("away_team"))
        for game in games
    }

    for home_key, away_key in unique_matchups:
        if not home_key or not away_key:
            continue
        summaries = []
        for feed in feeds:
            summary = _summarize_cached_odds_for_game(
                pipeline,
                feed["games"],
                home_key,
                away_key,
            )
            if not summary:
                continue
            summaries.append({
                "timestamp": feed["timestamp"],
                **summary,
            })
        if not summaries:
            continue

        earliest = summaries[0]
        latest = summaries[-1]
        min_spread = latest.get("min_home_spread")
        max_spread = latest.get("max_home_spread")
        spread_range_width = None
        if _coerce_float(min_spread) is not None and _coerce_float(max_spread) is not None:
            spread_range_width = abs(float(max_spread) - float(min_spread))

        line_move_value = None
        early_spread = _coerce_float(earliest.get("consensus_spread"))
        late_spread = _coerce_float(latest.get("consensus_spread"))
        if early_spread is not None and late_spread is not None and len(summaries) >= 2:
            line_move_value = late_spread - early_spread

        context_map[(home_key, away_key)] = {
            "line_provider_count": int(
                latest.get("spread_bookmaker_count")
                or latest.get("total_bookmaker_count")
                or 0
            ),
            "spread_range_display": _format_spread_range(min_spread, max_spread),
            "spread_range_width": spread_range_width,
            "line_move_value": line_move_value,
        }

    for game in games:
        key = _market_context_key(game.get("home_team"), game.get("away_team"))
        context = context_map.get(key, {})

        provider_count = context.get("line_provider_count")
        if provider_count is None:
            provider_count = _coerce_float(game.get("line_provider_count"))
        if provider_count is None:
            game["books_display"] = "—"
        else:
            provider_count = int(provider_count)
            game["line_provider_count"] = provider_count
            game["books_display"] = str(provider_count) if provider_count > 0 else "—"

        game["spread_range_display"] = context.get("spread_range_display", "—")
        game["spread_range_width"] = context.get("spread_range_width")

        line_move_value = context.get("line_move_value")
        game["line_move_value"] = line_move_value
        game["line_move_display"] = _format_line_move(
            line_move_value,
            game.get("home_abbr", "H"),
            game.get("away_abbr", "A"),
        )

    return games


def model_ats_pick(
    model_spread,
    market_spread,
    home_team,
    away_team,
    h_abbr,
    a_abbr,
    margin_uncertainty=None,
    confidence_score=None,
):
    """Return the ATS side the model would choose against the market spread."""
    model_val = _coerce_float(model_spread)
    market_val = _coerce_float(market_spread)
    if model_val is None or market_val is None:
        return "", "—"

    if math.isclose(model_val, market_val, abs_tol=0.05):
        return "No edge", "—"
    if abs(model_val - market_val) < _required_ats_edge(
        model_val,
        market_val,
        margin_uncertainty=margin_uncertainty,
        confidence_score=confidence_score,
    ):
        return "No edge", "—"
    if model_val < market_val:
        return str(home_team), str(h_abbr)
    return str(away_team), str(a_abbr)


def ats_pick_profile(
    model_spread,
    market_spread,
    home_team,
    away_team,
    h_abbr,
    a_abbr,
    margin_uncertainty=None,
    confidence_score=None,
    provider_count=None,
    spread_range_width=None,
    line_move_value=None,
):
    pick_name, pick_display = model_ats_pick(
        model_spread,
        market_spread,
        home_team,
        away_team,
        h_abbr,
        a_abbr,
        margin_uncertainty=margin_uncertainty,
        confidence_score=confidence_score,
    )

    model_val = _coerce_float(model_spread)
    market_val = _coerce_float(market_spread)
    if model_val is None or market_val is None:
        return {
            "pick_name": "",
            "pick_display": "—",
            "confidence_label": "—",
            "confidence_score": 0.0,
            "play_pass": "PASS",
            "edge": None,
        }

    edge = abs(model_val - market_val)
    if pick_name in {"", "No edge"}:
        return {
            "pick_name": pick_name,
            "pick_display": "—",
            "confidence_label": "NO EDGE",
            "confidence_score": 0.0,
            "play_pass": "PASS",
            "edge": edge,
        }

    required_edge = _required_ats_edge(
        model_val,
        market_val,
        margin_uncertainty=margin_uncertainty,
        confidence_score=confidence_score,
    )
    ratio = edge / max(required_edge, 0.1)
    score = 0.50 + min(0.22, max(0.0, ratio - 1.0) * 0.18)

    base_conf = _coerce_float(confidence_score)
    if base_conf is not None:
        score += (base_conf - 0.50) * 0.20

    spread_zone = max(abs(model_val), abs(market_val))
    if spread_zone <= 5.0:
        score -= 0.10
    elif 8.0 <= spread_zone < 18.0:
        score += 0.08
    elif spread_zone >= 18.0:
        score -= 0.08

    if model_val * market_val < 0.0:
        score += 0.10

    provider_count_val = _coerce_float(provider_count)
    books = int(provider_count_val) if provider_count_val is not None else None
    if books is not None:
        if books <= 1:
            score -= 0.12
        elif books == 2:
            score -= 0.06
        elif books >= 5:
            score += 0.04

    range_width = _coerce_float(spread_range_width)
    if range_width is not None:
        if range_width > 2.0:
            score -= 0.12
        elif range_width > 1.0:
            score -= 0.06
        elif range_width <= 0.5:
            score += 0.04

    move = _coerce_float(line_move_value)
    if move is not None:
        pick_home = model_val < market_val
        if abs(move) >= 1.5:
            if (pick_home and move < 0) or ((not pick_home) and move > 0):
                score += 0.03
            else:
                score -= 0.05

    score = max(0.0, min(0.95, score))

    if score >= 0.72:
        confidence_label = "HIGH"
    elif score >= 0.60:
        confidence_label = "MED"
    elif score >= 0.52:
        confidence_label = "LOW"
    else:
        confidence_label = "FADE"

    play_pass = "PASS"
    book_gate_ok = books is None or books >= 2
    if book_gate_ok and confidence_label in {"HIGH", "MED"}:
        if 8.0 <= spread_zone < 18.0:
            play_pass = "PLAY"
        elif model_val * market_val < 0.0 and score >= 0.58:
            play_pass = "PLAY"

    return {
        "pick_name": pick_name,
        "pick_display": pick_display,
        "confidence_label": confidence_label,
        "confidence_score": score,
        "play_pass": play_pass,
        "edge": edge,
    }


def result_is_finite(result) -> bool:
    numeric_fields = [
        result.home_score,
        result.away_score,
        result.spread,
        result.total,
        result.home_win_prob,
        result.away_win_prob,
        result.confidence_score,
    ]
    try:
        return all(pd.notna(value) and math.isfinite(float(value)) for value in numeric_fields)
    except (TypeError, ValueError):
        return False


def safe_predict(
    pipeline,
    engine,
    ml,
    home,
    away,
    venue=None,
    game_date=None,
    espn_spread=None,
    espn_total=None,
    tournament_round=1,
    home_seed=None,
    away_seed=None,
    is_ncaa_tournament=False,
):
    try:
        inputs = pipeline.get_game_inputs(
            home_team=home, away_team=away,
            tournament_round=tournament_round,
            venue_name=venue,
            season=SEASON,
            game_date=game_date,
            home_seed=home_seed,
            away_seed=away_seed,
            espn_spread=espn_spread, espn_total=espn_total,
            is_ncaa_tournament=is_ncaa_tournament,
        )
        if not inputs.get("home_efficiency"):
            return None, "Missing data"

        result = engine.predict(inputs)
        if result is None:
            return None, "Engine error"
        if not result_is_finite(result):
            return None, "Invalid numeric prediction"
        try:
            result = ml.enhance_prediction(result, inputs)
        except Exception:
            pass
        if not result_is_finite(result):
            return None, "Invalid numeric prediction"
        return result, None
    except Exception as e:
        if DEBUG:
            traceback.print_exc()
        return None, str(e)[:50]


def _team_match_key(pipeline, team_name):
    if not team_name:
        return ""
    resolved = pipeline.resolver.resolve_or_original(str(team_name).strip())
    cleaned = "".join(ch if ch.isalnum() else " " for ch in resolved.lower())
    return " ".join(cleaned.split())


def find_single_game(pipeline, date_str, home_team, away_team):
    games = fetch_games(pipeline, date_str)
    home_key = _team_match_key(pipeline, home_team)
    away_key = _team_match_key(pipeline, away_team)

    for game in games:
        if (
            _team_match_key(pipeline, game.get("home_team")) == home_key
            and _team_match_key(pipeline, game.get("away_team")) == away_key
        ):
            return game

    for game in games:
        if (
            _team_match_key(pipeline, game.get("home_team")) == away_key
            and _team_match_key(pipeline, game.get("away_team")) == home_key
        ):
            return game

    return None


def get_single_game_display_context(pipeline, game, team_one, team_two):
    team_one_key = _team_match_key(pipeline, team_one)
    home_key = _team_match_key(pipeline, game.get("home_team"))
    away_key = _team_match_key(pipeline, game.get("away_team"))

    if team_one_key == home_key:
        team_one_is_home = True
    elif team_one_key == away_key:
        team_one_is_home = False
    else:
        team_one_is_home = True

    team_one_abbr = game["home_abbr"] if team_one_is_home else game["away_abbr"]
    team_two_abbr = game["away_abbr"] if team_one_is_home else game["home_abbr"]

    return {
        "team_one": team_one,
        "team_two": team_two,
        "team_one_is_home": team_one_is_home,
        "team_one_abbr": team_one_abbr,
        "team_two_abbr": team_two_abbr,
    }


def print_single_game_report(pipeline, game, result, err, date_str, team_one, team_two):
    display = get_single_game_display_context(pipeline, game, team_one, team_two)
    matchup = f"{display['team_one']} vs {display['team_two']}"
    print(f"\n  🏀  SPORTS ORACLE — SINGLE GAME")
    print(f"      {matchup}")
    print(f"      Date: {display_date(date_str)}\n")

    if err or result is None:
        print(f"  ⚠️  Could not generate prediction: {err or 'Unknown error'}\n")
        return

    model_spread = format_spread(result.spread, game["home_abbr"], game["away_abbr"])
    vegas_spread = (
        game["odds_detail"]
        if game.get("odds_detail")
        else format_spread(game.get("betting_spread"), game["home_abbr"], game["away_abbr"])
    )

    try:
        vegas_total = (
            f"{float(game['over_under']):.1f}"
            if pd.notna(game.get("over_under")) and str(game.get("over_under")).strip() != ""
            else "—"
        )
    except (TypeError, ValueError):
        vegas_total = "—"

    projected_team_one = result.home_score if display["team_one_is_home"] else result.away_score
    projected_team_two = result.away_score if display["team_one_is_home"] else result.home_score

    print(f"  Projected Winner:      {result.predicted_winner} ({result.winner_prob:.0%})")
    print(
        f"  Projected Score:       "
        f"{display['team_one']} {projected_team_one:.1f} — "
        f"{display['team_two']} {projected_team_two:.1f}"
    )
    print(f"  Model Spread:          {model_spread}")
    print(f"  Market Spread:         {vegas_spread}")
    print(f"  Projected Total:       {result.total:.1f}")
    print(f"  Market Total:          {vegas_total}")
    print(f"  Confidence:            {result.confidence} ({result.confidence_score:.2f})")
    print(f"  Market Source:         {game.get('line_source', 'n/a')}")

    ats_profile = ats_pick_profile(
        result.spread,
        game.get("betting_spread"),
        game["home_team"],
        game["away_team"],
        game["home_abbr"],
        game["away_abbr"],
        margin_uncertainty=result.margin_uncertainty,
        confidence_score=result.confidence_score,
        provider_count=game.get("line_provider_count"),
        spread_range_width=game.get("spread_range_width"),
        line_move_value=game.get("line_move_value"),
    )
    print(f"  ATS Pick:              {ats_profile['pick_display']}")
    print(f"  ATS Confidence:        {ats_profile['confidence_label']}")
    print(f"  ATS Decision:          {ats_profile['play_pass']}")
    print(f"  Books:                 {game.get('books_display', '—')}")
    print(f"  Spread Range:          {game.get('spread_range_display', '—')}")
    print(f"  Line Move:             {game.get('line_move_display', '—')}")

    if game.get("venue"):
        print(f"  Venue:                 {game['venue']}")

    if game.get("is_final") and game.get("home_score") is not None and game.get("away_score") is not None:
        actual_margin = game["home_score"] - game["away_score"]
        actual_total = game["away_score"] + game["home_score"]
        model_picks_home = result.spread < 0
        actual_home_won = game["home_score"] > game["away_score"]
        su = "W" if model_picks_home == actual_home_won else "L"

        ats = "—"
        if game.get("betting_spread") is not None:
            try:
                v_spread = float(game["betting_spread"])
                home_covered = actual_margin > -v_spread
                push = actual_margin == -v_spread
                model_likes_home_ats = (-result.spread) > -v_spread
                if push:
                    ats = "P"
                else:
                    ats = "W" if model_likes_home_ats == home_covered else "L"
            except (TypeError, ValueError):
                pass

        ou = "—"
        ou_grade = "—"
        if game.get("over_under") is not None:
            try:
                market_total = float(game["over_under"])
                if actual_total > market_total:
                    ou = "O"
                elif actual_total < market_total:
                    ou = "U"
                else:
                    ou = "P"

                if ou == "P" or math.isclose(result.total, market_total, abs_tol=0.05):
                    ou_grade = "P"
                else:
                    model_likes_over = result.total > market_total
                    actual_over = actual_total > market_total
                    ou_grade = "W" if model_likes_over == actual_over else "L"
            except (TypeError, ValueError):
                pass

        actual_team_one = game["home_score"] if display["team_one_is_home"] else game["away_score"]
        actual_team_two = game["away_score"] if display["team_one_is_home"] else game["home_score"]
        print(f"  Actual Score:          {display['team_one']} {actual_team_one} — {display['team_two']} {actual_team_two}")
        print(f"  Straight Up (SU):      {su}")
        print(f"  Against Spread (ATS):  {ats}")
        print(f"  Over / Under Result:   {ou}")
        print(f"  Over / Under (O/U):    {ou_grade}")
        print(f"  Actual Total:          {actual_total}")
        print(f"  Total Difference:      {result.total - actual_total:+.1f}")

    print()


def run_single_game(
    pipeline,
    engine,
    ml,
    date_str,
    team_one,
    team_two,
    venue=None,
    home_seed=None,
    away_seed=None,
    tournament_round=1,
    is_ncaa_tournament=False,
):
    scoreboard_game = find_single_game(pipeline, date_str, team_one, team_two)

    game = {
        "home_team": team_one,
        "away_team": team_two,
        "home_abbr": team_one[:3].upper(),
        "away_abbr": team_two[:3].upper(),
        "home_score": None,
        "away_score": None,
        "is_final": False,
        "venue": venue,
        "betting_spread": None,
        "over_under": None,
        "odds_detail": "",
        "line_source": "manual",
    }

    if scoreboard_game:
        game.update(scoreboard_game)
        if venue:
            game["venue"] = venue

    selected_game_date = None
    try:
        selected_game_date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        selected_game_date = None

    result, err = safe_predict(
        pipeline,
        engine,
        ml,
        home=game["home_team"],
        away=game["away_team"],
        venue=game.get("venue"),
        game_date=selected_game_date,
        espn_spread=game.get("betting_spread"),
        espn_total=game.get("over_under"),
        tournament_round=tournament_round,
        home_seed=home_seed,
        away_seed=away_seed,
        is_ncaa_tournament=is_ncaa_tournament,
    )
    print_single_game_report(
        pipeline,
        game,
        result,
        err,
        date_str,
        team_one=team_one,
        team_two=team_two,
    )


def run_matchup_session(
    pipeline,
    engine,
    ml,
    date_str,
    venue=None,
    tournament_round=1,
    is_ncaa_tournament=False,
    refresh_odds=False,
):
    session_date = date_str
    session_round = tournament_round
    session_venue = venue

    print("  🔁  Matchup session ready.")
    print("      The pipeline, ratings, and ML model stay loaded in memory.")
    print("      Enter teams one matchup at a time and reuse the same session.\n")

    if refresh_odds:
        refresh_odds_cache_for_dates(pipeline, [session_date])
    fetch_games(pipeline, session_date)

    while True:
        settings_line = f"Date {display_date(session_date)}"
        if is_ncaa_tournament:
            settings_line += f" | Round {session_round}"
        if session_venue:
            settings_line += f" | Venue override: {session_venue}"
        print(f"  Current settings: {settings_line}")

        team_one = prompt_text("Team 1", required=True)
        team_two = prompt_text("Team 2", required=True)

        run_single_game(
            pipeline,
            engine,
            ml,
            date_str=session_date,
            team_one=team_one,
            team_two=team_two,
            venue=session_venue,
            tournament_round=session_round,
            is_ncaa_tournament=is_ncaa_tournament,
        )

        if not prompt_yes_no("Look up another matchup?", default=True):
            print("  Session ended.\n")
            return

        if prompt_yes_no("Change session settings?", default=False):
            session_date, session_round, session_venue = prompt_matchup_session_settings(
                session_date,
                session_round,
                session_venue,
                is_ncaa_tournament,
            )
            if refresh_odds:
                refresh_odds_cache_for_dates(pipeline, [session_date])
            fetch_games(pipeline, session_date)


def build_scoreboard_report(
    pipeline,
    engine,
    ml,
    date_str,
    include_date=False,
    is_ncaa_tournament=False,
):
    games = fetch_games(pipeline, date_str)
    if not games:
        return None

    slate_game_date = None
    try:
        slate_game_date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        slate_game_date = None

    rows = []
    su_w, su_l = 0, 0
    ats_w, ats_l, ats_p = 0, 0, 0
    ats_play_w, ats_play_l, ats_play_p = 0, 0, 0
    ou_w, ou_l, ou_p = 0, 0, 0
    errors = 0

    for g in games:
        v_spread_raw = g.get("betting_spread")
        v_ou_raw = g.get("over_under")
        h_abbr, a_abbr = g["home_abbr"], g["away_abbr"]
        matchup_str = f"{g['away_team']} @ {g['home_team']}"

        result, err = safe_predict(
            pipeline, engine, ml,
            g["home_team"], g["away_team"], g.get("venue"),
            game_date=slate_game_date,
            espn_spread=v_spread_raw, espn_total=v_ou_raw,
            is_ncaa_tournament=is_ncaa_tournament,
        )

        base_row = {
            "date_display": display_date(date_str) if include_date else "",
            "matchup": matchup_str,
        }

        if err:
            errors += 1
            rows.append({
                **base_row,
                "proj_score": "", "proj_total": "", "vegas_ou": "",
                "actual_total": "", "tot_diff": "",
                "model_spread": "", "vegas_spread": "",
                "ats_pick": "", "ats_pick_display": "",
                "ats_confidence": "", "play_pass": "",
                "books_display": "—",
                "spread_range_display": "—",
                "line_move_display": "—",
                "prob": "",
                "winner": "", "winner_display": "",
                "su": "", "ats": "", "ou": "", "ou_grade": "", "score": "", "error": err,
            })
            continue

        p_away = int(round(result.away_score))
        p_home = int(round(result.home_score))
        proj_score = f"{p_away}-{p_home}"
        proj_total = f"{result.total:.1f}"

        try:
            veg_ou_str = (
                f"{float(v_ou_raw):.1f}"
                if pd.notna(v_ou_raw) and str(v_ou_raw).strip() != ""
                else "—"
            )
        except (TypeError, ValueError):
            veg_ou_str = "—"

        model_spread = result.spread
        mod_str = format_spread(model_spread, h_abbr, a_abbr)
        veg_str = g["odds_detail"] if g["odds_detail"] else format_spread(v_spread_raw, h_abbr, a_abbr)
        ats_profile = ats_pick_profile(
            model_spread,
            v_spread_raw,
            g["home_team"],
            g["away_team"],
            h_abbr,
            a_abbr,
            margin_uncertainty=result.margin_uncertainty,
            confidence_score=result.confidence_score,
            provider_count=g.get("line_provider_count"),
            spread_range_width=g.get("spread_range_width"),
            line_move_value=g.get("line_move_value"),
        )
        ats_pick_name = ats_profile["pick_name"]
        ats_pick_display = ats_profile["pick_display"]
        ats_confidence = ats_profile["confidence_label"]
        play_pass = ats_profile["play_pass"]
        model_picks_home = model_spread < 0

        if g["is_final"]:
            actual_home_won = g["home_score"] > g["away_score"]
            winner_name = g["home_team"] if actual_home_won else g["away_team"]
            winner_display = h_abbr if actual_home_won else a_abbr
            su_correct = model_picks_home == actual_home_won

            if su_correct:
                su_w += 1
                su_icon = "W"
            else:
                su_l += 1
                su_icon = "L"

            ats_icon = "—"
            if v_spread_raw is not None:
                try:
                    actual_margin = g["home_score"] - g["away_score"]
                    v_spread_float = float(v_spread_raw)
                    home_covered = actual_margin > -v_spread_float
                    push = actual_margin == -v_spread_float
                    model_likes_home_ats = (-model_spread) > -v_spread_float

                    if push:
                        ats_icon = "P"
                        ats_p += 1
                    elif model_likes_home_ats == home_covered:
                        ats_icon = "W"
                        ats_w += 1
                    else:
                        ats_icon = "L"
                        ats_l += 1
                except (TypeError, ValueError):
                    pass

            ou_icon = "—"
            ou_grade = "—"
            if v_ou_raw is not None:
                try:
                    actual_total = g["away_score"] + g["home_score"]
                    market_total = float(v_ou_raw)
                    if actual_total > market_total:
                        ou_icon = "O"
                    elif actual_total < market_total:
                        ou_icon = "U"
                    else:
                        ou_icon = "P"

                    if ou_icon == "P" or math.isclose(result.total, market_total, abs_tol=0.05):
                        ou_grade = "P"
                        ou_p += 1
                    else:
                        model_likes_over = result.total > market_total
                        actual_over = actual_total > market_total
                        if model_likes_over == actual_over:
                            ou_grade = "W"
                            ou_w += 1
                        else:
                            ou_grade = "L"
                            ou_l += 1
                except (TypeError, ValueError):
                    pass

            actual_total_val = str(g["away_score"] + g["home_score"])
            try:
                tot_diff_val = result.total - (g["away_score"] + g["home_score"])
                tot_diff_str = f"{tot_diff_val:+.1f}"
            except (TypeError, ValueError):
                tot_diff_str = "—"

            rows.append({
                **base_row,
                "proj_score": proj_score, "proj_total": proj_total,
                "vegas_ou": veg_ou_str,
                "actual_total": actual_total_val,
                "tot_diff": tot_diff_str,
                "model_spread": mod_str, "vegas_spread": veg_str,
                "ats_pick": ats_pick_name,
                "ats_pick_display": ats_pick_display,
                "ats_confidence": ats_confidence,
                "play_pass": play_pass,
                "books_display": g.get("books_display", "—"),
                "spread_range_display": g.get("spread_range_display", "—"),
                "line_move_display": g.get("line_move_display", "—"),
                "prob": f"{result.winner_prob:.0%}",
                "winner": winner_name,
                "winner_display": winner_display,
                "su": su_icon, "ats": ats_icon, "ou": ou_icon, "ou_grade": ou_grade,
                "score": f"{g['away_score']}-{g['home_score']}",
                "error": "",
            })
            if play_pass == "PLAY":
                if ats_icon == "W":
                    ats_play_w += 1
                elif ats_icon == "L":
                    ats_play_l += 1
                elif ats_icon == "P":
                    ats_play_p += 1
        else:
            rows.append({
                **base_row,
                "proj_score": proj_score, "proj_total": proj_total,
                "vegas_ou": veg_ou_str,
                "actual_total": "", "tot_diff": "",
                "model_spread": mod_str, "vegas_spread": veg_str,
                "ats_pick": ats_pick_name,
                "ats_pick_display": ats_pick_display,
                "ats_confidence": ats_confidence,
                "play_pass": play_pass,
                "books_display": g.get("books_display", "—"),
                "spread_range_display": g.get("spread_range_display", "—"),
                "line_move_display": g.get("line_move_display", "—"),
                "prob": f"{result.winner_prob:.0%}",
                "winner": "",
                "winner_display": "",
                "su": "", "ats": "", "ou": "", "ou_grade": "", "score": "",
                "error": "",
            })

    return {
        "rows": rows,
        "games": len(games),
        "predicted": len(games) - errors,
        "errors": errors,
        "su_w": su_w,
        "su_l": su_l,
        "ats_w": ats_w,
        "ats_l": ats_l,
        "ats_p": ats_p,
        "ats_play_w": ats_play_w,
        "ats_play_l": ats_play_l,
        "ats_play_p": ats_play_p,
        "ou_w": ou_w,
        "ou_l": ou_l,
        "ou_p": ou_p,
        "date_count": 1,
    }


def render_scoreboard_report(report, header_label: str, export_label: str):
    rows = report["rows"]
    if not rows:
        return

    try:
        term_width = shutil.get_terminal_size((120, 20)).columns
    except Exception:
        term_width = 120
    term_width = max(110, min(term_width, 220))

    has_finals = any(r["score"] for r in rows)
    show_date_column = any(r.get("date_display") for r in rows)

    w_date = 10
    w_ps = 10
    w_pt = 8
    w_ou = 9
    w_at = 7
    w_td = 8
    w_mod = 13
    w_veg = 15
    w_vpick = 9
    w_aconf = 8
    w_play = 6
    w_books = 5
    w_range = 11
    w_move = 10
    w_prob = 5
    w_win = 10
    w_su = 4
    w_ats = 4
    w_ou_res = 4
    w_score = 9

    fixed_extra = w_date + 3 if show_date_column else 0
    if has_finals:
        fixed = (
            w_ps + w_pt + w_ou + w_at + w_td + w_mod + w_veg + w_prob
            + w_vpick + w_aconf + w_play + w_books + w_range + w_move
            + w_win + w_su + w_ats + w_ou_res + w_score + 60 + fixed_extra
        )
    else:
        fixed = (
            w_ps + w_pt + w_ou + w_mod + w_veg + w_vpick + w_aconf + w_play
            + w_books + w_range + w_move + w_prob + 36 + fixed_extra
        )

    w_match = max(20, term_width - fixed)

    def sep(char="─", join="┼", left="├", right="┤"):
        cols = [w_match, w_ps, w_pt, w_ou]
        if show_date_column:
            cols = [w_date] + cols
        if has_finals:
            cols += [w_at, w_td]
        cols += [w_mod, w_veg, w_vpick, w_aconf, w_play, w_books, w_range, w_move, w_prob]
        if has_finals:
            cols += [w_win, w_su, w_ats, w_ou_res, w_score]
        return left + join.join(char * (c + 2) for c in cols) + right

    def top():
        return sep("─", "┬", "┌", "┐")

    def bot():
        return sep("─", "┴", "└", "┘")

    meta_line = (
        f"{report['games']} games | {report['predicted']} predicted | "
        f"{report['errors']} no data"
    )
    if show_date_column and report.get("date_count", 1) > 1:
        meta_line = f"{report['date_count']} days | " + meta_line

    print(f"\n  🏀  SPORTS ORACLE — {header_label}")
    print(f"      {meta_line}\n")

    print(top())

    if has_finals:
        header = ""
        if show_date_column:
            header += f"│ {'Date':^{w_date}} "
        header += (
            f"│ {'Matchup':<{w_match}} "
            f"│ {'Proj Score':^{w_ps}} "
            f"│ {'Proj OT':^{w_pt}} "
            f"│ {'Veg O/U':^{w_ou}} "
            f"│ {'Act Tot':^{w_at}} "
            f"│ {'Tot Diff':^{w_td}} "
            f"│ {'Model Sprd':^{w_mod}} "
            f"│ {'Vegas Sprd':^{w_veg}} "
            f"│ {'ATS Pick':^{w_vpick}} "
            f"│ {'ATS Conf':^{w_aconf}} "
            f"│ {'Play':^{w_play}} "
            f"│ {'Books':^{w_books}} "
            f"│ {'Range':^{w_range}} "
            f"│ {'Move':^{w_move}} "
            f"│ {'Prob':>{w_prob}} "
            f"│ {'Winner':^{w_win}} "
            f"│ {'SU':^{w_su}} "
            f"│ {'ATS':^{w_ats}} "
            f"│ {'O/U':^{w_ou_res}} "
            f"│ {'Score':>{w_score}} │"
        )
        print(header)
    else:
        header = ""
        if show_date_column:
            header += f"│ {'Date':^{w_date}} "
        header += (
            f"│ {'Matchup':<{w_match}} "
            f"│ {'Proj Score':^{w_ps}} "
            f"│ {'Proj OT':^{w_pt}} "
            f"│ {'Veg O/U':^{w_ou}} "
            f"│ {'Model Sprd':^{w_mod}} "
            f"│ {'Vegas Sprd':^{w_veg}} "
            f"│ {'ATS Pick':^{w_vpick}} "
            f"│ {'ATS Conf':^{w_aconf}} "
            f"│ {'Play':^{w_play}} "
            f"│ {'Books':^{w_books}} "
            f"│ {'Range':^{w_range}} "
            f"│ {'Move':^{w_move}} "
            f"│ {'Prob':>{w_prob}} │"
        )
        print(header)

    print(sep())

    for r in rows:
        matchup_cell = r["matchup"]
        if len(matchup_cell) > w_match:
            matchup_cell = matchup_cell[:w_match - 2] + ".."

        if r["error"]:
            if has_finals:
                err_w = (
                    w_ps + w_pt + w_ou + w_at + w_td + w_mod + w_veg
                    + w_vpick + w_aconf + w_play + w_books + w_range + w_move
                    + w_prob + w_win + w_su + w_ats + w_ou_res + w_score + 56
                )
            else:
                err_w = (
                    w_ps + w_pt + w_ou + w_mod + w_veg + w_vpick + w_aconf
                    + w_play + w_books + w_range + w_move + w_prob + 32
                )
            if show_date_column:
                print(
                    f"│ {r['date_display']:<{w_date}} "
                    f"│ {matchup_cell:<{w_match}} │ {'⚠️ ' + r['error']:<{err_w}}│"
                )
            else:
                print(f"│ {matchup_cell:<{w_match}} │ {'⚠️ ' + r['error']:<{err_w}}│")
            continue

        row_prefix = f"│ {r['date_display']:<{w_date}} " if show_date_column else ""
        if has_finals:
            winner_cell = r["winner_display"]
            if len(winner_cell) > w_win:
                winner_cell = winner_cell[:w_win - 2] + ".."
            print(
                row_prefix
                + f"│ {matchup_cell:<{w_match}} "
                f"│ {r['proj_score']:^{w_ps}} "
                f"│ {r['proj_total']:^{w_pt}} "
                f"│ {r['vegas_ou']:^{w_ou}} "
                f"│ {r['actual_total']:^{w_at}} "
                f"│ {r['tot_diff']:^{w_td}} "
                f"│ {r['model_spread']:^{w_mod}} "
                f"│ {r['vegas_spread']:^{w_veg}} "
                f"│ {r['ats_pick_display']:^{w_vpick}} "
                f"│ {r['ats_confidence']:^{w_aconf}} "
                f"│ {r['play_pass']:^{w_play}} "
                f"│ {r['books_display']:^{w_books}} "
                f"│ {r['spread_range_display']:^{w_range}} "
                f"│ {r['line_move_display']:^{w_move}} "
                f"│ {r['prob']:>{w_prob}} "
                f"│ {winner_cell:^{w_win}} "
                f"│ {r['su']:^{w_su}} "
                f"│ {r['ats']:^{w_ats}} "
                f"│ {r['ou']:^{w_ou_res}} "
                f"│ {r['score']:>{w_score}} │"
            )
        else:
            print(
                row_prefix
                + f"│ {matchup_cell:<{w_match}} "
                f"│ {r['proj_score']:^{w_ps}} "
                f"│ {r['proj_total']:^{w_pt}} "
                f"│ {r['vegas_ou']:^{w_ou}} "
                f"│ {r['model_spread']:^{w_mod}} "
                f"│ {r['vegas_spread']:^{w_veg}} "
                f"│ {r['ats_pick_display']:^{w_vpick}} "
                f"│ {r['ats_confidence']:^{w_aconf}} "
                f"│ {r['play_pass']:^{w_play}} "
                f"│ {r['books_display']:^{w_books}} "
                f"│ {r['spread_range_display']:^{w_range}} "
                f"│ {r['line_move_display']:^{w_move}} "
                f"│ {r['prob']:>{w_prob}} │"
            )

    print(bot())

    su_w = report["su_w"]
    su_l = report["su_l"]
    ats_w = report["ats_w"]
    ats_l = report["ats_l"]
    ats_p = report["ats_p"]
    ats_play_w = report.get("ats_play_w", 0)
    ats_play_l = report.get("ats_play_l", 0)
    ats_play_p = report.get("ats_play_p", 0)
    ou_w = report.get("ou_w", 0)
    ou_l = report.get("ou_l", 0)
    ou_p = report.get("ou_p", 0)
    errors = report["errors"]
    total_su = su_w + su_l
    total_ats = ats_w + ats_l
    total_ats_play = ats_play_w + ats_play_l
    total_ou = ou_w + ou_l

    if total_su > 0 or errors > 0:
        print()
        print(f"  📊  RESULTS — {header_label}")
        print(f"  {'─' * 45}")

    if total_su > 0:
        su_pct = su_w / total_su
        print(f"  Straight Up (SU):      {su_w}-{su_l} ({su_pct:.1%})")
        if total_ats > 0:
            ats_pct = ats_w / total_ats
            ats_rec = f"{ats_w}-{ats_l}"
            if ats_p > 0:
                ats_rec += f"-{ats_p}"
            print(f"  Against Spread (ATS):  {ats_rec} ({ats_pct:.1%})")
        if total_ats_play > 0:
            ats_play_pct = ats_play_w / total_ats_play
            ats_play_rec = f"{ats_play_w}-{ats_play_l}"
            if ats_play_p > 0:
                ats_play_rec += f"-{ats_play_p}"
            print(f"  ATS Plays:             {ats_play_rec} ({ats_play_pct:.1%})")
        if total_ou > 0:
            ou_pct = ou_w / total_ou
            ou_rec = f"{ou_w}-{ou_l}"
            if ou_p > 0:
                ou_rec += f"-{ou_p}"
            print(f"  Over / Under (O/U):    {ou_rec} ({ou_pct:.1%})")

    if errors > 0:
        print(f"  Skipped:               {errors} (team not in BartTorvik)")

    if total_su > 0 or errors > 0:
        print()

    xlsx_path = export_table_to_xlsx(rows, export_label, has_finals)
    print(f"  💾  XLSX saved to {xlsx_path}\n")


def run(pipeline, engine, ml, date_str, is_ncaa_tournament=False):
    report = build_scoreboard_report(
        pipeline,
        engine,
        ml,
        date_str,
        is_ncaa_tournament=is_ncaa_tournament,
    )
    if not report:
        return
    render_scoreboard_report(
        report,
        header_label=display_date(date_str),
        export_label=date_str,
    )


def run_date_range(
    pipeline,
    engine,
    ml,
    start_date_str,
    end_date_str,
    is_ncaa_tournament=False,
):
    date_strings = iter_date_range(start_date_str, end_date_str)
    aggregate = {
        "rows": [],
        "games": 0,
        "predicted": 0,
        "errors": 0,
        "su_w": 0,
        "su_l": 0,
        "ats_w": 0,
        "ats_l": 0,
        "ats_p": 0,
        "ats_play_w": 0,
        "ats_play_l": 0,
        "ats_play_p": 0,
        "ou_w": 0,
        "ou_l": 0,
        "ou_p": 0,
        "date_count": len(date_strings),
    }

    for date_str in date_strings:
        report = build_scoreboard_report(
            pipeline,
            engine,
            ml,
            date_str,
            include_date=True,
            is_ncaa_tournament=is_ncaa_tournament,
        )
        if not report:
            continue

        aggregate["rows"].extend(report["rows"])
        aggregate["games"] += report["games"]
        aggregate["predicted"] += report["predicted"]
        aggregate["errors"] += report["errors"]
        aggregate["su_w"] += report["su_w"]
        aggregate["su_l"] += report["su_l"]
        aggregate["ats_w"] += report["ats_w"]
        aggregate["ats_l"] += report["ats_l"]
        aggregate["ats_p"] += report["ats_p"]
        aggregate["ats_play_w"] += report.get("ats_play_w", 0)
        aggregate["ats_play_l"] += report.get("ats_play_l", 0)
        aggregate["ats_play_p"] += report.get("ats_play_p", 0)
        aggregate["ou_w"] += report.get("ou_w", 0)
        aggregate["ou_l"] += report.get("ou_l", 0)
        aggregate["ou_p"] += report.get("ou_p", 0)

    if not aggregate["rows"]:
        print("  No games found in that date range.\n")
        return

    ordered_dates = iter_date_range(start_date_str, end_date_str)
    render_scoreboard_report(
        aggregate,
        header_label=f"{display_date(ordered_dates[0])} to {display_date(ordered_dates[-1])}",
        export_label=f"{ordered_dates[0]}_to_{ordered_dates[-1]}",
    )


def main():
    global DEBUG

    args = prompt_run_mode(parse_args())
    DEBUG = args.debug
    if args.end_date and not args.date:
        raise SystemExit("Date range mode requires a start date.")
    if args.end_date and (args.home or args.away):
        raise SystemExit("Date ranges are only supported for full-scoreboard mode.")
    if getattr(args, "odds_snapshot", False) and args.end_date:
        raise SystemExit("Odds snapshot export supports one date at a time.")

    date_str = parse_date_arg(args.date)
    end_date_str = parse_date_arg(args.end_date) if args.end_date else None
    if end_date_str:
        ordered_dates = iter_date_range(date_str, end_date_str)
        banner_date = f"{display_date(ordered_dates[0])} to {display_date(ordered_dates[-1])}"
    else:
        banner_date = display_date(date_str)
    if getattr(args, "odds_snapshot", False):
        banner_mode = "Odds Snapshot Export"
    elif getattr(args, "matchup_session", False):
        banner_mode = "Matchup Session"
    else:
        banner_mode = "NCAA Basketball Predictions"
    print(f"\n  🏀  SPORTS ORACLE — {banner_mode}")
    print(f"      Date: {banner_date}\n")
    if not getattr(args, "odds_snapshot", False):
        print(
            "      NCAA Tournament Mode: "
            f"{'ON' if args.ncaa_tournament else 'OFF'}"
        )
        print(
            "      Odds Refresh: "
            f"{'ON' if args.refresh_odds else 'OFF'}\n"
        )
    else:
        if args.snapshot_time:
            print(f"      Historical Snapshot: {args.snapshot_time}\n")
        else:
            print("      Snapshot Mode: automatic current/past-day selection\n")

    if getattr(args, "odds_snapshot", False):
        export_odds_snapshot(date_str, snapshot_time=args.snapshot_time)
        return

    pipeline, engine, ml = init(is_ncaa_tournament=bool(args.ncaa_tournament))

    if getattr(args, "matchup_session", False):
        run_matchup_session(
            pipeline,
            engine,
            ml,
            date_str=date_str,
            venue=args.venue,
            tournament_round=args.round,
            is_ncaa_tournament=bool(args.ncaa_tournament),
            refresh_odds=bool(args.refresh_odds),
        )
        return

    if args.home or args.away:
        if not args.home or not args.away:
            raise SystemExit("Single-game mode requires both --home and --away.")
        if args.refresh_odds:
            refresh_odds_cache_for_dates(pipeline, [date_str])
        run_single_game(
            pipeline,
            engine,
            ml,
            date_str=date_str,
            team_one=args.home,
            team_two=args.away,
            venue=args.venue,
            home_seed=args.home_seed,
            away_seed=args.away_seed,
            tournament_round=args.round,
            is_ncaa_tournament=bool(args.ncaa_tournament),
        )
        return

    if end_date_str:
        if args.refresh_odds:
            refresh_odds_cache_for_dates(
                pipeline,
                iter_date_range(date_str, end_date_str),
            )
        run_date_range(
            pipeline,
            engine,
            ml,
            date_str,
            end_date_str,
            is_ncaa_tournament=bool(args.ncaa_tournament),
        )
        return

    if args.refresh_odds:
        refresh_odds_cache_for_dates(pipeline, [date_str])

    run(
        pipeline,
        engine,
        ml,
        date_str,
        is_ncaa_tournament=bool(args.ncaa_tournament),
    )


if __name__ == "__main__":
    main()
