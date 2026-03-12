#!/usr/bin/env python3
"""
run_prediction.py

Pull all D1 men's basketball games for any date from ESPN,
predict every matchup, and display clean results.

USAGE:
  python run_prediction.py                    # today's games
  python run_prediction.py 03/12/2026         # specific date (MM/DD/YYYY)
  python run_prediction.py 03/11/2026 --debug # with full error tracebacks
"""

import os
import sys
import traceback
from datetime import datetime

SEASON = 2026
DEBUG = "--debug" in sys.argv


def parse_date() -> str:
    """Parse date from args. Returns YYYYMMDD. Default = today."""
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            continue
        for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%Y%m%d", "%Y-%m-%d"):
            try:
                return datetime.strptime(arg, fmt).strftime("%Y%m%d")
            except ValueError:
                continue
        print(f"  ⚠️  Could not parse date '{arg}'. Use MM/DD/YYYY.")
        sys.exit(1)
    return datetime.now().strftime("%Y%m%d")


def display_date(d: str) -> str:
    return f"{d[4:6]}/{d[6:]}/{d[:4]}"


# ══════════════════════════════════════════════════════════════════════════════

def init():
    from sports_oracle.collectors.pipeline import DataPipeline
    from sports_oracle.engine.prediction_engine import PredictionEngine
    from sports_oracle.engine.ml_model import MLPredictor
    from sports_oracle.backtest.historical_data import HistoricalDataBuilder

    print("  📡  Initializing pipeline (season 2026)...")
    pipeline = DataPipeline(
        cbbd_key=os.environ.get("CBBD_API_KEY", ""),
        odds_key=os.environ.get("ODDS_API_KEY", ""),
        season=SEASON,
    )
    engine = PredictionEngine()

    print("  🤖  Training ML model...")
    builder = HistoricalDataBuilder()
    df = builder.build_synthetic_training_set(n_seasons=14)
    ml = MLPredictor(blend_weight=0.35)
    ml.train(df)
    print(f"       Trained on {len(df)} games | CV accuracy: {ml.win_metrics.cv_mean:.1%}")
    print()
    return pipeline, engine, ml


# ══════════════════════════════════════════════════════════════════════════════

def fetch_games(pipeline, date_str):
    """Fetch all D1 games from ESPN for a given date."""
    print(f"  📅  Fetching games for {display_date(date_str)}...")
    scoreboard = pipeline.espn.get_scoreboard(date=date_str)
    if scoreboard.empty:
        print("       No games found.")
        return []

    games = []
    for _, row in scoreboard.iterrows():
        home = row.get("home_team")
        away = row.get("away_team")
        if not home or not away:
            continue

        status = str(row.get("status", "")).lower()
        is_final = "final" in status or "post" in status

        h_score = None
        a_score = None
        actual_winner = None
        if is_final:
            try:
                h_score = int(row.get("home_score"))
                a_score = int(row.get("away_score"))
                actual_winner = str(home) if h_score > a_score else str(away)
            except (TypeError, ValueError):
                is_final = False

        # Parse ESPN betting spread
        betting_spread = None
        odds_detail = str(row.get("odds_detail", ""))
        fav_abbr = str(row.get("favorite_abbr", ""))
        home_abbr = str(row.get("home_abbr", ""))
        away_abbr = str(row.get("away_abbr", ""))
        raw_spread = row.get("betting_spread")

        if raw_spread is not None and odds_detail:
            try:
                sp = abs(float(raw_spread))
                # ESPN spread is from the favorite's perspective (negative)
                # Convert to home spread: negative = home favored
                if fav_abbr == home_abbr:
                    betting_spread = -sp
                elif fav_abbr == away_abbr:
                    betting_spread = sp
                else:
                    betting_spread = float(raw_spread)
            except (ValueError, TypeError):
                pass

        games.append({
            "home_team": str(home),
            "away_team": str(away),
            "home_score": h_score,
            "away_score": a_score,
            "is_final": is_final,
            "actual_winner": actual_winner,
            "venue": row.get("venue_name"),
            "betting_spread": betting_spread,
            "odds_detail": odds_detail,
        })

    finals = sum(1 for g in games if g["is_final"])
    upcoming = len(games) - finals
    print(f"       Found {len(games)} games ({finals} final, {upcoming} upcoming)")
    print()
    return games


# ══════════════════════════════════════════════════════════════════════════════

def safe_predict(pipeline, engine, ml, home, away, venue=None):
    """Returns (PredictionResult, None) or (None, error_string)."""
    try:
        inputs = pipeline.get_game_inputs(
            home_team=home, away_team=away,
            tournament_round=1, venue_name=venue, season=SEASON,
        )
        home_eff = inputs.get("home_efficiency")
        away_eff = inputs.get("away_efficiency")
        if not home_eff or not home_eff.get("adj_oe"):
            return None, f"No data: {home}"
        if not away_eff or not away_eff.get("adj_oe"):
            return None, f"No data: {away}"

        result = engine.predict(inputs)
        if result is None:
            return None, "Engine error"

        try:
            result = ml.enhance_prediction(result, inputs)
        except Exception:
            pass

        return result, None
    except Exception as e:
        if DEBUG:
            traceback.print_exc()
        return None, str(e)[:50]


# ══════════════════════════════════════════════════════════════════════════════

def run(pipeline, engine, ml, date_str):
    import numpy as np

    games = fetch_games(pipeline, date_str)
    if not games:
        return

    # ── Predict all games ────────────────────────────────────────────
    rows = []
    correct = 0
    wrong = 0
    errors = 0
    margin_errors = []

    for g in games:
        result, err = safe_predict(
            pipeline, engine, ml, g["home_team"], g["away_team"], g.get("venue")
        )

        if err:
            errors += 1
            rows.append({
                "matchup": f"{g['away_team']} vs {g['home_team']}",
                "pick": "—",
                "model_spread": "",
                "vegas_spread": "",
                "prob": "",
                "result_icon": "⚠️",
                "winner": "",
                "score": "",
                "error": err,
            })
            continue

        pick = result.predicted_winner
        model_spread = result.spread

        # Format vegas spread
        vegas_str = ""
        if g["betting_spread"] is not None:
            vegas_str = f"{g['betting_spread']:+.1f}"

        if g["is_final"]:
            actual = g["actual_winner"]
            is_right = (pick == actual)
            if is_right:
                correct += 1
            else:
                wrong += 1

            actual_margin = g["home_score"] - g["away_score"]
            pred_margin = -model_spread
            margin_errors.append(abs(pred_margin - actual_margin))

            rows.append({
                "matchup": f"{g['away_team']} vs {g['home_team']}",
                "pick": pick,
                "model_spread": f"{model_spread:+.1f}",
                "vegas_spread": vegas_str,
                "prob": f"{result.winner_prob:.0%}",
                "result_icon": "✅" if is_right else "❌",
                "winner": actual,
                "score": f"{g['away_score']}-{g['home_score']}",
                "error": "",
            })
        else:
            rows.append({
                "matchup": f"{g['away_team']} vs {g['home_team']}",
                "pick": pick,
                "model_spread": f"{model_spread:+.1f}",
                "vegas_spread": vegas_str,
                "prob": f"{result.winner_prob:.0%}",
                "result_icon": "🔮",
                "winner": "",
                "score": "",
                "error": "",
            })

    # ══════════════════════════════════════════════════════════════════
    #  PRINT RESULTS
    # ══════════════════════════════════════════════════════════════════

    has_finals = any(r["winner"] for r in rows)
    total_predicted = correct + wrong
    upcoming = sum(1 for r in rows if r["result_icon"] == "🔮")

    print()
    print("┌" + "─" * 98 + "┐")
    print(f"│{'':2s}🏀  SPORTS ORACLE — {display_date(date_str)}" + " " * (98 - 26 - len(display_date(date_str))) + "│")
    print(f"│{'':2s}{len(games)} games found | {len(games)-errors} predicted | {errors} failed" + " " * (98 - 46 - len(str(len(games))) - len(str(len(games)-errors)) - len(str(errors))) + "│")
    print("├" + "─" * 98 + "┤")

    # Column header
    if has_finals:
        hdr = (f"│ {'Matchup':<38s}{'Pick':<14s}{'Model':>7s}{'Vegas':>7s}"
               f"{'Prob':>6s}  {'':2s} {'Winner':<14s}{'Score':>9s} │")
        div = "│ " + "─" * 38 + "─" * 14 + "─" * 7 + "─" * 7 + "─" * 6 + "──" + "─" * 2 + "─" * 14 + "─" * 9 + " │"
    else:
        hdr = (f"│ {'Matchup':<38s}{'Pick':<14s}{'Model':>7s}{'Vegas':>7s}"
               f"{'Prob':>6s}{'':>26s}│")
        div = "│ " + "─" * 38 + "─" * 14 + "─" * 7 + "─" * 7 + "─" * 6 + "─" * 26 + "│"

    print(hdr)
    print(div)

    for r in rows:
        if r["error"]:
            line = f"│ {r['matchup']:<38s}{'⚠️ ' + r['error']:<60s}│"
            print(line)
            continue

        matchup = r["matchup"]
        if len(matchup) > 37:
            matchup = matchup[:35] + ".."

        if has_finals:
            line = (f"│ {matchup:<38s}{r['pick']:<14s}{r['model_spread']:>7s}"
                    f"{r['vegas_spread']:>7s}{r['prob']:>6s}  "
                    f"{r['result_icon']:>2s} {r['winner']:<14s}{r['score']:>9s} │")
        else:
            line = (f"│ {matchup:<38s}{r['pick']:<14s}{r['model_spread']:>7s}"
                    f"{r['vegas_spread']:>7s}{r['prob']:>6s}{'':>26s}│")

        print(line)

    # ══════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════

    print("├" + "─" * 98 + "┤")

    if total_predicted > 0:
        pct = correct / total_predicted
        record_str = f"Record: {correct}-{wrong} ({pct:.1%})"
        print(f"│{'':2s}📊  COMPLETED GAMES{'':<80s}│")
        print(f"│{'':6s}{record_str:<93s}│")

        if margin_errors:
            mae = np.mean(margin_errors)
            med = np.median(margin_errors)
            stats_str = f"Avg margin error: {mae:.1f} pts  |  Median: {med:.1f} pts"
            print(f"│{'':6s}{stats_str:<93s}│")

    if upcoming > 0:
        print(f"│{'':2s}🔮  UPCOMING: {upcoming} games predicted{'':<60s}│")

    if errors > 0:
        err_str = f"⚠️  {errors} games skipped (team not in BartTorvik data)"
        print(f"│{'':2s}{err_str:<97s}│")

    print("└" + "─" * 98 + "┘")
    print()


# ══════════════════════════════════════════════════════════════════════════════

def main():
    date_str = parse_date()
    print()
    print("┌" + "─" * 58 + "┐")
    print(f"│{'':2s}🏀  SPORTS ORACLE — NCAA Basketball Predictions{'':11s}│")
    print(f"│{'':2s}    Date: {display_date(date_str)}{'':44s}│")
    print("└" + "─" * 58 + "┘")
    print()

    pipeline, engine, ml = init()
    run(pipeline, engine, ml, date_str)


if __name__ == "__main__":
    main()
