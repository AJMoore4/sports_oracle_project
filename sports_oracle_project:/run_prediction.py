#!/usr/bin/env python3
"""
run_prediction.py

Pull all D1 men's basketball games for any date from ESPN,
predict every matchup, and display clean results.

The Model Spread column IS the pick:
  - Negative spread = model favors the home team (listed second)
  - Positive spread = model favors the away team (listed first)
  - Example: "HOU -8.5" means the model picks Houston by 8.5

USAGE:
  python run_prediction.py                    # today's games
  python run_prediction.py 03/12/2026         # specific date (MM/DD/YYYY)
  python run_prediction.py 03/11/2026 --debug # with full error tracebacks
"""

import os
import sys
import re
import shutil
import traceback
import pandas as pd
from datetime import datetime

SEASON = 2026
DEBUG = "--debug" in sys.argv


def parse_date() -> str:
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            continue
        for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%Y%m%d", "%Y-%m-%d"):
            try:
                return datetime.strptime(arg, fmt).strftime("%Y%m%d")
            except ValueError:
                continue
    return datetime.now().strftime("%Y%m%d")


def display_date(d: str) -> str:
    return f"{d[4:6]}/{d[6:]}/{d[:4]}"


def init():
    from sports_oracle.collectors.pipeline import DataPipeline
    from sports_oracle.engine.prediction_engine import PredictionEngine
    from sports_oracle.engine.ml_model import MLPredictor
    from sports_oracle.backtest.historical_data import HistoricalDataBuilder

    print("  📡  Initializing pipeline (season 2026)...")
    pipeline = DataPipeline(
        cbbd_key=os.environ.get("CBBD_API_KEY", ""),
        season=SEASON,
    )
    engine = PredictionEngine()

    print("  🤖  Training ML model...")
    cbbd_key = os.environ.get("CBBD_API_KEY", "")
    if cbbd_key:
        try:
            from sports_oracle.backtest.live_training import LiveTrainingBuilder
            live_builder = LiveTrainingBuilder()
            print("  📡  Building live training data from CBBD + BartTorvik...")
            df = live_builder.build(
                cbbd_key=cbbd_key,
                seasons=list(range(2012, 2026)),
                cache_dir="data/training_cache",
            )
            if len(df) < 50:
                raise ValueError(f"Only {len(df)} rows — falling back to synthetic")
            print(f"  ✅  Live training data: {len(df)} real tournament games")
        except Exception as e:
            print(f"  ⚠️  Live training failed ({e}), using synthetic data...")
            builder = HistoricalDataBuilder()
            df = builder.build_synthetic_training_set(n_seasons=14)
    else:
        print("  ℹ️  No CBBD key — using synthetic training data")
        print("      (Set CBBD_API_KEY in .env for real historical training)")
        builder = HistoricalDataBuilder()
        df = builder.build_synthetic_training_set(n_seasons=14)
    ml = MLPredictor(blend_weight=0.45)
    ml.train(df)
    return pipeline, engine, ml


def fetch_games(pipeline, date_str):
    print(f"  📅  Fetching games for {display_date(date_str)}...")
    scoreboard = pipeline.espn.get_scoreboard(date=date_str)
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

        odds_detail = str(row.get("odds_detail", "")).strip()
        if odds_detail.lower() in ["nan", "none", ""]:
            odds_detail = ""

        # Parse ESPN spread
        betting_spread = None
        if odds_detail and odds_detail.upper() != "EVEN":
            try:
                match = re.search(r'[-+]?\d*\.?\d+', odds_detail)
                if match:
                    val = abs(float(match.group()))
                    fav_str = odds_detail.split()[0].upper()
                    if (fav_str in home.upper() or fav_str == h_abbr.upper()
                            or home.upper().startswith(fav_str[:3])):
                        betting_spread = -val
                    elif (fav_str in away.upper() or fav_str == a_abbr.upper()
                          or away.upper().startswith(fav_str[:3])):
                        betting_spread = val
                    else:
                        betting_spread = float(match.group())
            except Exception:
                pass

        if betting_spread is None:
            raw_spread = row.get("betting_spread")
            if pd.notna(raw_spread) and str(raw_spread) != "":
                try:
                    betting_spread = float(raw_spread)
                except ValueError:
                    pass

        games.append({
            "home_team": home, "away_team": away,
            "home_abbr": h_abbr, "away_abbr": a_abbr,
            "home_score": h_score, "away_score": a_score,
            "is_final": is_final,
            "venue": row.get("venue_name"),
            "betting_spread": betting_spread,
            "over_under": row.get("over_under"),
            "odds_detail": odds_detail,
        })

    return games


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


def safe_predict(pipeline, engine, ml, home, away, venue=None,
                 espn_spread=None, espn_total=None):
    try:
        inputs = pipeline.get_game_inputs(
            home_team=home, away_team=away,
            tournament_round=1, venue_name=venue, season=SEASON,
            espn_spread=espn_spread, espn_total=espn_total,
        )
        if not inputs.get("home_efficiency"):
            return None, "Missing data"

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


def run(pipeline, engine, ml, date_str):
    import numpy as np

    games = fetch_games(pipeline, date_str)
    if not games:
        return

    rows = []
    su_w, su_l = 0, 0
    ats_w, ats_l, ats_p = 0, 0, 0
    errors = 0

    for g in games:
        v_spread_raw = g.get("betting_spread")
        v_ou_raw = g.get("over_under")
        h_abbr, a_abbr = g["home_abbr"], g["away_abbr"]
        matchup_str = f"{g['away_team']} @ {g['home_team']}"

        result, err = safe_predict(
            pipeline, engine, ml,
            g["home_team"], g["away_team"], g.get("venue"),
            espn_spread=v_spread_raw, espn_total=v_ou_raw,
        )

        if err:
            errors += 1
            rows.append({
                "matchup": matchup_str,
                "proj_score": "", "proj_total": "", "vegas_ou": "",
                "actual_total": "", "tot_diff": "",
                "model_spread": "", "vegas_spread": "", "prob": "",
                "su": "", "ats": "", "score": "", "error": err,
            })
            continue

        # Projected score (away-home to match matchup column order)
        p_away = int(round(result.away_score))
        p_home = int(round(result.home_score))
        proj_score = f"{p_away}-{p_home}"
        proj_total = f"{result.total:.1f}"

        try:
            veg_ou_str = f"{float(v_ou_raw):.1f}" if pd.notna(v_ou_raw) and str(v_ou_raw).strip() != "" else "—"
        except (TypeError, ValueError):
            veg_ou_str = "—"

        # Model spread: negative = home favored
        model_spread = result.spread
        mod_str = format_spread(model_spread, h_abbr, a_abbr)

        # Vegas spread from ESPN
        if g["odds_detail"]:
            veg_str = g["odds_detail"]
        else:
            veg_str = format_spread(v_spread_raw, h_abbr, a_abbr)

        # SU determination: model spread sign tells us who the model picks
        # spread < 0 = home favored, spread > 0 = away favored
        model_picks_home = (model_spread < 0)

        if g["is_final"]:
            actual_home_won = g["home_score"] > g["away_score"]
            su_correct = (model_picks_home == actual_home_won)

            if su_correct:
                su_w += 1
                su_icon = "W"
            else:
                su_l += 1
                su_icon = "L"

            # ATS
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

            # Actual point total for completed games
            actual_total_val = str(g["away_score"] + g["home_score"])

            # Difference: model projected total minus actual total
            try:
                tot_diff_val = result.total - (g["away_score"] + g["home_score"])
                tot_diff_str = f"{tot_diff_val:+.1f}"
            except (TypeError, ValueError):
                tot_diff_str = "—"

            rows.append({
                "matchup": matchup_str,
                "proj_score": proj_score, "proj_total": proj_total,
                "vegas_ou": veg_ou_str,
                "actual_total": actual_total_val,
                "tot_diff": tot_diff_str,
                "model_spread": mod_str, "vegas_spread": veg_str,
                "prob": f"{result.winner_prob:.0%}",
                "su": su_icon, "ats": ats_icon,
                "score": f"{g['away_score']}-{g['home_score']}",
                "error": "",
            })
        else:
            rows.append({
                "matchup": matchup_str,
                "proj_score": proj_score, "proj_total": proj_total,
                "vegas_ou": veg_ou_str,
                "actual_total": "", "tot_diff": "",
                "model_spread": mod_str, "vegas_spread": veg_str,
                "prob": f"{result.winner_prob:.0%}",
                "su": "", "ats": "", "score": "",
                "error": "",
            })

    # ══════════════════════════════════════════════════════════════════
    #  TABLE RENDERING
    # ══════════════════════════════════════════════════════════════════

    try:
        term_width = shutil.get_terminal_size((120, 20)).columns
    except Exception:
        term_width = 120
    term_width = max(110, min(term_width, 220))

    has_finals = any(r["score"] for r in rows)

    # Column widths
    w_ps = 10     # Proj Score
    w_pt = 8      # Proj Tot
    w_ou = 9      # Vegas O/U
    w_at = 7      # Actual Total (only shown when has_finals)
    w_td = 8      # Total Diff (only shown when has_finals)
    w_mod = 13    # Model Spread
    w_veg = 15    # Vegas Spread
    w_prob = 5    # Prob
    w_su = 4      # SU
    w_ats = 4     # ATS
    w_score = 9   # Score

    if has_finals:
        fixed = w_ps + w_pt + w_ou + w_at + w_td + w_mod + w_veg + w_prob + w_su + w_ats + w_score + 36
    else:
        fixed = w_ps + w_pt + w_ou + w_mod + w_veg + w_prob + 21

    w_match = max(20, term_width - fixed)

    # ── Header ────────────────────────────────────────────────────────

    def sep(char="─", join="┼", left="├", right="┤"):
        cols = [w_match, w_ps, w_pt, w_ou]
        if has_finals:
            cols += [w_at, w_td]
        cols += [w_mod, w_veg, w_prob]
        if has_finals:
            cols += [w_su, w_ats, w_score]
        return left + join.join(char * (c + 2) for c in cols) + right

    def top():
        return sep("─", "┬", "┌", "┐")

    def bot():
        return sep("─", "┴", "└", "┘")

    print(f"\n  🏀  SPORTS ORACLE — {display_date(date_str)}")
    print(f"      {len(games)} games | {len(games)-errors} predicted | {errors} no data\n")

    print(top())

    if has_finals:
        print(
            f"│ {'Matchup':<{w_match}} "
            f"│ {'Proj Score':^{w_ps}} "
            f"│ {'Proj OT':^{w_pt}} "
            f"│ {'Veg O/U':^{w_ou}} "
            f"│ {'Act Tot':^{w_at}} "
            f"│ {'Tot Diff':^{w_td}} "
            f"│ {'Model Sprd':^{w_mod}} "
            f"│ {'Vegas Sprd':^{w_veg}} "
            f"│ {'Prob':>{w_prob}} "
            f"│ {'SU':^{w_su}} "
            f"│ {'ATS':^{w_ats}} "
            f"│ {'Score':>{w_score}} │"
        )
    else:
        print(
            f"│ {'Matchup':<{w_match}} "
            f"│ {'Proj Score':^{w_ps}} "
            f"│ {'Proj OT':^{w_pt}} "
            f"│ {'Veg O/U':^{w_ou}} "
            f"│ {'Model Sprd':^{w_mod}} "
            f"│ {'Vegas Sprd':^{w_veg}} "
            f"│ {'Prob':>{w_prob}} │"
        )

    print(sep())

    # ── Rows ──────────────────────────────────────────────────────────

    for r in rows:
        m = r["matchup"]
        if len(m) > w_match:
            m = m[:w_match - 2] + ".."

        if r["error"]:
            if has_finals:
                err_w = w_ps + w_pt + w_ou + w_at + w_td + w_mod + w_veg + w_prob + w_su + w_ats + w_score + 32
            else:
                err_w = w_ps + w_pt + w_ou + w_mod + w_veg + w_prob + 17
            print(f"│ {m:<{w_match}} │ {'⚠️ ' + r['error']:<{err_w}}│")
            continue

        if has_finals:
            print(
                f"│ {m:<{w_match}} "
                f"│ {r['proj_score']:^{w_ps}} "
                f"│ {r['proj_total']:^{w_pt}} "
                f"│ {r['vegas_ou']:^{w_ou}} "
                f"│ {r['actual_total']:^{w_at}} "
                f"│ {r['tot_diff']:^{w_td}} "
                f"│ {r['model_spread']:^{w_mod}} "
                f"│ {r['vegas_spread']:^{w_veg}} "
                f"│ {r['prob']:>{w_prob}} "
                f"│ {r['su']:^{w_su}} "
                f"│ {r['ats']:^{w_ats}} "
                f"│ {r['score']:>{w_score}} │"
            )
        else:
            print(
                f"│ {m:<{w_match}} "
                f"│ {r['proj_score']:^{w_ps}} "
                f"│ {r['proj_total']:^{w_pt}} "
                f"│ {r['vegas_ou']:^{w_ou}} "
                f"│ {r['model_spread']:^{w_mod}} "
                f"│ {r['vegas_spread']:^{w_veg}} "
                f"│ {r['prob']:>{w_prob}} │"
            )

    print(bot())

    # ══════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════

    total_su = su_w + su_l
    total_ats = ats_w + ats_l

    if total_su > 0 or errors > 0:
        print()
        print(f"  📊  RESULTS — {display_date(date_str)}")
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

    if errors > 0:
        print(f"  Skipped:               {errors} (team not in BartTorvik)")

    if total_su > 0 or errors > 0:
        print()


def main():
    date_str = parse_date()
    print(f"\n  🏀  SPORTS ORACLE — NCAA Basketball Predictions")
    print(f"      Date: {display_date(date_str)}\n")
    pipeline, engine, ml = init()
    run(pipeline, engine, ml, date_str)


if __name__ == "__main__":
    main()