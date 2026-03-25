"""
sports_oracle/engine/prediction_engine.py

Core prediction engine. Consumes the validated output
of pipeline.get_game_inputs() and produces:
  - Projected score for each team
  - Projected total points
  - Projected margin (spread)
  - Win probability for each team
  - Confidence level
  - Full breakdown of every adjustment

ARCHITECTURE:
  Layer 3 — Matchup Projection (formula baseline)
    Step 1: Expected possessions (Game_Pace)
    Step 2: Raw score projection per team
    Step 3: Raw margin and total

  Layer 4 — Additive Adjustments
    Momentum, Experience, Rest, Injury, Seed, Travel
    Each produces a signed point adjustment.
    Sum is added to Raw_Margin.

  Win Probability — Logistic function of Final_Margin
  Confidence — Based on data quality and agreement

USAGE:
    from engine.prediction_engine import PredictionEngine

    engine = PredictionEngine()
    result = engine.predict(game_inputs)

    print(result["home_score"])       # 74.2
    print(result["away_score"])       # 68.8
    print(result["spread"])           # -5.4 (home favored)
    print(result["total"])            # 143.0
    print(result["home_win_prob"])    # 0.72
    print(result["confidence"])       # "HIGH"
"""

from __future__ import annotations
import math
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("sports_oracle.engine")


# ── Constants ─────────────────────────────────────────────────────────────────

# Round pace modifiers — from config but local for engine isolation
ROUND_PACE_MOD = {
    0: 1.00,   # First Four
    1: 1.00,   # First Round
    2: 1.02,   # Second Round
    3: 1.00,   # Sweet 16
    4: 0.99,   # Elite 8
    5: 0.97,   # Final Four
    6: 0.96,   # Championship
}

# Logistic function calibration for margin → win probability
# Calibrated so that a 5-point margin ≈ 68% win probability
# σ = 10.5 fits NCAA tournament data well
LOGISTIC_SIGMA = 10.5

# Maximum total Layer 4 adjustment (points)
MAX_L4_ADJUSTMENT = 6.0
MAX_TOTAL_POINTS_ADJUSTMENT = 12.0

# Conservative total anchors by round.
# Used to pull totals back toward realistic postseason environments.
ROUND_TOTAL_ANCHOR = {
    0: 141.0,   # First Four
    1: 140.5,   # First Round / generic postseason opener
    2: 141.5,
    3: 141.5,
    4: 142.0,
    5: 141.0,
    6: 140.0,
}

# Total calibration weights.
TOTAL_BASELINE_SHRINK = 0.68
TOTAL_MARKET_BLEND = 0.61
TOTAL_MARKET_BLEND_MEDIUM_GAP = 0.63
TOTAL_MARKET_BLEND_HIGH_GAP = 0.70
TOTAL_MARKET_BLEND_EXTREME_GAP = 0.78
TOTAL_MARKET_BLEND_CAP = 0.80
TOTAL_MARKET_GAP_MEDIUM = 3.0
TOTAL_MARKET_GAP_HIGH = 5.0
TOTAL_MARKET_GAP_EXTREME = 8.0
TOTAL_HIGH_THRESHOLD = 150.0
TOTAL_VERY_HIGH_THRESHOLD = 160.0
PROJECTED_TOSSUP_THRESHOLD = 2.5
PROJECTED_SMALL_SPREAD_THRESHOLD = 5.0
PROJECTED_LARGE_FAVORITE_THRESHOLD = 12.0
PROJECTED_HUGE_FAVORITE_THRESHOLD = 18.0
BLOWOUT_TOTAL_UPLIFT_BASE = 1.2
BLOWOUT_TOTAL_UPLIFT_PER_POINT = 0.14
BLOWOUT_TOTAL_UPLIFT_MARKET_FACTOR = 0.08
BLOWOUT_TOTAL_UPLIFT_CAP = 4.5

# Momentum decay — weight for most recent game vs 10th most recent
MOMENTUM_DECAY_LAMBDA = 0.15

# Experience scoring weights
EXP_WEIGHT_ROSTER_AGE = 0.33
EXP_WEIGHT_COACH = 0.27
EXP_WEIGHT_RETURNING = 0.40

# Rest adjustment curve (days → point adjustment)
# 0 days (back-to-back): -1.5 pts
# 1 day: -0.5 pts
# 2 days: 0 (baseline)
# 3+ days: diminishing positive
REST_ADJUSTMENTS = {
    0: -1.5,
    1: -0.5,
    2: 0.0,
    3: 0.3,
    4: 0.4,
    5: 0.3,
    6: 0.2,
    7: 0.1,
}

# Travel distance thresholds (miles → point penalty)
TRAVEL_PENALTY_THRESHOLD = 500    # no penalty under 500 miles
TRAVEL_PENALTY_PER_1000 = 0.3    # 0.3 pts per 1000 miles beyond threshold

# Altitude threshold (feet difference → point penalty)
ALTITUDE_PENALTY_THRESHOLD = 2000  # no penalty under 2000ft diff
ALTITUDE_PENALTY_PER_1000 = 0.1   # 0.1 pts per 1000ft beyond threshold

GAME_TYPE_NCAA = "ncaa_tournament"
GAME_TYPE_CONFERENCE = "conference_tournament"
GAME_TYPE_GENERIC = "generic_postseason"

GAME_TYPE_PACE_MULTIPLIER = {
    GAME_TYPE_NCAA: 1.00,
    GAME_TYPE_CONFERENCE: 0.99,
    GAME_TYPE_GENERIC: 1.00,
}
GAME_TYPE_TOTAL_SHRINK_MULTIPLIER = {
    GAME_TYPE_NCAA: 1.00,
    GAME_TYPE_CONFERENCE: 0.72,
    GAME_TYPE_GENERIC: 0.00,
}
GAME_TYPE_TOTAL_MARKET_BLEND_BOOST = {
    GAME_TYPE_NCAA: 0.00,
    GAME_TYPE_CONFERENCE: 0.06,
    GAME_TYPE_GENERIC: 0.02,
}
GAME_TYPE_SPREAD_MARKET_BASE = {
    GAME_TYPE_NCAA: 0.08,
    GAME_TYPE_CONFERENCE: 0.16,
    GAME_TYPE_GENERIC: 0.12,
}

SPREAD_MARKET_BLEND_MEDIUM_GAP = 0.16
SPREAD_MARKET_BLEND_HIGH_GAP = 0.24
SPREAD_MARKET_BLEND_EXTREME_GAP = 0.32
SPREAD_MARKET_BLEND_CAP = 0.38
SPREAD_MARKET_GAP_MEDIUM = 2.5
SPREAD_MARKET_GAP_HIGH = 4.5
SPREAD_MARKET_GAP_EXTREME = 6.5
SPREAD_MARKET_BLEND_TOSSUP_BOOST = 0.14
SPREAD_MARKET_BLEND_SMALL_BOOST = 0.08
SPREAD_MARKET_BLEND_LARGE_FAVORITE_BOOST = 0.03
SPREAD_MARKET_BLEND_HUGE_FAVORITE_BOOST = 0.06

MATCHUP_INTERACTION_CAP = 6.0
MATCHUP_THREE_POINT_WEIGHT = 1.4
MATCHUP_TWO_POINT_WEIGHT = 1.1
MATCHUP_TURNOVER_WEIGHT = 1.0
MATCHUP_REBOUND_WEIGHT = 0.9
MATCHUP_FOUL_WEIGHT = 1.0


def infer_game_type_from_inputs(inputs: dict) -> str:
    """Infer the game environment for calibration/blending purposes."""
    explicit = str(inputs.get("game_type", "") or "").strip().lower()
    if explicit in {GAME_TYPE_NCAA, GAME_TYPE_CONFERENCE, GAME_TYPE_GENERIC}:
        return explicit

    if inputs.get("is_ncaa_tournament"):
        return GAME_TYPE_NCAA

    for side in ("home_extended", "away_extended"):
        extended = inputs.get(side, {}) or {}
        if int(extended.get("conf_tourney_wins", 0) or 0) > 0:
            return GAME_TYPE_CONFERENCE

    for side in ("home_momentum", "away_momentum"):
        momentum = inputs.get(side, {}) or {}
        conf_games = momentum.get("conf_tourney_games")
        if getattr(conf_games, "empty", True) is False:
            return GAME_TYPE_CONFERENCE

    return GAME_TYPE_GENERIC


# ── Calibration dataclass ─────────────────────────────────────────────────────

@dataclass
class CalibrationProfile:
    """Runtime-calibrated parameters learned from held-out seasons."""
    probability_intercept: float = 0.0
    probability_slope: float = 1.0 / LOGISTIC_SIGMA
    total_baseline_shrink: float = TOTAL_BASELINE_SHRINK
    total_market_blend: float = TOTAL_MARKET_BLEND
    round_total_anchor: dict[int, float] = field(
        default_factory=lambda: dict(ROUND_TOTAL_ANCHOR)
    )
    margin_mae: float = 8.0
    margin_rmse: float = 10.0
    total_mae: float = 11.5
    total_rmse: float = 14.0


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """Full prediction output with breakdown."""
    home_team: str = ""
    away_team: str = ""

    # Layer 3 — Raw projections
    game_pace: float = 68.0
    home_raw_score: float = 70.0
    away_raw_score: float = 70.0
    raw_margin: float = 0.0          # positive = home favored
    raw_total: float = 140.0
    home_matchup_interaction: float = 0.0
    away_matchup_interaction: float = 0.0

    # Layer 4 — Individual adjustments (each is home perspective)
    momentum_adj: float = 0.0
    experience_adj: float = 0.0
    rest_adj: float = 0.0
    injury_adj: float = 0.0
    seed_adj: float = 0.0
    travel_adj: float = 0.0
    total_adjustment: float = 0.0
    total_points_adjustment: float = 0.0

    # Final predictions
    home_score: float = 70.0
    away_score: float = 70.0
    spread: float = 0.0              # negative = home favored
    total: float = 140.0
    home_win_prob: float = 0.50
    away_win_prob: float = 0.50
    margin_uncertainty: float = 8.0
    total_uncertainty: float = 11.5

    # Market comparison
    market_spread: Optional[float] = None
    market_total: Optional[float] = None
    spread_edge: Optional[float] = None
    total_edge: Optional[float] = None

    # Metadata
    confidence: str = "MEDIUM"
    confidence_score: float = 0.50
    tournament_round: int = 1
    season: int = 2025
    game_type: str = GAME_TYPE_GENERIC
    spread_market_adjustment: float = 0.0
    total_market_adjustment: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def predicted_winner(self) -> str:
        return self.home_team if self.home_win_prob > 0.5 else self.away_team

    @property
    def predicted_loser(self) -> str:
        return self.away_team if self.home_win_prob > 0.5 else self.home_team

    @property
    def winner_prob(self) -> float:
        return max(self.home_win_prob, self.away_win_prob)

    def summary(self) -> str:
        """One-line summary string."""
        winner = self.predicted_winner
        prob = self.winner_prob
        return (
            f"{winner} {prob:.0%} | "
            f"{self.home_team} {self.home_score:.0f} – "
            f"{self.away_team} {self.away_score:.0f} | "
            f"Spread: {self.spread:+.1f} | Total: {self.total:.0f} | "
            f"Confidence: {self.confidence}"
        )

    def breakdown(self) -> str:
        """Detailed multi-line breakdown."""
        lines = [
            f"\n{'═'*60}",
            f"  {self.away_team} @ {self.home_team}",
            f"  Round {self.tournament_round} | Season {self.season} | {self.game_type}",
            f"{'═'*60}",
            f"",
            f"  LAYER 3 — Matchup Projection",
            f"  {'─'*40}",
            f"  Game Pace:        {self.game_pace:.1f} possessions",
            f"  {self.home_team:>20s}:  {self.home_raw_score:.1f} raw pts",
            f"  {self.away_team:>20s}:  {self.away_raw_score:.1f} raw pts",
            f"  Matchup Interact.: {self.home_matchup_interaction:+.2f} / {self.away_matchup_interaction:+.2f}",
            f"  Raw Margin:       {self.raw_margin:+.1f} pts",
            f"  Raw Total:        {self.raw_total:.1f} pts",
            f"",
            f"  LAYER 4 — Adjustments (home perspective)",
            f"  {'─'*40}",
            f"  Momentum:         {self.momentum_adj:+.2f} pts",
            f"  Experience:       {self.experience_adj:+.2f} pts",
            f"  Rest:             {self.rest_adj:+.2f} pts",
            f"  Injury:           {self.injury_adj:+.2f} pts",
            f"  Seed History:     {self.seed_adj:+.2f} pts",
            f"  Travel:           {self.travel_adj:+.2f} pts",
            f"  {'─'*40}",
            f"  Margin Adj Sum:   {self.total_adjustment:+.2f} pts",
            f"  Total Calibration:{self.total_points_adjustment:+.2f} pts",
            f"",
            f"  FINAL PREDICTION",
            f"  {'─'*40}",
            f"  {self.home_team:>20s}:  {self.home_score:.1f} pts",
            f"  {self.away_team:>20s}:  {self.away_score:.1f} pts",
            f"  Spread:           {self.spread:+.1f}",
            f"  Total:            {self.total:.1f}",
            f"  Win Probability:  {self.home_team} {self.home_win_prob:.1%} "
            f"| {self.away_team} {self.away_win_prob:.1%}",
            f"  Margin Range:     ±{self.margin_uncertainty:.1f} pts",
            f"  Total Range:      ±{self.total_uncertainty:.1f} pts",
            f"  Confidence:       {self.confidence} ({self.confidence_score:.0%})",
        ]

        if self.market_spread is not None:
            lines.extend([
                f"",
                f"  MARKET COMPARISON",
                f"  {'─'*40}",
                f"  Market Spread:    {self.market_spread:+.1f}",
                f"  Our Spread:       {self.spread:+.1f}",
                f"  Spread Shrink:    {self.spread_market_adjustment:+.2f} pts" if self.spread_market_adjustment else "",
                f"  Spread Edge:      {self.spread_edge:+.1f} pts" if self.spread_edge else "",
            ])
        if self.market_total is not None:
            lines.extend([
                f"  Market Total:     {self.market_total:.1f}",
                f"  Our Total:        {self.total:.1f}",
                f"  Total Shrink:     {self.total_market_adjustment:+.2f} pts" if self.total_market_adjustment else "",
                f"  Total Edge:       {self.total_edge:+.1f} pts" if self.total_edge else "",
            ])

        if self.warnings:
            lines.extend([
                f"",
                f"  ⚠️  WARNINGS",
                *[f"    • {w}" for w in self.warnings],
            ])

        lines.append(f"{'═'*60}")
        return "\n".join(lines)


# ── Prediction Engine ─────────────────────────────────────────────────────────

class PredictionEngine:
    """
    Core prediction engine. Takes validated pipeline inputs
    and produces scored predictions.
    """

    def __init__(
        self,
        calibration_profile: Optional[CalibrationProfile] = None,
    ):
        self.calibration_profile = calibration_profile or CalibrationProfile()

    def set_calibration_profile(
        self,
        calibration_profile: Optional[CalibrationProfile],
    ) -> None:
        self.calibration_profile = calibration_profile or CalibrationProfile()

    def predict(self, inputs: dict) -> PredictionResult:
        """
        Main entry point. Takes the output of
        pipeline.get_game_inputs() and returns a PredictionResult.
        """
        result = PredictionResult(
            home_team=inputs.get("home_team", "Home"),
            away_team=inputs.get("away_team", "Away"),
            tournament_round=inputs.get("tournament_round", 1),
            season=inputs.get("season", 2025),
        )
        is_ncaa_tournament = bool(inputs.get("is_ncaa_tournament", False))
        result.game_type = infer_game_type_from_inputs(inputs)

        home_eff = inputs.get("home_efficiency", {})
        away_eff = inputs.get("away_efficiency", {})
        venue = inputs.get("venue", {})

        if not home_eff or not away_eff:
            result.warnings.append("Missing efficiency data — using defaults")
            home_eff = home_eff or self._default_efficiency()
            away_eff = away_eff or self._default_efficiency()

        # ── Layer 3: Matchup Projection ──────────────────────────────────

        # Step 1: Expected possessions
        result.game_pace = self._compute_game_pace(
            home_eff,
            away_eff,
            venue,
            result.tournament_round,
            result.game_type,
            inputs,
        )

        # National average AdjOE — baseline for Massey scoring formula
        national_avg_oe = inputs.get("national_avg_oe", 105.0)

        home_matchup_eff_adj = self._compute_matchup_interaction(
            offense=home_eff,
            defense=away_eff,
            game_type=result.game_type,
        )
        away_matchup_eff_adj = self._compute_matchup_interaction(
            offense=away_eff,
            defense=home_eff,
            game_type=result.game_type,
        )

        # Step 2: Raw score projection (Massey additive formula)
        result.home_raw_score = self._project_score(
                 offense=home_eff,
                 defense=away_eff,
                 game_pace=result.game_pace,
                 venue=venue,
                 national_avg_oe=national_avg_oe,
                 matchup_eff_adj=home_matchup_eff_adj,
             )
        result.away_raw_score = self._project_score(
                 offense=away_eff,
                 defense=home_eff,
                 game_pace=result.game_pace,
                 venue=venue,
                 national_avg_oe=national_avg_oe,
                 matchup_eff_adj=away_matchup_eff_adj,
             )
        result.home_matchup_interaction = round(
            home_matchup_eff_adj * result.game_pace / 100.0,
            2,
        )
        result.away_matchup_interaction = round(
            away_matchup_eff_adj * result.game_pace / 100.0,
            2,
        )

        # Step 3: Raw margin and total
        result.raw_margin = result.home_raw_score - result.away_raw_score
        result.raw_total = result.home_raw_score + result.away_raw_score

        # ── Layer 4: Additive Adjustments ────────────────────────────────

        result.momentum_adj = self._compute_momentum_adjustment(
            inputs.get("home_momentum", {}),
            inputs.get("away_momentum", {}),
        )

        result.experience_adj = self._compute_experience_adjustment(
            inputs.get("home_experience", {}),
            inputs.get("away_experience", {}),
        )

        result.rest_adj = self._compute_rest_adjustment(
            inputs.get("home_rest", {}),
            inputs.get("away_rest", {}),
        )

        result.injury_adj = self._compute_injury_adjustment(
            inputs.get("injuries"),
            result.home_team,
            result.away_team,
        )

        result.seed_adj = self._compute_seed_adjustment(
            inputs.get("seed_context", {}),
        )

        result.travel_adj = self._compute_travel_adjustment(
            inputs.get("home_travel", {}),
            inputs.get("away_travel", {}),
        )

        # Sum and clamp spread-side adjustments
        raw_adj = (
            result.momentum_adj
            + result.experience_adj
            + result.rest_adj
            + result.injury_adj
            + result.seed_adj
            + result.travel_adj
        )
        result.total_adjustment = max(
            -MAX_L4_ADJUSTMENT,
            min(raw_adj, MAX_L4_ADJUSTMENT),
        )
        result.total_points_adjustment = self._compute_total_points_adjustment(
            inputs,
            result,
            result.game_type,
        )

        # ── Final Predictions ────────────────────────────────────────────

        final_margin = result.raw_margin + result.total_adjustment
        final_total = result.raw_total + result.total_points_adjustment
        final_margin, result.spread_market_adjustment = self._apply_spread_market_calibration(
            inputs=inputs,
            result=result,
            model_margin=final_margin,
        )
        final_total, result.total_market_adjustment = self._apply_total_market_calibration(
            inputs=inputs,
            result=result,
            model_total=final_total,
        )
        result.home_score = final_total / 2 + final_margin / 2
        result.away_score = final_total / 2 - final_margin / 2
        result.spread = -final_margin  # convention: negative = home favored
        result.total = final_total

        # Win probability via logistic function
        result.home_win_prob = self._margin_to_win_prob(final_margin)
        result.away_win_prob = 1.0 - result.home_win_prob
        result.margin_uncertainty = self._estimate_margin_uncertainty(inputs)
        result.total_uncertainty = self._estimate_total_uncertainty(inputs, result)

        # ── Market Comparison ────────────────────────────────────────────

        market = inputs.get("market_lines", {})
        if market:
            result.market_spread = market.get("consensus_spread")
            result.market_total = market.get("consensus_total")
            if result.market_spread is not None:
                try:
                    market_spread = float(result.market_spread)
                    if not math.isfinite(market_spread):
                        raise ValueError
                    if self._market_spread_is_anomalous(final_margin, market_spread):
                        result.market_spread = None
                    else:
                        result.spread_edge = round(
                            result.spread - market_spread, 1
                        )
                except (TypeError, ValueError):
                    result.market_spread = None
            if result.market_total is not None:
                result.total_edge = round(
                    result.total - result.market_total, 1
                )

        # ── Confidence ───────────────────────────────────────────────────

        result.confidence_score = self._compute_confidence(
            inputs, result
        )
        result.confidence = self._confidence_label(result.confidence_score)

        logger.info(f"Prediction: {result.summary()}")
        return result

    # ══════════════════════════════════════════════════════════════════════
    #  LAYER 3 — Matchup Projection
    # ══════════════════════════════════════════════════════════════════════

    def _compute_game_pace(
        self,
        home_eff: dict,
        away_eff: dict,
        venue: dict,
        tournament_round: int,
        game_type: str,
        inputs: dict,
    ) -> float:
        """
        Step 1: Expected possessions.
        Raw_Pace = (Team_A_Pace + Team_B_Pace) / 2
        Game_Pace = Raw_Pace × VPI × Round_Modifier
        """
        home_tempo = home_eff.get("adj_tempo", 68.0)
        away_tempo = away_eff.get("adj_tempo", 68.0)
        raw_pace = (home_tempo + away_tempo) / 2.0

        vpi = venue.get("vpi", 1.0)
        if game_type == GAME_TYPE_NCAA:
            round_mod = ROUND_PACE_MOD.get(tournament_round, 1.0)
        elif game_type == GAME_TYPE_CONFERENCE:
            round_mod = GAME_TYPE_PACE_MULTIPLIER[GAME_TYPE_CONFERENCE]
            avg_rest = self._average_rest_days(inputs)
            conf_pressure = self._conference_pressure_index(inputs)
            if avg_rest <= 1.0:
                round_mod *= 0.985
            elif avg_rest <= 2.0:
                round_mod *= 0.992
            round_mod *= max(0.96, 1.0 - conf_pressure * 0.015)
        else:
            round_mod = GAME_TYPE_PACE_MULTIPLIER.get(game_type, 1.0)

        game_pace = raw_pace * vpi * round_mod

        logger.debug(
            f"  Pace: ({home_tempo:.1f} + {away_tempo:.1f})/2 = {raw_pace:.1f} "
            f"× VPI={vpi:.3f} × Round={round_mod:.2f} = {game_pace:.1f}"
        )
        return game_pace

    def _project_score(
            self,
            offense: dict,
            defense: dict,
            game_pace: float,
            venue: dict,
            national_avg_oe: float = 105.0,
            matchup_eff_adj: float = 0.0,
    ) -> float:
        """
        Step 2: Raw score projection for one team.

        Massey-style additive formula:
          Score = (AdjOE + AdjDE - NatAvg) × (Pace / 100) × VSI × V3P_adj

        Why additive instead of multiplicative:
          The old formula (AdjOE × AdjDE / 100) systematically inflated
          totals by ~7 points because dividing by 100 instead of the
          actual national average (~105) under-corrects the cross-term.

          The additive form correctly calibrates: when both teams are
          exactly average (AdjOE=NatAvg, AdjDE=NatAvg), each team
          scores NatAvg × Pace/100, producing the correct national
          average total.

          The margin is algebraically identical between the two
          formulas — NatAvg cancels in the subtraction (Score_A - Score_B).
          Only totals are affected.

        national_avg_oe: D1 average AdjOE for the current season,
          computed from BartTorvik ratings and passed through the
          pipeline. Falls back to 105.0 if unavailable.
        """
        adj_oe = offense.get("adj_oe", 100.0)
        adj_de = defense.get("adj_de", 100.0)
        three_pt_rate = offense.get("three_pt_rate_off", 0.35)

        vsi = venue.get("vsi", 1.0)
        v3p = venue.get("v3p", 1.0)

        # Massey additive matchup efficiency:
        # How many points per 100 possessions this offense scores
        # against this specific defense, centered on the national average
        matchup_eff = adj_oe + adj_de - national_avg_oe

        # Scale to actual game possessions
        pace_factor = game_pace / 100.0

        # V3P adjustment — 3PT-heavy teams feel venue effects more
        v3p_adj = 1.0 + ((v3p - 1.0) * three_pt_rate * 1.5)

        score = (matchup_eff + matchup_eff_adj) * pace_factor * vsi * v3p_adj

        return score

    def _compute_total_points_adjustment(
        self,
        inputs: dict,
        result: PredictionResult,
        game_type: str,
    ) -> float:
        """
        Apply a conservative calibration to totals.

        1. Shrink raw totals toward a round-specific postseason anchor.
        2. If a market total exists, lean toward it even more.
        """
        conservative_total = self._calibrate_total_baseline(
            inputs=inputs,
            result=result,
            game_type=game_type,
        )
        conservative_total += self._blowout_total_uplift(
            inputs=inputs,
            result=result,
        )

        total_delta = conservative_total - result.raw_total
        return max(
            -MAX_TOTAL_POINTS_ADJUSTMENT,
            min(total_delta, MAX_TOTAL_POINTS_ADJUSTMENT),
        )

    def _dynamic_total_market_blend(
        self,
        base_blend: float,
        raw_total: float,
        market_total: float,
        game_type: str,
        inputs: Optional[dict] = None,
    ) -> float:
        blend = float(base_blend) + GAME_TYPE_TOTAL_MARKET_BLEND_BOOST.get(game_type, 0.0)
        market_gap = abs(float(raw_total) - float(market_total))

        if market_gap >= TOTAL_MARKET_GAP_EXTREME:
            blend = max(blend, TOTAL_MARKET_BLEND_EXTREME_GAP)
        elif market_gap >= TOTAL_MARKET_GAP_HIGH:
            blend = max(blend, TOTAL_MARKET_BLEND_HIGH_GAP)
        elif market_gap >= TOTAL_MARKET_GAP_MEDIUM:
            blend = max(blend, TOTAL_MARKET_BLEND_MEDIUM_GAP)

        if raw_total >= TOTAL_VERY_HIGH_THRESHOLD:
            blend = max(blend, min(TOTAL_MARKET_BLEND_CAP, blend + 0.05))
        elif raw_total >= TOTAL_HIGH_THRESHOLD:
            blend = max(blend, min(TOTAL_MARKET_BLEND_CAP, blend + 0.03))

        if inputs:
            venue_sample = float(inputs.get("venue", {}).get("sample_size", 0) or 0)
            if venue_sample < 5:
                blend += 0.03
            elif venue_sample < 15:
                blend += 0.01

            if game_type == GAME_TYPE_CONFERENCE:
                avg_rest = self._average_rest_days(inputs)
                if avg_rest <= 1.0:
                    blend += 0.04
                elif avg_rest <= 2.0:
                    blend += 0.02

        return max(0.0, min(TOTAL_MARKET_BLEND_CAP, blend))

    def _calibrate_total_baseline(
        self,
        inputs: dict,
        result: PredictionResult,
        game_type: str,
    ) -> float:
        if game_type == GAME_TYPE_NCAA:
            anchor = self.calibration_profile.round_total_anchor.get(
                result.tournament_round,
                self.calibration_profile.round_total_anchor.get(1, ROUND_TOTAL_ANCHOR[1]),
            )
            shrink = self.calibration_profile.total_baseline_shrink
            return anchor + ((result.raw_total - anchor) * shrink)

        if game_type == GAME_TYPE_CONFERENCE:
            avg_rest = self._average_rest_days(inputs)
            conf_pressure = self._conference_pressure_index(inputs)
            anchor = 140.0 - min(3.0, conf_pressure * 1.1)
            if avg_rest <= 1.0:
                anchor -= 1.0
            elif avg_rest <= 2.0:
                anchor -= 0.4
            shrink = self.calibration_profile.total_baseline_shrink * (
                GAME_TYPE_TOTAL_SHRINK_MULTIPLIER[GAME_TYPE_CONFERENCE]
            )
            shrink = max(0.35, min(0.80, shrink))
            return anchor + ((result.raw_total - anchor) * shrink)

        return result.raw_total

    def _dynamic_spread_market_blend(
        self,
        model_margin: float,
        market_spread: float,
        game_type: str,
        inputs: dict,
    ) -> float:
        blend = GAME_TYPE_SPREAD_MARKET_BASE.get(game_type, GAME_TYPE_SPREAD_MARKET_BASE[GAME_TYPE_GENERIC])
        model_spread = -float(model_margin)
        gap = abs(model_spread - float(market_spread))
        spread_zone = max(abs(model_spread), abs(float(market_spread)))

        if gap >= SPREAD_MARKET_GAP_EXTREME:
            blend = max(blend, SPREAD_MARKET_BLEND_EXTREME_GAP)
        elif gap >= SPREAD_MARKET_GAP_HIGH:
            blend = max(blend, SPREAD_MARKET_BLEND_HIGH_GAP)
        elif gap >= SPREAD_MARKET_GAP_MEDIUM:
            blend = max(blend, SPREAD_MARKET_BLEND_MEDIUM_GAP)

        if spread_zone <= PROJECTED_TOSSUP_THRESHOLD:
            blend += SPREAD_MARKET_BLEND_TOSSUP_BOOST
        elif spread_zone <= PROJECTED_SMALL_SPREAD_THRESHOLD:
            blend += SPREAD_MARKET_BLEND_SMALL_BOOST
        elif spread_zone >= PROJECTED_HUGE_FAVORITE_THRESHOLD:
            blend += SPREAD_MARKET_BLEND_HUGE_FAVORITE_BOOST
        elif spread_zone >= PROJECTED_LARGE_FAVORITE_THRESHOLD:
            blend += SPREAD_MARKET_BLEND_LARGE_FAVORITE_BOOST

        venue_sample = float(inputs.get("venue", {}).get("sample_size", 0) or 0)
        if venue_sample < 5:
            blend += 0.04
        elif venue_sample < 15:
            blend += 0.02

        if game_type == GAME_TYPE_CONFERENCE:
            avg_rest = self._average_rest_days(inputs)
            if avg_rest <= 1.0:
                blend += 0.03
            elif avg_rest <= 2.0:
                blend += 0.01

        if not inputs.get("home_efficiency") or not inputs.get("away_efficiency"):
            blend += 0.05

        return max(0.0, min(SPREAD_MARKET_BLEND_CAP, blend))

    def _market_spread_is_anomalous(
        self,
        model_margin: float,
        market_spread: float,
    ) -> bool:
        model_spread = -float(model_margin)
        market_spread = float(market_spread)

        if model_spread * market_spread >= 0.0:
            return False

        return (
            abs(model_spread) >= 14.0
            and abs(market_spread) <= 6.0
            and abs(model_spread - market_spread) >= 12.0
        )

    def _apply_spread_market_calibration(
        self,
        inputs: dict,
        result: PredictionResult,
        model_margin: float,
    ) -> tuple[float, float]:
        market_spread = inputs.get("market_lines", {}).get("consensus_spread")
        try:
            market_spread = float(market_spread)
            if not math.isfinite(market_spread):
                raise ValueError
        except (TypeError, ValueError):
            return model_margin, 0.0

        if self._market_spread_is_anomalous(model_margin, market_spread):
            logger.warning(
                "Ignoring anomalous market spread for %s @ %s: model=%+.1f market=%+.1f",
                result.away_team,
                result.home_team,
                -model_margin,
                market_spread,
            )
            return model_margin, 0.0

        market_margin = -market_spread
        blend = self._dynamic_spread_market_blend(
            model_margin=model_margin,
            market_spread=market_spread,
            game_type=result.game_type,
            inputs=inputs,
        )
        adjusted_margin = ((1.0 - blend) * model_margin) + (blend * market_margin)
        return adjusted_margin, adjusted_margin - model_margin

    def _apply_total_market_calibration(
        self,
        inputs: dict,
        result: PredictionResult,
        model_total: float,
    ) -> tuple[float, float]:
        market_total = inputs.get("market_lines", {}).get("consensus_total")
        try:
            market_total = float(market_total)
            if not math.isfinite(market_total):
                raise ValueError
        except (TypeError, ValueError):
            return model_total, 0.0

        market_blend = self._dynamic_total_market_blend(
            base_blend=self.calibration_profile.total_market_blend,
            raw_total=result.raw_total,
            market_total=market_total,
            game_type=result.game_type,
            inputs=inputs,
        )
        adjusted_total = ((1.0 - market_blend) * model_total) + (market_blend * market_total)
        return adjusted_total, adjusted_total - model_total

    @staticmethod
    def _average_rest_days(inputs: dict) -> float:
        values = []
        for side in ("home_rest", "away_rest"):
            rest_days = inputs.get(side, {}).get("rest_days")
            if rest_days is None:
                continue
            try:
                values.append(float(rest_days))
            except (TypeError, ValueError):
                continue
        if not values:
            return 2.0
        return sum(values) / len(values)

    @staticmethod
    def _conference_pressure_index(inputs: dict) -> float:
        total_wins = 0.0
        for side in ("home_extended", "away_extended"):
            extended = inputs.get(side, {}) or {}
            total_wins += float(extended.get("conf_tourney_wins", 0) or 0)
        return max(0.0, total_wins / 2.0)

    def _blowout_total_uplift(
        self,
        inputs: dict,
        result: PredictionResult,
    ) -> float:
        projected_margin = abs(float(result.raw_margin + result.total_adjustment))
        if projected_margin < PROJECTED_HUGE_FAVORITE_THRESHOLD:
            return 0.0

        uplift = BLOWOUT_TOTAL_UPLIFT_BASE + (
            (projected_margin - PROJECTED_HUGE_FAVORITE_THRESHOLD)
            * BLOWOUT_TOTAL_UPLIFT_PER_POINT
        )

        market_total = inputs.get("market_lines", {}).get("consensus_total")
        try:
            market_total = float(market_total)
            if math.isfinite(market_total) and market_total > float(result.raw_total):
                uplift += min(
                    1.0,
                    (market_total - float(result.raw_total)) * BLOWOUT_TOTAL_UPLIFT_MARKET_FACTOR,
                )
        except (TypeError, ValueError):
            pass

        avg_tempo = (
            float(inputs.get("home_efficiency", {}).get("adj_tempo", 68.0) or 68.0)
            + float(inputs.get("away_efficiency", {}).get("adj_tempo", 68.0) or 68.0)
        ) / 2.0
        if avg_tempo >= 70.0:
            uplift += 0.4
        elif avg_tempo <= 66.0:
            uplift -= 0.2

        return max(0.0, min(BLOWOUT_TOTAL_UPLIFT_CAP, uplift))

    def _compute_matchup_interaction(
        self,
        offense: dict,
        defense: dict,
        game_type: str,
    ) -> float:
        three_rate = float(offense.get("three_pt_rate_off", 0.35) or 0.35)
        defense_three_pct = float(defense.get("three_pt_pct_def", 0.34) or 0.34)
        two_point_off = float(offense.get("two_pt_pct_off", 0.48) or 0.48)
        two_point_def = float(defense.get("two_pt_pct_def", 0.48) or 0.48)
        turnover_off = float(offense.get("to_rate_off", 18.0) or 18.0)
        turnover_def = float(defense.get("to_rate_def", 18.0) or 18.0)
        orb_off = float(offense.get("orb_pct", 0.30) or 0.30)
        drb_def = float(defense.get("drb_pct", 0.70) or 0.70)
        fta_off = float(offense.get("fta_rate_off", 0.30) or 0.30)
        fta_def = float(defense.get("fta_rate_def", 0.30) or 0.30)

        three_point_interaction = (
            ((three_rate - 0.35) / 0.08)
            * ((defense_three_pct - 0.34) / 0.04)
            * MATCHUP_THREE_POINT_WEIGHT
        )
        two_point_interaction = (
            ((two_point_off - 0.48) / 0.06)
            * ((two_point_def - 0.48) / 0.06)
            * MATCHUP_TWO_POINT_WEIGHT
        )
        turnover_interaction = (
            -((turnover_off - 18.0) / 4.0)
            * ((turnover_def - 18.0) / 4.0)
            * MATCHUP_TURNOVER_WEIGHT
        )
        rebound_interaction = (
            ((orb_off - 0.30) / 0.06)
            * ((0.70 - drb_def) / 0.06)
            * MATCHUP_REBOUND_WEIGHT
        )
        foul_interaction = (
            ((fta_off - 0.30) / 0.07)
            * ((fta_def - 0.30) / 0.07)
            * MATCHUP_FOUL_WEIGHT
        )

        interaction = (
            three_point_interaction
            + two_point_interaction
            + turnover_interaction
            + rebound_interaction
            + foul_interaction
        )
        if game_type == GAME_TYPE_CONFERENCE:
            interaction *= 1.05
        return max(-MATCHUP_INTERACTION_CAP, min(MATCHUP_INTERACTION_CAP, interaction))

    # ══════════════════════════════════════════════════════════════════════
    #  LAYER 4 — Additive Adjustments
    # ══════════════════════════════════════════════════════════════════════

    def _compute_momentum_adjustment(
        self,
        home_momentum: dict,
        away_momentum: dict,
    ) -> float:
        """
        Momentum adjustment based on recent performance vs season average.

        Uses last 10 game margins with exponential decay.
        Compares recent efficiency to season average.

        Output: signed adjustment in points (positive = favors home).
        Range: roughly ±2.0 points.
        """
        home_score = self._score_momentum(home_momentum)
        away_score = self._score_momentum(away_momentum)
        return (home_score - away_score) * 1.0  # 1 pt per unit delta

    def _score_momentum(self, momentum: dict) -> float:
        """
        Score a team's momentum from -2 to +2.

        Factors:
          - Decay-weighted recent margins
          - Comparison to season average efficiency
          - Conference tournament performance
        """
        margins = momentum.get("recent_margins", [])
        if not margins:
            return 0.0

        # Decay-weighted average margin (most recent = highest weight)
        total_weight = 0.0
        weighted_sum = 0.0
        for i, margin in enumerate(margins):
            # margins[0] = most recent, margins[-1] = oldest
            weight = math.exp(-MOMENTUM_DECAY_LAMBDA * i)
            weighted_sum += float(margin) * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        weighted_avg_margin = weighted_sum / total_weight

        # Normalize to ±2 scale
        # A team winning by 15+ on average is ~+2, losing by 15+ is ~-2
        momentum_score = max(-2.0, min(2.0, weighted_avg_margin / 10.0))

        # Bonus for efficiency delta vs season average
        season_oe = momentum.get("season_adj_oe", 100)
        season_de = momentum.get("season_adj_de", 100)
        if season_oe > 0 and season_de > 0:
            # If they're trending up (margins better than efficiency suggests),
            # add a small bonus
            expected_margin = (season_oe - season_de) / 5.0
            actual_vs_expected = weighted_avg_margin - expected_margin
            efficiency_bonus = max(-0.5, min(0.5, actual_vs_expected / 10.0))
            momentum_score += efficiency_bonus

        return max(-2.0, min(2.0, momentum_score))

    def _compute_experience_adjustment(
        self,
        home_exp: dict,
        away_exp: dict,
    ) -> float:
        """
        Experience adjustment based on roster age, coach record,
        and roster continuity.

        Output: signed adjustment in points (positive = favors home).
        Range: roughly ±1.5 points.
        """
        home_score = self._score_experience(home_exp)
        away_score = self._score_experience(away_exp)
        return (home_score - away_score) * 0.75  # scale to points

    def _score_experience(self, exp: dict) -> float:
        """
        Score a team's tournament experience from -2 to +2.

        Components (when coach data available):
          1. Roster age (class year average) — 30% weight
          2. Coach tournament record — 35% weight
          3. Returning production % — 35% weight

        Fallback (when coach data unavailable):
          1. Roster age — 47.5% weight
          2. Returning production — 52.5% weight
        """
        score = 0.0
        import pandas as pd

        coach = exp.get("coach_record")
        has_coach_data = coach is not None and coach.get("appearances", 0) > 0

        # Determine weights based on data availability
        if has_coach_data:
            w_roster = EXP_WEIGHT_ROSTER_AGE    # 0.30
            w_coach = EXP_WEIGHT_COACH           # 0.35
            w_returning = EXP_WEIGHT_RETURNING   # 0.35
        else:
            # Redistribute coach weight to roster + returning
            w_roster = 0.475
            w_coach = 0.0
            w_returning = 0.525

        # 1. Roster age score
        roster = exp.get("roster")
        if isinstance(roster, pd.DataFrame) and not roster.empty:
            if "class_year_num" in roster.columns:
                avg_year = roster["class_year_num"].dropna().mean()
                if avg_year:
                    roster_score = (avg_year - 2.5) / 1.5
                    score += w_roster * max(-2, min(2, roster_score))

        # 2. Coach tournament record (only if available)
        if has_coach_data:
            appearances = coach.get("appearances", 0)
            win_rate = coach.get("win_rate", 0.0)
            app_factor = min(1.0, appearances / 10.0)
            rate_factor = (win_rate - 0.45) / 0.25
            coach_score = app_factor * rate_factor
            score += w_coach * max(-2, min(2, coach_score))
        elif coach and coach.get("first_yr_coach", False):
            # First-year coach from hardcoded data — small penalty
            score += 0.35 * (-0.8)

        # 3. Returning production
        returning_pct = exp.get("returning_pct", 0.5)
        returning_score = (returning_pct - 0.5) / 0.2
        score += w_returning * max(-2, min(2, returning_score))

        return max(-2.0, min(2.0, score))

    def _compute_rest_adjustment(
        self,
        home_rest: dict,
        away_rest: dict,
    ) -> float:
        """
        Rest adjustment based on days since last game.
        Output: signed adjustment (positive = favors home).
        """
        home_days = home_rest.get("rest_days")
        away_days = away_rest.get("rest_days")

        home_adj = self._rest_value(home_days)
        away_adj = self._rest_value(away_days)

        return home_adj - away_adj

    @staticmethod
    def _rest_value(days: Optional[int]) -> float:
        """Convert rest days to a point value."""
        if days is None:
            return 0.0
        days = int(days)
        if days in REST_ADJUSTMENTS:
            return REST_ADJUSTMENTS[days]
        if days > 7:
            # Rust factor — too much rest is slightly negative
            return max(-0.3, 0.1 - (days - 7) * 0.1)
        return 0.0

    def _compute_injury_adjustment(
        self,
        injuries,
        home_team: str,
        away_team: str,
    ) -> float:
        """
        Injury adjustment based on player availability.

        Without player usage data, we estimate impact by count:
          - Each injured starter ≈ -1.0 to -2.0 points
          - Each injured bench player ≈ -0.2 to -0.5 points
          - "Out" vs "Day-to-Day" weighted differently

        Output: signed adjustment (positive = favors home).
        """
        import pandas as pd

        if not isinstance(injuries, pd.DataFrame) or injuries.empty:
            return 0.0

        home_impact = self._team_injury_impact(injuries, home_team)
        away_impact = self._team_injury_impact(injuries, away_team)

        # Both are negative; diff favors the healthier team
        return home_impact - away_impact

    @staticmethod
    def _team_injury_impact(injuries, team: str) -> float:
        """
        Calculate injury impact for one team.
        Returns negative value (injuries always hurt).
        """
        import pandas as pd

        if injuries.empty or "team" not in injuries.columns:
            return 0.0

        team_lower = team.lower()
        team_injuries = injuries[
            injuries["team"].str.lower().str.contains(team_lower, na=False)
        ]

        if team_injuries.empty:
            return 0.0

        impact = 0.0
        for _, inj in team_injuries.iterrows():
            status = str(inj.get("status", "")).lower()
            position = str(inj.get("position", "")).lower()

            # Status-based weight
            if "out" in status:
                severity = 1.0
            elif "doubtful" in status:
                severity = 0.7
            elif "questionable" in status:
                severity = 0.3
            elif "day-to-day" in status or "dtd" in status:
                severity = 0.2
            else:
                severity = 0.4

            # Position-based impact (guards > forwards for most teams)
            if position in ("pg", "sg", "g"):
                pos_weight = 0.9
            elif position in ("sf", "pf", "f"):
                pos_weight = 0.7
            elif position in ("c",):
                pos_weight = 0.6
            else:
                pos_weight = 0.5

            impact -= severity * pos_weight * 1.2  # scale to ~1-2 pts per starter

        # Cap total injury impact
        return max(-4.0, impact)

    def _compute_seed_adjustment(self, seed_context: dict) -> float:
        """
        Seed history adjustment from pre-computed seed context.
        Already signed for home perspective.
        """
        return seed_context.get("seed_adjustment", 0.0)

    def _compute_travel_adjustment(
        self,
        home_travel: dict,
        away_travel: dict,
    ) -> float:
        """
        Travel + altitude adjustment.
        Teams traveling far or to high altitude get a penalty.
        Output: signed adjustment (positive = favors home).
        """
        home_penalty = self._travel_penalty(home_travel)
        away_penalty = self._travel_penalty(away_travel)
        return home_penalty - away_penalty

    @staticmethod
    def _travel_penalty(travel: dict) -> float:
        """
        Calculate travel penalty for one team.
        Returns negative value (travel always hurts).
        """
        penalty = 0.0
        distance = travel.get("travel_distance_miles", 0)
        alt_diff = travel.get("altitude_diff_ft", 0)

        # Distance penalty
        if distance > TRAVEL_PENALTY_THRESHOLD:
            excess = distance - TRAVEL_PENALTY_THRESHOLD
            penalty -= (excess / 1000.0) * TRAVEL_PENALTY_PER_1000

        # Altitude penalty (only for going UP significantly)
        if alt_diff > ALTITUDE_PENALTY_THRESHOLD:
            excess = alt_diff - ALTITUDE_PENALTY_THRESHOLD
            penalty -= (excess / 1000.0) * ALTITUDE_PENALTY_PER_1000

        return max(-1.5, penalty)

    # ══════════════════════════════════════════════════════════════════════
    #  Win Probability & Confidence
    # ══════════════════════════════════════════════════════════════════════

    def _margin_to_win_prob(self, margin: float) -> float:
        """
        Convert projected margin to win probability using logistic function.

        P(home wins) = 1 / (1 + exp(-(intercept + slope * margin)))

        Defaults reduce to the old fixed-sigma mapping:
          margin  0 → 50.0%
          margin  3 → 57.1%
          margin  5 → 62.0%
          margin  7 → 66.1%
          margin 10 → 72.1%
          margin 15 → 80.7%
        """
        intercept = self.calibration_profile.probability_intercept
        slope = self.calibration_profile.probability_slope
        return 1.0 / (1.0 + math.exp(-(intercept + slope * margin)))

    def _estimate_margin_uncertainty(self, inputs: dict) -> float:
        base = max(2.5, float(self.calibration_profile.margin_mae))
        multiplier = 1.0
        game_type = infer_game_type_from_inputs(inputs)

        if not inputs.get("home_efficiency"):
            multiplier += 0.12
        if not inputs.get("away_efficiency"):
            multiplier += 0.12

        home_margins = inputs.get("home_momentum", {}).get("recent_margins", [])
        away_margins = inputs.get("away_momentum", {}).get("recent_margins", [])
        if len(home_margins) < 5:
            multiplier += 0.05
        if len(away_margins) < 5:
            multiplier += 0.05

        home_std = float(inputs.get("home_extended", {}).get("margin_std", 12.0) or 12.0)
        away_std = float(inputs.get("away_extended", {}).get("margin_std", 12.0) or 12.0)
        avg_std = max(4.0, (home_std + away_std) / 2.0)
        volatility_boost = max(0.0, min(0.20, (avg_std - 12.0) / 30.0))
        multiplier += volatility_boost

        if game_type == GAME_TYPE_CONFERENCE:
            multiplier += 0.04
        elif game_type == GAME_TYPE_GENERIC:
            multiplier += 0.02

        return round(base * multiplier, 2)

    def _estimate_total_uncertainty(
        self,
        inputs: dict,
        result: Optional[PredictionResult] = None,
    ) -> float:
        base = max(3.5, float(self.calibration_profile.total_mae))
        multiplier = 1.0
        game_type = infer_game_type_from_inputs(inputs)

        venue_sample = float(inputs.get("venue", {}).get("sample_size", 0) or 0)
        if venue_sample < 5:
            multiplier += 0.06
        elif venue_sample < 15:
            multiplier += 0.03

        market_total = inputs.get("market_lines", {}).get("consensus_total")
        if market_total is None:
            multiplier += 0.05
        else:
            try:
                market_total = float(market_total)
                if math.isfinite(market_total) and result is not None:
                    raw_gap = abs(float(result.raw_total) - market_total)
                    if raw_gap >= TOTAL_MARKET_GAP_EXTREME:
                        multiplier += 0.10
                    elif raw_gap >= TOTAL_MARKET_GAP_HIGH:
                        multiplier += 0.06
                    elif raw_gap >= TOTAL_MARKET_GAP_MEDIUM:
                        multiplier += 0.03
            except (TypeError, ValueError):
                multiplier += 0.02

        home_tempo = float(inputs.get("home_efficiency", {}).get("adj_tempo", 68.0) or 68.0)
        away_tempo = float(inputs.get("away_efficiency", {}).get("adj_tempo", 68.0) or 68.0)
        tempo_gap = abs(home_tempo - away_tempo)
        multiplier += min(0.10, tempo_gap / 80.0)

        if result is not None:
            projected_total = float(result.raw_total)
            if projected_total >= TOTAL_VERY_HIGH_THRESHOLD:
                multiplier += 0.08
            elif projected_total >= TOTAL_HIGH_THRESHOLD:
                multiplier += 0.04
            projected_margin = abs(float(result.raw_margin + result.total_adjustment))
            if projected_margin >= PROJECTED_HUGE_FAVORITE_THRESHOLD:
                multiplier += 0.06
            elif projected_margin >= PROJECTED_LARGE_FAVORITE_THRESHOLD:
                multiplier += 0.03

        if game_type == GAME_TYPE_CONFERENCE:
            multiplier += 0.05
        elif game_type == GAME_TYPE_GENERIC:
            multiplier += 0.02

        return round(base * multiplier, 2)

    def _compute_confidence(
        self,
        inputs: dict,
        result: PredictionResult,
    ) -> float:
        """
        Confidence score from 0.0 to 1.0 based on:
          - Data completeness (did we get efficiency data?)
          - Agreement between formula and market
          - Sample size of venue data
          - Margin magnitude (close games = less confident)
        """
        score = 0.5  # baseline

        # Data completeness
        if inputs.get("home_efficiency"):
            score += 0.10
        if inputs.get("away_efficiency"):
            score += 0.10

        # Venue sample size
        venue_n = inputs.get("venue", {}).get("sample_size", 0)
        if venue_n > 20:
            score += 0.05
        elif venue_n > 5:
            score += 0.02

        # Margin signal-to-noise — larger edge relative to expected error is better
        abs_margin = abs(result.raw_margin + result.total_adjustment)
        margin_noise = max(1.0, result.margin_uncertainty)
        margin_signal = abs_margin / margin_noise
        if margin_signal > 1.5:
            score += 0.12
        elif margin_signal > 1.0:
            score += 0.06
        elif margin_signal < 0.5:
            score -= 0.10

        # Market agreement (if available)
        if result.spread_edge is not None:
            if abs(result.spread_edge) < max(1.5, margin_noise * 0.35):
                score += 0.10  # we agree with Vegas — good sign
            elif abs(result.spread_edge) > max(5.0, margin_noise * 0.8):
                score -= 0.10  # big disagreement — one of us is wrong
        if abs(result.spread_market_adjustment) > 1.0:
            score -= 0.03

        # Momentum data availability
        home_margins = inputs.get("home_momentum", {}).get("recent_margins", [])
        away_margins = inputs.get("away_momentum", {}).get("recent_margins", [])
        if len(home_margins) >= 5 and len(away_margins) >= 5:
            score += 0.05

        # Totals confidence helps overall confidence on scoring outputs.
        if result.total_uncertainty <= self.calibration_profile.total_mae:
            score += 0.03
        elif result.total_uncertainty >= self.calibration_profile.total_mae * 1.15:
            score -= 0.03
        if abs(result.total_market_adjustment) > 2.0:
            score -= 0.03

        return max(0.10, min(0.95, score))

    @staticmethod
    def _confidence_label(score: float) -> str:
        if score >= 0.80:
            return "VERY HIGH"
        if score >= 0.65:
            return "HIGH"
        if score >= 0.45:
            return "MEDIUM"
        if score >= 0.30:
            return "LOW"
        return "VERY LOW"

    @staticmethod
    def _default_efficiency() -> dict:
        """Return average D1 efficiency profile as fallback."""
        return {
            "adj_oe": 100.0, "adj_de": 100.0, "adj_tempo": 68.0,
            "efg_pct_off": 0.50, "efg_pct_def": 0.50,
            "to_rate_off": 18.0, "to_rate_def": 18.0,
            "three_pt_rate_off": 0.35, "three_pt_rate_def": 0.35,
            "three_pt_pct_off": 0.33, "three_pt_pct_def": 0.33,
            "fta_rate_off": 0.30, "fta_rate_def": 0.30,
            "orb_pct": 0.30, "drb_pct": 0.70,
            "sos": 0.0, "rank": 175, "barthag": 0.50,
        }

    # ── Batch Prediction ──────────────────────────────────────────────────

    def predict_batch(
        self,
        games: list[dict],
    ) -> list[PredictionResult]:
        """Predict multiple games. Each dict is a pipeline output."""
        return [self.predict(g) for g in games]


# ── Convenience function ──────────────────────────────────────────────────────

def predict_game(inputs: dict) -> PredictionResult:
    """Module-level convenience function."""
    engine = PredictionEngine()
    return engine.predict(inputs)


# ── Quick test with synthetic data ────────────────────────────────────────────
if __name__ == "__main__":
    engine = PredictionEngine()

    # Simulate a Duke (1-seed) vs Vermont (16-seed) first round game
    synthetic_inputs = {
        "home_team": "Duke",
        "away_team": "Vermont",
        "season": 2025,
        "tournament_round": 1,
        "home_seed": 1,
        "away_seed": 16,
        "home_efficiency": {
            "adj_oe": 120.5, "adj_de": 92.3, "adj_tempo": 71.2,
            "efg_pct_off": 0.56, "efg_pct_def": 0.44,
            "to_rate_off": 15.2, "to_rate_def": 20.1,
            "three_pt_rate_off": 0.38, "three_pt_rate_def": 0.32,
            "three_pt_pct_off": 0.37, "three_pt_pct_def": 0.30,
            "fta_rate_off": 0.35, "fta_rate_def": 0.25,
            "orb_pct": 0.33, "drb_pct": 0.75,
            "sos": 8.5, "rank": 5, "barthag": 0.95,
        },
        "away_efficiency": {
            "adj_oe": 105.2, "adj_de": 103.8, "adj_tempo": 66.5,
            "efg_pct_off": 0.51, "efg_pct_def": 0.49,
            "to_rate_off": 17.8, "to_rate_def": 18.2,
            "three_pt_rate_off": 0.40, "three_pt_rate_def": 0.36,
            "three_pt_pct_off": 0.34, "three_pt_pct_def": 0.33,
            "fta_rate_off": 0.28, "fta_rate_def": 0.30,
            "orb_pct": 0.28, "drb_pct": 0.68,
            "sos": -2.1, "rank": 85, "barthag": 0.72,
        },
        "venue": {
            "vsi": 1.02, "vpi": 0.99, "v3p": 0.97,
            "sample_size": 45,
        },
        "home_momentum": {
            "recent_margins": [15, 8, 22, -3, 12, 18, 5, 10, 7, 20],
            "season_adj_oe": 120.5, "season_adj_de": 92.3,
            "season_tempo": 71.2, "season_efg": 0.56, "season_to_rate": 15.2,
        },
        "away_momentum": {
            "recent_margins": [5, 12, -2, 8, 3, 15, 7, -5, 10, 6],
            "season_adj_oe": 105.2, "season_adj_de": 103.8,
            "season_tempo": 66.5, "season_efg": 0.51, "season_to_rate": 17.8,
        },
        "home_experience": {
            "roster": None,
            "coach_record": {
                "appearances": 12, "total_wins": 25, "total_losses": 10,
                "win_rate": 0.714, "first_yr_coach": False,
            },
            "returning_pct": 0.65,
        },
        "away_experience": {
            "roster": None,
            "coach_record": {
                "appearances": 3, "total_wins": 2, "total_losses": 3,
                "win_rate": 0.40, "first_yr_coach": False,
            },
            "returning_pct": 0.55,
        },
        "home_rest": {"rest_days": 5},
        "away_rest": {"rest_days": 5},
        "injuries": None,
        "home_travel": {
            "travel_distance_miles": 300, "altitude_diff_ft": 200,
        },
        "away_travel": {
            "travel_distance_miles": 800, "altitude_diff_ft": 500,
        },
        "seed_context": {
            "seed_adjustment": 1.85,  # favors 1-seed (home)
        },
        "market_lines": {
            "consensus_spread": -14.5,
            "consensus_total": 142.0,
        },
    }

    result = engine.predict(synthetic_inputs)
    print(result.breakdown())

    # Also test a close game: 5 vs 12
    print("\n\n")
    close_game = dict(synthetic_inputs)
    close_game["home_team"] = "Marquette"
    close_game["away_team"] = "McNeese"
    close_game["home_seed"] = 5
    close_game["away_seed"] = 12
    close_game["tournament_round"] = 1
    close_game["home_efficiency"]["adj_oe"] = 112.0
    close_game["home_efficiency"]["adj_de"] = 97.0
    close_game["home_efficiency"]["rank"] = 20
    close_game["home_efficiency"]["barthag"] = 0.88
    close_game["away_efficiency"]["adj_oe"] = 108.5
    close_game["away_efficiency"]["adj_de"] = 99.5
    close_game["away_efficiency"]["rank"] = 55
    close_game["away_efficiency"]["barthag"] = 0.80
    close_game["seed_context"] = {"seed_adjustment": 1.01}
    close_game["market_lines"] = {
        "consensus_spread": -5.5, "consensus_total": 145.0,
    }

    result2 = engine.predict(close_game)
    print(result2.breakdown())
