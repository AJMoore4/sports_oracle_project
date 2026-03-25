"""
sports_oracle/utils/data_validator.py

Validates and sanitizes all data flowing into the prediction engine.

PROBLEM:
  API endpoints return garbage sometimes — adj_oe of 0, pace of 999,
  null values masquerading as floats. If bad data reaches the formula
  layer, you get division-by-zero or wildly wrong predictions with
  no visible error.

APPROACH:
  Define reasonable bounds for every metric based on D1 basketball.
  Clamp out-of-range values. Log warnings for anything suspicious.
  Return a validation report alongside cleaned data.

USAGE:
    validator = DataValidator()
    clean_eff, report = validator.validate_efficiency(raw_efficiency_dict)
    clean_venue, report = validator.validate_venue(raw_venue_dict)
"""

from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("sports_oracle.validator")


# ── Reasonable bounds for D1 basketball metrics ───────────────────────────────
# Based on historical BartTorvik data (2010-2025).
# [min, default, max] — values outside min/max get clamped to default.

BOUNDS = {
    # Layer 1 — Efficiency
    "adj_oe":           (70.0,  100.0, 135.0),
    "adj_de":           (70.0,  100.0, 135.0),
    "adj_tempo":        (55.0,  68.0,  80.0),
    "efg_pct_off":      (0.30,  0.50,  0.65),
    "efg_pct_def":      (0.30,  0.50,  0.65),
    "to_rate_off":      (8.0,   18.0,  30.0),
    "to_rate_def":      (8.0,   18.0,  30.0),
    "three_pt_rate_off":(0.15,  0.35,  0.55),
    "three_pt_rate_def":(0.15,  0.35,  0.55),
    "three_pt_pct_off": (0.20,  0.33,  0.45),
    "three_pt_pct_def": (0.20,  0.33,  0.45),
    "fta_rate_off":     (0.10,  0.30,  0.55),
    "fta_rate_def":     (0.10,  0.30,  0.55),
    "orb_pct":          (0.15,  0.30,  0.45),
    "drb_pct":          (0.55,  0.70,  0.85),
    "sos":              (-15.0, 0.0,   15.0),
    "rank":             (1,     150,   365),
    "barthag":          (0.01,  0.50,  0.99),

    # Layer 2 — Venue
    "vsi":              (0.80,  1.00,  1.20),
    "vpi":              (0.85,  1.00,  1.15),
    "v3p":              (0.85,  1.00,  1.15),

    # Layer 4 — Contextual
    "rest_days":        (0,     2,     14),
    "returning_pct":    (0.0,   0.50,  1.0),
    "coach_win_rate":   (0.0,   0.50,  1.0),
    "momentum_score":   (-3.0,  0.0,   3.0),
    "experience_score": (-2.0,  0.0,   2.0),
    "injury_impact":    (-5.0,  0.0,   0.0),

    # Matchup
    "travel_distance":  (0,     0,     3000),
    "altitude_diff":    (-5000, 0,     5000),
}


@dataclass
class ValidationReport:
    """Tracks what was cleaned and why."""
    field_name: str = ""
    warnings: list[str] = field(default_factory=list)
    nulls_filled: int = 0
    values_clamped: int = 0
    is_valid: bool = True

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        logger.warning(msg)

    def __repr__(self):
        if not self.warnings:
            return f"ValidationReport(valid=True, clean)"
        return (
            f"ValidationReport(warnings={len(self.warnings)}, "
            f"nulls_filled={self.nulls_filled}, "
            f"clamped={self.values_clamped})"
        )


class DataValidator:
    """
    Validates and clamps all prediction engine inputs.
    Returns cleaned data + validation report.
    """

    def __init__(self, strict: bool = False):
        """
        strict=True: raise ValueError on out-of-range data
        strict=False: clamp to bounds and warn (default, for production)
        """
        self.strict = strict

    # ── Core validation ───────────────────────────────────────────────────

    def validate_value(
        self,
        value: Any,
        field_name: str,
        context: str = "",
    ) -> tuple[float, ValidationReport]:
        """
        Validate a single numeric value against known bounds.
        Returns (cleaned_value, report).
        """
        report = ValidationReport(field_name=field_name)
        bounds = BOUNDS.get(field_name)

        if bounds is None:
            # No bounds defined — pass through
            return value, report

        min_val, default, max_val = bounds

        # Handle None / missing
        if value is None:
            report.nulls_filled += 1
            report.add_warning(
                f"{context}{field_name}: null → using default {default}"
            )
            return default, report

        # Cast to float
        try:
            value = float(value)
        except (TypeError, ValueError):
            report.nulls_filled += 1
            report.add_warning(
                f"{context}{field_name}: non-numeric '{value}' → default {default}"
            )
            return default, report

        if not math.isfinite(value):
            report.nulls_filled += 1
            report.add_warning(
                f"{context}{field_name}: non-finite '{value}' → default {default}"
            )
            return default, report

        # Check bounds
        if value < min_val or value > max_val:
            if self.strict:
                raise ValueError(
                    f"{context}{field_name}={value} outside bounds "
                    f"[{min_val}, {max_val}]"
                )
            clamped = max(min_val, min(value, max_val))
            report.values_clamped += 1
            report.add_warning(
                f"{context}{field_name}={value:.2f} outside [{min_val}, {max_val}] "
                f"→ clamped to {clamped:.2f}"
            )
            return clamped, report

        return value, report

    # ── Dict-level validation ─────────────────────────────────────────────

    def validate_efficiency(
        self,
        eff: dict,
        team_label: str = "",
    ) -> tuple[dict, ValidationReport]:
        """Validate a team efficiency profile dict from pipeline."""
        context = f"[{team_label}] " if team_label else ""
        report = ValidationReport(field_name=f"{context}efficiency")
        cleaned = dict(eff)  # shallow copy

        eff_fields = [
            "adj_oe", "adj_de", "adj_tempo", "barthag",
            "efg_pct_off", "efg_pct_def",
            "to_rate_off", "to_rate_def",
            "three_pt_rate_off", "three_pt_rate_def",
            "three_pt_pct_off", "three_pt_pct_def",
            "fta_rate_off", "fta_rate_def",
            "orb_pct", "drb_pct", "sos", "rank",
        ]

        for f in eff_fields:
            val, sub_report = self.validate_value(
                cleaned.get(f), f, context
            )
            cleaned[f] = val
            report.warnings.extend(sub_report.warnings)
            report.nulls_filled += sub_report.nulls_filled
            report.values_clamped += sub_report.values_clamped

        if not eff:
            report.is_valid = False
            report.add_warning(f"{context}Empty efficiency profile")

        return cleaned, report

    def validate_venue(self, venue: dict) -> tuple[dict, ValidationReport]:
        """Validate a venue profile dict from pipeline."""
        report = ValidationReport(field_name="venue")
        cleaned = dict(venue)

        for f in ["vsi", "vpi", "v3p"]:
            val, sub_report = self.validate_value(
                cleaned.get(f), f, "[venue] "
            )
            cleaned[f] = val
            report.warnings.extend(sub_report.warnings)
            report.nulls_filled += sub_report.nulls_filled
            report.values_clamped += sub_report.values_clamped

        return cleaned, report

    def validate_rest(self, rest: dict, team_label: str = "") -> tuple[dict, ValidationReport]:
        """Validate rest data dict."""
        context = f"[{team_label}] " if team_label else ""
        report = ValidationReport(field_name=f"{context}rest")
        cleaned = dict(rest)

        val, sub_report = self.validate_value(
            cleaned.get("rest_days"), "rest_days", context
        )
        cleaned["rest_days"] = val
        report.warnings.extend(sub_report.warnings)
        report.nulls_filled += sub_report.nulls_filled
        report.values_clamped += sub_report.values_clamped

        return cleaned, report

    def validate_experience(
        self,
        exp: dict,
        team_label: str = "",
    ) -> tuple[dict, ValidationReport]:
        """Validate experience inputs used in Layer 4."""
        context = f"[{team_label}] " if team_label else ""
        report = ValidationReport(field_name=f"{context}experience")
        cleaned = dict(exp)

        val, sub_report = self.validate_value(
            cleaned.get("returning_pct"), "returning_pct", context
        )
        cleaned["returning_pct"] = val
        report.warnings.extend(sub_report.warnings)
        report.nulls_filled += sub_report.nulls_filled
        report.values_clamped += sub_report.values_clamped

        return cleaned, report

    def validate_momentum(
        self,
        momentum: dict,
        team_label: str = "",
    ) -> tuple[dict, ValidationReport]:
        """Validate momentum inputs used in Layer 4."""
        context = f"[{team_label}] " if team_label else ""
        report = ValidationReport(field_name=f"{context}momentum")
        cleaned = dict(momentum)

        recent_margins = []
        for raw in cleaned.get("recent_margins", []) or []:
            try:
                margin = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(margin):
                recent_margins.append(margin)
        cleaned["recent_margins"] = recent_margins

        for field_name in ("season_adj_oe", "season_adj_de"):
            base_field = field_name.replace("season_", "")
            val, sub_report = self.validate_value(
                cleaned.get(field_name),
                base_field,
                context,
            )
            cleaned[field_name] = val
            report.warnings.extend(sub_report.warnings)
            report.nulls_filled += sub_report.nulls_filled
            report.values_clamped += sub_report.values_clamped

        return cleaned, report

    def validate_game_inputs(self, inputs: dict) -> tuple[dict, list[ValidationReport]]:
        """
        Validate the full game_inputs dict from pipeline.get_game_inputs().
        Returns (cleaned_inputs, list_of_reports).
        """
        reports = []
        cleaned = dict(inputs)

        home = inputs.get("home_team", "Home")
        away = inputs.get("away_team", "Away")

        # Efficiency profiles
        if inputs.get("home_efficiency"):
            cleaned["home_efficiency"], r = self.validate_efficiency(
                inputs["home_efficiency"], home
            )
            reports.append(r)

        if inputs.get("away_efficiency"):
            cleaned["away_efficiency"], r = self.validate_efficiency(
                inputs["away_efficiency"], away
            )
            reports.append(r)

        # Venue
        if inputs.get("venue"):
            cleaned["venue"], r = self.validate_venue(inputs["venue"])
            reports.append(r)

        # Rest
        if inputs.get("home_rest"):
            cleaned["home_rest"], r = self.validate_rest(
                inputs["home_rest"], home
            )
            reports.append(r)
        if inputs.get("away_rest"):
            cleaned["away_rest"], r = self.validate_rest(
                inputs["away_rest"], away
            )
            reports.append(r)

        # Momentum
        if inputs.get("home_momentum"):
            cleaned["home_momentum"], r = self.validate_momentum(
                inputs["home_momentum"], home
            )
            reports.append(r)
        if inputs.get("away_momentum"):
            cleaned["away_momentum"], r = self.validate_momentum(
                inputs["away_momentum"], away
            )
            reports.append(r)

        # Experience
        if inputs.get("home_experience"):
            cleaned["home_experience"], r = self.validate_experience(
                inputs["home_experience"], home
            )
            reports.append(r)
        if inputs.get("away_experience"):
            cleaned["away_experience"], r = self.validate_experience(
                inputs["away_experience"], away
            )
            reports.append(r)

        # Tournament round
        tr = inputs.get("tournament_round", 1)
        if tr not in range(0, 7):
            logger.warning(
                f"Invalid tournament_round={tr}, defaulting to 1"
            )
            cleaned["tournament_round"] = 1

        total_warnings = sum(len(r.warnings) for r in reports)
        if total_warnings > 0:
            logger.info(
                f"Validation complete: {total_warnings} warnings across "
                f"{len(reports)} checks"
            )
        else:
            logger.debug("Validation complete: all inputs clean")

        return cleaned, reports


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    v = DataValidator()

    print("\n🔍 DataValidator — Sanity Bound Tests")
    print("=" * 50)

    # Good data
    val, r = v.validate_value(105.3, "adj_oe")
    print(f"  adj_oe=105.3 → {val} ({r})")

    # Null
    val, r = v.validate_value(None, "adj_oe")
    print(f"  adj_oe=None  → {val} ({r})")

    # Out of range
    val, r = v.validate_value(999, "adj_oe")
    print(f"  adj_oe=999   → {val} ({r})")

    val, r = v.validate_value(-50, "adj_de")
    print(f"  adj_de=-50   → {val} ({r})")

    # Full efficiency profile with some bad values
    print("\n  Full efficiency validation:")
    eff = {
        "adj_oe": 112.5,
        "adj_de": None,       # missing
        "adj_tempo": 200,     # way too high
        "barthag": 0.85,
        "efg_pct_off": 0.52,
        "rank": 15,
    }
    clean, report = v.validate_efficiency(eff, "Duke")
    print(f"  Cleaned adj_de: {clean['adj_de']}")
    print(f"  Cleaned adj_tempo: {clean['adj_tempo']}")
    print(f"  Report: {report}")
