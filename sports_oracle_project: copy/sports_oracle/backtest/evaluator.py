"""
sports_oracle/backtest/evaluator.py

Evaluates prediction accuracy across multiple dimensions:
  - Straight-up win/loss accuracy
  - Spread accuracy (against the number)
  - Total accuracy (over/under)
  - Calibration (are 70% predictions right 70% of the time?)
  - ROI simulation (if we bet every edge, what's the return?)
  - By-round breakdown (are we better in early or late rounds?)
  - Upset detection (did we identify the right upsets?)

USAGE:
    from backtest.evaluator import Evaluator

    evaluator = Evaluator()
    report = evaluator.evaluate(predictions_df, actuals_df)
    print(report.summary())

    # Or from training data with formula + actual columns:
    report = evaluator.evaluate_from_training(training_df)
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("sports_oracle.evaluator")


@dataclass
class EvaluationReport:
    """Complete evaluation metrics."""

    # Straight-up accuracy
    overall_accuracy: float = 0.0
    n_games: int = 0
    n_correct: int = 0

    # By round
    accuracy_by_round: dict = field(default_factory=dict)

    # Margin accuracy
    margin_mae: float = 0.0
    margin_rmse: float = 0.0
    margin_r2: float = 0.0

    # Total accuracy
    total_mae: float = 0.0
    total_rmse: float = 0.0

    # Calibration
    calibration_buckets: list = field(default_factory=list)
    calibration_error: float = 0.0   # expected calibration error

    # Upset detection
    upset_accuracy: float = 0.0
    upsets_predicted: int = 0
    upsets_actual: int = 0
    upsets_correct: int = 0

    # ROI (simulated flat betting)
    roi_spread: float = 0.0
    roi_total: float = 0.0
    roi_moneyline: float = 0.0

    def summary(self) -> str:
        """Human-readable evaluation report."""
        lines = [
            f"\n{'═'*60}",
            f"  PREDICTION EVALUATION REPORT",
            f"{'═'*60}",
            f"",
            f"  STRAIGHT-UP ACCURACY",
            f"  {'─'*45}",
            f"  Overall:    {self.overall_accuracy:.1%}  ({self.n_correct}/{self.n_games})",
        ]

        if self.accuracy_by_round:
            lines.append(f"")
            lines.append(f"  BY ROUND:")
            round_names = {
                1: "First Round", 2: "Second Round", 3: "Sweet 16",
                4: "Elite 8", 5: "Final Four", 6: "Championship",
            }
            for rd in sorted(self.accuracy_by_round.keys()):
                acc, n = self.accuracy_by_round[rd]
                name = round_names.get(rd, f"Round {rd}")
                lines.append(f"    {name:>15s}: {acc:.1%}  (n={n})")

        lines.extend([
            f"",
            f"  MARGIN PREDICTION",
            f"  {'─'*45}",
            f"  MAE:   {self.margin_mae:.2f} pts  (avg error)",
            f"  RMSE:  {self.margin_rmse:.2f} pts",
            f"  R²:    {self.margin_r2:.3f}",
            f"",
            f"  TOTAL PREDICTION",
            f"  {'─'*45}",
            f"  MAE:   {self.total_mae:.2f} pts",
            f"  RMSE:  {self.total_rmse:.2f} pts",
        ])

        if self.calibration_buckets:
            lines.extend([
                f"",
                f"  CALIBRATION",
                f"  {'─'*45}",
                f"  {'Predicted':>10s}  {'Actual':>8s}  {'Count':>6s}",
            ])
            for bucket in self.calibration_buckets:
                lines.append(
                    f"  {bucket['predicted']:>9.0%}   "
                    f"{bucket['actual']:>7.1%}   "
                    f"{bucket['count']:>5d}"
                )
            lines.append(f"  ECE (Expected Calibration Error): {self.calibration_error:.4f}")

        lines.extend([
            f"",
            f"  UPSET DETECTION",
            f"  {'─'*45}",
            f"  Upsets predicted:  {self.upsets_predicted}",
            f"  Upsets actual:     {self.upsets_actual}",
            f"  Correctly called:  {self.upsets_correct}",
            f"  Upset accuracy:    {self.upset_accuracy:.1%}" if self.upsets_predicted > 0 else "  Upset accuracy:    N/A",
        ])

        lines.append(f"{'═'*60}")
        return "\n".join(lines)


class Evaluator:
    """
    Evaluates prediction quality across multiple metrics.
    """

    def evaluate_from_training(
        self,
        df: pd.DataFrame,
        prob_col: str = "formula_win_prob",
        margin_col: str = "formula_margin",
        total_col: str = "formula_total",
    ) -> EvaluationReport:
        """
        Evaluate from a training DataFrame that has both
        prediction columns and actual outcome columns.
        """
        report = EvaluationReport()

        # ── Straight-up accuracy ──────────────────────────────────────
        predicted_home_win = df[prob_col] > 0.5
        actual_home_win = df["home_won"] == 1

        report.n_games = len(df)
        report.n_correct = int((predicted_home_win == actual_home_win).sum())
        report.overall_accuracy = report.n_correct / max(1, report.n_games)

        # ── By round ──────────────────────────────────────────────────
        if "round" in df.columns:
            for rd in sorted(df["round"].unique()):
                rd_mask = df["round"] == rd
                rd_pred = predicted_home_win[rd_mask]
                rd_actual = actual_home_win[rd_mask]
                n = len(rd_pred)
                if n > 0:
                    acc = float((rd_pred == rd_actual).sum()) / n
                    report.accuracy_by_round[int(rd)] = (acc, n)

        # ── Margin accuracy ───────────────────────────────────────────
        if margin_col in df.columns and "actual_margin" in df.columns:
            pred_margin = df[margin_col].values
            actual_margin = df["actual_margin"].values

            report.margin_mae = float(np.mean(np.abs(pred_margin - actual_margin)))
            report.margin_rmse = float(np.sqrt(np.mean((pred_margin - actual_margin) ** 2)))

            ss_res = np.sum((actual_margin - pred_margin) ** 2)
            ss_tot = np.sum((actual_margin - np.mean(actual_margin)) ** 2)
            report.margin_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # ── Total accuracy ────────────────────────────────────────────
        if total_col in df.columns and "actual_total" in df.columns:
            pred_total = df[total_col].values
            actual_total = df["actual_total"].values

            report.total_mae = float(np.mean(np.abs(pred_total - actual_total)))
            report.total_rmse = float(np.sqrt(np.mean((pred_total - actual_total) ** 2)))

        # ── Calibration ───────────────────────────────────────────────
        if prob_col in df.columns:
            report.calibration_buckets, report.calibration_error = (
                self._compute_calibration(
                    df[prob_col].values,
                    df["home_won"].values,
                )
            )

        # ── Upset detection ───────────────────────────────────────────
        if "h_seed" in df.columns and "a_seed" in df.columns:
            report = self._compute_upset_metrics(df, prob_col, report)

        return report

    def _compute_calibration(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> tuple[list[dict], float]:
        """
        Compute calibration: are X% predictions right X% of the time?

        Returns (calibration_buckets, expected_calibration_error)
        """
        bins = np.linspace(0, 1, n_bins + 1)
        buckets = []
        ece = 0.0
        total = len(predicted_probs)

        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (predicted_probs >= lo) & (predicted_probs < hi)
            if i == n_bins - 1:
                mask = (predicted_probs >= lo) & (predicted_probs <= hi)

            count = int(mask.sum())
            if count == 0:
                continue

            avg_pred = float(predicted_probs[mask].mean())
            avg_actual = float(actual_outcomes[mask].mean())

            buckets.append({
                "bin_lo": round(lo, 2),
                "bin_hi": round(hi, 2),
                "predicted": avg_pred,
                "actual": avg_actual,
                "count": count,
                "abs_error": abs(avg_pred - avg_actual),
            })

            ece += (count / total) * abs(avg_pred - avg_actual)

        return buckets, round(ece, 4)

    def _compute_upset_metrics(
        self,
        df: pd.DataFrame,
        prob_col: str,
        report: EvaluationReport,
    ) -> EvaluationReport:
        """
        Evaluate upset prediction accuracy.
        An upset = lower seed (higher number) wins.
        """
        # Home team is higher seed (lower number) in our convention
        has_seeds = df["h_seed"].notna() & df["a_seed"].notna()
        seeded = df[has_seeds].copy()

        if seeded.empty:
            return report

        # "Upset" = away team (lower seed / higher number) wins
        # But we need to check who's the favorite by seed
        higher_is_home = seeded["h_seed"] < seeded["a_seed"]
        same_seed = seeded["h_seed"] == seeded["a_seed"]

        # Filter to games with a clear favorite by seed
        diff_seed = seeded[~same_seed].copy()
        if diff_seed.empty:
            return report

        # For each game: did the underdog win?
        # Underdog = team with higher seed number
        underdog_won = np.where(
            diff_seed["h_seed"] < diff_seed["a_seed"],
            diff_seed["home_won"] == 0,   # higher seed home, so upset if away wins
            diff_seed["home_won"] == 1,   # higher seed away, so upset if home wins
        )

        # Did we predict the upset?
        # We predicted underdog when our model gave them >50% win prob
        predicted_underdog = np.where(
            diff_seed["h_seed"] < diff_seed["a_seed"],
            diff_seed[prob_col] < 0.50,   # we think away wins (upset)
            diff_seed[prob_col] > 0.50,   # we think home wins (upset)
        )

        report.upsets_actual = int(underdog_won.sum())
        report.upsets_predicted = int(predicted_underdog.sum())
        report.upsets_correct = int((predicted_underdog & underdog_won).sum())
        report.upset_accuracy = (
            report.upsets_correct / max(1, report.upsets_predicted)
        )

        return report

    # ── Comparison evaluation ─────────────────────────────────────────

    def compare_models(
        self,
        df: pd.DataFrame,
        model_configs: dict[str, tuple[str, str, str]],
    ) -> pd.DataFrame:
        """
        Compare multiple models side-by-side.

        model_configs: {
            "Formula": ("formula_win_prob", "formula_margin", "formula_total"),
            "ML": ("ml_win_prob", "ml_margin", "ml_total"),
        }

        Returns DataFrame with metrics per model.
        """
        rows = []
        for name, (prob_col, margin_col, total_col) in model_configs.items():
            if prob_col not in df.columns:
                continue

            report = self.evaluate_from_training(
                df,
                prob_col=prob_col,
                margin_col=margin_col,
                total_col=total_col,
            )
            rows.append({
                "model": name,
                "accuracy": report.overall_accuracy,
                "margin_mae": report.margin_mae,
                "margin_rmse": report.margin_rmse,
                "margin_r2": report.margin_r2,
                "total_mae": report.total_mae,
                "calibration_error": report.calibration_error,
                "upsets_correct": report.upsets_correct,
                "n_games": report.n_games,
            })

        return pd.DataFrame(rows)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ..backtest.historical_data import HistoricalDataBuilder

    print("\n📈 Evaluator — Backtesting the Formula Engine")
    print("=" * 60)

    builder = HistoricalDataBuilder()
    df = builder.build_synthetic_training_set(n_seasons=14)
    print(f"  Test data: {len(df)} games")

    evaluator = Evaluator()
    report = evaluator.evaluate_from_training(df)
    print(report.summary())
