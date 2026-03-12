"""
sports_oracle/engine/ml_model.py

ML adjustment layer. Trains on historical data and predicts
game outcomes using formula outputs + raw stats as features.

ARCHITECTURE:
  Three separate models, one per target:
    1. MarginModel  — predicts actual game margin (regression)
    2. TotalModel   — predicts actual game total (regression)
    3. WinModel     — predicts win probability (classification)

  All use the same feature set (~51 features).

MODELS:
  Primary: Ridge Regression (margin/total), Logistic Regression (win)
    - Strong regularization to prevent overfitting on 600 samples
    - Interpretable, fast, stable

  Optional: Gradient Boosting (if enough data and cross-val supports it)
    - Better at nonlinear interactions
    - Risk of overfitting with <1000 samples

USAGE:
    from engine.ml_model import MLPredictor

    predictor = MLPredictor()
    predictor.train(training_df)
    ml_result = predictor.predict(game_features_dict)

    # Or enhance a formula prediction:
    enhanced = predictor.enhance_prediction(formula_result, game_inputs)
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, log_loss, brier_score_loss,
)

from ..backtest.historical_data import HistoricalDataBuilder
from ..engine.prediction_engine import PredictionResult

logger = logging.getLogger("sports_oracle.ml")


@dataclass
class MLResult:
    """Output from the ML predictor."""
    ml_margin: float = 0.0
    ml_total: float = 140.0
    ml_win_prob: float = 0.50
    confidence: float = 0.50
    feature_importance: dict = field(default_factory=dict)


@dataclass
class ModelMetrics:
    """Training/validation metrics for a model."""
    name: str = ""
    mae: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    cv_mean: float = 0.0
    cv_std: float = 0.0
    accuracy: float = 0.0          # for classification only
    brier: float = 0.0             # for classification only
    n_samples: int = 0
    n_features: int = 0


class MLPredictor:
    """
    ML prediction layer. Trains three models and blends
    their predictions with the formula engine output.
    """

    def __init__(
        self,
        blend_weight: float = 0.35,
    ):
        """
        blend_weight: how much to weight ML vs formula (0=all formula, 1=all ML).
        0.35 means final = 65% formula + 35% ML.
        Intentionally conservative — formula is our backbone.
        """
        self.blend_weight = blend_weight
        self.feature_cols = HistoricalDataBuilder.get_feature_columns()

        # Models (initialized on train)
        self.margin_model: Optional[Ridge] = None
        self.total_model: Optional[Ridge] = None
        self.win_model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None

        # Metrics
        self.margin_metrics: Optional[ModelMetrics] = None
        self.total_metrics: Optional[ModelMetrics] = None
        self.win_metrics: Optional[ModelMetrics] = None

        self._is_trained = False

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        cv_folds: int = 5,
    ) -> dict[str, ModelMetrics]:
        """
        Train all three models on historical data.
        Returns metrics dict.
        """
        logger.info(f"Training ML models on {len(df)} games...")

        # Prepare features
        X = df[self.feature_cols].copy()
        X = X.fillna(0)

        # Targets
        y_margin = df["actual_margin"].values
        y_total = df["actual_total"].values
        y_win = df["home_won"].values

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # ── 1. Margin Model (Ridge Regression) ────────────────────────
        logger.info("  Training margin model (Ridge)...")
        self.margin_model = Ridge(alpha=10.0)  # strong regularization
        self.margin_model.fit(X_scaled, y_margin)

        cv_scores = cross_val_score(
            Ridge(alpha=10.0), X_scaled, y_margin,
            cv=kf, scoring="neg_mean_absolute_error",
        )

        y_pred_margin = self.margin_model.predict(X_scaled)
        self.margin_metrics = ModelMetrics(
            name="margin",
            mae=mean_absolute_error(y_margin, y_pred_margin),
            rmse=np.sqrt(mean_squared_error(y_margin, y_pred_margin)),
            r2=r2_score(y_margin, y_pred_margin),
            cv_mean=-cv_scores.mean(),
            cv_std=cv_scores.std(),
            n_samples=len(df),
            n_features=len(self.feature_cols),
        )

        # ── 2. Total Model (Ridge Regression) ─────────────────────────
        logger.info("  Training total model (Ridge)...")
        self.total_model = Ridge(alpha=10.0)
        self.total_model.fit(X_scaled, y_total)

        cv_scores_total = cross_val_score(
            Ridge(alpha=10.0), X_scaled, y_total,
            cv=kf, scoring="neg_mean_absolute_error",
        )

        y_pred_total = self.total_model.predict(X_scaled)
        self.total_metrics = ModelMetrics(
            name="total",
            mae=mean_absolute_error(y_total, y_pred_total),
            rmse=np.sqrt(mean_squared_error(y_total, y_pred_total)),
            r2=r2_score(y_total, y_pred_total),
            cv_mean=-cv_scores_total.mean(),
            cv_std=cv_scores_total.std(),
            n_samples=len(df),
            n_features=len(self.feature_cols),
        )

        # ── 3. Win Probability Model (Logistic Regression) ────────────
        logger.info("  Training win model (Logistic Regression)...")
        self.win_model = LogisticRegression(
            C=0.1,  # strong regularization (low C = more regularization)
            max_iter=1000,
            solver="lbfgs",
        )
        self.win_model.fit(X_scaled, y_win)

        cv_scores_win = cross_val_score(
            LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs"),
            X_scaled, y_win,
            cv=kf, scoring="accuracy",
        )

        y_pred_prob = self.win_model.predict_proba(X_scaled)[:, 1]
        y_pred_class = (y_pred_prob > 0.5).astype(int)
        self.win_metrics = ModelMetrics(
            name="win",
            accuracy=accuracy_score(y_win, y_pred_class),
            brier=brier_score_loss(y_win, y_pred_prob),
            cv_mean=cv_scores_win.mean(),
            cv_std=cv_scores_win.std(),
            n_samples=len(df),
            n_features=len(self.feature_cols),
        )

        self._is_trained = True

        metrics = {
            "margin": self.margin_metrics,
            "total": self.total_metrics,
            "win": self.win_metrics,
        }

        logger.info("  Training complete.")
        self._log_metrics()

        return metrics

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self, features: dict) -> MLResult:
        """
        Predict from a single game's features.
        features should contain all keys from get_feature_columns().
        """
        if not self._is_trained:
            logger.warning("ML models not trained — returning neutral prediction")
            return MLResult()

        # Build feature vector
        x = np.array([[features.get(col, 0.0) for col in self.feature_cols]])
        x_scaled = self.scaler.transform(x)

        ml_margin = float(self.margin_model.predict(x_scaled)[0])
        ml_total = float(self.total_model.predict(x_scaled)[0])
        ml_win_prob = float(self.win_model.predict_proba(x_scaled)[0, 1])

        # Feature importance (top 10 by absolute coefficient)
        importance = self._get_feature_importance()

        return MLResult(
            ml_margin=round(ml_margin, 2),
            ml_total=round(ml_total, 2),
            ml_win_prob=round(ml_win_prob, 4),
            confidence=self._ml_confidence(x_scaled),
            feature_importance=importance,
        )

    def enhance_prediction(
        self,
        formula_result: PredictionResult,
        game_inputs: dict,
    ) -> PredictionResult:
        """
        Blend ML predictions with formula predictions.
        Modifies the PredictionResult in place and returns it.

        Blend: final = (1 - weight) × formula + weight × ML
        """
        if not self._is_trained:
            return formula_result

        # Extract features from formula result + inputs
        features = self._extract_features(formula_result, game_inputs)
        ml = self.predict(features)

        w = self.blend_weight

        # Blend margin
        formula_margin = -formula_result.spread
        blended_margin = (1 - w) * formula_margin + w * ml.ml_margin

        # Blend total
        blended_total = (1 - w) * formula_result.total + w * ml.ml_total

        # Blend win probability
        blended_prob = (1 - w) * formula_result.home_win_prob + w * ml.ml_win_prob

        # Update result
        formula_result.home_score = blended_total / 2 + blended_margin / 2
        formula_result.away_score = blended_total / 2 - blended_margin / 2
        formula_result.spread = -blended_margin
        formula_result.total = blended_total
        formula_result.home_win_prob = blended_prob
        formula_result.away_win_prob = 1.0 - blended_prob

        # Recompute confidence with ML agreement
        formula_agrees = abs(formula_margin - ml.ml_margin) < 3.0
        if formula_agrees:
            formula_result.confidence_score = min(
                0.95, formula_result.confidence_score + 0.10
            )
        else:
            formula_result.confidence_score = max(
                0.20, formula_result.confidence_score - 0.05
            )
        formula_result.confidence = self._confidence_label(
            formula_result.confidence_score
        )

        return formula_result

    def _extract_features(
        self,
        result: PredictionResult,
        inputs: dict,
    ) -> dict:
        """Extract the flat feature dict from result + inputs."""
        h_eff = inputs.get("home_efficiency", {})
        a_eff = inputs.get("away_efficiency", {})
        venue = inputs.get("venue", {})

        features = {
            "formula_margin": -result.spread,
            "formula_total": result.total,
            "formula_win_prob": result.home_win_prob,
            "game_pace": result.game_pace,
            "adj_momentum": result.momentum_adj,
            "adj_experience": result.experience_adj,
            "adj_rest": result.rest_adj,
            "adj_seed": result.seed_adj,
            "adj_travel": result.travel_adj,
            "adj_total": result.total_adjustment,
            "h_adj_oe": h_eff.get("adj_oe", 100),
            "h_adj_de": h_eff.get("adj_de", 100),
            "h_tempo": h_eff.get("adj_tempo", 68),
            "h_efg_off": h_eff.get("efg_pct_off", 0.50),
            "h_efg_def": h_eff.get("efg_pct_def", 0.50),
            "h_to_rate_off": h_eff.get("to_rate_off", 18),
            "h_to_rate_def": h_eff.get("to_rate_def", 18),
            "h_3pt_rate": h_eff.get("three_pt_rate_off", 0.35),
            "h_3pt_pct": h_eff.get("three_pt_pct_off", 0.34),
            "h_fta_rate": h_eff.get("fta_rate_off", 0.30),
            "h_orb": h_eff.get("orb_pct", 0.30),
            "h_sos": h_eff.get("sos", 0),
            "h_barthag": h_eff.get("barthag", 0.50),
            "a_adj_oe": a_eff.get("adj_oe", 100),
            "a_adj_de": a_eff.get("adj_de", 100),
            "a_tempo": a_eff.get("adj_tempo", 68),
            "a_efg_off": a_eff.get("efg_pct_off", 0.50),
            "a_efg_def": a_eff.get("efg_pct_def", 0.50),
            "a_to_rate_off": a_eff.get("to_rate_off", 18),
            "a_to_rate_def": a_eff.get("to_rate_def", 18),
            "a_3pt_rate": a_eff.get("three_pt_rate_off", 0.35),
            "a_3pt_pct": a_eff.get("three_pt_pct_off", 0.34),
            "a_fta_rate": a_eff.get("fta_rate_off", 0.30),
            "a_orb": a_eff.get("orb_pct", 0.30),
            "a_sos": a_eff.get("sos", 0),
            "a_barthag": a_eff.get("barthag", 0.50),
            "oe_diff": h_eff.get("adj_oe", 100) - a_eff.get("adj_oe", 100),
            "de_diff": h_eff.get("adj_de", 100) - a_eff.get("adj_de", 100),
            "tempo_diff": h_eff.get("adj_tempo", 68) - a_eff.get("adj_tempo", 68),
            "barthag_diff": h_eff.get("barthag", 0.5) - a_eff.get("barthag", 0.5),
            "sos_diff": h_eff.get("sos", 0) - a_eff.get("sos", 0),
            "seed_diff": (inputs.get("away_seed") or 8) - (inputs.get("home_seed") or 8),
            "rest_diff": (inputs.get("home_rest", {}).get("rest_days") or 3) -
                         (inputs.get("away_rest", {}).get("rest_days") or 3),
            "round": inputs.get("tournament_round", 1),
            "vsi": venue.get("vsi", 1.0),
            "vpi": venue.get("vpi", 1.0),
            "v3p": venue.get("v3p", 1.0),
            "h_coach_app": inputs.get("home_experience", {}).get("coach_record", {}).get("appearances", 0),
            "a_coach_app": inputs.get("away_experience", {}).get("coach_record", {}).get("appearances", 0),
            "h_returning": inputs.get("home_experience", {}).get("returning_pct", 0.5),
            "a_returning": inputs.get("away_experience", {}).get("returning_pct", 0.5),
        }
        return features

    # ── Diagnostics ───────────────────────────────────────────────────────

    def _get_feature_importance(self, top_n: int = 10) -> dict:
        """Top features by absolute coefficient (margin model)."""
        if not self.margin_model:
            return {}

        coefs = self.margin_model.coef_
        importance = sorted(
            zip(self.feature_cols, coefs),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return {name: round(float(coef), 4) for name, coef in importance[:top_n]}

    def _ml_confidence(self, x_scaled: np.ndarray) -> float:
        """
        Estimate ML prediction confidence based on how far
        the input is from the training distribution.
        """
        if self.scaler is None:
            return 0.5
        # Mahalanobis-ish: how many std devs from center
        dist = np.sqrt(np.sum(x_scaled ** 2))
        # Typical scaled distance is sqrt(n_features) ≈ 7
        # Normalize so typical = 0.5, close to center = 0.7, far = 0.3
        confidence = max(0.2, min(0.8, 1.0 - dist / 20.0))
        return round(confidence, 2)

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

    def _log_metrics(self):
        """Log training metrics."""
        if self.margin_metrics:
            m = self.margin_metrics
            logger.info(
                f"  Margin: MAE={m.mae:.2f}, RMSE={m.rmse:.2f}, "
                f"R²={m.r2:.3f}, CV-MAE={m.cv_mean:.2f}±{m.cv_std:.2f}"
            )
        if self.total_metrics:
            m = self.total_metrics
            logger.info(
                f"  Total:  MAE={m.mae:.2f}, RMSE={m.rmse:.2f}, "
                f"R²={m.r2:.3f}, CV-MAE={m.cv_mean:.2f}±{m.cv_std:.2f}"
            )
        if self.win_metrics:
            m = self.win_metrics
            logger.info(
                f"  Win:    Acc={m.accuracy:.1%}, Brier={m.brier:.4f}, "
                f"CV-Acc={m.cv_mean:.1%}±{m.cv_std:.3f}"
            )

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def report(self) -> str:
        """Human-readable training report."""
        if not self._is_trained:
            return "ML models not yet trained."

        lines = [
            "\n" + "=" * 55,
            "  ML MODEL TRAINING REPORT",
            "=" * 55,
        ]

        if self.margin_metrics:
            m = self.margin_metrics
            lines.extend([
                f"\n  MARGIN MODEL (Ridge Regression)",
                f"  {'─'*40}",
                f"  Train MAE:     {m.mae:.2f} pts",
                f"  Train RMSE:    {m.rmse:.2f} pts",
                f"  Train R²:      {m.r2:.3f}",
                f"  CV MAE:        {m.cv_mean:.2f} ± {m.cv_std:.2f}",
                f"  Samples:       {m.n_samples}",
                f"  Features:      {m.n_features}",
            ])

        if self.total_metrics:
            m = self.total_metrics
            lines.extend([
                f"\n  TOTAL MODEL (Ridge Regression)",
                f"  {'─'*40}",
                f"  Train MAE:     {m.mae:.2f} pts",
                f"  Train RMSE:    {m.rmse:.2f} pts",
                f"  Train R²:      {m.r2:.3f}",
                f"  CV MAE:        {m.cv_mean:.2f} ± {m.cv_std:.2f}",
            ])

        if self.win_metrics:
            m = self.win_metrics
            lines.extend([
                f"\n  WIN PROBABILITY MODEL (Logistic Regression)",
                f"  {'─'*40}",
                f"  Train Accuracy:  {m.accuracy:.1%}",
                f"  Brier Score:     {m.brier:.4f}",
                f"  CV Accuracy:     {m.cv_mean:.1%} ± {m.cv_std:.3f}",
            ])

        # Feature importance
        importance = self._get_feature_importance(15)
        if importance:
            lines.extend([
                f"\n  TOP FEATURES (by margin model coefficient)",
                f"  {'─'*40}",
            ])
            for name, coef in importance.items():
                bar = "+" * int(min(20, abs(coef) * 5))
                lines.append(f"  {name:>20s}: {coef:+8.4f} {'█' * int(min(20, abs(coef)*5))}")

        lines.append("=" * 55)
        return "\n".join(lines)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ..backtest.historical_data import HistoricalDataBuilder

    print("\n🤖 ML Model — Training & Evaluation")
    print("=" * 55)

    # Build training data
    builder = HistoricalDataBuilder()
    df = builder.build_synthetic_training_set(n_seasons=14)
    print(f"  Training data: {len(df)} games")

    # Train
    predictor = MLPredictor(blend_weight=0.35)
    metrics = predictor.train(df)

    # Report
    print(predictor.report())

    # Test prediction
    print("\n  Sample prediction (single game):")
    test_features = {col: 0.0 for col in builder.get_feature_columns()}
    test_features["formula_margin"] = 8.0
    test_features["formula_total"] = 145.0
    test_features["formula_win_prob"] = 0.68
    test_features["h_adj_oe"] = 115.0
    test_features["h_adj_de"] = 95.0
    test_features["a_adj_oe"] = 108.0
    test_features["a_adj_de"] = 100.0
    test_features["seed_diff"] = 7
    test_features["barthag_diff"] = 0.15

    ml_result = predictor.predict(test_features)
    print(f"  ML margin:   {ml_result.ml_margin:+.1f}")
    print(f"  ML total:    {ml_result.ml_total:.1f}")
    print(f"  ML win prob: {ml_result.ml_win_prob:.1%}")
