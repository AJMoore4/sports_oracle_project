"""
sports_oracle/engine/ml_model.py

ML adjustment layer. Trains on historical data and predicts
game outcomes using formula outputs + raw stats as features.

ARCHITECTURE:
  Three separate models, one per target:
    1. MarginModel  — predicts actual game margin (regression)
    2. TotalModel   — predicts actual game total (regression)
    3. WinModel     — predicts win probability (classification)

  All use the same feature set (~68 features).

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
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, brier_score_loss,
)

from ..backtest.historical_data import HistoricalDataBuilder
from ..engine.prediction_engine import (
    PredictionResult,
    LOGISTIC_SIGMA,
    CalibrationProfile,
    ROUND_TOTAL_ANCHOR,
    infer_game_type_from_inputs,
    GAME_TYPE_NCAA,
    GAME_TYPE_CONFERENCE,
)

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
        blend_weight: float = 0.54,
        total_blend_weight: Optional[float] = None,
    ):
        """
        blend_weight: how much to weight ML vs formula (0=all formula, 1=all ML).
        0.42 means final = 58% formula + 42% ML.
        Intentionally conservative — formula is our backbone.
        """
        self.blend_weight = blend_weight
        self.total_blend_weight = (
            total_blend_weight
            if total_blend_weight is not None
            else min(blend_weight, 0.20)
        )
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
        self.calibration_profile = CalibrationProfile()

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
        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            logger.warning(
                "Training data missing %d feature columns; defaulting them to 0. "
                "Top missing: %s",
                len(missing_cols),
                ", ".join(missing_cols[:8]),
            )
        X = df.reindex(columns=self.feature_cols, fill_value=0.0).copy()
        X = (
            X.apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        # Targets
        y_margin = df["actual_margin"].values
        y_total = df["actual_total"].values
        y_win = df["home_won"].values

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        heldout = self._collect_time_based_predictions(
            X=X,
            df=df,
            y_margin=y_margin,
            y_total=y_total,
            y_win=y_win,
            margin_model_factory=lambda: Ridge(alpha=10.0),
            total_model_factory=lambda: Ridge(alpha=10.0),
            win_model_factory=lambda: LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
            ),
            max_folds=cv_folds,
        )
        cv_mean, cv_std = self._regression_summary(
            heldout,
            actual_col="actual_margin",
            predicted_col="blended_margin",
        )
        cv_total_mean, cv_total_std = self._regression_summary(
            heldout,
            actual_col="actual_total",
            predicted_col="blended_total",
        )
        cv_win_mean, cv_win_std = self._classification_summary(
            heldout,
            actual_col="home_won",
            predicted_col="home_win_pred",
        )

        # ── 1. Margin Model (Ridge Regression) ────────────────────────
        logger.info("  Training margin model (Ridge)...")
        self.margin_model = Ridge(alpha=10.0)  # strong regularization
        self.margin_model.fit(X_scaled, y_margin)

        y_pred_margin = self.margin_model.predict(X_scaled)
        self.margin_metrics = ModelMetrics(
            name="margin",
            mae=mean_absolute_error(y_margin, y_pred_margin),
            rmse=np.sqrt(mean_squared_error(y_margin, y_pred_margin)),
            r2=r2_score(y_margin, y_pred_margin),
            cv_mean=cv_mean,
            cv_std=cv_std,
            n_samples=len(df),
            n_features=len(self.feature_cols),
        )

        # ── 2. Total Model (Ridge Regression) ─────────────────────────
        logger.info("  Training total model (Ridge)...")
        self.total_model = Ridge(alpha=10.0)
        self.total_model.fit(X_scaled, y_total)

        y_pred_total = self.total_model.predict(X_scaled)
        self.total_metrics = ModelMetrics(
            name="total",
            mae=mean_absolute_error(y_total, y_pred_total),
            rmse=np.sqrt(mean_squared_error(y_total, y_pred_total)),
            r2=r2_score(y_total, y_pred_total),
            cv_mean=cv_total_mean,
            cv_std=cv_total_std,
            n_samples=len(df),
            n_features=len(self.feature_cols),
        )

        # ── 3. Win Probability Model (Logistic Regression) ────────────
        logger.info("  Training win model (Logistic Regression)...")
        self.win_model = LogisticRegression(
            C=1.0,  # strong regularization (low C = more regularization)
            max_iter=1000,
            solver="lbfgs",
        )
        self.win_model.fit(X_scaled, y_win)

        y_pred_prob = self.win_model.predict_proba(X_scaled)[:, 1]
        y_pred_class = (y_pred_prob > 0.5).astype(int)

        self.win_metrics = ModelMetrics(
            name="win",
            accuracy=accuracy_score(y_win, y_pred_class),
            brier=brier_score_loss(y_win, y_pred_prob),
            cv_mean=cv_win_mean,
            cv_std=cv_win_std,
            n_samples=len(df),
            n_features=len(self.feature_cols),
        )

        self._is_trained = True

        metrics = {
            "margin": self.margin_metrics,
            "total": self.total_metrics,
            "win": self.win_metrics,
        }

        self.calibration_profile = self._fit_calibration_profile(df, heldout)
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
        feature_row = pd.DataFrame(
            [{col: features.get(col, 0.0) for col in self.feature_cols}],
            columns=self.feature_cols,
        )
        feature_row = (
            feature_row.apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        x_scaled = self.scaler.transform(feature_row)

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

        w, w_total = self._adaptive_blend_weights(
            formula_result=formula_result,
            game_inputs=game_inputs,
            ml=ml,
        )

        # Blend margin
        formula_margin = -formula_result.spread
        blended_margin = (1 - w) * formula_margin + w * ml.ml_margin

        # Blend total
        blended_total = (1 - w_total) * formula_result.total + w_total * ml.ml_total

        # Keep probability internally consistent with the final blended margin.
        blended_prob = self._margin_to_win_prob(blended_margin)

        if not np.isfinite(
            [
                blended_margin,
                blended_total,
                blended_prob,
            ]
        ).all():
            logger.warning("ML blend produced non-finite values — keeping formula prediction")
            return formula_result

        # Update result
        formula_result.home_score = blended_total / 2 + blended_margin / 2
        formula_result.away_score = blended_total / 2 - blended_margin / 2
        formula_result.spread = -blended_margin
        formula_result.total = blended_total
        formula_result.home_win_prob = blended_prob
        formula_result.away_win_prob = 1.0 - blended_prob
        formula_result.margin_uncertainty = round(self.calibration_profile.margin_mae, 2)
        formula_result.total_uncertainty = round(self.calibration_profile.total_mae, 2)

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
            "adj_injury": result.injury_adj,
            "adj_seed": result.seed_adj,
            "adj_travel": result.travel_adj,
            "adj_total": result.total_adjustment,
            "adj_total_points": result.total_points_adjustment,
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
            # ── New BartTorvik features ──
            "h_drb": h_eff.get("drb_pct", 0.70),
            "a_drb": a_eff.get("drb_pct", 0.70),
            "h_fta_rate_def": h_eff.get("fta_rate_def", 0.30),
            "a_fta_rate_def": a_eff.get("fta_rate_def", 0.30),
            "h_2pt_off": h_eff.get("two_pt_pct_off", 0.48),
            "a_2pt_off": a_eff.get("two_pt_pct_off", 0.48),
            "h_2pt_def": h_eff.get("two_pt_pct_def", 0.48),
            "a_2pt_def": a_eff.get("two_pt_pct_def", 0.48),
            "h_3pt_rate_def": h_eff.get("three_pt_rate_def", 0.36),
            "a_3pt_rate_def": a_eff.get("three_pt_rate_def", 0.36),
            "h_elite_sos": h_eff.get("elite_sos", 0),
            "a_elite_sos": a_eff.get("elite_sos", 0),
            "h_nc_sos": h_eff.get("non_conf_sos", 0),
            "a_nc_sos": a_eff.get("non_conf_sos", 0),
            # ── Derived features (from extended stats) ──
            "h_close_pct": inputs.get("home_extended", {}).get("close_game_pct", 0.50),
            "a_close_pct": inputs.get("away_extended", {}).get("close_game_pct", 0.50),
            "h_close_rate": inputs.get("home_extended", {}).get("close_game_rate", 0.25),
            "a_close_rate": inputs.get("away_extended", {}).get("close_game_rate", 0.25),
            "h_close_games": inputs.get("home_extended", {}).get("close_game_games", 0),
            "a_close_games": inputs.get("away_extended", {}).get("close_game_games", 0),
            "h_margin_std": inputs.get("home_extended", {}).get("margin_std", 12.0),
            "a_margin_std": inputs.get("away_extended", {}).get("margin_std", 12.0),
            "h_conf_strength": inputs.get("home_extended", {}).get("conf_strength", 0.50),
            "a_conf_strength": inputs.get("away_extended", {}).get("conf_strength", 0.50),
            "h_conf_tourney_w": inputs.get("home_extended", {}).get("conf_tourney_wins", 0),
            "a_conf_tourney_w": inputs.get("away_extended", {}).get("conf_tourney_wins", 0),
            # ── Differentials ──
            "oe_diff": h_eff.get("adj_oe", 100) - a_eff.get("adj_oe", 100),
            "de_diff": h_eff.get("adj_de", 100) - a_eff.get("adj_de", 100),
            "tempo_diff": h_eff.get("adj_tempo", 68) - a_eff.get("adj_tempo", 68),
            "barthag_diff": h_eff.get("barthag", 0.5) - a_eff.get("barthag", 0.5),
            "sos_diff": h_eff.get("sos", 0) - a_eff.get("sos", 0),
            "seed_diff": (inputs.get("away_seed") or 8) - (inputs.get("home_seed") or 8),
            "rest_diff": (inputs.get("home_rest", {}).get("rest_days") or 3) -
                         (inputs.get("away_rest", {}).get("rest_days") or 3),
            "round": inputs.get("tournament_round", 1),
            "raw_total": result.raw_total,
            "vsi": venue.get("vsi", 1.0),
            "vpi": venue.get("vpi", 1.0),
            "v3p": venue.get("v3p", 1.0),
            "venue_sample_size": venue.get("sample_size", 0),
            "h_coach_app": inputs.get("home_experience", {}).get("coach_record", {}).get("appearances", 0),
            "a_coach_app": inputs.get("away_experience", {}).get("coach_record", {}).get("appearances", 0),
            "h_returning": inputs.get("home_experience", {}).get("returning_pct", 0.5),
            "a_returning": inputs.get("away_experience", {}).get("returning_pct", 0.5),
        }
        market_total = inputs.get("market_lines", {}).get("consensus_total")
        try:
            market_total = float(market_total)
            if not np.isfinite(market_total):
                raise ValueError
            features["market_total"] = market_total
            features["market_total_available"] = 1.0
            features["market_total_delta"] = result.total - market_total
        except (TypeError, ValueError):
            features["market_total"] = 0.0
            features["market_total_available"] = 0.0
            features["market_total_delta"] = 0.0
        return features

    def _adaptive_blend_weights(
        self,
        formula_result: PredictionResult,
        game_inputs: dict,
        ml: MLResult,
    ) -> tuple[float, float]:
        game_type = infer_game_type_from_inputs(game_inputs)

        margin_weight = float(self.blend_weight)
        total_weight = float(self.total_blend_weight)
        projected_spread = abs(float(formula_result.spread))

        if game_type == GAME_TYPE_NCAA:
            margin_weight *= 1.00
            total_weight *= 0.95
        elif game_type == GAME_TYPE_CONFERENCE:
            margin_weight *= 0.90
            total_weight *= 0.80
        else:
            margin_weight *= 0.85
            total_weight *= 0.75

        venue_sample = float(game_inputs.get("venue", {}).get("sample_size", 0) or 0)
        if venue_sample < 5:
            margin_weight *= 0.92
            total_weight *= 0.88
        elif venue_sample < 15:
            margin_weight *= 0.97
            total_weight *= 0.95

        if projected_spread <= 2.5:
            margin_weight *= 0.72
        elif projected_spread <= 5.0:
            margin_weight *= 0.84
        elif projected_spread >= 18.0:
            margin_weight *= 0.86
            total_weight *= 0.90
        elif projected_spread >= 12.0:
            margin_weight *= 0.94

        ml_conf = float(ml.confidence or 0.5)
        if ml_conf < 0.40:
            margin_weight *= 0.80
            total_weight *= 0.75
        elif ml_conf < 0.50:
            margin_weight *= 0.90
            total_weight *= 0.88
        elif ml_conf > 0.62:
            margin_weight *= 1.05

        market_spread = game_inputs.get("market_lines", {}).get("consensus_spread")
        try:
            market_spread = float(market_spread)
            if np.isfinite(market_spread):
                spread_gap = abs(float(formula_result.spread) - market_spread)
                if spread_gap >= 6.0:
                    margin_weight *= 0.55
                elif spread_gap >= 4.0:
                    margin_weight *= 0.70
                elif spread_gap >= 2.5:
                    margin_weight *= 0.82
        except (TypeError, ValueError):
            market_spread = None

        market_total = game_inputs.get("market_lines", {}).get("consensus_total")
        try:
            market_total = float(market_total)
            if np.isfinite(market_total):
                total_gap = abs(float(formula_result.total) - market_total)
                if total_gap >= 8.0:
                    total_weight *= 0.30
                elif total_gap >= 5.0:
                    total_weight *= 0.50
                elif total_gap >= 3.0:
                    total_weight *= 0.70
        except (TypeError, ValueError):
            market_total = None

        if float(formula_result.raw_total) >= 160.0:
            total_weight *= 0.60
        elif float(formula_result.raw_total) >= 150.0:
            total_weight *= 0.75

        if float(formula_result.margin_uncertainty) > float(self.calibration_profile.margin_mae) * 1.10:
            margin_weight *= 0.90
        if float(formula_result.total_uncertainty) > float(self.calibration_profile.total_mae) * 1.10:
            total_weight *= 0.85

        margin_weight = float(
            np.clip(margin_weight, 0.05, min(0.55, self.blend_weight * 1.10))
        )
        total_weight = float(
            np.clip(total_weight, 0.02, min(0.22, self.total_blend_weight * 1.10))
        )
        return margin_weight, total_weight

    @staticmethod
    def _score_summary(scores: np.ndarray) -> tuple[float, float]:
        if scores.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(scores)), float(np.std(scores))

    @staticmethod
    def _build_time_based_folds(
        df: pd.DataFrame,
        max_folds: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if "season" not in df.columns:
            return []

        season_values = pd.to_numeric(df["season"], errors="coerce").to_numpy()
        unique_seasons = sorted(
            int(season)
            for season in pd.Series(season_values).dropna().unique()
        )
        if len(unique_seasons) < 2:
            return []

        test_seasons = unique_seasons[1:]
        if max_folds and len(test_seasons) > max_folds:
            test_seasons = test_seasons[-max_folds:]

        folds = []
        for test_season in test_seasons:
            train_idx = np.flatnonzero(season_values < test_season)
            test_idx = np.flatnonzero(season_values == test_season)
            if len(train_idx) < 10 or len(test_idx) < 5:
                continue
            folds.append((train_idx, test_idx))
        return folds

    def _collect_time_based_predictions(
        self,
        X: pd.DataFrame,
        df: pd.DataFrame,
        y_margin: np.ndarray,
        y_total: np.ndarray,
        y_win: np.ndarray,
        margin_model_factory,
        total_model_factory,
        win_model_factory,
        max_folds: int,
    ) -> pd.DataFrame:
        rows = []
        for train_idx, test_idx in self._build_time_based_folds(df, max_folds):
            y_train = y_win[train_idx]
            if len(np.unique(y_train)) < 2:
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X.iloc[train_idx])
            X_test = scaler.transform(X.iloc[test_idx])
            margin_model = margin_model_factory()
            total_model = total_model_factory()
            win_model = win_model_factory()

            margin_model.fit(X_train, y_margin[train_idx])
            total_model.fit(X_train, y_total[train_idx])
            win_model.fit(X_train, y_train)

            margin_preds = margin_model.predict(X_test)
            total_preds = total_model.predict(X_test)
            win_preds = win_model.predict_proba(X_test)[:, 1]

            test_rows = df.iloc[test_idx].copy()
            formula_margin = pd.to_numeric(
                test_rows.get("formula_margin"),
                errors="coerce",
            ).fillna(0.0).to_numpy()
            formula_total = pd.to_numeric(
                test_rows.get("formula_total"),
                errors="coerce",
            ).fillna(0.0).to_numpy()

            blended_margin = (1 - self.blend_weight) * formula_margin + self.blend_weight * margin_preds
            blended_total = (1 - self.total_blend_weight) * formula_total + self.total_blend_weight * total_preds

            fold_df = pd.DataFrame({
                "season": pd.to_numeric(test_rows.get("season"), errors="coerce"),
                "actual_margin": y_margin[test_idx],
                "actual_total": y_total[test_idx],
                "home_won": y_win[test_idx],
                "formula_margin": formula_margin,
                "formula_total": formula_total,
                "blended_margin": blended_margin,
                "blended_total": blended_total,
                "home_win_pred": win_preds,
            })
            rows.append(fold_df)

        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    @staticmethod
    def _regression_summary(
        heldout: pd.DataFrame,
        actual_col: str,
        predicted_col: str,
    ) -> tuple[float, float]:
        if heldout.empty:
            return float("nan"), float("nan")

        scores = []
        for _, season_df in heldout.groupby("season"):
            actual = pd.to_numeric(season_df.get(actual_col), errors="coerce")
            predicted = pd.to_numeric(season_df.get(predicted_col), errors="coerce")
            mask = actual.notna() & predicted.notna()
            if mask.any():
                scores.append(mean_absolute_error(actual[mask], predicted[mask]))
        return MLPredictor._score_summary(np.asarray(scores, dtype=float))

    @staticmethod
    def _classification_summary(
        heldout: pd.DataFrame,
        actual_col: str,
        predicted_col: str,
    ) -> tuple[float, float]:
        if heldout.empty:
            return float("nan"), float("nan")

        scores = []
        for _, season_df in heldout.groupby("season"):
            actual = pd.to_numeric(season_df.get(actual_col), errors="coerce")
            predicted = pd.to_numeric(season_df.get(predicted_col), errors="coerce")
            mask = actual.notna() & predicted.notna()
            if mask.any():
                pred_class = (predicted[mask] >= 0.5).astype(int)
                scores.append(accuracy_score(actual[mask].astype(int), pred_class))
        return MLPredictor._score_summary(np.asarray(scores, dtype=float))

    def _fit_calibration_profile(
        self,
        df: pd.DataFrame,
        heldout: pd.DataFrame,
    ) -> CalibrationProfile:
        profile = CalibrationProfile()
        profile.probability_intercept = 0.0
        profile.probability_slope = 1.0 / LOGISTIC_SIGMA

        if not heldout.empty:
            calib = heldout.dropna(subset=["blended_margin", "home_won"]).copy()
            if len(calib) >= 20 and calib["home_won"].nunique() > 1:
                logistic = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000)
                logistic.fit(calib[["blended_margin"]], calib["home_won"].astype(int))
                slope = float(logistic.coef_[0][0])
                intercept = float(logistic.intercept_[0])
                if np.isfinite(slope) and abs(slope) > 1e-6:
                    profile.probability_slope = float(np.clip(slope, 0.02, 0.25))
                    profile.probability_intercept = float(np.clip(intercept, -1.0, 1.0))

            margin_resid = calib["actual_margin"] - calib["blended_margin"] if "actual_margin" in calib.columns else pd.Series(dtype=float)
            total_resid = calib["actual_total"] - calib["blended_total"] if "actual_total" in calib.columns else pd.Series(dtype=float)
            margin_resid = pd.to_numeric(margin_resid, errors="coerce").dropna()
            total_resid = pd.to_numeric(total_resid, errors="coerce").dropna()
            if not margin_resid.empty:
                profile.margin_mae = float(np.mean(np.abs(margin_resid)))
                profile.margin_rmse = float(np.sqrt(np.mean(np.square(margin_resid))))
            if not total_resid.empty:
                profile.total_mae = float(np.mean(np.abs(total_resid)))
                profile.total_rmse = float(np.sqrt(np.mean(np.square(total_resid))))

        numeric = df.copy()
        for column in ("round", "actual_total", "raw_total", "market_total"):
            if column in numeric.columns:
                numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

        round_anchors = dict(ROUND_TOTAL_ANCHOR)
        if {"round", "actual_total"}.issubset(numeric.columns):
            round_means = (
                numeric.dropna(subset=["round", "actual_total"])
                .groupby("round")["actual_total"]
                .mean()
            )
            for round_value, anchor in round_means.items():
                round_anchors[int(round_value)] = round(float(anchor), 2)
        profile.round_total_anchor = round_anchors

        if {"round", "actual_total", "raw_total"}.issubset(numeric.columns):
            regress_df = numeric.dropna(subset=["round", "actual_total", "raw_total"]).copy()
            if not regress_df.empty:
                anchors = regress_df["round"].apply(
                    lambda rnd: round_anchors.get(int(rnd), ROUND_TOTAL_ANCHOR[1])
                )
                x = (regress_df["raw_total"] - anchors).to_numpy()
                y = (regress_df["actual_total"] - anchors).to_numpy()
                denom = float(np.dot(x, x))
                if denom > 0:
                    shrink = float(np.dot(x, y) / denom)
                    profile.total_baseline_shrink = float(np.clip(shrink, 0.35, 0.95))

        if {"round", "actual_total", "raw_total", "market_total"}.issubset(numeric.columns):
            market_df = numeric.dropna(subset=["round", "actual_total", "raw_total", "market_total"]).copy()
            if not market_df.empty:
                anchors = market_df["round"].apply(
                    lambda rnd: round_anchors.get(int(rnd), ROUND_TOTAL_ANCHOR[1])
                )
                conservative = anchors + (
                    (market_df["raw_total"] - anchors) * profile.total_baseline_shrink
                )
                market_gap = (market_df["market_total"] - conservative).to_numpy()
                target_gap = (market_df["actual_total"] - conservative).to_numpy()
                denom = float(np.dot(market_gap, market_gap))
                if denom > 0:
                    blend = float(np.dot(market_gap, target_gap) / denom)
                    profile.total_market_blend = float(np.clip(blend, 0.0, 0.75))

        return profile

    def build_engine_calibration_profile(self) -> CalibrationProfile:
        return self.calibration_profile

    def _margin_to_win_prob(self, margin: float) -> float:
        return float(
            1.0
            / (
                1.0
                + np.exp(
                    -(
                        self.calibration_profile.probability_intercept
                        + self.calibration_profile.probability_slope * margin
                    )
                )
            )
        )

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
        dist = np.sqrt(np.sum(x_scaled ** 2))
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
                f"R²={m.r2:.3f}, CV MAE={m.cv_mean:.2f}±{m.cv_std:.3f}"
            )
        if self.total_metrics:
            m = self.total_metrics
            logger.info(
                f"  Total:  MAE={m.mae:.2f}, RMSE={m.rmse:.2f}, "
                f"R²={m.r2:.3f}, CV MAE={m.cv_mean:.2f}±{m.cv_std:.3f}"
            )
        if self.win_metrics:
            m = self.win_metrics
            logger.info(
                f"  Win:    Acc={m.accuracy:.1%}, Brier={m.brier:.4f}, "
                f"CV Acc={m.cv_mean:.1%}±{m.cv_std:.3f}"
            )
        logger.info(
            "  Calibration: "
            f"prob_intercept={self.calibration_profile.probability_intercept:+.3f}, "
            f"prob_slope={self.calibration_profile.probability_slope:.4f}, "
            f"total_shrink={self.calibration_profile.total_baseline_shrink:.3f}, "
            f"market_blend={self.calibration_profile.total_market_blend:.3f}, "
            f"margin_mae={self.calibration_profile.margin_mae:.2f}, "
            f"total_mae={self.calibration_profile.total_mae:.2f}"
        )

    def report(self) -> str:
        """Human-readable training report."""
        lines = [
            f"\n{'='*55}",
            f"  ML MODEL TRAINING REPORT",
            f"  Features: {len(self.feature_cols)}",
            f"{'='*55}",
        ]

        if self.margin_metrics:
            m = self.margin_metrics
            lines.extend([
                f"\n  MARGIN MODEL (Ridge Regression, α=10.0)",
                f"  {'─'*40}",
                f"  Train MAE:       {m.mae:.2f} pts",
                f"  Train RMSE:      {m.rmse:.2f} pts",
                f"  Train R²:        {m.r2:.3f}",
                f"  CV MAE:          {m.cv_mean:.2f} ± {m.cv_std:.3f}",
                f"  Samples:         {m.n_samples}",
                f"  Features:        {m.n_features}",
            ])

        if self.total_metrics:
            m = self.total_metrics
            lines.extend([
                f"\n  TOTAL MODEL (Ridge Regression, α=10.0)",
                f"  {'─'*40}",
                f"  Train MAE:       {m.mae:.2f} pts",
                f"  Train RMSE:      {m.rmse:.2f} pts",
                f"  Train R²:        {m.r2:.3f}",
                f"  CV MAE:          {m.cv_mean:.2f} ± {m.cv_std:.3f}",
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

        lines.extend([
            f"\n  CALIBRATION PROFILE",
            f"  {'─'*40}",
            f"  Prob Intercept:  {self.calibration_profile.probability_intercept:+.3f}",
            f"  Prob Slope:      {self.calibration_profile.probability_slope:.4f}",
            f"  Total Shrink:    {self.calibration_profile.total_baseline_shrink:.3f}",
            f"  Market Blend:    {self.calibration_profile.total_market_blend:.3f}",
            f"  Margin MAE:      {self.calibration_profile.margin_mae:.2f}",
            f"  Total MAE:       {self.calibration_profile.total_mae:.2f}",
        ])

        # Feature importance
        importance = self._get_feature_importance(15)
        if importance:
            lines.extend([
                f"\n  TOP FEATURES (by margin model coefficient)",
                f"  {'─'*40}",
            ])
            for name, coef in importance.items():
                bar = "█" * int(min(20, abs(coef) * 5))
                lines.append(f"  {name:>20s}: {coef:+8.4f} {bar}")

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
