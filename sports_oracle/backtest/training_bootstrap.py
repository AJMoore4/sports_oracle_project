"""
sports_oracle/backtest/training_bootstrap.py

Shared runtime ML bootstrap for the CLI and API server.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

import pandas as pd

from ..collectors.config import current_season
from ..engine.ml_model import MLPredictor
from .historical_data import HistoricalDataBuilder
from .live_training import LiveTrainingBuilder, TRAINING_START_SEASON

DEFAULT_ML_BLEND_WEIGHT = 0.45
DEFAULT_TOTAL_BLEND_WEIGHT = 0.15
MIN_LIVE_TRAINING_ROWS = 50
RUNTIME_ML_CACHE_VERSION = "v1"
RUNTIME_ML_LIVE_TTL_HOURS = 8.0


@dataclass
class RuntimeMLBootstrapResult:
    predictor: MLPredictor
    training_df: pd.DataFrame
    source: str


def _emit(status: Optional[Callable[[str], None]], message: str) -> None:
    if status is not None:
        status(message)


def _season_summary(df: pd.DataFrame) -> str:
    if "season" not in df.columns or df.empty:
        return "no season coverage"

    seasons = sorted(
        int(season)
        for season in pd.to_numeric(df["season"], errors="coerce")
        .dropna()
        .unique()
    )
    if not seasons:
        return "no season coverage"
    if len(seasons) == 1:
        return f"season {seasons[0]}"
    return f"seasons {seasons[0]}-{seasons[-1]}"


def _runtime_ml_cache_dir(cache_dir: str) -> str:
    path = os.path.join(cache_dir, "runtime_ml", RUNTIME_ML_CACHE_VERSION)
    os.makedirs(path, exist_ok=True)
    return path


def _runtime_ml_cache_path(
    cache_dir: str,
    source: str,
    season: int,
    blend_weight: float,
    total_blend_weight: float,
    min_live_rows: int,
    use_live_tournament_fit: bool,
) -> str:
    raw_key = "|".join(
        [
            f"source={source}",
            f"season={int(season)}",
            f"blend={blend_weight:.4f}",
            f"total_blend={total_blend_weight:.4f}",
            f"start={TRAINING_START_SEASON}",
            f"min_live_rows={int(min_live_rows)}",
            f"use_live={int(bool(use_live_tournament_fit))}",
            f"cache_dir={os.path.abspath(cache_dir)}",
        ]
    )
    digest = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:16]
    filename = f"{source}_{int(season)}_{digest}.pkl"
    return os.path.join(_runtime_ml_cache_dir(cache_dir), filename)


def _runtime_ml_cache_ttl(season: int, source: str) -> Optional[timedelta]:
    if source != "live_ncaa_tournament":
        return None
    if int(season) < current_season():
        return None
    return timedelta(hours=RUNTIME_ML_LIVE_TTL_HOURS)


def _load_runtime_ml_cache(
    path: str,
    ttl: Optional[timedelta],
) -> Optional[RuntimeMLBootstrapResult]:
    if not os.path.exists(path):
        return None

    if ttl is not None:
        try:
            modified = datetime.fromtimestamp(os.path.getmtime(path))
        except OSError:
            return None
        if datetime.now() - modified > ttl:
            return None

    try:
        cached = pd.read_pickle(path)
    except Exception:
        return None

    if not isinstance(cached, RuntimeMLBootstrapResult):
        return None
    if not isinstance(cached.training_df, pd.DataFrame):
        return None
    if not isinstance(cached.predictor, MLPredictor):
        return None
    return cached


def _save_runtime_ml_cache(path: str, result: RuntimeMLBootstrapResult) -> None:
    try:
        pd.to_pickle(result, path)
    except Exception:
        return


def build_runtime_ml_predictor(
    season: int,
    blend_weight: float = DEFAULT_ML_BLEND_WEIGHT,
    total_blend_weight: float = DEFAULT_TOTAL_BLEND_WEIGHT,
    cbbd_key: Optional[str] = None,
    cache_dir: str = "data/training_cache",
    min_live_rows: int = MIN_LIVE_TRAINING_ROWS,
    use_live_tournament_fit: bool = False,
    status: Optional[Callable[[str], None]] = None,
) -> RuntimeMLBootstrapResult:
    """
    Train the runtime ML predictor from live historical data when possible,
    otherwise fall back to the synthetic historical builder.
    """
    seasons = list(range(TRAINING_START_SEASON, int(season) + 1))
    if cbbd_key is None:
        cbbd_key = os.environ.get("CBBD_API_KEY", "")
    training_df: Optional[pd.DataFrame] = None
    source = "synthetic"

    def train_from_df(df: pd.DataFrame, trained_source: str) -> RuntimeMLBootstrapResult:
        predictor = MLPredictor(
            blend_weight=blend_weight,
            total_blend_weight=total_blend_weight,
        )
        predictor.train(df)
        result = RuntimeMLBootstrapResult(
            predictor=predictor,
            training_df=df,
            source=trained_source,
        )
        cache_path = _runtime_ml_cache_path(
            cache_dir=cache_dir,
            source=trained_source,
            season=season,
            blend_weight=blend_weight,
            total_blend_weight=total_blend_weight,
            min_live_rows=min_live_rows,
            use_live_tournament_fit=use_live_tournament_fit,
        )
        _save_runtime_ml_cache(cache_path, result)
        return result

    if use_live_tournament_fit:
        live_cache_path = _runtime_ml_cache_path(
            cache_dir=cache_dir,
            source="live_ncaa_tournament",
            season=season,
            blend_weight=blend_weight,
            total_blend_weight=total_blend_weight,
            min_live_rows=min_live_rows,
            use_live_tournament_fit=use_live_tournament_fit,
        )
        cached_live = _load_runtime_ml_cache(
            path=live_cache_path,
            ttl=_runtime_ml_cache_ttl(season, "live_ncaa_tournament"),
        )
        if cached_live is not None:
            _emit(
                status,
                "  ⚡  Loaded cached live NCAA tournament model "
                f"({len(cached_live.training_df)} games, {_season_summary(cached_live.training_df)})",
            )
            return cached_live

        try:
            _emit(status, "  📡  Loading NCAA tournament training data from SQLite cache / APIs...")
            live_builder = LiveTrainingBuilder()
            training_df = live_builder.build(
                cbbd_key=cbbd_key,
                seasons=seasons,
                cache_dir=cache_dir,
            )
            if len(training_df) < min_live_rows:
                raise ValueError(f"Only {len(training_df)} rows")

            source = "live_ncaa_tournament"
            _emit(
                status,
                "  ✅  Live training data: "
                f"{len(training_df)} real tournament games ({_season_summary(training_df)})",
            )
        except Exception as exc:
            if cbbd_key:
                _emit(status, f"  ⚠️  Live training failed ({exc}), using synthetic data...")
            else:
                _emit(status, "  ℹ️  No CBBD key/cache coverage - using synthetic training data")
                _emit(status, "      (Set CBBD_API_KEY in .env for live historical training)")

    if training_df is None:
        synthetic_cache_path = _runtime_ml_cache_path(
            cache_dir=cache_dir,
            source="synthetic",
            season=season,
            blend_weight=blend_weight,
            total_blend_weight=total_blend_weight,
            min_live_rows=min_live_rows,
            use_live_tournament_fit=use_live_tournament_fit,
        )
        cached_synthetic = _load_runtime_ml_cache(
            path=synthetic_cache_path,
            ttl=_runtime_ml_cache_ttl(season, "synthetic"),
        )
        if cached_synthetic is not None:
            _emit(
                status,
                "  ⚡  Loaded cached synthetic model "
                f"({len(cached_synthetic.training_df)} games, {_season_summary(cached_synthetic.training_df)})",
            )
            return cached_synthetic

        if use_live_tournament_fit:
            _emit(status, "  ℹ️  Falling back to generic synthetic training data")
        else:
            _emit(status, "  ℹ️  NCAA tournament fit disabled - using generic synthetic training data")
        builder = HistoricalDataBuilder()
        training_df = builder.build_synthetic_training_set(n_seasons=len(seasons))
        source = "synthetic"

    return train_from_df(training_df, source)
