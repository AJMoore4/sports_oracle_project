# Sports Oracle — API Reference

Complete reference for all public classes and methods.

---

## Pipeline

### `DataPipeline`

The main orchestrator. Fetches data from all sources and assembles model-ready inputs.

```python
from sports_oracle.collectors.pipeline import DataPipeline

pipeline = DataPipeline(
    cbbd_key="",      # CBBD API key (optional)
    odds_key="",      # Odds API key (optional)
    season=2025,      # Default season (auto-detected if omitted)
)
```

#### `get_game_inputs(home_team, away_team, ...)`

Assembles all inputs for the prediction engine.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `home_team` | str | required | Team name (any format — auto-resolved) |
| `away_team` | str | required | Team name |
| `season` | int | current | NCAA season year |
| `tournament_round` | int | 1 | 0=FirstFour, 1=First, 2=Second, 3=Sweet16, 4=Elite8, 5=FinalFour, 6=Championship |
| `venue_id` | int | None | CBBD venue ID |
| `venue_name` | str | None | Venue name for geo lookup |
| `game_date` | str | today | 'YYYY-MM-DD' |
| `home_seed` | int | None | Tournament seed (1-16) |
| `away_seed` | int | None | Tournament seed (1-16) |

Returns: `dict` — validated inputs ready for `PredictionEngine.predict()`.

#### `get_team_efficiency(team, season)` → `dict`

Layer 1 efficiency profile. Keys: `adj_oe`, `adj_de`, `adj_tempo`, `efg_pct_off`, `efg_pct_def`, `to_rate_off`, `to_rate_def`, `three_pt_rate_off`, `three_pt_pct_off`, `fta_rate_off`, `orb_pct`, `drb_pct`, `sos`, `rank`, `barthag`.

#### `get_venue_profile(venue_id, venue_name, seasons)` → `dict`

Layer 2 venue analysis. Keys: `vsi`, `vpi`, `v3p`, `sample_size`, `rounds`.

#### `health_check()` → `dict`

Tests connectivity to all data sources. Returns `{"barttorvik": "OK", "espn": "OK", ...}`.

#### `get_scoreboard_with_historical_lines(date=None, groups="50")` → `DataFrame`

Fetches the ESPN scoreboard, then applies market lines with Odds API as the primary source. Upcoming games prefer current Odds API consensus, completed games prefer historical Odds API consensus, and CBBD is used as a fallback for finals if Odds API has no archived line. Original ESPN values are preserved in `espn_betting_spread`, `espn_over_under`, and `espn_odds_detail`.

---

## Prediction Engine

### `PredictionEngine`

Core formula engine. Layer 3 + Layer 4.

```python
from sports_oracle.engine.prediction_engine import PredictionEngine

engine = PredictionEngine()
result = engine.predict(game_inputs)
```

#### `predict(inputs)` → `PredictionResult`

Main entry point. Takes pipeline output, returns full prediction.

#### `predict_batch(games)` → `list[PredictionResult]`

Predict multiple games.

### `PredictionResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `home_team` | str | Home team name |
| `away_team` | str | Away team name |
| `home_score` | float | Projected home score |
| `away_score` | float | Projected away score |
| `spread` | float | Spread (negative = home favored) |
| `total` | float | Projected total points |
| `home_win_prob` | float | Home win probability (0-1) |
| `away_win_prob` | float | Away win probability (0-1) |
| `confidence` | str | "VERY HIGH", "HIGH", "MEDIUM", "LOW", "VERY LOW" |
| `confidence_score` | float | Confidence as float (0-1) |
| `game_pace` | float | Projected possessions |
| `raw_margin` | float | Layer 3 margin before adjustments |
| `momentum_adj` | float | Momentum adjustment (pts) |
| `experience_adj` | float | Experience adjustment (pts) |
| `rest_adj` | float | Rest adjustment (pts) |
| `injury_adj` | float | Injury adjustment (pts) |
| `seed_adj` | float | Seed history adjustment (pts) |
| `travel_adj` | float | Travel/altitude adjustment (pts) |
| `total_adjustment` | float | Sum of all Layer 4 adjustments |
| `market_spread` | float? | Vegas spread (if available) |
| `spread_edge` | float? | Our spread - market spread |
| `predicted_winner` | str | Team with >50% win probability |
| `winner_prob` | float | Winner's win probability |

Methods: `summary()` → one-line string, `breakdown()` → multi-line detailed report.

---

## ML Model

### `MLPredictor`

```python
from sports_oracle.engine.ml_model import MLPredictor

predictor = MLPredictor(blend_weight=0.35)
predictor.train(training_df)
enhanced = predictor.enhance_prediction(formula_result, game_inputs)
```

#### `train(df, cv_folds=5)` → `dict[str, ModelMetrics]`

Trains margin, total, and win probability models. Returns metrics for each.

#### `predict(features)` → `MLResult`

Predict from a flat feature dict. Returns `MLResult(ml_margin, ml_total, ml_win_prob, confidence, feature_importance)`.

#### `enhance_prediction(formula_result, game_inputs)` → `PredictionResult`

Blends ML predictions into an existing formula result. Modifies and returns the `PredictionResult`.

#### `report()` → `str`

Human-readable training report with metrics and feature importance.

---

## Utilities

### `TeamResolver`

```python
from sports_oracle.utils.team_resolver import TeamResolver, resolve_team

resolver = TeamResolver()     # 356 teams, 1084 aliases
resolver.resolve("Duke Blue Devils")  # → "Duke"
resolver.resolve("UConn")            # → "Connecticut"
resolver.resolve("Zags")             # → "Gonzaga"

# Convenience function (uses singleton):
resolve_team("Michigan State Spartans")  # → "Michigan St."
```

### `DataValidator`

```python
from sports_oracle.utils.data_validator import DataValidator

validator = DataValidator(strict=False)
clean_value, report = validator.validate_value(999, "adj_oe")
# → (135.0, ValidationReport(clamped=1))

clean_inputs, reports = validator.validate_game_inputs(raw_inputs)
```

### `GeoLookup`

```python
from sports_oracle.utils.geo import GeoLookup

geo = GeoLookup()
dist = geo.travel_distance("Duke", "Madison Square Garden")  # → 423.4
alt = geo.altitude_diff("Florida", "Delta Center")           # → 4126
ctx = geo.travel_context("Gonzaga", "Caesars Superdome")
# → {"travel_distance_miles": 1896.1, "altitude_diff_ft": -1917, ...}
```

### `SeedHistory`

```python
from sports_oracle.utils.seed_history import SeedHistory

sh = SeedHistory(decay_lambda=0.10)
sh.get_win_rate(1, 16)           # → 0.963 (recency-weighted)
sh.get_upset_rate(5, 12)         # → 0.246
sh.get_seed_adjustment(1, 16)    # → +1.85 (pts, favors 1-seed)
sh.get_matchup_context(5, 12)    # → full context dict
```

---

## Backtesting

### `HistoricalDataBuilder`

```python
from sports_oracle.backtest.historical_data import HistoricalDataBuilder

builder = HistoricalDataBuilder()
df = builder.build_synthetic_training_set(n_seasons=14)
# → DataFrame with 868 rows, 63 columns, 51 ML features
```

### `LiveTrainingBuilder`

```python
from sports_oracle.backtest.live_training import LiveTrainingBuilder

builder = LiveTrainingBuilder()
df = builder.build(
    cbbd_key="your_key",
    seasons=list(range(2015, 2027)),
    cache_dir="data/training_cache",
)
```

Builds real historical tournament training data from CBBD + BartTorvik. When `cache_dir` is provided, finalized seasons are stored in `data/training_cache/live_training.sqlite` and reused on later runs; only uncached/current-season data is fetched again.

### `Evaluator`

```python
from sports_oracle.backtest.evaluator import Evaluator

evaluator = Evaluator()
report = evaluator.evaluate_from_training(df)
print(report.summary())

# Compare models:
comparison = evaluator.compare_models(df, {
    "Formula": ("formula_win_prob", "formula_margin", "formula_total"),
    "ML": ("ml_win_prob", "ml_margin", "ml_total"),
})
```

---

## Collectors

All collectors inherit from `BaseClient` which provides retry logic, rate limiting, and error handling.

| Collector | Class | Key? | Delay |
|-----------|-------|:----:|:-----:|
| BartTorvik | `BartTorvik` | No | 3.5s |
| ESPN | `ESPNCollector` | No | 1.0s |
| CBBD | `CBBDCollector` | Yes | 0.5s |
| Sports Reference | `SportsRefCollector` | No | 3.5s |
| Odds API | `OddsCollector` | Yes | — |
| NCAA API | `NCAACollector` | No | 2.0s |
