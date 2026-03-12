# Sports Oracle

**NCAA Tournament Prediction Engine**

A hybrid formula + ML system that predicts NCAA men's basketball tournament outcomes using efficiency metrics, venue analysis, contextual factors, and market data.

---

## What It Does

Sports Oracle takes two teams, their stats, and a tournament context, and produces:

- **Projected score** for each team
- **Spread prediction** (point margin)
- **Total prediction** (over/under)
- **Win probability** with confidence level
- **Market edge detection** (our line vs Vegas)

The system works in layers: a KenPom-style formula engine produces a baseline prediction, then an ML model (trained on historical tournament data) refines it.

---

## Architecture

```
BartTorvik → ESPN → CBBD → SportsRef → Odds API → NCAA API
                         ↓
              TeamResolver + DataValidator
                         ↓
                    DataPipeline
                         ↓
              Layer 3: Matchup Projection
              (pace → scores → margin)
                         ↓
              Layer 4: Additive Adjustments
              (momentum, experience, rest,
               injuries, seed history, travel)
                         ↓
                ML Layer (Ridge + Logistic)
                         ↓
            Spread · Total · Win Probability
```

## Quick Start

### 1. Install dependencies

```bash
pip install pandas requests beautifulsoup4 lxml scikit-learn numpy
```

### 2. Set up API keys

Create a `.env` file in the project root:

```env
# Required for full functionality — free tiers available
CBBD_API_KEY=your_key_here        # collegebasketballdata.com/key
ODDS_API_KEY=your_key_here        # the-odds-api.com

# Optional
ANTHROPIC_API_KEY=your_key_here   # For future AI features
```

The system works without API keys (using BartTorvik + ESPN, which are free and keyless), but venue data and betting lines require CBBD and Odds API keys respectively.

### 3. Run a prediction

```python
from sports_oracle.collectors.pipeline import DataPipeline
from sports_oracle.engine.prediction_engine import PredictionEngine

# Initialize
pipeline = DataPipeline(season=2025)
engine = PredictionEngine()

# Fetch data and predict
inputs = pipeline.get_game_inputs(
    home_team="Duke",
    away_team="Vermont",
    tournament_round=1,       # 1=First Round
    home_seed=1,
    away_seed=16,
    venue_name="State Farm Stadium",
)
result = engine.predict(inputs)

# Output
print(result.summary())
# Duke 89% | Duke 87 – Vermont 65 | Spread: -22.1 | Total: 152 | Confidence: VERY HIGH

print(result.breakdown())
# Full layer-by-layer breakdown
```

### 4. Train the ML layer (optional, improves accuracy)

```python
from sports_oracle.backtest.historical_data import HistoricalDataBuilder
from sports_oracle.engine.ml_model import MLPredictor

# Build training data (synthetic for development, live for production)
builder = HistoricalDataBuilder()
df = builder.build_synthetic_training_set(n_seasons=14)

# Train
predictor = MLPredictor(blend_weight=0.35)
predictor.train(df)

# Enhance a formula prediction with ML
enhanced = predictor.enhance_prediction(result, inputs)
print(enhanced.summary())
```

### 5. Evaluate accuracy

```python
from sports_oracle.backtest.evaluator import Evaluator

evaluator = Evaluator()
report = evaluator.evaluate_from_training(df)
print(report.summary())
```

---

## Project Structure

```
sports_oracle/
├── collectors/           # Data collection from external sources
│   ├── config.py         # Shared config, HTTP client, rate limiting
│   ├── barttorvik_collector.py   # T-Rank efficiency data (primary)
│   ├── cbbd_collector.py         # Venue, game results, betting lines
│   ├── espn_collector.py         # Rosters, injuries, scores
│   ├── sportsref_collector.py    # Coach records, historical data
│   ├── odds_collector.py         # The Odds API (betting lines)
│   ├── ncaa_collector.py         # NCAA official bracket/results
│   └── pipeline.py               # Orchestrates all collectors
│
├── engine/               # Prediction engine
│   ├── prediction_engine.py  # Layer 3 + Layer 4 formula engine
│   └── ml_model.py           # ML adjustment layer
│
├── utils/                # Utilities
│   ├── team_resolver.py  # Canonical team name mapping (356 teams)
│   ├── data_validator.py # Sanity bounds on all inputs
│   ├── geo.py            # Travel distance + altitude
│   └── seed_history.py   # Dynamic seed matchup rates
│
└── backtest/             # Backtesting & evaluation
    ├── historical_data.py # Training data assembly
    └── evaluator.py       # Accuracy & calibration metrics
```

---

## Data Sources

| Source | Key Required | Primary Use |
|--------|:---:|-------------|
| BartTorvik | No | Efficiency ratings (AdjOE, AdjDE, tempo, SOS) |
| ESPN | No | Rosters, injuries, scores, conference tournaments |
| CBBD | Yes (free) | Venue data, game results, betting lines |
| Sports Reference | No | Coach records, tournament history |
| The Odds API | Yes (free) | Live betting lines for edge detection |
| NCAA API | No | Official bracket, seeds, results |

---

## Accuracy (Synthetic Backtest)

| Metric | Formula | ML | Blended |
|--------|:---:|:---:|:---:|
| Win Accuracy | 78% | 82% | 80% |
| Margin MAE | 6.9 pts | 6.7 pts | 6.8 pts |
| Calibration Error | 0.093 | 0.037 | 0.086 |
| Upset Detection | 66% | — | — |

These numbers are from synthetic data. Real-world accuracy will be validated once live tournament data is collected.

---

## Key Design Decisions

- **Hybrid formula + ML**: Formula handles the physics of basketball (efficiency, pace, matchup), ML handles the patterns humans miss (contextual interactions, market calibration)
- **Additive Layer 4 adjustments**: Simple, interpretable. Each factor adds or subtracts points from the formula baseline
- **35% ML blend weight**: Conservative. The formula is the backbone; ML fine-tunes
- **Dynamic seed history**: Recency-weighted so recent upsets (UMBC 2018, FDU 2023) increase projected upset rates
- **KenPom-style scoring**: `AdjOE × AdjDE / 100 × Pace/100` — the gold standard for college basketball projection

---

## License

Educational research project. Not for commercial use.
