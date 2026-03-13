# Sports Oracle v1 — Final Architecture Spec

**Decisions Locked — March 11, 2026**

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Layer 3 | Matchup Projection Engine (user spec) | Steps 1-3: Game_Pace → Raw Scores → Margin |
| Layer 4 stacking | Additive | Simple, interpretable: `Final = Raw_Margin + adjustments` |
| ML layer | Predict outcome directly | Formula outputs become features; model has maximum freedom |
| Seed history | Dynamic from CBBD/SportsRef | Recency-weighted, more accurate than static lookup |
| New data sources | Odds API + NCAA API + travel + altitude | Full feature set for ML layer |

---

## File Map

```
sports_oracle/
├── collectors/
│   ├── config.py                  ✅ EXISTS (bug-fixed)
│   ├── barttorvik_collector.py    ✅ EXISTS (bug-fixed)
│   ├── cbbd_collector.py          ✅ EXISTS (bug-fixed)
│   ├── espn_collector.py          ✅ EXISTS (bug-fixed)
│   ├── sportsref_collector.py     ✅ EXISTS (unchanged)
│   ├── pipeline.py                ✅ EXISTS (bug-fixed, needs VPI/V3P)
│   ├── odds_collector.py          🆕 NEW — The Odds API
│   └── ncaa_collector.py          🆕 NEW — NCAA API
│
├── engine/
│   ├── prediction_engine.py       🆕 NEW — Layer 3 + Layer 4
│   ├── ml_model.py                🆕 NEW — ML adjustment layer
│   ├── bracket_predictor.py       🆕 NEW — 63-game simulation
│   └── edge_finder.py             🆕 NEW — market edge scanner
│
├── utils/
│   ├── team_resolver.py           🆕 NEW — canonical name mapping
│   ├── data_validator.py          🆕 NEW — sanity bounds
│   ├── geo.py                     🆕 NEW — travel distance + altitude
│   └── seed_history.py            🆕 NEW — dynamic seed rates
│
├── backtest/
│   ├── historical_data.py         🆕 NEW — training data assembly
│   └── evaluator.py               🆕 NEW — accuracy metrics
│
└── docs/
    ├── README.md                  🆕 NEW
    ├── ARCHITECTURE.md            🆕 NEW
    ├── FORMULAS.md                🆕 NEW
    └── API_REFERENCE.md           🆕 NEW
```

---

## Data Flow

```
 BartTorvik ──┐    ESPN ──┐    CBBD ──┐    SportsRef ──┐
 (efficiency) │  (roster) │  (venue) │    (coach/hist) │
              ▼           ▼          ▼                 ▼
 Odds API ──┐  NCAA API ─┐  Geo ───┐  Seed History ──┐
 (lines)    │  (bracket) │  (dist) │  (dynamic rates) │
            └──────┬─────┴────┬────┴────────┬─────────┘
                   ▼          ▼             ▼
            TeamResolver + DataValidator
                        │
                   pipeline.py (assembles all inputs)
                        │
              ┌─────────▼──────────┐
              │  LAYER 3: Matchup  │
              │  Step 1: Game_Pace │
              │  Step 2: Raw Score │
              │  Step 3: Margin    │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │  LAYER 4: Additive │
              │  + momentum        │
              │  + experience      │
              │  + rest            │
              │  + injury          │
              │  + seed history    │
              │  + travel/altitude │
              │  = Formula outputs │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │  ML LAYER          │
              │  Features: formula │
              │    + raw + context │
              │  Predicts: margin, │
              │    total, win_prob │
              └─────────┬──────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     Bracket       Game Predict   Edge Finder
     (63 games)    (detailed)     (vs Vegas)
```

---

## Layer 3 — Matchup Projection Engine

### Step 1: Expected Possessions

```
Raw_Pace  = (Team_A_Pace + Team_B_Pace) / 2
Game_Pace = Raw_Pace × VPI × Round_Modifier

Round_Modifier:
  First/Second Round  → 1.00
  Sweet 16            → 0.97
  Elite 8             → 0.94
  Final Four          → 0.96
  Championship        → 0.95
```

### Step 2: Raw Score Projection

```
Team_A_Score = (Team_A_AdjOE × (100 / Team_B_AdjDE))
             × (Game_Pace / 100)
             × VSI
             × V3P_adjustment_A

Team_B_Score = (Team_B_AdjOE × (100 / Team_A_AdjDE))
             × (Game_Pace / 100)
             × VSI
             × V3P_adjustment_B

V3P_adjustment = 1.0 + ((V3P - 1.0) × 3PA_Rate × 1.5)
```

### Step 3: Projected Total & Margin

```
Projected_Total = Team_A_Score + Team_B_Score
Raw_Margin      = Team_A_Score - Team_B_Score
```

---

## Layer 4 — Additive Adjustments

```
Formula_Margin = Raw_Margin
               + momentum_adj
               + experience_adj
               + rest_adj
               + injury_adj
               + seed_adj
               + travel_adj
```

Each adjustment is a signed float (positive favors Team A).

---

## ML Feature Vector (~62 features)

| Category | Count | Examples |
|----------|-------|---------|
| Formula outputs | 6 | formula_margin, formula_total, formula_win_prob, game_pace |
| Layer 1 raw (×2 teams) | ~30 | adj_oe, adj_de, efg_off/def, to_rate, 3pt splits, sos, barthag |
| Layer 2 venue | 4 | vsi, vpi, v3p, sample_size |
| Layer 4 context (×2 teams) | ~16 | momentum, experience, coach record, rest, injuries |
| Matchup-specific | 6 | seed_diff, upset_base_rate, rank_diff, travel_diff, altitude_diff, round |

Training set: ~600 tournament games (2010–2025, ~40/year)

---

## Build Order

| Phase | Files | Depends On |
|-------|-------|-----------|
| 1. Data Layer | team_resolver, data_validator, geo, seed_history, odds_collector, ncaa_collector, pipeline updates | Bug-fixed collectors |
| 2. Engine | prediction_engine.py | Phase 1 |
| 3. ML | historical_data, ml_model, evaluator | Phase 2 |
| 4. Outputs | bracket_predictor, edge_finder | Phase 3 |
| 5. Docs | README, ARCHITECTURE, FORMULAS, API_REFERENCE | All phases |
