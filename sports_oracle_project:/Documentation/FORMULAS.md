# Sports Oracle — Formula Reference

Complete mathematical documentation for every formula in the prediction engine.

---

## Layer 3: Matchup Projection Engine

### Step 1 — Expected Possessions

```
Raw_Pace   = (Team_A_Tempo + Team_B_Tempo) / 2
Game_Pace  = Raw_Pace × VPI × Round_Modifier
```

**VPI** (Venue Pace Index): How much a specific venue speeds up or slows down games relative to the national average. Computed from era-adjusted historical games at that venue with exponential decay weighting (λ=0.15).

**Round Modifiers:**

| Round | Modifier | Rationale |
|-------|:---:|-----------|
| First Four | 1.00 | Baseline |
| First Round | 1.00 | Baseline |
| Second Round | 1.00 | Baseline |
| Sweet 16 | 0.97 | Defenses tighten with preparation time |
| Elite 8 | 0.94 | Lowest scoring round — elite defensive matchups |
| Final Four | 0.96 | Slight uptick — elite offenses that survived |
| Championship | 0.95 | Championship pressure slightly depresses pace |

### Step 2 — Raw Score Projection

```
Score_A = (AdjOE_A × AdjDE_B / 100) × (Game_Pace / 100) × VSI × V3P_adj_A
Score_B = (AdjOE_B × AdjDE_A / 100) × (Game_Pace / 100) × VSI × V3P_adj_B
```

**Why `AdjOE × AdjDE / 100`:**
Both AdjOE and AdjDE are calibrated to "points per 100 possessions against an average D1 team" with a national average of ~100. Multiplying them and dividing by 100 produces the expected scoring rate for a specific offense against a specific defense:

- Duke AdjOE=120 vs Vermont AdjDE=104: `120 × 104 / 100 = 124.8` pts per 100 poss
- Vermont AdjOE=105 vs Duke AdjDE=92: `105 × 92 / 100 = 96.6` pts per 100 poss

A good defense (low AdjDE) correctly suppresses the opponent's scoring.

**VSI** (Venue Scoring Index): Era-adjusted ratio of venue scoring to national average. `VSI=1.05` means 5% more scoring than average at this venue.

**V3P Adjustment:**

```
V3P_adj = 1.0 + ((V3P - 1.0) × 3PA_Rate × 1.5)
```

3PT-heavy teams feel venue effects more. If a venue suppresses 3PT shooting (`V3P=0.95`) and a team takes 42% of shots from 3 (`3PA_Rate=0.42`):

```
V3P_adj = 1.0 + ((0.95 - 1.0) × 0.42 × 1.5) = 1.0 + (-0.0315) = 0.9685
```

That team scores ~3% less than a balanced team at the same venue.

### Step 3 — Raw Margin & Total

```
Raw_Margin = Score_A - Score_B    (positive = Team A favored)
Raw_Total  = Score_A + Score_B
```

---

## Layer 4: Additive Adjustments

All adjustments are signed floats from Team A's perspective (positive favors Team A). They are summed and added to Raw_Margin.

```
Final_Margin = Raw_Margin + Σ(adjustments)
```

Total adjustment is capped at ±8.0 points.

### Momentum Adjustment (±2.0 pts)

```
momentum_score(team) = clamp(weighted_avg_margin / 10, -2, +2) + efficiency_bonus

weighted_avg_margin = Σ(margin_i × e^(-0.15 × i)) / Σ(e^(-0.15 × i))
    where i = 0 for most recent game, i = 9 for 10th most recent

efficiency_bonus = clamp((weighted_avg - expected_margin) / 10, -0.5, +0.5)
    where expected_margin = (season_AdjOE - season_AdjDE) / 5

momentum_adj = (home_score - away_score) × 1.0
```

### Experience Adjustment (±1.5 pts)

```
experience_score(team) = 0.30 × roster_age + 0.35 × coach_score + 0.35 × returning_score

roster_age = clamp((avg_class_year - 2.5) / 1.5, -2, +2)
    Fr=1, So=2, Jr=3, Sr=4, Gr=5

coach_score = clamp(min(1, appearances/10) × (win_rate - 0.45) / 0.25, -2, +2)
    First-year tournament coach: -0.8

returning_score = clamp((returning_pct - 0.5) / 0.2, -2, +2)

experience_adj = (home_score - away_score) × 0.75
```

### Rest Adjustment (±1.5 pts)

| Days Rest | Adjustment |
|:---------:|:----------:|
| 0 (B2B) | -1.5 |
| 1 | -0.5 |
| 2 | 0.0 |
| 3 | +0.3 |
| 4 | +0.4 |
| 5 | +0.3 |
| 6 | +0.2 |
| 7 | +0.1 |
| 8+ | Decreasing (rust) |

```
rest_adj = rest_value(home_days) - rest_value(away_days)
```

### Injury Adjustment (±4.0 pts per team)

```
player_impact = severity × position_weight × 1.2

severity:
  Out         → 1.0
  Doubtful    → 0.7
  Questionable → 0.3
  Day-to-Day  → 0.2

position_weight:
  PG/SG/G     → 0.9
  SF/PF/F     → 0.7
  C           → 0.6

team_impact = max(-4.0, Σ(player_impacts))    [always negative]
injury_adj  = home_impact - away_impact
```

### Seed History Adjustment (±2.0 pts)

```
seed_adj = (win_rate - 0.5) × 4.0 × direction

win_rate = recency-weighted historical rate for this seed matchup
    weighted by e^(-0.10 × age)

direction = +1 if home is higher seed, -1 if away is higher seed
```

### Travel Adjustment (±1.5 pts)

```
distance_penalty:
  < 500 miles:  0.0
  ≥ 500 miles:  -0.4 × (excess_miles / 1000)

altitude_penalty:
  < 2000 ft:    0.0
  ≥ 2000 ft:    -0.2 × (excess_ft / 1000)

travel_adj = home_penalty - away_penalty    [each penalty is ≤ 0]
```

---

## Win Probability

Logistic function of final margin:

```
P(home wins) = 1 / (1 + e^(-margin / σ))
σ = 10.5
```

| Margin | Win Prob |
|:------:|:--------:|
| 0 | 50.0% |
| 3 | 57.1% |
| 5 | 62.0% |
| 7 | 66.1% |
| 10 | 72.1% |
| 15 | 80.7% |
| 20 | 87.1% |
| 25 | 91.5% |

---

## ML Layer

**Models:** Ridge Regression (margin, total), Logistic Regression (win probability)

**Blend formula:**

```
final_value = (1 - w) × formula_value + w × ml_value
w = 0.35    (65% formula, 35% ML)
```

**Feature vector:** 51 features including formula outputs (most important), raw efficiency stats for both teams, differentials, venue indices, Layer 4 adjustments, matchup context (seeds, rest, round), and experience metrics.

**Regularization:** α=10.0 (Ridge), C=0.1 (Logistic) — strong regularization to prevent overfitting on ~600 training samples.

---

## Venue Index Calculations

All three venue indices use the same framework: era-adjusted ratios with exponential decay and bubble-season downweighting.

```
For each historical game at the venue:
  game_ratio   = game_metric / national_average_that_season
  decay_weight = e^(0.15 × (season - oldest_season))
  bubble_weight = 0.05 if season == 2021, else 1.0
  weight = decay_weight × bubble_weight

Index = Σ(game_ratio × weight) / Σ(weight)
```

| Index | Metric | Interpretation |
|-------|--------|---------------|
| VSI | Total points / national avg total | >1.0 = more scoring |
| VPI | Game tempo / national avg tempo | >1.0 = faster pace |
| V3P | 3PT% / national avg 3PT% | <1.0 = suppresses 3PT |
