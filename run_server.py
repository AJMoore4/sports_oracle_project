#!/usr/bin/env python3
"""
run_server.py

FastAPI server that exposes the prediction engine as a REST API.
The React dashboard can connect to this for live predictions.

SETUP:
  1. Install additional dependency:
       pip install fastapi uvicorn
  2. Run:
       python run_server.py
  3. Open http://localhost:8000/docs for Swagger UI
  4. Dashboard connects to http://localhost:8000/api/predict

ENDPOINTS:
  POST /api/predict        — predict a single game
  POST /api/predict/batch  — predict multiple games
  GET  /api/teams          — list all known teams
  GET  /api/health         — check data source status
"""

import os
import logging
from typing import Optional

logging.basicConfig(level=logging.WARNING)

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    print("\n⚠️  FastAPI not installed. Run:")
    print("     pip install fastapi uvicorn")
    print("   Then re-run this script.\n")
    raise SystemExit(1)

from sports_oracle.collectors.pipeline import DataPipeline
from sports_oracle.engine.prediction_engine import PredictionEngine
from sports_oracle.backtest.training_bootstrap import build_runtime_ml_predictor
from sports_oracle.utils.team_resolver import get_resolver


# ── Initialize everything on startup ─────────────────────────────────────────

print("🏀 Sports Oracle API starting...")

pipeline = DataPipeline(
    cbbd_key=os.environ.get("CBBD_API_KEY", ""),
    odds_key=os.environ.get("ODDS_API_KEY", ""),
    season=2026,
)
engine = PredictionEngine()
resolver = get_resolver()

# Train ML model
print("  Training ML model...")
bootstrap = build_runtime_ml_predictor(
    season=2026,
    cbbd_key=os.environ.get("CBBD_API_KEY", ""),
    cache_dir="data/training_cache",
    status=print,
)
ml = bootstrap.predictor
engine.set_calibration_profile(ml.build_engine_calibration_profile())
print(f"  ✅ ML trained on {len(bootstrap.training_df)} games ({bootstrap.source})")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sports Oracle API",
    description="NCAA Tournament Prediction Engine",
    version="1.0.0",
)

# Allow dashboard to connect from any origin (dev mode)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ───────────────────────────────────────────────────

class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    home_seed: Optional[int] = None
    away_seed: Optional[int] = None
    tournament_round: int = 1
    venue_name: Optional[str] = None
    season: int = 2026

class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    home_score: float
    away_score: float
    spread: float
    total: float
    home_win_prob: float
    away_win_prob: float
    predicted_winner: str
    confidence: str
    confidence_score: float
    game_pace: float
    raw_margin: float
    adjustments: dict
    market_spread: Optional[float] = None
    market_total: Optional[float] = None
    spread_edge: Optional[float] = None
    home_efficiency: dict = Field(default_factory=dict)
    away_efficiency: dict = Field(default_factory=dict)

class BatchRequest(BaseModel):
    games: list[PredictRequest]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/predict", response_model=PredictResponse)
def predict_game(req: PredictRequest):
    """Predict a single game."""
    inputs = pipeline.get_game_inputs(
        home_team=req.home_team,
        away_team=req.away_team,
        tournament_round=req.tournament_round,
        home_seed=req.home_seed,
        away_seed=req.away_seed,
        venue_name=req.venue_name,
        season=req.season,
    )

    result = engine.predict(inputs)
    result = ml.enhance_prediction(result, inputs)

    return PredictResponse(
        home_team=result.home_team,
        away_team=result.away_team,
        home_score=round(result.home_score, 1),
        away_score=round(result.away_score, 1),
        spread=round(result.spread, 1),
        total=round(result.total, 1),
        home_win_prob=round(result.home_win_prob, 4),
        away_win_prob=round(result.away_win_prob, 4),
        predicted_winner=result.predicted_winner,
        confidence=result.confidence,
        confidence_score=round(result.confidence_score, 2),
        game_pace=round(result.game_pace, 1),
        raw_margin=round(result.raw_margin, 1),
        adjustments={
            "momentum": round(result.momentum_adj, 2),
            "experience": round(result.experience_adj, 2),
            "rest": round(result.rest_adj, 2),
            "injury": round(result.injury_adj, 2),
            "seed": round(result.seed_adj, 2),
            "travel": round(result.travel_adj, 2),
            "margin": round(result.total_adjustment, 2),
            "total": round(result.total_points_adjustment, 2),
        },
        market_spread=result.market_spread,
        market_total=result.market_total,
        spread_edge=result.spread_edge,
        home_efficiency=inputs.get("home_efficiency", {}),
        away_efficiency=inputs.get("away_efficiency", {}),
    )


@app.post("/api/predict/batch")
def predict_batch(req: BatchRequest):
    """Predict multiple games."""
    results = []
    for game in req.games:
        try:
            results.append(predict_game(game))
        except Exception as e:
            results.append({"error": str(e), "game": game.dict()})
    return {"predictions": results}


@app.get("/api/teams")
def list_teams():
    """List all known teams with their canonical names."""
    return {
        "teams": resolver.all_teams,
        "count": resolver.team_count,
    }


@app.get("/api/health")
def health_check():
    """Check data source connectivity."""
    return pipeline.health_check()


@app.get("/")
def root():
    return {
        "name": "Sports Oracle API",
        "version": "1.0.0",
        "endpoints": [
            "POST /api/predict",
            "POST /api/predict/batch",
            "GET /api/teams",
            "GET /api/health",
        ],
        "docs": "/docs",
    }


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Sports Oracle API running at http://localhost:8000")
    print("   Swagger docs: http://localhost:8000/docs")
    print("   Press Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
