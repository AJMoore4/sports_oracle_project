#!/usr/bin/env python3
"""
setup_project.py

Run this ONCE after downloading all files from Claude.
It verifies the project structure and checks dependencies.

Usage:
    python setup_project.py
"""

import os
import sys
import subprocess


def check_file(path, required=True):
    exists = os.path.isfile(path)
    icon = "✅" if exists else ("❌" if required else "⚠️ ")
    status = "found" if exists else ("MISSING" if required else "optional")
    print(f"  {icon} {path:<50s} {status}")
    return exists


def main():
    print("\n" + "=" * 60)
    print("  Sports Oracle — Project Setup")
    print("=" * 60)

    # ── Check project structure ──────────────────────────────────
    print("\n📁 Checking project structure...\n")

    all_good = True
    required_files = [
        "sports_oracle/__init__.py",
        "sports_oracle/collectors/__init__.py",
        "sports_oracle/collectors/config.py",
        "sports_oracle/collectors/barttorvik_collector.py",
        "sports_oracle/collectors/cbbd_collector.py",
        "sports_oracle/collectors/espn_collector.py",
        "sports_oracle/collectors/sportsref_collector.py",
        "sports_oracle/collectors/odds_collector.py",
        "sports_oracle/collectors/ncaa_collector.py",
        "sports_oracle/collectors/pipeline.py",
        "sports_oracle/engine/__init__.py",
        "sports_oracle/engine/prediction_engine.py",
        "sports_oracle/engine/ml_model.py",
        "sports_oracle/utils/__init__.py",
        "sports_oracle/utils/team_resolver.py",
        "sports_oracle/utils/data_validator.py",
        "sports_oracle/utils/geo.py",
        "sports_oracle/utils/seed_history.py",
        "sports_oracle/backtest/__init__.py",
        "sports_oracle/backtest/historical_data.py",
        "sports_oracle/backtest/evaluator.py",
    ]

    # Create __init__.py files if missing
    init_dirs = [
        "sports_oracle",
        "sports_oracle/collectors",
        "sports_oracle/engine",
        "sports_oracle/utils",
        "sports_oracle/backtest",
    ]
    for d in init_dirs:
        os.makedirs(d, exist_ok=True)
        init_path = os.path.join(d, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                f.write("")
            print(f"  📝 Created {init_path}")

    for f in required_files:
        if not check_file(f):
            all_good = False

    # Optional files
    check_file("run_prediction.py", required=False)
    check_file("run_server.py", required=False)
    check_file(".env", required=False)

    # ── Check dependencies ──────────────────────────────────────
    print("\n📦 Checking Python dependencies...\n")

    deps = {
        "pandas": "pandas",
        "requests": "requests",
        "bs4": "beautifulsoup4",
        "lxml": "lxml",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
    }
    optional_deps = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
    }

    missing = []
    for import_name, pip_name in deps.items():
        try:
            __import__(import_name)
            print(f"  ✅ {pip_name}")
        except ImportError:
            print(f"  ❌ {pip_name} — MISSING")
            missing.append(pip_name)

    for import_name, pip_name in optional_deps.items():
        try:
            __import__(import_name)
            print(f"  ✅ {pip_name} (optional)")
        except ImportError:
            print(f"  ⚠️  {pip_name} (optional — needed for API server)")

    if missing:
        print(f"\n  Install missing dependencies:")
        print(f"    pip install {' '.join(missing)}")
        all_good = False

    # ── Check API keys ──────────────────────────────────────────
    print("\n🔑 Checking API keys...\n")

    # Load .env if present
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

    keys = {
        "CBBD_API_KEY": ("CBBD (venue data)", "collegebasketballdata.com/key"),
        "ODDS_API_KEY": ("Odds API (betting lines)", "the-odds-api.com"),
    }
    for key_name, (desc, signup_url) in keys.items():
        val = os.environ.get(key_name, "")
        if val:
            print(f"  ✅ {key_name} — configured")
        else:
            print(f"  ⚠️  {key_name} — not set ({desc})")
            print(f"       Sign up free: {signup_url}")

    print(f"\n  The engine works without API keys (uses BartTorvik + ESPN).")
    print(f"  Keys add venue analysis and betting line comparison.\n")

    # ── Quick import test ───────────────────────────────────────
    if all_good:
        print("🧪 Testing imports...\n")
        try:
            from sports_oracle.collectors.pipeline import DataPipeline
            from sports_oracle.engine.prediction_engine import PredictionEngine
            from sports_oracle.engine.ml_model import MLPredictor
            from sports_oracle.utils.team_resolver import resolve_team
            print("  ✅ All imports successful")
            print(f"  ✅ resolve_team('Duke Blue Devils') → '{resolve_team('Duke Blue Devils')}'")
        except Exception as e:
            print(f"  ❌ Import error: {e}")
            all_good = False

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_good:
        print("  ✅ Setup complete! Ready to run.")
        print()
        print("  NEXT STEPS:")
        print("    1. Run predictions:  python run_prediction.py")
        print("    2. Start API server: python run_server.py")
        print("    3. Open Swagger UI:  http://localhost:8000/docs")
    else:
        print("  ⚠️  Setup incomplete — fix the issues above first.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
