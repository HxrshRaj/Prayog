"""Backfill MLflow (SQLite, ./mlflow.db) from every experiments/*/results.json.  Run: python scripts/log_mlflow.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prayog.common import log_mlflow

for p in sorted((ROOT / "experiments").glob("*/results.json")):
    res = json.loads(p.read_text())
    log_mlflow(p.parent.name, res["runs"])
    print("logged", p.parent.name, len(res["runs"]), "runs")
