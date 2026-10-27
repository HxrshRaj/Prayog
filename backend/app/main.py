"""Prayog API: experiment registry, MLflow view, classifier inference, and the built dashboard."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("PRAYOG_ROOT", Path(__file__).resolve().parents[2]))
EXPERIMENTS = ROOT / "experiments"
DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="Prayog API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def registry() -> dict[str, dict]:
    """experiment name -> {results, card}. Only experiments that have a saved results.json are listed."""
    out = {}
    for d in sorted(EXPERIMENTS.iterdir()) if EXPERIMENTS.exists() else []:
        if not d.is_dir():
            continue
        results, card = _read(d / "results.json"), _read(d / "card.json")
        if results or card:
            out[d.name] = {"results": results, "card": card}
    return out


@app.get("/api/health")
def health():
    return {"status": "ok", "experiments": len(registry())}


@app.get("/api/experiments")
def experiments():
    items = []
    for name, e in registry().items():
        card, res = e["card"] or {}, e["results"] or {}
        items.append({"name": name, "title": card.get("title", name), "paper": card.get("paper"),
                      "verdict": card.get("verdict", "not run"), "has_results": res is not None and bool(res),
                      "n_runs": len(res.get("runs", [])) if res else 0,
                      "git_commit": (res.get("env") or {}).get("git_commit") if res else None})
    return items


@app.get("/api/experiments/{name}")
def experiment(name: str):
    e = registry().get(name)
    if not e:
        raise HTTPException(404, f"unknown experiment {name}")
    return e


@app.get("/api/runs")
def runs(experiment: str | None = None):
    rows = []
    for name, e in registry().items():
        res = e["results"]
        if not res or (experiment and name != experiment):
            continue
        env = res.get("env", {})
        for r in res["runs"]:
            rows.append({"experiment": name, "variant": r["variant"], "seed": r["seed"],
                         "metrics": r["metrics"], "git_commit": env.get("git_commit"),
                         "libraries": env.get("libraries"), "timestamp": env.get("timestamp")})
    return rows


@app.get("/api/curves/{name}")
def curves(name: str):
    """Mean +- std across seeds, per variant, for experiments that saved learning curves."""
    import numpy as np
    e = registry().get(name)
    if not e or not e["results"]:
        raise HTTPException(404, "no results")
    by_variant: dict[str, list] = {}
    for r in e["results"]["runs"]:
        if r.get("curve"):
            by_variant.setdefault(r["variant"], []).append(np.array(r["curve"]))
    out = {}
    for v, cs in by_variant.items():
        arr = np.stack([c[:, 1] for c in cs])
        out[v] = {"x": cs[0][:, 0].tolist(), "mean": arr.mean(0).tolist(),
                  "std": (arr.std(0, ddof=1) if len(cs) > 1 else np.zeros(arr.shape[1])).tolist(),
                  "seeds": len(cs), "per_seed": arr.tolist()}
    return out


@app.get("/api/mlflow")
def mlflow_runs():
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        store = ROOT / "mlflow.db"
        if not store.exists():
            return []
        client = MlflowClient(tracking_uri=f"sqlite:///{store.as_posix()}")
        rows = []
        for exp in client.search_experiments():
            for r in client.search_runs([exp.experiment_id], max_results=500):
                rows.append({"experiment": exp.name, "run": r.info.run_name, "metrics": r.data.metrics,
                             "params": r.data.params})
        return rows
    except Exception as e:  # MLflow is optional on the hosted instance
        return {"error": str(e)}


class PredictIn(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def _classifier():
    import torch
    from peft import PeftModel
    from transformers import BertForSequenceClassification, BertTokenizerFast
    adapter = ROOT / "models" / "lora_sst2"
    if not adapter.exists():
        raise FileNotFoundError("LoRA adapter not found; run experiments/lora_vs_full/run.py first")
    base_name = "prajjwal1/bert-tiny"
    tok = BertTokenizerFast.from_pretrained(base_name)
    base = BertForSequenceClassification.from_pretrained(base_name, num_labels=2)
    model = PeftModel.from_pretrained(base, str(adapter)).eval()
    return tok, model, torch


@app.post("/api/predict")
def predict(body: PredictIn):
    try:
        tok, model, torch = _classifier()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    enc = tok(body.text, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, -1)[0].tolist()
    return {"label": "positive" if probs[1] > probs[0] else "negative",
            "probabilities": {"negative": probs[0], "positive": probs[1]},
            "model": "prajjwal1/bert-tiny + LoRA r=4 (SST-2 subset)"}


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
