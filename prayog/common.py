"""Shared helpers: seeding, environment capture, result files, aggregation."""
from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LIBS = ["torch", "transformers", "peft", "trl", "datasets", "stable-baselines3",
        "gymnasium", "numpy", "scikit-learn", "mlflow"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def versions() -> dict:
    out = {}
    for lib in LIBS:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            pass
    return out


def env_info() -> dict:
    return {"git_commit": git_commit(), "python": sys.version.split()[0],
            "platform": platform.platform(), "libraries": versions(),
            "device": "cpu", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}


def mean_std(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": int(len(a))}


def summarise(runs: list[dict]) -> dict:
    """Group runs by `variant` and aggregate every scalar metric across seeds."""
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(r["variant"], []).append(r)
    summary = {}
    for variant, rs in groups.items():
        keys = [k for k, v in rs[0]["metrics"].items() if isinstance(v, (int, float))]
        summary[variant] = {k: mean_std([r["metrics"][k] for r in rs]) for k in keys}
    return summary


def write_results(exp_dir: Path, config: dict, runs: list[dict], extra: dict | None = None,
                  smoke: bool = False) -> dict:
    res = {"experiment": exp_dir.name, "config": config, "env": env_info(),
           "runs": runs, "summary": summarise(runs)}
    if extra:
        res.update(extra)
    (exp_dir / ("results.smoke.json" if smoke else "results.json")).write_text(json.dumps(res, indent=2))
    return res


def load_config(exp_dir: Path, smoke: bool) -> dict:
    import yaml
    cfg = yaml.safe_load((exp_dir / "config.yaml").read_text())
    if smoke:
        cfg.update(cfg.get("smoke", {}))
    cfg.pop("smoke", None)
    return cfg


def log_mlflow(exp_name: str, runs: list[dict]) -> None:
    """Best-effort MLflow logging (file store under ./mlruns)."""
    try:
        import mlflow
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
        mlflow.set_experiment(exp_name)
        for r in runs:
            with mlflow.start_run(run_name=f"{r['variant']}-seed{r['seed']}"):
                mlflow.log_param("variant", r["variant"])
                mlflow.log_param("seed", r["seed"])
                mlflow.log_param("git_commit", git_commit())
                for k, v in r["metrics"].items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
    except Exception as e:  # tracking must never break an experiment
        print(f"[mlflow] skipped: {e}")


def experiment_cli(description: str):
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--smoke", action="store_true", help="1-step CPU smoke run; writes results.smoke.json only")
    return p.parse_args()
