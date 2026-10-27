import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    exp = tmp_path / "experiments" / "toy"
    exp.mkdir(parents=True)
    runs = [{"variant": "a", "seed": s, "metrics": {"accuracy": 0.5 + s / 10},
             "curve": [[0, 1.0 + s], [10, 2.0 + s]]} for s in range(3)]
    (exp / "results.json").write_text(json.dumps(
        {"experiment": "toy", "env": {"git_commit": "abc1234", "libraries": {}}, "runs": runs, "summary": {}}))
    (exp / "card.json").write_text(json.dumps({"title": "Toy", "paper": "P", "verdict": "reproduced"}))
    monkeypatch.setenv("PRAYOG_ROOT", str(tmp_path))
    for m in [m for m in sys.modules if m.startswith("app")]:
        del sys.modules[m]
    from app.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["experiments"] == 1


def test_experiments_list(client):
    (item,) = client.get("/api/experiments").json()
    assert item["name"] == "toy" and item["verdict"] == "reproduced" and item["n_runs"] == 3


def test_runs_include_provenance(client):
    rows = client.get("/api/runs").json()
    assert len(rows) == 3 and rows[0]["git_commit"] == "abc1234"


def test_curves_mean_and_spread(client):
    c = client.get("/api/curves/toy").json()["a"]
    assert c["seeds"] == 3 and c["mean"] == [2.0, 3.0] and c["std"][0] == pytest.approx(1.0)


def test_unknown_experiment_404(client):
    assert client.get("/api/experiments/nope").status_code == 404


def test_predict_without_adapter_is_503(client):
    assert client.post("/api/predict", json={"text": "great movie"}).status_code == 503


def test_predict_validates_input(client):
    assert client.post("/api/predict", json={"text": ""}).status_code == 422
