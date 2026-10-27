import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "rl_ppo"))
sys.path.insert(0, str(ROOT / "scripts"))

from prayog.common import mean_std, summarise, write_results  # noqa: E402
from ppo_scratch import compute_gae  # noqa: E402


def test_mean_std_uses_sample_std():
    m = mean_std([1.0, 2.0, 3.0])
    assert m["mean"] == 2.0 and m["std"] == pytest.approx(1.0) and m["n"] == 3


def test_summarise_groups_by_variant():
    runs = [{"variant": "a", "seed": s, "metrics": {"x": float(s)}} for s in range(3)]
    runs.append({"variant": "b", "seed": 0, "metrics": {"x": 10.0}})
    s = summarise(runs)
    assert s["a"]["x"]["mean"] == 1.0 and s["b"]["x"]["std"] == 0.0


def test_write_results_smoke_does_not_clobber(tmp_path):
    runs = [{"variant": "a", "seed": 0, "metrics": {"x": 1.0}}]
    write_results(tmp_path, {}, runs, smoke=True)
    assert (tmp_path / "results.smoke.json").exists() and not (tmp_path / "results.json").exists()
    res = json.loads((tmp_path / "results.smoke.json").read_text())
    assert "git_commit" in res["env"] and "libraries" in res["env"]


def test_gae_matches_manual_computation():
    r, v, d = [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.0, 0.0, 1.0]
    gamma, lam = 0.9, 0.8
    adv, ret = compute_gae(r, v, d, last_value=0.0, gamma=gamma, lam=lam)
    d2 = 1.0 - 0.5                    # terminal step: no bootstrap
    d1 = 1.0 + gamma * 0.5 - 0.5
    d0 = d1
    a2 = d2
    a1 = d1 + gamma * lam * a2
    a0 = d0 + gamma * lam * a1
    np.testing.assert_allclose(adv, [a0, a1, a2], rtol=1e-6)
    np.testing.assert_allclose(ret, adv + 0.5, rtol=1e-6)


def test_gae_lambda_one_is_monte_carlo_minus_value():
    r, v, d = [1.0, 1.0], [0.0, 0.0], [0.0, 1.0]
    adv, _ = compute_gae(r, v, d, 0.0, gamma=0.5, lam=1.0)
    assert adv[0] == pytest.approx(1.0 + 0.5 * 1.0)


def test_card_verdict_rule():
    from build_cards import verdict_gain
    assert verdict_gain(0.05, 0.01) == "reproduced"
    assert verdict_gain(0.015, 0.01) == "partly reproduced"
    assert verdict_gain(0.005, 0.01) == "not reproduced"
