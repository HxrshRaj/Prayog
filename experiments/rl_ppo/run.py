"""Experiment 4: PPO (Schulman et al. 2017): Stable-Baselines3 vs our from-scratch PyTorch PPO."""
import sys, time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from prayog.common import experiment_cli, load_config, log_mlflow, set_seed, write_results
from ppo_scratch import evaluate, train_ppo

HERE = Path(__file__).parent


def curve_from_returns(points, total):
    """points: list of (timestep, mean return over last 10 episodes) -> fixed 25-point grid."""
    grid = np.linspace(0, total, 26)[1:]
    ts = np.array([p[0] for p in points])
    vs = np.array([p[1] for p in points])
    return [[float(g), float(np.interp(g, ts, vs))] for g in grid]


def run_sb3(cfg, seed):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    pts = []

    class CB(BaseCallback):
        def _on_rollout_end(self):
            eps = [e["r"] for e in self.model.ep_info_buffer]
            if eps:
                pts.append((self.num_timesteps, float(np.mean(eps[-10:]))))

        def _on_step(self):
            return True

    env = Monitor(gym.make(cfg["env"]))
    model = PPO("MlpPolicy", env, seed=seed, n_steps=cfg["n_steps"], batch_size=cfg["batch_size"],
                n_epochs=cfg["n_epochs"], learning_rate=cfg["lr"], gamma=cfg["gamma"],
                gae_lambda=cfg["gae_lambda"], clip_range=cfg["clip_range"], ent_coef=cfg["ent_coef"],
                vf_coef=cfg["vf_coef"], max_grad_norm=cfg["max_grad_norm"], device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=[cfg["hidden"]] * 2, vf=[cfg["hidden"]] * 2),
                                   activation_fn=torch.nn.Tanh), verbose=0)
    t0 = time.perf_counter()
    model.learn(cfg["total_timesteps"], callback=CB())
    t = time.perf_counter() - t0
    rets = []
    e = gym.make(cfg["env"])
    for ep in range(cfg["eval_episodes"]):
        obs, _ = e.reset(seed=10_000 + seed * 100 + ep)
        done, tot = False, 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, te, tr, _ = e.step(int(a))
            tot += r
            done = te or tr
        rets.append(tot)
    return float(np.mean(rets)), pts, t


def run_scratch(cfg, seed):
    env_fn = lambda: gym.make(cfg["env"])
    pts = []

    def cb(steps, eps):
        if eps:
            pts.append((steps, float(np.mean(eps[-10:]))))

    t0 = time.perf_counter()
    net = train_ppo(env_fn, cfg, seed, cb)
    t = time.perf_counter() - t0
    return evaluate(net, env_fn, cfg["eval_episodes"], seed), pts, t


def main():
    args = experiment_cli(__doc__)
    cfg = load_config(HERE, args.smoke)
    runs = []
    for seed in cfg["seeds"]:
        for name, fn in [("sb3_ppo", run_sb3), ("scratch_ppo", run_scratch)]:
            set_seed(seed)
            final, pts, t = fn(cfg, seed)
            print(f"seed={seed} {name}: eval_return={final:.1f} ({t:.0f}s)", flush=True)
            runs.append({"variant": name, "seed": seed,
                         "metrics": {"eval_return": final, "train_time_s": t},
                         "curve": curve_from_returns(pts, cfg["total_timesteps"]) if pts else []})
    write_results(HERE, cfg, runs, smoke=args.smoke)
    if not args.smoke:
        log_mlflow("rl_ppo", runs)


if __name__ == "__main__":
    main()
