"""Experiment 4b: same PPO comparison (SB3 vs from-scratch) on LunarLander-v3 at a small step budget."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "rl_ppo"))
import run as cartpole  # experiments/rl_ppo/run.py (same training + evaluation code)
from prayog.common import experiment_cli, load_config, log_mlflow, set_seed, write_results

HERE = Path(__file__).parent


def main():
    args = experiment_cli(__doc__)
    cfg = load_config(HERE, args.smoke)
    runs = []
    for seed in cfg["seeds"]:
        for name, fn in [("sb3_ppo", cartpole.run_sb3), ("scratch_ppo", cartpole.run_scratch)]:
            set_seed(seed)
            final, pts, t = fn(cfg, seed)
            print(f"seed={seed} {name}: eval_return={final:.1f} ({t:.0f}s)", flush=True)
            runs.append({"variant": name, "seed": seed,
                         "metrics": {"eval_return": final, "train_time_s": t},
                         "curve": cartpole.curve_from_returns(pts, cfg["total_timesteps"]) if pts else []})
    write_results(HERE, cfg, runs, smoke=args.smoke)
    if not args.smoke:
        log_mlflow("rl_ppo_lunarlander", runs)


if __name__ == "__main__":
    main()
