"""PPO written from scratch in PyTorch (single env, GAE, clipped objective)."""
import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_act, hidden):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                nn.Tanh(), nn.Linear(hidden, n_act))
        self.v = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                               nn.Tanh(), nn.Linear(hidden, 1))

    def dist(self, x):
        return torch.distributions.Categorical(logits=self.pi(x))


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """delta_t = r_t + gamma*V(s_{t+1})*(1-done_t) - V(s_t);  A_t = delta_t + gamma*lam*(1-done_t)*A_{t+1}.

    `dones[t]` marks that the episode ended after step t, so nothing bootstraps across it.
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nxt = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nxt * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + np.asarray(values, dtype=np.float32)


def train_ppo(env_fn, cfg, seed, on_rollout=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = env_fn()
    obs, _ = env.reset(seed=seed)
    net = ActorCritic(env.observation_space.shape[0], env.action_space.n, cfg["hidden"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], eps=1e-5)
    n_steps, steps_done, ep_ret, ep_returns = cfg["n_steps"], 0, 0.0, []
    while steps_done < cfg["total_timesteps"]:
        O, A, LP, R, D, V = [], [], [], [], [], []
        for _ in range(n_steps):
            ot = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                d = net.dist(ot)
                a = d.sample()
                v = net.v(ot).item()
            nobs, r, term, trunc, _ = env.step(a.item())
            O.append(ot); A.append(a); LP.append(d.log_prob(a)); R.append(r); V.append(v)
            D.append(float(term or trunc))
            ep_ret += r
            steps_done += 1
            obs = nobs
            if term or trunc:
                ep_returns.append(ep_ret)
                ep_ret = 0.0
                obs, _ = env.reset()
        with torch.no_grad():
            last_v = net.v(torch.as_tensor(obs, dtype=torch.float32)).item()
        adv, ret = compute_gae(R, V, D, last_v, cfg["gamma"], cfg["gae_lambda"])
        O, A, LP = torch.stack(O), torch.stack(A), torch.stack(LP)
        adv, ret = torch.as_tensor(adv), torch.as_tensor(ret)
        for _ in range(cfg["n_epochs"]):
            perm = torch.randperm(n_steps)
            for i in range(0, n_steps, cfg["batch_size"]):
                b = perm[i:i + cfg["batch_size"]]
                d = net.dist(O[b])
                ratio = torch.exp(d.log_prob(A[b]) - LP[b])
                ab = (adv[b] - adv[b].mean()) / (adv[b].std() + 1e-8)
                clipped = torch.clamp(ratio, 1 - cfg["clip_range"], 1 + cfg["clip_range"])
                pg = -torch.min(ratio * ab, clipped * ab).mean()
                vl = ((net.v(O[b]).squeeze(-1) - ret[b]) ** 2).mean()
                loss = pg + cfg["vf_coef"] * vl - cfg["ent_coef"] * d.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
                opt.step()
        if on_rollout:
            on_rollout(steps_done, ep_returns)
    return net


@torch.no_grad()
def evaluate(net, env_fn, episodes, seed):
    env = env_fn()
    rets = []
    for e in range(episodes):
        obs, _ = env.reset(seed=10_000 + seed * 100 + e)
        done, total = False, 0.0
        while not done:
            a = net.dist(torch.as_tensor(obs, dtype=torch.float32)).probs.argmax().item()
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        rets.append(total)
    return float(np.mean(rets))
