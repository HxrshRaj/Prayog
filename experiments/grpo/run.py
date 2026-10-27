"""Experiment 5: a tiny GRPO-style loop with a verifiable reward on toy arithmetic.

A small GRU encoder-decoder is trained FROM SCRATCH (no pretraining) with group-relative advantages:
for each prompt sample G answers, reward = exact-match (verifiable, no reward model), advantage = (r - mean_group) / std_group.
There is no value network. Updates are on-policy (one gradient step per batch), so no ratio clipping or KL term is used.
This is a toy: it shows the mechanism works, not that it scales to LLM reasoning.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prayog.common import experiment_cli, load_config, log_mlflow, set_seed, write_results

HERE = Path(__file__).parent
EOS, PLUS, EQ, BOS = 10, 11, 12, 13   # digits 0-9 are tokens 0-9
VOCAB = 14


class Policy(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, 32)
        self.enc = nn.GRU(32, hidden, batch_first=True)
        self.dec = nn.GRUCell(32, hidden)
        self.out = nn.Linear(hidden, VOCAB)

    def sample(self, prompts, max_tokens, greedy=False):
        """prompts: (B, 4) tokens [a, +, b, =]. Returns tokens (B, T), sum log-prob (B,)."""
        _, h = self.enc(self.emb(prompts))
        h = h[0]
        tok = torch.full((len(prompts),), BOS)
        toks, logp = [], torch.zeros(len(prompts))
        alive = torch.ones(len(prompts), dtype=torch.bool)
        for _ in range(max_tokens):
            h = self.dec(self.emb(tok), h)
            dist = torch.distributions.Categorical(logits=self.out(h))
            tok = dist.probs.argmax(-1) if greedy else dist.sample()
            logp = logp + torch.where(alive, dist.log_prob(tok), torch.zeros(()))
            toks.append(torch.where(alive, tok, torch.full_like(tok, EOS)))
            alive = alive & (tok != EOS)
        return torch.stack(toks, 1), logp


def decode(tokens):
    """Digits before the first EOS -> int, or None if malformed/empty."""
    out = []
    for t in tokens.tolist():
        if t == EOS:
            break
        if t > 9:
            return None
        out.append(str(t))
    return int("".join(out)) if out else None


def rewards(tokens, a, b):
    return torch.tensor([float(decode(t) == int(x + y)) for t, x, y in zip(tokens, a, b)])


def make_prompts(pairs):
    a = torch.tensor([p[0] for p in pairs])
    b = torch.tensor([p[1] for p in pairs])
    return torch.stack([a, torch.full_like(a, PLUS), b, torch.full_like(a, EQ)], 1), a, b


def accuracy(model, pairs, max_tokens):
    prompts, a, b = make_prompts(pairs)
    with torch.no_grad():
        toks, _ = model.sample(prompts, max_tokens, greedy=True)
    return rewards(toks, a, b).mean().item()


def train(cfg, seed):
    set_seed(seed)
    n = cfg["max_operand"] + 1
    allp = [(i, j) for i in range(n) for j in range(n)]
    rng = np.random.RandomState(seed)
    rng.shuffle(allp)
    k = int(len(allp) * cfg["holdout_frac"])
    held, train_p = allp[:k], allp[k:]
    model = Policy(cfg["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    G, curve = cfg["group_size"], []
    start_train, start_held = accuracy(model, train_p, cfg["max_answer_tokens"]), accuracy(model, held, cfg["max_answer_tokens"])
    for step in range(1, cfg["steps"] + 1):
        batch = [train_p[i] for i in rng.randint(0, len(train_p), cfg["prompts_per_step"])]
        prompts, a, b = make_prompts(batch)
        rp, ra, rb = (x.repeat_interleave(G, 0) for x in (prompts, a, b))
        toks, logp = model.sample(rp, cfg["max_answer_tokens"])
        r = rewards(toks, ra, rb)
        rg = r.view(-1, G)
        adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-6)).view(-1)  # group-relative
        loss = -(adv.detach() * logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % cfg["curve_every"] == 0 or step == cfg["steps"]:
            curve.append([step, r.mean().item()])
    return {"train_acc_start": start_train, "heldout_acc_start": start_held,
            "train_acc": accuracy(model, train_p, cfg["max_answer_tokens"]),
            "heldout_acc": accuracy(model, held, cfg["max_answer_tokens"]),
            "n_train_pairs": len(train_p), "n_heldout_pairs": len(held)}, curve


def main():
    args = experiment_cli(__doc__)
    cfg = load_config(HERE, args.smoke)
    runs = []
    for seed in cfg["seeds"]:
        m, curve = train(cfg, seed)
        print(f"seed={seed}: heldout {m['heldout_acc_start']:.2f} -> {m['heldout_acc']:.2f}, "
              f"train {m['train_acc_start']:.2f} -> {m['train_acc']:.2f}", flush=True)
        runs.append({"variant": "grpo", "seed": seed, "metrics": m, "curve": curve})
    write_results(HERE, cfg, runs, smoke=args.smoke)
    if not args.smoke:
        log_mlflow("grpo", runs)


if __name__ == "__main__":
    main()
