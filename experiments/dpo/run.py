"""Experiment 3: DPO (Rafailov et al. 2023) on pythia-70m, with human labels vs AI-judge (RLAIF-style) labels."""
import json, sys, tempfile
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prayog.common import experiment_cli, load_config, log_mlflow, set_seed, write_results
from prayog.judge import make_judge

HERE = Path(__file__).parent


def load_pairs(cfg):
    """Parse Anthropic HH transcripts into (prompt, chosen_reply, rejected_reply) with a short-length filter."""
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
    marker = "\n\nAssistant:"
    pairs = []
    for ex in ds:
        c, r = ex["chosen"], ex["rejected"]
        i, j = c.rfind(marker), r.rfind(marker)
        if i < 0 or j < 0 or c[:i] != r[:j]:
            continue
        prompt = c[:i] + marker
        cc, rr = c[i + len(marker):].strip(), r[j + len(marker):].strip()
        if (len(prompt) > cfg["max_prompt_chars"] or cc == rr
                or not all(cfg["min_reply_chars"] <= len(x) <= cfg["max_reply_chars"] for x in (cc, rr))):
            continue
        pairs.append({"prompt": prompt, "chosen": " " + cc, "rejected": " " + rr})
        if len(pairs) >= cfg["n_pairs"]:
            break
    return pairs


def label_with_judge(pairs, cfg):
    """Cache judge verdicts: does the judge prefer the human-chosen reply?"""
    cache = HERE / ("judge_cache.smoke.json" if cfg.get("_smoke") else "judge_cache.json")
    if cache.exists():
        return json.loads(cache.read_text())
    judge = make_judge(cfg["judge"])
    out = []
    for k, p in enumerate(pairs):
        pref, consistent = judge.prefers_first(p["prompt"], p["chosen"], p["rejected"])
        out.append({"judge_prefers_human": bool(pref), "consistent": bool(consistent)})
        if k % 50 == 0:
            print(f"judge {k}/{len(pairs)}", flush=True)
    cache.write_text(json.dumps(out))
    return out


def seq_logps(model, tok, prompt, reply, max_length):
    """(sum logp, n_tokens) of `reply` given `prompt`."""
    p_ids = tok(prompt)["input_ids"]
    r_ids = tok(reply)["input_ids"] + [tok.eos_token_id]
    ids = (p_ids + r_ids)[:max_length]
    n_r = len(ids) - len(p_ids)
    if n_r <= 0:
        return 0.0, 1
    x = torch.tensor([ids])
    with torch.no_grad():
        lp = torch.log_softmax(model(x).logits[0, :-1], -1)
    tgt = x[0, 1:]
    tok_lp = lp.gather(1, tgt[:, None])[:, 0][-n_r:]
    return tok_lp.sum().item(), n_r


def pref_metrics(policy, ref, tok, heldout, max_length):
    """reward_acc: DPO implicit reward (logp ratio vs frozen reference) ranks human-chosen above rejected.
    logp_acc: length-normalised policy log-likelihood ranks chosen above rejected (works for the untrained model)."""
    policy.eval()
    ra = la = 0
    for ex in heldout:
        pc, nc = seq_logps(policy, tok, ex["prompt"], ex["chosen"], max_length)
        pr, nr = seq_logps(policy, tok, ex["prompt"], ex["rejected"], max_length)
        rc, _ = seq_logps(ref, tok, ex["prompt"], ex["chosen"], max_length)
        rr, _ = seq_logps(ref, tok, ex["prompt"], ex["rejected"], max_length)
        ra += (pc - rc) > (pr - rr)
        la += (pc / nc) > (pr / nr)
    n = len(heldout)
    return ra / n, la / n


def rule_check(policy, tok, heldout, n_prompts, new_tokens):
    """Rule-based quality: reply non-empty, no role leak ('Human:'), and not degenerate (distinct-2 >= 0.6)."""
    policy.eval()
    ok = 0
    for ex in heldout[:n_prompts]:
        enc = tok(ex["prompt"], return_tensors="pt")
        with torch.no_grad():
            out = policy.generate(**enc, max_new_tokens=new_tokens, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        ids = out[0, enc["input_ids"].shape[1]:].tolist()
        text = tok.decode(ids, skip_special_tokens=True).strip()
        bigrams = list(zip(ids, ids[1:]))
        distinct = len(set(bigrams)) / max(len(bigrams), 1)
        ok += bool(text) and "Human:" not in text and distinct >= 0.6
    return ok / max(n_prompts, 1)


def train_arm(cfg, seed, train_pairs, tok):
    set_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32)
    args = DPOConfig(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=cfg["batch_size"],
                     gradient_accumulation_steps=cfg["grad_accum"], num_train_epochs=cfg["epochs"],
                     learning_rate=cfg["lr"], beta=cfg["beta"], max_length=cfg["max_length"], seed=seed,
                     use_cpu=True, bf16=False, fp16=False, report_to=[], save_strategy="no", logging_steps=10_000,
                     disable_tqdm=True)
    trainer = DPOTrainer(model=model, args=args, train_dataset=Dataset.from_list(train_pairs),
                         processing_class=tok)
    trainer.train()
    return trainer.model


def main():
    cli = experiment_cli(__doc__)
    cfg = load_config(HERE, cli.smoke)
    cfg["_smoke"] = cli.smoke
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    tok.pad_token = tok.eos_token
    pairs = load_pairs(cfg)
    verdicts = label_with_judge(pairs, cfg)
    cfg.pop("_smoke")
    n_tr = cfg["n_train"]
    train_h, held = pairs[:n_tr], pairs[n_tr:]
    agree = [v["judge_prefers_human"] for v in verdicts]
    cons = [v["consistent"] for v in verdicts]
    cons_agree = [a for a, c in zip(agree, cons) if c]
    judge_stats = {"judge": cfg["judge"], "judge_model": cfg["judge_model"], "n_pairs": len(pairs),
                   "agreement_with_human": sum(agree) / len(agree),
                   "position_consistent_fraction": sum(cons) / len(cons),
                   "agreement_when_consistent": sum(cons_agree) / max(len(cons_agree), 1),
                   "n_consistent": len(cons_agree)}
    print("judge:", judge_stats, flush=True)
    # AI-labelled training set: swap chosen/rejected where the judge disagrees with the human label
    train_j = [p if a else {**p, "chosen": p["rejected"], "rejected": p["chosen"]}
               for p, a in zip(train_h, agree[:n_tr])]

    ref = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).eval()
    runs = []
    for seed in cfg["seeds"]:
        set_seed(seed)
        arms = {"untrained": None, "dpo_human_labels": train_h, "dpo_judge_labels": train_j}
        for name, data in arms.items():
            if name == "untrained" and seed != cfg["seeds"][0]:
                continue  # deterministic: no seed dependence, one run
            policy = ref if data is None else train_arm(cfg, seed, data, tok)
            ra, la = pref_metrics(policy, ref, tok, held, cfg["max_length"])
            rc = rule_check(policy, tok, held, cfg["gen_prompts"], cfg["gen_new_tokens"])
            print(f"seed={seed} {name}: reward_acc={ra:.3f} logp_acc={la:.3f} rule_pass={rc:.3f}", flush=True)
            runs.append({"variant": name, "seed": seed,
                         "metrics": {"heldout_reward_acc": ra, "heldout_logp_acc": la, "rule_pass_rate": rc}})
    write_results(HERE, cfg, runs, {"judge_stats": judge_stats, "n_heldout": len(held)}, smoke=cli.smoke)
    if not cli.smoke:
        log_mlflow("dpo", runs)


if __name__ == "__main__":
    main()
