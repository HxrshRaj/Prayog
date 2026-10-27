"""Experiment 1: LoRA (Hu et al. 2021) vs full fine-tuning on SST-2 with bert-tiny."""
import sys, time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import BertForSequenceClassification, BertTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prayog.common import experiment_cli, load_config, log_mlflow, set_seed, write_results
from prayog.data import (accuracy, count_params, load_sst2, predict_logits, tokenise, train_state_mb)

HERE = Path(__file__).parent


def train_one(cfg, variant, rank, seed, data, tok_model_name):
    set_seed(seed)
    (tr_ids, tr_mask, tr_y), (va_ids, va_mask, va_y) = data
    model = BertForSequenceClassification.from_pretrained(tok_model_name, num_labels=2)
    lr = cfg["full_lr"]
    if variant == "lora":
        lc = LoraConfig(task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=cfg["lora_alpha_factor"] * rank,
                        target_modules=cfg["lora_target"], lora_dropout=0.0)
        model = get_peft_model(model, lc)
        lr = cfg["lora_lr"]
    total, trainable = count_params(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    g = torch.Generator().manual_seed(seed)
    n, bs = len(tr_ids), cfg["batch_size"]
    t0 = time.perf_counter()
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = model(input_ids=tr_ids[b], attention_mask=tr_mask[b], labels=tr_y[b]).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
    train_time = time.perf_counter() - t0
    acc = accuracy(predict_logits(model, va_ids, va_mask), va_y)
    return model, {"accuracy": acc, "train_time_s": train_time, "trainable_params": trainable,
                   "total_params": total, "trainable_pct": 100 * trainable / total,
                   "train_state_mb": train_state_mb(model)}


def main():
    args = experiment_cli(__doc__)
    cfg = load_config(HERE, args.smoke)
    tok = BertTokenizerFast.from_pretrained(cfg["model"])
    tr, va = load_sst2(cfg["n_train"], cfg["n_val"])
    data = (tokenise(tok, tr, cfg["max_len"]), tokenise(tok, va, cfg["max_len"]))
    variants = [("full", 0)] + [("lora", r) for r in cfg["lora_ranks"]]
    runs = []
    for seed in cfg["seeds"]:
        for kind, r in variants:
            name = "full_ft" if kind == "full" else f"lora_r{r}"
            model, m = train_one(cfg, kind, r, seed, data, cfg["model"])
            print(f"seed={seed} {name}: acc={m['accuracy']:.4f} trainable={m['trainable_pct']:.2f}% "
                  f"time={m['train_time_s']:.1f}s", flush=True)
            runs.append({"variant": name, "seed": seed, "metrics": m})
            if not args.smoke and name == "lora_r4" and seed == 0:  # adapter served by /predict
                out = ROOT / "models" / "lora_sst2"
                model.save_pretrained(out)
    write_results(HERE, cfg, runs, smoke=args.smoke)
    if not args.smoke:
        log_mlflow("lora_vs_full", runs)


if __name__ == "__main__":
    main()
