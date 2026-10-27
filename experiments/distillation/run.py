"""Experiment 2: knowledge distillation (Hinton et al. 2015): DistilBERT teacher -> bert-tiny student."""
import sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification, BertTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prayog.common import experiment_cli, load_config, log_mlflow, set_seed, write_results
from prayog.data import accuracy, count_params, load_sst2, predict_logits, tokenise

HERE = Path(__file__).parent


def train_student(cfg, seed, data, teacher_logits, T):
    """T=None -> student alone (hard-label CE only)."""
    set_seed(seed)
    ids, mask, y = data[0]
    model = BertForSequenceClassification.from_pretrained(cfg["student"], num_labels=2)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    g = torch.Generator().manual_seed(seed)
    n, bs = len(ids), cfg["batch_size"]
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            logits = model(input_ids=ids[b], attention_mask=mask[b]).logits
            loss = F.cross_entropy(logits, y[b])
            if T is not None:
                soft = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(teacher_logits[b] / T, -1),
                                reduction="batchmean") * T * T  # T^2 keeps gradient scale (Hinton)
                loss = (1 - cfg["alpha"]) * loss + cfg["alpha"] * soft
            loss.backward()
            opt.step()
            opt.zero_grad()
    va_ids, va_mask, va_y = data[1]
    return accuracy(predict_logits(model, va_ids, va_mask), va_y)


def main():
    args = experiment_cli(__doc__)
    cfg = load_config(HERE, args.smoke)
    tr, va = load_sst2(cfg["n_train"], cfg["n_val"])
    t_tok = AutoTokenizer.from_pretrained(cfg["teacher"])
    s_tok = BertTokenizerFast.from_pretrained(cfg["student_tokenizer"])
    teacher = AutoModelForSequenceClassification.from_pretrained(cfg["teacher"])
    tt_tr, tt_va = tokenise(t_tok, tr, cfg["max_len"]), tokenise(t_tok, va, cfg["max_len"])
    teacher_logits = predict_logits(teacher, tt_tr[0], tt_tr[1])
    teacher_acc = accuracy(predict_logits(teacher, tt_va[0], tt_va[1]), tt_va[2])
    t_params = count_params(teacher)[0]
    s_params = count_params(BertForSequenceClassification.from_pretrained(cfg["student"]))[0]
    print(f"teacher acc={teacher_acc:.4f} params={t_params/1e6:.1f}M; student params={s_params/1e6:.1f}M")
    data = (tokenise(s_tok, tr, cfg["max_len"]), tokenise(s_tok, va, cfg["max_len"]))
    runs = []
    for seed in cfg["seeds"]:
        for T in [None] + cfg["temperatures"]:
            name = "student_alone" if T is None else f"distilled_T{T}"
            t0 = time.perf_counter()
            acc = train_student(cfg, seed, data, teacher_logits, T)
            print(f"seed={seed} {name}: acc={acc:.4f}", flush=True)
            runs.append({"variant": name, "seed": seed,
                         "metrics": {"accuracy": acc, "train_time_s": time.perf_counter() - t0}})
    extra = {"teacher": {"name": cfg["teacher"], "val_accuracy": teacher_acc, "params": t_params},
             "student_params": s_params}
    write_results(HERE, cfg, runs, extra, smoke=args.smoke)
    if not args.smoke:
        log_mlflow("distillation", runs)


if __name__ == "__main__":
    main()
