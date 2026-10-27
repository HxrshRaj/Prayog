"""Dataset + model-loading helpers shared by the NLP experiments."""
from __future__ import annotations

import numpy as np
import torch


def load_sst2(n_train: int, n_val: int, data_seed: int = 0):
    """SST-2 (GLUE): fixed random subset of train, and the official validation set (or a prefix)."""
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "sst2")
    rng = np.random.RandomState(data_seed)
    tr = ds["train"]
    idx = rng.permutation(len(tr))[:n_train]
    tr = tr.select(idx.tolist())
    va = ds["validation"]
    if n_val and n_val < len(va):
        va = va.select(range(n_val))
    return tr, va


def tokenise(tok, ds, max_len: int = 64):
    enc = tok(list(ds["sentence"]), truncation=True, padding="max_length", max_length=max_len,
              return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], torch.tensor(list(ds["label"]))


@torch.no_grad()
def predict_logits(model, ids, mask, bs: int = 256):
    model.eval()
    out = []
    for i in range(0, len(ids), bs):
        out.append(model(input_ids=ids[i:i + bs], attention_mask=mask[i:i + bs]).logits)
    return torch.cat(out)


def accuracy(logits, labels) -> float:
    return (logits.argmax(-1) == labels).float().mean().item()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_state_mb(model) -> float:
    """Analytic training-state memory for trainable params: weights + grads + Adam (m, v), fp32.
    Frozen base weights are counted once as well. Activations are excluded."""
    total, trainable = count_params(model)
    frozen = total - trainable
    return (frozen * 4 + trainable * 4 * 4) / 2**20
