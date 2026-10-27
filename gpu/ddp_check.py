"""Multi-GPU NCCL check: all-reduce bandwidth + a small DDP training run.

Run on Kaggle 2x T4:  torchrun --nproc_per_node=2 gpu/ddp_check.py
Writes gpu/results/ddp_rank0.json
"""
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def allreduce_bw(world):
    rows = []
    for mb in [1, 4, 16, 64, 256]:
        x = torch.ones(mb * 2**20 // 4, device="cuda")
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        # ring all-reduce bus bandwidth = 2*(n-1)/n * size / time
        rows.append({"size_mb": mb, "time_ms": dt * 1e3,
                     "bus_gbps": 2 * (world - 1) / world * mb / 1024 / dt})
    return rows


def ddp_train(rank, steps=200):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    X = torch.randn(4096, 32, generator=g)
    y = (X[:, :4].sum(1) > 0).long()
    model = DDP(nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 2)).cuda(), device_ids=[rank])
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    shard = slice(rank * 2048, (rank + 1) * 2048)
    Xs, ys = X[shard].cuda(), y[shard].cuda()
    losses = []
    for _ in range(steps):
        i = torch.randint(0, 2048, (128,), device="cuda")
        loss = nn.functional.cross_entropy(model(Xs[i]), ys[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        acc = (model(X.cuda()).argmax(1) == y.cuda()).float().mean().item()
    return {"first_loss": losses[0], "last_loss": losses[-1], "full_data_acc": acc}


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    bw = allreduce_bw(world)
    tr = ddp_train(rank)
    if rank == 0:
        out = Path(__file__).parent / "results"
        out.mkdir(exist_ok=True)
        res = {"world_size": world, "gpus": [torch.cuda.get_device_name(i) for i in range(world)],
               "allreduce": bw, "ddp": tr, "torch": torch.__version__}
        (out / "ddp_rank0.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
    dist.destroy_process_group()
