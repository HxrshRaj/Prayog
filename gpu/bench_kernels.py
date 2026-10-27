"""Benchmark Triton kernels vs PyTorch; writes gpu/results/triton_bench.json and prints a markdown table."""
import json
from pathlib import Path

import torch
import triton
from triton_kernels import layer_norm, softmax

rows = []
for n in [256, 512, 1024, 2048, 4096, 8192]:
    x = torch.randn(4096, n, device="cuda")
    w, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
    for name, tri, ref in [("softmax", lambda: softmax(x), lambda: torch.softmax(x, 1)),
                           ("layer_norm", lambda: layer_norm(x, w, b),
                            lambda: torch.nn.functional.layer_norm(x, (n,), w, b))]:
        t_tri, t_ref = triton.testing.do_bench(tri), triton.testing.do_bench(ref)
        rows.append({"kernel": name, "cols": n, "triton_ms": t_tri, "torch_ms": t_ref,
                     "speedup": t_ref / t_tri})
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
(out / "triton_bench.json").write_text(json.dumps(
    {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
     "rows": rows}, indent=2))
print("| kernel | cols | triton ms | torch ms | speedup |\n|---|---|---|---|---|")
for r in rows:
    print(f"| {r['kernel']} | {r['cols']} | {r['triton_ms']:.4f} | {r['torch_ms']:.4f} | {r['speedup']:.2f}x |")
