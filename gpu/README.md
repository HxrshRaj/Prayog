# GPU systems (separate from the CPU experiments)

| Part | File | Needs | Status |
|---|---|---|---|
| Triton fused softmax + layer norm, correctness tests | `triton_kernels.py`, `test_kernels.py` | CUDA GPU (Colab T4) | **not run** |
| Triton benchmark table | `bench_kernels.py` | CUDA GPU | **not run** |
| NCCL all-reduce bandwidth + DDP run | `ddp_check.py` | 2 GPUs (Kaggle 2x T4) | **not run** |

The development machine has no NVIDIA GPU, so none of this has been executed and **no GPU numbers exist in this repo**.
When you run them, results land in `gpu/results/*.json`; change a status above only once its results file is committed.

- Colab T4: `cd gpu && pytest test_kernels.py && python bench_kernels.py`
- Kaggle 2x T4: `torchrun --nproc_per_node=2 gpu/ddp_check.py`
