"""Fused row-wise softmax and layer-norm forward kernels in Triton.

Needs a CUDA GPU. NOT run on the development laptop (no NVIDIA GPU).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)  # subtract the row max for numerical stability
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_stride + cols, y, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)
    _softmax_kernel[(n_rows,)](out, x, x.stride(0), out.stride(0), n_cols, BLOCK=block,
                               num_warps=4 if block <= 2048 else 8)
    return out


@triton.jit
def _layernorm_kernel(out_ptr, in_ptr, w_ptr, b_ptr, stride, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_cols
    y = xc * (1.0 / tl.sqrt(var + eps))
    w = tl.load(w_ptr + cols, mask=mask, other=1.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * stride + cols, y * w + b, mask=mask)


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    assert x.is_cuda and x.ndim == 2 and x.is_contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)
    _layernorm_kernel[(n_rows,)](out, x, weight, bias, x.stride(0), n_cols, eps, BLOCK=block,
                                 num_warps=4 if block <= 2048 else 8)
    return out
