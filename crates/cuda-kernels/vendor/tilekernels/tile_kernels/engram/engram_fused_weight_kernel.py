import os

import torch
import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_engram_fused_weight_kernel(hidden_size: int, hc_mult: int):
    """Elementwise bf16 x bf16 -> fp32 for weight_hidden * weight_embed."""
    threads = 32
    vec_size = 8
    blk_d = threads * vec_size
    assert hidden_size % blk_d == 0
    num_blk = hidden_size // blk_d

    @T.prim_func
    def engram_fused_weight_kernel(
        weight_hidden: T.Tensor[(hc_mult, hidden_size), T.bfloat16],
        weight_embed: T.Tensor[(hc_mult, hidden_size), T.bfloat16],
        weight_fused: T.Tensor[(hc_mult, hidden_size), T.float],
    ):
        with T.Kernel(hc_mult, num_blk, threads=threads) as (pid_h, pid_b):
            tid = T.get_thread_binding()
            a_local = T.alloc_local((vec_size,), T.float)
            b_local = T.alloc_local((vec_size,), T.float)
            for i_k in T.vectorized(vec_size):
                a_local[i_k] = weight_hidden[pid_h, pid_b * blk_d + tid * vec_size + i_k]
                b_local[i_k] = weight_embed[pid_h, pid_b * blk_d + tid * vec_size + i_k]
            for i_k in T.vectorized(vec_size):
                weight_fused[pid_h, pid_b * blk_d + tid * vec_size + i_k] = a_local[i_k] * b_local[i_k]

    return engram_fused_weight_kernel


def fused_weight(weight_hidden: torch.Tensor, weight_embed: torch.Tensor) -> torch.Tensor:
    """Compute weight_hidden * weight_embed in fp32.

    Args:
        weight_hidden: Shape (hc_mult, hidden_size), bfloat16.
        weight_embed: Shape (hc_mult, hidden_size), bfloat16.

    Returns:
        weight_fused: Shape (hc_mult, hidden_size), float32.
    """
    hc_mult, hidden_size = weight_hidden.shape

    kernel = get_engram_fused_weight_kernel(hidden_size, hc_mult)
    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    weight_fused = torch.empty(hc_mult, hidden_size, dtype=torch.float32, device=weight_hidden.device)
    kernel(weight_hidden, weight_embed, weight_fused)

    return weight_fused
