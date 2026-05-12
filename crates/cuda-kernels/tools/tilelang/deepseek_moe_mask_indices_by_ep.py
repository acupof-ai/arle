# Copyright (c) 2026 DeepSeek
# SPDX-License-Identifier: MIT
#
# Adapted from deepseek-ai/TileKernels:
# tile_kernels/moe/mask_indices_by_tp_kernel.py
#
# This TileLang kernel implements the DeepSeek MoE expert-id mask/remap used
# when EP and TP are both active. It keeps only experts owned by the current
# TP rank and compacts their global expert ids into the local expert range.

import os

import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_mask_indices_by_ep_kernel(num_topk: int, dtype: T.dtype):
    num_threads = 128

    num_tokens = T.dynamic("num_tokens")
    num_blocks = T.ceildiv(num_tokens * num_topk, num_threads)

    @T.prim_func
    def mask_indices_by_ep_kernel(
        indices: T.Tensor[(num_tokens, num_topk), dtype],
        masked_indices: T.Tensor[(num_tokens, num_topk), dtype],
        experts_per_ep_rank: T.int32,
        experts_per_moe_dp_group: T.int32,
        num_tp_ranks: T.int32,
        tp_rank: T.int32,
    ):
        with T.Kernel(num_blocks, threads=num_threads) as (pid,):
            indices_1d = T.reshape(indices, (num_tokens * num_topk,))
            masked_indices_1d = T.reshape(masked_indices, (num_tokens * num_topk,))
            thread_idx = T.get_thread_binding()
            index = pid * num_threads + thread_idx

            value = T.alloc_var(dtype)
            if index < num_tokens * num_topk:
                value = indices_1d[index]
                if (
                    value < 0
                    or T.truncmod(T.truncdiv(value, experts_per_ep_rank), num_tp_ranks)
                    != tp_rank
                ):
                    masked_indices_1d[index] = -1
                else:
                    value -= tp_rank * experts_per_ep_rank
                    dp_rank = T.truncdiv(value, experts_per_moe_dp_group)
                    value -= dp_rank * (experts_per_moe_dp_group - experts_per_ep_rank)
                    masked_indices_1d[index] = T.Select(value < 0, T.int64(-1), value)

    return mask_indices_by_ep_kernel


def print_kernel_source(num_topk: int, dtype: T.dtype) -> None:
    kernel = get_mask_indices_by_ep_kernel(num_topk, dtype)
    print(kernel.get_kernel_source())


if __name__ == "__main__":
    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", "1")):
        print_kernel_source(6, T.int64)
