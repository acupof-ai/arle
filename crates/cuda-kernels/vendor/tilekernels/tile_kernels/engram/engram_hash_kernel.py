import os

import torch
import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    },
)
def get_engram_hash_kernel(
    max_ngram_size: int = 3,
    num_ngram_layers: int = 2,
    num_embed_table_per_ngram: int = 8,
):
    num_tokens = T.dynamic('num_tokens')
    threads = 32
    blk_m = threads
    num_out_cols = (max_ngram_size - 1) * num_embed_table_per_ngram

    @T.prim_func
    def engram_hash_kernel(
        ngram_token_ids: T.Tensor[(num_tokens, max_ngram_size), T.int32],
        multipliers: T.Tensor[(num_ngram_layers, max_ngram_size), T.int64],
        vocab_sizes: T.Tensor[(num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), T.int32],
        offsets: T.Tensor[(num_ngram_layers, num_out_cols), T.int32],
        output: T.Tensor[(num_ngram_layers, num_tokens, num_out_cols), T.int32],
    ):
        with T.Kernel(num_ngram_layers, T.ceildiv(num_tokens, blk_m), threads=threads) as (pid_h, pid_s):
            tid = T.get_thread_binding()
            token_idx = pid_s * blk_m + tid
            if token_idx >= num_tokens:
                T.thread_return()
            x_local = T.alloc_local((max_ngram_size,), T.int32)
            multipliers_local = T.alloc_local((max_ngram_size,), T.int64)
            vocab_sizes_local = T.alloc_local((max_ngram_size - 1, num_embed_table_per_ngram), T.int32)
            offsets_local = T.alloc_local((num_out_cols,), T.int32)
            output_local = T.alloc_local((num_out_cols,), T.int32)
            hash_local = T.alloc_var(T.int64)

            T.copy(multipliers[pid_h, :], multipliers_local)
            T.copy(vocab_sizes[pid_h, :, :], vocab_sizes_local)
            T.copy(offsets[pid_h, :], offsets_local)
            T.copy(ngram_token_ids[token_idx, :], x_local)

            hash_local = 0
            for ngram_idx in T.unroll(0, max_ngram_size):
                hash_local = T.bitwise_xor(
                    hash_local,
                    T.cast(x_local[ngram_idx], T.int64) * multipliers_local[ngram_idx],
                )
                if ngram_idx > 0:
                    for j in T.unroll(num_embed_table_per_ngram):
                        col = (ngram_idx - 1) * num_embed_table_per_ngram + j
                        output_local[col] = (hash_local % T.cast(vocab_sizes_local[ngram_idx - 1, j], T.int64)) + offsets_local[col]

            T.copy(output_local, output[pid_h, token_idx, :])

    return engram_hash_kernel


def engram_hash(
    ngram_token_ids: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Compute n-gram hash embedding indices.

    Args:
        ngram_token_ids: N-gram token IDs, shape (num_tokens, max_ngram_size), int32.
        multipliers: Per-layer hash multipliers, shape (num_ngram_layers, max_ngram_size), int64.
        vocab_sizes: Per-layer embedding table sizes,
            shape (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), int32.
        offsets: Per-layer embedding table offsets,
            shape (num_ngram_layers, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.

    Returns:
        Embedding indices, shape (num_ngram_layers, num_tokens, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.
    """
    num_tokens, max_ngram_size = ngram_token_ids.shape
    num_ngram_layers, _, num_embed_table_per_ngram = vocab_sizes.shape
    num_out_cols = (max_ngram_size - 1) * num_embed_table_per_ngram

    output = torch.empty((num_ngram_layers, num_tokens, num_out_cols), dtype=torch.int32, device=ngram_token_ids.device)

    kernel = get_engram_hash_kernel(max_ngram_size, num_ngram_layers, num_embed_table_per_ngram)
    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    if num_tokens > 0:
        kernel(ngram_token_ids, multipliers, vocab_sizes, offsets, output)

    return output
