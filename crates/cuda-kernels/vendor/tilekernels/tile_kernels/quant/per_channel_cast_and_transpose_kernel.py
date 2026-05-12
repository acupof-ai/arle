import os
import torch
import tilelang
from tilelang import language as T
from tile_kernels.quant.common import *


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_per_channel_cast_and_transpose_kernel(
    hidden: int,
    in_dtype: T.dtype,
    out_config: CastOutputConfig,
):
    num_per_tokens, num_per_channels = out_config.sf_block
    assert num_per_tokens in [32, 128] and num_per_channels == 1
    num_threads = 256

    TILE_X, TILE_Y, TILE_K = 128, 64, 4
    num_threads_per_token = TILE_Y // TILE_K

    assert TILE_X % num_per_tokens == 0
    assert hidden % TILE_Y == 0

    # Runtime symbols
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def per_channel_cast_and_transpose_kernel(
        x: T.Tensor[(num_tokens, hidden), in_dtype],
        out: T.Tensor[(hidden, num_tokens), out_config.dtype],
        out_sf: T.Tensor[(num_tokens // num_per_tokens, hidden), out_config.sf_dtype],
    ):
        with T.Kernel(hidden // TILE_Y, num_tokens // TILE_X, threads=num_threads) as (pid_y, pid_x):
            # Shared padding to reduce bank conflict
            out_shared = T.alloc_shared((TILE_Y, TILE_X + TILE_K), in_dtype)
            tid = T.get_thread_binding()
            row, col = tid // num_threads_per_token, tid % num_threads_per_token

            T.assume(num_tokens % TILE_X == 0)

            # Read and transpose
            tmp = T.alloc_local((TILE_K, TILE_K), in_dtype)
            tmp_row = T.alloc_local((TILE_K, ), in_dtype)
            for i_ in T.unroll(TILE_X // TILE_K // (num_threads // num_threads_per_token)):
                i = i_ * (num_threads // num_threads_per_token) + row
                # Read into registers
                for j in T.unroll(TILE_K):
                    for k in T.vectorized(TILE_K):
                        tmp_row[k] = x[pid_x * TILE_X + i * TILE_K + j, pid_y * TILE_Y + col * TILE_K + k]
                    for k in T.unroll(TILE_K):
                        tmp[k, j] = tmp_row[k]

                # Copy into shared memory
                for j in T.unroll(TILE_K):
                    swizzle_j = (j + tid // 4) % TILE_K
                    for k in T.vectorized(TILE_K):
                        out_shared[col * TILE_K + swizzle_j, i * TILE_K + k] = tmp[swizzle_j, k]

            # Write into output
            # NOTE: Use multiple stages to reduce register pressure
            num_stages = 4
            tile_y_per_stage = TILE_Y // num_stages
            out_fragment = T.alloc_fragment((tile_y_per_stage, TILE_X), T.float32)
            amax_fragment = T.alloc_fragment((tile_y_per_stage, TILE_X // num_per_tokens), T.float32)
            for stage in T.unroll(num_stages):
                T.copy(out_shared[tile_y_per_stage * stage: tile_y_per_stage * (stage + 1), 0: TILE_X], out_fragment)
                out_fragment_reshaped = T.reshape(out_fragment, (tile_y_per_stage, TILE_X // num_per_tokens, num_per_tokens))
                T.reduce_absmax(out_fragment_reshaped, amax_fragment, dim=2)

                for i, j in T.Parallel(tile_y_per_stage, TILE_X // num_per_tokens):
                    sf, sf_inv = get_sf_and_inv(T.cast(amax_fragment[i, j], T.float32), out_config)
                    out_sf[pid_x * (TILE_X // num_per_tokens) + j, pid_y * TILE_Y + stage * tile_y_per_stage + i] = sf
                    amax_fragment[i, j] = sf_inv

                for i, j in T.Parallel(tile_y_per_stage, TILE_X):
                    out[pid_y * TILE_Y + stage * tile_y_per_stage + i, pid_x * TILE_X + j] = out_fragment[i, j] * amax_fragment[i, j // num_per_tokens]

    return per_channel_cast_and_transpose_kernel


def per_channel_cast_and_transpose(x: torch.Tensor, fmt: str, num_per_tokens: int, round_sf: bool = False) -> QuantTensor:
    """Cast a BF16 matrix to FP8, transpose it, and produce sf factors.

    Args:
        x: Input 2D contiguous BF16 tensor of shape (num_tokens, hidden).
        fmt: Target FP8 format.
        num_per_tokens: Number of tokens in each scaling block.
        round_sf: Whether to round scaling factors.

    Returns:
        A tuple `(out, out_sf)` with transposed FP8 output and sf tensor.
    """
    assert x.dim() == 2 and x.is_contiguous()
    assert x.dtype == torch.bfloat16
    num_tokens, hidden = x.shape

    assert fmt == 'e4m3'
    assert num_tokens % 128 == 0 and hidden % 64 == 0
    assert num_per_tokens in [32, 128]

    out_config = get_cast_output_config(fmt, (num_per_tokens, 1), round_sf=round_sf)

    # Get kernel implement
    kernel = get_per_channel_cast_and_transpose_kernel(
        hidden=hidden,
        in_dtype=T.dtype(x.dtype),
        out_config=out_config,
    )

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    # Allocate output and launch
    out = torch.empty((hidden, num_tokens), dtype=torch.float8_e4m3fn, device=x.device)
    out_sf = torch.empty((num_tokens // num_per_tokens, hidden), dtype=torch.float32, device=x.device)
    if num_tokens > 0:
        kernel(x, out, out_sf)

    return out, out_sf
