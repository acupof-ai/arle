import os
import torch
import tilelang
from tilelang import language as T
from tile_kernels.quant.common import *
from typing import Optional


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_swiglu_forward_and_per_channel_cast_and_transpose_kernel(
    hidden: int,
    without_transpose: bool,
    use_clamp: bool,
    in_dtype: T.dtype,
    out_config: CastOutputConfig,
    swiglu_clamp_value: float,
):
    num_per_tokens, _ = out_config.sf_block
    num_threads = 256

    TILE_X, TILE_Y, TILE_K = 128, 64, 4
    while TILE_X // 2 % num_per_tokens == 0 and hidden % (TILE_Y * 2) == 0:
        TILE_X //= 2
        TILE_Y *= 2

    num_vec = get_best_vectorize_size(in_dtype)
    num_threads_per_global_token = TILE_Y // num_vec
    num_threads_per_shared_token = TILE_Y // TILE_K
    thread_global_step = num_threads // num_threads_per_global_token
    thread_shared_step = num_threads // num_threads_per_shared_token
    num_split_blocks = TILE_X // num_per_tokens

    assert TILE_X % num_per_tokens == 0

    # Runtime symbols
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def swiglu_forward_and_per_channel_cast_and_transpose_kernel(
        x: T.Tensor[(num_tokens, hidden * 2), in_dtype],
        out: T.Tensor[(num_tokens, hidden) if without_transpose else (hidden, num_tokens), out_config.dtype],
        out_sf: T.Tensor[(num_tokens // num_per_tokens, hidden), T.float32],
    ):
        with T.Kernel(hidden // TILE_Y, num_tokens // TILE_X, threads=num_threads) as (pid_y, pid_x):
            # Add padding (TILE_K) to reduce bank conflicts
            act_shared = T.alloc_shared((TILE_X, TILE_Y) if without_transpose else (TILE_Y, TILE_X + TILE_K), in_dtype)
            tid = T.get_thread_binding()
            T.assume(num_tokens % TILE_X == 0)

            # Per channel cast
            if without_transpose:
                out_shared = T.alloc_shared((TILE_X, TILE_Y), out_config.dtype)
                row, col = tid // num_threads_per_global_token, tid % num_threads_per_global_token
                tmp_l_local = T.alloc_local((num_vec,), in_dtype)
                tmp_r_local = T.alloc_local((num_vec,), in_dtype)
                x_act_local = T.alloc_local((TILE_K,), T.float32)
                for i_ in T.unroll(TILE_X // thread_global_step):
                    i = i_ * thread_global_step + row
                    for j in T.vectorized(num_vec):
                        tmp_l_local[j] = x[pid_x * TILE_X + i, pid_y * TILE_Y + col * num_vec + j]
                        tmp_r_local[j] = x[pid_x * TILE_X + i, pid_y * TILE_Y + col * num_vec + hidden + j]
                    for j in T.unroll(num_vec // TILE_K):
                        for k in T.unroll(TILE_K):
                            val_l = T.alloc_var(T.float32)
                            val_r = T.alloc_var(T.float32)
                            if use_clamp:
                                val_l = T.min(tmp_l_local[j * TILE_K + k], swiglu_clamp_value)
                                val_r = T.max(T.min(tmp_r_local[j * TILE_K + k], swiglu_clamp_value), -swiglu_clamp_value)
                            else:
                                val_l = T.float32(tmp_l_local[j * TILE_K + k])
                                val_r = T.float32(tmp_r_local[j * TILE_K + k])

                            val = val_l / (1 + T.exp(-val_l)) * val_r
                            x_act_local[k] = val
                        for k in T.vectorized(TILE_K):
                            act_shared[i, col * num_vec + j * TILE_K + k] = x_act_local[k]

                T.sync_threads()
                shared_row = tid // num_threads_per_shared_token
                shared_col = tid % num_threads_per_shared_token
                tmp_local = T.alloc_local((TILE_K,), in_dtype)
                amax_local = T.alloc_local((TILE_K // 2,), T.bfloat16x2)
                amax_local_view = T.view(amax_local, (TILE_K, ), T.bfloat16)
                amax_shared = T.alloc_shared((num_threads // num_threads_per_shared_token, TILE_Y), T.bfloat16)
                sf_shared = T.alloc_shared((num_split_blocks, TILE_Y), T.float32)

                for i_ in T.unroll(TILE_X // thread_shared_step):
                    i = i_ + shared_row * (TILE_X // thread_shared_step)
                    for j in T.vectorized(TILE_K):
                        tmp_local[j] = act_shared[i, shared_col * TILE_K + j]
                    for j in T.unroll(TILE_K // 2):
                        packed = T.bfloat16x2(tmp_local[j * 2], tmp_local[j * 2 + 1])
                        if i_ == 0:
                            amax_local[j] = T.abs2(packed)
                        else:
                            amax_local[j] = T.max2(T.abs2(packed), amax_local[j])

                for j in T.vectorized(TILE_K):
                    amax_shared[shared_row, shared_col * TILE_K + j] = amax_local_view[j]

                T.sync_threads()
                amax_var = T.alloc_var(T.bfloat16, init=0.0)
                if tid < TILE_Y * num_split_blocks:
                    row_offset = tid // TILE_Y
                    col_offset = tid % TILE_Y
                    for i in T.unroll(thread_shared_step // num_split_blocks):
                        amax_var = T.max(amax_var, amax_shared[row_offset * (thread_shared_step // num_split_blocks) + i, col_offset])
                    sf, sf_inv = get_sf_and_inv(T.float32(amax_var), out_config)
                    out_sf[pid_x * (TILE_X // num_per_tokens) + row_offset, pid_y * TILE_Y + col_offset] = sf
                    sf_shared[row_offset, col_offset] = sf_inv
                T.sync_threads()

                for i, j in T.Parallel(TILE_X, TILE_Y):
                    out_shared[i, j] = act_shared[i, j] * sf_shared[i // num_per_tokens, j]
                T.copy(out_shared, out[pid_x * TILE_X, pid_y * TILE_Y], disable_tma=True)

            else:
                row, col = tid // num_threads_per_shared_token, tid % num_threads_per_shared_token
                x_act_local = T.alloc_local((TILE_K, TILE_K), in_dtype)
                tmp_l = T.alloc_local((TILE_K,), in_dtype)
                tmp_r = T.alloc_local((TILE_K,), in_dtype)

                # Swiglu forward & transpose
                for i_ in T.unroll(TILE_X // TILE_K // thread_shared_step):
                    i = i_ * thread_shared_step + row
                    # Read into registers
                    for j in T.unroll(TILE_K):
                        for k in T.vectorized(TILE_K):
                            tmp_l[k] = x[pid_x * TILE_X + i * TILE_K + j, pid_y * TILE_Y + col * TILE_K + k]
                        for k in T.vectorized(TILE_K):
                            tmp_r[k] = x[pid_x * TILE_X + i * TILE_K + j, pid_y * TILE_Y + col * TILE_K + k + hidden]
                        for k in T.unroll(TILE_K):
                            val_l = T.alloc_var(T.float32)
                            val_r = T.alloc_var(T.float32)
                            if use_clamp:
                                val_l = T.min(tmp_l[k], swiglu_clamp_value)
                                val_r = T.max(T.min(tmp_r[k], swiglu_clamp_value), -swiglu_clamp_value)
                            else:
                                val_l = T.float32(tmp_l[k])
                                val_r = T.float32(tmp_r[k])

                            val = val_l / (1 + T.exp(-val_l)) * val_r
                            x_act_local[k, j] = val

                    for j in T.unroll(TILE_K):
                        for k in T.vectorized(TILE_K):
                            # Accept 4x bank conflicts here, because swizzle overhead outweighs benefits
                            act_shared[col * TILE_K + j, i * TILE_K + k] = x_act_local[j, k]

                # Use multiple stages to reduce register pressure
                num_stages = 4
                tile_y_per_stage = TILE_Y // num_stages
                out_fragment = T.alloc_fragment((tile_y_per_stage, TILE_X), T.float32)
                amax_fragment = T.alloc_fragment((tile_y_per_stage, TILE_X // num_per_tokens), T.float32)
                for stage in T.unroll(num_stages):
                    T.copy(act_shared[tile_y_per_stage * stage : tile_y_per_stage * (stage + 1), 0:TILE_X], out_fragment)
                    out_fragment_reshaped = T.reshape(out_fragment, (tile_y_per_stage, TILE_X // num_per_tokens, num_per_tokens))
                    T.reduce_absmax(out_fragment_reshaped, amax_fragment, dim=2)

                    for i, j in T.Parallel(tile_y_per_stage, TILE_X // num_per_tokens):
                        sf, sf_inv = get_sf_and_inv(T.cast(amax_fragment[i, j], T.float32), out_config)
                        out_sf[pid_x * (TILE_X // num_per_tokens) + j, pid_y * TILE_Y + stage * tile_y_per_stage + i] = sf
                        amax_fragment[i, j] = sf_inv

                    for i, j in T.Parallel(tile_y_per_stage, TILE_X):
                        out[pid_y * TILE_Y + stage * tile_y_per_stage + i, pid_x * TILE_X + j] = (
                            out_fragment[i, j] * amax_fragment[i, j // num_per_tokens]
                        )

    return swiglu_forward_and_per_channel_cast_and_transpose_kernel


def swiglu_forward_and_per_channel_cast_and_transpose(
    x: torch.Tensor,
    fmt: str,
    num_per_tokens: int,
    round_sf: bool = False,
    without_transpose: bool = False,
    swiglu_clamp_value: Optional[float] = None,
) -> QuantTensor:
    """Fuse SwiGLU forward pass with per-channel FP8 cast and optional transpose.

    Args:
        x: Input 2D contiguous BF16 tensor of shape (num_tokens, hidden * 2).
        fmt: Target FP8 format (must be ``'e4m3'``).
        num_per_tokens: Number of tokens in each scaling block, must be 128.
        round_sf: Whether to round scaling factors to powers of two.
        without_transpose: If True, output keeps (num_tokens, hidden) layout
            instead of transposed (hidden, num_tokens).
        swiglu_clamp_value: Optional clamp threshold for SwiGLU activations.

    Returns:
        A tuple ``(out, out_sf)`` with FP8 output and sf-factor tensor.
    """
    assert fmt == 'e4m3'
    assert x.dim() == 2 and x.is_contiguous()
    assert x.dtype == torch.bfloat16
    num_tokens, hidden = x.shape

    # Swiglu forward : hidden -> hidden // 2
    assert num_tokens % 128 == 0 and hidden % 128 == 0
    assert num_per_tokens in (32, 128)

    hidden = hidden // 2
    use_clamp = swiglu_clamp_value is not None

    # Get kernel implement
    out_config = get_cast_output_config(fmt, (num_per_tokens, 1), round_sf=round_sf)
    kernel = get_swiglu_forward_and_per_channel_cast_and_transpose_kernel(
        hidden=hidden,
        without_transpose=without_transpose,
        use_clamp=use_clamp,
        in_dtype=T.dtype(x.dtype),
        out_config=out_config,
        swiglu_clamp_value=swiglu_clamp_value,
    )

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    # Allocate output and launch
    out = torch.empty((num_tokens, hidden) if without_transpose else (hidden, num_tokens), dtype=torch.float8_e4m3fn, device=x.device)
    out_sf = torch.empty((num_tokens // num_per_tokens, hidden), dtype=torch.float32, device=x.device)
    if num_tokens > 0:
        kernel(x, out, out_sf)

    return out, out_sf
