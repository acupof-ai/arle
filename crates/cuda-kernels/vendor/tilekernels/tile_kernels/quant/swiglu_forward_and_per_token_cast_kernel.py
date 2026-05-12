import os
import torch
import tilelang
from tilelang import language as T
from tile_kernels.utils import is_power_of_two
from tile_kernels.config import get_num_sms
from tile_kernels.quant.common import *
from typing import Optional


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_swiglu_forward_and_per_token_cast_kernel(
    hidden: int,
    with_weight: bool,
    with_pos_to_expert: bool,
    use_clamp: bool,
    count_clamp: bool,
    in_dtype: T.dtype,
    out_config: CastOutputConfig,
    num_sms: Optional[int],
):
    num_elems_per_block = 4096
    num_threads = 256
    _, num_per_channels = out_config.sf_block

    TILE_X = 1
    TILE_Y = num_per_channels

    while TILE_Y * 2 <= num_elems_per_block and hidden % (TILE_Y * 2) == 0:
        TILE_Y *= 2

    while TILE_X * TILE_Y % num_threads != 0:
        TILE_X *= 2

    if TILE_X != 1 or TILE_Y < 2048:
        if TILE_X == 1 and hidden <= 8192:
            TILE_Y = hidden

        if is_power_of_two(TILE_Y):
            while TILE_X * TILE_Y * 2 <= num_elems_per_block:
                TILE_X *= 2

    # Runtime symbols
    num_expanded_tokens = T.dynamic('num_expanded_tokens')
    num_tokens = T.dynamic('num_tokens')
    num_topk = T.dynamic('num_topk')
    sf_shape = get_sf_shape((num_expanded_tokens, hidden), out_config)
    sf_stride = T.dynamic('sf_stride')

    num_blocks = T.ceildiv(num_expanded_tokens, TILE_X) * T.ceildiv(hidden, TILE_Y)
    # If need to count clamp, use persistent kernel
    if count_clamp:
        num_blocks = num_sms * 4

    num_groups = TILE_Y // num_per_channels

    @T.prim_func
    def swiglu_forward_and_per_token_cast_kernel(
        x: T.Tensor[(num_expanded_tokens, hidden * 2), in_dtype],
        out: T.Tensor[(num_expanded_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],
        pos_to_token_topk: T.Tensor[(num_expanded_tokens,), T.int32],
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        pos_to_expert: T.Tensor[(num_expanded_tokens,), T.int32],
        clamped_count: T.Tensor[(3,), T.int64],
        swiglu_clamp_value: T.float32,
    ):
        with T.Kernel(num_blocks, threads=num_threads) as pid:
            tid = T.get_thread_binding()

            topk_weights_1d = T.reshape(topk_weights, (num_tokens * num_topk,))
            x_fragment = T.alloc_fragment((TILE_X, TILE_Y), T.float32)
            x_fragment_reshaped = T.reshape(x_fragment, [TILE_X, num_groups, num_per_channels])
            xl_fragment = T.alloc_fragment((TILE_X, TILE_Y), in_dtype)
            xr_fragment = T.alloc_fragment((TILE_X, TILE_Y), in_dtype)

            count_silu = T.alloc_reducer((1,), T.int64, 'sum', replication='all')
            count_upper = T.alloc_reducer((1,), T.int64, 'sum', replication='all')
            count_lower = T.alloc_reducer((1,), T.int64, 'sum', replication='all')

            T.fill(count_silu, 0)
            T.fill(count_upper, 0)
            T.fill(count_lower, 0)

            if count_clamp:
                upper = T.ceildiv(T.ceildiv(num_expanded_tokens, TILE_X) * T.ceildiv(hidden, TILE_Y) - pid, num_blocks)
            else:
                upper = 1

            for iter in T.serial(upper):
                pid_iter = iter * num_blocks + pid
                pid_x, pid_y = pid_iter // T.ceildiv(hidden, TILE_Y), pid_iter % T.ceildiv(hidden, TILE_Y)

                topk_weights_fragment = T.alloc_fragment((TILE_X,), T.float32)
                pos_to_expert_fragment = T.alloc_fragment((TILE_X,), T.int32)
                sf_inv_fragment = T.alloc_fragment((TILE_X, num_groups), T.float32)
                out_fragment = T.alloc_fragment((TILE_X, TILE_Y), out_config.dtype)

                if with_weight:
                    for i in T.Parallel(TILE_X):
                        pos = pos_to_token_topk[pid_x * TILE_X + i]
                        if pos >= 0:
                            T.assume(pos < num_tokens * num_topk)
                            topk_weights_fragment[i] = topk_weights_1d[pos]

                if with_pos_to_expert:
                    for i in T.Parallel(TILE_X):
                        pos_to_expert_fragment[i] = pos_to_expert[pid_x * TILE_X + i]

                if not with_pos_to_expert or TILE_X != 1 or pos_to_expert_fragment[0] >= 0:
                    for i, j in T.Parallel(TILE_X, TILE_Y):
                        if (not with_pos_to_expert) or pos_to_expert_fragment[i] >= 0:
                            xl_fragment[i, j] = x[pid_x * TILE_X + i, pid_y * TILE_Y + j]
                            xr_fragment[i, j] = x[pid_x * TILE_X + i, pid_y * TILE_Y + j + hidden]

                    for i, j in T.Parallel(TILE_X, TILE_Y):
                        if (not with_pos_to_expert) or pos_to_expert_fragment[i] >= 0:
                            val_l = T.alloc_var(T.float32)
                            val_r = T.alloc_var(T.float32)
                            val_l = T.float32(xl_fragment[i, j])
                            val_r = T.float32(xr_fragment[i, j])
                            if use_clamp:
                                if count_clamp:
                                    clamp_silu = val_l > swiglu_clamp_value
                                    val_l = T.Select(clamp_silu, swiglu_clamp_value, val_l)
                                    count_silu[0] += clamp_silu
                                    clamp_upper = val_r > swiglu_clamp_value
                                    clamp_lower = val_r < -swiglu_clamp_value
                                    val_r = T.Select(clamp_upper, swiglu_clamp_value, val_r)
                                    val_r = T.Select(clamp_lower, -swiglu_clamp_value, val_r)
                                    count_upper[0] += clamp_upper
                                    count_lower[0] += clamp_lower
                                else:
                                    val_l = T.min(val_l, swiglu_clamp_value)
                                    val_r = T.max(T.min(val_r, swiglu_clamp_value), -swiglu_clamp_value)
                            if with_weight:
                                val = val_l / (1 + T.exp(-val_l)) * val_r * topk_weights_fragment[i]
                            else:
                                val = val_l / (1 + T.exp(-val_l)) * val_r
                            x_fragment[i, j] = val

                    # Reduce SF
                    T.reduce_absmax(x_fragment_reshaped, sf_inv_fragment, dim=2)
                    for i, j in T.Parallel(TILE_X, num_groups):
                        if (not with_pos_to_expert) or pos_to_expert_fragment[i] >= 0:
                            sf, sf_inv = get_sf_and_inv(sf_inv_fragment[i, j], out_config)
                            x_idx = pid_x * TILE_X + i
                            y_idx = pid_y * num_groups + j
                            store_sf(out_sf, sf, x_idx, y_idx, out_config)
                            sf_inv_fragment[i, j] = sf_inv

                    # Store casted values
                    for i, j in T.Parallel(TILE_X, TILE_Y):
                        if (not with_pos_to_expert) or pos_to_expert_fragment[i] >= 0:
                            out_fragment[i, j] = x_fragment[i, j] * sf_inv_fragment[i, j // num_per_channels]
                    T.copy(out_fragment, out[pid_x * TILE_X, pid_y * TILE_Y])

            if count_clamp:
                T.finalize_reducer(count_silu)
                T.finalize_reducer(count_upper)
                T.finalize_reducer(count_lower)

                if tid == 0:
                    T.atomic_add(clamped_count[0], count_silu[0])
                    T.atomic_add(clamped_count[1], count_upper[0])
                    T.atomic_add(clamped_count[2], count_lower[0])

    return swiglu_forward_and_per_token_cast_kernel


def swiglu_forward_and_per_token_cast(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    pos_to_token_topk: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    pos_to_expert: Optional[torch.Tensor] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    swiglu_clamp_value: Optional[float] = None,
    clamped_count: Optional[torch.Tensor] = None,
    sf_clamp_min: Optional[float] = None,
) -> QuantTensor:
    """Fuse SwiGLU forward pass with per-token FP8 quantization.

    Args:
        x: Input 2D contiguous tensor of shape (num_expanded_tokens, hidden * 2).
        fmt: Target FP8 format (must be ``'e4m3'``).
        num_per_channels: Number of channels in each scaling block (0 = hidden).
        pos_to_token_topk: Optional mapping from expanded position to (token, topk) index.
        topk_weights: Optional top-k routing weights of shape (num_tokens, num_topk).
        pos_to_expert: Optional mapping from expanded position to expert index.
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.
        swiglu_clamp_value: Optional clamp threshold for SwiGLU activations.
        clamped_count: Optional int64 tensor of shape (3,) to accumulate clamp counts.
        sf_clamp_min: Optional custom minimum clamp value for sf factors.

    Returns:
        A tuple ``(out, out_sf)`` with FP8 output and sf-factor tensor.
    """
    assert x.dim() == 2 and x.is_contiguous()
    num_expanded_tokens, hidden = x.shape
    hidden = hidden // 2

    if pos_to_token_topk is not None:
        assert pos_to_token_topk.dim() == 1
        assert x.shape[0] == num_expanded_tokens
        assert topk_weights is not None
        assert topk_weights.dim() == 2

    if pos_to_expert is not None:
        assert pos_to_expert.dim() == 1
        assert pos_to_expert.shape[0] == num_expanded_tokens

    if clamped_count is not None:
        assert swiglu_clamp_value is not None
        assert clamped_count.dim() == 1
        assert clamped_count.shape[0] == 3

    # Swiglu forward : hidden -> hidden // 2
    assert hidden % 128 == 0
    assert num_per_channels == 128 or num_per_channels == hidden
    assert num_per_channels == 128 or (not use_tma_aligned_col_major_sf)
    assert fmt == 'e4m3'

    # Get kernel implement
    out_config = get_cast_output_config(
        fmt, (1, num_per_channels), use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0, custom_clamp_min_value=sf_clamp_min
    )
    kernel = get_swiglu_forward_and_per_token_cast_kernel(
        hidden,
        pos_to_token_topk is not None,
        pos_to_expert is not None,
        swiglu_clamp_value is not None,
        clamped_count is not None,
        in_dtype=T.dtype(x.dtype),
        out_config=out_config,
        num_sms=get_num_sms() if clamped_count is not None else None,
    )

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    # Allocate output and launch
    out = torch.empty((num_expanded_tokens, hidden), dtype=torch.float8_e4m3fn, device='cuda')
    out_sf = alloc_scaling_factors((num_expanded_tokens, hidden), out_config)
    swiglu_clamp_value = 0 if swiglu_clamp_value is None else swiglu_clamp_value
    if num_expanded_tokens > 0:
        kernel(x, out, out_sf, pos_to_token_topk, topk_weights, pos_to_expert, clamped_count, swiglu_clamp_value)

    out_sf = cast_epilogue(out_sf, num_expanded_tokens, hidden, out_config)

    return out, out_sf
