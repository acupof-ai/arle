import os
import torch
import tilelang
from tilelang import language as T

from tile_kernels.utils import align
from tile_kernels.quant.common import *
from typing import Optional


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_swiglu_backward_and_per_token_cast_kernel(
    hidden: int,
    out_config: CastOutputConfig,
    use_clamp: bool,
):
    num_threads = 64
    align_length = 512
    hidden_aligned = align(hidden, align_length)
    _, num_per_channels = out_config.sf_block
    num_sf_per_align = align_length // num_per_channels

    # Runtime symbols
    num_expand_tokens = T.dynamic('num_expand_tokens')
    num_tokens = T.dynamic('num_tokens')
    num_topk = T.dynamic('num_topk')

    num_blocks = T.max(num_expand_tokens, num_tokens * num_topk)

    @T.prim_func
    def swiglu_backward_and_per_token_cast_kernel(
        x: T.Tensor[(num_expand_tokens, hidden * 2), out_config.dtype],
        x_sf: T.Tensor[(num_expand_tokens, hidden * 2 // num_per_channels), T.float32],
        grad_out: T.Tensor[(num_expand_tokens, hidden), T.bfloat16],
        weight: T.Tensor[(num_tokens, num_topk), T.float32],
        pos_to_token_topk: T.Tensor[(num_expand_tokens, ), T.int32],
        token_topk_to_pos: T.Tensor[(num_tokens, num_topk), T.int32],
        out: T.Tensor[(num_expand_tokens, hidden), T.bfloat16],
        x_grad_fp8: T.Tensor[(num_expand_tokens, hidden * 2), out_config.dtype],
        x_grad_fp8_sf_invs: T.Tensor[(num_expand_tokens, hidden * 2 // num_per_channels), T.float32],
        x_grad: T.Tensor[(num_expand_tokens, hidden * 2), T.bfloat16],
        weight_grad: T.Tensor[(num_tokens, num_topk), T.float32],
        swiglu_clamp_value: T.float32,
    ):
        with T.Kernel(num_blocks, threads=num_threads) as (pid_token, ):
            x_fragment = T.alloc_fragment((align_length, ), out_config.dtype)
            y_fragment = T.alloc_fragment((align_length, ), out_config.dtype)
            x_sf_fragment = T.alloc_fragment((num_sf_per_align, ), T.float32)
            y_sf_fragment = T.alloc_fragment((num_sf_per_align, ), T.float32)
            grad_out_fragment = T.alloc_fragment((align_length, ), T.bfloat16)

            x_grad_fragment = T.alloc_fragment((align_length, ), T.float32)
            y_grad_fragment = T.alloc_fragment((align_length, ), T.float32)
            x_grad_fragment_reshaped = T.reshape(x_grad_fragment, (num_sf_per_align, num_per_channels))
            y_grad_fragment_reshaped = T.reshape(y_grad_fragment, (num_sf_per_align, num_per_channels))
            xmax_fragment = T.alloc_fragment((num_sf_per_align, ), T.float32)
            ymax_fragment = T.alloc_fragment((num_sf_per_align, ), T.float32)
            out_fragment = T.alloc_fragment((align_length, ), T.bfloat16)

            acc = T.alloc_reducer((1, ), T.float32, 'sum', replication='all')
            T.fill(acc, 0.0)

            if pid_token < num_tokens * num_topk:
                pos = token_topk_to_pos[pid_token // num_topk, pid_token % num_topk]
                if pos == -1:
                    weight_grad[pid_token // num_topk, pid_token % num_topk] = 0

            if pid_token >= num_expand_tokens:
                T.thread_return()

            index = pos_to_token_topk[pid_token]
            T.assume(index < num_tokens * num_topk)
            w = T.Select(index >= 0, weight[index // num_topk, index % num_topk], 0.0)

            for k in T.serial(hidden_aligned // align_length):
                for i in T.Parallel(align_length):
                    if i + k * align_length < hidden:
                        x_fragment[i] = x[pid_token, i + k * align_length]
                        y_fragment[i] = x[pid_token, i + hidden + k * align_length]
                        grad_out_fragment[i] = grad_out[pid_token, i + k * align_length]

                for i in T.Parallel(num_sf_per_align):
                    if i + k * num_sf_per_align < hidden // num_per_channels:
                        x_sf_fragment[i] = x_sf[pid_token, i + k * num_sf_per_align]
                        y_sf_fragment[i] = x_sf[pid_token, i + hidden // num_per_channels + k * num_sf_per_align]

                for i in T.Parallel(align_length):
                    if i + k * align_length < hidden:
                        x_fp32 = T.alloc_var(T.float32)
                        y_fp32 = T.alloc_var(T.float32)
                        x_fp32 = x_fragment[i] * x_sf_fragment[i // num_per_channels]
                        y_fp32 = y_fragment[i] * y_sf_fragment[i // num_per_channels]

                        is_clamped_x = x_fp32 > swiglu_clamp_value if use_clamp else False
                        is_clamped_y_upper = y_fp32 > swiglu_clamp_value if use_clamp else False
                        is_clamped_y_lower = y_fp32 < -swiglu_clamp_value if use_clamp else False
                        is_clamped_y = is_clamped_y_upper or is_clamped_y_lower

                        if use_clamp:
                            x_fp32 = T.Select(is_clamped_x, swiglu_clamp_value, x_fp32)
                            y_fp32 = T.Select(is_clamped_y_upper, swiglu_clamp_value, y_fp32)
                            y_fp32 = T.Select(is_clamped_y_lower, -swiglu_clamp_value, y_fp32)

                        grad_out_fp32 = T.cast(grad_out_fragment[i], T.float32)
                        tmp_fp32 = 1 + T.exp(-x_fp32)
                        act_out = x_fp32 / tmp_fp32 * y_fp32
                        acc[0] += grad_out_fp32 * act_out
                        s_fp32 = 1.0 / tmp_fp32
                        grad_out_fp32_ws = grad_out_fp32 * w * s_fp32

                        # Write to fragment
                        out_fragment[i] = act_out * w
                        x_grad_fragment[i] = T.Select(is_clamped_x, 0.0, grad_out_fp32_ws * y_fp32 * (1 + x_fp32 * (1 - s_fp32)))
                        y_grad_fragment[i] = T.Select(is_clamped_y, 0.0, grad_out_fp32_ws * x_fp32)

                for i in T.Parallel(align_length):
                    if i + k * align_length < hidden:
                        x_grad[pid_token, i + k * align_length] = x_grad_fragment[i]
                        x_grad[pid_token, i + hidden + k * align_length] = y_grad_fragment[i]
                        out[pid_token, i + k * align_length] = out_fragment[i]

                T.reduce_absmax(x_grad_fragment_reshaped, xmax_fragment, dim=1)
                T.reduce_absmax(y_grad_fragment_reshaped, ymax_fragment, dim=1)

                for i in T.Parallel(num_sf_per_align):
                    if i + k * num_sf_per_align < hidden // num_per_channels:
                        x_sf, x_sf_inv = get_sf_and_inv(xmax_fragment[i], out_config)
                        y_sf, y_sf_inv = get_sf_and_inv(ymax_fragment[i], out_config)
                        x_grad_fp8_sf_invs[pid_token, i + k * num_sf_per_align] = x_sf
                        x_grad_fp8_sf_invs[pid_token, i + hidden // num_per_channels + k * num_sf_per_align] = y_sf
                        xmax_fragment[i] = x_sf_inv
                        ymax_fragment[i] = y_sf_inv

                x_grad_fp8_fragment = T.alloc_fragment((align_length, ), out_config.dtype)
                y_grad_pt_fragment = T.alloc_fragment((align_length, ), out_config.dtype)

                for i in T.Parallel(align_length):
                    if i + k * align_length < hidden:
                        x_grad_fp8_fragment[i] = x_grad_fragment[i] * xmax_fragment[i // num_per_channels]
                        y_grad_pt_fragment[i] = y_grad_fragment[i] * ymax_fragment[i // num_per_channels]

                for i in T.Parallel(align_length):
                    if i + k * align_length < hidden:
                        x_grad_fp8[pid_token, i + k * align_length] = x_grad_fp8_fragment[i]
                        x_grad_fp8[pid_token, i + hidden + k * align_length] = y_grad_pt_fragment[i]

            T.finalize_reducer(acc)
            if index >= 0:
                weight_grad[index // num_topk, index % num_topk] = acc[0]

    return swiglu_backward_and_per_token_cast_kernel


def swiglu_backward_and_per_token_cast(
    x: QuantTensor,
    grad_out: torch.Tensor,
    weight: torch.Tensor,
    pos_to_token_topk: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    num_per_channels: int,
    round_sf: bool = False,
    swiglu_clamp_value: Optional[float] = None
) -> tuple[torch.Tensor, QuantTensor, torch.Tensor, torch.Tensor]:
    """Fuse SwiGLU backward pass with per-token FP8 quantization of the input gradient.

    Args:
        x: FP8 forward input as a ``QuantTensor`` ``(data, sf)`` where data has
            shape (num_expand_tokens, hidden * 2) and sf has shape
            (num_expand_tokens, hidden * 2 // num_per_channels).
        grad_out: BF16 gradient of shape (num_expand_tokens, hidden).
        weight: Top-k routing weights of shape (num_tokens, num_topk).
        pos_to_token_topk: Mapping from expanded position to (token, topk) index.
        token_topk_to_pos: Mapping from (token, topk) to expanded position.
        num_per_channels: Number of channels in each scaling block (32 or 128).
        round_sf: Whether to round scaling factors to powers of two.
        swiglu_clamp_value: Optional clamp threshold for SwiGLU activations.

    Returns:
        A tuple ``(out, (x_grad_fp8, x_grad_fp8_sf_invs), x_grad, weight_grad)``
        containing BF16 SwiGLU output, quantized input gradient pair, BF16 input
        gradient, and routing weight gradient.
    """
    # Only support num_per_channels in (32, 128)
    assert num_per_channels in (32, 128)
    x, x_sf = x

    assert x.dtype == torch.float8_e4m3fn
    assert (x.dim() == 2 or x.dim() == 3) and x.is_contiguous()
    assert x_sf.dim() == 2 and x_sf.is_contiguous()
    assert weight.dim() == 2 and weight.is_contiguous()
    assert pos_to_token_topk.dim() == 1
    assert token_topk_to_pos.dim() == 2 and token_topk_to_pos.is_contiguous()

    # Assert `hidden % (2 * num_per_channels) == 0`
    assert x.size(-1) % (2 * num_per_channels) == 0
    hidden = x.size(-1) // 2

    x = x.view(-1, hidden * 2)
    grad_out = grad_out.view(-1, hidden)
    num_expand_tokens = x.size(0)
    num_tokens, num_topk = token_topk_to_pos.shape

    assert x_sf.shape == (num_expand_tokens, 2 * hidden // num_per_channels)
    assert grad_out.shape == (num_expand_tokens, hidden)
    assert weight.shape == (num_tokens, num_topk)
    assert pos_to_token_topk.shape == (num_expand_tokens, )
    assert token_topk_to_pos.shape == (num_tokens, num_topk)

    fmt = 'e4m3'
    out_config = get_cast_output_config(fmt, (1, num_per_channels), round_sf=round_sf)

    # Get kernel implement
    kernel = get_swiglu_backward_and_per_token_cast_kernel(
        hidden,
        out_config,
        swiglu_clamp_value is not None,
    )

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    # Allocate output and launch
    out = torch.empty((num_expand_tokens, hidden), dtype=grad_out.dtype, device=grad_out.device)
    x_grad_fp8 = torch.empty((num_expand_tokens, hidden * 2), dtype=x.dtype, device=x.device)
    x_grad_fp8_sf_invs = torch.empty((num_expand_tokens, 2 * hidden // num_per_channels), dtype=torch.float32, device=x.device)
    x_grad = torch.empty((num_expand_tokens, hidden * 2), dtype=grad_out.dtype, device=grad_out.device)
    weight_grad = torch.empty((num_tokens, num_topk), dtype=torch.float32, device=x.device)

    swiglu_clamp_value = 0 if swiglu_clamp_value is None else swiglu_clamp_value
    if num_expand_tokens > 0:
        kernel(x, x_sf, grad_out, weight, pos_to_token_topk, token_topk_to_pos, out, x_grad_fp8, x_grad_fp8_sf_invs, x_grad, weight_grad, swiglu_clamp_value)

    return out, (x_grad_fp8, x_grad_fp8_sf_invs), x_grad, weight_grad
