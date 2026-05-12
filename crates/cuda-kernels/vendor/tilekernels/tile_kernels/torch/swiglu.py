from typing import Optional, Tuple

import torch
from torch.types import Number

from tile_kernels.quant.types import QuantTensor


def swiglu_forward(
    x: torch.Tensor,
    pos_to_token_topk: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    swiglu_clamp_value: Optional[float] = None,
    clamped_count: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    PyTorch implementation of the SwiGLU forward pass.

    Computes ``silu(x_left) * x_right`` where ``x_left`` and ``x_right`` are
    the two halves of the last dimension of ``x``, then optionally scales each
    row by its corresponding top-k routing weight.

    Args:
        x: Input 2D contiguous tensor of shape ``(num_expanded_tokens, hidden * 2)``
            in BF16 or FP32.
        pos_to_token_topk: Optional 1-D int32 tensor of shape
            ``(num_expanded_tokens,)`` mapping each expanded position to a
            flat ``(token, topk)`` index into ``topk_weights``.  Entries that
            are ``< 0`` indicate padding and the corresponding output rows are
            left as zero.
        topk_weights: Optional 2-D float32 tensor of shape
            ``(num_tokens, num_topk)`` containing routing weights.  Required
            when ``pos_to_token_topk`` is provided.
        swiglu_clamp_value: Optional clamp threshold applied before the
            activation.  ``x_left`` is clamped to
            ``(-inf, swiglu_clamp_value]`` and ``x_right`` is clamped to
            ``[-swiglu_clamp_value, swiglu_clamp_value]``.
        clamped_count: Optional 1-D int64 tensor of length 3.  When provided
            alongside ``swiglu_clamp_value``, the counts of clamped elements
            are added in-place: index 0 counts ``x_left > swiglu_clamp_value``,
            index 1 counts ``x_right > swiglu_clamp_value``, index 2 counts
            ``x_right < -swiglu_clamp_value``.

    Returns:
        FP32 output tensor of shape ``(num_expanded_tokens, hidden)``.
    """
    assert x.dim() == 2 and x.is_contiguous()
    assert x.dtype in (torch.bfloat16, torch.float32)

    num_expanded_tokens, hidden2 = x.shape
    assert hidden2 % 2 == 0
    hidden = hidden2 // 2

    if pos_to_token_topk is not None:
        assert pos_to_token_topk.dim() == 1
        assert pos_to_token_topk.shape[0] == num_expanded_tokens
        assert topk_weights is not None
        assert topk_weights.dim() == 2

    # Split into left (gate) and right (value) halves
    x_fp32 = x.float()
    x_left = x_fp32[:, :hidden]
    x_right = x_fp32[:, hidden:]

    # Optional clamp before activation
    if swiglu_clamp_value is not None:
        if clamped_count is not None:
            clamped_count[0] += (x_left > swiglu_clamp_value).sum()
            clamped_count[1] += (x_right > swiglu_clamp_value).sum()
            clamped_count[2] += (x_right < -swiglu_clamp_value).sum()
        x_left = torch.clamp(x_left, max=swiglu_clamp_value)
        x_right = torch.clamp(x_right, min=-swiglu_clamp_value, max=swiglu_clamp_value)

    # SwiGLU: silu(x_left) * x_right  where silu(x) = x * sigmoid(x)
    out = x_left / (1.0 + torch.exp(-x_left)) * x_right

    # Optional per-row weight scaling
    if pos_to_token_topk is not None:
        num_tokens, num_topk = topk_weights.shape
        pos_mask = pos_to_token_topk >= 0
        token_indices = torch.div(pos_to_token_topk[pos_mask], num_topk, rounding_mode='floor')
        topk_indices = pos_to_token_topk[pos_mask] % num_topk

        w_expanded = torch.zeros(num_expanded_tokens, device=x.device, dtype=torch.float32)
        w_expanded[pos_mask] = topk_weights[token_indices, topk_indices].float()
        out = out * w_expanded.unsqueeze(1)

    return out


# Explicit extract fma pattern for torch.compile to capture.
# This is for precision issue.
@torch.compile
def elementwise_fma(a: torch.Tensor, b: Number | torch.Tensor, c: Number | torch.Tensor) -> torch.Tensor:
    return a * b + c


def swiglu_backward(
    x: QuantTensor,
    grad_out: torch.Tensor,
    weight: torch.Tensor,
    pos_to_token_topk: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    num_per_channels: int,
    swiglu_clamp_value: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation of the SwiGLU backward pass.

    Args:
        x: Quantized input as a QuantTensor ``(data, sf)`` where data has shape
            ``(num_expand_tokens, hidden * 2)`` in FP8 format and sf has shape
            ``(num_expand_tokens, hidden * 2 // num_per_channels)``.
        grad_out: Gradient of output of shape (num_expand_tokens, hidden)
        weight: Weight tensor of shape (num_tokens, num_topk)
        pos_to_token_topk: Mapping from expanded token position to token-topk index
        token_topk_to_pos: Mapping from token-topk index to expanded token position
        num_per_channels: Number of channels per sf factor (32 or 128)
        swiglu_clamp_value: Clamp value for SwiGLU activation

    Returns:
        out: FP32 output tensor of shape (num_expand_tokens, hidden)
        x_grad: FP32 gradient of x, shape (num_expand_tokens, hidden * 2)
        weight_grad: Gradient of weight
    """
    x_data, x_sf = x

    # Only support num_per_channels in (32, 128)
    assert num_per_channels in (32, 128)

    assert (x_data.dim() == 2 or x_data.dim() == 3) and x_data.is_contiguous()
    assert x_sf.dim() == 2 and x_sf.is_contiguous()
    assert weight.dim() == 2 and weight.is_contiguous()
    assert pos_to_token_topk.dim() == 1
    assert token_topk_to_pos.dim() == 2 and token_topk_to_pos.is_contiguous()

    # Assert `hidden % num_per_channels == 0`
    assert x_data.size(-1) % (2 * num_per_channels) == 0
    hidden = x_data.size(-1) // 2

    x_data = x_data.view(-1, hidden * 2)
    grad_out = grad_out.view(-1, hidden)
    num_expand_tokens = x_data.size(0)
    num_tokens, num_topk = token_topk_to_pos.shape

    assert x_sf.shape == (num_expand_tokens, 2 * hidden // num_per_channels)
    assert grad_out.shape == (num_expand_tokens, hidden)
    assert weight.shape == (num_tokens, num_topk)
    assert pos_to_token_topk.shape == (num_expand_tokens,)
    assert token_topk_to_pos.shape == (num_tokens, num_topk)

    # Dequantize x from FP8 to FP32
    # Expand sf to match x shape
    x_sf_expanded = x_sf.repeat_interleave(num_per_channels, dim=1)
    x_fp32 = x_data.float() * x_sf_expanded

    # Split x into x and y parts
    x_part = x_fp32[:, :hidden]
    y_part = x_fp32[:, hidden:]

    # Apply SwiGLU clamp if needed
    use_clamp = swiglu_clamp_value is not None
    clamp_value = swiglu_clamp_value
    x_clamped = None
    y_clamped = None

    # Apply clamp
    if use_clamp:
        # For x: clamp when x > clamp_value
        x_clamped = x_part > clamp_value
        x_part[x_clamped] = clamp_value

        # For y: clamp when y > clamp_value or y < -clamp_value
        y_clamped_upper = y_part > clamp_value
        y_clamped_lower = y_part < -clamp_value
        y_clamped = y_clamped_upper | y_clamped_lower
        y_part[y_clamped_upper] = clamp_value
        y_part[y_clamped_lower] = -clamp_value

    # Compute SwiGLU activation: x * sigmoid(x) * y
    tmp_x = 1.0 + torch.exp(-x_part)
    sigmoid_x = torch.ones_like(x_part) / tmp_x

    # Compute output with weight scaling
    # Get weight for each expanded token
    pos_mask = pos_to_token_topk >= 0
    token_indices = torch.div(pos_to_token_topk[pos_mask], num_topk, rounding_mode='floor')
    topk_indices = pos_to_token_topk[pos_mask] % num_topk

    # Initialize weight tensor for expanded tokens
    w_expanded = torch.zeros(num_expand_tokens, device=x_data.device, dtype=torch.float32)
    w_expanded[pos_mask] = weight[token_indices, topk_indices]

    # Convert grad_out to FP32
    grad_out_fp32 = grad_out.float()

    # grad_out_ws = grad_out * w * s
    grad_out_ws = grad_out_fp32 * w_expanded.unsqueeze(1) * sigmoid_x

    # x_grad = grad_out_ws * y * (1 + x * (1 - s)) if not clamped
    x_grad = grad_out_ws * y_part * elementwise_fma(x_part, 1.0 - sigmoid_x, 1.0)

    # y_grad = grad_out_ws * x if not clamped
    y_grad = grad_out_ws * x_part

    # Apply clamp gradients
    if use_clamp:
        x_grad[x_clamped] = 0.0
        y_grad[y_clamped] = 0.0

    # Output
    act_out = x_part / tmp_x * y_part
    out = act_out * w_expanded.unsqueeze(1)

    # Combine x_grad and y_grad
    x_grad_full = torch.cat([x_grad, y_grad], dim=1)

    # Compute weight gradient: sum(grad_out * act_out) for each token-topk
    weight_grad = torch.zeros_like(weight)

    # Compute dot product for each expanded token
    dot_products = (grad_out_fp32 * act_out).sum(dim=1)

    # Accumulate to weight_grad based on pos_to_token_topk mapping
    weight_grad[token_indices, topk_indices] = dot_products[pos_mask]

    return out, x_grad_full, weight_grad
