from tilelang import language as T

from tile_kernels.quant.common import *

from .per_channel_cast_fused_kernel import per_channel_cast_fused


def per_channel_cast(x: torch.Tensor, fmt: str, num_per_tokens: int, round_sf: bool = False) -> QuantTensor:
    """Cast a matrix to FP8 with per-channel (column-wise) scaling factors.

    Args:
        x: Input 2D contiguous tensor of shape (num_tokens, hidden).
        fmt: Target FP8 format (must be ``'e4m3'``).
        num_per_tokens: Number of tokens in each scaling block.
        round_sf: Whether to round scaling factors to powers of two.

    Returns:
        A tuple ``(out, out_sf)`` with FP8 output and sf-factor tensor.
    """
    assert x.is_contiguous() and x.dim() == 2
    assert fmt == 'e4m3'

    num_tokens, hidden = x.shape

    assert num_tokens % 128 == 0 and hidden % 64 == 0
    assert num_per_tokens == 128

    return per_channel_cast_fused(x, 'e4m3', num_per_tokens=num_per_tokens, round_sf=round_sf)
