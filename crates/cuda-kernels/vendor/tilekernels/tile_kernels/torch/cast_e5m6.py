import torch
import torch.nn.functional as F


def right_shift_unsigned(x, shift):
    # CUDA torch does not support bit ops on uint32, so we need to mask to get unsigned right shift
    return (x >> shift) & ((1 << (32 - shift)) - 1)


def transform_sf(sf: torch.Tensor) -> torch.Tensor:
    if sf.dtype == torch.float32:
        return sf
    assert sf.dtype == torch.int32
    sf = sf.contiguous()
    if sf.stride(-1) != 1:
        sf = sf.as_strided(size=sf.shape, stride=(sf.shape[-1], 1))
    sf = sf.view(torch.uint8)
    sf = sf.to(torch.int32)
    sf = (sf << 23).view(torch.float32)
    return sf


def _make_col_major(sf: torch.Tensor, tma_alignment: int) -> torch.Tensor:
    """Return a column-major view of sf with TMA-aligned token dimension.

    The token dimension (dim 0) is padded to a multiple of ``tma_alignment``
    and the result is stored in column-major order (row stride = 1).
    """
    num_tokens, num_groups = sf.shape
    pad_tokens = (tma_alignment - num_tokens % tma_alignment) % tma_alignment
    num_tokens_padded = num_tokens + pad_tokens
    buf = torch.zeros(num_groups, num_tokens_padded, dtype=sf.dtype, device=sf.device)
    buf[:, :num_tokens] = sf.T
    return buf.T[:num_tokens, :]


def _float32_to_fp16_rtz_bits(x: torch.Tensor) -> torch.Tensor:
    x_bits = x.contiguous().view(torch.int32)
    sign = (x_bits >> 16) & 0x8000
    exp = (x_bits >> 23) & 0xFF
    mant = x_bits & 0x7FFFFF

    normal = (exp >= 113) & (exp <= 142)
    subnormal = (exp >= 103) & (exp <= 112)
    overflow = (exp > 142) & (exp < 255)
    underflow = exp < 103
    is_nan = exp == 255

    exp_f16 = (exp - 112).to(torch.int32)
    mant_f16 = (mant >> 13).to(torch.int32)

    shift = (113 - exp).to(torch.int32)
    mant_sub = right_shift_unsigned(0x800000 | mant, shift + 13)

    result = sign.to(torch.int32)
    result = torch.where(normal, result | (exp_f16 << 10) | mant_f16, result)
    result = torch.where(subnormal, result | mant_sub, result)
    result = torch.where(overflow | (is_nan & (mant == 0)), result | 0x7C00, result)
    result = torch.where(is_nan & (mant != 0), result | 0x7FFF, result)
    result = torch.where(underflow, sign.to(torch.int32), result)

    return result.to(torch.uint16)


def _cast_to_e5m6(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    assert x.dtype in (torch.float32, torch.bfloat16)
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float32)

    num_tokens, hidden = x.shape
    assert hidden % 8 == 0

    x_bits = x.contiguous().view(torch.int32)
    fp16_bits = _float32_to_fp16_rtz_bits(x)

    remain_bits = x_bits & 0x1FFFF
    e5m6_bits = right_shift_unsigned(fp16_bits.to(torch.int32), 4)
    lsb = e5m6_bits & 1
    cond = (lsb.to(torch.int64) + remain_bits.to(torch.int64)) > 0x10000
    e5m6_bits = (e5m6_bits + cond.to(torch.int32)) & 0xFFF

    e5m6 = e5m6_bits.to(torch.int64).view(num_tokens, hidden // 8, 8)

    h0 = e5m6[..., 0]
    h1 = e5m6[..., 1]
    h2 = e5m6[..., 2]
    h3 = e5m6[..., 3]
    h4 = e5m6[..., 4]
    h5 = e5m6[..., 5]
    h6 = e5m6[..., 6]
    h7 = e5m6[..., 7]

    w0 = (h0 << 20) | (h1 << 8) | (h2 >> 4)
    w1 = (h2 << 28) | (h3 << 16) | (h4 << 4) | (h5 >> 8)
    w2 = (h5 << 24) | (h6 << 12) | h7

    packed = torch.stack([w0, w1, w2], dim=-1)
    packed = packed.to(torch.uint32).view(num_tokens, hidden // 8 * 3)
    return packed.view(torch.uint8)


def _cast_back_from_e5m6(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.dtype == torch.uint8

    num_tokens = x.shape[0]
    packed_hidden = x.shape[1]
    hidden = packed_hidden * 2 // 3
    assert hidden % 8 == 0

    words = x.contiguous().view(torch.uint32)
    words = words.view(num_tokens, hidden // 8, 3)

    w0 = words[..., 0].to(torch.int64)
    w1 = words[..., 1].to(torch.int64)
    w2 = words[..., 2].to(torch.int64)

    f16_0 = ((w0 >> 16) & 0xFFF0).to(torch.int32)
    f16_1 = ((w0 >> 4) & 0xFFF0).to(torch.int32)
    f16_2 = (((w0 << 8) | (w1 >> 24)) & 0xFFF0).to(torch.int32)
    f16_3 = ((w1 >> 12) & 0xFFF0).to(torch.int32)
    f16_4 = (w1 & 0xFFF0).to(torch.int32)
    f16_5 = (((w1 << 12) | (w2 >> 20)) & 0xFFF0).to(torch.int32)
    f16_6 = ((w2 >> 8) & 0xFFF0).to(torch.int32)
    f16_7 = ((w2 << 4) & 0xFFF0).to(torch.int32)

    f16_all = torch.stack([f16_0, f16_1, f16_2, f16_3, f16_4, f16_5, f16_6, f16_7], dim=-1)
    f16_all = f16_all.view(num_tokens, hidden)

    f16_as_int16 = ((f16_all & 0xFFFF) ^ 0x8000) - 0x8000
    f16_values = f16_as_int16.to(torch.int16).view(torch.float16)
    return f16_values.to(torch.float32)


def cast_to_e5m6(
    x: torch.Tensor,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast a 2D tensor to E5M6 format with per-token scaling factors (PyTorch reference).

    Args:
        x: Input 2D tensor of shape ``(num_tokens, hidden)``.
        num_per_channels: Number of elements per scaling group along the channel dim.
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf layout.
        round_sf: Whether to round sf to a power-of-two boundary.
        use_packed_ue8m0: Whether to store sf in packed UE8M0 format.

    Returns:
        Tuple of ``(packed, sf_out)`` where ``packed`` is the E5M6-packed uint8 tensor
        and ``sf_out`` is the scaling-factor tensor.
    """
    assert x.ndim == 2
    assert x.dtype in (torch.float32, torch.bfloat16)
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float32)

    num_tokens, hidden = x.shape
    assert hidden % num_per_channels == 0
    assert hidden % 8 == 0

    num_groups = hidden // num_per_channels
    clamp_min = 1e-4
    # NOTE: For precision, construct a tensor on device instead of a python float
    max_value = torch.tensor(65024.0, dtype=torch.float32, device=x.device)

    x_view = x.view(num_tokens, num_groups, num_per_channels)
    amax = x_view.abs().amax(dim=-1)
    amax = torch.clamp(amax, min=clamp_min)

    dequant_sf = amax / max_value
    dequant_sf_int = dequant_sf.view(torch.int32)

    if round_sf:
        exp_sf = ((dequant_sf_int - 1) >> 23) + 1 - 127
        sf_inv_bits = (127 - exp_sf).clamp(min=0) << 23
        sf_inv = sf_inv_bits.view(torch.float32)
        sf_inv = torch.where(dequant_sf_int == 0, torch.tensor(0.0, device=x.device, dtype=torch.float32), sf_inv)
    else:
        exp_sf = None
        sf_inv = torch.where(
            dequant_sf_int == 0,
            torch.tensor(0.0, device=x.device, dtype=torch.float32),
            max_value / amax,
        )

    sf_inv_expanded = sf_inv.unsqueeze(-1).expand(num_tokens, num_groups, num_per_channels)
    sf_inv_expanded = sf_inv_expanded.reshape(num_tokens, hidden)
    x_scaled = x * sf_inv_expanded

    packed = _cast_to_e5m6(x_scaled)

    tma_alignment = 4

    if use_packed_ue8m0:
        if round_sf:
            sf_raw = (exp_sf + 127).to(torch.uint8)
        else:
            sf_raw = ((dequant_sf_int >> 23) & 0xFF).to(torch.uint8)

        pad_groups = (4 - num_groups % 4) % 4
        if pad_groups > 0:
            sf_raw = F.pad(sf_raw, (0, pad_groups))
        sf_out = sf_raw.view(torch.int32)
        if use_tma_aligned_col_major_sf:
            sf_out = _make_col_major(sf_out, tma_alignment)
    else:
        if round_sf:
            sf_out_bits = (exp_sf + 127) << 23
            sf_out = sf_out_bits.view(torch.float32)
        else:
            sf_out = dequant_sf
        if use_tma_aligned_col_major_sf:
            sf_out = _make_col_major(sf_out, tma_alignment)

    return packed, sf_out


def cast_back_from_e5m6(
    x: tuple[torch.Tensor, torch.Tensor],
    fmt: str,
    x_block_size: tuple[int, int],
) -> torch.Tensor:
    """Dequantize an E5M6 tensor back to BF16 or FP32 (PyTorch reference).

    Args:
        x: Quantized tensor pair ``(data, sf_factors)`` where ``data`` is a uint8
           E5M6-packed tensor and ``sf_factors`` are the per-token scaling factors.
        fmt: Target output format, either ``'bf16'`` or ``'fp32'``.
        x_block_size: Block size as ``(num_per_tokens, num_per_channels)``.

    Returns:
        Dequantized tensor in the requested format.
    """
    x_data, x_sf = x
    assert x_data.ndim == 2 and x_data.dtype == torch.uint8
    assert fmt in ('bf16', 'fp32')

    num_tokens = x_data.shape[0]
    packed_hidden = x_data.shape[1]
    hidden = packed_hidden * 2 // 3
    num_per_tokens, num_per_channels = x_block_size
    assert hidden % num_per_channels == 0
    num_groups = hidden // num_per_channels

    unpacked = _cast_back_from_e5m6(x_data)

    sf_float = transform_sf(x_sf)
    if x_sf.dtype != torch.float32:
        sf_float = sf_float[:, :num_groups]
    sf_expanded = sf_float.repeat_interleave(num_per_tokens, dim=0).repeat_interleave(num_per_channels, dim=1)
    sf_expanded = sf_expanded[:num_tokens, :hidden]

    result = unpacked * sf_expanded
    out_dtype = torch.bfloat16 if fmt == 'bf16' else torch.float32
    return result.to(out_dtype)
