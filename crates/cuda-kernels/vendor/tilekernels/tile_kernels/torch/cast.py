import torch
import torch.nn.functional as F
from typing import Optional, Union

from tile_kernels.utils import align, ceil_div
from tile_kernels.quant.common import unpack_from_e2m1fn_x2
from tile_kernels.quant.types import QuantTensor


def right_shift_unsigned(x, shift):
    # CUDA torch does not support bit ops on uint32, so we need to mask to get unsigned right shift
    return (x >> shift) & ((1 << (32 - shift)) - 1)


def get_min_clamp_val(dtype: torch.dtype):
    min_clamp_by_dtype = {torch.int8: 6.0 * 2 ** (-126), torch.float8_e4m3fn: 0.0001}
    assert dtype in min_clamp_by_dtype
    return min_clamp_by_dtype[dtype]


def get_max_quant_val(dtype: torch.dtype):
    max_quant_by_dtype = {torch.int8: 6.0, torch.float8_e4m3fn: 448.0}
    assert dtype in max_quant_by_dtype
    return max_quant_by_dtype[dtype]


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


def cast_back(x: QuantTensor, fmt: str, block_size: tuple[int, int] = (32, 32)) -> torch.Tensor:
    """Dequantize an FP8/FP4 tensor back to BF16 or FP32 (PyTorch reference).

    Args:
        x: Quantized tensor pair ``(data, sf_factors)``.
        fmt: Target output format, either ``'bf16'`` or ``'fp32'``.
        block_size: Scaling block size as ``(block_h, block_w)``.

    Returns:
        Dequantized tensor in the requested format.
    """
    input_tensor, input_sf = x
    assert input_tensor.dtype in (torch.float8_e4m3fn, torch.int8)
    # Expand x_sf and do elementwise multiply with x
    input_sf = transform_sf(input_sf)
    input_sf = input_sf.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=1)
    input_tensor = unpack_from_e2m1fn_x2(input_tensor) if input_tensor.dtype == torch.int8 else input_tensor.to(torch.float32)
    input_sf = input_sf[: input_tensor.shape[0], : input_tensor.shape[1]]
    x = input_tensor * input_sf
    return x.to(dtype=torch.float32 if fmt == 'fp32' else torch.bfloat16)


def cast(
    x: Union[torch.Tensor, QuantTensor],
    fmt: str,
    block_size: tuple[int, int] = (32, 32),
    sf: Optional[torch.Tensor] = None,
    x_block_size: Optional[tuple[int, int]] = None,
    round_sf: bool = False,
    use_tma_aligned_col_major_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> Union[torch.Tensor, QuantTensor]:
    """Cast a 2D tensor to MX format with 2D block-shared sf.

    Args:
        x: Input 2D tensor (H, W), optionally with scaling factors.
        fmt: Target quantization format, e.g. "e4m3".
        block_size: Block shape (block_h, block_w), e.g. (32, 32).
        sf: Optional precomputed dequant sf.
        x_block_size: Input block shape when `x_packed` carries sf.
        round_sf: Whether to round sf.
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf.
        use_packed_ue8m0: Whether to store sf in packed UE8M0.

    Returns:
        Quantized tensor (packed on the last dim for mxfp4) and optional sf.
    """
    has_input_sf = isinstance(x, tuple)
    if has_input_sf:
        input_tensor, input_sf = x
        assert input_tensor.dtype in (torch.float8_e4m3fn, torch.int8)
        assert input_sf is not None and x_block_size is not None
        # Expand x_sf and do elementwise multiply with x
        input_sf = transform_sf(input_sf)
        input_sf = input_sf.repeat_interleave(x_block_size[0], dim=0).repeat_interleave(x_block_size[1], dim=1)
        input_tensor = unpack_from_e2m1fn_x2(input_tensor) if input_tensor.dtype == torch.int8 else input_tensor.to(torch.float32)
        input_sf = input_sf[: input_tensor.shape[0], : input_tensor.shape[1]]
        x = input_tensor * input_sf
    else:
        assert x.dtype in (torch.bfloat16, torch.float32)

    assert x.ndim == 2, 'Only 2D tensors are supported for 2D blocking.'
    h, w = x.shape
    bh, bw = block_size

    out_dtype = {'e2m1': torch.int8, 'e4m3': torch.float8_e4m3fn}[fmt]
    max_quant_val = get_max_quant_val(out_dtype)
    is_fp4 = out_dtype == torch.int8
    device = x.device

    if h == 0:
        out_w = w // 2 if is_fp4 else w
        out_weight = torch.empty((0, out_w), dtype=out_dtype, device=device)
        if sf is not None:
            return out_weight
        sf_h = 0
        sf_w = (w + bw - 1) // bw
        if use_packed_ue8m0:
            dq_sf = torch.empty((ceil_div(sf_w, 4), sf_h), dtype=torch.int32, device=device).T
        elif use_tma_aligned_col_major_sf:
            dq_sf = torch.empty((sf_w, sf_h), dtype=torch.float32, device=device).T
        else:
            dq_sf = torch.empty((sf_h, sf_w), dtype=torch.float32, device=device)
        return out_weight, dq_sf

    if is_fp4:
        assert w % 2 == 0, 'For mxfp4, the width must be even for packing.'

    # 1) Padding
    pad_h = (bh - h % bh) % bh
    pad_w = (bw - w % bw) % bw

    # F.pad args are (left, right, top, bottom).
    padded_src = F.pad(x.to(torch.float32), (0, pad_w, 0, pad_h))
    valid_mask = F.pad(torch.ones_like(x, dtype=torch.bool), (0, pad_w, 0, pad_h))

    ph, pw = padded_src.shape

    if sf is None:
        # 2) Block-wise max: (Hb, bh, Wb, bw) -> (Hb, Wb, bh*bw)
        reshaped_for_max = padded_src.view(ph // bh, bh, pw // bw, bw).permute(0, 2, 1, 3).reshape(ph // bh, pw // bw, -1)
        reshaped_mask = valid_mask.view(ph // bh, bh, pw // bw, bw).permute(0, 2, 1, 3).reshape(ph // bh, pw // bw, -1)

        abs_f = torch.abs(reshaped_for_max)
        # Exclude padded positions from max.
        abs_f = torch.where(reshaped_mask, abs_f, torch.tensor(-1.0, device=device, dtype=abs_f.dtype))
        max_val, _ = abs_f.max(dim=-1, keepdim=True)  # shape: (ph/bh, pw/bw, 1)
        max_val = torch.clamp(max_val, min=get_min_clamp_val(out_dtype))

        # 3) Compute sf.
        assert max_val.dtype == torch.float32
        # NOTE: Do manual filling to prevent pytorch from doing reciprocal on the CPU
        # Refer to https://github.com/pytorch/pytorch/blob/4d4613b6227ed156543c25e6ea94b99d25d3aff4/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu#L36
        max_quant_val_expanded = torch.full_like(max_val, max_quant_val, dtype=torch.float32)
        dequant_sf = max_val / max_quant_val_expanded

        ds_int = dequant_sf.view(torch.int32)
        # Upcast to ue8m0
        if round_sf:
            ds_int_rounded = (ds_int + 0x007FFFFF) & 0x7F800000
            dequant_sf_rounded = ds_int_rounded.view(torch.float32)
            quant_sf = torch.where(dequant_sf_rounded == 0, torch.tensor(0.0, device=device), 1.0 / dequant_sf_rounded)
        else:
            ds_int_rounded = ds_int
            quant_sf = torch.where(ds_int_rounded == 0, torch.tensor(0.0, device=device), max_quant_val_expanded / max_val)
    else:
        assert not use_packed_ue8m0 and not use_tma_aligned_col_major_sf
        expected_sf_shape = (ph // bh, pw // bw)
        assert sf.ndim == 2, f'sf must be 2D, got {sf.ndim}D'
        assert tuple(sf.shape) == expected_sf_shape, f'sf shape mismatch: expected {expected_sf_shape}, got {tuple(sf.shape)}'
        quant_sf = sf.reciprocal().unsqueeze(-1)

    # 4) Quantize data.
    if has_input_sf:
        quant_sf_extended = quant_sf.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=1).squeeze(-1)
        quant_sf_extended = quant_sf_extended[:h, :w]
        quant_tensor = x * quant_sf_extended
    else:
        # Map to block layout: (Hb, bh, Wb, bw) * (Hb, 1, Wb, 1)
        padded_src_view = padded_src.view(ph // bh, bh, pw // bw, bw)
        quant_sf_view = quant_sf.view(ph // bh, 1, pw // bw, 1)

        assert padded_src_view.dtype == torch.float32 and quant_sf_view.dtype == torch.float32
        quant_tensor = (padded_src_view * quant_sf_view).reshape(ph, pw)
        # Crop back to original shape.
        quant_tensor = quant_tensor[:h, :w]

    # 5) Cast type and pack FP4.
    if not is_fp4:
        quant_tensor = torch.clamp(quant_tensor, -max_quant_val, max_quant_val)
        out_weight = quant_tensor.to(out_dtype)
    else:
        e2m1_value = convert_to_e2m1_bits(quant_tensor, max_quant_val, device)
        # Pack (H, W) -> (H, W/2)
        e2m1_value = e2m1_value.view(h, w // 2, 2)
        out_weight = e2m1_value[..., 0] | (e2m1_value[..., 1] << 4)
        out_weight = out_weight.view(torch.int8)

    if sf is not None:
        return out_weight

    # 6) Format sf output.
    ds_int_rounded = ds_int_rounded.squeeze(-1)
    if use_tma_aligned_col_major_sf:
        # TMA alignment requirements of 16 bytes.
        # Either packed UE8M0 or float32, all 4 bytes per element.
        # Therefore, align to 16 / 4 = 4 elements.
        tma_alignment = 4
        packing_alignment = 4 if use_packed_ue8m0 else 1
        pad_h = align(ds_int_rounded.shape[0], tma_alignment) - ds_int_rounded.shape[0]
        pad_w = align(ds_int_rounded.shape[1], packing_alignment) - ds_int_rounded.shape[1]
        ds_int_rounded_padded = F.pad(ds_int_rounded, (0, pad_w, 0, pad_h))
        if use_packed_ue8m0:
            dq_sf = (ds_int_rounded_padded >> 23).to(torch.int8).view(torch.int32)  # (H/bh, W/bw)
        else:
            dq_sf = ds_int_rounded_padded.view(torch.float32)
        dq_sf = dq_sf.T.contiguous().T[: ds_int_rounded.shape[0], :]

    else:
        dq_sf = ds_int_rounded.view(torch.float32)

    return out_weight, dq_sf


def convert_to_e2m1_bits(quant_tensor, max_quant_val, device):
    """FP4 cast"""
    q_int = quant_tensor.contiguous().view(torch.int32)
    signs = q_int & 0x80000000
    exponents = (q_int >> 23) & 0xFF
    mantissas_orig = q_int & 0x7FFFFF

    E8_BIAS, E2_BIAS = 127, 1
    # Adjust mantissas for subnormals.
    is_subnormal = exponents < E8_BIAS
    shift = E8_BIAS - exponents - 1
    mantissas_pre = 0x400000 | right_shift_unsigned(mantissas_orig, 1)
    bit0_dropped = (mantissas_orig & 0x1) != 0
    mask = (1 << shift.clamp(max=31)) - 1
    dropped_post = (mantissas_pre & mask) != 0
    sticky = is_subnormal & (bit0_dropped | dropped_post)
    mantissas = torch.where(is_subnormal, mantissas_pre >> shift, mantissas_orig)
    exponents = torch.maximum(exponents, torch.tensor(E8_BIAS - E2_BIAS, device=device)) - (E8_BIAS - E2_BIAS)
    # Round to nearest, ties to even (RTNE)
    m2bits = right_shift_unsigned(mantissas, 21) & 0x3
    lsb_keep = right_shift_unsigned(m2bits, 1) & 0x1
    guard = m2bits & 0x1
    sticky |= (mantissas & ((1 << 21) - 1)) != 0
    round_inc = guard & (sticky.to(torch.int32) | lsb_keep)
    e2m1_tmp = right_shift_unsigned(((exponents << 2) | m2bits) + round_inc, 1)
    e2m1_tmp = torch.minimum(e2m1_tmp, torch.tensor(0x7, device=device))
    e2m1_value = (right_shift_unsigned(signs, 28) | e2m1_tmp).to(torch.uint8)  # shape: (..., even_axis_shape)

    return e2m1_value
