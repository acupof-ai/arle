import math
import os

import tilelang
import torch
from tilelang import language as T

from tile_kernels.quant.common import *
from tile_kernels.utils import align


@T.macro
def e5m6_to_float(
    inp: T.LocalBuffer[(3,), T.uint32],
    out: T.LocalBuffer[(8,), T.float32],
):
    """Unpack 3 uint32 (96 bits) into 8 float32 from e5m6 packed format.

    E5M6 stores each value as the top 12 bits of an IEEE fp16
    (1-bit sign + 5-bit exponent + 6-bit mantissa).
    8 values x 12 bits = 96 bits = 3 x uint32.

    This is the exact inverse of ``float_to_e5m6`` in ``per_token_cast_to_e5m6.py``.
    """
    f16_local = T.alloc_local((8,), T.float16)
    kMask = T.uint32(0xFFF0)

    # Extract 8 x 12-bit values, each placed in the top 12 bits of a uint16.
    # Reinterpreting as fp16 recovers the truncated half-precision value.
    f16_local[0] = T.reinterpret(T.cast((inp[0] >> 16) & kMask, T.uint16), T.float16)
    f16_local[1] = T.reinterpret(T.cast((inp[0] >> 4) & kMask, T.uint16), T.float16)
    f16_local[2] = T.reinterpret(T.cast(((inp[0] << 8) | (inp[1] >> 24)) & kMask, T.uint16), T.float16)
    f16_local[3] = T.reinterpret(T.cast((inp[1] >> 12) & kMask, T.uint16), T.float16)
    f16_local[4] = T.reinterpret(T.cast(inp[1] & kMask, T.uint16), T.float16)
    f16_local[5] = T.reinterpret(T.cast(((inp[1] << 12) | (inp[2] >> 20)) & kMask, T.uint16), T.float16)
    f16_local[6] = T.reinterpret(T.cast((inp[2] >> 8) & kMask, T.uint16), T.float16)
    f16_local[7] = T.reinterpret(T.cast((inp[2] << 4) & kMask, T.uint16), T.float16)

    for i in T.vectorized(8):
        out[i] = T.cast(f16_local[i], T.float32)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
    },
)
def get_cast_back_e5m6_kernel(
    hidden: int,
    in_config: CastInputConfig,
    out_dtype: T.dtype,
):
    num_threads = 128
    num_elems_per_block = 8192
    num_per_tokens, num_per_channels = in_config.sf_block
    packed_hidden = hidden * 3 // 8

    if num_per_tokens == 1:
        # Default case: 1 * gcd(hidden, 8192)
        TILE_K = math.gcd(hidden, num_elems_per_block)

        # Replication optimization case: 1 * hidden
        if hidden <= num_elems_per_block:
            TILE_K = align(hidden, num_threads * 8)

        TILE_M = num_elems_per_block // TILE_K

        # When TILE_M is small, set TILE_M = 1 to avoid predicated loads for better performance
        if TILE_M <= 3:
            TILE_M = 1
    else:
        # Per block cast case
        TILE_K = align(max(64, num_per_channels), 8)
        TILE_M = num_elems_per_block // TILE_K

    TILE_K_packed = TILE_K * 3 // 8

    # Runtime symbols
    num_tokens = T.dynamic('num_tokens')
    sf_shape = get_sf_shape((num_tokens, hidden), in_config)

    sf_stride = T.dynamic('sf_stride')

    def out_forward_fn(i: int, j: int):
        elems = i * TILE_K + j
        return elems // 8 % num_threads, elems % 8 + elems // (8 * num_threads) * 8

    @T.prim_func
    def cast_back_e5m6_kernel(
        x: T.Tensor[(num_tokens, packed_hidden), T.uint32],
        x_sf: T.StridedTensor[sf_shape, (sf_stride, 1), in_config.sf_dtype],
        out: T.Tensor[(num_tokens, hidden), out_dtype],
    ):
        with T.Kernel(T.ceildiv(num_tokens, TILE_M), T.ceildiv(hidden, TILE_K), threads=num_threads) as (pid_token, pid_hidden):
            sf_shared = T.alloc_shared((T.ceildiv(TILE_M, num_per_tokens), T.ceildiv(TILE_K, num_per_channels)), T.float32)
            out_fragment = T.alloc_fragment((TILE_M, TILE_K), out_dtype)

            for i, j in T.Parallel(T.ceildiv(TILE_M, num_per_tokens), T.ceildiv(TILE_K, num_per_channels)):
                token_index = pid_token * TILE_M // num_per_tokens + i
                channel_index = pid_hidden * TILE_K // num_per_channels + j
                sf = load_sf(x_sf, token_index, channel_index, in_config)
                sf_shared[i, j] = transform_sf(sf, in_config)

            # Unpack e5m6 groups and apply scale
            in_local = T.alloc_local((3,), T.uint32)
            out_local = T.alloc_local((8,), T.float32)

            T.annotate_layout({
                out_fragment: T.Fragment(
                    (TILE_M, TILE_K),
                    forward_fn=out_forward_fn,
                ),
            })

            for i, j in T.Parallel(TILE_M, TILE_K // 8):
                # Load 3 packed uint32
                for k in T.serial(3):
                    in_local[k] = x[pid_token * TILE_M + i, pid_hidden * TILE_K_packed + j * 3 + k]

                e5m6_to_float(in_local, out_local)

                # Scale and store
                for k in T.vectorized(8):
                    channel_in_tile = j * 8 + k
                    scaled = out_local[k] * sf_shared[i // num_per_tokens, channel_in_tile // num_per_channels]
                    out_fragment[i, channel_in_tile] = scaled

            T.copy(out_fragment, out[pid_token * TILE_M, pid_hidden * TILE_K])

    return cast_back_e5m6_kernel


def cast_back_e5m6(
    x: QuantTensor,
    fmt: str,
    x_block_size: tuple[int, int],
) -> torch.Tensor:
    """Dequantize an e5m6 packed tensor back to BF16 or FP32.

    Args:
        x: Quantized tensor pair ``(packed_data, scale_factors)``.
            ``packed_data`` is uint8 with shape ``(num_tokens, hidden * 3 // 2)`.
        fmt: Target output format, either ``'bf16'`` or ``'fp32'``.
        x_block_size: Scaling block size as ``(num_per_tokens, num_per_channels)``.

    Returns:
        Dequantized tensor of shape ``(num_tokens, hidden)`` in the requested format.
    """
    assert fmt in ('bf16', 'fp32')
    out_dtype = torch.bfloat16 if fmt == 'bf16' else torch.float32

    x, x_sf = x
    num_tokens = x.size(0)
    assert x.dim() == 2
    assert x_sf.dim() == 2
    assert x_sf.dtype in (torch.int32, torch.float32)

    # Compute logical hidden from packed representation
    assert x.dtype == torch.uint8
    hidden = x.size(1) * 2 // 3
    assert hidden % 8 == 0

    x, x_sf, in_config = get_cast_input_and_config((x, x_sf), x_block_size)
    x = x.view(torch.uint32)

    kernel = get_cast_back_e5m6_kernel(hidden=hidden, in_config=in_config, out_dtype=T.dtype(out_dtype))

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    out = torch.empty((num_tokens, hidden), dtype=out_dtype, device=x.device)
    if num_tokens > 0:
        kernel(x, x_sf, out)

    return out
