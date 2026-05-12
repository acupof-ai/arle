import os
import pytest
import torch

import tile_kernels
from tile_kernels.testing.bench import dtype_to_str, make_param_id
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_tokens
from tile_kernels.testing.numeric import assert_equal, count_bytes

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    dtype = params['dtype']

    x = torch.randn((num_tokens, hidden), dtype=dtype, device='cuda')

    return x


def generate_test_params(is_benchmark: bool) -> list[dict]:
    return [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'round_sf': round_sf,
            'dtype': dtype,
            'num_per_channels': num_per_channels,
        }
        for num_tokens in generate_num_tokens(128, is_benchmark=is_benchmark)
        for hidden_size in generate_hidden_sizes()
        for round_sf in (True, False)
        for dtype in (torch.bfloat16,)
        for num_per_channels in (32, 128)
    ]


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_per_channel_cast_and_transpose(params):
    round_sf = params['round_sf']
    num_per_channels = params['num_per_channels']

    x = generate_test_data(params)

    x_fp8, x_sf = tile_kernels.quant.per_channel_cast_and_transpose(x, 'e4m3', num_per_channels, round_sf)
    x_fp8_ref, x_sf_ref = tile_kernels.torch.cast(x, 'e4m3', block_size=(num_per_channels, 1), round_sf=round_sf)
    x_fp8_ref = x_fp8_ref.T.contiguous()
    assert_equal(x_fp8, x_fp8_ref)
    assert_equal(x_sf, x_sf_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_per_channel_cast_and_transpose_benchmark(benchmark_timer, benchmark_record, params):
    round_sf = params['round_sf']
    num_per_channels = params['num_per_channels']

    x = generate_test_data(params)

    x_fp8, x_sf = tile_kernels.quant.per_channel_cast_and_transpose(x, 'e4m3', num_per_channels, round_sf)
    num_bytes = count_bytes(x, x_fp8, x_sf)

    t_us = benchmark_timer(
        lambda: tile_kernels.quant.per_channel_cast_and_transpose(x, 'e4m3', num_per_channels, round_sf)
    )

    params['dtype'] = dtype_to_str(x.dtype)
    benchmark_record(
        kernel='per_channel_cast_and_transpose',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )
