import os
import pytest
import torch

import tile_kernels
from tile_kernels.testing.bench import dtype_to_str, make_param_id
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_tokens
from tile_kernels.testing.numeric import assert_equal, count_bytes, check_bias

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
            'num_per_tokens': num_per_tokens,
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'round_sf': round_sf,
            'dtype': dtype,
        }
        for num_per_tokens in (128,)
        for num_tokens in generate_num_tokens(128, is_benchmark=is_benchmark)
        for hidden_size in generate_hidden_sizes()
        for round_sf in (False, True)
        for dtype in (torch.bfloat16,)
    ]


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_per_channel_cast(params):
    num_per_tokens = params['num_per_tokens']
    round_sf = params['round_sf']

    x = generate_test_data(params)
    x_fp8, per_channel_sf_inv = tile_kernels.quant.per_channel_cast(x, 'e4m3', num_per_tokens, round_sf)
    x_fp8_ref, per_channel_sf_inv_ref = tile_kernels.torch.cast(x, 'e4m3', block_size=(num_per_tokens, 1), round_sf=round_sf)

    assert_equal(x_fp8, x_fp8_ref)
    assert_equal(per_channel_sf_inv, per_channel_sf_inv_ref)

    # Check bias
    x_casted_back = tile_kernels.torch.cast_back((x_fp8_ref, per_channel_sf_inv_ref), 'fp32', (num_per_tokens, 1))
    check_bias(x_casted_back, x)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_per_channel_cast_benchmark(benchmark_timer, benchmark_record, params):
    num_per_tokens = params['num_per_tokens']
    round_sf = params['round_sf']
    dtype = params['dtype']

    x = generate_test_data(params)
    x_fp8, per_channel_sf_inv = tile_kernels.quant.per_channel_cast(x, 'e4m3', num_per_tokens, round_sf)

    t_us = benchmark_timer(lambda: tile_kernels.quant.per_channel_cast(x, 'e4m3', num_per_tokens, round_sf))
    num_bytes = count_bytes(x, x_fp8, per_channel_sf_inv)

    benchmark_record(
        kernel='per_channel_cast',
        operation='fwd',
        params={**params, 'dtype': dtype_to_str(dtype)},
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )
