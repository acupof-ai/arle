import os
import pytest
import torch

import tile_kernels
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.testing.bench import dtype_to_str, make_param_id
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_tokens

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def twice_stride(w):
    # Make a 2D tensor's leading dim twice strided
    twice_w = w.new_empty((w.shape[0], w.shape[1] * 2))
    ret = torch.chunk(twice_w, 2, dim=1)[0]
    ret[:] = w
    assert not ret.is_contiguous()
    return ret


def generate_test_data_transpose(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    dtype = params['dtype']
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    if dtype == torch.float8_e4m3fn:
        x = x.to(torch.float8_e4m3fn)
    if num_tokens > 0:
        x = twice_stride(x)
    return (x,)


def generate_test_data_batched_transpose(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    num_experts = params['num_experts']
    dtype = params['dtype']
    x = torch.randn((num_experts, num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    if dtype == torch.float8_e4m3fn:
        x = x.to(torch.float8_e4m3fn)
    return (x,)


def generate_test_params_transpose(is_benchmark: bool) -> list[dict]:
    return [
        {'num_tokens': t, 'hidden': hidden_size, 'dtype': dtype}
        for t in generate_num_tokens(64, is_benchmark=is_benchmark)
        for hidden_size in generate_hidden_sizes()
        for dtype in (torch.float8_e4m3fn, torch.bfloat16)
    ]


def generate_test_params_batched_transpose(is_benchmark: bool) -> list[dict]:
    return [
        {'num_tokens': num_tokens, 'hidden': hidden_size, 'num_experts': num_experts, 'dtype': dtype}
        for num_tokens in generate_num_tokens(64, is_benchmark=is_benchmark)
        for hidden_size in generate_hidden_sizes()
        for num_experts in (8, 32)
        for dtype in (torch.float8_e4m3fn, torch.bfloat16, torch.float32)
    ]


@pytest.mark.parametrize('params', generate_test_params_transpose(is_benchmark=False), ids=make_param_id)
def test_transpose(params):
    num_tokens = params['num_tokens']
    (x,) = generate_test_data_transpose(params)
    y = tile_kernels.transpose.transpose(x)
    if num_tokens == 0:
        return
    y_ref = x.T.contiguous()

    assert_equal(y, y_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params_transpose(is_benchmark=True), ids=make_param_id)
def test_transpose_benchmark(benchmark_timer, benchmark_record, params):
    (x,) = generate_test_data_transpose(params)

    num_bytes = count_bytes(x, tile_kernels.transpose.transpose(x))
    t_us = benchmark_timer(lambda: tile_kernels.transpose.transpose(x))

    benchmark_record(
        kernel='transpose',
        operation='fwd',
        params={**params, 'dtype': dtype_to_str(params['dtype'])},
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )


@pytest.mark.parametrize('params', generate_test_params_batched_transpose(is_benchmark=False), ids=make_param_id)
def test_batched_transpose(params):
    (x,) = generate_test_data_batched_transpose(params)
    y = tile_kernels.transpose.batched_transpose(x)
    y_ref = torch.transpose(x, 1, 2).contiguous()

    assert_equal(y, y_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params_batched_transpose(is_benchmark=True), ids=make_param_id)
def test_batched_transpose_benchmark(benchmark_timer, benchmark_record, params):
    (x,) = generate_test_data_batched_transpose(params)

    num_bytes = count_bytes(x, tile_kernels.transpose.batched_transpose(x))
    t_us = benchmark_timer(lambda: tile_kernels.transpose.batched_transpose(x))

    benchmark_record(
        kernel='batched_transpose',
        operation='fwd',
        params={**params, 'dtype': dtype_to_str(params['dtype'])},
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )
