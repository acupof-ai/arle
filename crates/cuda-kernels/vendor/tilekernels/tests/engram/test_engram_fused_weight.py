import os
import pytest
import torch

from tile_kernels.engram import fused_weight
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.testing.generator import generate_hidden_sizes
from tile_kernels.testing.bench import make_param_id

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    hc_mult = params['hc']
    hidden_size = params['hidden']
    wh_data = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device='cuda')
    we_data = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device='cuda')
    return (wh_data, we_data)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    return [
        {'hc': hc, 'hidden': hidden_size}
        for hc in (4,)
        for hidden_size in generate_hidden_sizes(128)
    ]


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_engram_fused_weight(params):
    wh_data, we_data = generate_test_data(params)

    ref = wh_data.float() * we_data.float()
    out = fused_weight(wh_data, we_data)

    assert_equal(out, ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_engram_fused_weight_benchmark(benchmark_timer, benchmark_record, params):
    wh_data, we_data = generate_test_data(params)
    out = fused_weight(wh_data, we_data)

    t_us = benchmark_timer(lambda: fused_weight(wh_data, we_data))

    num_bytes = count_bytes(wh_data, we_data, out)
    bandwidth_gbs = num_bytes / t_us / 1e3
    benchmark_record(
        kernel='fused_weight',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )
