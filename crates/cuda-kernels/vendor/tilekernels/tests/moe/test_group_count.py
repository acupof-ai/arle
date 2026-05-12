import os
import torch

import pytest

import tile_kernels
from tile_kernels.config import set_num_sms
from tile_kernels.testing.generator import generate_topk_idx, generate_moe_params, generate_num_sms
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.torch import group_count as torch_group_count
from tile_kernels.testing.bench import make_param_id

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]

    return (topk_idx, num_tokens)


@pytest.mark.parametrize('params', list(generate_moe_params(is_benchmark=False)), ids=make_param_id)
def test_group_count(params):
    topk_idx, num_tokens = generate_test_data(params)
    num_experts = params['num_experts']

    # Test correctness
    count_ref = torch_group_count(topk_idx, num_experts)

    for num_sms in generate_num_sms():
        set_num_sms(num_sms)
        count = tile_kernels.moe.group_count(topk_idx, num_experts)
        assert_equal(count, count_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', list(generate_moe_params(is_benchmark=True)), ids=make_param_id)
def test_group_count_benchmark(benchmark_timer, benchmark_record, params):
    topk_idx, num_tokens = generate_test_data(params)
    num_experts = params['num_experts']

    t_us = benchmark_timer(lambda: tile_kernels.moe.group_count(topk_idx, num_experts))
    num_bytes = count_bytes(topk_idx)
    bandwidth_gbs = num_bytes / t_us / 1e3

    params.pop('num_send_tokens')
    benchmark_record(
        kernel='group_count',
        operation='fwd',
        params={'num_tokens': num_tokens, **params},
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )
