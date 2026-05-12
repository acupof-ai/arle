import os
import torch

import pytest

import tile_kernels
from tile_kernels.config import set_num_sms
from tile_kernels.testing.generator import generate_topk_idx, generate_moe_params, generate_num_sms
from tile_kernels.testing.numeric import calc_diff, count_bytes
from tile_kernels.torch import aux_fi as torch_aux_fi
from tile_kernels.testing.bench import make_param_id

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]

    return (topk_idx, num_tokens)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    return [
        {**moe, 'num_aux_topk': num_aux_topk}
        for moe in generate_moe_params(is_benchmark=is_benchmark)
        for num_aux_topk in (1, moe['num_topk'])
    ]


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_aux_fi(params):
    topk_idx, num_tokens = generate_test_data(params)
    num_experts = params['num_experts']
    num_aux_topk = params['num_aux_topk']

    # Test correctness
    fi_ref = torch_aux_fi(topk_idx, num_experts, num_aux_topk)

    for num_sms in generate_num_sms():
        set_num_sms(num_sms)
        fi = tile_kernels.moe.aux_fi(topk_idx, num_experts, num_aux_topk)
        assert calc_diff(fi, fi_ref) < 2e-7, f'aux_fi mismatch\n{fi}\nvs\n{fi_ref}'


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_aux_fi_benchmark(benchmark_timer, benchmark_record, params):
    topk_idx, num_tokens = generate_test_data(params)
    num_experts = params['num_experts']
    num_aux_topk = params['num_aux_topk']

    t_us = benchmark_timer(lambda: tile_kernels.moe.aux_fi(topk_idx, num_experts, num_aux_topk))
    num_bytes = count_bytes(topk_idx)
    bandwidth_gbs = num_bytes / t_us / 1e3

    params.pop('num_send_tokens')
    benchmark_record(
        kernel='aux_fi',
        operation='fwd',
        params={'num_tokens': num_tokens, **params},
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )
