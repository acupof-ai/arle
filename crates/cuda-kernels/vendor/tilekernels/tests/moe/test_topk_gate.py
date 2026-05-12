import os
import torch
import pytest

import tile_kernels
from tile_kernels.testing.generator import generate_num_tokens
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.testing.bench import make_param_id
from tile_kernels.torch import stable_topk as torch_stable_topk
# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


_EXPERT_CONFIGS = [
    (72, 6),
    (32, 6),
    (64, 6),
    (96, 6),
    (16, 6),
    (36, 6),
    (108, 6),
    (128, 6),
    (144, 6),
    (256, 8),
]


def generate_test_data(params):
    num_tokens = params['num_tokens']
    num_experts = params['num_experts']
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    return scores


def generate_test_params(is_benchmark: bool) -> list[dict]:
    return [
        {
            'num_tokens': num_tokens,
            'num_experts': num_experts,
            'num_topk': num_topk,
        }
        for num_tokens in generate_num_tokens(is_benchmark=is_benchmark)
        for num_experts, num_topk in _EXPERT_CONFIGS
    ]


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_topk_gate(params):
    scores = generate_test_data(params)
    num_topk = params['num_topk']

    topk_idx_ref = torch_stable_topk(scores, num_topk)
    topk_idx = tile_kernels.moe.topk_gate(scores, num_topk)
    assert_equal(topk_idx, topk_idx_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_topk_gate_benchmark(benchmark_timer, benchmark_record, params):
    scores = generate_test_data(params)
    num_topk = params['num_topk']

    topk_idx = tile_kernels.moe.topk_gate(scores, num_topk)

    t_us = benchmark_timer(lambda: tile_kernels.moe.topk_gate(scores, num_topk))
    num_bytes = count_bytes(scores, topk_idx)
    bandwidth_gbs = num_bytes / t_us / 1e3

    benchmark_record(
        kernel='topk_gate',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )
