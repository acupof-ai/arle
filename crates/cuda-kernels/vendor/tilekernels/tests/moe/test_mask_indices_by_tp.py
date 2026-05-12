import os

import torch
import pytest

import tile_kernels
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.testing.generator import generate_moe_params, generate_topk_idx
from tile_kernels.torch import mask_indices_by_tp as torch_mask_indices_by_tp
from tile_kernels.testing.bench import make_param_id

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    num_experts = params['num_experts']
    num_ep_ranks = params['num_ep_ranks']
    num_tp_ranks = params['num_tp_ranks']

    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]
    tp_rank = torch.randint(0, num_tp_ranks, (1,)).item()
    n = num_experts * num_ep_ranks

    return (topk_idx, num_tokens, tp_rank, n)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    return [
        {**moe, 'num_tp_ranks': num_tp_ranks}
        for moe in generate_moe_params(is_benchmark=is_benchmark)
        for num_tp_ranks in (2, 4, 8)
    ]


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_mask_indices_by_tp(params):
    topk_idx, num_tokens, tp_rank, n = generate_test_data(params)
    num_ep_ranks = params['num_ep_ranks']
    num_tp_ranks = params['num_tp_ranks']

    masked_indices = tile_kernels.moe.mask_indices_by_tp(topk_idx, n, num_ep_ranks, tp_rank, num_tp_ranks)

    # Test correctness: torch reference
    masked_ref = torch_mask_indices_by_tp(topk_idx, n, num_ep_ranks, tp_rank, num_tp_ranks)
    assert_equal(masked_indices, masked_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_mask_indices_by_tp_benchmark(benchmark_timer, benchmark_record, params):
    topk_idx, num_tokens, tp_rank, n = generate_test_data(params)
    num_ep_ranks = params['num_ep_ranks']
    num_tp_ranks = params['num_tp_ranks']

    masked_indices = tile_kernels.moe.mask_indices_by_tp(topk_idx, n, num_ep_ranks, tp_rank, num_tp_ranks)

    t_us = benchmark_timer(lambda: tile_kernels.moe.mask_indices_by_tp(topk_idx, n, num_ep_ranks, tp_rank, num_tp_ranks))
    num_bytes = count_bytes(topk_idx, masked_indices)
    bandwidth_gbs = num_bytes / t_us / 1e3

    params.pop('num_send_tokens')
    benchmark_record(
        kernel='mask_indices_by_tp',
        operation='fwd',
        params={'num_tokens': num_tokens, **params, 'tp_rank': tp_rank},
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )
