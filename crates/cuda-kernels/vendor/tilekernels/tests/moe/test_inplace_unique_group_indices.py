import os
import torch

import pytest

import tile_kernels
from tile_kernels.torch import inplace_unique_group_indices as torch_inplace_unique_group_indices
from tile_kernels.config import set_num_sms
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.testing.generator import generate_moe_params, generate_topk_idx, generate_num_sms
from tile_kernels.testing.bench import make_param_id

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    num_experts = params['num_experts']
    num_ep_ranks = params['num_ep_ranks']
    num_groups = params['num_groups']

    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]
    _group_indices = topk_idx // (num_experts * num_ep_ranks // num_groups)

    return (_group_indices, num_tokens)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    return [
        {**moe, 'num_groups': num_groups}
        for moe in generate_moe_params(is_benchmark=is_benchmark)
        for num_groups in (8, 16, 72) if moe['num_experts'] * moe['num_ep_ranks'] % num_groups == 0
    ]


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_inplace_unique_group_indices(params):
    _group_indices, num_tokens = generate_test_data(params)
    num_groups = params['num_groups']

    func = lambda group_indices: tile_kernels.moe.inplace_unique_group_indices(group_indices, num_groups)
    func_ref = lambda group_indices: torch_inplace_unique_group_indices(group_indices, num_groups)

    group_indices_ref = _group_indices.clone()
    func_ref(group_indices_ref)

    for num_sms in generate_num_sms():
        set_num_sms(num_sms)
        group_indices = _group_indices.clone()
        func(group_indices)
        assert_equal(group_indices, group_indices_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_inplace_unique_group_indices_benchmark(benchmark_timer, benchmark_record, params):
    _group_indices, num_tokens = generate_test_data(params)
    num_groups = params['num_groups']

    func = lambda group_indices: tile_kernels.moe.inplace_unique_group_indices(group_indices, num_groups)

    group_indices = _group_indices.clone()
    func(group_indices)

    num_bytes = count_bytes(group_indices)
    num_bytes += torch.count_nonzero(group_indices != _group_indices).item() * 4

    t_us = benchmark_timer(lambda: func(_group_indices.clone()))
    bandwidth_gbs = num_bytes / t_us / 1e3

    params.pop('num_send_tokens')
    benchmark_record(
        kernel='inplace_unique_group_indices',
        operation='fwd',
        params={'num_tokens': num_tokens, **params},
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )
