import random
import os
from typing import Iterable

import torch
from tile_kernels.utils import align
from tile_kernels.config import get_device_num_sms


def generate_num_tokens(alignment: int = 1, is_benchmark: bool = False) -> list[int]:
    do_full_test = os.getenv('TK_FULL_TEST') in ['1', 'true', 'True']
    base_list = [4001, 8001]
    if do_full_test and not is_benchmark:
        full_list = [0] + base_list
    else:
        full_list = base_list
    return [align(num_tokens, alignment) for num_tokens in full_list]


def generate_hidden_sizes(align: int = 64) -> list[int]:
    base_list = [576, 2048, 2560, 3072, 4096, 6144, 7168]
    full_list = [hidden_size for hidden_size in base_list if hidden_size % align == 0]
    return full_list


def generate_num_sms() -> list[int]:
    device_num_sms = get_device_num_sms()
    do_full_test = os.getenv('TK_FULL_TEST') in ['1', 'true', 'True']
    extra_list = [1, ]
    base_list = [device_num_sms - 20, device_num_sms, ]
    # Ensure `device_num_sms` is the last one in the list for convenience of testing
    return extra_list + base_list if do_full_test else base_list


def generate_moe_params(is_benchmark: bool = False) -> Iterable[dict]:
    do_full_test = os.getenv('TK_FULL_TEST') in ['1', 'true', 'True']
    extra_num_topk_list = (1, 7) if do_full_test else ()
    extra_num_experts_list = (288, 384) if do_full_test else ()
    extra_num_ep_ranks_list = (1, 72, 256) if do_full_test else ()

    if do_full_test and not is_benchmark:
        yield {'num_send_tokens': 0, 'num_topk': 1, 'num_experts': 1, 'num_ep_ranks': 1}

    for num_tokens in (4001,):
        for num_topk in (2, 6, 8, 9) + extra_num_topk_list:
            for num_experts in (72, 256) + extra_num_experts_list:
                for num_ep_ranks in (8, 64) + extra_num_ep_ranks_list:
                    if num_experts % num_ep_ranks == 0:
                        yield {'num_send_tokens': num_tokens, 'num_topk': num_topk,
                               'num_experts': num_experts // num_ep_ranks, 'num_ep_ranks': num_ep_ranks}


@torch.compile
def generate_topk_idx(params: dict) -> torch.Tensor:
    num_send_tokens = params['num_send_tokens']
    num_experts = params['num_experts']
    num_topk = params['num_topk']
    num_ep_ranks = params['num_ep_ranks']

    if num_send_tokens == 0:
        return torch.empty((0, num_topk), dtype=torch.int64, device='cuda')
    scores = torch.rand((num_send_tokens * num_ep_ranks, num_experts * num_ep_ranks), dtype=torch.bfloat16, device='cuda')
    _, topk_idx = torch.topk(scores, k=num_topk, dim=-1, sorted=False)
    mask = topk_idx >= num_experts
    topk_idx[mask] = -1
    mask = mask.all(dim=1)
    topk_idx = topk_idx[~mask]
    return topk_idx


# E5M6 format: 1 sign + 5 exponent + 6 mantissa, bias=15
#   max normal:    2^15 * (1 + 63/64) = 65024.0
#   min normal:    2^(-14)
#   max subnormal: 2^(-14) * (63/64)
#   min subnormal: 2^(-14) * (1/64) = 2^(-20)
_E5M6_SPECIAL_VALUES = (
    pow(2, -20),            # min subnormal
    pow(2, -14) * 63 / 64,  # max subnormal
    pow(2, -14),            # min normal
)


def generate_e5m6_inputs(num_tokens: int, hidden: int, dtype: torch.dtype) -> Iterable[tuple[torch.Tensor, bool]]:
    '''Yield (x, is_special) pairs: one random tensor, then e5m6 special-value tensors.'''
    yield torch.randn((num_tokens, hidden), dtype=dtype, device='cuda'), False
    for value in _E5M6_SPECIAL_VALUES:
        x = torch.full((num_tokens, hidden), value, dtype=dtype, device='cuda')
        x[:, -1] = 65024.0
        yield x, True


def generate_rand_float(shape: tuple[int, ...]) -> torch.Tensor:
    # We want to sample from a uniform distribution over the exponent of sf
    exp = random.randint(-110, 126)
    sf = float(2**exp)
    float_tensor = torch.randn(shape, dtype=torch.float32, device='cuda') * sf

    mask = torch.logical_or(torch.isnan(float_tensor), torch.isinf(float_tensor))
    if mask.any():
        num_values = mask.to(torch.int32).sum().item()
        normal_values = torch.randn((num_values,), dtype=torch.float32, device='cuda')
        float_tensor[mask] = normal_values

    max_value = torch.finfo(torch.float32).max / 8
    return torch.clamp(float_tensor, -max_value, max_value)
