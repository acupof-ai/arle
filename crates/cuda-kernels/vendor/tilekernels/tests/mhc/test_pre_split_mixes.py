from typing import Callable

import pytest
import torch
from tile_kernels.modeling.mhc.ops import mhc_pre_split_mixes
from tile_kernels.torch.mhc import mhc_pre_split_mixes_ref


def generate_pre_split_mixes_test_data(
    n0: int, n1: int, mhc_mult: int, device: str = 'cuda'
) -> dict[str, torch.Tensor]:
    mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult

    input_mixes = torch.randn((n0, n1, mhc_mult3), dtype=torch.float, device=device)
    mhc_scale = torch.randn((3,), dtype=torch.float, device=device)
    mhc_base = torch.randn((mhc_mult3,), dtype=torch.float, device=device)

    pre_layer_mix_grad = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float, device=device)
    post_layer_mix_grad = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float, device=device)
    comb_res_mix_grad = torch.randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float, device=device)

    return {
        'input_mixes': input_mixes,
        'mhc_scale': mhc_scale,
        'mhc_base': mhc_base,
        'pre_layer_mix_grad': pre_layer_mix_grad,
        'post_layer_mix_grad': post_layer_mix_grad,
        'comb_res_mix_grad': comb_res_mix_grad,
        'mhc_post_mult_value': 2.0,
        'mhc_pre_eps': 1e-2,
    }


def _tester(
    impl: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, int, float, float],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    test_data: dict[str, torch.Tensor],
    mhc_mult: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_mixes_ = test_data['input_mixes'].clone().requires_grad_()
    mhc_scale_ = test_data['mhc_scale'].clone().requires_grad_()
    mhc_base_ = test_data['mhc_base'].clone().requires_grad_()

    pre_layer_mix_, post_layer_mix_, comb_res_mix_ = impl(
        input_mixes_,
        mhc_scale_,
        mhc_base_,
        mhc_mult,
        test_data['mhc_post_mult_value'],
        test_data['mhc_pre_eps'],
    )

    torch.autograd.backward(
        [pre_layer_mix_, post_layer_mix_, comb_res_mix_],
        [
            test_data['pre_layer_mix_grad'],
            test_data['post_layer_mix_grad'],
            test_data['comb_res_mix_grad'],
        ],
    )

    return (
        pre_layer_mix_,
        post_layer_mix_,
        comb_res_mix_,
        input_mixes_.grad,
        mhc_scale_.grad,
        mhc_base_.grad,
    )


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1024, 4096])
@pytest.mark.parametrize('mhc_mult', [4])
def test_pre_split_mixes_comprehensive(n0: int, n1: int, mhc_mult: int) -> None:
    test_data = generate_pre_split_mixes_test_data(n0=n0, n1=n1, mhc_mult=mhc_mult)

    (
        pre_layer_mix_tl,
        post_layer_mix_tl,
        comb_res_mix_tl,
        grad_input_mixes_tl,
        grad_mhc_scale_tl,
        grad_mhc_base_tl,
    ) = _tester(mhc_pre_split_mixes, test_data, mhc_mult)

    (
        pre_layer_mix_ref,
        post_layer_mix_ref,
        comb_res_mix_ref,
        grad_input_mixes_ref,
        grad_mhc_scale_ref,
        grad_mhc_base_ref,
    ) = _tester(mhc_pre_split_mixes_ref, test_data, mhc_mult)

    torch.testing.assert_close(pre_layer_mix_tl, pre_layer_mix_ref, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(post_layer_mix_tl, post_layer_mix_ref, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(comb_res_mix_tl, comb_res_mix_ref, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(grad_input_mixes_tl, grad_input_mixes_ref, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(grad_mhc_scale_tl, grad_mhc_scale_ref, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(grad_mhc_base_tl, grad_mhc_base_ref, rtol=1e-5, atol=2e-5)
