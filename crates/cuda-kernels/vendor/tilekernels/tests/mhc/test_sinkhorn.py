from typing import Callable

import pytest
import torch
from tile_kernels.modeling.mhc.ops import sinkhorn_normalize
from tile_kernels.torch.mhc import sinkhorn_normalize_ref


def generate_sinkhorn_test_data(
    n0: int, n1: int, mhc: int, device: str = 'cuda'
) -> dict[str, torch.Tensor]:
    comb_res_mix = torch.randn((n0, n1, mhc, mhc), dtype=torch.float32, device=device)
    out_grad = torch.randn((n0, n1, mhc, mhc), dtype=torch.float32, device=device)

    return {
        'comb_res_mix': comb_res_mix,
        'out_grad': out_grad,
        'repeat': 10,
        'eps': 1e-6,
    }


def _tester(
    impl: Callable[[torch.Tensor, int, float], torch.Tensor],
    test_data: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    comb_res_mix_ = test_data['comb_res_mix'].clone().requires_grad_()
    out_ = impl(comb_res_mix_, test_data['repeat'], test_data['eps'])
    torch.autograd.backward([out_], [test_data['out_grad']])
    return out_, comb_res_mix_.grad


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1, 1024, 4096])
@pytest.mark.parametrize('mhc', [4])
def test_sinkhorn_comprehensive(n0: int, n1: int, mhc: int) -> None:
    test_data = generate_sinkhorn_test_data(n0=n0, n1=n1, mhc=mhc)

    out_tl, grad_tl = _tester(sinkhorn_normalize, test_data)
    out_ref, grad_ref = _tester(sinkhorn_normalize_ref, test_data)

    torch.testing.assert_close(out_tl, out_ref)
    torch.testing.assert_close(grad_tl, grad_ref)
