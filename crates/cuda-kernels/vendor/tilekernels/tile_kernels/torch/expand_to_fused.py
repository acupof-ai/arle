import torch
from tile_kernels.utils import align
from tile_kernels.quant.types import QuantTensor


def expand_to_fused(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
) -> torch.Tensor:
    """Expand token activations into the fused expert layout.

    For each token t and topk slot k, copies x[t, :] to out[pos, :] where
    pos = token_topk_to_pos[t, k].  Positions where pos_to_expert < 0 are
    zero-filled.

    Args:
        x: Input tensor of shape (num_tokens, hidden).
        token_topk_to_pos: Mapping from (token, topk) to expanded position,
            shape (num_tokens, num_topk). -1 means unused.
        pos_to_expert: Mapping from expanded position to expert index,
            shape (num_expanded_tokens,). Negative means padding.

    Returns:
        Expanded tensor of shape (num_expanded_tokens, hidden).
    """
    num_tokens, hidden = x.shape
    num_expanded_tokens = pos_to_expert.shape[0]

    out = torch.zeros((num_expanded_tokens, hidden), dtype=x.dtype, device=x.device)

    pos_flat = token_topk_to_pos.reshape(-1)
    mask = pos_flat >= 0
    valid_pos = pos_flat[mask]
    num_topk = token_topk_to_pos.shape[1]
    x_repeated = x.unsqueeze(1).expand(-1, num_topk, -1).reshape(-1, hidden)
    out[valid_pos] = x_repeated[mask]

    return out


def expand_to_fused_with_sf(
    x: QuantTensor,
    num_per_channels: int,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
    use_tma_aligned_col_major_sf: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand token activations and sf factors into the fused expert layout.

    Args:
        x: Input QuantTensor (data, sf) where data has shape
            (num_tokens, hidden) and sf has shape (num_tokens, hidden_sf).
        num_per_channels: Number of channels per scaling block (e.g. 128).
        token_topk_to_pos: Mapping from (token, topk) to expanded position,
            shape (num_tokens, num_topk). -1 means unused.
        pos_to_expert: Mapping from expanded position to expert index,
            shape (num_expanded_tokens,). Negative means padding.
        use_tma_aligned_col_major_sf: Whether sf uses TMA-aligned col-major layout.

    Returns:
        A tuple (out, out_sf) with expanded activation and sf-factor tensors.
    """
    x_data, x_sf = x
    num_tokens, hidden = x_data.shape
    num_expanded_tokens = pos_to_expert.shape[0]
    hidden_sf = x_sf.shape[1]

    out = torch.zeros((num_expanded_tokens, hidden), dtype=x_data.dtype, device=x_data.device)

    # Construct output scaling factor tensor.
    if use_tma_aligned_col_major_sf:
        num_expanded_sf_tokens = align(num_expanded_tokens, 4)
        out_sf = torch.zeros((hidden_sf, num_expanded_sf_tokens), dtype=x_sf.dtype, device=x_sf.device)
        out_sf = out_sf[:, :num_expanded_tokens]
        out_sf = out_sf.T
    else:
        out_sf = torch.zeros((num_expanded_tokens, hidden_sf), dtype=x_sf.dtype, device=x_sf.device)

    num_topk = token_topk_to_pos.shape[1]
    pos_flat = token_topk_to_pos.reshape(-1)  # (num_tokens * num_topk,)
    mask = pos_flat >= 0
    valid_pos = pos_flat[mask]

    x_data_rep = x_data.unsqueeze(1).expand(-1, num_topk, -1).reshape(-1, hidden)
    out[valid_pos] = x_data_rep[mask]

    x_sf_rep = x_sf.unsqueeze(1).expand(-1, num_topk, -1).reshape(-1, hidden_sf)
    out_sf[valid_pos] = x_sf_rep[mask]

    return out, out_sf
