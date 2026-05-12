import torch

from tile_kernels.utils import align, ceil_div


def clear_unused_sf(sf: torch.Tensor, hidden: int, num_per_channels: int) -> torch.Tensor:
    """Zero out unused sf entries beyond the actual channel block count.

    Args:
        sf: Scale-factor tensor to clean up.
        hidden: Number of hidden channels in the original tensor.
        num_per_channels: Number of channels per scaling block.

    Returns:
        Flattened sf tensor with unused trailing entries set to zero.
    """
    num_channel_blocks = ceil_div(hidden, num_per_channels)
    aligned_num_channel_blocks = align(num_channel_blocks, 4)
    sf_flattened = (sf.contiguous().flatten().view(torch.uint8)).view(-1, aligned_num_channel_blocks)
    sf_flattened[:, num_channel_blocks:] = 0
    return sf_flattened
