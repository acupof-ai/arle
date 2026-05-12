import torch


def aux_fi(topk_idx: torch.Tensor, num_experts: int, num_aux_topk: int) -> torch.Tensor:
    """Compute auxiliary load-balancing frequency indicator f_i for each expert.

    ``f_i[e] = count[e] * num_experts / (num_tokens * num_aux_topk)``

    Args:
        topk_idx: Expert indices of shape ``(num_tokens, num_topk)``.
            Entries ``< 0`` are treated as padding and ignored.
        num_experts: Total number of experts.
        num_aux_topk: Number of top-k slots used for the auxiliary loss.

    Returns:
        Float32 tensor of shape ``(num_experts,)`` with the f_i values.
    """
    num_tokens, num_topk = topk_idx.shape
    if num_tokens == 0:
        return torch.zeros(num_experts, dtype=torch.float32, device=topk_idx.device)
    valid_idx = topk_idx[topk_idx >= 0]
    counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_idx.device)
    counts.scatter_add_(0, valid_idx, torch.ones_like(valid_idx))
    return counts.float() * num_experts / (num_tokens * num_aux_topk)


def group_count(group_idx: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Count the number of tokens assigned to each group, ignoring padding.

    Args:
        group_idx: Group indices tensor.  Entries ``< 0`` are ignored.
        num_groups: Total number of groups.

    Returns:
        Int32 tensor of shape ``(num_groups,)`` with per-group counts.
    """
    valid_idx = group_idx[group_idx >= 0]
    counts = torch.zeros(num_groups, dtype=torch.int32, device=group_idx.device)
    counts.scatter_add_(0, valid_idx, torch.ones_like(valid_idx, dtype=torch.int32))
    return counts


def mask_indices_by_tp(
    indices: torch.Tensor,
    n: int,
    num_ep_ranks: int,
    tp_rank: int,
    num_tp_ranks: int,
) -> torch.Tensor:
    """Mask expert indices to keep only those belonging to the given TP rank.

    Args:
        indices: Expert index tensor.
        n: Total number of experts across all EP ranks (``num_experts * num_ep_ranks``).
        num_ep_ranks: Number of expert-parallel ranks.
        tp_rank: Tensor-parallel rank to keep.
        num_tp_ranks: Total number of tensor-parallel ranks.

    Returns:
        Tensor of the same shape with non-local indices set to ``-1`` and local
        indices remapped to the local expert numbering.
    """
    per_gpu = n // num_ep_ranks
    per_dp = num_tp_ranks * per_gpu

    value = indices.clone()
    invalid = (value < 0) | ((value // per_gpu) % num_tp_ranks != tp_rank)

    value = value - tp_rank * per_gpu
    dp_rank = value // per_dp
    value = value - dp_rank * (per_dp - per_gpu)

    value[invalid | (value < 0)] = -1
    return value


def normalize_weight(topk_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize each token's top-k weights so they sum to one.

    Args:
        topk_weights: Float32 tensor of shape ``(num_tokens, num_topk)``.

    Returns:
        Tuple of ``(denominator, normalized_weights)`` where *denominator* has
        shape ``(num_tokens,)`` and *normalized_weights* has the same shape as
        the input.
    """
    num_tokens, num_topk = topk_weights.shape
    denominator = torch.full((num_tokens,), 1e-20, dtype=torch.float32, device=topk_weights.device)
    for k in range(num_topk):
        denominator = denominator + topk_weights[:, k]
    normalized_weights = topk_weights / denominator.unsqueeze(1)
    return denominator, normalized_weights


def inplace_unique_group_indices(group_indices: torch.Tensor, num_groups: int) -> None:
    """Deduplicate group indices in-place, keeping only the first occurrence per row.

    For each row, if a group index appears more than once, all but the first
    (leftmost) occurrence are replaced with ``-1``.

    Args:
        group_indices: Int tensor of shape ``(num_tokens, num_topk)``, modified in-place.
        num_groups: Total number of groups (unused, kept for API consistency).
    """
    num_tokens, num_topk = group_indices.shape

    # stable sort within each row
    vals, idx = torch.sort(group_indices, dim=1, stable=True)

    # find first occurrence in the sorted order (per row)
    first_in_sorted = torch.ones((num_tokens, num_topk), dtype=torch.bool, device=group_indices.device)
    first_in_sorted[:, 1:] = vals[:, 1:] != vals[:, :-1]
    dup_in_sorted = ~first_in_sorted

    # map duplicate markers back to original positions
    dup_in_orig = torch.zeros((num_tokens, num_topk), dtype=torch.bool, device=group_indices.device)
    dup_in_orig.scatter_(1, idx, dup_in_sorted)

    group_indices[dup_in_orig] = -1
