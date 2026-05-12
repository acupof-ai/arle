import torch
import torch.nn.functional as F
from typing import Optional

from tile_kernels.moe.scoring import ScoringFunc


def stable_topk(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    _, sorted_indices = torch.sort(scores, dim=1, descending=True, stable=True)
    return sorted_indices[:, :num_topk].contiguous()


def topk_sum_and_topk_group_idx(
    scores: torch.Tensor,
    num_group_sum_topk: int,
    num_topk_groups: int,
) -> torch.Tensor:
    group_scores_ref = scores.topk(num_group_sum_topk, dim=-1, sorted=False).values.sum(-1)
    return stable_topk(group_scores_ref, num_topk_groups)


def top2_sum_gate(
    logits: torch.Tensor,
    bias: torch.Tensor,
    num_topk: int,
    num_topk_groups: int,
    num_groups: int,
    use_shared_as_routed: bool,
    num_shared_experts: int,
    routed_scaling_factor: float,
    ep_rank: int,
    num_ep_ranks: int,
    tp_rank: int,
    num_tp_ranks: int,
    scoring_func: str,
    mask: Optional[torch.Tensor] = None,
    fix_routing_mask: Optional[torch.Tensor] = None,
    to_physical_map: Optional[torch.Tensor] = None,
    logical_count: Optional[torch.Tensor] = None,
    unmapped_topk_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for top-k expert routing with top-2 sum grouping.

    Args:
        logits: Raw token-expert logits, shape ``(num_tokens, num_routed_experts)``,
            ``float32``.
        bias: Per-expert bias added to scores before ranking, shape
            ``(num_routed_experts,)``, ``float32``.
        num_topk: Number of routed experts to select per token.
        num_topk_groups: Number of expert groups to keep (0 means no grouping).
        num_groups: Total number of expert groups (0 means no grouping).
        use_shared_as_routed: Whether shared experts are appended as extra routed
            slots in the output.
        num_shared_experts: Number of shared experts.
        routed_scaling_factor: Multiplicative scaling applied to normalised weights.
        ep_rank: Expert-parallelism rank of this process.
        num_ep_ranks: Total number of expert-parallelism ranks.
        tp_rank: Tensor-parallelism rank of this process.
        num_tp_ranks: Total number of tensor-parallelism ranks.
        scoring_func: One of ``'sigmoid'``, ``'sqrtsoftplus'``, ``'softmax'``.
        mask: Boolean mask, shape ``(num_tokens,)``.  ``True`` → route the token,
            ``False`` → fill outputs with ``-1`` / ``0``.
        fix_routing_mask: Boolean mask, shape ``(num_tokens,)``.  When ``True`` for
            a token, use the indices already stored in *unmapped_topk_idx* instead of
            running the selection algorithm.
        to_physical_map: Logical-to-physical expert map, shape
            ``(num_logical_experts, num_duplicate_experts + 1)``, ``int32``.
        logical_count: Number of active duplicates per logical expert, shape
            ``(num_logical_experts,)``, ``int32``.
        unmapped_topk_idx: Output tensor (updated in-place) for unmasked expert
            indices, shape ``(num_tokens, num_topk)``, ``int64``.

    Returns:
        topk_idx: Selected (post-EP/TP masking) expert indices, shape
            ``(num_tokens, num_topk + num_shared_experts)``, ``int64``.
        topk_weights: Normalised expert weights, same shape, ``float32``.
    """
    num_tokens_full, num_routed_experts = logits.shape
    scoring = ScoringFunc.from_str(scoring_func)

    if not use_shared_as_routed:
        num_shared_experts = 0

    num_physical_topk = num_topk + num_shared_experts
    num_logical_experts = num_routed_experts + num_shared_experts
    device = logits.device

    topk_idx_out = torch.full((num_tokens_full, num_physical_topk), -1, dtype=torch.int64, device=device)
    topk_weights_out = torch.zeros((num_tokens_full, num_physical_topk), dtype=torch.float32, device=device)

    if num_tokens_full == 0:
        return topk_idx_out, topk_weights_out

    active = mask if mask is not None else torch.ones(num_tokens_full, dtype=torch.bool, device=device)
    active_indices = active.nonzero(as_tuple=False).squeeze(1)
    num_tokens = active_indices.numel()

    if num_tokens == 0:
        if unmapped_topk_idx is not None:
            unmapped_topk_idx[~active] = -1
        return topk_idx_out, topk_weights_out

    logits_a = logits[active_indices]
    bias_b = bias.unsqueeze(0)

    # 1. Apply scoring function
    if scoring == ScoringFunc.SIGMOID:
        scores_wo_bias = torch.sigmoid(logits_a)
    elif scoring == ScoringFunc.SQRTSOFTPLUS:
        scores_wo_bias = F.softplus(logits_a).sqrt()
    else:  # SOFTMAX
        scores_wo_bias = torch.softmax(logits_a, dim=-1)

    # 2. Biased scores for ranking (softmax uses raw logits + bias)
    scores_biased = (logits_a + bias_b) if scoring == ScoringFunc.SOFTMAX else (scores_wo_bias + bias_b)

    # 3. Split tokens into normal routing and fix_routing
    fix_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    if fix_routing_mask is not None and unmapped_topk_idx is not None:
        fix_mask = fix_routing_mask[active_indices]

    topk_idx_local = torch.full((num_tokens, num_topk), -1, dtype=torch.int64, device=device)
    topk_score_local = torch.zeros((num_tokens, num_topk), dtype=torch.float32, device=device)

    # 4. Normal routing: select top-k experts
    normal_mask = ~fix_mask
    if normal_mask.any():
        normal_indices = normal_mask.nonzero(as_tuple=False).squeeze(1)
        sb = scores_biased[normal_indices]

        if num_groups != num_topk_groups:
            num_per_group = num_routed_experts // num_groups
            top_group_idx = topk_sum_and_topk_group_idx(sb.view(-1, num_groups, num_per_group), 2, num_topk_groups)
            group_mask = torch.ones((normal_indices.numel(), num_groups), dtype=torch.bool, device=device)
            group_mask.scatter_(1, top_group_idx, False)
            sb = sb.masked_fill(
                group_mask.unsqueeze(-1).expand(-1, num_groups, num_per_group).reshape(-1, num_routed_experts),
                float('-inf'),
            )

        selected = stable_topk(sb, num_topk)
        topk_idx_local[normal_indices] = selected
        topk_score_local[normal_indices] = scores_wo_bias[normal_indices].gather(1, selected)

    # 5. Fix routing: use pre-stored indices
    if fix_mask.any() and unmapped_topk_idx is not None:
        fix_indices = fix_mask.nonzero(as_tuple=False).squeeze(1)
        pre_idx = unmapped_topk_idx[active_indices[fix_indices]]
        topk_idx_local[fix_indices] = pre_idx
        topk_score_local[fix_indices] = scores_wo_bias[fix_indices].gather(1, pre_idx.clamp(min=0))

    # 6. Write unmapped_topk_idx
    if unmapped_topk_idx is not None:
        unmapped_topk_idx[active_indices] = topk_idx_local
        if mask is not None:
            unmapped_topk_idx[~active] = -1

    # 7. Normalise weights (top-sum normalisation)
    topk_sum = topk_score_local.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    topk_weights_routed = topk_score_local / topk_sum * routed_scaling_factor

    # 8. Append shared-expert slots
    if num_shared_experts > 0:
        shared_idx = torch.arange(num_routed_experts, num_logical_experts, dtype=torch.int64, device=device)
        topk_idx_all = torch.cat([topk_idx_local, shared_idx.expand(num_tokens, -1)], dim=1)
        topk_weights_all = torch.cat(
            [
                topk_weights_routed,
                torch.ones((num_tokens, num_shared_experts), dtype=torch.float32, device=device),
            ],
            dim=1,
        )
    else:
        topk_idx_all, topk_weights_all = topk_idx_local, topk_weights_routed

    # 9. Map logical → physical experts
    if to_physical_map is not None and logical_count is not None:
        for lane in range(num_physical_topk):
            logical = topk_idx_all[:, lane]
            valid = logical >= 0
            if valid.any():
                global_idx = active_indices[valid].to(torch.int64)
                dup_idx = (ep_rank + global_idx * 23333) % logical_count[logical[valid]].to(torch.int64)
                topk_idx_all[valid, lane] = to_physical_map[logical[valid], dup_idx].to(torch.int64)

    # 10. EP / TP masking
    num_extra = to_physical_map.shape[1] - 1 if to_physical_map is not None else 0
    experts_per_rank = (num_routed_experts + num_extra) // num_ep_ranks
    experts_per_dp = experts_per_rank * num_tp_ranks

    idx = topk_idx_all
    valid = idx >= 0
    ep_of = torch.where(valid, idx // experts_per_rank, torch.zeros_like(idx))
    idx = torch.where(valid & (ep_of % num_tp_ranks != tp_rank), -1, idx)

    valid = idx >= 0
    local = idx - tp_rank * experts_per_rank
    dp_of = torch.where(valid, local // experts_per_dp, torch.zeros_like(local))
    remapped = local - dp_of * (experts_per_dp - experts_per_rank)
    idx = torch.where(valid & (remapped >= 0), remapped, torch.where(valid, -1, idx))

    # 11. Write outputs
    topk_idx_out[active_indices] = idx
    topk_weights_out[active_indices] = topk_weights_all

    return topk_idx_out, topk_weights_out
