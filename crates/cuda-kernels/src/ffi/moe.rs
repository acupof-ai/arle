use super::{CUresult, CUstream, Half};

#[allow(dead_code)]
unsafe extern "C" {
    pub fn dsv4_mask_indices_by_ep_i64_cuda(
        indices: *const i64,
        masked_indices: *mut i64,
        num_tokens: i32,
        num_topk: i32,
        experts_per_ep_rank: i32,
        experts_per_moe_dp_group: i32,
        num_tp_ranks: i32,
        tp_rank: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_mask_indices_by_ep_i32_cuda(
        indices: *const i32,
        masked_indices: *mut i32,
        num_tokens: i32,
        num_topk: i32,
        experts_per_ep_rank: i32,
        experts_per_moe_dp_group: i32,
        num_tp_ranks: i32,
        tp_rank: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_route_cuda(
        logits: *const Half,
        bias: *const Half,
        tid2eid: *const i64,
        token_ids: *const u32,
        indices: *mut i32,
        weights: *mut f32,
        num_tokens: i32,
        n_experts: i32,
        topk: i32,
        routing_kind: i32,
        scoring_kind: i32,
        routed_scaling_factor: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_count_local_experts_cuda(
        indices: *const i32,
        counts: *mut i32,
        num_tokens: i32,
        topk: i32,
        local_expert_start: i32,
        experts_per_rank: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_count_expert_ranks_cuda(
        indices: *const i32,
        counts: *mut i32,
        num_tokens: i32,
        topk: i32,
        experts_per_rank: i32,
        ep_world_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_pack_local_experts_cuda(
        hidden: *const Half,
        indices: *const i32,
        weights: *const f32,
        offsets: *const i32,
        cursors: *mut i32,
        packed_hidden: *mut Half,
        packed_token: *mut i32,
        packed_weight: *mut f32,
        num_tokens: i32,
        hidden_dim: i32,
        topk: i32,
        local_expert_start: i32,
        experts_per_rank: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_pack_expert_ranks_cuda(
        hidden: *const Half,
        indices: *const i32,
        weights: *const f32,
        offsets: *const i32,
        cursors: *mut i32,
        packed_hidden: *mut Half,
        packed_token: *mut i32,
        packed_route_slot: *mut i32,
        packed_meta: *mut i32,
        num_tokens: i32,
        hidden_dim: i32,
        topk: i32,
        experts_per_rank: i32,
        ep_world_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_count_packed_local_experts_cuda(
        packed_meta: *const i32,
        counts: *mut i32,
        num_routes: i32,
        local_expert_start: i32,
        experts_per_rank: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_pack_received_experts_cuda(
        received_hidden: *const Half,
        received_meta: *const i32,
        offsets: *const i32,
        cursors: *mut i32,
        expert_hidden: *mut Half,
        expert_token: *mut i32,
        expert_weight: *mut f32,
        expert_route_slot: *mut i32,
        num_routes: i32,
        hidden_dim: i32,
        local_expert_start: i32,
        experts_per_rank: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_scatter_packed_expert_cuda(
        expert_out: *const Half,
        routed_out: *mut Half,
        packed_token: *const i32,
        packed_weight: *const f32,
        start_slot: i32,
        count: i32,
        hidden_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_scatter_packed_route_slot_cuda(
        expert_out: *const Half,
        route_out: *mut Half,
        expert_route_slot: *const i32,
        expert_weight: *const f32,
        start_slot: i32,
        count: i32,
        hidden_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_scatter_all_route_slots_cuda(
        expert_out: *const Half,
        route_out: *mut Half,
        expert_route_slot: *const i32,
        expert_weight: *const f32,
        num_routes: i32,
        hidden_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_scatter_route_outputs_by_slot_cuda(
        packed_route_out: *const Half,
        route_slot_out: *mut Half,
        packed_route_slot: *const i32,
        num_routes: i32,
        hidden_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_combine_route_slot_outputs_cuda(
        route_slot_out: *const Half,
        routed_out: *mut Half,
        num_tokens: i32,
        topk: i32,
        hidden_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_combine_route_outputs_cuda(
        route_out: *const Half,
        packed_token: *const i32,
        routed_out: *mut Half,
        num_tokens: i32,
        num_routes: i32,
        hidden_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_add_local_expert_cuda(
        expert_out: *const Half,
        routed_out: *mut Half,
        indices: *const i32,
        weights: *const f32,
        num_tokens: i32,
        hidden_dim: i32,
        topk: i32,
        global_expert_idx: i32,
        stream: CUstream,
    ) -> CUresult;
}
