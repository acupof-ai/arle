use super::{CUresult, CUstream};

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
}
