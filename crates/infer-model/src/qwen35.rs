//! Qwen3.6 host geometry: per-rank shard dims, recurrent-state sizes, the
//! joint KV-budget solve, and decode routing decisions. All pure arithmetic
//! on [`Qwen35Config`] + the per-rank head counts; the device probes (free-VRAM
//! query, NCCL min-reduce) stay in `infer-cuda` and enter here as plain values.

use infer_moe::MoeConfig;
use infer_seam::{PROFILE_KV_TOKENS_FLOOR, SlotBudget, profile_kv_pool_tokens};
use qwen35_spec::Qwen35Config;

/// This rank's per-shard head counts (= global config on a single GPU). The
/// forward sizes its buffers, slot state, and kernel launches from these, not
/// the config.
#[derive(Debug, Clone, Copy)]
pub struct LocalShard {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub linear_k_heads: usize,
    pub linear_v_heads: usize,
}

pub fn local_full_attn_q_dim(cfg: &Qwen35Config, shard: LocalShard) -> usize {
    shard.q_heads * cfg.head_dim
}

/// The gated q_proj output width: the projection interleaves `[query; gate]`
/// per head, so each local head contributes `2*head_dim` rows.
pub fn local_full_attn_q_proj_dim(cfg: &Qwen35Config, shard: LocalShard) -> usize {
    shard.q_heads * cfg.head_dim * 2
}

/// Single source for every full-attn K/V cache / pool size on this rank.
pub fn local_full_attn_kv_dim(cfg: &Qwen35Config, shard: LocalShard) -> usize {
    shard.kv_heads * cfg.head_dim
}

pub fn local_linear_qkv_dim(cfg: &Qwen35Config, shard: LocalShard) -> usize {
    let qk = 2 * shard.linear_k_heads * cfg.linear_key_head_dim;
    qk + shard.linear_v_heads * cfg.linear_value_head_dim
}

pub fn local_linear_z_dim(cfg: &Qwen35Config, shard: LocalShard) -> usize {
    shard.linear_v_heads * cfg.linear_value_head_dim
}

/// B2 CP decode (T3.1): the cp group can divide every attention head count
/// evenly. `false` when cp=1 or a head count is indivisible — decode then runs
/// replicated.
pub fn cp_decode_possible(cp_size: usize, shard: LocalShard) -> bool {
    cp_size > 1
        && shard.q_heads.is_multiple_of(cp_size)
        && shard.kv_heads.is_multiple_of(cp_size)
        && shard.linear_k_heads.is_multiple_of(cp_size)
        && shard.linear_v_heads.is_multiple_of(cp_size)
}

pub fn decode_q_heads(shard: LocalShard, cp_size: usize) -> usize {
    shard.q_heads / cp_size
}

pub fn decode_kv_heads(shard: LocalShard, cp_size: usize) -> usize {
    shard.kv_heads / cp_size
}

pub fn decode_linear_k_heads(shard: LocalShard, cp_size: usize) -> usize {
    shard.linear_k_heads / cp_size
}

pub fn decode_linear_v_heads(shard: LocalShard, cp_size: usize) -> usize {
    shard.linear_v_heads / cp_size
}

/// `(num_linear, gdr_state_len, conv_len)` for this rank's recurrent state,
/// sized from LOCAL shard widths. Single source for both slot acquisition and
/// the spec snapshot scratch.
pub fn recurrent_dims(cfg: &Qwen35Config, shard: LocalShard) -> (usize, usize, usize) {
    let num_linear = cfg.num_hidden_layers - cfg.num_full_attention_layers();
    let gdr_state_len = shard.linear_v_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim;
    let conv_len = local_linear_qkv_dim(cfg, shard) * (cfg.linear_conv_kernel_dim - 1);
    (num_linear, gdr_state_len, conv_len)
}

/// B2 CP decode dims for the 1/cp-head recurrent pair: same layer count,
/// 1/cp the gdr/conv widths (the engage guard makes both head counts
/// divisible by cp).
pub fn recurrent_dims_decode(
    cfg: &Qwen35Config,
    shard: LocalShard,
    cp_size: usize,
) -> (usize, usize, usize) {
    let (num_linear, gdr, conv) = recurrent_dims(cfg, shard);
    (num_linear, gdr / cp_size, conv / cp_size)
}

/// `(gdr_bytes, conv_bytes)` of one slot's linear state (f32 gated-delta +
/// bf16 conv). Snapshot scratch matches this exactly so a snapshot/restore is
/// a straight D2D copy.
pub fn linear_state_bytes(cfg: &Qwen35Config, shard: LocalShard) -> (usize, usize) {
    let (_, gdr, conv) = recurrent_dims(cfg, shard);
    (
        gdr * std::mem::size_of::<f32>(),
        conv * std::mem::size_of::<half::bf16>(),
    )
}

/// `(per_slot, kv_bytes, gdr_bytes, conv_bytes)` — full-attn K/V is paged
/// (shared pool), so `kv_bytes` is 0.
pub fn per_slot_kv_bytes(cfg: &Qwen35Config, shard: LocalShard) -> (usize, usize, usize, usize) {
    let num_linear = cfg.num_hidden_layers - cfg.num_full_attention_layers();
    let bf16 = std::mem::size_of::<half::bf16>();
    let f32sz = std::mem::size_of::<f32>();
    let (_, gdr_len, conv_len) = recurrent_dims(cfg, shard);
    let gdr_bytes = num_linear.saturating_mul(gdr_len).saturating_mul(f32sz);
    let conv_bytes = num_linear.saturating_mul(conv_len).saturating_mul(bf16);
    (
        gdr_bytes.saturating_add(conv_bytes),
        0,
        gdr_bytes,
        conv_bytes,
    )
}

/// Decode-graph gate: TRUE iff a `seq_len == 1` MoE step is a pure device-kernel
/// sequence — the device router (no host sync + D2H) and `R = top_k` below the
/// DeepGEMM floor, whose JIT is not capture-safe. `min_routes` is the runtime
/// flag's value, read in `infer-cuda` and passed as a host number.
pub fn moe_decode_graph_capturable(cfg: &MoeConfig, min_routes: usize) -> bool {
    cfg.device_route_eligible() && cfg.top_k < min_routes
}

/// Why a decode step cannot run as a captured graph, if anything. `has_moe`
/// and `moe` come from the loaded model; `min_routes` from the runtime flag.
pub fn decode_graph_unsupported_reason(
    has_moe: bool,
    moe: Option<&MoeConfig>,
    min_routes: usize,
) -> Option<&'static str> {
    if !has_moe {
        return None;
    }
    let Some(cfg) = moe else {
        return Some("MoE layers present but no moe_config");
    };
    if !moe_decode_graph_capturable(cfg, min_routes) {
        return Some(
            "MoE decode is not device-routable (host router fallback active — \
             non-greedy/grouped routing)",
        );
    }
    None
}

/// Fraction of measured free VRAM the KV budget may claim. Same neutral kernel
/// as DSv4.
pub const KV_MEM_FRACTION: f64 = 0.7;

/// Phase 1 of the joint KV-budget solve: the largest state-affordable slot
/// count whose pool remainder still funds one full-length (`max_seq_len`)
/// request, from this rank's free-VRAM probe. Feasibility is monotone in
/// decreasing n, so the first feasible n scanning down is the max.
///
/// `probe` is the measured free AFTER any startup-fixed grant cap — the cap
/// policy and its log live in the caller. Returns `i32::MAX` when the probe
/// failed — that rank must not bind the cross-rank min. A 0 return means
/// post-weights free VRAM holds no slot at all; the reject guard fires on the
/// reduced value in the caller.
///
/// `attn_ws_per_slot` is the quantized-FA3 split-KV workspace, charged per
/// slot because the kernel sizes it `num_splits × num_slots × …` (0 for
/// non-quantized pools). It is deducted alongside the recurrent per-slot
/// grant so the pool remainder funds only what survives both.
pub fn kv_slot_budget_local(
    probe: Option<(usize, usize)>,
    per_slot: usize,
    attn_ws_per_slot: usize,
    requested: usize,
    max_seq_len: usize,
    mem_fraction_static: f64,
    cell_bytes_per_token: u64,
) -> i32 {
    let Some((free, total)) = probe else {
        return i32::MAX;
    };
    let granted_per_slot = per_slot.saturating_add(attn_ws_per_slot);
    let pool_tokens_at = |free: usize, n: usize| -> u64 {
        profile_kv_pool_tokens(
            (free as u64).saturating_sub(granted_per_slot.saturating_mul(n) as u64),
            total as u64,
            cell_bytes_per_token,
            mem_fraction_static,
        )
    };
    let budget = SlotBudget::from_free(free, KV_MEM_FRACTION, 0, granted_per_slot);
    let affordable = budget.affordable().unwrap_or(usize::MAX);
    let mut n = requested.max(1).min(affordable);
    while n > 1 && pool_tokens_at(free, n) < max_seq_len as u64 {
        n -= 1;
    }
    if affordable == 0 {
        n = 0;
    }
    i32::try_from(n).unwrap_or(i32::MAX)
}

/// Phase 2: the shared-pool page count at the REDUCED slot count. `i32::MAX`
/// when the probe failed (every rank failing → the requested floor, in
/// [`finalize_pool_pages`]). `num_full` / `local_kv_heads` / `head_dim` are the
/// cell's decomposition, carried for the budget log (the cell itself is
/// computed caller-side, as it needs the backend's KV-format type).
pub fn kv_pool_pages_local(
    probe: Option<(usize, usize)>,
    planned: usize,
    per_slot: usize,
    attn_ws_per_slot: usize,
    cell_bytes_per_token: u64,
    num_full: usize,
    local_kv_heads: usize,
    head_dim: usize,
    requested_pages: usize,
    mem_fraction_static: f64,
    page_size: usize,
) -> i32 {
    let Some((free, total)) = probe else {
        return i32::MAX;
    };
    let granted_per_slot = per_slot.saturating_add(attn_ws_per_slot);
    let recurrent_bytes = per_slot.saturating_mul(planned);
    let attn_ws_bytes = attn_ws_per_slot.saturating_mul(planned);
    let profiled_tokens = profile_kv_pool_tokens(
        (free as u64).saturating_sub(granted_per_slot.saturating_mul(planned) as u64),
        total as u64,
        cell_bytes_per_token,
        mem_fraction_static,
    );
    if profiled_tokens == PROFILE_KV_TOKENS_FLOOR {
        // `mem_fraction_static` bounds the engine's share of TOTAL, so a value
        // under the weights' own share caps admission at the floor even at
        // num_slots 1.
        let reserve = (total as f64 * (1.0 - mem_fraction_static)) as u64;
        log::warn!(
            "KV pool collapsed to the {}-token floor even at num_slots {planned}: \
             free {}MB − recurrent {}MB − FA3 attn workspace {}MB − reserve {}MB \
             (= total {}MB × (1 − mem_fraction_static {mem_fraction_static})) leaves \
             nothing for {cell_bytes_per_token}B/tok cells. Raise mem_fraction_static, \
             or free VRAM: every prompt over {} tokens will abort.",
            PROFILE_KV_TOKENS_FLOOR,
            free >> 20,
            recurrent_bytes >> 20,
            attn_ws_bytes >> 20,
            reserve >> 20,
            total >> 20,
            PROFILE_KV_TOKENS_FLOOR,
        );
    }
    let profiled_pages = (profiled_tokens / page_size as u64).max(1) as usize;
    log::info!(
        "CUDA Qwen3.6 full-attn KV pool profiled from measured VRAM: free {}MB / \
         total {}MB, recurrent {}MB + FA3 attn workspace {}MB ({planned} slots × \
         {}MB recurrent + {}MB workspace), mem_fraction_static {mem_fraction_static}, \
         cell {cell_bytes_per_token}B/tok ({num_full} full-attn layers × {local_kv_heads} \
         kv-heads × {head_dim} hd) -> max_total_tokens {profiled_tokens} ({profiled_pages} \
         pages); requested {requested_pages} pages (advisory)",
        free >> 20,
        total >> 20,
        recurrent_bytes >> 20,
        attn_ws_bytes >> 20,
        per_slot >> 20,
        attn_ws_per_slot >> 20,
    );
    i32::try_from(profiled_pages).unwrap_or(i32::MAX)
}

/// The cross-rank-reduced page count → the plan value: every rank's probe
/// failed (`i32::MAX`) → the requested floor, matching the old
/// profile-failure fallback.
pub fn finalize_pool_pages(reduced: usize, requested_pages: usize) -> usize {
    if reduced == i32::MAX as usize {
        log::warn!(
            "CUDA Qwen3.6 full-attn KV pool: free-VRAM probe failed on every rank; \
             falling back to requested floor {requested_pages} pages"
        );
        requested_pages
    } else {
        reduced
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cp_decode_possible_divisibility() {
        let shard = LocalShard {
            q_heads: 16,
            kv_heads: 8,
            linear_k_heads: 4,
            linear_v_heads: 4,
        };
        assert!(!cp_decode_possible(1, shard));
        assert!(cp_decode_possible(4, shard));
        assert!(!cp_decode_possible(3, shard));
        assert!(!cp_decode_possible(8, shard)); // kv 8 ok, lin 4 not
    }

    #[test]
    fn decode_heads_divide_by_cp() {
        let shard = LocalShard {
            q_heads: 16,
            kv_heads: 8,
            linear_k_heads: 4,
            linear_v_heads: 4,
        };
        assert_eq!(decode_q_heads(shard, 4), 4);
        assert_eq!(decode_kv_heads(shard, 4), 2);
        assert_eq!(decode_linear_v_heads(shard, 4), 1);
    }

    #[test]
    fn kv_slot_budget_joint_solve() {
        // free=1e6, reserve=1e5 (frac 0.9), cell=1, max_seq_len=300_000.
        // affordable = 0.7*1e6/1e4 = 70, but at n=70 the pool remainder
        // (900_000 − 700_000 = 200_000 tokens) cannot fund 300_000 → the scan
        // sheds to n=60 (remainder 300_000, exactly feasible).
        let n = kv_slot_budget_local(Some((1_000_000, 1_000_000)), 10_000, 0, 80, 300_000, 0.9, 1);
        assert_eq!(n, 60);
        // Probe failure must not bind the min.
        assert_eq!(kv_slot_budget_local(None, 1, 0, 1, 1, 0.9, 1), i32::MAX);
        // Nothing affordable → 0 (the caller rejects on the reduced value).
        assert_eq!(
            kv_slot_budget_local(Some((1024, 1024)), 1 << 30, 0, 1, 1, 0.9, 1),
            0
        );
    }

    #[test]
    fn kv_slot_budget_charges_fa3_workspace_per_slot() {
        // Same inputs as kv_slot_budget_joint_solve but with a 2_000B/slot FA3
        // workspace: granted/slot rises 10_000 → 12_000, so the scan sheds from
        // n=60 to n=50 (900_000 − 12_000·50 = 300_000, exactly feasible).
        let n = kv_slot_budget_local(
            Some((1_000_000, 1_000_000)),
            10_000,
            2_000,
            80,
            300_000,
            0.9,
            1,
        );
        assert_eq!(n, 50);
    }

    #[test]
    fn kv_pool_pages_and_finalize() {
        let pages = kv_pool_pages_local(
            Some((32 << 30, 32 << 30)),
            64,
            64 << 20,
            0,
            584,
            4,
            8,
            128,
            4096,
            0.9,
            128,
        );
        assert!(pages > 0 && pages != i32::MAX);
        assert_eq!(
            kv_pool_pages_local(None, 64, 1, 0, 1, 4, 8, 128, 4096, 0.9, 128),
            i32::MAX
        );
        assert_eq!(finalize_pool_pages(i32::MAX as usize, 4096), 4096);
        assert_eq!(finalize_pool_pages(123, 4096), 123);
    }

    #[test]
    fn decode_graph_reason() {
        assert_eq!(decode_graph_unsupported_reason(false, None, 1024), None);
        assert_eq!(
            decode_graph_unsupported_reason(true, None, 1024),
            Some("MoE layers present but no moe_config")
        );
        let capturable = MoeConfig::qwen36(64, 8, true, 5120);
        assert_eq!(
            decode_graph_unsupported_reason(true, Some(&capturable), 1024),
            None
        );
        // top_k at/above the floor → host router fallback.
        assert!(decode_graph_unsupported_reason(true, Some(&capturable), 8).is_some());
    }
}
