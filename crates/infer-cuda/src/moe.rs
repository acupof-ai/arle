//! Single-GPU BF16 MoE forward — Qwen3.5/3.6 SparseMoeBlock (all experts local).
//!
//! ```text
//!  router gemm  → logits[T, E]
//!    → dsv4_route (DEVICE, zero bias)  → route_indices[T*topk], route_weights[T*topk]
//!    → qwen36_renorm_topk_weights      → norm_topk_prob renorm (in-place)
//!      [ARLE_QWEN35_GPU_ROUTER=0 fallback: infer_moe::route (HOST) + flatten_routing
//!       + ctx.sync + logits D2H + indices/weights H2D]
//!    → dsv4_count_local_experts        → counts[E]
//!    → dsv4_exclusive_scan_i32         → offsets[E]
//!    → dsv4_pack_local_experts_with_slots
//!                                      → packed_hidden[R,H], packed_route_slot[R],
//!                                        packed_weight[R]   (R = T*topk)
//!    → R ≤ 256: moe_bf16_grouped_gemm_swiglu_decode (fused gate+up+SwiGLU,
//!               weight-read-bound decode kernels; ARLE_QWEN35_MOE_DECODE_KERNEL=0
//!               opts out) → moe_bf16_grouped_gemm_decode (down)
//!      R > 256: moe_bf16_grouped_gemm_pair_batch (gate+up)
//!               → silu_mul (unclamped SwiGLU — Qwen3.6 has no clamp)
//!               → moe_bf16_grouped_gemm_batch (down)
//!    → dsv4_scatter_all_route_slots    → route_out[R,H]   (weight·expert_out, by slot)
//!    → dsv4_combine_route_slot_outputs → routed[T,H]      (sum over topk)
//!    → shared expert dense SwiGLU + qwen36_add_shared_expert_gated
//! ```
//!
//! Routing runs on DEVICE by default (`dsv4_route` with a zero selection bias is
//! exactly greedy top-k; audit lane MOE-OK-3 established host/device semantic
//! identity for Qwen3.6). The host `infer_moe::route` path stays as the
//! `ARLE_QWEN35_GPU_ROUTER=0` fallback for the pod A/B license — it costs a
//! full-stream `ctx.sync` + logits D2H + 2×H2D per layer per step.
//! W4/4-bit is a separate follow-up: the two `moe_bf16_grouped_gemm_*` call sites
//! are the swap points; everything else is dtype-agnostic on BF16 activations.

use infer_moe::RoutingDecision;

/// Host hash routing for a DSv4 hash-routed MoE layer: each token's experts come
/// straight from the `tid2eid` table (`token_id * topk .. + topk`), weighted by
/// the layer's router scores (sqrtsoftplus + normalize + `routed_scaling_factor`,
/// via `config.moe_routes_from_scores`). No learned router gate / bias.
///
/// `logits` is the row-major `[num_tokens, n_routed_experts]` router gemm output
/// (used only for the SCORES that weight the hash-picked experts). Returns one
/// [`RoutingDecision`] per token with exactly `topk` experts, matching the bias
/// path's contract so `flatten_routing` is shared. DSv4-only, so gated on `cuda`
/// (the `deepseek-spec` dep is a `cuda`-feature dependency).
#[cfg(feature = "cuda")]
pub(crate) fn hash_route(
    config: &deepseek_spec::DeepSeekV4Config,
    tid2eid: &[i64],
    tokens: &[u32],
    logits: &[f32],
) -> anyhow::Result<Vec<RoutingDecision>> {
    use infer_moe::ExpertWeight;

    let num_experts = config.n_routed_experts;
    let topk = config.num_experts_per_tok;
    anyhow::ensure!(
        logits.len() == tokens.len() * num_experts,
        "DSv4 hash route logits len {} != tokens {} * experts {num_experts}",
        logits.len(),
        tokens.len()
    );
    let mut decisions = Vec::with_capacity(tokens.len());
    for (token_idx, &token) in tokens.iter().enumerate() {
        let token = token as usize;
        anyhow::ensure!(
            token < config.vocab_size,
            "DSv4 hash route token {token} exceeds vocab_size {}",
            config.vocab_size
        );
        let start = token * topk;
        let end = start + topk;
        anyhow::ensure!(
            end <= tid2eid.len(),
            "DSv4 tid2eid too short: need {end} for token {token}, have {}",
            tid2eid.len()
        );
        let experts: Vec<usize> = tid2eid[start..end]
            .iter()
            .map(|&e| {
                anyhow::ensure!(e >= 0, "DSv4 tid2eid has negative expert id {e}");
                usize::try_from(e)
                    .map_err(|_| anyhow::anyhow!("DSv4 tid2eid expert id {e} overflow"))
            })
            .collect::<anyhow::Result<_>>()?;

        let row = &logits[token_idx * num_experts..(token_idx + 1) * num_experts];
        let scores = config
            .router_scores_from_logits(row)
            .map_err(|e| anyhow::anyhow!("DSv4 hash route scores: {e}"))?;
        let routes = config
            .moe_routes_from_scores(0, token_idx, &scores, None, Some(&experts))
            .map_err(|e| anyhow::anyhow!("DSv4 hash route weights: {e}"))?;
        decisions.push(RoutingDecision {
            experts: routes
                .into_iter()
                .map(|r| ExpertWeight {
                    expert: r.expert_idx,
                    weight: r.weight,
                })
                .collect(),
        });
    }
    Ok(decisions)
}

/// Flatten per-token [`RoutingDecision`]s into the token-major flat buffers the
/// `dsv4_*` kernels read at `route = token * topk + k`: `route_indices` (expert
/// id), `route_weights` (gate weight), each length `num_tokens * topk`. Each
/// decision must carry exactly `topk` experts; selection order is preserved.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn flatten_routing(
    decisions: &[RoutingDecision],
    topk: usize,
) -> anyhow::Result<(Vec<i32>, Vec<f32>)> {
    let total = decisions.len() * topk;
    let mut indices = Vec::with_capacity(total);
    let mut weights = Vec::with_capacity(total);
    for (token, decision) in decisions.iter().enumerate() {
        anyhow::ensure!(
            decision.experts.len() == topk,
            "routing decision for token {token} has {} experts, expected topk {topk}",
            decision.experts.len()
        );
        for ew in &decision.experts {
            indices.push(
                i32::try_from(ew.expert).map_err(|_| {
                    anyhow::anyhow!("expert id {} exceeds i32 kernel ABI", ew.expert)
                })?,
            );
            weights.push(ew.weight);
        }
    }
    Ok((indices, weights))
}

/// DeepGEMM m-grouped-contiguous row alignment on SM90 (the upstream
/// `get_mk_alignment_for_contiguous_layout()`): the kernel resolves the B-side
/// expert group ONCE per BLOCK_M output tile from `m_indices[tile_start]`
/// (vendor `deep_gemm/scheduler/gemm.cuh::get_global_idx`), and BLOCK_M for
/// the contiguous layout is pinned to 128 — so every expert group's row
/// segment must start at a multiple of 128, with `-1` on every pad row.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const DEEPGEMM_CONTIG_ALIGN: usize = 128;

/// Host upper bound on the 128-aligned packed row count for the DeepGEMM
/// contiguous grouped layout. The true total `Σ_g align(count_g, 128)` is
/// device-resident (counts come from the on-device router), and the GEMM's
/// `m` / the TMA descriptors are host values — so the launch uses this cap
/// instead of a per-layer D2H sync:
///
/// `Σ_g align(c_g, 128) ≤ R + 127·G_active ≤ align(R, 128) + 128·min(R, G)`
///
/// (G_active ≤ min(R, G) since every active group has ≥ 1 route). Tiles past
/// the true aligned total carry `m_indices = -1` and compute excluded garbage.
/// Always a multiple of 128 (the native bridge rejects unaligned `m`).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn deepgemm_contig_rows_cap(total_routes: usize, local_experts: usize) -> usize {
    let a = DEEPGEMM_CONTIG_ALIGN;
    total_routes.div_ceil(a) * a + a * local_experts.min(total_routes)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
/// Routed-row floor below which the DeepGEMM grouped path loses to the hand
/// CUDA-core kernels (pod A/B 2026-06-11: decode R=8 hand wins +8%, prefill
/// R=16384 DeepGEMM wins -33% needle wall; 1024 = 128-token chunk x top-8
/// keeps both measured regimes on their winning side).
pub(crate) const QWEN35_DEEPGEMM_MIN_ROUTES: usize = 1024;

/// Routed-row ceiling for the decode-specialized weight-read-bound grouped
/// kernels (`moe_bf16_grouped_gemm_{swiglu_,}decode`).
///
/// Formula: `256 = top_k(8) × B_max(32)` — the batched-decode envelope. The
/// decode band is exactly `R = top_k · B` (stage-1 batched decode steps;
/// B=1 single decode is R=8), where each expert receives ≤ B rows (one route
/// per token per expert), so:
/// * `B ≤ ACT_TILE(8)`: every touched weight row is read EXACTLY once
///   (nsys 2026-06-11 root cause: the batch kernels burn 487 µs/layer at R=8
///   on scalar 2B loads + zero M-reuse ≈ 3% of HBM bandwidth; ideal bytes
///   `E_active × (gate[512,2048] + up[512,2048] + down[2048,512]) × 2B =
///   E_active × 6.29 MB` → 50.3 MB at E_active=8 → 12.6 µs at 4 TB/s,
///   realistic 25-60 µs/layer target).
/// * `8 < B ≤ 32`: ≤ `ceil(32/8) = 4` weight passes with ≥8-way activation
///   reuse per pass — still weight-traffic-dominated.
/// * `R > 256`: only prefill remainder chunks land here (full 128-token
///   chunks go DeepGEMM at `R ≥ 1024`). The batch kernels' 32-row M-tile
///   reads weights `ceil(c/32)` times vs the decode kernels' `ceil(c/8)` —
///   up to 4× less traffic — and the regime is unmeasured, so the batch
///   kernels keep it (status quo; folding the mid-band is a follow-up pod
///   A/B, not a default flip).
///
/// Must stay `< QWEN35_DEEPGEMM_MIN_ROUTES` so the decode band never
/// shadows the licensed DeepGEMM prefill dispatch.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const QWEN35_MOE_DECODE_MAX_ROUTES: usize = 256;

/// `ARLE_QWEN35_DEEPGEMM=1`: swap the Qwen3.5/3.6 hand CUDA-core grouped
/// expert GEMMs (~3.9 TFLOP/s class) for DeepGEMM SM90 BF16 m-grouped GEMMs
/// (vendored official kernels, JIT-compiled through the torch-free native
/// bridge; requires a binary built with `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`,
/// preflight fails loud otherwise).
///
/// Default ON (licensed 2026-06-11 pod A/B, warm JIT cache, n=3 each):
/// hybrid dispatch keeps decode on the hand kernels (40.86 vs 40.46 tok/s,
/// neutral) and puts prefill on DeepGEMM tensor cores — needle 3k wall
/// 9.10 -> 2.32 s (-74.5%); smoke x3 consistent + needle PASS both arms.
/// First-ever run on a box pays the DeepGEMM JIT compile into
/// `~/.deep_gemm` (~3.7 s across needle shapes, once per cache lifetime).
/// `ARLE_QWEN35_DEEPGEMM=0` restores the hand-kernel-only path. Read at
/// LOAD time as well: the loader builds the contiguous grouped-B weight
/// caches (and drops the per-expert copies) only when enabled, so flipping
/// requires a process restart. Requires a binary built with
/// `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`; without it the preflight fails
/// loud and `=0` is the operator's escape hatch.
#[cfg(feature = "cuda")]
pub(crate) fn qwen35_deepgemm_enabled() -> bool {
    !matches!(
        std::env::var("ARLE_QWEN35_DEEPGEMM").as_deref(),
        Ok("0" | "false" | "FALSE" | "no" | "off" | "OFF")
    )
}

/// `ARLE_QWEN35_MOE_DECODE_KERNEL=0`: run the original hand batch grouped
/// kernels at every routed-row count below the DeepGEMM floor — the
/// same-binary A/B arm for the pod license of the decode-band
/// weight-read-bound kernels. Default ON. Read per call (per layer per
/// step), mirroring [`qwen35_deepgemm_enabled`]; inside a captured decode
/// graph the value read at capture time is what replays.
#[cfg(feature = "cuda")]
pub(crate) fn qwen35_moe_decode_kernel_enabled() -> bool {
    !matches!(
        std::env::var("ARLE_QWEN35_MOE_DECODE_KERNEL").as_deref(),
        Ok("0" | "false" | "FALSE" | "no" | "off" | "OFF")
    )
}

/// `ARLE_QWEN35_MOE_FUSED_SGLANG=1`: route the Qwen3.6 MoE FFN through the
/// vendored SGLang `fused_moe` triton lane (U3) — moe_align → fused_moe_kernel
/// GEMM1 (gate||up) → act_and_mul (silu·up) → fused_moe_kernel GEMM2 (down,
/// MUL_ROUTED_WEIGHT) → moe_sum_reduce — instead of the in-tree pack +
/// grouped-GEMM + scatter/combine path. Default OFF: flag-off is byte-for-byte
/// the current hand decode-band path (this helper is read once, gating the
/// whole fused branch). The triton cubins are Hopper-only AOT; a
/// non-sm90 / no-triton build links NOT_SUPPORTED stubs and the runtime
/// loud-fails when this flag is on and a stub is hit. The lazily-built stacked
/// `w1/w2` cache (`MoeLayerWeights::fused_sglang`, +~1.5 GB) materializes only
/// on the first fused forward, so load-time is unaffected. bf16-only for now
/// (fp8 is a 1-shape AOT signature add). Read per call, mirroring the sibling
/// MoE gates; inside a captured decode graph the value read at capture replays.
#[cfg(feature = "cuda")]
pub(crate) fn qwen35_moe_fused_sglang_enabled() -> bool {
    matches!(
        std::env::var("ARLE_QWEN35_MOE_FUSED_SGLANG").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
    )
}

/// Allocate an i32 buffer pre-filled with `-1` ON DEVICE (memset 0xFF).
///
/// A `clone_htod(&vec![-1; n])` here would record a CUDA-graph memcpy node
/// whose HOST source dies with this call — replay reads freed memory (the
/// same hazard class as the topk_length IMA, 2026-06-10). Device memset is
/// capture-safe and cheaper. Shared by the BF16 (`gpu`) and DSv4 FP8
/// (`dsv4_gpu`) forwards — both use `-1` as the "route slot not packed on this
/// rank" sentinel.
#[cfg(feature = "cuda")]
fn alloc_neg1_i32(
    ctx: &cuda_kernels::prelude::DeviceContext,
    len: usize,
) -> anyhow::Result<cudarc::driver::CudaSlice<i32>> {
    use cudarc::driver::DevicePtrMut;

    let mut buf = ctx
        .stream
        .alloc_zeros::<i32>(len)
        .map_err(|e| anyhow::anyhow!("MoE -1 sentinel alloc failed: {e}"))?;
    let n_bytes = len * std::mem::size_of::<i32>();
    let (ptr, _guard) = buf.device_ptr_mut(&ctx.stream);
    // SAFETY: `ptr` is a live device allocation of `n_bytes` on this stream.
    unsafe {
        cudarc::driver::result::memset_d8_async(ptr, 0xFF, n_bytes, ctx.stream.cu_stream())
            .map_err(|e| anyhow::anyhow!("MoE -1 sentinel memset failed: {e}"))?;
    }
    drop(_guard);
    Ok(buf)
}

#[cfg(feature = "cuda")]
mod gpu {
    use anyhow::{Result, ensure};
    use cuda_kernels::moe;
    use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, HiddenStates, RawDevicePtr};
    use cuda_kernels::tensor::cache_ptr;
    use half::bf16;
    use infer_moe::{MoeConfig, ScoringFunc, TopkMethod};

    use super::{
        DEEPGEMM_CONTIG_ALIGN, QWEN35_DEEPGEMM_MIN_ROUTES, QWEN35_MOE_DECODE_MAX_ROUTES,
        deepgemm_contig_rows_cap, qwen35_deepgemm_enabled, qwen35_moe_decode_kernel_enabled,
        qwen35_moe_fused_sglang_enabled,
    };
    use crate::loader::MoeLayerWeights;
    use crate::moe_config::ExpertSplit;
    use crate::ops::{gemm_batch, silu_mul};
    use crate::workspace::{HiddenSlot, SliceSlot};

    /// DeepGEMM masked per-group band capacity (rows). Must be a multiple of
    /// 128: the masked GEMM's TMA store writes full BLOCK_M (64/128) output
    /// tiles, so a smaller band would let a tile cross into the next group's
    /// band. The masked layout is host-shape-fixed (`[G, 128, K]` regardless
    /// of routing), which also makes it the CUDA-graph-safe variant.
    const DEEPGEMM_MASKED_BAND: usize = 128;

    /// Persistent device scratch for [`moe_forward_into`] — one slot per
    /// per-call buffer the old path allocated fresh (~10 device allocations per
    /// MoE layer per step). Exact-shape reuse: decode (`R = topk`) hits the
    /// cache every step; prefill re-allocates only when the chunk shape
    /// changes. (audit MOE-P3-1)
    ///
    /// Write-before-read proof per slot (kernels verified in
    /// `csrc/moe/dsv4_route.cu`, `csrc/gemm/moe_grouped_gemm.cu`,
    /// `csrc/gemm/gemv.cu`, `csrc/misc/elementwise_basic.cu`):
    ///
    /// | slot              | shape       | init on reuse | proof |
    /// |-------------------|-------------|---------------|-------|
    /// | `logits`          | `[E, T]`    | none          | router `gemm_cuda` is beta=0, writes all `E*T` |
    /// | `route_indices`   | `[R]` i32   | none / H2D    | device route: `dsv4_route` phase 3 writes all `topk` slots per token unconditionally; host fallback: H2D overwrite every call |
    /// | `route_weights`   | `[R]` f32   | none / H2D    | same as `route_indices` (renorm kernel is in-place over freshly written slots) |
    /// | `router_bias_zero`| `[E]` bf16  | `upload_const`| pure function of length (all-zero greedy selection bias), uploaded once |
    /// | `counts`          | `[E_l]` i32 | memset 0      | REQUIRED: count kernel `atomicAdd`s |
    /// | `offsets`         | `[E_l]` i32 | none          | exclusive scan writes all `tid < n` |
    /// | `scan_total`      | `[1]` i32   | none          | scan writes it before any read |
    /// | `packed_hidden`   | `[H, R]`    | none          | single-GPU: pack writes every slot row (offsets+cursors partition `[0,R)`); EP: tail rows unwritten but only read by the grouped GEMMs WITHIN `offsets/counts` |
    /// | `packed_route_slot`| `[R]` i32  | memset -1 (EP only) | single-GPU: pack writes all `R` slots; EP: sentinel REQUIRED (scatter skips `< 0`; stale/zero tail would alias slot 0) |
    /// | `packed_weight`   | `[R]` f32   | none          | scatter reads `packed_weight[r]` only where `packed_route_slot[r] >= 0`, which pack wrote |
    /// | `cursors`         | `[E_l]` i32 | memset 0      | REQUIRED: pack kernel `atomicAdd`s slot cursors |
    /// | `expert_indices`  | `[E_l]` i32 | none          | identity table, pure function of length (`upload_const`) |
    /// | `gate_out`/`up_out`| `[I, R]`   | none          | batch path only (`R > QWEN35_MOE_DECODE_MAX_ROUTES` or decode kernels opted out): grouped GEMM writes rows within `counts`; `silu_mul` READS the (EP) tail but its outputs there land in the `act` tail, which nothing reads (down-GEMM reads `act` within `counts` only) — final bytes unchanged. Decode-kernel path never allocates/touches these slots |
    /// | `act`             | `[I, R]`    | none          | batch path: `silu_mul` writes all `I*R`. Decode path: the fused swiglu kernel writes rows within `offsets/counts` only; the (EP) tail keeps stale rows, read by nothing (the down decode GEMM also reads strictly within `offsets/counts`) |
    /// | `expert_out`      | `[H, R]`    | none          | down grouped GEMM (batch or decode) writes within `counts`; scatter reads only packed slots |
    /// | `route_out`       | `[H, R]`    | memset 0 (EP only) | single-GPU: scatter writes ALL `R` slots (route↔slot bijection) then combine reads all; EP: zero-init LOAD-BEARING (combine sums non-local slots as 0 into the partial) |
    /// | `shared_gate`/`shared_up`| `[S_i, T]` | none    | dense `gemm_cuda` beta=0 |
    /// | `shared_act`      | `[S_i, T]`  | none          | `silu_mul` writes all |
    /// | `shared_out`      | `[H, T]`    | none          | dense `gemm_cuda` beta=0 |
    /// | `gate_logit`      | `[1, T]`    | none          | dense `gemm_cuda` beta=0 |
    ///
    /// The caller-provided `out` (`[H, T]`) needs no init: the combine kernel
    /// writes every element before `qwen36_add_shared_expert_gated` RMWs it.
    ///
    /// DeepGEMM path (`ARLE_QWEN35_DEEPGEMM=1`) — extra slots + reused slots
    /// at PADDED shapes (`rows_p` = `G·128` masked band / `contig_rows_cap`):
    ///
    /// | slot                 | shape        | init on reuse | proof |
    /// |----------------------|--------------|---------------|-------|
    /// | `dg_band_offsets`    | `[E_l]` i32  | `upload_const`| pure function of length (`g * 128` band bases) |
    /// | `dg_aligned_offsets` | `[E_l]` i32  | none          | aligned scan writes all `tid < n` |
    /// | `dg_m_indices`       | `[rows_p]`   | memset -1     | REQUIRED: fill kernel writes only `count_g` rows per group; every pad row must read -1 (contiguous-GEMM group resolution + garbage marker) |
    /// | `packed_route_slot`  | `[rows_p]`   | memset -1     | REQUIRED even single-GPU (unlike the hand path): pack writes only the `R` real rows; pad rows must read -1 so the scatter skips them |
    /// | `packed_hidden`      | `[H, rows_p]`| none          | pack writes the `R` real rows; PAD rows: zero from the slot's `alloc_zeros`, or stale rows from a previous call — both safe by induction: a pad A-row only feeds the matching pad D-row (GEMM rows are independent), every pad D-row has `packed_route_slot = -1` so the scatter never reads it, and every stale value traces back to a finite prior GEMM output or the zero init (never uninitialized memory) |
    /// | `packed_weight`      | `[rows_p]`   | none          | scatter reads it only where `packed_route_slot >= 0`, which pack wrote |
    /// | `gate_out`/`up_out`  | `[I, rows_p]`| none          | masked/contiguous GEMM writes rows within its tile coverage; rows outside coverage are read only by `silu_mul`, whose outputs land in the matching `act` pad rows (down-GEMM tile coverage is identical, scatter skips pads) |
    /// | `act`                | `[I, rows_p]`| none          | `silu_mul` writes all elements |
    /// | `expert_out`         | `[H, rows_p]`| none          | GEMM writes within tile coverage; scatter reads only `packed_route_slot >= 0` rows, all within coverage |
    ///
    /// SGLang fused lane (`ARLE_QWEN35_MOE_FUSED_SGLANG=1`) — extra slots, never
    /// touched on the default path (`numel = T·topk`, `2N`/`N`/`K` from cfg):
    ///
    /// | slot                  | shape            | init on reuse | proof |
    /// |-----------------------|------------------|---------------|-------|
    /// | `fm_sorted_token_ids` | `[max_padded]` i32 | none        | the native align kernel int4-fills the whole buffer with the `numel` pad sentinel (block 1) then scatters every real route id (block 0) — every slot written each call |
    /// | `fm_expert_ids`       | `[num_blocks]` i32 | none        | the align kernel writes the per-block expert id for every block |
    /// | `fm_total_padded`     | `[1]` i32        | memset 0      | REQUIRED: align kernel `atomicAdd`s the cumsum/total |
    /// | `fm_cumsum`           | `[E + 1]` i32    | memset 0      | REQUIRED: align kernel `atomicAdd`s per-expert counts then scans in place |
    /// | `fm_cache1`           | `[numel, 2N]` bf16 | none        | GEMM1 writes every valid route row (`offs_token < numel`); pad-tile rows are masked stores, never read by act (act runs over `numel` rows) |
    /// | `fm_cache2`           | `[numel, N]` bf16  | none        | act_and_mul writes all `numel` rows (silu·up); GEMM2 reads exactly these |
    /// | `fm_cache3`           | `[numel, K]` bf16  | none        | GEMM2 writes every valid route row; sum-reduce reads `[T, topk, K]` = all `numel` rows |
    #[derive(Default)]
    pub(crate) struct MoeForwardScratch {
        logits: HiddenSlot,
        route_indices: SliceSlot<i32>,
        route_weights: SliceSlot<f32>,
        router_bias_zero: SliceSlot<bf16>,
        counts: SliceSlot<i32>,
        offsets: SliceSlot<i32>,
        scan_total: SliceSlot<i32>,
        packed_hidden: HiddenSlot,
        packed_route_slot: SliceSlot<i32>,
        packed_weight: SliceSlot<f32>,
        cursors: SliceSlot<i32>,
        expert_indices: SliceSlot<i32>,
        dg_band_offsets: SliceSlot<i32>,
        dg_aligned_offsets: SliceSlot<i32>,
        dg_m_indices: SliceSlot<i32>,
        gate_out: HiddenSlot,
        up_out: HiddenSlot,
        act: HiddenSlot,
        expert_out: HiddenSlot,
        route_out: HiddenSlot,
        shared_gate: HiddenSlot,
        shared_up: HiddenSlot,
        shared_act: HiddenSlot,
        shared_out: HiddenSlot,
        gate_logit: HiddenSlot,
        // ── SGLang fused_moe lane (U3, `ARLE_QWEN35_MOE_FUSED_SGLANG`). All
        // unused (never allocated) on the default hand path. ────────────────
        /// `[max_num_tokens_padded]` i32 — moe_align sorted route ids.
        fm_sorted_token_ids: SliceSlot<i32>,
        /// `[num_blocks]` i32 — moe_align per-block expert id. Sized to the
        /// max block count `cdiv(max_num_tokens_padded, BLOCK_SIZE_M)`.
        fm_expert_ids: SliceSlot<i32>,
        /// `[1]` i32 — moe_align total padded tokens (zeroed every call).
        fm_total_padded: SliceSlot<i32>,
        /// `[num_experts + 1]` i32 — moe_align cumsum scratch (zeroed).
        fm_cumsum: SliceSlot<i32>,
        /// cache1 `[num_tokens*topk, 2*moe_inter]` bf16 (gate||up GEMM1 out).
        fm_cache1: SliceSlot<bf16>,
        /// cache2 `[num_tokens*topk, moe_inter]` bf16 (silu·up act out).
        fm_cache2: SliceSlot<bf16>,
        /// cache3 `[num_tokens*topk, hidden]` bf16 (down GEMM2 out, summed).
        fm_cache3: SliceSlot<bf16>,
    }

    impl MoeForwardScratch {
        pub(crate) fn new() -> Self {
            Self::default()
        }

        /// Drop every cached buffer (frees the VRAM; OPD weight time-share).
        pub(crate) fn release(&mut self) {
            let Self {
                logits,
                route_indices,
                route_weights,
                router_bias_zero,
                counts,
                offsets,
                scan_total,
                packed_hidden,
                packed_route_slot,
                packed_weight,
                cursors,
                expert_indices,
                dg_band_offsets,
                dg_aligned_offsets,
                dg_m_indices,
                gate_out,
                up_out,
                act,
                expert_out,
                route_out,
                shared_gate,
                shared_up,
                shared_act,
                shared_out,
                gate_logit,
                fm_sorted_token_ids,
                fm_expert_ids,
                fm_total_padded,
                fm_cumsum,
                fm_cache1,
                fm_cache2,
                fm_cache3,
            } = self;
            logits.release();
            route_indices.release();
            route_weights.release();
            router_bias_zero.release();
            counts.release();
            offsets.release();
            scan_total.release();
            packed_hidden.release();
            packed_route_slot.release();
            packed_weight.release();
            cursors.release();
            expert_indices.release();
            dg_band_offsets.release();
            dg_aligned_offsets.release();
            dg_m_indices.release();
            gate_out.release();
            up_out.release();
            act.release();
            expert_out.release();
            route_out.release();
            shared_gate.release();
            shared_up.release();
            shared_act.release();
            shared_out.release();
            gate_logit.release();
            fm_sorted_token_ids.release();
            fm_expert_ids.release();
            fm_total_padded.release();
            fm_cumsum.release();
            fm_cache1.release();
            fm_cache2.release();
            fm_cache3.release();
        }
    }

    /// Default ON: route fully on-device (no per-layer logits D2H +
    /// `ctx.sync()` full-stream drain). Opt out with
    /// `ARLE_QWEN35_GPU_ROUTER=0` (or `false`) to run the verified
    /// `infer_moe::route` host reference — the pod A/B license lever. Mirrors
    /// the DSv4 `ARLE_DSV4_GPU_ROUTER` gate (`dsv4_gpu::use_gpu_router`).
    fn use_gpu_router() -> bool {
        !matches!(
            std::env::var("ARLE_QWEN35_GPU_ROUTER").as_deref(),
            Ok("0" | "false")
        )
    }

    /// Whether this routing config can run on the DEVICE route kernel (greedy
    /// top-k, no group-limited masking). Shared by the per-layer dispatch in
    /// [`moe_forward_into`] and the decode-graph gate below.
    fn device_route_eligible(cfg: &MoeConfig) -> bool {
        cfg.topk_method == TopkMethod::Greedy && cfg.n_group.is_none() && cfg.topk_group.is_none()
    }

    /// Decode-graph gate: TRUE iff a `seq_len == 1` MoE step through
    /// [`moe_forward_into`] is a pure device-kernel sequence (CUDA-graph
    /// capturable) for `cfg` — i.e. it takes the device router (no per-layer
    /// `ctx.sync` + logits D2H + indices/weights H2D of the host fallback) AND
    /// the decode routed-row count stays under the DeepGEMM hybrid-dispatch
    /// floor so the shape-constant hand grouped kernels run (DeepGEMM JIT is
    /// not capture-safe and never fires at decode: `R = top_k <
    /// QWEN35_DEEPGEMM_MIN_ROUTES`).
    pub(crate) fn qwen35_decode_moe_graph_capturable(cfg: &MoeConfig) -> bool {
        use_gpu_router() && device_route_eligible(cfg) && cfg.top_k < QWEN35_DEEPGEMM_MIN_ROUTES
    }

    /// BF16 MoE forward for one sparse layer (allocate-per-call compatibility
    /// wrapper): builds a transient [`MoeForwardScratch`] + output buffer and
    /// delegates to [`moe_forward_into`]. Same allocation behavior as the
    /// pre-scratch path; the Qwen3.5/3.6 hot loop passes a persistent scratch
    /// through [`moe_forward_into`] instead.
    pub(crate) fn moe_forward(
        ctx: &DeviceContext,
        weights: &MoeLayerWeights,
        normed: &HiddenStates,
        cfg: &MoeConfig,
        split: &ExpertSplit,
    ) -> Result<HiddenStates> {
        let mut scratch = MoeForwardScratch::new();
        let mut out = HiddenStates::zeros(ctx, normed.hidden_dim, normed.seq_len)?;
        moe_forward_into(ctx, weights, normed, cfg, split, &mut scratch, &mut out)?;
        Ok(out)
    }

    /// BF16 MoE forward for one sparse layer. `normed` is the post-LN hidden
    /// `[num_tokens, hidden]`; writes the block output (routed + sigmoid-gated
    /// shared expert) into `out` (`[hidden, num_tokens]`, fully overwritten).
    ///
    /// `split` is this rank's EP expert ownership. Routing runs over ALL
    /// `cfg.num_experts` (router gate replicated; activations identical across
    /// ranks because every row-parallel output was all-reduced upstream), but
    /// only routes landing on `split`'s local experts contribute — so under
    /// `ep_size > 1` the output buffer is a PARTIAL sum (routed locals +
    /// column/row-sharded shared expert) and the caller must `all_reduce_sum`
    /// it once before the residual add. `ExpertSplit::single` is the unchanged
    /// single-GPU path (every expert local, nothing to reduce).
    pub(crate) fn moe_forward_into(
        ctx: &DeviceContext,
        weights: &MoeLayerWeights,
        normed: &HiddenStates,
        cfg: &MoeConfig,
        split: &ExpertSplit,
        scratch: &mut MoeForwardScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let num_tokens = normed.seq_len;
        let hidden_dim = normed.hidden_dim;
        let num_experts = cfg.num_experts;
        let topk = cfg.top_k;
        ensure!(
            split.num_experts == num_experts,
            "MoE expert split covers {} experts but the router has {num_experts}",
            split.num_experts
        );
        let local_experts = split.experts_per_rank;
        ensure!(
            weights.router_gate.cols == hidden_dim && weights.router_gate.rows == num_experts,
            "MoE router shape mismatch: gate={}x{} hidden_dim={} num_experts={}",
            weights.router_gate.rows,
            weights.router_gate.cols,
            hidden_dim,
            num_experts
        );
        // DeepGEMM expert path: grouped-B caches present (built at load when
        // `ARLE_QWEN35_DEEPGEMM=1`) and the env gate still set. HYBRID
        // dispatch by routed-row count (pod A/B 2026-06-11, same binary,
        // n=3): at decode R=8 the masked grouped path loses to the hand
        // CUDA-core kernels (37.5 vs 40.8 tok/s, JIT/TMA fixed overhead on
        // tiny bands), while at prefill R=16384 the contiguous tensor-core
        // path wins big (needle 3k wall 9.07 -> 6.10 s). The crossover
        // between the measured endpoints is uncharacterized; 1024 routed
        // rows (= 128-token chunk x top-8) keeps every measured regime on
        // its winning side and only remainder chunks land in between.
        let use_deepgemm = qwen35_deepgemm_enabled()
            && weights.gate_grouped.is_some()
            && num_tokens * topk >= QWEN35_DEEPGEMM_MIN_ROUTES;
        if !use_deepgemm {
            // Grouped-mode loads cleared the per-expert Vecs (the hand
            // kernels run through the rebuilt ptr tables into the grouped
            // buffer), so accept either weight form here.
            ensure!(
                weights.gate_grouped.is_some()
                    || (weights.gate.len() == local_experts
                        && weights.up.len() == local_experts
                        && weights.down.len() == local_experts),
                "MoE expert count mismatch: gate={} up={} down={} local_experts={local_experts} \
                 (ep_size={} ep_rank={})",
                weights.gate.len(),
                weights.up.len(),
                weights.down.len(),
                split.ep_size,
                split.ep_rank
            );
        }
        ensure!(
            out.hidden_dim == hidden_dim && out.seq_len == num_tokens,
            "MoE output shape mismatch: out={}x{} expected {hidden_dim}x{num_tokens}",
            out.hidden_dim,
            out.seq_len
        );

        // ── 1. Router gemm → logits[T, E] (token-major). ────────────────────
        let logits = scratch.logits.get(ctx, num_experts, num_tokens)?;
        gemm_batch(ctx, &weights.router_gate, normed, logits)?;

        // ── 2. Route: DEVICE kernel (default) or host reference fallback. ───
        let total_routes = num_tokens * topk;
        // The device kernel implements greedy top-k (zero selection bias) with
        // no group-limited masking; any other routing config falls back to the
        // host reference rather than silently mis-routing.
        let (route_indices, route_weights) = if use_gpu_router() && device_route_eligible(cfg) {
            // `dsv4_route` with routing_kind=1 (learned-bias selection key) and
            // an all-zero bias IS greedy top-k: key = scores + 0.0 exactly
            // (softmax scores are >= +0.0, so `x + 0.0f == x` bitwise).
            // scoring_kind=0 (softmax) pins the weight denom at 1.0 — raw
            // softmax probs — and the Qwen3.6 `norm_topk_prob` renorm is the
            // separate `qwen36_renorm_topk_weights` launch, mirroring
            // `infer_moe::route`'s separate step 5 (same 1e-20 zero guard,
            // same serial sum order). Audit lane MOE-OK-3 (2026-06-11)
            // established host/device semantic identity for Qwen3.6: stable
            // softmax over all experts, greedy top-k ties-to-lower-index,
            // renorm to sum 1, scaling 1.0. The bias table is a pure function
            // of its length (all zeros), so `upload_const` uploads once.
            let bias_zero_host = vec![bf16::ZERO; num_experts];
            let bias_zero = scratch
                .router_bias_zero
                .upload_const(ctx, &bias_zero_host)?;
            let bias_zero_ptr = cache_ptr(bias_zero, ctx);
            let route_indices = scratch.route_indices.get(ctx, total_routes)?;
            let route_weights = scratch.route_weights.get(ctx, total_routes)?;
            // SAFETY: logits `[E, T]`, indices/weights `[T*topk]`, bias `[E]`
            // are live scratch buffers on ctx.stream. Phase 3 of the route
            // kernel writes EVERY indices/weights slot unconditionally, so
            // length-matched slot reuse needs no re-init; the renorm kernel
            // rewrites the same freshly written slots in place.
            unsafe {
                moe::dsv4_route(
                    cache_ptr(&logits.data, ctx),
                    Some(bias_zero_ptr),
                    None,
                    None,
                    cache_ptr(route_indices, ctx),
                    cache_ptr(route_weights, ctx),
                    num_tokens,
                    num_experts,
                    topk,
                    1, // learned-bias selection key; zero bias ⇒ greedy
                    cfg.scoring_func.scoring_kind(),
                    cfg.routed_scaling_factor,
                    ctx.stream.cu_stream(),
                )?;
                // Host-route gate equivalent: norm_topk_prob ∧ Greedy ∧
                // softmax (non-softmax scoring already normalizes inside the
                // route kernel; Greedy is guaranteed by the eligibility gate).
                if cfg.norm_topk_prob && cfg.scoring_func == ScoringFunc::Softmax {
                    moe::qwen36_renorm_topk_weights(
                        cache_ptr(route_weights, ctx),
                        num_tokens,
                        topk,
                        ctx.stream.cu_stream(),
                    )?;
                }
            }
            (&*route_indices, &*route_weights)
        } else {
            // HOST route (verified infer_moe reference,
            // `ARLE_QWEN35_GPU_ROUTER=0` or a non-greedy/grouped config).
            // Sync so the gemm has landed before the D2H read.
            ctx.sync()?;
            let logits_bf16: Vec<bf16> = ctx
                .stream
                .clone_dtoh(&logits.data)
                .map_err(|e| anyhow::anyhow!("MoE router logits D2H failed: {e}"))?;
            let logits_host: Vec<f32> = logits_bf16.iter().map(|&v| v.to_f32()).collect();
            let decisions = infer_moe::route(&logits_host, &[], cfg)
                .map_err(|e| anyhow::anyhow!("MoE host route failed: {e}"))?;
            let (indices_host, weights_host) = super::flatten_routing(&decisions, topk)?;
            ensure!(
                indices_host.len() == total_routes && weights_host.len() == total_routes,
                "flattened routing length mismatch"
            );

            let route_indices = scratch.route_indices.upload(ctx, &indices_host)?;
            let route_weights = scratch.route_weights.upload(ctx, &weights_host)?;
            (route_indices, route_weights)
        };

        // Raw-ptr conversions end the `route_indices`/`route_weights` scratch
        // field borrows so the downstream tails (fused / DeepGEMM) can take
        // `&mut scratch`; the buffers themselves persist in the scratch.
        let route_indices_ptr = cache_ptr(route_indices, ctx);
        let route_weights_ptr = cache_ptr(route_weights, ctx);

        // ── 2b. SGLang fused_moe lane (opt-in, `ARLE_QWEN35_MOE_FUSED_SGLANG`).
        // Single-GPU only (global == local expert ids); flag-off this branch is
        // never entered, so the default path below is byte-for-byte unchanged.
        if qwen35_moe_fused_sglang_enabled() && device_route_eligible(cfg) {
            ensure!(
                split.ep_size == 1,
                "ARLE_QWEN35_MOE_FUSED_SGLANG requires ep_size == 1 (single-GPU); \
                 got ep_size={} ep_rank={}. The fused lane consumes GLOBAL expert ids \
                 directly — EP local remap is unsupported. Unset the flag for EP runs.",
                split.ep_size,
                split.ep_rank
            );
            // `routed_scaling_factor` is BAKED into the sum-reduce cubin (1.0);
            // a different value would silently mis-scale. Loud-fail instead.
            ensure!(
                (cfg.routed_scaling_factor - 1.0).abs() < f32::EPSILON,
                "ARLE_QWEN35_MOE_FUSED_SGLANG: routed_scaling_factor={} but the \
                 fused sum-reduce cubin bakes 1.0. Rebuild the cubin or unset the flag.",
                cfg.routed_scaling_factor
            );
            moe_forward_fused_sglang(
                ctx,
                weights,
                normed,
                cfg,
                scratch,
                route_indices_ptr,
                route_weights_ptr,
                num_tokens,
                hidden_dim,
                num_experts,
                topk,
                out,
            )?;
            return add_shared_expert_gated(ctx, weights, normed, scratch, out);
        }

        // ── 3. Per-expert route counts (LOCAL experts only). ────────────────
        // Kernels write these through raw device pointers, so no Rust `mut`.
        // counts MUST be zeroed every call (atomicAdd accumulator).
        let counts_ptr = cache_ptr(scratch.counts.get_zeroed(ctx, local_experts)?, ctx);
        // SAFETY: all buffers are valid on ctx.stream for the given shapes.
        unsafe {
            moe::dsv4_count_local_experts(
                route_indices_ptr,
                counts_ptr,
                num_tokens,
                topk,
                split.local_expert_start,
                local_experts,
                ctx.stream.cu_stream(),
            )?;
        }

        if use_deepgemm {
            // ── 4-8 (DeepGEMM): aligned/banded pack → SM90 BF16 m-grouped
            // GEMMs → scatter/combine. Fills `out` exactly like step 8.
            deepgemm_routed_tail(
                ctx,
                weights,
                normed,
                split,
                scratch,
                route_indices_ptr,
                route_weights_ptr,
                counts_ptr,
                out,
                topk,
            )?;
            return add_shared_expert_gated(ctx, weights, normed, scratch, out);
        }

        // ── 3b. Group offsets for the hand path (compact exclusive scan).
        // offsets and scan_total are fully written by the scan before any read.
        let offsets = scratch.offsets.get(ctx, local_experts)?;
        let scan_total = scratch.scan_total.get(ctx, 1)?;
        // SAFETY: counts/offsets/scan_total valid on ctx.stream.
        unsafe {
            moe::dsv4_exclusive_scan_i32(
                counts_ptr,
                cache_ptr(offsets, ctx),
                cache_ptr(scan_total, ctx),
                local_experts,
                ctx.stream.cu_stream(),
            )?;
        }

        // ── 4. Pack routed tokens grouped-by-expert (with route slots). ─────
        // Single-GPU packs every one of the `total_routes` slots exactly once
        // (offsets + cursors partition `[0, R)`), so the reused buffers need no
        // re-init. Under EP only routes hitting LOCAL experts get packed, so
        // the tail of `packed_route_slot` past this rank's route count would
        // keep a STALE slot id on reuse. The scatter kernel skips
        // `route_slot < 0` — a stale/zero tail would alias a live route slot
        // and race its legitimate write — so EP re-fills the -1 sentinel every
        // call (same memset the old per-call `alloc_neg1_i32` did).
        // cursors MUST be zeroed every call (atomicAdd slot allocator).
        let packed_hidden = scratch.packed_hidden.get(ctx, hidden_dim, total_routes)?;
        let packed_route_slot = if split.ep_size == 1 {
            scratch.packed_route_slot.get(ctx, total_routes.max(1))?
        } else {
            scratch
                .packed_route_slot
                .neg1_filled(ctx, total_routes.max(1))?
        };
        let packed_weight = scratch.packed_weight.get(ctx, total_routes.max(1))?;
        let cursors = scratch.cursors.get_zeroed(ctx, local_experts)?;
        // SAFETY: buffers valid on ctx.stream; shapes checked by the kernel.
        unsafe {
            moe::dsv4_pack_local_experts_with_slots(
                cache_ptr(&normed.data, ctx),
                route_indices_ptr,
                route_weights_ptr,
                cache_ptr(offsets, ctx),
                cache_ptr(cursors, ctx),
                cache_ptr(&packed_hidden.data, ctx),
                cache_ptr(packed_route_slot, ctx),
                cache_ptr(packed_weight, ctx),
                num_tokens,
                hidden_dim,
                topk,
                split.local_expert_start,
                local_experts,
                ctx.stream.cu_stream(),
            )?;
        }

        // Identity compact→local remap: the weight-pointer tables hold ONLY this
        // rank's experts (loader walks `split`'s range), so group g indexes table
        // entry g (explicit table keeps the RawDevicePtr contract — no null hack).
        // The table is a pure function of `local_experts`, so a length-matched
        // reuse skips the per-call H2D copy entirely.
        let expert_index_table: Vec<i32> = (0..local_experts as i32).collect();
        let expert_indices = scratch
            .expert_indices
            .upload_const(ctx, &expert_index_table)?;

        // ── 5-7 shape resolution. Grouped-mode loads carry shapes on the
        // group (per-expert Vecs are cleared); concat already enforced
        // uniformity + the same [n, k] slab layout the ptr tables expose.
        let moe_inter = match (&weights.gate_grouped, weights.gate.first()) {
            (Some(g), _) => g.rows,
            (None, Some(first)) => {
                let mi = first.rows;
                expert_shape_ok(first, &weights.up[0], hidden_dim, mi)?;
                mi
            }
            (None, None) => {
                anyhow::bail!("MoE weights carry neither grouped nor per-expert experts")
            }
        };
        // `max_count` only sizes grid Y; total_routes is a safe upper bound.
        let max_count = total_routes.max(1);
        // Down dims up front (shared by both expert-GEMM paths). Grouped-mode
        // loads cleared the per-expert Vec; the group carries the (uniform,
        // concat-ensured) dims instead.
        let (down_rows, down_cols) = match (&weights.down_grouped, weights.down.first()) {
            (Some(g), _) => (g.rows, g.cols),
            (None, Some(first)) => (first.rows, first.cols),
            (None, None) => anyhow::bail!("MoE weights carry neither grouped nor per-expert down"),
        };
        ensure!(
            down_cols == moe_inter && down_rows == hidden_dim,
            "MoE expert down shape {}x{} != hidden_dim {} / moe_inter {}",
            down_rows,
            down_cols,
            hidden_dim,
            moe_inter
        );

        // Decode-band dispatch: at `R <= QWEN35_MOE_DECODE_MAX_ROUTES` the
        // weight-read-bound decode kernels replace pair-GEMM + silu_mul +
        // down-GEMM (3 launches -> 2; one warp per weight row, 16B-vector
        // loads, weights of touched experts read once per <=8-row activation
        // chunk — see the formula on the const). Both contraction dims must
        // be 16B-vector rows (gate/up k = H, down k = I); otherwise the batch
        // kernels keep the shape. `ARLE_QWEN35_MOE_DECODE_KERNEL=0` is the
        // same-binary A/B arm. Shape-constant pure kernel launches — decode
        // CUDA-graph capture-safe, same as the batch kernels.
        let use_decode_kernels = total_routes <= QWEN35_MOE_DECODE_MAX_ROUTES
            && hidden_dim.is_multiple_of(8)
            && moe_inter.is_multiple_of(8)
            && qwen35_moe_decode_kernel_enabled();

        // ── 5+6. Gate+up GEMM + SwiGLU (UNCLAMPED — Qwen3.6 has no clamp). ──
        // act rows are written within `offsets/counts` (decode path) or fully
        // (batch path's silu_mul); under EP the act tail past the local route
        // count is stale either way, and nothing reads it (both down GEMMs
        // consume `act` strictly within offsets/counts).
        let act = scratch.act.get(ctx, moe_inter, total_routes)?;
        if use_decode_kernels {
            // Fused: act = silu(gate·x) * (up·x), no gate_out/up_out slots.
            // SAFETY: weight-ptr tables + packed buffers valid on ctx.stream;
            // k = hidden_dim % 8 == 0 checked by the dispatch above.
            unsafe {
                moe::moe_bf16_grouped_gemm_swiglu_decode(
                    &weights.gate_ptrs,
                    &weights.up_ptrs,
                    cache_ptr(&packed_hidden.data, ctx),
                    cache_ptr(&act.data, ctx),
                    cache_ptr(offsets, ctx),
                    counts_ptr,
                    cache_ptr(expert_indices, ctx),
                    local_experts,
                    max_count,
                    moe_inter,
                    hidden_dim,
                    ctx,
                    ctx.stream.cu_stream(),
                )?;
            }
        } else {
            let gate_out = scratch.gate_out.get(ctx, moe_inter, total_routes)?;
            let up_out = scratch.up_out.get(ctx, moe_inter, total_routes)?;
            // SAFETY: weight-ptr tables + packed buffers valid on ctx.stream.
            unsafe {
                moe::moe_bf16_grouped_gemm_pair_batch(
                    &weights.gate_ptrs,
                    &weights.up_ptrs,
                    cache_ptr(&packed_hidden.data, ctx),
                    cache_ptr(&gate_out.data, ctx),
                    cache_ptr(&up_out.data, ctx),
                    cache_ptr(offsets, ctx),
                    counts_ptr,
                    cache_ptr(expert_indices, ctx),
                    local_experts,
                    max_count,
                    moe_inter,
                    hidden_dim,
                    ctx,
                    ctx.stream.cu_stream(),
                )?;
            }
            // silu_mul covers all `moe_inter * total_routes` elements; under
            // EP the gate/up tails past the local route count are stale (not
            // zero), but their products land only in the matching `act` tail,
            // which nothing reads.
            silu_mul(ctx, gate_out, up_out, act)?;
        }

        // ── 7. Grouped down GEMM → expert_out[R, H]. ────────────────────────
        let expert_out = scratch.expert_out.get(ctx, hidden_dim, total_routes)?;
        if use_decode_kernels {
            // SAFETY: down weight table + act/expert_out valid on ctx.stream;
            // k = moe_inter % 8 == 0 checked by the dispatch above.
            unsafe {
                moe::moe_bf16_grouped_gemm_decode(
                    &weights.down_ptrs,
                    cache_ptr(&act.data, ctx),
                    cache_ptr(&expert_out.data, ctx),
                    cache_ptr(offsets, ctx),
                    counts_ptr,
                    cache_ptr(expert_indices, ctx),
                    local_experts,
                    max_count,
                    hidden_dim,
                    moe_inter,
                    ctx,
                    ctx.stream.cu_stream(),
                )?;
            }
        } else {
            // SAFETY: down weight table + act/expert_out valid on ctx.stream.
            unsafe {
                moe::moe_bf16_grouped_gemm_batch(
                    &weights.down_ptrs,
                    cache_ptr(&act.data, ctx),
                    cache_ptr(&expert_out.data, ctx),
                    cache_ptr(offsets, ctx),
                    counts_ptr,
                    cache_ptr(expert_indices, ctx),
                    local_experts,
                    max_count,
                    hidden_dim,
                    moe_inter,
                    ctx,
                    ctx.stream.cu_stream(),
                )?;
            }
        }

        // ── 8. Scatter weighted expert outputs to route slots, combine topk. ─
        // route_out[slot] = weight · expert_out[slot]; combine sums over topk.
        // Single-GPU the scatter writes ALL `total_routes` slots (route↔slot
        // bijection), so reuse needs no re-init; under EP the unwritten
        // non-local slots MUST read zero in the combine (the partial-sum
        // contract), so EP re-zeros every call — exactly the zero-init the old
        // per-call alloc provided.
        let route_out = if split.ep_size == 1 {
            scratch.route_out.get(ctx, hidden_dim, total_routes)?
        } else {
            scratch
                .route_out
                .get_zeroed(ctx, hidden_dim, total_routes)?
        };
        // SAFETY: all buffers valid on ctx.stream for the given shapes.
        unsafe {
            moe::dsv4_scatter_all_route_slots(
                cache_ptr(&expert_out.data, ctx),
                cache_ptr(&route_out.data, ctx),
                cache_ptr(packed_route_slot, ctx),
                cache_ptr(packed_weight, ctx),
                total_routes,
                hidden_dim,
                ctx.stream.cu_stream(),
            )?;
            moe::dsv4_combine_route_slot_outputs(
                cache_ptr(&route_out.data, ctx),
                cache_ptr(&out.data, ctx),
                num_tokens,
                topk,
                hidden_dim,
                ctx.stream.cu_stream(),
            )?;
        }

        // ── 9. Shared expert: dense SwiGLU · sigmoid(x @ shared_gate_router).
        add_shared_expert_gated(ctx, weights, normed, scratch, out)
    }

    /// Step 9 of the MoE block, shared by the hand and DeepGEMM expert paths:
    /// dense shared-expert SwiGLU, sigmoid-gated and accumulated into the
    /// routed output (`out` is RMW'd; the combine fully wrote it in step 8).
    fn add_shared_expert_gated(
        ctx: &DeviceContext,
        weights: &MoeLayerWeights,
        normed: &HiddenStates,
        scratch: &mut MoeForwardScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let num_tokens = normed.seq_len;
        let hidden_dim = normed.hidden_dim;
        let shared = shared_expert_forward(
            ctx,
            weights,
            normed,
            &mut scratch.shared_gate,
            &mut scratch.shared_up,
            &mut scratch.shared_act,
            &mut scratch.shared_out,
        )?;
        let gate_logit = scratch.gate_logit.get(ctx, 1, num_tokens)?;
        gemm_batch(ctx, &weights.shared_gate_router, normed, gate_logit)?;
        // SAFETY: out/shared/gate_logit valid on ctx.stream.
        unsafe {
            moe::qwen36_add_shared_expert_gated(
                cache_ptr(&out.data, ctx),
                cache_ptr(&shared.data, ctx),
                cache_ptr(&gate_logit.data, ctx),
                num_tokens,
                hidden_dim,
                ctx.stream.cu_stream(),
            )?;
        }

        Ok(())
    }

    /// Loud-fail message when the fused triton AOT lane resolves to a
    /// NOT_SUPPORTED stub (non-sm90 / no-triton build) but the flag is on.
    fn fused_moe_stub_err(kernel: &str) -> anyhow::Error {
        anyhow::anyhow!(
            "ARLE_QWEN35_MOE_FUSED_SGLANG is on but the fused_moe triton AOT lane is not built \
             ({kernel} returned CUDA_ERROR_NOT_SUPPORTED). Rebuild with an sm_90 target and \
             INFER_TRITON_PYTHON set to a triton-capable interpreter, or unset \
             ARLE_QWEN35_MOE_FUSED_SGLANG to use the hand kernels. \
             See crates/cuda-kernels/tools/triton/README.md."
        )
    }

    /// Lazily build (once per layer) the SGLang stacked expert-weight cache
    /// `w1 [E_l, 2N, K]` (gate rows then up rows per expert) + `w2 [E_l, K, N]`
    /// (down verbatim) from the per-expert `DeviceMatrix` Vecs. Pure D2D
    /// gather-concat (~+1.5 GB on the Qwen3.6 single-GPU shard); returns a
    /// borrow of the populated `OnceCell`. Requires the per-expert Vecs to be
    /// present — DeepGEMM grouped-mode loads clear them, so the fused lane is
    /// incompatible with `ARLE_QWEN35_DEEPGEMM=1` and loud-fails here.
    fn build_fused_sglang_weights<'w>(
        ctx: &DeviceContext,
        weights: &'w MoeLayerWeights,
        hidden_dim: usize,
    ) -> Result<&'w crate::loader::MoeFusedSglangWeights> {
        use cudarc::driver::CudaSlice;

        // Already built this layer? (Single-threaded `!Sync` executor, so this
        // check-then-`set` cannot race — `get_or_try_init` is still unstable.)
        if let Some(cached) = weights.fused_sglang.get() {
            return Ok(cached);
        }

        let built = (|| -> Result<crate::loader::MoeFusedSglangWeights> {
            let num_experts = weights.gate.len();
            ensure!(
                num_experts > 0
                    && weights.up.len() == num_experts
                    && weights.down.len() == num_experts,
                "ARLE_QWEN35_MOE_FUSED_SGLANG needs the per-expert weight Vecs \
                 (gate={} up={} down={}); they are cleared under ARLE_QWEN35_DEEPGEMM=1. \
                 Unset DeepGEMM to use the fused lane.",
                weights.gate.len(),
                weights.up.len(),
                weights.down.len()
            );

            // Per-expert dims (gate/up `[N, K]`, down `[K, N]`); uniform across
            // experts (the loader concat path already enforces this).
            let moe_inter = weights.gate[0].rows; // N
            let k = weights.gate[0].cols; // K (= hidden_dim)
            ensure!(
                k == hidden_dim,
                "fused MoE gate K {k} != hidden_dim {hidden_dim}"
            );
            let gate_up_rows = 2 * moe_inter; // 2N

            let w1_elems = num_experts * gate_up_rows * k;
            let w2_elems = num_experts * k * moe_inter;
            let mut w1: CudaSlice<bf16> = ctx
                .stream
                .alloc_zeros(w1_elems)
                .map_err(|e| anyhow::anyhow!("fused MoE w1 alloc failed: {e}"))?;
            let mut w2: CudaSlice<bf16> = ctx
                .stream
                .alloc_zeros(w2_elems)
                .map_err(|e| anyhow::anyhow!("fused MoE w2 alloc failed: {e}"))?;

            let expert_nk = moe_inter * k; // gate / up slab elements
            let down_kn = k * moe_inter; // down slab elements
            for e in 0..num_experts {
                let gate = &weights.gate[e];
                let up = &weights.up[e];
                let down = &weights.down[e];
                ensure!(
                    gate.rows == moe_inter
                        && gate.cols == k
                        && up.rows == moe_inter
                        && up.cols == k,
                    "fused MoE expert {e} gate/up shape ({}x{} / {}x{}) != {moe_inter}x{k}",
                    gate.rows,
                    gate.cols,
                    up.rows,
                    up.cols
                );
                ensure!(
                    down.rows == k && down.cols == moe_inter,
                    "fused MoE expert {e} down shape {}x{} != {k}x{moe_inter}",
                    down.rows,
                    down.cols
                );
                // w1[e]: rows [0, N) = gate `[N, K]`, rows [N, 2N) = up `[N, K]`.
                let w1_base = e * gate_up_rows * k;
                ctx.stream
                    .memcpy_dtod(&gate.data, &mut w1.slice_mut(w1_base..w1_base + expert_nk))
                    .map_err(|err| anyhow::anyhow!("fused MoE w1 gate D2D failed: {err}"))?;
                ctx.stream
                    .memcpy_dtod(
                        &up.data,
                        &mut w1.slice_mut(w1_base + expert_nk..w1_base + 2 * expert_nk),
                    )
                    .map_err(|err| anyhow::anyhow!("fused MoE w1 up D2D failed: {err}"))?;
                // w2[e]: down `[K, N]` verbatim.
                let w2_base = e * down_kn;
                ctx.stream
                    .memcpy_dtod(&down.data, &mut w2.slice_mut(w2_base..w2_base + down_kn))
                    .map_err(|err| anyhow::anyhow!("fused MoE w2 down D2D failed: {err}"))?;
            }

            Ok(crate::loader::MoeFusedSglangWeights {
                w1,
                w2,
                num_experts,
                gate_up_rows,
                hidden: k,
                moe_inter,
            })
        })()?;

        // `set` only fails if another thread won the race — impossible on the
        // `!Sync` executor (we already returned early on a populated cell).
        weights
            .fused_sglang
            .set(built)
            .map_err(|_| anyhow::anyhow!("fused MoE weight cache double-init (unreachable)"))?;
        weights
            .fused_sglang
            .get()
            .ok_or_else(|| anyhow::anyhow!("fused MoE weight cache empty after set"))
    }

    /// SGLang `fused_moe` routed-expert path (opt-in lane, single-GPU only). The
    /// `route_*_ptr` are the token-major `[num_tokens*topk]` GLOBAL expert ids /
    /// renormed weights produced by step 2 (== SGLang's `topk_ids`/
    /// `topk_weights`). Writes the routed sum into `out` (token-major `[T, K]`,
    /// fully overwritten by the final sum-reduce); the caller adds the shared
    /// expert.
    ///
    /// Data flow (all caches keyed by flattened route id = `token*topk + k`):
    ///   moe_align → GEMM1 (`a`=normed `[T,K]`, w1 `[E,2N,K]` → cache1
    ///   `[T*topk, 2N]`) → act_and_mul (silu(gate)·up → cache2 `[T*topk, N]`) →
    ///   GEMM2 (cache2, w2 `[E,K,N]`, MUL_ROUTED_WEIGHT → cache3 `[T*topk, K]`)
    ///   → sum_reduce (cache3 viewed `[T, topk, K]` → out `[T, K]`).
    ///
    /// `normed` is token-major (token-contiguous) `[T, K]`: a token's K features
    /// are contiguous, so GEMM1 `a` uses `stride_am=hidden_dim, stride_ak=1`.
    /// `out.data` is the same token-major `[T, K]` (it is added to the
    /// token-major residual downstream), so sum-reduce uses
    /// `output_stride_0=hidden_dim, output_stride_1=1`. (Confirmed against the
    /// validated hand pack kernel `dsv4_pack_local_experts_kernel`, which reads
    /// `normed` at `token*hidden_dim + col`.)
    #[allow(clippy::too_many_arguments)]
    fn moe_forward_fused_sglang(
        ctx: &DeviceContext,
        weights: &MoeLayerWeights,
        normed: &HiddenStates,
        cfg: &MoeConfig,
        scratch: &mut MoeForwardScratch,
        route_indices_ptr: RawDevicePtr<i32>,
        route_weights_ptr: RawDevicePtr<f32>,
        num_tokens: usize,
        hidden_dim: usize,
        num_experts: usize,
        topk: usize,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let _ = cfg; // routed_scaling_factor asserted == 1.0 by the caller (baked in cubin).
        let fused = build_fused_sglang_weights(ctx, weights, hidden_dim)?;
        let moe_inter = fused.moe_inter; // N
        let gate_up_rows = fused.gate_up_rows; // 2N
        debug_assert_eq!(fused.hidden, hidden_dim);
        debug_assert_eq!(fused.num_experts, num_experts);

        let numel = num_tokens * topk; // flattened topk_ids length
        let block_size = moe::FUSED_MOE_BLOCK_SIZE_M;
        let max_padded = moe::fused_moe_max_num_tokens_padded(numel, num_experts, block_size);
        let num_blocks = max_padded.div_ceil(block_size);

        // ── moe_align scratch. sorted_token_ids must be the numel sentinel for
        // pad rows so GEMM tiles skip them; the native kernel int4-fills the
        // whole buffer with `numel` (block 1) then scatters the real ids, so a
        // plain `get` is enough — the kernel writes every slot. total_padded +
        // cumsum are atomicAdd accumulators → zeroed every call. expert_ids is
        // fully written by the kernel (per-block expert id). ─────────────────
        let sorted_token_ids = scratch.fm_sorted_token_ids.get(ctx, max_padded)?;
        let sorted_token_ids_ptr = cache_ptr(sorted_token_ids, ctx);
        let expert_ids = scratch.fm_expert_ids.get(ctx, num_blocks)?;
        let expert_ids_ptr = cache_ptr(expert_ids, ctx);
        let total_padded = scratch.fm_total_padded.get_zeroed(ctx, 1)?;
        let total_padded_ptr = cache_ptr(total_padded, ctx);
        let cumsum = scratch.fm_cumsum.get_zeroed(ctx, num_experts + 1)?;
        let cumsum_ptr = cache_ptr(cumsum, ctx);

        // SAFETY: all buffers valid on ctx.stream for the documented shapes;
        // route_indices_ptr is the token-major `[numel]` global expert id buffer.
        unsafe {
            moe::moe_align_block_size(
                route_indices_ptr,
                sorted_token_ids_ptr,
                expert_ids_ptr,
                total_padded_ptr,
                cumsum_ptr,
                num_experts,
                block_size,
                numel,
                max_padded,
                ctx.stream.cu_stream(),
            )
            .map_err(|_| fused_moe_stub_err("arle_moe_align_block_size"))?;
        }

        // ── cache slots. cache1/2/3 are fully written by their producer kernel
        // within the valid route rows; pad rows past `numel` are never read by
        // the next stage (act over numel rows; sum-reduce over `numel` route
        // ids = `T*topk`). Sized exactly `[numel, *]`. ───────────────────────
        let cache1 = scratch.fm_cache1.get(ctx, numel * gate_up_rows)?;
        let cache1_ptr = cache_ptr(cache1, ctx);
        let cache2 = scratch.fm_cache2.get(ctx, numel * moe_inter)?;
        let cache2_ptr = cache_ptr(cache2, ctx);
        let cache3 = scratch.fm_cache3.get(ctx, numel * hidden_dim)?;
        let cache3_ptr = cache_ptr(cache3, ctx);

        let w1_ptr = cache_ptr(&fused.w1, ctx);
        let w2_ptr = cache_ptr(&fused.w2, ctx);

        // ── GEMM1: a = normed `[T, K]`, b = w1 `[E, 2N, K]`, c = cache1
        // `[T*topk, 2N]`. normed is token-major (token-contiguous) `[T, K]`:
        // stride_am=hidden_dim (next token = +K), stride_ak=1 (next feature =
        // +1). w1 row-major `[E, 2N, K]`: stride_be=2N*K, stride_bk=1,
        // stride_bn=K. cache1 row-major: stride_cm=2N, stride_cn=1. ───────────
        let n1 = gate_up_rows; // 2N
        let k1 = hidden_dim; // K
        // SAFETY: a/b/c + align outputs valid on ctx.stream; route_weights
        // unused in GEMM1 (MUL_ROUTED_WEIGHT=0 baked) but passed for the shared
        // kernel signature.
        unsafe {
            moe::fused_moe_gemm1(
                cache_ptr(&normed.data, ctx),
                w1_ptr,
                cache1_ptr,
                route_weights_ptr,
                sorted_token_ids_ptr,
                expert_ids_ptr,
                total_padded_ptr,
                n1,
                k1,
                max_padded,
                numel,
                k1,      // stride_am = hidden_dim (token-major: next token = +K)
                1,       // stride_ak = 1 (token-contiguous: next feature = +1)
                n1 * k1, // stride_be = 2N*K
                1,       // stride_bk
                k1,      // stride_bn = K
                n1,      // stride_cm = 2N
                1,       // stride_cn
                ctx.stream.cu_stream(),
            )
            .map_err(|_| fused_moe_stub_err("arle_fused_moe_gemm1"))?;
        }

        // ── act_and_mul: cache1 `[numel, 2N]` → cache2 `[numel, N]`,
        // cache2 = silu(gate) * up. hidden_size = 2N (kernel halves it).
        // expert_ids row buffer = route_indices (pad-row skip; never fires on
        // the dense bf16 path). Grid = (numel,). ─────────────────────────────
        // SAFETY: cache1/cache2/route_indices valid on ctx.stream.
        unsafe {
            moe::fused_moe_act_and_mul(
                cache1_ptr,
                cache2_ptr,
                route_indices_ptr,
                numel,
                gate_up_rows,
                ctx.stream.cu_stream(),
            )
            .map_err(|_| fused_moe_stub_err("arle_fused_moe_act"))?;
        }

        // ── GEMM2: a = cache2 `[numel, N]`, b = w2 `[E, K, N]`, c = cache3
        // `[numel, K]`. top_k=1 baked → a_ptrs uses offs_token//1 = route id.
        // MUL_ROUTED_WEIGHT=1 baked → applies route_weights here. cache2
        // row-major: stride_am=N, stride_ak=1. w2 row-major `[E, K, N]`:
        // stride_be=K*N, stride_bk=1, stride_bn=N. cache3 row-major:
        // stride_cm=K, stride_cn=1. ──────────────────────────────────────────
        let n2 = hidden_dim; // K
        let k2 = moe_inter; // N
        // SAFETY: a/b/c + align outputs + route_weights valid on ctx.stream.
        unsafe {
            moe::fused_moe_gemm2(
                cache2_ptr,
                w2_ptr,
                cache3_ptr,
                route_weights_ptr,
                sorted_token_ids_ptr,
                expert_ids_ptr,
                total_padded_ptr,
                n2,
                k2,
                max_padded,
                numel,
                k2,      // stride_am = N
                1,       // stride_ak
                n2 * k2, // stride_be = K*N
                1,       // stride_bk
                k2,      // stride_bn = N
                n2,      // stride_cm = K
                1,       // stride_cn
                ctx.stream.cu_stream(),
            )
            .map_err(|_| fused_moe_stub_err("arle_fused_moe_gemm2"))?;
        }

        // ── sum_reduce: cache3 viewed `[T, topk, K]` → out `[T, K]`, summed
        // over topk × routed_scaling_factor (1.0 baked). cache3 row-major
        // `[numel, K]` = `[T, topk, K]` contiguous: input_stride_0=topk*K,
        // input_stride_1=K, input_stride_2=1. out is token-major `[T, K]`
        // (token-contiguous): output_stride_0=hidden_dim, output_stride_1=1. ──
        // SAFETY: cache3/out valid on ctx.stream for the documented shapes.
        unsafe {
            moe::fused_moe_sum_reduce(
                cache3_ptr,
                topk * hidden_dim, // input_stride_0
                hidden_dim,        // input_stride_1
                1,                 // input_stride_2
                cache_ptr(&out.data, ctx),
                hidden_dim, // output_stride_0 = hidden_dim (token-major: next token = +K)
                1,          // output_stride_1 = 1 (token-contiguous: next feature = +1)
                num_tokens,
                topk,
                hidden_dim,
                ctx.stream.cu_stream(),
            )
            .map_err(|_| fused_moe_stub_err("arle_fused_moe_sum"))?;
        }

        Ok(())
    }

    /// Steps 4-8 of the MoE block on the DeepGEMM SM90 BF16 m-grouped GEMMs
    /// (`ARLE_QWEN35_DEEPGEMM=1`): pack → gate GEMM + up GEMM → silu_mul →
    /// down GEMM → scatter/combine into `out`. `counts` was already filled by
    /// step 3's count kernel.
    ///
    /// Per-call dispatch (masked vs contiguous):
    ///
    /// * **Masked** (`R = total_routes <= 128`, i.e. decode): A/D are fixed
    ///   per-group bands `[G, 128, *]` and `masked_m = counts` bounds each
    ///   group on-device. The band capacity is a HOST constant while the
    ///   per-group counts are device-resident, so the only host-provable
    ///   capacity bound is `max_g count_g <= Σ_g count_g = R <= 128` — which
    ///   is exactly the dispatch threshold. Wasted compute is
    ///   `Σ_g (align(count_g, BLOCK_M ∈ {64,128}) − count_g) <= 64·G_active`
    ///   rows per GEMM; shapes are routing-independent (CUDA-graph-safe).
    /// * **Contiguous** (`R > 128`, i.e. prefill): compact-ish `[m_cap, K]` A
    ///   with 128-aligned per-group segments and `m_indices` row→group ids
    ///   (-1 pads). Masked CANNOT serve this regime — a single hot expert can
    ///   receive up to R rows, so the band capacity would have to be R per
    ///   group (`G·R` rows of scratch). Contiguous pays
    ///   `m_cap − Σ_g count_g <= align(R,128) + 128·min(R,G) − R` padding
    ///   rows of garbage compute (pad tiles resolve to group 0 and are
    ///   excluded from the scatter via the route-slot -1 sentinel).
    ///
    /// Gate and up run as TWO grouped GEMMs over the same A (fusing them into
    /// one `[G, 2I, K]` grouped B — the checkpoint's native stacked layout —
    /// is a follow-up: it halves A traffic and launch count but needs the
    /// loader to keep the fused `gate_up` tensor un-split plus a strided
    /// silu_mul over `[gate | up]` halves; deferred to keep this diff at the
    /// licensed two-GEMM numerics of the hand path).
    #[allow(clippy::too_many_arguments)]
    fn deepgemm_routed_tail(
        ctx: &DeviceContext,
        weights: &MoeLayerWeights,
        normed: &HiddenStates,
        split: &ExpertSplit,
        scratch: &mut MoeForwardScratch,
        route_indices: RawDevicePtr<i32>,
        route_weights: RawDevicePtr<f32>,
        counts: RawDevicePtr<i32>,
        out: &mut HiddenStates,
        topk: usize,
    ) -> Result<()> {
        let num_tokens = normed.seq_len;
        let hidden_dim = normed.hidden_dim;
        let total_routes = num_tokens * topk;
        let local_experts = split.experts_per_rank;
        let stream = ctx.stream.cu_stream();

        let (gate_g, up_g, down_g) = match (
            &weights.gate_grouped,
            &weights.up_grouped,
            &weights.down_grouped,
        ) {
            (Some(g), Some(u), Some(d)) => (g, u, d),
            _ => anyhow::bail!(
                "DeepGEMM MoE path requires the grouped expert caches (load with ARLE_QWEN35_DEEPGEMM=1)"
            ),
        };
        let moe_inter = gate_g.rows;
        ensure!(
            gate_g.groups == local_experts
                && up_g.groups == local_experts
                && down_g.groups == local_experts,
            "DeepGEMM MoE group count mismatch: gate={} up={} down={} local_experts={local_experts}",
            gate_g.groups,
            up_g.groups,
            down_g.groups
        );
        ensure!(
            gate_g.cols == hidden_dim
                && up_g.rows == moe_inter
                && up_g.cols == hidden_dim
                && down_g.rows == hidden_dim
                && down_g.cols == moe_inter,
            "DeepGEMM MoE grouped cache shape mismatch: gate={}x{} up={}x{} down={}x{} H={hidden_dim} I={moe_inter}",
            gate_g.rows,
            gate_g.cols,
            up_g.rows,
            up_g.cols,
            down_g.rows,
            down_g.cols
        );
        // BF16 kernel constraints: K % 64 (BLOCK_K) and N % 8, in both GEMM
        // directions (gate/up: n=I k=H; down: n=H k=I) → both dims % 64.
        ensure!(
            hidden_dim.is_multiple_of(64) && moe_inter.is_multiple_of(64),
            "DeepGEMM BF16 MoE needs H and I aligned to 64, got H={hidden_dim} I={moe_inter}"
        );
        // Fail loud if the native DeepGEMM bridge is a build-time stub.
        moe::dsv4_deepgemm_native_preflight()?;

        let use_masked = total_routes <= DEEPGEMM_MASKED_BAND;
        let rows = if use_masked {
            local_experts * DEEPGEMM_MASKED_BAND
        } else {
            deepgemm_contig_rows_cap(total_routes, local_experts)
        };
        ensure!(
            rows.checked_mul(hidden_dim.max(moe_inter))
                .is_some_and(|v| i32::try_from(v).is_ok()),
            "DeepGEMM MoE padded rows {rows} x max(H={hidden_dim}, I={moe_inter}) exceeds the i32 kernel ABI"
        );

        // ── 4 (DG). Pack routed tokens + pad-row sentinels. ─────────────────
        // packed_route_slot MUST be -1-refilled every call even single-GPU:
        // the pack writes only the R real rows, and every PAD row has to read
        // -1 so the scatter skips it (pad GEMM outputs are garbage — masked
        // rows beyond count_g, contiguous tiles resolved against group 0).
        // packed_hidden pad rows are zero-from-alloc or stale-from-prior-call;
        // safe by the slot-table induction (GEMM rows are independent, pads
        // never reach `out`).
        let offsets_ptr = if use_masked {
            // Fixed band bases `g * 128` — a pure function of the length.
            let band_table: Vec<i32> = (0..local_experts as i32)
                .map(|g| g * DEEPGEMM_MASKED_BAND as i32)
                .collect();
            cache_ptr(scratch.dg_band_offsets.upload_const(ctx, &band_table)?, ctx)
        } else {
            // 128-aligned per-group segment starts (device-side scan of the
            // device-resident counts; the total is not read back — `rows` is
            // the host cap).
            let aligned_offsets =
                cache_ptr(scratch.dg_aligned_offsets.get(ctx, local_experts)?, ctx);
            let scan_total = cache_ptr(scratch.scan_total.get(ctx, 1)?, ctx);
            // SAFETY: counts/offsets/total valid on ctx.stream for E_l groups.
            unsafe {
                moe::moe_exclusive_scan_aligned_i32(
                    counts,
                    aligned_offsets,
                    scan_total,
                    local_experts,
                    DEEPGEMM_CONTIG_ALIGN,
                    stream,
                )?;
            }
            aligned_offsets
        };
        let cursors = cache_ptr(scratch.cursors.get_zeroed(ctx, local_experts)?, ctx);
        let packed_route_slot = cache_ptr(scratch.packed_route_slot.neg1_filled(ctx, rows)?, ctx);
        let packed_weight = cache_ptr(scratch.packed_weight.get(ctx, rows)?, ctx);
        let packed_hidden = cache_ptr(&scratch.packed_hidden.get(ctx, hidden_dim, rows)?.data, ctx);
        // SAFETY: buffers valid on ctx.stream; the pack writes every route's
        // row at offsets[local] + cursor < rows (masked: count_g <= R <= 128
        // = band capacity by the dispatch threshold; contiguous: aligned
        // offsets + counts <= the rows cap by construction).
        unsafe {
            moe::dsv4_pack_local_experts_with_slots(
                cache_ptr(&normed.data, ctx),
                route_indices,
                route_weights,
                offsets_ptr,
                cursors,
                packed_hidden,
                packed_route_slot,
                packed_weight,
                num_tokens,
                hidden_dim,
                topk,
                split.local_expert_start,
                local_experts,
                stream,
            )?;
        }

        // Contiguous only: row → local-expert map (-1 pads, refilled every
        // call for the same sentinel reason as packed_route_slot).
        let m_indices = if use_masked {
            None
        } else {
            let m_indices = cache_ptr(scratch.dg_m_indices.neg1_filled(ctx, rows)?, ctx);
            // SAFETY: counts/offsets are per-group; m_indices has `rows` slots.
            unsafe {
                moe::dsv4_fill_m_indices_from_counts(
                    counts,
                    offsets_ptr,
                    m_indices,
                    local_experts,
                    rows,
                    stream,
                )?;
            }
            Some(m_indices)
        };

        // ── 5+6+7 (DG). gate GEMM + up GEMM → silu_mul → down GEMM. ─────────
        // Heuristics-only hint: expected valid rows per group.
        let expected_m = total_routes.div_ceil(local_experts).max(1);
        let gate_out = scratch.gate_out.get(ctx, moe_inter, rows)?;
        let up_out = scratch.up_out.get(ctx, moe_inter, rows)?;
        // SAFETY: A `[rows, H]`, B `[G, I, H]`, D `[rows, I]` row-major BF16
        // on ctx.stream; masked_m = counts (read-only); m_indices contract
        // holds by the aligned scan (group segments start 128-aligned).
        unsafe {
            for (grouped_b, d) in [(gate_g, &*gate_out), (up_g, &*up_out)] {
                if use_masked {
                    moe::deepgemm_m_grouped_bf16_gemm_nt_masked(
                        packed_hidden,
                        cache_ptr(&grouped_b.data, ctx),
                        cache_ptr(&d.data, ctx),
                        counts,
                        local_experts,
                        DEEPGEMM_MASKED_BAND,
                        moe_inter,
                        hidden_dim,
                        expected_m,
                        stream,
                    )?;
                } else {
                    moe::deepgemm_m_grouped_bf16_gemm_nt_contiguous(
                        packed_hidden,
                        cache_ptr(&grouped_b.data, ctx),
                        cache_ptr(&d.data, ctx),
                        m_indices.expect("contiguous path fills m_indices"),
                        local_experts,
                        rows,
                        moe_inter,
                        hidden_dim,
                        stream,
                    )?;
                }
            }
        }
        // SwiGLU over the full padded buffers (UNCLAMPED — Qwen3.6 has no
        // clamp); pad-row products land in `act` pad rows, which the down
        // GEMM tiles cover only as further pad rows the scatter skips.
        let act = scratch.act.get(ctx, moe_inter, rows)?;
        silu_mul(ctx, gate_out, up_out, act)?;
        let expert_out = scratch.expert_out.get(ctx, hidden_dim, rows)?;
        // SAFETY: A `[rows, I]`, B `[G, H, I]`, D `[rows, H]`; same contracts
        // as the gate/up GEMMs above.
        unsafe {
            if use_masked {
                moe::deepgemm_m_grouped_bf16_gemm_nt_masked(
                    cache_ptr(&act.data, ctx),
                    cache_ptr(&down_g.data, ctx),
                    cache_ptr(&expert_out.data, ctx),
                    counts,
                    local_experts,
                    DEEPGEMM_MASKED_BAND,
                    hidden_dim,
                    moe_inter,
                    expected_m,
                    stream,
                )?;
            } else {
                moe::deepgemm_m_grouped_bf16_gemm_nt_contiguous(
                    cache_ptr(&act.data, ctx),
                    cache_ptr(&down_g.data, ctx),
                    cache_ptr(&expert_out.data, ctx),
                    m_indices.expect("contiguous path fills m_indices"),
                    local_experts,
                    rows,
                    hidden_dim,
                    moe_inter,
                    stream,
                )?;
            }
        }

        // ── 8 (DG). Scatter weighted expert rows to route slots, combine. ───
        // The scatter walks ALL `rows` padded rows and skips route_slot < 0,
        // so exactly the R real rows land in route_out. Single-GPU that is a
        // route↔slot bijection (all experts local → all R slots written);
        // under EP the unwritten non-local slots MUST read zero in the
        // combine (partial-sum contract) → zero-refill, same as the hand path.
        let route_out = if split.ep_size == 1 {
            scratch.route_out.get(ctx, hidden_dim, total_routes)?
        } else {
            scratch
                .route_out
                .get_zeroed(ctx, hidden_dim, total_routes)?
        };
        // SAFETY: all buffers valid on ctx.stream for the given shapes.
        unsafe {
            moe::dsv4_scatter_all_route_slots(
                cache_ptr(&expert_out.data, ctx),
                cache_ptr(&route_out.data, ctx),
                packed_route_slot,
                packed_weight,
                rows,
                hidden_dim,
                stream,
            )?;
            moe::dsv4_combine_route_slot_outputs(
                cache_ptr(&route_out.data, ctx),
                cache_ptr(&out.data, ctx),
                num_tokens,
                topk,
                hidden_dim,
                stream,
            )?;
        }
        Ok(())
    }

    /// Dense shared-expert SwiGLU: `down(silu(gate(x)) * up(x))` into the
    /// `out_slot` buffer (every stage fully overwrites its buffer: beta=0
    /// GEMMs + full-range silu_mul). Takes the four slots individually so the
    /// caller's other scratch fields stay borrowable while the returned
    /// reference (tied to `out_slot` only) is alive.
    fn shared_expert_forward<'a>(
        ctx: &DeviceContext,
        weights: &MoeLayerWeights,
        normed: &HiddenStates,
        gate_slot: &mut HiddenSlot,
        up_slot: &mut HiddenSlot,
        act_slot: &mut HiddenSlot,
        out_slot: &'a mut HiddenSlot,
    ) -> Result<&'a HiddenStates> {
        let shared_inter = weights.shared_gate.rows;
        let hidden_dim = normed.hidden_dim;
        let gate = gate_slot.get(ctx, shared_inter, normed.seq_len)?;
        let up = up_slot.get(ctx, shared_inter, normed.seq_len)?;
        gemm_batch(ctx, &weights.shared_gate, normed, gate)?;
        gemm_batch(ctx, &weights.shared_up, normed, up)?;
        let act = act_slot.get(ctx, shared_inter, normed.seq_len)?;
        silu_mul(ctx, gate, up, act)?;
        let out = out_slot.get(ctx, hidden_dim, normed.seq_len)?;
        gemm_batch(ctx, &weights.shared_down, act, out)?;
        Ok(out)
    }

    fn expert_shape_ok(
        gate: &DeviceMatrix,
        up: &DeviceMatrix,
        hidden_dim: usize,
        moe_inter: usize,
    ) -> Result<()> {
        ensure!(
            gate.cols == hidden_dim && up.cols == hidden_dim && up.rows == moe_inter,
            "MoE expert gate/up shape mismatch: gate={}x{} up={}x{} hidden_dim={} moe_inter={}",
            gate.rows,
            gate.cols,
            up.rows,
            up.cols,
            hidden_dim,
            moe_inter
        );
        Ok(())
    }
}

// The DSv4 FP8 MoE forward is consumed by the Piece 2 `model.rs` layer loop
// (the `RealCudaExecutor` DSv4 branch); until that integration edit lands there
// is no in-crate caller, matching the `dsv4.rs` pending-consumer gate. The
// `dead_code`/`unused_imports` allowance marks pending-consumer infra, not cruft
// (see `feedback_necessity_not_callers`).
#[cfg(feature = "cuda")]
#[allow(dead_code)]
mod dsv4_gpu {
    use anyhow::{Result, ensure};
    use cuda_kernels::moe;
    use cuda_kernels::prelude::{DeviceContext, HiddenStates};
    use cuda_kernels::tensor::{Dsv4Fp8DeepGemmWeightCache, RawDevicePtr, cache_ptr};
    use cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut};
    use half::bf16;
    use std::sync::Arc;

    use super::{DEEPGEMM_CONTIG_ALIGN, alloc_neg1_i32, deepgemm_contig_rows_cap};
    use crate::dsv4::{Dsv4ForwardKeepalive, Dsv4Model, Dsv4MoeLayer};
    use crate::moe_config::ExpertSplit;
    use crate::ops::gemm_batch;

    const CONTIG_ROUTE_ALIGN: usize = 128;

    struct DeviceRouting {
        indices: CudaSlice<i32>,
        weights: CudaSlice<f32>,
    }

    pub(crate) struct Dsv4MoeDecodeScratch {
        topk: usize,
        num_groups: usize,
        hidden_dim: usize,
        intermediate: usize,
        shared_intermediate: usize,
        route_indices: CudaSlice<i32>,
        route_weights: CudaSlice<f32>,
        token_ids: CudaSlice<u32>,
        router_logits: HiddenStates,
        counts: CudaSlice<i32>,
        offsets: CudaSlice<i32>,
        scan_total: CudaSlice<i32>,
        packed_hidden: HiddenStates,
        packed_route_slot: CudaSlice<i32>,
        packed_weight: CudaSlice<f32>,
        cursors: CudaSlice<i32>,
        route_out: HiddenStates,
        grouped: Dsv4GroupedDecodeScratch,
        grouped_contig: Dsv4GroupedContiguousDecodeScratch,
        shared: Dsv4SharedDecodeScratch,
    }

    struct Dsv4GroupedDecodeScratch {
        max_m: usize,
        scale_stride_m: usize,
        input_fp8: CudaSlice<u8>,
        input_scales: CudaSlice<f32>,
        w13_out: HiddenStates,
        act_fp8: CudaSlice<u8>,
        act_scales: CudaSlice<f32>,
        out_padded: HiddenStates,
        out_compact: HiddenStates,
        masked_m: CudaSlice<i32>,
        active_experts: CudaSlice<i32>,
    }

    struct Dsv4GroupedContiguousDecodeScratch {
        rows: usize,
        scale_stride_m: usize,
        input_fp8: CudaSlice<u8>,
        input_scales: CudaSlice<f32>,
        w13_out: HiddenStates,
        act_fp8: CudaSlice<u8>,
        act_scales: CudaSlice<f32>,
        out: HiddenStates,
        packed_hidden: HiddenStates,
        packed_route_slot: CudaSlice<i32>,
        packed_weight: CudaSlice<f32>,
        m_indices: CudaSlice<i32>,
        active_experts: CudaSlice<i32>,
        active_offsets: CudaSlice<i32>,
        active_counts: CudaSlice<i32>,
    }

    struct Dsv4SharedDecodeScratch {
        max_m: usize,
        scale_stride_m: usize,
        input_fp8: CudaSlice<u8>,
        input_scales: CudaSlice<f32>,
        w13_out: HiddenStates,
        act_fp8: CudaSlice<u8>,
        act_scales: CudaSlice<f32>,
        out: HiddenStates,
        active_experts: CudaSlice<i32>,
        active_offsets: CudaSlice<i32>,
        counts: CudaSlice<i32>,
        masked_m: CudaSlice<i32>,
    }

    impl Dsv4MoeDecodeScratch {
        pub(crate) fn new(
            ctx: &DeviceContext,
            cfg: &infer_moe::MoeConfig,
            split: &ExpertSplit,
            layer: &Dsv4MoeLayer,
        ) -> Result<Self> {
            let topk = cfg.top_k;
            let num_groups = layer.num_groups;
            let hidden_dim = layer.hidden_dim;
            let intermediate = layer.intermediate;
            let shared_intermediate = layer.shared_w2.cols;
            ensure!(
                topk > 0 && num_groups == split.experts_per_rank,
                "DSv4 decode scratch shape mismatch: topk={topk} groups={num_groups} experts_per_rank={}",
                split.experts_per_rank
            );
            ensure!(
                hidden_dim.is_multiple_of(128)
                    && intermediate.is_multiple_of(128)
                    && shared_intermediate.is_multiple_of(128),
                "DSv4 decode scratch needs 128-aligned dims: H={hidden_dim} I={intermediate} shared_I={shared_intermediate}"
            );

            let route_indices = ctx.stream.alloc_zeros::<i32>(topk).map_err(|e| {
                anyhow::anyhow!("DSv4 decode route-index scratch alloc failed: {e}")
            })?;
            let route_weights = ctx.stream.alloc_zeros::<f32>(topk).map_err(|e| {
                anyhow::anyhow!("DSv4 decode route-weight scratch alloc failed: {e}")
            })?;
            let token_ids = ctx
                .stream
                .alloc_zeros::<u32>(1)
                .map_err(|e| anyhow::anyhow!("DSv4 decode token-id scratch alloc failed: {e}"))?;
            let router_logits = unsafe { HiddenStates::uninit(ctx, cfg.num_experts, 1)? };
            let counts = ctx
                .stream
                .alloc_zeros::<i32>(num_groups)
                .map_err(|e| anyhow::anyhow!("DSv4 decode count scratch alloc failed: {e}"))?;
            let offsets = ctx
                .stream
                .alloc_zeros::<i32>(num_groups)
                .map_err(|e| anyhow::anyhow!("DSv4 decode offset scratch alloc failed: {e}"))?;
            let scan_total = ctx
                .stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| anyhow::anyhow!("DSv4 decode scan-total scratch alloc failed: {e}"))?;
            let packed_hidden = HiddenStates::zeros(ctx, hidden_dim, topk)?;
            let packed_route_slot = ctx
                .stream
                .alloc_zeros::<i32>(topk)
                .map_err(|e| anyhow::anyhow!("DSv4 decode route-slot scratch alloc failed: {e}"))?;
            let packed_weight = ctx.stream.alloc_zeros::<f32>(topk).map_err(|e| {
                anyhow::anyhow!("DSv4 decode packed-weight scratch alloc failed: {e}")
            })?;
            let cursors = ctx
                .stream
                .alloc_zeros::<i32>(num_groups)
                .map_err(|e| anyhow::anyhow!("DSv4 decode cursor scratch alloc failed: {e}"))?;
            let route_out = HiddenStates::zeros(ctx, hidden_dim, topk)?;
            let grouped =
                Dsv4GroupedDecodeScratch::new(ctx, num_groups, hidden_dim, intermediate, topk)?;
            let grouped_contig =
                Dsv4GroupedContiguousDecodeScratch::new(ctx, hidden_dim, intermediate, topk)?;
            let shared = Dsv4SharedDecodeScratch::new(ctx, hidden_dim, shared_intermediate)?;

            Ok(Self {
                topk,
                num_groups,
                hidden_dim,
                intermediate,
                shared_intermediate,
                route_indices,
                route_weights,
                token_ids,
                router_logits,
                counts,
                offsets,
                scan_total,
                packed_hidden,
                packed_route_slot,
                packed_weight,
                cursors,
                route_out,
                grouped,
                grouped_contig,
                shared,
            })
        }

        /// Device bytes of the ONE shared [`Dsv4MoeDecodeScratch`].
        /// MUST mirror [`Dsv4MoeDecodeScratch::new`]'s allocations.
        pub(crate) fn device_bytes(
            cfg: &infer_moe::MoeConfig,
            split: &ExpertSplit,
            layer: &Dsv4MoeLayer,
        ) -> usize {
            fn b(elems: usize, elem_bytes: usize) -> usize {
                elems.saturating_mul(elem_bytes)
            }
            fn bf16(hidden_dim: usize, seq_len: usize) -> usize {
                hidden_dim.saturating_mul(seq_len).saturating_mul(2)
            }
            let topk = cfg.top_k;
            let num_groups = layer.num_groups.max(split.experts_per_rank);
            let hidden_dim = layer.hidden_dim;
            let intermediate = layer.intermediate;
            let shared_intermediate = layer.shared_w2.cols;
            let hidden_scale_cols = hidden_dim.div_ceil(128);
            let inter_scale_cols = intermediate.div_ceil(128);

            let mut total = 0usize;
            total = total.saturating_add(b(topk, 4)); // route_indices
            total = total.saturating_add(b(topk, 4)); // route_weights
            total = total.saturating_add(b(1, 4)); // token_ids
            total = total.saturating_add(bf16(cfg.num_experts, 1)); // router_logits
            total = total.saturating_add(b(num_groups, 4)); // counts
            total = total.saturating_add(b(num_groups, 4)); // offsets
            total = total.saturating_add(b(1, 4)); // scan_total
            total = total.saturating_add(bf16(hidden_dim, topk)); // packed_hidden
            total = total.saturating_add(b(topk, 4)); // packed_route_slot
            total = total.saturating_add(b(topk, 4)); // packed_weight
            total = total.saturating_add(b(num_groups, 4)); // cursors
            total = total.saturating_add(bf16(hidden_dim, topk)); // route_out

            let grouped_max_m = topk.max(128);
            let grouped_scale_stride_m = grouped_max_m.div_ceil(4) * 4;
            let grouped_rows = num_groups.saturating_mul(grouped_max_m);
            total = total.saturating_add(b(grouped_rows.saturating_mul(hidden_dim), 1));
            total = total.saturating_add(b(
                num_groups
                    .saturating_mul(grouped_scale_stride_m)
                    .saturating_mul(hidden_scale_cols),
                4,
            ));
            total = total.saturating_add(bf16(2usize.saturating_mul(intermediate), grouped_rows));
            total = total.saturating_add(b(grouped_rows.saturating_mul(intermediate), 1));
            total = total.saturating_add(b(
                num_groups
                    .saturating_mul(grouped_scale_stride_m)
                    .saturating_mul(inter_scale_cols),
                4,
            ));
            total = total.saturating_add(bf16(hidden_dim, grouped_rows));
            total = total.saturating_add(bf16(hidden_dim, topk.max(1)));
            total = total.saturating_add(b(num_groups, 4)); // masked_m
            total = total.saturating_add(b(num_groups, 4)); // active_experts

            let contig_rows = topk.max(1).saturating_mul(CONTIG_ROUTE_ALIGN);
            let contig_scale_stride_m = contig_rows.div_ceil(4) * 4;
            total = total.saturating_add(b(contig_rows.saturating_mul(hidden_dim), 1));
            total = total.saturating_add(b(
                contig_scale_stride_m.saturating_mul(hidden_scale_cols),
                4,
            ));
            total = total.saturating_add(bf16(2usize.saturating_mul(intermediate), contig_rows));
            total = total.saturating_add(b(contig_rows.saturating_mul(intermediate), 1));
            total =
                total.saturating_add(b(contig_scale_stride_m.saturating_mul(inter_scale_cols), 4));
            total = total.saturating_add(bf16(hidden_dim, contig_rows)); // out
            total = total.saturating_add(bf16(hidden_dim, contig_rows)); // packed_hidden
            total = total.saturating_add(b(contig_rows, 4)); // packed_route_slot
            total = total.saturating_add(b(contig_rows, 4)); // packed_weight
            total = total.saturating_add(b(contig_rows, 4)); // m_indices
            total = total.saturating_add(b(3, 4)); // active_experts + offsets + counts

            let shared_max_m = 128usize;
            let shared_scale_stride_m = shared_max_m.div_ceil(4) * 4;
            let shared_inter_scale_cols = shared_intermediate.div_ceil(128);
            total = total.saturating_add(b(shared_max_m.saturating_mul(hidden_dim), 1));
            total = total.saturating_add(b(
                shared_scale_stride_m.saturating_mul(hidden_scale_cols),
                4,
            ));
            total = total.saturating_add(bf16(
                2usize.saturating_mul(shared_intermediate),
                shared_max_m,
            ));
            total = total.saturating_add(b(shared_max_m.saturating_mul(shared_intermediate), 1));
            total = total.saturating_add(b(
                shared_scale_stride_m.saturating_mul(shared_inter_scale_cols),
                4,
            ));
            total = total.saturating_add(bf16(hidden_dim, shared_max_m));
            total = total.saturating_add(b(4, 4)); // active_experts + offsets + counts + masked_m
            total
        }

        fn reset_routed(&mut self, ctx: &DeviceContext) -> Result<()> {
            ctx.stream
                .memset_zeros(&mut self.counts)
                .map_err(|e| anyhow::anyhow!("DSv4 decode count scratch reset failed: {e}"))?;
            ctx.stream
                .memset_zeros(&mut self.cursors)
                .map_err(|e| anyhow::anyhow!("DSv4 decode cursor scratch reset failed: {e}"))?;
            ctx.stream
                .memset_zeros(&mut self.packed_weight)
                .map_err(|e| {
                    anyhow::anyhow!("DSv4 decode packed-weight scratch reset failed: {e}")
                })?;
            ctx.stream
                .memset_zeros(&mut self.route_out.data)
                .map_err(|e| anyhow::anyhow!("DSv4 decode route-out scratch reset failed: {e}"))?;
            memset_i32_minus_one(ctx, &mut self.packed_route_slot)?;
            // grouped_contig.{packed_route_slot,m_indices} are consumed ONLY on the
            // contiguous decode path (the `use_contiguous` branch reads them); on the
            // default masked path they are never read, so skip the per-layer memset
            // there — pure waste (review #6, the measured 10.4% memset bucket).
            if use_contiguous_decode_moe() {
                memset_i32_minus_one(ctx, &mut self.grouped_contig.packed_route_slot)?;
                memset_i32_minus_one(ctx, &mut self.grouped_contig.m_indices)?;
            }
            Ok(())
        }

        fn validate_routed(
            &self,
            hidden_dim: usize,
            topk: usize,
            layer: &Dsv4MoeLayer,
        ) -> Result<()> {
            ensure!(
                self.topk == topk
                    && self.num_groups == layer.num_groups
                    && self.hidden_dim == hidden_dim
                    && self.intermediate == layer.intermediate,
                "DSv4 decode scratch mismatch: scratch topk={} groups={} H={} I={} vs topk={topk} groups={} H={hidden_dim} I={}",
                self.topk,
                self.num_groups,
                self.hidden_dim,
                self.intermediate,
                layer.num_groups,
                layer.intermediate
            );
            Ok(())
        }
    }

    impl Dsv4GroupedDecodeScratch {
        fn new(
            ctx: &DeviceContext,
            num_groups: usize,
            hidden_dim: usize,
            intermediate: usize,
            route_capacity: usize,
        ) -> Result<Self> {
            let max_m = route_capacity.max(128);
            let scale_stride_m = max_m.div_ceil(4) * 4;
            let rows = num_groups * max_m;
            let hidden_scale_cols = hidden_dim.div_ceil(128);
            let inter_scale_cols = intermediate.div_ceil(128);
            let input_fp8 = alloc_u8(ctx, rows * hidden_dim)?;
            let input_scales =
                alloc_zeros_f32(ctx, num_groups * scale_stride_m * hidden_scale_cols)?;
            let w13_out = HiddenStates::zeros(ctx, 2 * intermediate, rows)?;
            let act_fp8 = alloc_u8(ctx, rows * intermediate)?;
            let act_scales = alloc_zeros_f32(ctx, num_groups * scale_stride_m * inter_scale_cols)?;
            let out_padded = HiddenStates::zeros(ctx, hidden_dim, rows)?;
            let out_compact = HiddenStates::zeros(ctx, hidden_dim, route_capacity.max(1))?;
            let masked_m = ctx
                .stream
                .alloc_zeros::<i32>(num_groups)
                .map_err(|e| anyhow::anyhow!("DSv4 decode masked-m scratch alloc failed: {e}"))?;
            let active_experts = ctx
                .stream
                .clone_htod(&(0..num_groups as i32).collect::<Vec<i32>>())
                .map_err(|e| {
                    anyhow::anyhow!("DSv4 decode active-expert scratch H2D failed: {e}")
                })?;
            Ok(Self {
                max_m,
                scale_stride_m,
                input_fp8,
                input_scales,
                w13_out,
                act_fp8,
                act_scales,
                out_padded,
                out_compact,
                masked_m,
                active_experts,
            })
        }
    }

    impl Dsv4GroupedContiguousDecodeScratch {
        fn new(
            ctx: &DeviceContext,
            hidden_dim: usize,
            intermediate: usize,
            route_capacity: usize,
        ) -> Result<Self> {
            // DeepGEMM's contiguous grouped layout expects each expert segment
            // to be M-tile aligned, with invalid padding rows marked -1. Decode
            // has at most `topk` active routes, so give each route one aligned
            // tile instead of materialising every local expert group.
            let rows = route_capacity.max(1) * CONTIG_ROUTE_ALIGN;
            let scale_stride_m = rows.div_ceil(4) * 4;
            let hidden_scale_cols = hidden_dim.div_ceil(128);
            let inter_scale_cols = intermediate.div_ceil(128);
            let input_fp8 = alloc_u8(ctx, rows * hidden_dim)?;
            let input_scales = alloc_zeros_f32(ctx, scale_stride_m * hidden_scale_cols)?;
            let w13_out = HiddenStates::zeros(ctx, 2 * intermediate, rows)?;
            let act_fp8 = alloc_u8(ctx, rows * intermediate)?;
            let act_scales = alloc_zeros_f32(ctx, scale_stride_m * inter_scale_cols)?;
            let out = HiddenStates::zeros(ctx, hidden_dim, rows)?;
            let packed_hidden = HiddenStates::zeros(ctx, hidden_dim, rows)?;
            let packed_route_slot = ctx.stream.alloc_zeros::<i32>(rows).map_err(|e| {
                anyhow::anyhow!("DSv4 contiguous route-slot scratch alloc failed: {e}")
            })?;
            let packed_weight = ctx.stream.alloc_zeros::<f32>(rows).map_err(|e| {
                anyhow::anyhow!("DSv4 contiguous packed-weight scratch alloc failed: {e}")
            })?;
            let m_indices = ctx
                .stream
                .alloc_zeros::<i32>(rows)
                .map_err(|e| anyhow::anyhow!("DSv4 decode contiguous m-index alloc failed: {e}"))?;
            let active_experts = ctx
                .stream
                .clone_htod(&[0i32])
                .map_err(|e| anyhow::anyhow!("DSv4 contiguous active scratch H2D failed: {e}"))?;
            let active_offsets = ctx
                .stream
                .clone_htod(&[0i32])
                .map_err(|e| anyhow::anyhow!("DSv4 contiguous offset scratch H2D failed: {e}"))?;
            let active_counts = ctx
                .stream
                .clone_htod(&[i32::try_from(rows)?])
                .map_err(|e| anyhow::anyhow!("DSv4 contiguous count scratch H2D failed: {e}"))?;
            Ok(Self {
                rows,
                scale_stride_m,
                input_fp8,
                input_scales,
                w13_out,
                act_fp8,
                act_scales,
                out,
                packed_hidden,
                packed_route_slot,
                packed_weight,
                m_indices,
                active_experts,
                active_offsets,
                active_counts,
            })
        }
    }

    impl Dsv4SharedDecodeScratch {
        fn new(ctx: &DeviceContext, hidden_dim: usize, shared_inter: usize) -> Result<Self> {
            let max_m = 128usize;
            let scale_stride_m = max_m.div_ceil(4) * 4;
            let hidden_scale_cols = hidden_dim.div_ceil(128);
            let inter_scale_cols = shared_inter.div_ceil(128);
            let input_fp8 = alloc_u8(ctx, max_m * hidden_dim)?;
            let input_scales = alloc_zeros_f32(ctx, scale_stride_m * hidden_scale_cols)?;
            let w13_out = HiddenStates::zeros(ctx, 2 * shared_inter, max_m)?;
            let act_fp8 = alloc_u8(ctx, max_m * shared_inter)?;
            let act_scales = alloc_zeros_f32(ctx, scale_stride_m * inter_scale_cols)?;
            let out = HiddenStates::zeros(ctx, hidden_dim, max_m)?;
            let active_experts = ctx
                .stream
                .clone_htod(&[0i32])
                .map_err(|e| anyhow::anyhow!("DSv4 shared active scratch H2D failed: {e}"))?;
            let active_offsets = ctx
                .stream
                .clone_htod(&[0i32])
                .map_err(|e| anyhow::anyhow!("DSv4 shared offset scratch H2D failed: {e}"))?;
            let counts = ctx
                .stream
                .clone_htod(&[1i32])
                .map_err(|e| anyhow::anyhow!("DSv4 shared count scratch H2D failed: {e}"))?;
            let masked_m = ctx
                .stream
                .clone_htod(&[1i32])
                .map_err(|e| anyhow::anyhow!("DSv4 shared masked-m scratch H2D failed: {e}"))?;
            Ok(Self {
                max_m,
                scale_stride_m,
                input_fp8,
                input_scales,
                w13_out,
                act_fp8,
                act_scales,
                out,
                active_experts,
                active_offsets,
                counts,
                masked_m,
            })
        }
    }

    fn memset_i32_minus_one(ctx: &DeviceContext, slice: &mut CudaSlice<i32>) -> Result<()> {
        let bytes = slice.len() * std::mem::size_of::<i32>();
        let (ptr, _record) = slice.device_ptr_mut(&ctx.stream);
        unsafe {
            cudarc::driver::result::memset_d8_async(ptr, 0xFF, bytes, ctx.stream.cu_stream())
                .map_err(|e| anyhow::anyhow!("DSv4 decode i32 -1 memset failed: {e}"))?;
        }
        Ok(())
    }

    fn use_gpu_router() -> bool {
        // Default ON: route fully on-device (no per-layer logits D2H + `ctx.sync()`).
        // DSv4-Flash ships no group-limited routing (`MoeConfig::dsv4` hardcodes
        // `n_group/topk_group = None`, config.rs:151), so the device kernel's plain
        // bias-corrected top-k is algorithmically identical to the host
        // `route_token`. The decode-graph path (`dsv4_moe_forward_decode_graph`)
        // already calls this same kernel unconditionally and is token-verified, so
        // the eager/deepep paths are just converging onto the proven path. Opt out
        // with `ARLE_DSV4_GPU_ROUTER=0`.
        !matches!(std::env::var("ARLE_DSV4_GPU_ROUTER").as_deref(), Ok("0"))
    }

    fn use_contiguous_decode_moe() -> bool {
        matches!(
            std::env::var("ARLE_DSV4_MOE_CONTIG_DECODE").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
        )
    }

    fn dsv4_route_device(
        model: &Dsv4Model,
        layer: &Dsv4MoeLayer,
        tokens: &[u32],
        logits: &HiddenStates,
        decode_scratch: Option<&mut Dsv4MoeDecodeScratch>,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<DeviceRouting> {
        use deepseek_spec::DeepSeekV4MoeRoutingKind;

        let ctx = &model.ctx;
        let cfg = &model.moe_config;
        let num_tokens = logits.seq_len;
        let total_routes = num_tokens * cfg.top_k;
        ensure!(
            tokens.len() == num_tokens,
            "DSv4 device route token count {} != logits seq_len {num_tokens}",
            tokens.len()
        );
        ensure!(
            logits.hidden_dim == cfg.num_experts,
            "DSv4 device route logits hidden_dim {} != num_experts {}",
            logits.hidden_dim,
            cfg.num_experts
        );

        let mut decode_scratch = decode_scratch;
        let (route_indices, route_weights, token_ids) = if let Some(scratch) =
            decode_scratch.as_mut()
        {
            let scratch = &mut **scratch;
            ensure!(
                num_tokens == 1 && total_routes == scratch.topk,
                "DSv4 decode route scratch only supports one token: tokens={num_tokens} routes={total_routes} scratch_topk={}",
                scratch.topk
            );
            let token_ids = if matches!(layer.routing_kind, DeepSeekV4MoeRoutingKind::Hash) {
                ctx.stream
                    .memcpy_htod(tokens, &mut scratch.token_ids)
                    .map_err(|e| anyhow::anyhow!("DSv4 decode route token-id H2D failed: {e}"))?;
                Some(scratch.token_ids.clone())
            } else {
                None
            };
            (
                scratch.route_indices.clone(),
                scratch.route_weights.clone(),
                token_ids,
            )
        } else {
            let route_indices = ctx
                .stream
                .alloc_zeros::<i32>(total_routes)
                .map_err(|e| anyhow::anyhow!("DSv4 device route-index alloc failed: {e}"))?;
            let route_weights = ctx
                .stream
                .alloc_zeros::<f32>(total_routes)
                .map_err(|e| anyhow::anyhow!("DSv4 device route-weight alloc failed: {e}"))?;
            let token_ids = if matches!(layer.routing_kind, DeepSeekV4MoeRoutingKind::Hash) {
                let token_ids = ctx
                    .stream
                    .clone_htod(tokens)
                    .map_err(|e| anyhow::anyhow!("DSv4 device route token-id H2D failed: {e}"))?;
                keepalive.keep_route_u32(&token_ids);
                Some(token_ids)
            } else {
                None
            };
            (route_indices, route_weights, token_ids)
        };

        let routing_kind = match layer.routing_kind {
            DeepSeekV4MoeRoutingKind::Hash => 0,
            DeepSeekV4MoeRoutingKind::LearnedBias => 1,
        };
        let bias = layer
            .gate_bias
            .as_ref()
            .map(|bias| cache_ptr(&bias.data, ctx));
        let tid2eid = layer
            .hash_tid2eid_device
            .as_ref()
            .map(|table| cache_ptr(table, ctx));
        let token_ids_ptr = token_ids.as_ref().map(|ids| cache_ptr(ids, ctx));

        keepalive.keep_route_hidden(logits);
        keepalive.keep_route_i32(&route_indices);
        keepalive.keep_route_f32(&route_weights);
        // SAFETY: buffers are allocated for `[num_tokens * topk]`; optional
        // pointers are validated by the wrapper according to `routing_kind`.
        unsafe {
            moe::dsv4_route(
                cache_ptr(&logits.data, ctx),
                bias,
                tid2eid,
                token_ids_ptr,
                cache_ptr(&route_indices, ctx),
                cache_ptr(&route_weights, ctx),
                num_tokens,
                cfg.num_experts,
                cfg.top_k,
                routing_kind,
                cfg.scoring_func.scoring_kind(),
                cfg.routed_scaling_factor,
                ctx.stream.cu_stream(),
            )?;
        }
        Ok(DeviceRouting {
            indices: route_indices,
            weights: route_weights,
        })
    }

    /// Decode-graph MoE path: all mutable buffers are fixed scratch addresses,
    /// and the token id is read from a caller-staged device buffer so graph
    /// replay does not bake the capture step's host token value.
    pub(crate) fn dsv4_moe_forward_decode_graph(
        model: &Dsv4Model,
        layer: &Dsv4MoeLayer,
        token_ids: &CudaSlice<u32>,
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        scratch: &mut Dsv4MoeDecodeScratch,
    ) -> Result<()> {
        use deepseek_spec::DeepSeekV4MoeRoutingKind;

        let ctx = &model.ctx;
        let cfg = &model.moe_config;
        let topk = cfg.top_k;
        let hidden_dim = hidden.hidden_dim;
        ensure!(
            hidden.seq_len == 1 && token_ids.len() == 1,
            "DSv4 decode-graph MoE requires B=1, got hidden seq={} token_ids={}",
            hidden.seq_len,
            token_ids.len()
        );
        scratch.validate_routed(hidden_dim, topk, layer)?;
        scratch.reset_routed(ctx)?;
        moe::dsv4_deepgemm_native_preflight()?;

        gemm_batch(ctx, &layer.gate, hidden, &mut scratch.router_logits)?;

        let routing_kind = match layer.routing_kind {
            DeepSeekV4MoeRoutingKind::Hash => 0,
            DeepSeekV4MoeRoutingKind::LearnedBias => 1,
        };
        let bias = layer
            .gate_bias
            .as_ref()
            .map(|bias| cache_ptr(&bias.data, ctx));
        let tid2eid = layer
            .hash_tid2eid_device
            .as_ref()
            .map(|table| cache_ptr(table, ctx));
        let token_ids_ptr = if matches!(layer.routing_kind, DeepSeekV4MoeRoutingKind::Hash) {
            Some(cache_ptr(token_ids, ctx))
        } else {
            None
        };
        unsafe {
            moe::dsv4_route(
                cache_ptr(&scratch.router_logits.data, ctx),
                bias,
                tid2eid,
                token_ids_ptr,
                cache_ptr(&scratch.route_indices, ctx),
                cache_ptr(&scratch.route_weights, ctx),
                1,
                cfg.num_experts,
                cfg.top_k,
                routing_kind,
                cfg.scoring_func.scoring_kind(),
                cfg.routed_scaling_factor,
                ctx.stream.cu_stream(),
            )?;
        }

        // Pooled tail: EVERY buffer is a persistent scratch address. The
        // masked per-call-alloc tail was tried here and produced degenerate
        // output at +24% "speed" — its stream-ordered allocs (and the
        // clone_htod'd active_counts constants) become graph nodes whose
        // Rust-side frees land after capture, so replays write/read aliased
        // VAs and the MoE partially no-ops. Captured bodies must allocate
        // NOTHING. The pooled padded-GEMM tax is avoided by the contiguous
        // variant (ARLE_DSV4_MOE_CONTIG_DECODE=1): same persistent buffers,
        // compact 8-row GEMM shape.
        let route_indices = cache_ptr(&scratch.route_indices, ctx);
        let route_weights = cache_ptr(&scratch.route_weights, ctx);
        dsv4_moe_forward_decode_pooled(
            ctx,
            layer,
            &model.split,
            route_indices,
            route_weights,
            hidden,
            out,
            topk,
            model.split.local_expert_start,
            model.config.swiglu_limit,
            scratch,
        )
    }

    /// FP8 DeepGEMM MoE forward for one DSv4 routed-MoE layer (this EP rank's
    /// experts only). Mirrors the BF16 [`super::gpu::moe_forward`] route/pack/
    /// scatter/combine plumbing but swaps the two BF16 grouped GEMMs for the
    /// native DeepGEMM 5-call FP8 pipeline (`f8f8bf16`, 128-block scale).
    ///
    /// Routing is per-layer: bias-routed layers run the learned router gemm +
    /// `noaux_tc` correction bias through `infer_moe::route`; hash-routed layers
    /// (`layer.hash_tid2eid` present) pick experts directly from the `tid2eid`
    /// table by token id (no router gate), weighting them by the router scores.
    ///
    /// `tokens` are the input token ids (needed for hash routing); `hidden` is
    /// the post-LN `[num_tokens, hidden]`; `out` receives routed experts only.
    /// Callers all-reduce this EP-sharded output, then add the replicated shared
    /// expert exactly once per rank via [`dsv4_shared_expert_forward`].
    pub(crate) fn dsv4_moe_forward(
        model: &Dsv4Model,
        layer: &Dsv4MoeLayer,
        tokens: &[u32],
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        decode_scratch: Option<&mut Dsv4MoeDecodeScratch>,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<()> {
        let ctx = &model.ctx;
        let cfg = &model.moe_config;
        let split = &model.split;
        let swiglu_limit = model.config.swiglu_limit;

        let num_tokens = hidden.seq_len;
        let hidden_dim = hidden.hidden_dim;
        let topk = cfg.top_k;
        let experts_per_rank = split.experts_per_rank;
        let local_start = split.local_expert_start;

        ensure!(
            tokens.len() == num_tokens,
            "DSv4 MoE token count {} != hidden seq_len {num_tokens}",
            tokens.len()
        );
        ensure!(
            out.hidden_dim == hidden_dim && out.seq_len == num_tokens,
            "DSv4 MoE out shape {}x{} != hidden {}x{}",
            out.hidden_dim,
            out.seq_len,
            hidden_dim,
            num_tokens
        );
        ensure!(
            layer.gate.cols == hidden_dim && layer.gate.rows == cfg.num_experts,
            "DSv4 router shape mismatch: gate={}x{} hidden={} num_experts={}",
            layer.gate.rows,
            layer.gate.cols,
            hidden_dim,
            cfg.num_experts
        );
        ensure!(
            layer.num_groups == experts_per_rank,
            "DSv4 expert group count {} != experts_per_rank {experts_per_rank}",
            layer.num_groups
        );
        ensure!(
            layer.hidden_dim == hidden_dim,
            "DSv4 expert hidden dim {} != runtime hidden dim {hidden_dim}",
            layer.hidden_dim
        );
        if let Some(scratch) = decode_scratch.as_ref() {
            scratch.validate_routed(hidden_dim, topk, layer)?;
            ensure!(
                num_tokens == 1,
                "DSv4 decode MoE scratch is only valid for one-token decode, got {num_tokens}"
            );
        }

        // Fail loud if the native DeepGEMM bridge is a build-time stub.
        moe::dsv4_deepgemm_native_preflight()?;

        // ── 1+2. Router gemm → logits[T, E] → route (host oracle or device). ───
        let mut decode_scratch = decode_scratch;
        let (route_indices, route_weights) =
            crate::stage_profile::profile(ctx, "dsv4/stage/moe_route", || -> Result<_> {
                let _nvtx = crate::nvtx::range("dsv4/moe_route");
                // SAFETY: router gemm writes the full logits buffer.
                let mut logits = unsafe { HiddenStates::uninit(ctx, cfg.num_experts, num_tokens)? };
                gemm_batch(ctx, &layer.gate, hidden, &mut logits)?;
                if use_gpu_router() {
                    let routing = dsv4_route_device(
                        model,
                        layer,
                        tokens,
                        &logits,
                        decode_scratch.as_deref_mut(),
                        keepalive,
                    )?;
                    Ok((routing.indices, routing.weights))
                } else {
                    keepalive.keep_hidden(&logits);
                    ctx.sync()?;
                    let logits_bf16: Vec<bf16> = ctx
                        .stream
                        .clone_dtoh(&logits.data)
                        .map_err(|e| anyhow::anyhow!("DSv4 router logits D2H failed: {e}"))?;
                    let logits_host: Vec<f32> = logits_bf16.iter().map(|&v| v.to_f32()).collect();
                    let decisions =
                        dsv4_route(ctx, &model.config, cfg, layer, tokens, &logits_host)?;
                    let (indices_host, weights_host) = super::flatten_routing(&decisions, topk)?;

                    let route_indices = ctx
                        .stream
                        .clone_htod(&indices_host)
                        .map_err(|e| anyhow::anyhow!("DSv4 route-index H2D failed: {e}"))?;
                    let route_weights = ctx
                        .stream
                        .clone_htod(&weights_host)
                        .map_err(|e| anyhow::anyhow!("DSv4 route-weight H2D failed: {e}"))?;
                    keepalive.keep_i32(&route_indices);
                    keepalive.keep_f32(&route_weights);
                    Ok((route_indices, route_weights))
                }
            })?;

        if let Some(scratch) = decode_scratch.as_mut() {
            let scratch = &mut **scratch;
            let route_indices = cache_ptr(&route_indices, ctx);
            let route_weights = cache_ptr(&route_weights, ctx);
            return dsv4_moe_forward_decode_pooled(
                ctx,
                layer,
                split,
                route_indices,
                route_weights,
                hidden,
                out,
                topk,
                local_start,
                swiglu_limit,
                scratch,
            );
        }

        dsv4_moe_forward_masked_tail(
            model,
            layer,
            &route_indices,
            &route_weights,
            hidden,
            out,
            keepalive,
        )
    }

    /// Masked-path MoE tail (count → scan → pack → DeepGEMM grouped → scatter/
    /// combine), shared by the eager forward and the decode graph. Fully
    /// device-driven: routing buffers are read by kernels, intermediates are
    /// stream-ordered allocs (legal inside graph capture, freed via
    /// AUTO_FREE_ON_LAUNCH), and the -1 route-slot sentinel is a device memset.
    #[allow(clippy::too_many_arguments)]
    fn dsv4_moe_forward_masked_tail(
        model: &Dsv4Model,
        layer: &Dsv4MoeLayer,
        route_indices: &CudaSlice<i32>,
        route_weights: &CudaSlice<f32>,
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<()> {
        let ctx = &model.ctx;
        let cfg = &model.moe_config;
        let split = &model.split;
        let swiglu_limit = model.config.swiglu_limit;
        let num_tokens = hidden.seq_len;
        let hidden_dim = hidden.hidden_dim;
        let topk = cfg.top_k;
        let experts_per_rank = split.experts_per_rank;
        let local_start = split.local_expert_start;
        let total_routes = num_tokens * topk;
        // ── 3. Per-local-expert counts → group offsets (EP-aware start/range). ──
        let counts = ctx
            .stream
            .alloc_zeros::<i32>(experts_per_rank)
            .map_err(|e| anyhow::anyhow!("DSv4 count alloc failed: {e}"))?;
        let offsets = ctx
            .stream
            .alloc_zeros::<i32>(experts_per_rank)
            .map_err(|e| anyhow::anyhow!("DSv4 offset alloc failed: {e}"))?;
        let scan_total = ctx
            .stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| anyhow::anyhow!("DSv4 scan-total alloc failed: {e}"))?;
        keepalive.keep_i32(&counts);
        keepalive.keep_i32(&offsets);
        keepalive.keep_i32(&scan_total);
        // SAFETY: all buffers valid on ctx.stream for the given shapes.
        unsafe {
            moe::dsv4_count_local_experts(
                cache_ptr(route_indices, ctx),
                cache_ptr(&counts, ctx),
                num_tokens,
                topk,
                local_start,
                experts_per_rank,
                ctx.stream.cu_stream(),
            )?;
            moe::moe_exclusive_scan_aligned_i32(
                cache_ptr(&counts, ctx),
                cache_ptr(&offsets, ctx),
                cache_ptr(&scan_total, ctx),
                experts_per_rank,
                DEEPGEMM_CONTIG_ALIGN,
                ctx.stream.cu_stream(),
            )?;
        }

        // ── 4. Pack routed tokens grouped-by-local-expert (128-aligned rows). ──
        let packed_rows = deepgemm_contig_rows_cap(total_routes.max(1), experts_per_rank);
        let packed_hidden = HiddenStates::zeros(ctx, hidden_dim, packed_rows)?;
        keepalive.keep_hidden(&packed_hidden);
        // Initialize to -1 (the invalid sentinel), NOT 0. The scatter kernel
        // treats only route_slot < 0 as invalid; zero-init left unfilled compact
        // rows looking like valid slot-0 rows, which in m=1 decode overwrote
        // route slot 0 with zero output. Pad rows from the aligned layout must
        // also stay -1 so the scatter skips their garbage GEMM output.
        let packed_route_slot = alloc_neg1_i32(ctx, packed_rows)?;
        let packed_weight = ctx
            .stream
            .alloc_zeros::<f32>(packed_rows)
            .map_err(|e| anyhow::anyhow!("DSv4 packed_weight alloc failed: {e}"))?;
        let cursors = ctx
            .stream
            .alloc_zeros::<i32>(experts_per_rank)
            .map_err(|e| anyhow::anyhow!("DSv4 cursors alloc failed: {e}"))?;
        keepalive.keep_i32(&packed_route_slot);
        keepalive.keep_f32(&packed_weight);
        keepalive.keep_i32(&cursors);
        // SAFETY: buffers valid on ctx.stream; shapes checked by the kernel.
        unsafe {
            moe::dsv4_pack_local_experts_with_slots(
                cache_ptr(&hidden.data, ctx),
                cache_ptr(route_indices, ctx),
                cache_ptr(route_weights, ctx),
                cache_ptr(&offsets, ctx),
                cache_ptr(&cursors, ctx),
                cache_ptr(&packed_hidden.data, ctx),
                cache_ptr(&packed_route_slot, ctx),
                cache_ptr(&packed_weight, ctx),
                num_tokens,
                hidden_dim,
                topk,
                local_start,
                experts_per_rank,
                ctx.stream.cu_stream(),
            )?;
        }

        // ── 5. FP8 DeepGEMM 5-call grouped expert pipeline → aligned rows. ─────
        let intermediate = layer.intermediate;
        ensure!(
            hidden_dim.is_multiple_of(128) && intermediate.is_multiple_of(128),
            "DSv4 DeepGEMM needs H and I aligned to 128, got H={hidden_dim} I={intermediate}"
        );
        ensure!(
            layer.w13_grouped.rows == intermediate * 2 && layer.w13_grouped.cols == hidden_dim,
            "DSv4 grouped w13 cache shape {}x{} != [2*I={}, H={}]",
            layer.w13_grouped.rows,
            layer.w13_grouped.cols,
            intermediate * 2,
            hidden_dim
        );
        ensure!(
            layer.w2_grouped.rows == hidden_dim && layer.w2_grouped.cols == intermediate,
            "DSv4 grouped w2 cache shape {}x{} != [H={}, I={}]",
            layer.w2_grouped.rows,
            layer.w2_grouped.cols,
            hidden_dim,
            intermediate
        );

        let expert_out = {
            let _nvtx = crate::nvtx::range("dsv4/deepgemm_grouped");
            deepgemm_grouped_experts(
                ctx,
                layer,
                &packed_hidden,
                &counts,
                &offsets,
                swiglu_limit,
                keepalive,
            )?
        };
        keepalive.keep_hidden(&expert_out);

        // ── 6. Scatter weighted expert outputs to route slots, combine topk. ────
        let nvtx_combine = crate::nvtx::range("dsv4/combine_scatter");
        let route_out = HiddenStates::zeros(ctx, hidden_dim, total_routes.max(1))?;
        keepalive.keep_hidden(&route_out);
        // SAFETY: all buffers valid on ctx.stream for the given shapes.
        unsafe {
            moe::dsv4_scatter_all_route_slots(
                cache_ptr(&expert_out.data, ctx),
                cache_ptr(&route_out.data, ctx),
                cache_ptr(&packed_route_slot, ctx),
                cache_ptr(&packed_weight, ctx),
                expert_out.seq_len,
                hidden_dim,
                ctx.stream.cu_stream(),
            )?;
            moe::dsv4_combine_route_slot_outputs(
                cache_ptr(&route_out.data, ctx),
                cache_ptr(&out.data, ctx),
                num_tokens,
                topk,
                hidden_dim,
                ctx.stream.cu_stream(),
            )?;
        }
        drop(nvtx_combine);

        // The shared expert is replicated on every rank. Callers must all-reduce
        // the routed local expert contribution first, then add the shared expert
        // exactly once per rank.
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn dsv4_moe_forward_decode_pooled(
        ctx: &DeviceContext,
        layer: &Dsv4MoeLayer,
        split: &ExpertSplit,
        route_indices: RawDevicePtr<i32>,
        route_weights: RawDevicePtr<f32>,
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        topk: usize,
        local_start: usize,
        swiglu_limit: f32,
        scratch: &mut Dsv4MoeDecodeScratch,
    ) -> Result<()> {
        let num_tokens = hidden.seq_len;
        let hidden_dim = hidden.hidden_dim;
        let total_routes = num_tokens * topk;
        ensure!(
            num_tokens == 1 && total_routes == scratch.topk,
            "DSv4 pooled decode MoE expected one token/topk routes, got tokens={num_tokens} routes={total_routes} scratch_topk={}",
            scratch.topk
        );
        let use_contiguous = use_contiguous_decode_moe();
        crate::stage_profile::profile(ctx, "dsv4/stage/moe_pack", || -> Result<()> {
            scratch.reset_routed(ctx)?;
            unsafe {
                moe::dsv4_count_local_experts(
                    route_indices,
                    cache_ptr(&scratch.counts, ctx),
                    num_tokens,
                    topk,
                    local_start,
                    split.experts_per_rank,
                    ctx.stream.cu_stream(),
                )?;
                moe::dsv4_exclusive_scan_i32(
                    cache_ptr(&scratch.counts, ctx),
                    cache_ptr(&scratch.offsets, ctx),
                    cache_ptr(&scratch.scan_total, ctx),
                    split.experts_per_rank,
                    ctx.stream.cu_stream(),
                )?;
                if use_contiguous {
                    moe::dsv4_pack_local_experts_with_slots_and_indices(
                        cache_ptr(&hidden.data, ctx),
                        route_indices,
                        route_weights,
                        cache_ptr(&scratch.offsets, ctx),
                        cache_ptr(&scratch.cursors, ctx),
                        cache_ptr(&scratch.grouped_contig.packed_hidden.data, ctx),
                        cache_ptr(&scratch.grouped_contig.packed_route_slot, ctx),
                        cache_ptr(&scratch.grouped_contig.packed_weight, ctx),
                        cache_ptr(&scratch.grouped_contig.m_indices, ctx),
                        num_tokens,
                        hidden_dim,
                        topk,
                        local_start,
                        split.experts_per_rank,
                        ctx.stream.cu_stream(),
                    )?;
                } else {
                    moe::dsv4_pack_local_experts_with_slots(
                        cache_ptr(&hidden.data, ctx),
                        route_indices,
                        route_weights,
                        cache_ptr(&scratch.offsets, ctx),
                        cache_ptr(&scratch.cursors, ctx),
                        cache_ptr(&scratch.packed_hidden.data, ctx),
                        cache_ptr(&scratch.packed_route_slot, ctx),
                        cache_ptr(&scratch.packed_weight, ctx),
                        num_tokens,
                        hidden_dim,
                        topk,
                        local_start,
                        split.experts_per_rank,
                        ctx.stream.cu_stream(),
                    )?;
                }
            }
            Ok(())
        })?;

        let expert_out = {
            let _nvtx = crate::nvtx::range("dsv4/deepgemm_grouped");
            crate::stage_profile::profile(ctx, "dsv4/stage/moe_deepgemm_grouped", || {
                if use_contiguous {
                    deepgemm_grouped_experts_contiguous_pooled(
                        ctx,
                        layer,
                        swiglu_limit,
                        &mut scratch.grouped_contig,
                    )
                } else {
                    deepgemm_grouped_experts_pooled(
                        ctx,
                        layer,
                        &scratch.packed_hidden,
                        &scratch.counts,
                        &scratch.offsets,
                        swiglu_limit,
                        &mut scratch.grouped,
                    )
                }
            })?
        };

        let nvtx_combine = crate::nvtx::range("dsv4/combine_scatter");
        crate::stage_profile::profile(ctx, "dsv4/stage/moe_combine_scatter", || -> Result<()> {
            unsafe {
                if use_contiguous {
                    moe::dsv4_scatter_all_route_slots(
                        cache_ptr(&expert_out.data, ctx),
                        cache_ptr(&scratch.route_out.data, ctx),
                        cache_ptr(&scratch.grouped_contig.packed_route_slot, ctx),
                        cache_ptr(&scratch.grouped_contig.packed_weight, ctx),
                        expert_out.seq_len,
                        hidden_dim,
                        ctx.stream.cu_stream(),
                    )?;
                } else {
                    moe::dsv4_scatter_all_route_slots(
                        cache_ptr(&expert_out.data, ctx),
                        cache_ptr(&scratch.route_out.data, ctx),
                        cache_ptr(&scratch.packed_route_slot, ctx),
                        cache_ptr(&scratch.packed_weight, ctx),
                        total_routes,
                        hidden_dim,
                        ctx.stream.cu_stream(),
                    )?;
                }
                moe::dsv4_combine_route_slot_outputs(
                    cache_ptr(&scratch.route_out.data, ctx),
                    cache_ptr(&out.data, ctx),
                    num_tokens,
                    topk,
                    hidden_dim,
                    ctx.stream.cu_stream(),
                )?;
            }
            Ok(())
        })?;
        drop(nvtx_combine);
        Ok(())
    }

    #[cfg(feature = "deepep")]
    pub(crate) fn dsv4_moe_forward_deepep(
        model: &Dsv4Model,
        transport: &crate::deepep::DeepEpTransport,
        layer: &Dsv4MoeLayer,
        tokens: &[u32],
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<()> {
        let ctx = &model.ctx;
        let cfg = &model.moe_config;
        let split = &model.split;
        let swiglu_limit = model.config.swiglu_limit;

        let num_tokens = hidden.seq_len;
        let hidden_dim = hidden.hidden_dim;
        let topk = cfg.top_k;
        let experts_per_rank = split.experts_per_rank;

        ensure!(
            tokens.len() == num_tokens,
            "DSv4 DeepEP token count {} != hidden seq_len {num_tokens}",
            tokens.len()
        );
        ensure!(
            out.hidden_dim == hidden_dim && out.seq_len == num_tokens,
            "DSv4 DeepEP out shape {}x{} != hidden {}x{}",
            out.hidden_dim,
            out.seq_len,
            hidden_dim,
            num_tokens
        );
        ensure!(
            layer.num_groups == experts_per_rank,
            "DSv4 DeepEP expert group count {} != experts_per_rank {experts_per_rank}",
            layer.num_groups
        );
        ensure!(
            layer.hidden_dim == hidden_dim,
            "DSv4 DeepEP expert hidden dim {} != runtime hidden dim {hidden_dim}",
            layer.hidden_dim
        );
        // Local DSv4 MoE is DeepGEMM-only (no scalar/native expert fallback exists
        // in this tree), so the production path is always the DeepGEMM backend. The
        // preflight below is the real guard against a build-time stub.
        moe::dsv4_deepgemm_native_preflight()?;

        // Route exactly like the allreduce path. DeepEP dispatch consumes global
        // expert ids as i64 and remaps received ids to rank-local expert ids.
        let mut logits = HiddenStates::zeros(ctx, cfg.num_experts, num_tokens)?;
        gemm_batch(ctx, &layer.gate, hidden, &mut logits)?;
        let total_routes = num_tokens * topk;
        let (topk_idx_i64, route_weights) = if use_gpu_router() {
            let routing = dsv4_route_device(model, layer, tokens, &logits, None, keepalive)?;
            let topk_idx_i64 = ctx
                .stream
                .alloc_zeros::<i64>(total_routes)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP route-index i64 alloc failed: {e}"))?;
            keepalive.keep_route_i64(&topk_idx_i64);
            unsafe {
                moe::dsv4_cast_i32_to_i64(
                    cache_ptr(&routing.indices, ctx),
                    cache_ptr(&topk_idx_i64, ctx),
                    total_routes,
                    ctx.stream.cu_stream(),
                )?;
            }
            (topk_idx_i64, routing.weights)
        } else {
            keepalive.keep_hidden(&logits);
            ctx.sync()?;
            let logits_bf16: Vec<bf16> = ctx
                .stream
                .clone_dtoh(&logits.data)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP router logits D2H failed: {e}"))?;
            let logits_host: Vec<f32> = logits_bf16.iter().map(|&v| v.to_f32()).collect();
            let decisions = dsv4_route(ctx, &model.config, cfg, layer, tokens, &logits_host)?;
            let (indices_host, weights_host) = super::flatten_routing(&decisions, topk)?;
            let indices_i64: Vec<i64> = indices_host.iter().map(|&v| i64::from(v)).collect();
            let topk_idx_i64 = ctx
                .stream
                .clone_htod(&indices_i64)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP route-index i64 H2D failed: {e}"))?;
            let route_weights = ctx
                .stream
                .clone_htod(&weights_host)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP route-weight H2D failed: {e}"))?;
            keepalive.keep_i64(&topk_idx_i64);
            keepalive.keep_f32(&route_weights);
            (topk_idx_i64, route_weights)
        };

        let num_sms = crate::deepep::DeepEpTransport::num_sms()?;
        let mut scratch =
            transport.alloc_scratch(ctx, hidden_dim, num_tokens, topk, cfg.num_experts, num_sms)?;
        keepalive.keep_hidden(&scratch.recv_x);
        keepalive.keep_i32(&scratch.recv_src_idx);
        keepalive.keep_i64(&scratch.recv_topk_idx);
        keepalive.keep_f32(&scratch.recv_topk_weights);
        keepalive.keep_i32(&scratch.rank_prefix);
        keepalive.keep_i32(&scratch.recv_channel_prefix);
        keepalive.keep_i32(&scratch.send_head);
        keepalive.keep_i32(&scratch.num_tokens_per_rank);
        keepalive.keep_i32(&scratch.num_tokens_per_expert);
        keepalive.keep_u8(&scratch.is_token_in_rank);
        keepalive.keep_i32(&scratch.channel_prefix_matrix);
        keepalive.keep_f32(&scratch.combined_topk_weights);
        let num_recv = transport.dispatch(
            ctx,
            &mut scratch,
            hidden,
            &topk_idx_i64,
            &route_weights,
            cfg.num_experts,
            topk,
            num_sms,
        )?;

        let recv_slots = num_recv.saturating_mul(topk);
        let local_routed = HiddenStates::zeros(ctx, hidden_dim, scratch.capacity_recv)?;
        if recv_slots > 0 {
            // DeepEP gives rank-local expert ids as i64. Convert the valid prefix
            // to i32 for the existing local count/pack kernels.
            let recv_i64 = ctx
                .stream
                .clone_dtoh(&scratch.recv_topk_idx)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP recv_topk_idx D2H failed: {e}"))?;
            let mut recv_i32 = vec![0i32; scratch.capacity_recv.saturating_mul(topk)];
            for (dst, &src) in recv_i32.iter_mut().zip(recv_i64.iter()).take(recv_slots) {
                *dst = i32::try_from(src).map_err(|_| {
                    anyhow::anyhow!("DSv4 DeepEP recv expert id {src} overflows i32")
                })?;
            }
            let recv_topk_i32 = ctx
                .stream
                .clone_htod(&recv_i32)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP recv_topk_i32 H2D failed: {e}"))?;
            let counts = ctx
                .stream
                .alloc_zeros::<i32>(experts_per_rank)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP local count alloc failed: {e}"))?;
            let offsets = ctx
                .stream
                .alloc_zeros::<i32>(experts_per_rank)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP local offset alloc failed: {e}"))?;
            let scan_total = ctx
                .stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP scan-total alloc failed: {e}"))?;
            keepalive.keep_i32(&recv_topk_i32);
            keepalive.keep_i32(&counts);
            keepalive.keep_i32(&offsets);
            keepalive.keep_i32(&scan_total);
            unsafe {
                moe::dsv4_count_local_experts(
                    cache_ptr(&recv_topk_i32, ctx),
                    cache_ptr(&counts, ctx),
                    num_recv,
                    topk,
                    0,
                    experts_per_rank,
                    ctx.stream.cu_stream(),
                )?;
                moe::dsv4_exclusive_scan_i32(
                    cache_ptr(&counts, ctx),
                    cache_ptr(&offsets, ctx),
                    cache_ptr(&scan_total, ctx),
                    experts_per_rank,
                    ctx.stream.cu_stream(),
                )?;
            }

            let packed_hidden = HiddenStates::zeros(ctx, hidden_dim, recv_slots.max(1))?;
            let packed_route_slot = alloc_neg1_i32(ctx, recv_slots.max(1))?;
            let packed_weight = ctx
                .stream
                .alloc_zeros::<f32>(recv_slots.max(1))
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP packed_weight alloc failed: {e}"))?;
            let cursors = ctx
                .stream
                .alloc_zeros::<i32>(experts_per_rank)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP cursors alloc failed: {e}"))?;
            keepalive.keep_hidden(&packed_hidden);
            keepalive.keep_i32(&packed_route_slot);
            keepalive.keep_f32(&packed_weight);
            keepalive.keep_i32(&cursors);
            unsafe {
                moe::dsv4_pack_local_experts_with_slots(
                    cache_ptr(&scratch.recv_x.data, ctx),
                    cache_ptr(&recv_topk_i32, ctx),
                    cache_ptr(&scratch.recv_topk_weights, ctx),
                    cache_ptr(&offsets, ctx),
                    cache_ptr(&cursors, ctx),
                    cache_ptr(&packed_hidden.data, ctx),
                    cache_ptr(&packed_route_slot, ctx),
                    cache_ptr(&packed_weight, ctx),
                    num_recv,
                    hidden_dim,
                    topk,
                    0,
                    experts_per_rank,
                    ctx.stream.cu_stream(),
                )?;
            }

            let expert_out = deepgemm_grouped_experts(
                ctx,
                layer,
                &packed_hidden,
                &counts,
                &offsets,
                swiglu_limit,
                keepalive,
            )?;
            keepalive.keep_hidden(&expert_out);
            let route_out = HiddenStates::zeros(ctx, hidden_dim, recv_slots.max(1))?;
            keepalive.keep_hidden(&route_out);
            unsafe {
                moe::dsv4_scatter_all_route_slots(
                    cache_ptr(&expert_out.data, ctx),
                    cache_ptr(&route_out.data, ctx),
                    cache_ptr(&packed_route_slot, ctx),
                    cache_ptr(&packed_weight, ctx),
                    recv_slots,
                    hidden_dim,
                    ctx.stream.cu_stream(),
                )?;
                moe::dsv4_combine_route_slot_outputs(
                    cache_ptr(&route_out.data, ctx),
                    cache_ptr(&local_routed.data, ctx),
                    num_recv,
                    topk,
                    hidden_dim,
                    ctx.stream.cu_stream(),
                )?;
            }
        }

        transport.combine(
            ctx,
            &mut scratch,
            &local_routed,
            out,
            num_recv,
            num_tokens,
            topk,
            num_sms,
        )?;
        let _ = total_routes;
        Ok(())
    }

    pub(crate) fn dsv4_shared_expert_forward(
        ctx: &DeviceContext,
        stream: &Arc<CudaStream>,
        layer: &Dsv4MoeLayer,
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        swiglu_limit: f32,
        decode_scratch: Option<&mut Dsv4MoeDecodeScratch>,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<()> {
        if let Some(scratch) = decode_scratch {
            ensure!(
                hidden.seq_len == 1
                    && scratch.hidden_dim == hidden.hidden_dim
                    && scratch.shared_intermediate == layer.shared_w2.cols,
                "DSv4 shared decode scratch mismatch: tokens={} scratch_H={} hidden_H={} scratch_I={} layer_I={}",
                hidden.seq_len,
                scratch.hidden_dim,
                hidden.hidden_dim,
                scratch.shared_intermediate,
                layer.shared_w2.cols
            );
            return dsv4_shared_expert_pooled(
                ctx,
                stream,
                layer,
                hidden,
                out,
                swiglu_limit,
                &mut scratch.shared,
            );
        }
        let shared = dsv4_shared_expert(ctx, stream, layer, hidden, swiglu_limit, keepalive)?;
        ensure!(
            shared.hidden_dim == out.hidden_dim && shared.seq_len >= out.seq_len,
            "DSv4 shared expert shape {}x{} != output {}x{}",
            shared.hidden_dim,
            shared.seq_len,
            out.hidden_dim,
            out.seq_len
        );
        let elems = out.hidden_dim * out.seq_len;
        let src = shared.data.slice(0..elems);
        stream
            .memcpy_dtod(&src, &mut out.data)
            .map_err(|e| anyhow::anyhow!("DSv4 shared expert output D2D failed: {e}"))?;
        keepalive.keep_hidden(&shared);
        Ok(())
    }

    /// Dispatch DSv4 host routing for one layer: bias-routed → learned router +
    /// `noaux_tc` correction bias via `infer_moe::route` (the validated path);
    /// hash-routed → `tid2eid` table via [`super::hash_route`]. Returns one
    /// [`RoutingDecision`] per token.
    pub(crate) fn dsv4_route(
        ctx: &DeviceContext,
        config: &deepseek_spec::DeepSeekV4Config,
        moe_config: &infer_moe::MoeConfig,
        layer: &Dsv4MoeLayer,
        tokens: &[u32],
        logits_host: &[f32],
    ) -> Result<Vec<infer_moe::RoutingDecision>> {
        use deepseek_spec::DeepSeekV4MoeRoutingKind;
        match layer.routing_kind {
            DeepSeekV4MoeRoutingKind::Hash => {
                let tid2eid = layer.hash_tid2eid.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("DSv4 hash-routed MoE layer missing tid2eid table")
                })?;
                super::hash_route(config, tid2eid, tokens, logits_host)
            }
            DeepSeekV4MoeRoutingKind::LearnedBias => {
                let gate_bias = layer.gate_bias.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("DSv4 bias-routed MoE layer missing gate bias")
                })?;
                // bias D2H is cheap (one [n_routed] vec); thread the REAL bias.
                let bias_bf16: Vec<bf16> = ctx
                    .stream
                    .clone_dtoh(&gate_bias.data)
                    .map_err(|e| anyhow::anyhow!("DSv4 gate bias D2H failed: {e}"))?;
                let bias_host: Vec<f32> = bias_bf16.iter().map(|&v| v.to_f32()).collect();
                infer_moe::route(logits_host, &bias_host, moe_config)
                    .map_err(|e| anyhow::anyhow!("DSv4 host route failed: {e}"))
            }
        }
    }

    /// Run the FP8 DeepGEMM 5-call grouped expert pipeline over this rank's local
    /// experts; returns the padded/aligned expert output. The caller's
    /// `packed_route_slot = -1` pad rows decide which rows reach route slots.
    fn deepgemm_grouped_experts(
        ctx: &DeviceContext,
        layer: &Dsv4MoeLayer,
        packed_hidden: &HiddenStates,
        counts: &cudarc::driver::CudaSlice<i32>,
        offsets: &cudarc::driver::CudaSlice<i32>,
        swiglu_limit: f32,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<HiddenStates> {
        let num_groups = layer.num_groups;
        let hidden_dim = packed_hidden.hidden_dim;
        let intermediate = layer.intermediate;
        let w13 = &layer.w13_grouped;
        let w2 = &layer.w2_grouped;
        ensure!(
            layer.hidden_dim == hidden_dim,
            "DSv4 grouped expert hidden dim {} != packed hidden dim {hidden_dim}",
            layer.hidden_dim
        );
        ensure!(
            w13.groups == num_groups
                && w2.groups == num_groups
                && w13.rows == 2 * intermediate
                && w13.cols == hidden_dim
                && w2.rows == hidden_dim
                && w2.cols == intermediate,
            "DSv4 grouped expert cache metadata mismatch: groups={} w13={}x{} g={} w2={}x{} g={} hidden={hidden_dim} inter={intermediate}",
            num_groups,
            w13.rows,
            w13.cols,
            w13.groups,
            w2.rows,
            w2.cols,
            w2.groups,
        );

        // Prefill can have tens of thousands of routes. The old masked layout
        // materialized `num_groups * max_m` rows and overflowed the unpad kernel's
        // work-size at ~1.5K prompt tokens (32 * T * topk * H > i32::MAX), long
        // before the 8K/32K SLO shapes. Use DeepGEMM's contiguous grouped layout
        // over 128-aligned route rows instead: `m_indices[row]` names the local
        // expert for each activation row, and every pad row stays -1.
        let rows = packed_hidden.seq_len.max(1);
        let scale_stride_m = rows.div_ceil(4) * 4;
        let hidden_scale_cols = hidden_dim.div_ceil(128);
        let inter_scale_cols = intermediate.div_ceil(128);

        let input_fp8 = alloc_u8(ctx, rows * hidden_dim)?;
        let input_scales = alloc_zeros_f32(ctx, scale_stride_m * hidden_scale_cols)?;
        let w13_out = HiddenStates::zeros(ctx, 2 * intermediate, rows)?;
        let act_fp8 = alloc_u8(ctx, rows * intermediate)?;
        let act_scales = alloc_zeros_f32(ctx, scale_stride_m * inter_scale_cols)?;
        let mut out_compact = HiddenStates::zeros(ctx, hidden_dim, packed_hidden.seq_len.max(1))?;
        let m_indices = alloc_neg1_i32(ctx, rows)?;
        let active_experts = ctx.stream.clone_htod(&[0i32]).map_err(|e| {
            anyhow::anyhow!("DSv4 DeepGEMM contiguous active-expert H2D failed: {e}")
        })?;
        let active_offsets = ctx
            .stream
            .clone_htod(&[0i32])
            .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM contiguous offset H2D failed: {e}"))?;
        let active_counts = ctx
            .stream
            .clone_htod(&[i32::try_from(rows)?])
            .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM contiguous count H2D failed: {e}"))?;
        keepalive.keep_u8(&input_fp8);
        keepalive.keep_f32(&input_scales);
        keepalive.keep_hidden(&w13_out);
        keepalive.keep_u8(&act_fp8);
        keepalive.keep_f32(&act_scales);
        keepalive.keep_hidden(&out_compact);
        keepalive.keep_i32(&m_indices);
        keepalive.keep_i32(&active_experts);
        keepalive.keep_i32(&active_offsets);
        keepalive.keep_i32(&active_counts);

        let p_hidden = cache_ptr(&packed_hidden.data, ctx);
        let p_in_fp8 = cache_ptr(&input_fp8, ctx);
        let p_in_scales = cache_ptr(&input_scales, ctx);
        let p_active = cache_ptr(&active_experts, ctx);
        let p_active_offsets = cache_ptr(&active_offsets, ctx);
        let p_active_counts = cache_ptr(&active_counts, ctx);
        let p_m_indices = cache_ptr(&m_indices, ctx);
        let p_w13_out = cache_ptr(&w13_out.data, ctx);
        let p_act_fp8 = cache_ptr(&act_fp8, ctx);
        let p_act_scales = cache_ptr(&act_scales, ctx);
        let p_out_compact = cache_ptr(&out_compact.data, ctx);
        let stream = ctx.stream.cu_stream();

        // SAFETY: every buffer/ptr above is valid on ctx.stream for the shapes
        // checked against the loaded caches; the kernels bound rows by masked_m.
        unsafe {
            moe::dsv4_fill_m_indices_from_counts(
                cache_ptr(counts, ctx),
                cache_ptr(offsets, ctx),
                p_m_indices,
                num_groups,
                rows,
                stream,
            )?;
            moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                p_hidden,
                p_in_fp8,
                p_in_scales,
                p_active,
                p_active_offsets,
                p_active_counts,
                1,
                rows,
                hidden_dim,
                scale_stride_m,
                stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                p_in_fp8,
                p_in_scales,
                cache_ptr(&w13.weight, ctx),
                cache_ptr(&w13.scales, ctx),
                p_w13_out,
                p_m_indices,
                num_groups,
                rows,
                2 * intermediate,
                hidden_dim,
                scale_stride_m,
                stream,
            )?;
            moe::dsv4_deepgemm_swiglu_quantize_w13(
                p_w13_out,
                p_act_fp8,
                p_act_scales,
                p_active,
                p_active_counts,
                1,
                rows,
                intermediate,
                scale_stride_m,
                swiglu_limit,
                stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                p_act_fp8,
                p_act_scales,
                cache_ptr(&w2.weight, ctx),
                cache_ptr(&w2.scales, ctx),
                p_out_compact,
                p_m_indices,
                num_groups,
                rows,
                hidden_dim,
                intermediate,
                scale_stride_m,
                stream,
            )?;
        }

        out_compact.seq_len = packed_hidden.seq_len;
        Ok(out_compact)
    }

    fn deepgemm_grouped_experts_pooled(
        ctx: &DeviceContext,
        layer: &Dsv4MoeLayer,
        packed_hidden: &HiddenStates,
        counts: &CudaSlice<i32>,
        offsets: &CudaSlice<i32>,
        swiglu_limit: f32,
        scratch: &mut Dsv4GroupedDecodeScratch,
    ) -> Result<HiddenStates> {
        let num_groups = layer.num_groups;
        let hidden_dim = packed_hidden.hidden_dim;
        let intermediate = layer.intermediate;
        let w13 = &layer.w13_grouped;
        let w2 = &layer.w2_grouped;
        ensure!(
            scratch.max_m >= packed_hidden.seq_len
                && scratch.out_compact.seq_len >= packed_hidden.seq_len
                && scratch.out_padded.hidden_dim == hidden_dim
                && scratch.w13_out.hidden_dim == 2 * intermediate,
            "DSv4 pooled grouped scratch mismatch: max_m={} out_cap={} packed={} H={} I={}",
            scratch.max_m,
            scratch.out_compact.seq_len,
            packed_hidden.seq_len,
            hidden_dim,
            intermediate
        );
        ensure!(
            w13.groups == num_groups
                && w2.groups == num_groups
                && w13.rows == 2 * intermediate
                && w13.cols == hidden_dim
                && w2.rows == hidden_dim
                && w2.cols == intermediate,
            "DSv4 grouped expert cache metadata mismatch: groups={} w13={}x{} g={} w2={}x{} g={} hidden={hidden_dim} inter={intermediate}",
            num_groups,
            w13.rows,
            w13.cols,
            w13.groups,
            w2.rows,
            w2.cols,
            w2.groups,
        );
        // `masked_m` is the per-group valid-row count = `counts`; alias it directly
        // (the masked GEMM only reads it, and `counts` is already passed read-only to
        // pack_quantize + swiglu below) instead of a per-layer D2D copy into
        // `scratch.masked_m` — kills one cuMemcpyDtoD/layer (the 17.8% D2D bucket).
        let p_hidden = cache_ptr(&packed_hidden.data, ctx);
        let p_in_fp8 = cache_ptr(&scratch.input_fp8, ctx);
        let p_in_scales = cache_ptr(&scratch.input_scales, ctx);
        let p_active = cache_ptr(&scratch.active_experts, ctx);
        let p_offsets = cache_ptr(offsets, ctx);
        let p_counts = cache_ptr(counts, ctx);
        // Alias masked_m to counts (same data, read-only in the masked GEMM).
        let p_masked = p_counts;
        let p_w13_out = cache_ptr(&scratch.w13_out.data, ctx);
        let p_act_fp8 = cache_ptr(&scratch.act_fp8, ctx);
        let p_act_scales = cache_ptr(&scratch.act_scales, ctx);
        let p_out_padded = cache_ptr(&scratch.out_padded.data, ctx);
        let p_out_compact = cache_ptr(&scratch.out_compact.data, ctx);
        let stream = ctx.stream.cu_stream();

        unsafe {
            moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                p_hidden,
                p_in_fp8,
                p_in_scales,
                p_active,
                p_offsets,
                p_counts,
                num_groups,
                scratch.max_m,
                hidden_dim,
                scratch.scale_stride_m,
                stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
                p_in_fp8,
                p_in_scales,
                cache_ptr(&w13.weight, ctx),
                cache_ptr(&w13.scales, ctx),
                p_w13_out,
                p_masked,
                num_groups,
                scratch.max_m,
                2 * intermediate,
                hidden_dim,
                scratch.scale_stride_m,
                stream,
            )?;
            moe::dsv4_deepgemm_swiglu_quantize_w13(
                p_w13_out,
                p_act_fp8,
                p_act_scales,
                p_active,
                p_counts,
                num_groups,
                scratch.max_m,
                intermediate,
                scratch.scale_stride_m,
                swiglu_limit,
                stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
                p_act_fp8,
                p_act_scales,
                cache_ptr(&w2.weight, ctx),
                cache_ptr(&w2.scales, ctx),
                p_out_padded,
                p_masked,
                num_groups,
                scratch.max_m,
                hidden_dim,
                intermediate,
                scratch.scale_stride_m,
                stream,
            )?;
            moe::dsv4_deepgemm_unpad_grouped_bf16(
                p_out_padded,
                p_out_compact,
                p_active,
                p_offsets,
                p_counts,
                num_groups,
                scratch.max_m,
                hidden_dim,
                stream,
            )?;
        }

        Ok(HiddenStates {
            data: scratch.out_compact.data.clone(),
            hidden_dim,
            seq_len: packed_hidden.seq_len,
        })
    }

    fn deepgemm_grouped_experts_contiguous_pooled(
        ctx: &DeviceContext,
        layer: &Dsv4MoeLayer,
        swiglu_limit: f32,
        scratch: &mut Dsv4GroupedContiguousDecodeScratch,
    ) -> Result<HiddenStates> {
        let num_groups = layer.num_groups;
        let packed_hidden = &scratch.packed_hidden;
        let hidden_dim = packed_hidden.hidden_dim;
        let intermediate = layer.intermediate;
        let w13 = &layer.w13_grouped;
        let w2 = &layer.w2_grouped;
        ensure!(
            scratch.rows >= packed_hidden.seq_len
                && scratch.out.hidden_dim == hidden_dim
                && scratch.w13_out.hidden_dim == 2 * intermediate,
            "DSv4 contiguous grouped scratch mismatch: rows={} packed={} H={} I={}",
            scratch.rows,
            packed_hidden.seq_len,
            hidden_dim,
            intermediate
        );
        ensure!(
            w13.groups == num_groups
                && w2.groups == num_groups
                && w13.rows == 2 * intermediate
                && w13.cols == hidden_dim
                && w2.rows == hidden_dim
                && w2.cols == intermediate,
            "DSv4 contiguous grouped cache metadata mismatch: groups={} w13={}x{} g={} w2={}x{} g={} hidden={hidden_dim} inter={intermediate}",
            num_groups,
            w13.rows,
            w13.cols,
            w13.groups,
            w2.rows,
            w2.cols,
            w2.groups,
        );

        let p_hidden = cache_ptr(&packed_hidden.data, ctx);
        let p_in_fp8 = cache_ptr(&scratch.input_fp8, ctx);
        let p_in_scales = cache_ptr(&scratch.input_scales, ctx);
        let p_active = cache_ptr(&scratch.active_experts, ctx);
        let p_offsets = cache_ptr(&scratch.active_offsets, ctx);
        let p_counts = cache_ptr(&scratch.active_counts, ctx);
        let p_m_indices = cache_ptr(&scratch.m_indices, ctx);
        let p_w13_out = cache_ptr(&scratch.w13_out.data, ctx);
        let p_act_fp8 = cache_ptr(&scratch.act_fp8, ctx);
        let p_act_scales = cache_ptr(&scratch.act_scales, ctx);
        let p_out = cache_ptr(&scratch.out.data, ctx);
        let stream = ctx.stream.cu_stream();

        unsafe {
            moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                p_hidden,
                p_in_fp8,
                p_in_scales,
                p_active,
                p_offsets,
                p_counts,
                1,
                scratch.rows,
                hidden_dim,
                scratch.scale_stride_m,
                stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                p_in_fp8,
                p_in_scales,
                cache_ptr(&w13.weight, ctx),
                cache_ptr(&w13.scales, ctx),
                p_w13_out,
                p_m_indices,
                num_groups,
                scratch.rows,
                2 * intermediate,
                hidden_dim,
                scratch.scale_stride_m,
                stream,
            )?;
            moe::dsv4_deepgemm_swiglu_quantize_w13(
                p_w13_out,
                p_act_fp8,
                p_act_scales,
                p_active,
                p_counts,
                1,
                scratch.rows,
                intermediate,
                scratch.scale_stride_m,
                swiglu_limit,
                stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                p_act_fp8,
                p_act_scales,
                cache_ptr(&w2.weight, ctx),
                cache_ptr(&w2.scales, ctx),
                p_out,
                p_m_indices,
                num_groups,
                scratch.rows,
                hidden_dim,
                intermediate,
                scratch.scale_stride_m,
                stream,
            )?;
        }

        Ok(HiddenStates {
            data: scratch.out.data.clone(),
            hidden_dim,
            seq_len: packed_hidden.seq_len,
        })
    }

    /// DSv4 dense shared expert via a single-group FP8 DeepGEMM pass: w13 fused
    /// gate+up → clamped SwiGLU → w2 down, over every token. No routing/scatter.
    fn dsv4_shared_expert_pooled(
        ctx: &DeviceContext,
        stream: &Arc<CudaStream>,
        layer: &Dsv4MoeLayer,
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        swiglu_limit: f32,
        scratch: &mut Dsv4SharedDecodeScratch,
    ) -> Result<()> {
        let hidden_dim = hidden.hidden_dim;
        let num_tokens = hidden.seq_len;
        let shared_inter = layer.shared_w2.cols;
        ensure!(
            num_tokens == 1
                && scratch.out.hidden_dim == hidden_dim
                && scratch.w13_out.hidden_dim == 2 * shared_inter,
            "DSv4 pooled shared scratch mismatch: tokens={num_tokens} H={hidden_dim} I={shared_inter}"
        );
        ensure!(
            out.hidden_dim == hidden_dim && out.seq_len == num_tokens,
            "DSv4 pooled shared out shape {}x{} != hidden {}x{}",
            out.hidden_dim,
            out.seq_len,
            hidden_dim,
            num_tokens
        );
        let p_hidden = cache_ptr(&hidden.data, ctx);
        let p_in_fp8 = cache_ptr(&scratch.input_fp8, ctx);
        let p_in_scales = cache_ptr(&scratch.input_scales, ctx);
        let p_active = cache_ptr(&scratch.active_experts, ctx);
        let p_offsets = cache_ptr(&scratch.active_offsets, ctx);
        let p_counts = cache_ptr(&scratch.counts, ctx);
        let p_masked = cache_ptr(&scratch.masked_m, ctx);
        let p_w13_out = cache_ptr(&scratch.w13_out.data, ctx);
        let p_act_fp8 = cache_ptr(&scratch.act_fp8, ctx);
        let p_act_scales = cache_ptr(&scratch.act_scales, ctx);
        let p_out = cache_ptr(&scratch.out.data, ctx);
        let cu_stream = stream.cu_stream();

        unsafe {
            moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                p_hidden,
                p_in_fp8,
                p_in_scales,
                p_active,
                p_offsets,
                p_counts,
                1,
                scratch.max_m,
                hidden_dim,
                scratch.scale_stride_m,
                cu_stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
                p_in_fp8,
                p_in_scales,
                cache_ptr(&layer.shared_w13.weight, ctx),
                cache_ptr(&layer.shared_w13.scales, ctx),
                p_w13_out,
                p_masked,
                1,
                scratch.max_m,
                2 * shared_inter,
                hidden_dim,
                scratch.scale_stride_m,
                cu_stream,
            )?;
            moe::dsv4_deepgemm_swiglu_quantize_w13(
                p_w13_out,
                p_act_fp8,
                p_act_scales,
                p_active,
                p_counts,
                1,
                scratch.max_m,
                shared_inter,
                scratch.scale_stride_m,
                swiglu_limit,
                cu_stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
                p_act_fp8,
                p_act_scales,
                cache_ptr(&layer.shared_w2.weight, ctx),
                cache_ptr(&layer.shared_w2.scales, ctx),
                p_out,
                p_masked,
                1,
                scratch.max_m,
                hidden_dim,
                shared_inter,
                scratch.scale_stride_m,
                cu_stream,
            )?;
        }

        let elems = hidden_dim * num_tokens;
        let src = scratch.out.data.slice(0..elems);
        stream
            .memcpy_dtod(&src, &mut out.data)
            .map_err(|e| anyhow::anyhow!("DSv4 pooled shared output D2D failed: {e}"))?;
        Ok(())
    }

    /// DSv4 dense shared expert via a single-group FP8 DeepGEMM pass: w13 fused
    /// gate+up → clamped SwiGLU → w2 down, over every token. No routing/scatter.
    fn dsv4_shared_expert(
        ctx: &DeviceContext,
        stream: &Arc<CudaStream>,
        layer: &Dsv4MoeLayer,
        hidden: &HiddenStates,
        swiglu_limit: f32,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<HiddenStates> {
        let hidden_dim = hidden.hidden_dim;
        let num_tokens = hidden.seq_len;
        let shared_inter = layer.shared_w2.cols;
        ensure!(
            layer.shared_w13.rows == shared_inter * 2 && layer.shared_w13.cols == hidden_dim,
            "DSv4 shared w13 shape {}x{} != [2*I={}, H={}]",
            layer.shared_w13.rows,
            layer.shared_w13.cols,
            shared_inter * 2,
            hidden_dim
        );
        ensure!(
            layer.shared_w2.rows == hidden_dim,
            "DSv4 shared w2 rows {} != hidden {hidden_dim}",
            layer.shared_w2.rows
        );
        ensure!(
            hidden_dim.is_multiple_of(128) && shared_inter.is_multiple_of(128),
            "DSv4 shared expert needs H and I aligned to 128, got H={hidden_dim} I={shared_inter}"
        );

        // Floor at 128 for the SAME reason as the routed grouped path: the shared
        // expert runs the identical `dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked`
        // kernel, whose small-m (m=1 decode) tile path diverges below 128. The
        // routed-only floor left this reachable (codex review P2); a prompt whose
        // next-token margin leans on the shared expert could flip at m<128.
        let max_m = num_tokens.max(128);
        let scale_stride_m = max_m.div_ceil(4) * 4;
        let hidden_scale_cols = hidden_dim.div_ceil(128);
        let inter_scale_cols = shared_inter.div_ceil(128);
        let use_ctx_stream = Arc::ptr_eq(stream, &ctx.stream);

        let input_fp8 = if use_ctx_stream {
            alloc_u8(ctx, max_m * hidden_dim)?
        } else {
            alloc_u8_on(stream, max_m * hidden_dim)?
        };
        let input_scales = if use_ctx_stream {
            alloc_zeros_f32(ctx, scale_stride_m * hidden_scale_cols)?
        } else {
            alloc_zeros_f32_on(stream, scale_stride_m * hidden_scale_cols)?
        };
        let w13_out = if use_ctx_stream {
            HiddenStates::zeros(ctx, 2 * shared_inter, max_m)?
        } else {
            hidden_zeros_on(stream, 2 * shared_inter, max_m)?
        };
        let act_fp8 = if use_ctx_stream {
            alloc_u8(ctx, max_m * shared_inter)?
        } else {
            alloc_u8_on(stream, max_m * shared_inter)?
        };
        let act_scales = if use_ctx_stream {
            alloc_zeros_f32(ctx, scale_stride_m * inter_scale_cols)?
        } else {
            alloc_zeros_f32_on(stream, scale_stride_m * inter_scale_cols)?
        };
        // DeepGEMM's TMA D descriptor is built with `m = max_m`, so the output
        // allocation must cover the padded row capacity even though downstream
        // consumers only read the first `num_tokens` rows.
        let out = if use_ctx_stream {
            HiddenStates::zeros(ctx, hidden_dim, max_m)?
        } else {
            hidden_zeros_on(stream, hidden_dim, max_m)?
        };
        keepalive.keep_u8(&input_fp8);
        keepalive.keep_f32(&input_scales);
        keepalive.keep_hidden(&w13_out);
        keepalive.keep_u8(&act_fp8);
        keepalive.keep_f32(&act_scales);
        keepalive.keep_hidden(&out);

        // Single group spanning all tokens: identity expert 0, offset 0, count T.
        let active_experts = if use_ctx_stream {
            ctx.stream.clone_htod(&[0i32])
        } else {
            stream.clone_htod(&[0i32])
        }
        .map_err(|e| anyhow::anyhow!("DSv4 shared active H2D failed: {e}"))?;
        let active_offsets = if use_ctx_stream {
            ctx.stream.clone_htod(&[0i32])
        } else {
            stream.clone_htod(&[0i32])
        }
        .map_err(|e| anyhow::anyhow!("DSv4 shared offset H2D failed: {e}"))?;
        let counts = if use_ctx_stream {
            ctx.stream.clone_htod(&[num_tokens as i32])
        } else {
            stream.clone_htod(&[num_tokens as i32])
        }
        .map_err(|e| anyhow::anyhow!("DSv4 shared count H2D failed: {e}"))?;
        let masked_m = if use_ctx_stream {
            ctx.stream.clone_htod(&[num_tokens as i32])
        } else {
            stream.clone_htod(&[num_tokens as i32])
        }
        .map_err(|e| anyhow::anyhow!("DSv4 shared masked_m H2D failed: {e}"))?;
        keepalive.keep_i32(&active_experts);
        keepalive.keep_i32(&active_offsets);
        keepalive.keep_i32(&counts);
        keepalive.keep_i32(&masked_m);

        let p_hidden = cache_ptr(&hidden.data, ctx);
        let p_in_fp8 = cache_ptr(&input_fp8, ctx);
        let p_in_scales = cache_ptr(&input_scales, ctx);
        let p_active = cache_ptr(&active_experts, ctx);
        let p_offsets = cache_ptr(&active_offsets, ctx);
        let p_counts = cache_ptr(&counts, ctx);
        let p_masked = cache_ptr(&masked_m, ctx);
        let p_w13_out = cache_ptr(&w13_out.data, ctx);
        let p_act_fp8 = cache_ptr(&act_fp8, ctx);
        let p_act_scales = cache_ptr(&act_scales, ctx);
        let p_out = cache_ptr(&out.data, ctx);
        let cu_stream = if use_ctx_stream {
            ctx.stream.cu_stream()
        } else {
            stream.cu_stream()
        };

        // SAFETY: single-group buffers valid on the selected stream; masked_m bounds rows.
        unsafe {
            moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                p_hidden,
                p_in_fp8,
                p_in_scales,
                p_active,
                p_offsets,
                p_counts,
                1,
                max_m,
                hidden_dim,
                scale_stride_m,
                cu_stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
                p_in_fp8,
                p_in_scales,
                cache_ptr(&layer.shared_w13.weight, ctx),
                cache_ptr(&layer.shared_w13.scales, ctx),
                p_w13_out,
                p_masked,
                1,
                max_m,
                2 * shared_inter,
                hidden_dim,
                scale_stride_m,
                cu_stream,
            )?;
            moe::dsv4_deepgemm_swiglu_quantize_w13(
                p_w13_out,
                p_act_fp8,
                p_act_scales,
                p_active,
                p_counts,
                1,
                max_m,
                shared_inter,
                scale_stride_m,
                swiglu_limit,
                cu_stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
                p_act_fp8,
                p_act_scales,
                cache_ptr(&layer.shared_w2.weight, ctx),
                cache_ptr(&layer.shared_w2.scales, ctx),
                p_out,
                p_masked,
                1,
                max_m,
                hidden_dim,
                shared_inter,
                scale_stride_m,
                cu_stream,
            )?;
        }

        Ok(HiddenStates {
            data: out.data.clone(),
            hidden_dim,
            seq_len: num_tokens,
        })
    }

    /// One contiguous group-major FP8 weight + scale buffer the masked GEMM
    /// strides per group (`b + g * n * k`, `sfb + g * scale_rows * scale_cols`).
    pub(crate) struct GroupedCache {
        pub(crate) weight: cudarc::driver::CudaSlice<u8>,
        pub(crate) scales: cudarc::driver::CudaSlice<f32>,
        pub(crate) groups: usize,
        pub(crate) rows: usize,
        pub(crate) cols: usize,
    }

    /// Concatenate the per-expert FP8 caches into one contiguous group-major
    /// buffer (D2D), validating uniform `[rows, cols]` shape + 128-row alignment.
    ///
    /// The loader stores this grouped cache in [`crate::dsv4::Dsv4MoeLayer`] and
    /// drops the per-expert Vecs. The weights are static after load; rebuilding
    /// this concat during decode would copy hundreds of MiB per layer per token.
    pub(crate) fn build_grouped_cache(
        ctx: &DeviceContext,
        caches: &[Dsv4Fp8DeepGemmWeightCache],
        rows: usize,
        cols: usize,
    ) -> Result<GroupedCache> {
        let first = caches
            .first()
            .ok_or_else(|| anyhow::anyhow!("DSv4 DeepGEMM grouped cache has no local experts"))?;
        ensure!(
            first.rows == rows && first.cols == cols,
            "DSv4 grouped cache shape {}x{} != [{rows}, {cols}]",
            first.rows,
            first.cols
        );
        ensure!(
            rows.is_multiple_of(128),
            "DSv4 grouped cache rows {rows} must be 128-aligned for group-major scale concat"
        );
        let num_groups = caches.len();
        let weight_stride = first.weight.len();
        let scale_stride = first.scales.len();
        let mut weight = ctx
            .stream
            .alloc_zeros::<u8>(num_groups * weight_stride)
            .map_err(|e| anyhow::anyhow!("DSv4 grouped weight alloc failed: {e}"))?;
        let mut scales = ctx
            .stream
            .alloc_zeros::<f32>(num_groups * scale_stride)
            .map_err(|e| anyhow::anyhow!("DSv4 grouped scale alloc failed: {e}"))?;
        for (g, cache) in caches.iter().enumerate() {
            ensure!(
                cache.rows == rows
                    && cache.cols == cols
                    && cache.weight.len() == weight_stride
                    && cache.scales.len() == scale_stride,
                "DSv4 grouped cache group {g} non-uniform: {}x{}",
                cache.rows,
                cache.cols
            );
            {
                let mut dst = weight.slice_mut(g * weight_stride..(g + 1) * weight_stride);
                ctx.stream
                    .memcpy_dtod(&cache.weight, &mut dst)
                    .map_err(|e| anyhow::anyhow!("DSv4 grouped weight D2D failed: {e}"))?;
            }
            let mut dst = scales.slice_mut(g * scale_stride..(g + 1) * scale_stride);
            ctx.stream
                .memcpy_dtod(&cache.scales, &mut dst)
                .map_err(|e| anyhow::anyhow!("DSv4 grouped scale D2D failed: {e}"))?;
        }
        Ok(GroupedCache {
            weight,
            scales,
            groups: num_groups,
            rows,
            cols,
        })
    }

    fn alloc_u8(ctx: &DeviceContext, len: usize) -> Result<cudarc::driver::CudaSlice<u8>> {
        ctx.stream
            .alloc_zeros::<u8>(len.max(1))
            .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM u8 scratch alloc failed: {e}"))
    }

    fn alloc_u8_on(stream: &Arc<CudaStream>, len: usize) -> Result<cudarc::driver::CudaSlice<u8>> {
        stream
            .alloc_zeros::<u8>(len.max(1))
            .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM u8 scratch alloc failed: {e}"))
    }

    fn alloc_zeros_f32(ctx: &DeviceContext, len: usize) -> Result<cudarc::driver::CudaSlice<f32>> {
        ctx.stream
            .alloc_zeros::<f32>(len.max(1))
            .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM f32 scratch alloc failed: {e}"))
    }

    fn alloc_zeros_f32_on(
        stream: &Arc<CudaStream>,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<f32>> {
        stream
            .alloc_zeros::<f32>(len.max(1))
            .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM f32 scratch alloc failed: {e}"))
    }

    fn hidden_zeros_on(
        stream: &Arc<CudaStream>,
        hidden_dim: usize,
        seq_len: usize,
    ) -> Result<HiddenStates> {
        let len = hidden_dim * seq_len;
        let data = stream
            .alloc_zeros::<bf16>(len.max(1))
            .map_err(|e| anyhow::anyhow!("DSv4 hidden scratch alloc failed: {e}"))?;
        Ok(HiddenStates {
            data,
            hidden_dim,
            seq_len,
        })
    }

    /// NVSHMEM low-latency (token-owned) DSv4 MoE forward for THIS rank's owned
    /// token slice. Implements the EP pipeline over the owned `[hidden, owned_n]`
    /// activations: route → LL dispatch (FP8 pack) → masked grouped GEMM w13 →
    /// masked SwiGLU+requant → masked grouped GEMM w2 → LL combine → add shared
    /// expert. The caller (dsv4.rs) owns the token slicing + the final all-gather.
    ///
    /// `out` is this rank's owned routed+shared output `[hidden, owned_n]`. The
    /// LL packed recv / GEMM-output scratch is pre-allocated once in `scratch`;
    /// this path only overwrites it (no per-step alloc beyond the small route +
    /// topk-id buffers + the routed/shared temporaries).
    #[cfg(feature = "deepep")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dsv4_moe_forward_deepep_ll(
        model: &Dsv4Model,
        transport: &crate::deepep::DeepEpTransport,
        scratch: &mut crate::deepep::DeepEpLlScratch,
        layer: &Dsv4MoeLayer,
        tokens: &[u32],
        global_tokens: usize,
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<()> {
        let ctx = &model.ctx;
        let cfg = &model.moe_config;
        let swiglu_limit = model.config.swiglu_limit;

        let owned_n = hidden.seq_len;
        let hidden_dim = hidden.hidden_dim;
        let topk = cfg.top_k;
        let intermediate = layer.intermediate;
        let num_local_experts = transport.ll_num_local_experts()?;

        ensure!(
            tokens.len() == owned_n,
            "deepep_ll token count {} != owned hidden seq_len {owned_n}",
            tokens.len()
        );
        ensure!(
            out.hidden_dim == hidden_dim && out.seq_len == owned_n,
            "deepep_ll out shape {}x{} != owned {hidden_dim}x{owned_n}",
            out.hidden_dim,
            out.seq_len,
        );
        ensure!(
            layer.num_groups == num_local_experts,
            "deepep_ll expert group count {} != num_local_experts {num_local_experts}",
            layer.num_groups
        );
        ensure!(
            layer.hidden_dim == hidden_dim,
            "deepep_ll expert hidden dim {} != runtime hidden dim {hidden_dim}",
            layer.hidden_dim
        );
        moe::dsv4_deepgemm_native_preflight()?;

        // ── Step 2: route the OWNED tokens on device → topk_idx (i64, global
        // expert ids, kept on device — no host i32→i64 glue) + topk weights.
        // owned_n may be 0 (seq_len < world): this rank still participates in the
        // dispatch/combine COLLECTIVE with num_tokens=0 (sends nothing, receives
        // tokens routed to its local experts), so allocate 1-slot dummies for the
        // route buffers and skip the actual route compute.
        let total_routes = owned_n * topk;
        let topk_idx_i64 = ctx
            .stream
            .alloc_zeros::<i64>(total_routes.max(1))
            .map_err(|e| anyhow::anyhow!("deepep_ll route-index i64 alloc failed: {e}"))?;
        keepalive.keep_route_i64(&topk_idx_i64);
        let route_weights = if owned_n > 0 {
            let mut logits = HiddenStates::zeros(ctx, cfg.num_experts, owned_n)?;
            gemm_batch(ctx, &layer.gate, hidden, &mut logits)?;
            keepalive.keep_hidden(&logits);
            let routing = dsv4_route_device(model, layer, tokens, &logits, None, keepalive)?;
            keepalive.keep_route_f32(&routing.weights);
            // SAFETY: both buffers hold `total_routes` elements on `ctx.stream`.
            unsafe {
                moe::dsv4_cast_i32_to_i64(
                    cache_ptr(&routing.indices, ctx),
                    cache_ptr(&topk_idx_i64, ctx),
                    total_routes,
                    ctx.stream.cu_stream(),
                )?;
            }
            routing.weights
        } else {
            let w = ctx
                .stream
                .alloc_zeros::<f32>(1)
                .map_err(|e| anyhow::anyhow!("deepep_ll empty route-weight alloc failed: {e}"))?;
            keepalive.keep_route_f32(&w);
            w
        };

        // ── Step 3: LL dispatch the owned tokens → packed FP8 recv into scratch.
        // `scratch` is model-owned (outlives the forward), so it does not need
        // the forward-keepalive guard the transient buffers below get.
        let _expected_m = transport.ll_dispatch(ctx, scratch, hidden, &topk_idx_i64, topk)?;

        let m = scratch.m_padded;
        let sfa_aligned_m = scratch.sfa_aligned_m;
        let w13 = &layer.w13_grouped;
        let w2 = &layer.w2_grouped;
        ensure!(
            w13.groups == num_local_experts
                && w2.groups == num_local_experts
                && w13.rows == 2 * intermediate
                && w13.cols == hidden_dim
                && w2.rows == hidden_dim
                && w2.cols == intermediate,
            "deepep_ll grouped weight metadata mismatch: groups={} w13={}x{}g{} w2={}x{}g{} H={hidden_dim} I={intermediate}",
            num_local_experts,
            w13.rows,
            w13.cols,
            w13.groups,
            w2.rows,
            w2.cols,
            w2.groups,
        );

        let p_recv_x = cache_ptr(&scratch.recv_x_fp8, ctx);
        let p_recv_sc = cache_ptr(&scratch.recv_x_scales, ctx);
        let p_masked = cache_ptr(&scratch.recv_count, ctx);
        let p_w13_out = cache_ptr(&scratch.w13_out, ctx);
        let p_act_fp8 = cache_ptr(&scratch.act_fp8, ctx);
        let p_act_sc = cache_ptr(&scratch.act_scales, ctx);
        let p_expert_out = cache_ptr(&scratch.expert_out, ctx);
        let stream = ctx.stream.cu_stream();

        // ── Step 4: masked grouped GEMM w13 (fused gate+up): recv FP8 → bf16
        //    [E_local, m, 2*intermediate]. masked_m = recv_count (per expert).
        // SAFETY: all buffers are scratch sized for `[E_local, m, *]`; masked_m
        // bounds rows; sfa_aligned_m == m (TMA-aligned, asserted at scratch alloc).
        unsafe {
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
                p_recv_x,
                p_recv_sc,
                cache_ptr(&w13.weight, ctx),
                cache_ptr(&w13.scales, ctx),
                p_w13_out,
                p_masked,
                num_local_experts,
                m,
                2 * intermediate,
                hidden_dim,
                sfa_aligned_m,
                stream,
            )?;
            // ── Step 5: masked SwiGLU(clamp) + per-128-block FP8 requant →
            //    [E_local, m, intermediate] FP8 + column-major scales. The grid
            //    covers only `expected_m = min(global_tokens, m)` rows per
            //    expert (per-expert recv count ≤ the step's global token count;
            //    a full-band grid measured 631 µs/layer of empty-block drain at
            //    B=1 — 52.9% of deepep_ll GPU time).
            moe::dsv4_deepgemm_silu_mul_masked_quant(
                p_w13_out,
                p_act_fp8,
                p_act_sc,
                p_masked,
                num_local_experts,
                m,
                global_tokens.max(1).min(m),
                2 * intermediate,
                swiglu_limit,
                stream,
            )?;
            // ── Step 6: masked grouped GEMM w2 (down): act FP8 → bf16
            //    [E_local, m, hidden]. This IS the LL-combine input layout.
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
                p_act_fp8,
                p_act_sc,
                cache_ptr(&w2.weight, ctx),
                cache_ptr(&w2.scales, ctx),
                p_expert_out,
                p_masked,
                num_local_experts,
                m,
                hidden_dim,
                intermediate,
                sfa_aligned_m,
                stream,
            )?;
        }

        // ── Step 7: LL combine → this rank's owned routed output [hidden, owned_n].
        // The shared expert is NOT added here: the dsv4.rs forward adds the
        // replicated shared expert on the FULL gathered `moe_out` afterward
        // (identical to the intranode `dsv4_moe_forward_deepep` contract), so
        // adding it here would double-count. `out` holds routed-only.
        transport.ll_combine(
            ctx,
            scratch,
            out,
            &topk_idx_i64,
            &route_weights,
            owned_n,
            topk,
        )?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
#[allow(unused_imports)] // consumed by the Piece 2 model.rs DSv4 branch
pub(crate) use dsv4_gpu::{
    Dsv4MoeDecodeScratch, GroupedCache, build_grouped_cache, dsv4_moe_forward,
    dsv4_moe_forward_decode_graph, dsv4_shared_expert_forward,
};
#[cfg(feature = "deepep")]
pub(crate) use dsv4_gpu::{dsv4_moe_forward_deepep, dsv4_moe_forward_deepep_ll};
#[cfg(feature = "cuda")]
pub(crate) use gpu::{
    MoeForwardScratch, moe_forward, moe_forward_into, qwen35_decode_moe_graph_capturable,
};

#[cfg(test)]
mod tests {
    use super::*;
    use infer_moe::{MoeConfig, route};

    fn cfg(num_experts: usize, top_k: usize, norm: bool) -> MoeConfig {
        MoeConfig::qwen36(num_experts, top_k, norm, 8)
    }

    #[test]
    fn flatten_matches_route_token_major() {
        // 2 tokens, 4 experts, top-2: check the flat buffers mirror the
        // route → (token*topk + k) layout the kernels read.
        let c = cfg(4, 2, true);
        // token 0: experts 3,2 ; token 1: experts 0,1 (descending logit).
        let logits = vec![0.1f32, 0.2, 3.0, 4.0, 4.0, 3.0, 0.2, 0.1];
        let decisions = route(&logits, &[], &c).unwrap();
        let (idx, w) = flatten_routing(&decisions, c.top_k).unwrap();
        assert_eq!(idx.len(), 4);
        assert_eq!(w.len(), 4);
        // token 0 routes at flat slots [0,1]; token 1 at [2,3].
        assert_eq!(idx[0], decisions[0].experts[0].expert as i32);
        assert_eq!(idx[1], decisions[0].experts[1].expert as i32);
        assert_eq!(idx[2], decisions[1].experts[0].expert as i32);
        assert_eq!(idx[3], decisions[1].experts[1].expert as i32);
        assert_eq!(idx[0], 3);
        assert_eq!(idx[1], 2);
        assert_eq!(idx[2], 0);
        assert_eq!(idx[3], 1);
        // Weights mirror selection order and renormalize to sum 1 per token.
        let t0: f32 = w[0] + w[1];
        let t1: f32 = w[2] + w[3];
        assert!((t0 - 1.0).abs() < 1e-5, "token0 weights sum {t0}");
        assert!((t1 - 1.0).abs() < 1e-5, "token1 weights sum {t1}");
    }

    #[test]
    fn flatten_rejects_wrong_topk() {
        let c = cfg(4, 2, false);
        let logits = vec![0.1f32, 0.2, 3.0, 4.0];
        let decisions = route(&logits, &[], &c).unwrap();
        // Ask for the wrong topk → mismatch error.
        assert!(flatten_routing(&decisions, 3).is_err());
    }

    // ── DSv4 noaux_tc gate-bias plumbing (the Piece 3 router thread). ────────
    // `dsv4_moe_forward` reads the real per-expert correction bias and passes it
    // to `infer_moe::route` (not `&[]`). These CPU tests lock that the bias
    // changes selection while the weight stays the unbiased sqrtsoftplus score,
    // and that `flatten_routing` mirrors the selection token-major.

    fn dsv4_cfg(num_experts: usize, top_k: usize, scaling: f32) -> MoeConfig {
        MoeConfig::dsv4(num_experts, 1, top_k, scaling, 8)
    }

    #[test]
    fn dsv4_gate_bias_flips_selection_not_weight() {
        // 4 experts, top-1. Raw sqrtsoftplus scores rank expert 3 first; a big
        // positive bias on expert 0 must steer noaux_tc selection to expert 0,
        // but the emitted weight is expert 0's UNBIASED score × scaling.
        let c = dsv4_cfg(4, 1, 1.0);
        let logits = vec![0.1f32, 0.2, 0.3, 0.9];
        let no_bias = route(&logits, &[], &c).unwrap();
        assert_eq!(no_bias[0].experts[0].expert, 3, "raw top-1 is expert 3");

        let bias = vec![5.0f32, 0.0, 0.0, 0.0];
        let biased = route(&logits, &bias, &c).unwrap();
        assert_eq!(
            biased[0].experts[0].expert, 0,
            "bias steers noaux_tc selection to expert 0"
        );
        // Weight is the unbiased score of expert 0 (sum-normalized over the single
        // selected expert ⇒ ~1.0 × scaling), not the biased key.
        let (idx, w) = flatten_routing(&biased, c.top_k).unwrap();
        assert_eq!(idx, vec![0]);
        assert!(
            (w[0] - 1.0).abs() < 1e-4,
            "single-select weight ≈ 1.0, got {}",
            w[0]
        );
    }

    #[test]
    fn dsv4_routed_scaling_factor_threads_through() {
        // routed_scaling_factor multiplies the normalized weight; top-1 over a
        // single expert ⇒ weight ≈ scaling.
        let c = dsv4_cfg(4, 1, 2.5);
        let logits = vec![0.1f32, 0.2, 0.3, 0.9];
        let decisions = route(&logits, &[], &c).unwrap();
        let (_idx, w) = flatten_routing(&decisions, c.top_k).unwrap();
        assert!(
            (w[0] - 2.5).abs() < 1e-3,
            "weight ≈ routed_scaling 2.5, got {}",
            w[0]
        );
    }

    // ── DSv4 hash routing (`super::hash_route`). Hash layers pick experts from
    // the tid2eid table by token id (ignoring the router gate selection), then
    // weight them by the sqrtsoftplus router scores. cuda-gated: `deepseek_spec`
    // is a `cuda`-feature dep.
    #[cfg(feature = "cuda")]
    fn hash_test_config() -> deepseek_spec::DeepSeekV4Config {
        deepseek_spec::DeepSeekV4Config::from_json_str(
            r#"{
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4", "torch_dtype": "bfloat16",
            "vocab_size": 8, "hidden_size": 256, "num_hidden_layers": 3,
            "num_attention_heads": 8, "num_key_value_heads": 1, "head_dim": 512,
            "hidden_act": "silu", "swiglu_limit": 10.0, "q_lora_rank": 256,
            "o_lora_rank": 256, "o_groups": 8, "qk_rope_head_dim": 64,
            "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
            "moe_intermediate_size": 256, "routed_scaling_factor": 1.0,
            "norm_topk_prob": true, "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc", "index_n_heads": 8, "index_head_dim": 64,
            "index_topk": 64, "num_hash_layers": 3, "sliding_window": 64,
            "compress_ratios": [0, 4, 0], "compress_rope_theta": 160000.0,
            "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1.0e-6,
            "num_nextn_predict_layers": 0, "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "rope_scaling": {"type": "yarn", "factor": 16.0,
                "original_max_position_embeddings": 2048, "beta_fast": 32.0, "beta_slow": 1.0},
            "rms_norm_eps": 1.0e-6, "initializer_range": 0.02,
            "tie_word_embeddings": false, "attention_bias": false,
            "attention_dropout": 0.0, "bos_token_id": 0, "eos_token_id": 1
        }"#,
        )
        .unwrap()
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn hash_route_picks_tid2eid_experts_ignoring_logits() {
        let cfg = hash_test_config();
        // tid2eid: token 0 → experts [1,2]; token 1 → experts [3,0]. topk=2.
        let tid2eid: Vec<i64> = vec![1, 2, /*tok0*/ 3, 0 /*tok1*/];
        let tokens = vec![0u32, 1u32];
        // Logits rank expert 3 highest for both tokens; hash must IGNORE that
        // for selection (experts come from tid2eid), using logits only to weight.
        let logits = vec![
            0.1f32, 0.2, 0.3, 0.9, // token 0
            0.1, 0.2, 0.3, 0.9, // token 1
        ];
        let decisions = super::hash_route(&cfg, &tid2eid, &tokens, &logits).unwrap();
        assert_eq!(decisions.len(), 2);
        assert_eq!(
            decisions[0].expert_ids(),
            vec![1, 2],
            "token 0 from tid2eid"
        );
        assert_eq!(
            decisions[1].expert_ids(),
            vec![3, 0],
            "token 1 from tid2eid"
        );
        // Weights are positive normalized scores (each token's two weights sum ~1).
        let (idx, w) = flatten_routing(&decisions, cfg.num_experts_per_tok).unwrap();
        assert_eq!(idx, vec![1, 2, 3, 0]);
        assert!((w[0] + w[1] - 1.0).abs() < 1e-3, "token 0 weights sum ~1");
        assert!((w[2] + w[3] - 1.0).abs() < 1e-3, "token 1 weights sum ~1");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn hash_route_rejects_token_beyond_table() {
        let cfg = hash_test_config();
        // Only token 0's two experts are present; token 1 needs entries [2,3].
        let tid2eid: Vec<i64> = vec![0, 1];
        assert!(super::hash_route(&cfg, &tid2eid, &[0u32, 1u32], &[0.0; 8]).is_err());
    }

    // ── DeepGEMM m-grouped layout math (kernel-foundation #2). ──────────────
    // Host references mirror the device kernels (`moe_exclusive_scan_aligned_
    // i32`, `dsv4_fill_m_indices_from_counts` over a -1-prefilled buffer, the
    // `dsv4_pack_local_experts_with_slots` slot arithmetic) and the DeepGEMM
    // scheduler contract (vendor `deep_gemm/scheduler/gemm.cuh`,
    // `get_global_idx` for `GemmType::MGroupedContiguous`): the kernel
    // resolves the B-side expert group ONCE per BLOCK_M output tile from
    // `m_indices[m_block_idx * BLOCK_M]`, so a tile must never mix groups.

    const ALIGN: usize = super::DEEPGEMM_CONTIG_ALIGN;

    /// Mirror of `moe_exclusive_scan_aligned_i32_cuda`: exclusive prefix sum
    /// of per-group counts, each group's span padded up to `alignment`.
    fn aligned_scan_ref(counts: &[i32], alignment: i32) -> (Vec<i32>, i32) {
        let mut offsets = Vec::with_capacity(counts.len());
        let mut acc = 0i32;
        for &c in counts {
            offsets.push(acc);
            // (signed `div_ceil` is unstable; counts are non-negative)
            acc += (c + alignment - 1) / alignment * alignment;
        }
        (offsets, acc)
    }

    /// Mirror of -1-memset + `dsv4_fill_m_indices_from_counts_cuda`.
    fn fill_m_indices_ref(counts: &[i32], offsets: &[i32], cap: usize) -> Vec<i32> {
        let mut m = vec![-1i32; cap];
        for (g, (&c, &o)) in counts.iter().zip(offsets).enumerate() {
            for r in 0..c {
                let dst = (o + r) as usize;
                if dst < cap {
                    m[dst] = g as i32;
                }
            }
        }
        m
    }

    /// The MGroupedContiguous scheduler contract: within every BLOCK_M tile,
    /// every VALID row's group must equal the group read at the tile's first
    /// row (pad rows are -1 and excluded downstream).
    fn tile_contract_holds(m_indices: &[i32], block_m: usize) -> bool {
        m_indices.chunks(block_m).all(|tile| {
            let head = tile[0];
            tile.iter().all(|&g| g < 0 || g == head)
        })
    }

    #[test]
    fn aligned_scan_offsets_are_aligned_and_within_cap() {
        // Mixed counts incl. zero and exact-multiple groups.
        let counts = vec![3i32, 0, 130, 128, 1];
        let total_routes: i32 = counts.iter().sum();
        let (offsets, total) = aligned_scan_ref(&counts, ALIGN as i32);
        assert_eq!(offsets, vec![0, 128, 128, 384, 512]);
        assert_eq!(total, 640);
        for (g, &o) in offsets.iter().enumerate() {
            assert_eq!(o as usize % ALIGN, 0, "group {g} offset {o} unaligned");
            // Segments never overlap: offset + count <= next offset.
            let end = o + counts[g];
            let next = offsets.get(g + 1).copied().unwrap_or(total);
            assert!(
                end <= next,
                "group {g} segment [{o}, {end}) overlaps {next}"
            );
        }
        // The host launch cap bounds the device-side aligned total.
        let cap = super::deepgemm_contig_rows_cap(total_routes as usize, counts.len());
        assert!(
            total as usize <= cap,
            "aligned total {total} > host cap {cap}"
        );
        assert_eq!(cap % ALIGN, 0, "cap must stay 128-aligned for the bridge");
    }

    #[test]
    fn contig_rows_cap_bounds_worst_case_skew() {
        // Worst cases: all routes on one expert; one route per expert; empty.
        for (counts, experts) in [
            (vec![16384i32], 256usize),
            (vec![1i32; 256], 256),
            (vec![64i32; 256], 256),
            (vec![0i32; 256], 256),
        ] {
            let mut padded = vec![0i32; experts];
            padded[..counts.len()].copy_from_slice(&counts);
            let routes: i32 = padded.iter().sum();
            let (_, total) = aligned_scan_ref(&padded, ALIGN as i32);
            let cap = super::deepgemm_contig_rows_cap(routes.max(1) as usize, experts);
            assert!(
                total as usize <= cap,
                "aligned total {total} > cap {cap} for counts {padded:?}"
            );
        }
    }

    /// Item-6 contract test, POSITIVE: the 128-aligned layout keeps every
    /// BLOCK_M=128 tile single-group, so `m_indices[tile_start]` (what the
    /// kernel reads) is correct for every valid row of the tile.
    #[test]
    fn aligned_layout_satisfies_per_tile_group_contract() {
        let counts = vec![100i32, 100, 3, 0, 257];
        let (offsets, total) = aligned_scan_ref(&counts, ALIGN as i32);
        let cap = total as usize + ALIGN; // trailing all-pad tile stays -1
        let m = fill_m_indices_ref(&counts, &offsets, cap);
        assert!(tile_contract_holds(&m, ALIGN));
        // Pure-pad tiles (e.g. the trailing cap slack) read -1 at the head.
        assert_eq!(m[total as usize], -1);
    }

    /// Item-6 contract test, NEGATIVE: the COMPACT (unaligned) layout puts a
    /// group boundary inside a 128-row tile, so the kernel computes the tail
    /// rows of the tile against the WRONG expert's weights. This is the
    /// regression the DSv4 FP8 prefill path must not reintroduce.
    #[test]
    fn compact_layout_violates_per_tile_group_contract() {
        let counts = vec![100i32, 100];
        // Plain (compact) exclusive scan: offsets [0, 100].
        let offsets = vec![0i32, 100];
        let m = fill_m_indices_ref(&counts, &offsets, 200);
        assert!(
            !tile_contract_holds(&m, ALIGN),
            "compact layout must violate the per-tile contract (rows 100..128 \
             are group 1 inside a tile whose head row is group 0)"
        );
    }

    #[test]
    fn dsv4_fp8_prefill_rows_cap_keeps_pad_rows_invalid() {
        let counts = vec![100i32, 100];
        let total_routes: i32 = counts.iter().sum();
        let (offsets, aligned_total) = aligned_scan_ref(&counts, ALIGN as i32);
        let rows_cap = super::deepgemm_contig_rows_cap(total_routes as usize, counts.len());
        let m = fill_m_indices_ref(&counts, &offsets, rows_cap);

        assert_eq!(offsets, vec![0, 128]);
        assert!(aligned_total as usize <= rows_cap);
        assert!(tile_contract_holds(&m, ALIGN));
        assert!(
            m[100..128].iter().all(|&g| g == -1),
            "rows between expert 0 and expert 1 must be invalid pads"
        );
        assert!(
            m[228..rows_cap].iter().all(|&g| g == -1),
            "rows after the aligned total must be invalid pads"
        );
    }

    /// Pad rows are excluded from the combine: the pack writes a route slot
    /// for the R real rows only, every pad row keeps the -1 sentinel the
    /// scatter skips, and each route lands in exactly one packed row.
    #[test]
    fn pad_rows_keep_sentinel_and_every_route_packs_once() {
        let counts = vec![2i32, 0, 3];
        let (offsets, total) = aligned_scan_ref(&counts, ALIGN as i32);
        let cap = total as usize;
        // Mirror the pack: slot = offsets[g] + cursor[g]++ per route.
        let mut route_slot = vec![-1i32; cap];
        let mut cursors = vec![0i32; counts.len()];
        let routes: Vec<usize> = vec![0, 2, 2, 2, 0]; // route -> expert
        for (route, &g) in routes.iter().enumerate() {
            let slot = (offsets[g] + cursors[g]) as usize;
            cursors[g] += 1;
            route_slot[slot] = route as i32;
        }
        // Scatter-skip semantics: only slot >= 0 rows reach route_out.
        let written: Vec<i32> = route_slot.iter().copied().filter(|&s| s >= 0).collect();
        assert_eq!(
            written.len(),
            routes.len(),
            "every route packs exactly once"
        );
        let mut sorted = written.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
        // Every pad row (outside [offset, offset+count)) keeps the sentinel.
        for (row, &slot) in route_slot.iter().enumerate() {
            let in_segment = counts
                .iter()
                .zip(&offsets)
                .any(|(&c, &o)| row >= o as usize && row < (o + c) as usize);
            assert_eq!(
                slot >= 0,
                in_segment,
                "row {row}: packed rows and segment rows must coincide"
            );
        }
    }

    // ── Decode-band weight-read-bound kernel dispatch + grid mirrors. ───────

    /// The decode band must never shadow the licensed DeepGEMM prefill
    /// dispatch, and must cover the whole batched-decode envelope
    /// (`R = top_k(8) × B` for B ≤ 32) on 16B-vector-aligned chunks.
    #[test]
    fn decode_band_sits_below_deepgemm_floor_and_covers_batched_decode() {
        const MAX: usize = super::QWEN35_MOE_DECODE_MAX_ROUTES;
        assert!(
            MAX < super::QWEN35_DEEPGEMM_MIN_ROUTES,
            "decode band {MAX} must stay below the DeepGEMM floor {}",
            super::QWEN35_DEEPGEMM_MIN_ROUTES
        );
        // Batched decode R = 8B; the band covers B up to 32 slots.
        assert_eq!(MAX, 8 * 32, "decode band is the batched-decode envelope");
        // grid.y chunking is ACT_TILE=8-row; the band is whole chunks.
        assert_eq!(MAX % 8, 0, "decode band must be whole 8-row chunks");
    }

    /// Host mirror of the `moe_bf16_grouped_gemm_{swiglu_,}decode` launch
    /// grid: `grid.y = ceil(max_count / ACT_TILE)` chunks over per-expert
    /// `offsets/counts`. Every routed row must be covered by exactly one
    /// (expert, chunk) block — chunks beyond an expert's count early-exit,
    /// counts above ACT_TILE span multiple chunks (the extra weight passes),
    /// and zero-count experts contribute nothing.
    #[test]
    fn decode_kernel_chunk_grid_covers_every_route_exactly_once() {
        const ACT_TILE: usize = 8;
        // Mixed per-expert counts: zero, sub-tile, exact-tile, multi-chunk
        // (count 32 = B_max under the R<=256 band), odd remainder.
        let counts = [1usize, 0, 8, 3, 32, 0, 9];
        let total: usize = counts.iter().sum();
        let mut offsets = vec![0usize; counts.len()];
        let mut acc = 0usize;
        for (g, &c) in counts.iter().enumerate() {
            offsets[g] = acc;
            acc += c;
        }
        // Launcher: max_count = total_routes upper bound.
        let chunks = total.max(1).div_ceil(ACT_TILE);
        let mut touched = vec![0u32; total];
        for (e, &c) in counts.iter().enumerate() {
            for chunk in 0..chunks {
                let chunk_base = chunk * ACT_TILE;
                if chunk_base >= c {
                    continue; // kernel early-exit: chunk_base >= counts[e]
                }
                let tile = (c - chunk_base).min(ACT_TILE);
                for b in 0..tile {
                    touched[offsets[e] + chunk_base + b] += 1;
                }
            }
        }
        for (row, &t) in touched.iter().enumerate() {
            assert_eq!(t, 1, "packed row {row} covered {t} times (want exactly 1)");
        }
    }

    /// Masked-band dispatch threshold: R <= 128 is the only host-provable
    /// bound on any single group's count (counts are device-resident), and it
    /// keeps every pack slot inside its group's 128-row band.
    #[test]
    fn masked_band_threshold_bounds_per_group_counts() {
        const BAND: i32 = 128;
        // Worst skew at the threshold: all R = 128 routes on one expert.
        let counts = vec![128i32, 0, 0];
        let total: i32 = counts.iter().sum();
        assert!(total <= BAND);
        for (g, &c) in counts.iter().enumerate() {
            assert!(c <= BAND, "group {g} count {c} exceeds the band");
            // Band pack slot arithmetic: g * BAND + cursor < (g + 1) * BAND.
            let last_slot = g as i32 * BAND + (c - 1).max(0);
            assert!(last_slot < (g as i32 + 1) * BAND);
        }
    }
}
