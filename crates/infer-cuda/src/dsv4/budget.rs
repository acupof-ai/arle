//! DSv4 capacity: the MLA latent KV arena shape and the joint
//! `(num_slots, pool_tokens)` budget solve. Split out of `dsv4.rs` — sizing is
//! self-contained and runs BEFORE any slot or `kv_adapter` exists.

use anyhow::{Result, bail, ensure};
use deepseek_spec::{DeepSeekV4AttentionMode, DeepSeekV4Config};

use super::{Dsv4Model, Dsv4SlotState, MAX_SPEC_DRAFT_DEPTH, MAX_SPEC_VERIFY_ROWS};

/// Unlike the per-head BF16 [`cuda_kernels::prelude::PagedKVPool`], MLA caches a
/// single compressed latent per token in the flat FP8 block layout FlashMLA's
/// sparse-decode consumes: `[NoPE | RoPE]` packed to `bytes_per_token` bytes
/// (`cuda-kernels/src/attention.rs` `dsv4_fp8_kv_pack`, 584 B/token for the
/// canonical NoPE=448 / RoPE=64 / head_dim=512 shape).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Dsv4MlaKvArena {
    /// RoPE-carrying dims (`qk_rope_head_dim`, 64 for DSv4-Flash).
    pub rope_dim: usize,
    /// NoPE latent dims (`head_dim - qk_rope_head_dim`, 448 for DSv4-Flash).
    pub nope_dim: usize,
    /// FlashMLA paged block size (`page_block_size`, 64 for DSv4-Flash MODEL1).
    pub page_block_size: usize,
    /// Packed bytes per token in the FP8 arena (NoPE FP8 + RoPE bf16 + e8m0).
    pub bytes_per_token: usize,
    pub num_layers: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Dsv4KvBudgetPlan {
    pub(crate) num_slots: usize,
    /// Shared FlashMLA comp token capacity the demand-paged layer pools are
    /// sized for (#154 Phase 3b) — the engine's admission page count is
    /// `flashmla_pool_tokens / page_block_size`. For identity (V32) models
    /// this is bookkeeping only (`num_slots × max_seq_len`).
    pub(crate) flashmla_pool_tokens: usize,
}

/// Packed bytes per token the FlashMLA sparse-FP8 decode reads for the canonical
/// MODEL1 NoPE=448 / RoPE=64 shape (`dsv4_fp8_kv_pack` doc):
/// 448 fp8 NoPE + 128 bf16 RoPE + 8 e8m0 scales = 584.
const DSV4_FLASH_KV_BYTES_PER_TOKEN: usize = 584;
/// V32 / GLM shape NoPE=512 (= `kv_lora_rank`) / RoPE=64:
/// 512 fp8 NoPE + 128 bf16 RoPE + 16 scales = 656 (matches
/// `arle_flashmla_decode_shim.cu:62-63` `V32_BYTES_PER_TOKEN`).
const DSV4_V32_KV_BYTES_PER_TOKEN: usize = 656;
const DSV4_FLASH_PAGE_BLOCK_SIZE: usize = 64;

impl Dsv4MlaKvArena {
    pub(super) fn from_config(config: &DeepSeekV4Config) -> Result<Self> {
        let rope_dim = config.qk_rope_head_dim;
        let nope_dim = config
            .head_dim
            .checked_sub(rope_dim)
            .filter(|&d| d > 0)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "DSv4 head_dim {} must exceed qk_rope_head_dim {rope_dim}",
                    config.head_dim
                )
            })?;
        // Two FP8 pack/decode layouts are wired (`arle_flashmla_decode_shim.cu`):
        //   • MODEL1: NoPE=448 / RoPE=64 → 584 B/token (DSv4-Flash, pre-absorbed).
        //   • V32:    NoPE=512 / RoPE=64 → 656 B/token (GLM-5.2, NoPE == kv_lora_rank).
        // GPU-UNVERIFIABLE (pod): the V32 *pack* kernel emitting the 656 B/token /
        // 512-NoPE layout is wired in Tranche D — this arena only sizes the buffer.
        let bytes_per_token = match (nope_dim, rope_dim) {
            (448, 64) => DSV4_FLASH_KV_BYTES_PER_TOKEN,
            // V32: NoPE latent is the full kv_lora_rank (512 for GLM-5.2).
            (n, 64) if n == config.kv_lora_rank && config.kv_lora_rank > 0 => {
                DSV4_V32_KV_BYTES_PER_TOKEN
            }
            _ => bail!(
                "DSv4 MLA KV arena wires only the FlashMLA MODEL1 NoPE=448/RoPE=64 \
                 (584 B/token) or V32 NoPE={}=kv_lora_rank/RoPE=64 (656 B/token) packs, \
                 got NoPE={nope_dim} RoPE={rope_dim} kv_lora_rank={}",
                config.kv_lora_rank,
                config.kv_lora_rank
            ),
        };
        Ok(Self {
            rope_dim,
            nope_dim,
            page_block_size: DSV4_FLASH_PAGE_BLOCK_SIZE,
            bytes_per_token,
            num_layers: config.num_hidden_layers,
        })
    }
}

impl Dsv4Model {
    pub(crate) fn new_kv_adapter(
        &self,
        max_seq_len: usize,
        budget: Dsv4KvBudgetPlan,
    ) -> Result<crate::attention::Dsv4KvAdapter> {
        let specs: Vec<_> = self
            .layers
            .iter()
            .map(|layer| {
                let local_width = layer.attention.wq_b.rows;
                ensure!(
                    local_width.is_multiple_of(self.config.head_dim),
                    "DSv4 attention pool local width {local_width} is not a multiple of head_dim {}",
                    self.config.head_dim
                );
                Ok((layer.mode, layer.compress_ratio, local_width / self.config.head_dim))
            })
            .collect::<Result<Vec<_>>>()?;
        let mla_decode: Vec<_> = self
            .layers
            .iter()
            .map(|layer| {
                let model1 = layer.attention.w_kc.is_none()
                    && layer.attention.w_vc.is_none()
                    && layer.attention.o_proj.is_none();
                model1
                    .then(|| {
                        crate::attention::Dsv4MlaDecodeScratch::new(
                            &self.ctx,
                            &self.config,
                            &layer.attention,
                            layer.mode,
                            layer.compress_ratio,
                        )
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        crate::attention::Dsv4KvAdapter::new(
            &self.ctx,
            &self.config,
            &specs,
            max_seq_len,
            &self.kv_arena,
            self.tp.config().world_size,
            budget.num_slots,
            budget.flashmla_pool_tokens,
            mla_decode,
            self.layers.iter().find_map(|layer| layer.moe.as_ref()),
            self.split.experts_per_rank,
            self.config.hidden_size,
        )
    }

    pub(crate) fn new_slot_state(
        &self,
        max_seq_len: usize,
        slot_idx: usize,
        kv_adapter: &crate::attention::Dsv4KvAdapter,
    ) -> Result<Dsv4SlotState> {
        Dsv4SlotState::new(self, max_seq_len, slot_idx, kv_adapter)
    }

    /// Verify rows a slot can ever be asked for: DSpark verifies its block plus
    /// the anchor, MTP a depth-clamped chain plus the anchor
    /// ([`crate::executor::spec_decode`] clamps to `MAX_SPEC_DRAFT_DEPTH`).
    /// `MAX_SPEC_VERIFY_ROWS` is the validation ceiling, not the allocation
    /// width — sizing the per-slot scratch at it booked ~300MB/slot nothing
    /// touches, and capped DSpark at 22 slots (#184).
    pub(crate) fn spec_verify_rows(&self) -> usize {
        let rows = if self.config.is_dspark() {
            self.config.dspark_block_size + 1
        } else {
            MAX_SPEC_DRAFT_DEPTH + 1
        };
        rows.min(MAX_SPEC_VERIFY_ROWS)
    }

    pub(crate) fn per_slot_device_bytes(&self, max_seq_len: usize) -> Result<usize> {
        let bf16 = std::mem::size_of::<half::bf16>();
        let rows = self.spec_verify_rows();
        let hidden = self.config.hidden_size;
        let stream_dim = hidden * self.config.hc_mult;
        let n = self.layers.len();
        // attention(per-layer): Σ over layers + the start_pos scalar. The
        // per-component sum is logged once: the total alone cannot say whether a
        // buffer is worth quantizing, which is the number every KV-precision
        // question has needed and nobody had.
        let mut total = std::mem::size_of::<i32>();
        let mut per_component: std::collections::BTreeMap<&'static str, usize> =
            std::collections::BTreeMap::new();
        for layer in &self.layers {
            for (name, bytes) in
                crate::attention::Dsv4LayerAttentionState::device_bytes_for_breakdown(
                    &self.config,
                    layer.mode,
                    layer.compress_ratio,
                    max_seq_len,
                )?
            {
                total = total.saturating_add(bytes);
                *per_component.entry(name).or_default() += bytes;
            }
        }
        if !per_component.is_empty() {
            log::info!(
                "[dsv4-slot-ledger] max_seq_len={max_seq_len} attn_total={:.1}MB {}",
                total as f64 / (1024.0 * 1024.0),
                per_component
                    .iter()
                    .map(|(k, v)| format!("{k}={:.1}MB", *v as f64 / (1024.0 * 1024.0)))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        if self.spec_decode_on {
            let ring = crate::attention::Dsv4SpecRingSnapshot::device_bytes_for(
                &self.config,
                &self.kv_arena,
            )?;
            total = total.saturating_add(n.saturating_mul(ring));
            total = total.saturating_add(n.saturating_mul(hidden * rows * bf16));
            // spec_verify: embeddings + initial_stream, then per layer 7 row-major
            // temporaries (5 hidden-wide + 2 stream-wide), all `rows` columns. This
            // is the dominant per-slot term the old budget missed entirely.
            let verify_scratch = (hidden + stream_dim) * rows * bf16;
            let verify_per_layer = (5 * hidden + 2 * stream_dim) * rows * bf16;
            total = total
                .saturating_add(verify_scratch)
                .saturating_add(n.saturating_mul(verify_per_layer));
        }
        // DSpark adds per-slot taps, latent KV, and one ratio-0 attention state
        // per stage. Keep this aligned with `load_dspark_exec`.
        if self.config.is_dspark() {
            let head_dim = self.config.head_dim;
            let num_stages = self.config.dspark_num_stages();
            let block = self.config.dspark_block_size;
            let draft_span = self.config.sliding_window + block;
            let latent_cap = draft_span;
            total = total.saturating_add(
                self.config
                    .dspark_target_layer_ids
                    .len()
                    .saturating_mul(stream_dim)
                    .saturating_mul(block.saturating_add(1))
                    .saturating_mul(bf16),
            );
            total = total.saturating_add(
                num_stages
                    .saturating_mul(latent_cap)
                    .saturating_mul(head_dim)
                    .saturating_mul(bf16),
            );
            let stage_mode = self.config.attention_mode_for_compress_ratio(0);
            let stage_bytes = crate::attention::Dsv4LayerAttentionState::device_bytes_for(
                &self.config,
                stage_mode,
                0,
                draft_span,
            )?;
            total = total.saturating_add(num_stages.saturating_mul(stage_bytes));
        }
        Ok(total)
    }

    /// Clamp `requested` decode slots to what the KV budget affords, from
    /// `cudaMemGetInfo() × MEM_FRACTION ÷ per-slot KV bytes`. This is the dynamic-mem-budget
    /// fix for the c=32 OOM CRASH (root cause: a fixed `num_slots` whose arena alloc OOMs at
    /// high concurrency × long `max_seq_len`). The per-slot cost is an itemized
    /// ledger: the EXACT FP8 arena term (`max_seq_len × bytes_per_token ×
    /// num_layers`) scaled ×2 to cover compressor/SW per-slot buffers + forward
    /// activations, PLUS the official-DSA indexer scratch (one
    /// `Dsv4DsaOfficialState` per CSA layer per slot — its `logits` tile scales
    /// with `max_seq/cr` and dwarfs the arena at long context; un-budgeted it
    /// OOMs engine build at 256K, issue #67).
    ///
    /// Cross-rank consistency: per-rank `mem_get_info` is NOT guaranteed identical
    /// (allocator state differs per rank), and the clamped count feeds the
    /// scheduler's slot gate — any per-rank divergence in scheduler-visible
    /// capacity diverges the deterministic planner and deadlocks NCCL. The local
    /// affordable count is therefore NCCL min-reduced; every rank calls this at
    /// the same construction point (collective). A rank that cannot query its
    /// memory contributes `i32::MAX` (does not bind) instead of skipping the
    /// collective.
    /// Peak transient working set of ONE prefill chunk, itemized from the
    /// allocation sites. `DSV4_PREFILL_QUERY_CHUNK` (4096) is the ceiling of
    /// `chunked_prefill_size` (`loaded.rs` clamps the flag to [128, 4096]), so
    /// this bounds any admissible chunk. Terms:
    /// - MoE masked-tail (`moe/dsv4.rs` `dsv4_moe_forward_masked_tail` +
    ///   `deepgemm_grouped_experts`): packed_hidden [H, rows] bf16, activation
    ///   FP8 copy [rows, H], w13_out [2I, rows] bf16, act FP8 [rows, I],
    ///   out_compact [H, rows] bf16, route_out [H, routes] bf16 — where
    ///   routes = chunk×top_k and rows is the 128-aligned contiguous cap.
    /// - Attention (`attention.rs` `mla_attention_prepare`/`_fwd`): normed +
    ///   attn_out [H, chunk], q_prepared + local_attn [local_width, chunk],
    ///   O-LoRA latent [o_lora_rank, chunk], all bf16.
    ///
    /// Freed buffers go back to the retained mempool between layers, so the
    /// peak is one layer's live set, not a sum over layers.
    fn prefill_transient_reserve_bytes(&self) -> usize {
        const BF16: usize = 2;
        let chunk = crate::attention::DSV4_PREFILL_QUERY_CHUNK;
        let h = self.config.hidden_size;
        let i_dim = self.config.moe_intermediate_size;
        let routes = chunk.saturating_mul(self.moe_config.top_k);
        let rows = crate::moe::deepgemm_contig_rows_cap(
            routes.max(1),
            self.split.experts_per_rank,
            crate::moe::DEEPGEMM_CONTIG_ALIGN,
        );
        let local_width = (self.config.num_attention_heads / self.tp.config().world_size.max(1))
            .saturating_mul(self.config.head_dim);
        let moe = rows.saturating_mul(h * BF16)        // packed_hidden
            + rows.saturating_mul(h)                   // input_fp8
            + rows.saturating_mul(2 * i_dim * BF16)    // w13_out
            + rows.saturating_mul(i_dim)               // act_fp8
            + rows.saturating_mul(h * BF16)            // out_compact
            + routes.saturating_mul(h * BF16); // route_out
        let attn = chunk.saturating_mul(2 * h * BF16)  // normed + attn_out
            + chunk.saturating_mul(2 * local_width * BF16) // q_prepared + local_attn
            + chunk.saturating_mul(self.config.o_lora_rank * BF16); // latent
        moe.saturating_add(attn)
    }

    pub(crate) fn kv_budget_plan(
        &self,
        requested: usize,
        max_seq_len: usize,
        extra_per_slot_bytes: usize,
    ) -> Result<Dsv4KvBudgetPlan> {
        // NO term here for the prefix-state pool (#154 Phase 2): it is
        // host-DRAM-resident by design (`attention/prefix_state.rs`), funded
        // by the --kv-dram share — HBM sizes slots, DRAM sizes pool heat.
        const MEM_FRACTION: f64 = 0.9;
        // Official-DSA selector memory splits into the ONE model-wide shared
        // scratch (a fixed subtraction from the budget) and the per-(slot,
        // CSA-layer) transient rotated_keys staging (a per-slot term). #67.
        let official_on = true;
        // First indexer layer's indexer ratio (CSA = compress_ratio; GLM
        // SparseIndexed → 1, full-sequence every-token-a-key). Widened from CSA-only
        // so the shared DSA scratch + batched per-slot scratch are budgeted for GLM.
        let idx_cr = self
            .layers
            .iter()
            .find(|layer| layer.mode.has_indexer())
            .map(|layer| {
                if layer.mode == DeepSeekV4AttentionMode::SparseIndexed {
                    1
                } else {
                    layer.compress_ratio
                }
            });
        let dsa_shared_bytes: usize = match (official_on, idx_cr) {
            (true, Some(cr)) => {
                crate::attention::dsv4_dsa_shared_scratch_bytes(&self.config, cr, max_seq_len)
            }
            _ => 0,
        };
        // ONE model-wide FP32 compressor-probe scratch (hoisted off the per-slot
        // compressor state) — a fixed subtraction, mirroring `dsa_shared_bytes`.
        // 0 when the model has no compressor layer (matches the adapter's gate).
        let compressor_fp32_bytes = crate::attention::Dsv4CompressorFp32Scratch::device_bytes_for(
            crate::attention::dsv4_compressor_fp32_max_width(
                &self.config,
                self.layers.iter().map(|l| (l.mode, l.compress_ratio)),
            ),
            max_seq_len,
        );
        // Eager B=1 uses the FP8 decode-band MoE lane, so no shared routed-MoE
        // scratch is allocated; only the shared-expert output below counts.
        // The model-wide shared-expert output is allocated unconditionally on
        // the adapter (#60). It is sized for the bounded MTP verify chunk and
        // reused by B=1 decode with `seq_len = 1`; count it as a fixed term
        // regardless of the GPU-router path.
        let shared_expert_out_bytes = self
            .config
            .hidden_size
            .saturating_mul(MAX_SPEC_VERIFY_ROWS)
            .saturating_mul(std::mem::size_of::<half::bf16>());
        let moe_decode_shared_bytes = shared_expert_out_bytes;
        let shared_expert_scratch_bytes = self
            .layers
            .iter()
            .find_map(|layer| layer.moe.as_ref())
            .map(|layer| {
                crate::moe::Dsv4SharedDecodeScratch::device_bytes(
                    layer.hidden_dim,
                    layer.shared_w2.cols,
                )
            })
            .unwrap_or(0);
        // Compact-FP8 MoE tail scratch (launch-bound Step 1) is allocated on the
        // adapter whenever the model has a MoE layer; count it as a fixed term so
        // the KV pool sizing doesn't over-commit → OOM.
        let moe_tail_scratch_bytes = self
            .layers
            .iter()
            .find_map(|layer| layer.moe.as_ref())
            .map(|layer| {
                crate::moe::Dsv4MoeTailScratch::device_bytes(
                    layer.hidden_dim,
                    layer.intermediate,
                    self.split.experts_per_rank,
                )
            })
            .unwrap_or(0);
        let mla_decode_bytes: usize = self
            .layers
            .iter()
            .map(|layer| {
                crate::attention::Dsv4MlaDecodeScratch::device_bytes_for(
                    &self.config,
                    &layer.attention,
                    layer.mode,
                )
            })
            .sum();
        // TRUE per-slot cost = the slot struct itself (statically — no slot exists
        // yet). Fixes the 43→382 MB under-count — the old hand-roll missed
        // spec_verify, the dominant term — that ran `affordable` ~9× high.
        let slot_state_bytes = self.per_slot_device_bytes(max_seq_len)?;
        // Per-(slot,CSA-layer) DSA key-cache band lives in `Dsv4LayerKvLayout`
        // (`dsa_slot_bytes × num_slots`), NOT the slot struct, so it is a per-slot
        // budget term on top of `slot_state_bytes`. Official-gated to match the
        // pool alloc gate (`dsv4_dsa_key_cache_bytes`; index_ratio=1 for GLM).
        let mut dsa_key_cache_per_slot: usize = 0;
        if official_on {
            for layer in &self.layers {
                if !layer.mode.has_indexer() {
                    continue;
                }
                let index_ratio = if layer.mode == DeepSeekV4AttentionMode::SparseIndexed {
                    1
                } else {
                    layer.compress_ratio
                };
                dsa_key_cache_per_slot = dsa_key_cache_per_slot.saturating_add(
                    crate::attention::dsv4_dsa_key_cache_bytes(
                        &self.config,
                        index_ratio,
                        max_seq_len,
                    )
                    .unwrap_or(0),
                );
            }
        }
        // The N-row batched-decode DSA scratch (`*_batch` buffers inside the ONE
        // shared scratch) is sized by `decode_max_batch == num_slots`, so it is a
        // per-SLOT cost (NOT a fixed subtraction — that would be circular, since
        // num_slots is derived from this budget). One term per slot (the scratch
        // is one shared instance, not per CSA layer), gated on a CSA layer existing.
        let dsa_batched_per_slot = match (official_on, idx_cr) {
            (true, Some(cr)) => crate::attention::dsv4_dsa_batched_scratch_bytes_per_slot(
                &self.config,
                cr,
                max_seq_len,
            ),
            _ => 0,
        };
        let per_slot = slot_state_bytes
            .saturating_add(dsa_key_cache_per_slot)
            .saturating_add(dsa_batched_per_slot)
            .saturating_add(extra_per_slot_bytes);
        let (affordable_local, budget_bytes_local): (i32, usize) =
            match cudarc::driver::result::mem_get_info() {
                Ok((free, _total)) => {
                    let fixed_without_pool = dsa_shared_bytes
                        .saturating_add(compressor_fp32_bytes)
                        .saturating_add(moe_decode_shared_bytes)
                        .saturating_add(shared_expert_scratch_bytes)
                        .saturating_add(moe_tail_scratch_bytes)
                        .saturating_add(mla_decode_bytes);
                    let budget_before_pool = infer_seam::SlotBudget::from_free(
                        free,
                        MEM_FRACTION,
                        fixed_without_pool,
                        per_slot.max(1),
                    );
                    let affordable = budget_before_pool
                        .affordable()
                        .map_or(i32::MAX, |n| i32::try_from(n).unwrap_or(i32::MAX));
                    let slots_cap = requested.max(1);
                    let reserved_for_slots =
                        per_slot.saturating_mul(slots_cap.min(affordable.max(0) as usize));
                    let pool_budget_total = budget_before_pool
                        .budget_bytes
                        .saturating_sub(reserved_for_slots);
                    log::info!(
                        "DSv4 KV budget: free {}MB, per_slot {}MB (slot-state {}MB + DSA key-cache {}MB + DSA batched {}MB; \
                         FP8 arena in shared pool), shared DSA {}MB, shared compressor FP32 {}MB, shared MoE decode {}MB, \
                         shared expert scratch {}MB, shared MLA decode {}MB, pool_total {}MB, affordable {}",
                        free >> 20,
                        per_slot >> 20,
                        slot_state_bytes >> 20,
                        dsa_key_cache_per_slot >> 20,
                        dsa_batched_per_slot >> 20,
                        dsa_shared_bytes >> 20,
                        compressor_fp32_bytes >> 20,
                        moe_decode_shared_bytes >> 20,
                        shared_expert_scratch_bytes >> 20,
                        mla_decode_bytes >> 20,
                        pool_budget_total >> 20,
                        affordable,
                    );
                    (affordable, budget_before_pool.budget_bytes)
                }
                Err(_) => (i32::MAX, 0),
            };
        let affordable =
            self.tp
                .all_reduce_min_scalar_i32(&self.ctx, affordable_local)? as usize;
        // Reduce the pre-reservation budget (not a per-layer share, and not
        // the post-reservation remainder): the joint (num_slots, pool_tokens)
        // solve below needs the budget at ARBITRARY slot counts, and the
        // actual reservation (`per_slot × planned`) is rank-identical once
        // `planned` derives from reduced scalars. Reduced in MiB — a byte
        // count saturates the i32 collective at 2047MB (the old
        // pool_budget_total reduce silently did exactly that).
        let budget_bytes = (self.tp.all_reduce_min_scalar_i32(
            &self.ctx,
            i32::try_from((budget_bytes_local >> 20).min(i32::MAX as usize)).unwrap_or(i32::MAX),
        )? as usize)
            << 20;
        // Prefill-transient reserve: one chunk's peak working set, itemized
        // from the allocation sites, subtracted BEFORE the slot solve. Without
        // it the solve hands every budget byte to slot state + pool and the
        // first admitted long prefill OOMs mid-serve (18 slots x 1049MB filled
        // a 19.7GB budget; c=8 died at CUDA_ERROR_OUT_OF_MEMORY, 2026-08-24).
        // Deterministic from config ⇒ rank-identical, no reduce needed.
        let prefill_reserve = self.prefill_transient_reserve_bytes();
        let budget_bytes = budget_bytes.saturating_sub(prefill_reserve);
        log::info!(
            "DSv4 KV budget: prefill-transient reserve {}MB (chunk {} tokens)",
            prefill_reserve >> 20,
            crate::attention::DSV4_PREFILL_QUERY_CHUNK,
        );
        // Reject-below-fixed guard (parity with Metal's fits_fixed): a
        // cross-rank-min affordable of 0 means post-weights free VRAM cannot
        // hold even one slot's KV arena + selector/compressor state at this
        // max_seq_len. Fail closed uniformly — every rank branches on the same
        // reduced scalar, so this is lockstep-safe — instead of admitting one
        // slot (the former `max(1)`) and OOMing at arena allocation.
        anyhow::ensure!(
            affordable > 0,
            "DSv4 KV budget rejected startup: post-weights free VRAM affords 0 slots at \
             max_seq_len {max_seq_len} (per_slot ~{}MB + shared DSA {}MB + shared MoE decode {}MB \
             + shared expert scratch {}MB + shared MLA decode {}MB exceed {MEM_FRACTION} of free). Lower --max-total-tokens or free VRAM.",
            per_slot >> 20,
            dsa_shared_bytes >> 20,
            moe_decode_shared_bytes >> 20,
            shared_expert_scratch_bytes >> 20,
            mla_decode_bytes >> 20,
        );
        // The `affordable` gate above only covers PER-SLOT costs; the shared
        // FlashMLA pool (`pool_budget_total`, the coherent remainder after
        // those) separately needs at least one slot's fixed band **summed
        // across every layer** (`kv_layout.rs`'s `flashmla_slot_pages` is a
        // per-layer page count — a SlidingWindow-only layer needs far fewer
        // pages than a CompressedSparse one, so summing real per-layer need
        // replaces the old `.max()` + uniform-divide, which either starved
        // the big layers or wasted budget on the small ones). NOT covered by
        // the per-slot reservation above. Pod-verified 2026-07-06: without
        // this check, a `max_seq_len` that clears `affordable > 0` can still
        // be too large for the pool's own band, and the mismatch
        // surfaces as a hard `ensure!` panic in `kv_layout.rs` that crashes
        // every worker rank instead of this same clean startup rejection.
        let flashmla_page_bytes = self
            .kv_arena
            .page_block_size
            .saturating_mul(self.kv_arena.bytes_per_token);
        let per_layer_flashmla_pages: Vec<usize> = self
            .layers
            .iter()
            .map(|layer| {
                crate::attention::dsv4_flashmla_slot_pages(
                    &self.config,
                    layer.mode,
                    layer.compress_ratio,
                    max_seq_len,
                    self.kv_arena.page_block_size,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let total_slot_pages: usize = per_layer_flashmla_pages.iter().sum();
        let flashmla_band_bytes_per_slot = total_slot_pages.saturating_mul(flashmla_page_bytes);
        let demand_paged =
            crate::attention::dsv4_flashmla_demand_paged(&self.config) && total_slot_pages > 0;
        if !demand_paged {
            // Identity (V32) bands: the pool draws a FULL fixed band per slot
            // up front, so concurrency is ALSO capped by how many whole bands
            // the post-reservation remainder affords (pod-verified
            // 2026-07-06: without this, a 2-request concurrent burst
            // exhausted the pool mid-serve).
            let pool_budget_total = budget_bytes
                .saturating_sub(per_slot.saturating_mul(requested.max(1).min(affordable)));
            anyhow::ensure!(
                total_slot_pages == 0 || flashmla_band_bytes_per_slot <= pool_budget_total,
                "DSv4 KV budget rejected startup: the shared FlashMLA pool's \
                 remainder ({}MB) cannot hold even one slot's band across all \
                 layers at max_seq_len {max_seq_len} ({total_slot_pages} pages, \
                 {}MB). Lower --max-total-tokens or free VRAM.",
                pool_budget_total >> 20,
                flashmla_band_bytes_per_slot >> 20,
            );
            let pool_affordable_slots = if flashmla_band_bytes_per_slot == 0 {
                usize::MAX
            } else {
                infer_seam::SlotBudget::from_limit(
                    pool_budget_total,
                    0,
                    flashmla_band_bytes_per_slot,
                )
                .affordable()
                .unwrap_or(usize::MAX)
            };
            let affordable = affordable.min(pool_affordable_slots);
            let (planned, clamped) = infer_seam::clamp_to_affordable(requested, affordable);
            if clamped {
                log::warn!(
                    "DSv4 KV budget: requested {requested} slots × ~{}MB/slot (slot-state {}MB + DSA key-cache/batched; \
                     FP8 arena out of divisor) exceeds the cross-rank-min affordable {affordable} \
                     (local affordable {affordable_local}, pool-band-affordable {pool_affordable_slots}, \
                     {MEM_FRACTION} of post-weights free); clamping num_slots to {affordable}.",
                    per_slot >> 20,
                    slot_state_bytes >> 20,
                );
            }
            return Ok(Dsv4KvBudgetPlan {
                num_slots: planned,
                flashmla_pool_tokens: planned.saturating_mul(max_seq_len),
            });
        }
        // #154 Phase 3b — demand-paged bands: num_slots stops being the
        // pool's sizing unit. Jointly pick (num_slots, pool_tokens): the
        // largest state-affordable slot count whose pool remainder still
        // holds the per-slot ring/safety reserve PLUS one full-length
        // request's comp capacity, then the largest shared token capacity
        // that remainder funds (capped at num_slots × max_seq_len — beyond
        // that no admissible mix can consume it). Feasibility is monotone in
        // decreasing n, so the first feasible n scanning down is the max.
        // All inputs are cross-rank-reduced ⇒ the loop is lockstep-identical.
        let page = self.kv_arena.page_block_size;
        let pages_needed = |n: usize, pool_tokens: usize| -> Result<usize> {
            let mut acc = 0usize;
            for layer in &self.layers {
                acc = acc.saturating_add(crate::attention::dsv4_flashmla_layer_pool_pages(
                    &self.config,
                    layer.mode,
                    layer.compress_ratio,
                    max_seq_len,
                    page,
                    n,
                    pool_tokens,
                )?);
            }
            Ok(acc)
        };
        let n_max = requested.max(1).min(affordable);
        let mut chosen: Option<(usize, usize)> = None;
        for n in (1..=n_max).rev() {
            let pool_pages = budget_bytes.saturating_sub(per_slot.saturating_mul(n))
                / flashmla_page_bytes.max(1);
            if pages_needed(n, max_seq_len)? > pool_pages {
                continue;
            }
            let (mut lo, mut hi) = (max_seq_len, n.saturating_mul(max_seq_len));
            while lo < hi {
                let mid = lo + (hi - lo).div_ceil(2);
                if pages_needed(n, mid)? <= pool_pages {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            chosen = Some((n, lo));
            break;
        }
        let Some((planned, flashmla_pool_tokens)) = chosen else {
            anyhow::bail!(
                "DSv4 KV budget rejected startup: even one slot's ring reserve plus a \
                 full-length comp band ({total_slot_pages} pages, {}MB) does not fit the \
                 pool remainder at max_seq_len {max_seq_len}. Lower --max-total-tokens or free VRAM.",
                flashmla_band_bytes_per_slot >> 20,
            );
        };
        if planned < requested {
            log::warn!(
                "DSv4 KV budget: requested {requested} slots clamped to {planned} \
                 (cross-rank-min state-affordable {affordable}, local {affordable_local})."
            );
        }
        log::info!(
            "DSv4 KV budget (demand-paged bands): num_slots {planned}, shared comp capacity \
             {flashmla_pool_tokens} tokens ({} engine pages), per_slot {}MB, budget {}MB",
            flashmla_pool_tokens / page.max(1),
            per_slot >> 20,
            budget_bytes >> 20,
        );
        Ok(Dsv4KvBudgetPlan {
            num_slots: planned,
            flashmla_pool_tokens,
        })
    }
}
