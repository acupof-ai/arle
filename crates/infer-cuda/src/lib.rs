//! CUDA backend executor.
//!
//! [`CudaKvPool`] is the CUDA name for the backend-neutral host page manager
//! implementing the [`KvPool`] seam. It is the SINGLE page allocator for the
//! Qwen-dense paged path: the executor lowers each scheduled row's host page
//! table into the device pool (`TokenKVPool::mirror_slot`), which is what makes
//! radix prefix attach serve real device KV. [`CudaExecutor`] implements
//! [`BackendExecutor`] (CPU-testable placeholder without `cuda`) and dispatches
//! three model arms: dense BF16 Qwen3 (paged, host-mirrored), Qwen3.5/3.6 hybrid
//! MoE (per-slot arena, recurrent state), and DSv4-Flash FP8 (per-slot MLA
//! arena, multi-GPU TP/EP).
//!
//! Depends only on `infer-plan` + `infer-seam`, never engine-core.

use std::fmt;
#[cfg(feature = "cuda")]
use std::path::Path;

use infer_plan::{ForwardPlan, SlotToken, StepOutput};
use infer_seam::{
    BackendExecutor, HostPagedKvPool, KvPool, KvTierLocation, PollResult, PrefixBlock,
};

#[cfg(feature = "cuda")]
mod attention;
#[cfg(all(feature = "cuda", feature = "deepep"))]
mod deepep;
// DSv4-Flash FP8 model (loader + structs + MLA KV arena). cuda-gated: holds
// device weight matrices + the shared DSv4 FP8 DeepGEMM caches.
#[cfg(feature = "cuda")]
mod dsv4;
#[cfg(feature = "cuda")]
mod executor;
#[cfg(feature = "cuda")]
pub mod graph;
// Shared paged-KV host math: page table -> byte-offset / physical-row
// translation. Two consumers — DSv4 FlashMLA arena (#85 P2, packed MLA latent
// pool; FlashMLA's FFI calls a page a "block") and the Qwen quant-KV store
// kernels (#68, physical_token_rows for the identity-assuming quant kernels).
// Not cuda-gated: pure host math, CPU-testable without nvcc.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
mod paged_kv_table;
// Not cuda-gated under the explicit `no-cuda` feature: host-only resident
// weight-quant checkpoint ABI detection and numeric codecs.
#[cfg(any(feature = "cuda", feature = "no-cuda"))]
#[allow(dead_code)]
mod quant_format;
// DSv4 hyper-connections (`hc_mult > 1`): the wide residual stream wrap. cuda-
// gated (device kernels + DSv4 weight matrices).
#[cfg(feature = "cuda")]
mod hc;
#[cfg(feature = "cuda")]
mod linear_profile;
#[cfg(feature = "cuda")]
mod loader;
// GPU-NUMA worker pinning (boot one-shot; rank-skew mitigation).
#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
mod numa_pin;
// CLI-driven runtime toggles (EngineLoadConfig.cuda → statics; no env reads).
#[cfg(feature = "cuda")]
mod runtime_flags;
#[cfg(feature = "cuda")]
pub use runtime_flags::apply_runtime_flags;
#[cfg(feature = "cuda")]
mod nvtx;
#[cfg(feature = "cuda")]
mod ops;
#[cfg(feature = "cuda")]
mod profile;
#[cfg(feature = "cuda")]
mod stage_profile;
// Qwen3.5 / Qwen3.6 HYBRID model (gated-delta linear attention + periodic full
// attention, BF16 MoE). cuda-gated: device weight matrices + recurrent state.
#[cfg(feature = "cuda")]
mod qwen35;
// Persistent forward-workspace buffer slots (exact-shape reuse cache). cuda-
// gated: owns device buffers.
#[cfg(feature = "cuda")]
mod workspace;

// Per-step student LoRA re-merge contract (OPD P2). The host-side data types
// the train crate pushes into the student engine; re-exported from `infer-api`.
#[cfg(feature = "cuda")]
pub use qwen35::{
    SharedBf16BaseProjection, SharedFp8BaseProjection, StudentLoraLayer, StudentLoraMatrices,
    StudentLoraProjection, StudentLoraProjectionUpdate, StudentLoraUpdate,
};

// Load-time decode-graph default setter (CLI `--cuda-graph` → engine). Lets the
// `enable_cuda_graph` load flag actually gate the B=1 decode graph instead of
// being discarded; the `--no-cuda-graph` flag controls it.
/// CUDA KV-cache dtype resolution (#68): resolves the seam request against the
/// CUDA support matrix, failing loud on unwired paged quant modes.
#[cfg(feature = "cuda")]
pub use executor::CudaKvCacheDtype;
/// DSpark train sidecar: experience buffer for test-time training of the draft
/// model. The inference hot path pushes (draft_tokens, draft_logits,
/// target_logits, accepted_count) tuples; a separate trainer drains them and
/// runs acceptance-weighted policy gradient against the acceptance reward.
#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
/// Tier budget resolution: machine-derived disk budget when `--kv-disk` has no
/// `--kv-disk-limit` (probes free disk), and the per-rank L2 share from a
/// deployment-total `--kv-dram` request.
pub use kv_native_sys::{default_t2_budget_bytes, resolve_dram_budget_bytes};
/// Rank-0 NCCL `unique_id` mint for multiproc launchers (see [`loader::mint_nccl_unique_id_hex`]).
#[cfg(feature = "nccl")]
pub use loader::mint_nccl_unique_id_hex;
/// Decode the NCCL `unique_id` a launcher published via `INFER_NCCL_UNIQUE_ID`
/// (context-parallel training reuses the serve rendezvous channel).
#[cfg(feature = "nccl")]
pub use loader::nccl_unique_id_from_env;

/// Process-local override for DSv4 FlashMLA decode dispatch. `None` restores
/// the `--dsv4-flashmla-decode` flag default. Intended for resident A/B
/// harnesses that need to compare scalar vs FlashMLA after one model load.
#[cfg(feature = "cuda")]
pub fn set_dsv4_flashmla_decode_override(enabled: Option<bool>) {
    attention::set_dsv4_flashmla_decode_override(enabled);
}

#[cfg(feature = "cuda")]
pub fn set_dsv4_fused_wqkv_decode_override(enabled: Option<bool>) {
    attention::set_dsv4_fused_wqkv_decode_override(enabled);
}

/// Process-local toggle for the DSv4 contiguous-decode MoE path
/// (`--dsv4-moe-contig-decode`). Same A/B-harness intent as the overrides above.
#[cfg(feature = "cuda")]
/// Make the Qwen3.6 MoE loader keep routed experts as per-expert BF16
/// `DeviceMatrix` (dequantized from FP8 at load) so the OPD rollout student can
/// re-merge LoRA into experts each step. Call BEFORE loading the student engine;
/// off = serving default (fused grouped-FP8 experts, no per-expert re-merge).
#[cfg(feature = "cuda")]
pub fn set_qwen35_moe_experts_bf16_resident(enabled: bool) {
    runtime_flags::set_qwen35_moe_experts_bf16_resident(enabled);
}

/// Declare the row counts this engine's forwards can present: the most a decode
/// step can carry (the executor's slot budget) and the most a prefill chunk can.
/// Call BEFORE the executor loads its weights — the dense FP8 routing floor and
/// the load-time DeepGEMM preparation that pairs with it both read this, and an
/// undeclared envelope keeps every dense GEMM on its M-independent arm.
#[cfg(feature = "cuda")]
pub fn apply_dense_gemm_row_envelope(decode_rows: usize, prefill_rows: usize) {
    runtime_flags::set_dense_gemm_row_envelope(decode_rows, prefill_rows);
}

#[cfg(feature = "cuda")]
pub fn reset_dsv4_linear_profile() {
    linear_profile::reset();
}

#[cfg(feature = "cuda")]
pub fn print_dsv4_linear_profile(tag: &str) {
    linear_profile::print_rank0(tag);
}

#[cfg(feature = "cuda")]
pub fn reset_dsv4_stage_profile() {
    stage_profile::reset();
}

#[cfg(feature = "cuda")]
pub fn set_dsv4_stage_profile_active(active: bool) {
    stage_profile::set_active(active);
}

#[cfg(feature = "cuda")]
pub fn print_dsv4_stage_profile(tag: &str, timed_tokens: usize, timed_wall_ms: f64) {
    stage_profile::print_rank0(tag, timed_tokens, timed_wall_ms);
}

// Not cuda-gated: env→TpConfig resolution is CPU-testable; only the NCCL comm
// variant is feature-gated.
pub mod tp;

// Not cuda-gated: pure-CPU per-rank weight-shard byte slicing; the device upload
// that consumes it stays in `loader`.
pub mod shard_slice;

// Not cuda-gated: Qwen35Config → infer_moe::MoeConfig bridge + per-rank expert
// split arithmetic.
pub mod moe_config;

// Not cuda-gated: the host route→assignment flattening is CPU-tested; the device
// `moe_forward_into` lives in the inner `cuda`-gated module.
mod moe;

pub type CudaKvPool = HostPagedKvPool;

/// In-flight handle for a submitted CUDA step. Resolves synchronously today.
#[derive(Debug)]
pub struct CudaInflight {
    output: StepOutput,
}

/// CUDA backend executor.
///
/// `new()` is the no-GPU placeholder for host tests; the `from_*_safetensors`
/// constructors (feature `cuda`) build the real executor.
#[derive(Default)]
pub struct CudaExecutor {
    inner: CudaExecutorInner,
}

#[derive(Default)]
enum CudaExecutorInner {
    #[default]
    Placeholder,
    #[cfg(feature = "cuda")]
    Real(Box<executor::RealCudaExecutor>),
}

impl fmt::Debug for CudaExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            CudaExecutorInner::Placeholder => f
                .debug_struct("CudaExecutor")
                .field("inner", &"placeholder")
                .finish(),
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => {
                f.debug_struct("CudaExecutor").field("inner", real).finish()
            }
        }
    }
}

impl CudaExecutor {
    /// Build a CUDA executor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: CudaExecutorInner::Placeholder,
        }
    }

    /// Re-budget the host-demoted KV tier (`0` disables). Pre-serve only — must be
    /// called before the engine starts demoting; existing entries are dropped.
    pub fn set_kv_tier_budget_bytes(&mut self, bytes: usize) {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => {
                let _ = bytes;
            }
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.set_kv_tier_budget_bytes(bytes),
        }
    }

    /// Attach the opt-in disk spill level under `root` (pre-serve only).
    /// Returns whether the loaded model's arm consumed it; callers fail
    /// closed on `false` so an explicit `--kv-disk` is never a silent
    /// no-op.
    pub fn set_kv_tier_disk(&mut self, root: std::path::PathBuf, budget_bytes: usize) -> bool {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => {
                let _ = (root, budget_bytes);
                false
            }
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.set_kv_tier_disk(root, budget_bytes),
        }
    }

    /// Build the real CUDA executor for a BF16 Qwen3.5/3.6 hybrid dense-or-MoE
    /// checkpoint. BF16 only; the W4/4-bit canonical needs the W4 grouped-GEMM
    /// follow-up.
    ///
    /// The full-attn KV is a SHARED paged pool sized from measured free VRAM
    /// after weights load (`mem_fraction_static`, SGLang-style); `total_pages`
    /// is the per-request ceiling + the host-admission floor. The ACTUAL device
    /// pool page count the host [`CudaKvPool`] must mirror is reported by
    /// [`Self::effective_total_pages`].
    /// `dspark_draft_model`: `Some(dir)` loads the DSpark/DFlash block drafter
    /// checkpoint and enables `--spec-type dspark` speculative decode.
    #[cfg(feature = "cuda")]
    pub fn from_qwen35_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
        max_total_tokens: usize,
        kv_dtype: CudaKvCacheDtype,
        mem_fraction_static: f64,
        dspark_draft_model: Option<&std::path::Path>,
        dspark_sps_bias_ms: f32,
        dspark_sps_row_ms: f32,
        markov_head_rank: Option<usize>,
        dspark_block_size: Option<usize>,
        mtp_draft_tokens: Option<usize>,
        memory_budget_bytes: Option<usize>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            inner: CudaExecutorInner::Real(Box::new(
                executor::RealCudaExecutor::from_qwen35_safetensors(
                    model_path,
                    num_slots,
                    total_pages,
                    max_total_tokens,
                    kv_dtype,
                    mem_fraction_static,
                    dspark_draft_model,
                    dspark_sps_bias_ms,
                    dspark_sps_row_ms,
                    markov_head_rank,
                    dspark_block_size,
                    mtp_draft_tokens,
                    memory_budget_bytes,
                )?,
            )),
        })
    }

    /// Build the real CUDA executor for a DSv4-Flash FP8 checkpoint (MLA + HC +
    /// FP8 DeepGEMM MoE). DSv4 is multi-GPU only (TP=8/EP=8): the per-rank EP
    /// expert split + NCCL TP groups resolve from the env (`INFER_NCCL_UNIQUE_ID`,
    /// `INFER_CUDA_DEVICES`/world-size), so the launcher binds one rank per GPU.
    /// DSv4 owns its MLA KV state inside the forward, so no `total_pages`/
    /// `CudaKvPool` page budget is needed (a host pool is still attached for slot
    /// bookkeeping).
    /// `max_seq_len` is a runtime knob: the serve path threads it from
    /// `--max-total-tokens`/`EngineConfig::max_total_tokens` — the same global
    /// cap every backend uses, no DSv4-only knob. Standalone microbenchmarks
    /// with no CLI-args layer (`dsv4_resident_ab`) pass their own
    /// literal constant. The executor itself never reads an env var for this.
    /// `mtp_draft_tokens`: `Some(n)` turns on the checkpoint-native MTP
    /// speculative-decode head with draft depth `n` (config-driven, the serve
    /// path's `--spec-type mtp` / `--mtp-draft-tokens`); `mtp_draft_topk`
    /// controls the per-level draft candidate width while verifier rows stay
    /// chain-shaped. `None` = no MTP head (spec decode off unless
    /// `--spec-type dspark` supplies a draft model).
    #[cfg(feature = "cuda")]
    pub fn from_dsv4_fp8_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        max_seq_len: usize,
        mtp_draft_tokens: Option<usize>,
        mtp_draft_topk: Option<usize>,
        dspark_draft_model: Option<&std::path::Path>,
        dspark_sps_bias_ms: f32,
        dspark_sps_row_ms: f32,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            inner: CudaExecutorInner::Real(Box::new(
                executor::RealCudaExecutor::from_dsv4_fp8_safetensors(
                    model_path,
                    num_slots,
                    max_seq_len,
                    mtp_draft_tokens,
                    mtp_draft_topk,
                    dspark_draft_model,
                    dspark_sps_bias_ms,
                    dspark_sps_row_ms,
                )?,
            )),
        })
    }

    /// Effective slot count after any model-side KV-budget clamp, or `None` on
    /// the no-GPU placeholder. The DSv4 constructor may clamp below the
    /// requested `num_slots` (dynamic KV mem budget, cross-rank min-reduced);
    /// the engine's scheduler and admission pool MUST be sized from this value,
    /// not the requested one, or admission targets slots the executor has no
    /// arena for.
    #[must_use]
    pub fn effective_num_slots(&self) -> Option<usize> {
        match &self.inner {
            CudaExecutorInner::Placeholder => None,
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => Some(real.effective_num_slots()),
        }
    }

    /// Actual shared device-pool page count for the paged-pool models (dense
    /// Qwen3 + Qwen3.6 + DSv4 MLA latent pool), profiled from measured free VRAM
    /// at construction. The host admission `CudaKvPool` must mirror this 1:1 —
    /// not the requested `total_pages`. `None` only for the placeholder. DSv4
    /// returns `Some(flashmla_total_pages)` (its MLA pool is free-VRAM-sized),
    /// paired with [`Self::effective_page_size`] (64-tok pages, not config 16).
    #[must_use]
    pub fn effective_total_pages(&self) -> Option<usize> {
        match &self.inner {
            CudaExecutorInner::Placeholder => None,
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.effective_total_pages(),
        }
    }

    /// Device pool page size (tokens/page) for arms whose host admission pool
    /// must mirror device granularity. DSv4's MLA pool pages at 64
    /// (`page_block_size`), not `config.page_size` (16) — the host pool must use
    /// this or it admits at 1/4 the device token capacity (H3). `None` (and the
    /// placeholder) ⇒ the host page size already matches the device default.
    #[must_use]
    pub fn effective_page_size(&self) -> Option<usize> {
        match &self.inner {
            CudaExecutorInner::Placeholder => None,
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.effective_page_size(),
        }
    }

    /// Fixed logical page-band width per slot for backends whose device cache is
    /// not sequential in token position. DSv4 FlashMLA uses a full
    /// `[SW ring | compressed]` band; the host pool must allocate that band once
    /// so `KvBatchDescriptor` can carry the row's complete page table.
    #[must_use]
    pub fn effective_fixed_pages_per_slot(&self) -> Option<usize> {
        match &self.inner {
            CudaExecutorInner::Placeholder => None,
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.effective_fixed_pages_per_slot(),
        }
    }

    /// OPD teacher raw-logits forward (Qwen3.5/3.6 hybrid only).
    ///
    /// Runs the full hybrid forward over `(input_ids, positions)` and returns the
    /// FULL `[seq_len, vocab]` logits (every row, no sampling) plus a clone of the
    /// model's [`DeviceContext`] so the caller can sync/consume the device buffer.
    /// `infer-api` wraps this triple into its public `RawLogits`. The placeholder
    /// (no real GPU executor) bails.
    #[cfg(feature = "cuda")]
    pub fn forward_token_logits(
        &mut self,
        input_ids: &[u32],
        positions: &[u32],
    ) -> anyhow::Result<(
        cuda_kernels::prelude::DeviceVec,
        [usize; 2],
        cuda_kernels::prelude::DeviceContext,
    )> {
        match &mut self.inner {
            CudaExecutorInner::Real(real) => {
                let (logits, shape) = real.forward_token_logits(input_ids, positions)?;
                let device = real.device().clone();
                Ok((logits, shape, device))
            }
            CudaExecutorInner::Placeholder => {
                anyhow::bail!(
                    "forward_token_logits requires a loaded CUDA model, not the placeholder executor"
                )
            }
        }
    }

    /// Trunk taps at `target_layer_ids` as `[seq, taps·hidden]` plus the
    /// final-normed hidden states as `[seq, hidden]`, both host f32 — the two
    /// trunk inputs `spec_train::trainer::Target` needs per sample.
    #[cfg(feature = "cuda")]
    pub fn forward_training_taps(
        &mut self,
        input_ids: &[u32],
        target_layer_ids: &[i64],
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        match &mut self.inner {
            CudaExecutorInner::Real(real) => {
                real.forward_training_taps(input_ids, target_layer_ids)
            }
            CudaExecutorInner::Placeholder => anyhow::bail!(
                "forward_training_taps requires a loaded CUDA model, not the placeholder executor"
            ),
        }
    }

    /// Fold a fresh student LoRA update into the resident Qwen3.5/3.6
    /// projection weights (OPD per-step re-merge). Errors on the no-GPU
    /// placeholder and on non-student CUDA models (DSv4).
    #[cfg(feature = "cuda")]
    pub fn remerge_student_lora(
        &mut self,
        update: qwen35::StudentLoraUpdate,
    ) -> anyhow::Result<()> {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => anyhow::bail!(
                "student LoRA re-merge requires the real CUDA executor; \
                 the no-GPU placeholder has no resident weights"
            ),
            CudaExecutorInner::Real(real) => real.remerge_student_lora(update),
        }
    }

    /// Read-only borrow of resident FP8 block-scaled base projection pointers
    /// for train-infer weight sharing (`--share-frozen-base`). Errors on the
    /// no-GPU placeholder and on non-student CUDA models.
    #[cfg(feature = "cuda")]
    pub fn frozen_base_fp8_pointers(&self) -> anyhow::Result<Vec<qwen35::SharedFp8BaseProjection>> {
        match &self.inner {
            CudaExecutorInner::Placeholder => anyhow::bail!(
                "frozen-base FP8 sharing requires the real CUDA executor; \
                 the no-GPU placeholder has no resident weights"
            ),
            CudaExecutorInner::Real(real) => real.frozen_base_fp8_pointers(),
        }
    }

    /// Non-owning views of every resident dense-BF16 base projection's device
    /// pointer, for refreshing the train student's frozen base AFTER a LoRA
    /// re-merge.
    #[cfg(feature = "cuda")]
    pub fn frozen_base_bf16_pointers(
        &self,
    ) -> anyhow::Result<Vec<qwen35::SharedBf16BaseProjection>> {
        match &self.inner {
            CudaExecutorInner::Placeholder => anyhow::bail!(
                "frozen-base BF16 sharing requires the real CUDA executor; \
                 the no-GPU placeholder has no resident weights"
            ),
            CudaExecutorInner::Real(real) => real.frozen_base_bf16_pointers(),
        }
    }

    /// Hot-swap the DSpark Markov head weights from a host f32 snapshot.
    /// Called by the train sidecar after each acceptance-weighted step.
    #[cfg(feature = "cuda")]
    pub fn update_dspark_markov_weights(&mut self, w1: &[f32], w2: &[f32]) -> anyhow::Result<()> {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => anyhow::bail!(
                "DSpark Markov weight update requires the real CUDA executor; \
                 the no-GPU placeholder has no resident weights"
            ),
            CudaExecutorInner::Real(real) => real.update_dspark_markov_weights(w1, w2),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn dsv4_verify_forward_selftest(&mut self, prompt: &[u32]) -> anyhow::Result<()> {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => {
                anyhow::bail!("DSv4 verify-forward selftest requires the real CUDA executor")
            }
            CudaExecutorInner::Real(real) => real.dsv4_verify_forward_selftest(prompt),
        }
    }

    /// Placeholder forward — produces one deterministic token per scheduled row.
    fn placeholder_forward(plan: &ForwardPlan) -> StepOutput {
        let mut tokens = Vec::with_capacity(plan.decode_rows.len() + plan.prefill_rows.len());
        for row in &plan.decode_rows {
            tokens.push(SlotToken {
                slot: row.slot,
                token: row.last_token.wrapping_add(1),
                logprob: None,
                top_logprobs: Vec::new(),
                finish: None,
            });
        }
        for row in &plan.prefill_rows {
            let token = row.tokens.last().copied().unwrap_or(0).wrapping_add(1);
            tokens.push(SlotToken {
                slot: row.slot,
                token,
                logprob: None,
                top_logprobs: Vec::new(),
                finish: None,
            });
        }
        StepOutput { tokens }
    }
}

impl BackendExecutor for CudaExecutor {
    type Inflight = CudaInflight;

    fn submit(
        &mut self,
        plan: &ForwardPlan,
        _kv: &mut dyn KvPool,
    ) -> anyhow::Result<Self::Inflight> {
        let output = match &mut self.inner {
            CudaExecutorInner::Placeholder => Self::placeholder_forward(plan),
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.submit(plan, _kv)?,
        };
        Ok(CudaInflight { output })
    }

    fn poll(&mut self, inflight: Self::Inflight) -> anyhow::Result<PollResult<Self::Inflight>> {
        Ok(PollResult::Ready(inflight.output))
    }

    fn poll_background(&mut self) -> anyhow::Result<()> {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => {}
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.poll_background(),
        }
        Ok(())
    }

    fn warmup(&mut self) -> anyhow::Result<()> {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => Ok(()),
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.warmup(),
        }
    }

    fn model_stop_token_ids(&self) -> Vec<u32> {
        match &self.inner {
            CudaExecutorInner::Placeholder => Vec::new(),
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.model_stop_token_ids(),
        }
    }

    fn step_limits(&self) -> infer_seam::StepLimits {
        match &self.inner {
            CudaExecutorInner::Placeholder => infer_seam::StepLimits::default(),
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => infer_seam::StepLimits {
                max_tokens_per_step: real.max_tokens_per_step(),
                max_prefill_chunk: real.max_prefill_chunk(),
                prefill_restore_boundary_alignment: real.prefill_restore_boundary_alignment(),
                spec_row_tokens: real.spec_row_tokens(),
                ..infer_seam::StepLimits::default()
            },
        }
    }

    fn stats(&self) -> infer_seam::BackendStats {
        let (spec_decode, operator_dispatch, op_timing) = match &self.inner {
            CudaExecutorInner::Placeholder => Default::default(),
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => (
                real.spec_decode_stats(),
                real.operator_dispatch_stats(),
                real.op_timing_stats(),
            ),
        };
        infer_seam::BackendStats {
            spec_decode,
            operator_dispatch,
            op_timing,
            artifact: infer_seam::BackendArtifactIdentity {
                kernel_bundle_id: cuda_kernels::KERNEL_BUILD_ID.to_string(),
            },
            gpu: None,
        }
    }

    fn tp_sync_min(&self, local: usize) -> anyhow::Result<usize> {
        match &self.inner {
            CudaExecutorInner::Placeholder => Ok(local),
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.tp_sync_min(local),
        }
    }

    fn kv_shard_spec(&self) -> Option<(usize, usize)> {
        match &self.inner {
            CudaExecutorInner::Placeholder => None,
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.kv_shard_spec(),
        }
    }

    fn release_kv_slot(&mut self, slot: usize) {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => {
                let _ = slot;
            }
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.release_kv_slot(slot),
        }
    }

    fn prefix_reuse(&mut self) -> Option<&mut dyn infer_seam::PrefixReuse> {
        Some(self)
    }

    fn kv_page_tier(&mut self) -> Option<&mut dyn infer_seam::KvPageTier> {
        Some(self)
    }

    fn kv_page_tier_view(&self) -> Option<&dyn infer_seam::KvPageTier> {
        Some(self)
    }

    fn kv_slot_tier(&mut self) -> Option<&mut dyn infer_seam::KvSlotTier> {
        // Presence replaces the old `kv_slot_tier_enabled` bool: only models
        // with a whole-slot store expose the capability.
        let enabled = match &self.inner {
            CudaExecutorInner::Placeholder => false,
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.kv_slot_tier_enabled(),
        };
        if enabled { Some(self) } else { None }
    }

    fn device_kv_fit(&self) -> Option<&dyn infer_seam::DeviceKvFit> {
        // Presence replaces the old `kv_device_gate_active` bool.
        let active = match &self.inner {
            CudaExecutorInner::Placeholder => false,
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.kv_device_gate_active(),
        };
        if active { Some(self) } else { None }
    }

    fn weight_residency(&mut self) -> Option<&mut dyn infer_seam::WeightResidency> {
        Some(self)
    }
}

impl infer_seam::PrefixReuse for CudaExecutor {
    fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
        match &self.inner {
            CudaExecutorInner::Placeholder => {
                let _ = blocks;
                0
            }
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.reusable_prefix_blocks(blocks),
        }
    }

    fn reusable_prefix_blocks_for_prompt(&self, blocks: &[PrefixBlock], tokens: &[u32]) -> usize {
        match &self.inner {
            CudaExecutorInner::Placeholder => {
                let _ = (blocks, tokens);
                0
            }
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.reusable_prefix_blocks_for_prompt(blocks, tokens),
        }
    }

    fn release_prefix_pages(&mut self, pages: &[u32]) {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => {
                let _ = pages;
            }
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.release_prefix_pages(pages),
        }
    }

    fn release_provisional_prefix_pages(&mut self, pages: &[u32]) {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => {
                let _ = pages;
            }
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.release_provisional_prefix_pages(pages),
        }
    }

    fn cached_prefix_match_len(&self, tokens: &[u32]) -> anyhow::Result<usize> {
        match &self.inner {
            CudaExecutorInner::Placeholder => {
                let _ = tokens;
                Ok(0)
            }
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.cached_prefix_match_len(tokens),
        }
    }

    fn restore_cached_prefix(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
        slot_pages: &[u32],
    ) -> anyhow::Result<()> {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => {
                let _ = (slot, tokens, matched_len, slot_pages);
                anyhow::bail!("placeholder CUDA executor has no position-0 prefix store")
            }
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => {
                real.restore_cached_prefix(slot, tokens, matched_len, slot_pages)
            }
        }
    }

    fn restore_prefix_sidecar(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
        prefix_pages: &[u32],
    ) -> anyhow::Result<usize> {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => {
                let _ = (slot, tokens, prefix_pages);
                Ok(matched_len)
            }
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => {
                real.restore_prefix_sidecar(slot, tokens, matched_len, prefix_pages)
            }
        }
    }

    fn capture_finish_frontier(
        &mut self,
        slot: usize,
        tokens: &[u32],
        slot_pages: &[u32],
    ) -> anyhow::Result<()> {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => {
                let _ = (slot, tokens, slot_pages);
                Ok(())
            }
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.capture_finish_frontier(slot, tokens, slot_pages),
        }
    }

    fn save_prefix_sidecar(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
        prefix_pages: &[u32],
        slot_pages: &[u32],
        newly_cached: &[u32],
    ) -> anyhow::Result<()> {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => {
                let _ = (
                    slot,
                    tokens,
                    matched_len,
                    prefix_pages,
                    slot_pages,
                    newly_cached,
                );
                Ok(())
            }
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.save_prefix_sidecar(
                slot,
                tokens,
                matched_len,
                prefix_pages,
                slot_pages,
                newly_cached,
            ),
        }
    }
}

impl infer_seam::KvPageTier for CudaExecutor {
    // No CUDA arm has a page-addressable device pool; Qwen3.5/3.6 and DSv4
    // park whole slots via the slot-tier hooks instead.
    fn kv_tier_capacity_pages(&self) -> usize {
        0
    }

    fn kv_tier_page_bytes(&self) -> usize {
        0
    }

    fn kv_tier_host_demoted_pages(&self) -> usize {
        match &self.inner {
            CudaExecutorInner::Placeholder => 0,
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.kv_tier_host_demoted_pages(),
        }
    }

    fn kv_tier_disk_pages(&self) -> usize {
        match &self.inner {
            CudaExecutorInner::Placeholder => 0,
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.kv_tier_disk_pages(),
        }
    }

    fn kv_tier_read_hits(&self) -> infer_seam::KvTierReadHits {
        match &self.inner {
            CudaExecutorInner::Placeholder => infer_seam::KvTierReadHits::default(),
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.kv_tier_read_hits(),
        }
    }

    fn kv_tier_io_stats(&self) -> infer_seam::KvTierIoStats {
        match &self.inner {
            CudaExecutorInner::Placeholder => infer_seam::KvTierIoStats::default(),
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.kv_tier_io_stats(),
        }
    }

    fn kv_tier_transfer_is_zero_copy(&self) -> bool {
        false
    }

    fn kv_tier_location(&self, key: u64) -> Option<KvTierLocation> {
        match &self.inner {
            CudaExecutorInner::Placeholder => {
                let _ = key;
                None
            }
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.kv_tier_location(key),
        }
    }

    fn demote_prefix_pages(&mut self, _entries: &[(u32, u64)]) -> anyhow::Result<usize> {
        Ok(0)
    }

    fn promote_prefix_pages(&mut self, _entries: &[(u64, u32)]) -> anyhow::Result<()> {
        anyhow::bail!("CUDA executor has no page-granular KV tier store")
    }

    fn drop_kv_tier_entries(&mut self, _keys: &[u64]) {}
}

impl infer_seam::KvSlotTier for CudaExecutor {
    fn demote_slot(&mut self, slot: usize, key: u64) -> anyhow::Result<bool> {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => {
                let _ = (slot, key);
                Ok(false)
            }
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.demote_slot(slot, key),
        }
    }

    fn promote_slot(&mut self, key: u64, slot: usize, slot_pages: &[u32]) -> anyhow::Result<()> {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => {
                let _ = (key, slot, slot_pages);
                anyhow::bail!("placeholder CUDA executor has no whole-slot KV tier store")
            }
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.promote_slot(key, slot, slot_pages),
        }
    }

    fn drop_kv_slot_entries(&mut self, keys: &[u64]) {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => {
                let _ = keys;
            }
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.drop_kv_slot_entries(keys),
        }
    }
}

impl infer_seam::DeviceKvFit for CudaExecutor {
    fn kv_device_fit(&self, rows: &[infer_seam::DeviceRowDemand], unfit: &mut Vec<usize>) {
        match &self.inner {
            CudaExecutorInner::Placeholder => {
                let _ = (rows, unfit);
            }
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.kv_device_fit(rows, unfit),
        }
    }
}

impl infer_seam::WeightResidency for CudaExecutor {
    fn offload_weights(&mut self) -> anyhow::Result<usize> {
        match &mut self.inner {
            // No real device weights to offload without the cuda backend.
            CudaExecutorInner::Placeholder => Ok(0),
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.offload_engine_weights(),
        }
    }

    fn reload_weights(&mut self) -> anyhow::Result<()> {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => Ok(()),
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.reload_engine_weights(),
        }
    }

    fn release_inference_scratch(&mut self) -> anyhow::Result<()> {
        match &mut self.inner {
            // No real device scratch to release without the cuda backend.
            CudaExecutorInner::Placeholder => Ok(()),
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.release_inference_scratch(),
        }
    }

    fn release_kv_pool(&mut self) -> anyhow::Result<()> {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => Ok(()),
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.release_kv_pool(),
        }
    }

    fn ensure_kv_pool(&mut self) -> anyhow::Result<()> {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => Ok(()),
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.ensure_kv_pool(),
        }
    }
}
