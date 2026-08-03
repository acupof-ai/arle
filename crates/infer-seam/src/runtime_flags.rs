//! CLI-driven backend runtime toggles (runtime config = CLI flags, not env).
//!
//! Plain host data carried by `EngineLoadConfig` (serde → multiproc workers
//! read the SAME values via `ARLE_WORKER_ENGINE_CONFIG`); each backend applies
//! its struct once, before executor construction. Field defaults are the
//! shipped defaults — the serde defaults keep old worker payloads loadable.

/// TP collective transport selection (`--comm-backend`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommBackend {
    /// One-shot custom allreduce where eligible, NCCL otherwise.
    #[default]
    Auto,
    /// NCCL everywhere (disables the one-shot custom allreduce).
    Nccl,
}

fn d_true() -> bool {
    true
}
fn d_fa3_decode_splits() -> usize {
    8
}
fn d_deepgemm_min_routes() -> usize {
    1024
}
fn d_dsa_indexer_sms() -> usize {
    78
}
fn d_mtp_min_accept() -> f32 {
    0.55
}
fn d_spec_max_batch() -> usize {
    // Qwen3.5 DSpark's measured envelope; every unbatched scheme clamps to 1
    // at its own call site.
    16
}
fn d_deepep_num_sms() -> u32 {
    20
}

/// CUDA executor runtime toggles, applied via `infer_cuda::apply_runtime_flags`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CudaRuntimeFlags {
    /// Whole-step Qwen3.5/3.6 decode graph (paged lane licensed 2026-08-03:
    /// −7.9% ITL, byte-identical greedy, MMLU parity).
    #[serde(default = "d_true")]
    pub qwen35_decode_graph: bool,
    /// Batched rows>1 Qwen3.5/3.6 decode; off = sequential per-row A/B arm.
    #[serde(default = "d_true")]
    pub qwen35_batched_decode: bool,
    /// DeepGEMM grouped expert GEMMs (load-time: builds grouped weight caches).
    #[serde(default = "d_true")]
    pub qwen35_deepgemm: bool,
    /// Decode-band batch grouped MoE kernels; off = hand batch kernels A/B arm.
    #[serde(default = "d_true")]
    pub qwen35_moe_decode_kernel: bool,
    /// On-device MoE router; off = host `infer_moe::route` reference.
    #[serde(default = "d_true")]
    pub qwen35_gpu_router: bool,
    /// FA3 full-attention prefill (silently ignored on stub builds).
    #[serde(default = "d_true")]
    pub qwen35_fa3: bool,
    /// FA3 decode split count (clamped to \[2, 256\]).
    #[serde(default = "d_fa3_decode_splits")]
    pub qwen35_fa3_decode_splits: usize,
    /// Routed-row floor for the DeepGEMM grouped expert path. Default 1024,
    /// the compile-time `QWEN35_DEEPGEMM_MIN_ROUTES`; lower it to reach the
    /// uncharacterized mid-band (batched decode is `R = top_k * B`).
    #[serde(default = "d_deepgemm_min_routes")]
    pub qwen35_deepgemm_min_routes: usize,
    /// FlashQLA chunked GDN prefill (sm_90a baked Qwen3.6 shard only).
    #[serde(default)]
    pub qwen35_gdr_chunked: bool,
    /// Fast page-16 decode-metadata kernel path.
    #[serde(default)]
    pub decode_metadata_fast_page16: bool,
    /// Marlin W4 FP8 prefill weights at load.
    #[serde(default)]
    pub marlin_w4_fp8_prefill: bool,
    /// Retain the cuMemAllocAsync pool across syncs (caching allocator).
    #[serde(default = "d_true")]
    pub mempool_retain: bool,
    /// Safetensor shard read-ahead cache budget in bytes (None = built-in default).
    #[serde(default)]
    pub shard_cache_bytes: Option<usize>,
    /// Pin multi-rank workers to their GPU's NUMA node.
    #[serde(default = "d_true")]
    pub numa_pin: bool,
    /// TP collective transport (`--comm-backend`).
    #[serde(default)]
    pub comm_backend: CommBackend,
    /// DSv4 FlashMLA sparse decode: None = default (on when compiled in).
    #[serde(default)]
    pub dsv4_flashmla_decode: Option<bool>,
    /// DSv4 DSA indexer SM budget.
    #[serde(default = "d_dsa_indexer_sms")]
    pub dsv4_dsa_indexer_sms: usize,
    /// DSv4 contiguous-decode MoE path (KILLED as default; A/B lever).
    #[serde(default)]
    pub dsv4_moe_contig_decode: bool,
    /// DSv4 decode-region prefix reuse: capture the finished slot's frontier at
    /// finish, restore a later turn to the exact finish position (opt-in until
    /// the pod perf+correctness license; OFF = byte-identical).
    #[serde(default)]
    pub dsv4_decode_reuse: bool,
    /// Adaptive MTP gate at B=1 (skip speculation below the accept break-even).
    #[serde(default)]
    pub mtp_adaptive: bool,
    /// Minimum accept-rate EMA to keep speculating under --mtp-adaptive.
    #[serde(default = "d_mtp_min_accept")]
    pub mtp_min_accept: f32,
    /// Speculate (MTP/DSpark) only when the decode batch is ≤ this. Above it,
    /// spec is a compute-bound loss, so route decode to the plain batched path.
    /// Default 1: only true c=1 speculates.
    #[serde(default = "d_spec_max_batch")]
    pub spec_max_batch: usize,
    /// DeepEP intranode SM budget (positive, even).
    #[serde(default = "d_deepep_num_sms")]
    pub deepep_num_sms: u32,
    /// DeepEP LL per-rank dispatch-token cap (None = SGLANG env or 256).
    #[serde(default)]
    pub deepep_max_dispatch_tokens_per_rank: Option<u32>,
}

impl Default for CudaRuntimeFlags {
    fn default() -> Self {
        Self {
            qwen35_decode_graph: true,
            qwen35_batched_decode: true,
            qwen35_deepgemm: true,
            qwen35_moe_decode_kernel: true,
            qwen35_gpu_router: true,
            qwen35_fa3: true,
            qwen35_fa3_decode_splits: d_fa3_decode_splits(),
            qwen35_deepgemm_min_routes: d_deepgemm_min_routes(),
            qwen35_gdr_chunked: false,
            decode_metadata_fast_page16: false,
            marlin_w4_fp8_prefill: false,
            mempool_retain: true,
            shard_cache_bytes: None,
            numa_pin: true,
            comm_backend: CommBackend::Auto,
            dsv4_flashmla_decode: None,
            dsv4_dsa_indexer_sms: d_dsa_indexer_sms(),
            dsv4_moe_contig_decode: false,
            dsv4_decode_reuse: false,
            mtp_adaptive: false,
            mtp_min_accept: d_mtp_min_accept(),
            spec_max_batch: d_spec_max_batch(),
            deepep_num_sms: d_deepep_num_sms(),
            deepep_max_dispatch_tokens_per_rank: None,
        }
    }
}

/// Metal executor runtime toggles, applied via `infer_metal::apply_runtime_flags`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MetalRuntimeFlags {
    /// Overlapped c=1 greedy decode pipeline.
    #[serde(default = "d_true")]
    pub pipeline: bool,
    /// Load-time JIT warmup forward.
    #[serde(default = "d_true")]
    pub warmup: bool,
    /// Paged-prefix SDPA read path for single-token decode.
    #[serde(default = "d_true")]
    pub paged_kv_read: bool,
    /// Host (blocking D2H) non-greedy sampler; off = device greedy argmax.
    #[serde(default)]
    pub host_sampling: bool,
    /// Speculative decode master switch (`--no-speculative` clears it).
    #[serde(default = "d_true")]
    pub speculative: bool,
    /// Explicit DFlash draft head (HF id or dir); None = model auto-resolve.
    #[serde(default)]
    pub draft_model: Option<String>,
    /// Draft depth (`--speculative-tokens`); None = resolver default.
    #[serde(default)]
    pub speculative_tokens: Option<usize>,
    /// Acceptance width (`--spec-accept-topk`); 1 = exact greedy verify.
    #[serde(default = "d_spec_accept_topk")]
    pub spec_accept_topk: i32,
}

fn d_spec_accept_topk() -> i32 {
    1
}

impl Default for MetalRuntimeFlags {
    fn default() -> Self {
        Self {
            pipeline: true,
            warmup: true,
            paged_kv_read: true,
            host_sampling: false,
            speculative: true,
            draft_model: None,
            speculative_tokens: None,
            spec_accept_topk: d_spec_accept_topk(),
        }
    }
}
