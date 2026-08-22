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
fn d_deepgemm_min_routes() -> usize {
    1024
}
fn d_dsa_indexer_sms() -> usize {
    78
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
    /// Whole-step Qwen3.5/3.6 decode graph (paged lane); off = the eager
    /// same-binary A/B arm. Serve hardcodes on; the OPD rollout engine keeps
    /// its own off-by-default lever.
    #[serde(default = "d_true")]
    pub qwen35_decode_graph: bool,
    /// Routed-row floor for the DeepGEMM grouped expert path. Default 1024,
    /// the compile-time `QWEN35_DEEPGEMM_MIN_ROUTES`; lower it to reach the
    /// uncharacterized mid-band (batched decode is `R = top_k * B`).
    #[serde(default = "d_deepgemm_min_routes")]
    pub qwen35_deepgemm_min_routes: usize,
    /// FlashQLA chunked GDN prefill (sm_90a baked Qwen3.6 shard only).
    #[serde(default = "d_true")]
    pub qwen35_gdr_chunked: bool,
    /// Retain the cuMemAllocAsync pool across syncs (caching allocator).
    #[serde(default = "d_true")]
    pub mempool_retain: bool,
    #[serde(default)]
    pub shard_cache_bytes: Option<usize>,
    #[serde(default = "d_true")]
    pub numa_pin: bool,
    #[serde(default)]
    pub comm_backend: CommBackend,
    #[serde(default = "d_dsa_indexer_sms")]
    pub dsv4_dsa_indexer_sms: usize,
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
    /// DSv4 MoE transport override (`allreduce`|`deepep`|`deepep_ll`|`mega_moe`);
    /// None = `ARLE_DSV4_MOE_TRANSPORT` env or allreduce.
    #[serde(default)]
    pub dsv4_moe_transport: Option<String>,
}

impl Default for CudaRuntimeFlags {
    fn default() -> Self {
        Self {
            qwen35_decode_graph: d_true(),
            qwen35_deepgemm_min_routes: d_deepgemm_min_routes(),
            qwen35_gdr_chunked: d_true(),
            mempool_retain: d_true(),
            shard_cache_bytes: None,
            numa_pin: d_true(),
            comm_backend: CommBackend::default(),
            dsv4_dsa_indexer_sms: d_dsa_indexer_sms(),
            spec_max_batch: d_spec_max_batch(),
            deepep_num_sms: d_deepep_num_sms(),
            deepep_max_dispatch_tokens_per_rank: None,
            dsv4_moe_transport: None,
        }
    }
}

/// Metal executor runtime toggles, applied via `infer_metal::apply_runtime_flags`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MetalRuntimeFlags {
    #[serde(default)]
    pub warmup: bool,
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
            warmup: false,
            host_sampling: false,
            speculative: d_true(),
            draft_model: None,
            speculative_tokens: None,
            spec_accept_topk: d_spec_accept_topk(),
        }
    }
}
