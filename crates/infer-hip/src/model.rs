//! DSv4-Flash forward orchestration for the HIP lane (#77, plan §2.2).
//!
//! Pinned mode flags (plan §2.2): the FlashMLA, official-DSA and DeepGEMM
//! datacenter paths are EXCLUDED — this is the shim-portable fallback
//! lane. The launcher sequence is extracted from `infer-cuda/src/dsv4.rs`
//! `forward_tokens_stream_impl` + `attention.rs` `mla_attention`'s
//! non-FlashMLA branch (call sites at attention.rs:4372/4472/4533/4611/
//! 4735 on today's tree):
//!
//! embed → mhc_expand → per layer {
//!   gemv(hc_attn.fn) → mhc_params → mhc_pre_rms_norm(attn_norm)
//!   → gemv(wq_a) → rms_norm(q_norm) → gemv(wq_b); gemv(wkv) → rms_norm(kv_norm)
//!   → prepare_qk(start_pos_ptr)
//!   → [SW] swa_attention(write_window_cache=1)
//!   → [CSA/HCA] compressor_update; [CSA] indexer compressor_update +
//!     gemv(indexer.wq_b)/gemv(indexer.proj) + csa_select; hybrid_attention(wwc=1)
//!   → gemv(wo_a) → gemv(wo_b) → mhc_post
//!   → gemv(hc_ffn.fn) → mhc_params → mhc_pre_rms_norm(ffn_norm)
//!   → gemv(router) → host route → per-expert up/gate → swiglu_clamp → down
//!   → shared expert (same shape) → add → mhc_post
//! } → mhc_head_pre → rms_norm(output_norm) → gemv(lm_head) → host f32 logits.
//!
//! §0.1 mutated-buffer enumeration: see [`ATTENTION_LAUNCHER_WRITES`]; only
//! the four ring/compressed families plus the SW window ring persist across
//! steps — everything else is per-step scratch fully rewritten before read.
//! The CUDA-only `dsv4_update_window_cache` / `arle_dsv4_output_inverse_rope`
//! launchers are FlashMLA adjuncts (callers at attention.rs:3195/3866) and
//! never run on this path: SW/hybrid kernels update the ring and inverse-rope
//! the output inline.
//!
//! All launches go through the `*_start_pos_ptr_cuda` variants on ONE HIP
//! stream; host routing is the only mid-token sync (correctness-first MVP;
//! batched mmq prefill and graph capture are later-work knobs).

use anyhow::{Result, bail};
use deepseek_spec::v4::{DeepSeekV4AttentionLayerPlan, DeepSeekV4Config};

use crate::kv_pool::Dsv4SlotShape;
use crate::loader::Residency;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemvKind {
    Q4K,
    Q5K,
    Q6K,
    /// q8_1 activation quantize + `arle_mmvq_iq2_xxs_cuda`.
    MmvqIq2Xxs,
    /// q8_1 activation quantize + `arle_mmvq_q2_k_cuda`.
    MmvqQ2K,
}

/// Kernel choice for a residency tier; `None` = no HIP matmul exists
/// (bf16 GEMV is cuBLAS-only on CUDA and excluded here, plan §2.0).
pub fn gemv_kind(residency: Residency) -> Option<GemvKind> {
    use crate::loader::KQuant;
    match residency {
        Residency::KeepKQuant(KQuant::Q4K) => Some(GemvKind::Q4K),
        Residency::KeepKQuant(KQuant::Q5K) => Some(GemvKind::Q5K),
        Residency::KeepKQuant(KQuant::Q6K) => Some(GemvKind::Q6K),
        Residency::KeepIq2Xxs => Some(GemvKind::MmvqIq2Xxs),
        Residency::KeepQ2K => Some(GemvKind::MmvqQ2K),
        Residency::DequantBf16 | Residency::DequantF32 | Residency::HostOnly => None,
    }
}

/// Whether a tensor participates in a matmul on this forward path.
/// `HyperConnection`/`HeadHyperConnection` cover both the mixer matrix
/// (`*_fn`, 2-D) and the base/scale vectors, so the rank disambiguates;
/// the embedding table only matmuls when it doubles as the tied lm_head.
pub fn is_matmul_bearing(
    kind: infer_gguf::deepseek4::Dsv4TensorKind,
    ndims: usize,
    tied: bool,
) -> bool {
    use infer_gguf::deepseek4::Dsv4TensorKind::*;
    match kind {
        TokenEmbedding => tied,
        OutputHead | AttnQA | AttnQB | AttnKvLatent | AttnOutA | AttnOutB | CompressorKv
        | CompressorGate | IndexerProj | IndexerQB | IndexerCompressKv | IndexerCompressGate
        | RouterGate | RoutedExpertGate | RoutedExpertUp | RoutedExpertDown | SharedExpertGate
        | SharedExpertUp | SharedExpertDown => true,
        RoutedExpertGateUp => false,
        HyperConnection | HeadHyperConnection => ndims >= 2,
        _ => false,
    }
}

/// Fail-loud load gate: every matmul-bearing tensor must land on a residency
/// the HIP GEMV surface serves. Names the offending tensor so a recipe gap
/// (e.g. f32 router gate in a community GGUF) is a clear load error, not a
/// silent garbage forward. `ndims` comes from the GGUF tensor info.
pub fn validate_matmul_residency<'a>(
    tensors: impl Iterator<
        Item = (
            &'a str,
            infer_gguf::deepseek4::Dsv4TensorKind,
            Residency,
            usize,
        ),
    >,
    tied_embeddings: bool,
) -> Result<()> {
    for (name, kind, residency, ndims) in tensors {
        if kind == infer_gguf::deepseek4::Dsv4TensorKind::RoutedExpertGateUp {
            bail!(
                "tensor {name} ({kind:?}) uses fused routed gate/up layout, but HIP DSv4 \
                 forward requires split ffn_gate_exps.weight and ffn_up_exps.weight tensors"
            );
        }
        if is_matmul_bearing(kind, ndims, tied_embeddings) && gemv_kind(residency).is_none() {
            bail!(
                "tensor {name} ({kind:?}) is matmul-bearing but lands {residency:?}: no HIP \
                 bf16 GEMV exists — requantize it to Q4_K/Q5_K/Q6_K/IQ2_XXS/Q2_K"
            );
        }
    }
    Ok(())
}

pub struct LauncherWrites {
    pub launcher: &'static str,
    pub writes: &'static [&'static str],
}

/// §0.1 mutated-buffer table for every attention-path launcher this lane
/// uses, sourced from the CUDA call sites (attention.rs:4372 prepare_qk,
/// :4472 swa, :5024 compressor ×2 — main + indexer, :5224 csa_select,
/// :4735 hybrid). `slot.*` buffers persist across steps; `scratch.*` are
/// fully rewritten before every read.
pub const ATTENTION_LAUNCHER_WRITES: &[LauncherWrites] = &[
    LauncherWrites {
        launcher: "dsv4_prepare_qk_start_pos_ptr_cuda",
        writes: &["scratch.q_prepared", "scratch.k_prepared"],
    },
    LauncherWrites {
        launcher: "dsv4_swa_attention_start_pos_ptr_cuda(write_window_cache=1)",
        writes: &["scratch.attn_local", "slot.sw_window_ring"],
    },
    LauncherWrites {
        launcher: "dsv4_compressor_update_start_pos_ptr_cuda(compressor)",
        writes: &[
            "slot.compressor.pending_kv",
            "slot.compressor.pending_score",
            "slot.compressor.prev_overlap_kv",
            "slot.compressor.prev_overlap_score",
            "slot.compressor.compressed",
        ],
    },
    LauncherWrites {
        launcher: "dsv4_compressor_update_start_pos_ptr_cuda(indexer)",
        writes: &[
            "slot.indexer.pending_kv",
            "slot.indexer.pending_score",
            "slot.indexer.prev_overlap_kv",
            "slot.indexer.prev_overlap_score",
            "slot.indexer.compressed",
        ],
    },
    LauncherWrites {
        launcher: "dsv4_csa_select_start_pos_ptr_cuda",
        writes: &["scratch.selected"],
    },
    LauncherWrites {
        launcher: "dsv4_hybrid_attention_start_pos_ptr_cuda(write_window_cache=1)",
        writes: &["scratch.attn_local", "slot.sw_window_ring"],
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressorOffsets {
    pub pending_kv: usize,
    pub pending_score: usize,
    pub prev_overlap_kv: usize,
    pub prev_overlap_score: usize,
    pub compressed: usize,
    /// Capacity of the compressed region in rows of `head_dim` elems.
    pub compressed_rows: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerOffsets {
    pub sw_window: usize,
    pub compressor: Option<CompressorOffsets>,
    pub indexer: Option<CompressorOffsets>,
}

/// Arena layout: layers in order, each `sw_window` then compressor family
/// then indexer family, families in `Dsv4CompressorShapeElems` field order.
/// Must stay additive-exact with `Dsv4SlotShape::total_elems`.
pub fn slot_layer_offsets(shape: &Dsv4SlotShape, head_dims: (usize, usize)) -> Vec<LayerOffsets> {
    fn family(
        cursor: &mut usize,
        elems: &crate::kv_pool::Dsv4CompressorShapeElems,
        dim: usize,
    ) -> CompressorOffsets {
        let base = *cursor;
        let off = CompressorOffsets {
            pending_kv: base,
            pending_score: base + elems.pending_kv,
            prev_overlap_kv: base + elems.pending_kv + elems.pending_score,
            prev_overlap_score: base
                + elems.pending_kv
                + elems.pending_score
                + elems.prev_overlap_kv,
            compressed: base
                + elems.pending_kv
                + elems.pending_score
                + elems.prev_overlap_kv
                + elems.prev_overlap_score,
            compressed_rows: elems.compressed / dim.max(1),
        };
        *cursor += elems.total();
        off
    }
    let (head_dim, index_head_dim) = head_dims;
    let mut cursor = 0usize;
    let mut out = Vec::with_capacity(shape.layers.len());
    for layer in &shape.layers {
        let sw_window = cursor;
        cursor += layer.sw_window;
        let compressor = layer
            .compressor
            .as_ref()
            .map(|c| family(&mut cursor, c, head_dim));
        let indexer = layer
            .indexer
            .as_ref()
            .map(|c| family(&mut cursor, c, index_head_dim));
        out.push(LayerOffsets {
            sw_window,
            compressor,
            indexer,
        });
    }
    out
}

/// Per-token host routing — the verified `deepseek-spec` reference math
/// (sqrt(softplus) scores, bias-corrected noaux top-k, normalize-by-selected
/// then `routed_scaling_factor`; hash layers take experts from `tid2eid`).
/// Mirrors `infer-cuda/src/moe.rs` `hash_route` + the host route path.
pub fn route_layer_token(
    config: &DeepSeekV4Config,
    layer_idx: usize,
    token: u32,
    logits_row: &[f32],
    bias: Option<&[f32]>,
    tid2eid: Option<&[i64]>,
) -> Result<Vec<(usize, f32)>> {
    let scores = config
        .router_scores_from_logits(logits_row)
        .map_err(|e| anyhow::anyhow!("DSv4 router scores: {e}"))?;
    let hash_experts: Option<Vec<usize>> = if layer_idx < config.num_hash_layers {
        let table = tid2eid
            .ok_or_else(|| anyhow::anyhow!("hash-routed layer {layer_idx} missing tid2eid"))?;
        let topk = config.num_experts_per_tok;
        let start = token as usize * topk;
        if start + topk > table.len() {
            bail!(
                "tid2eid too short: token {token} needs [{start}, {}), have {}",
                start + topk,
                table.len()
            );
        }
        Some(
            table[start..start + topk]
                .iter()
                .map(|&e| {
                    usize::try_from(e).map_err(|_| anyhow::anyhow!("tid2eid expert id {e} invalid"))
                })
                .collect::<Result<_>>()?,
        )
    } else {
        None
    };
    let routes = config
        .moe_routes_from_scores(layer_idx, 0, &scores, bias, hash_experts.as_deref())
        .map_err(|e| anyhow::anyhow!("DSv4 route: {e}"))?;
    Ok(routes
        .into_iter()
        .map(|r| (r.expert_idx, r.weight))
        .collect())
}

pub fn layer_schedule(config: &DeepSeekV4Config) -> Vec<DeepSeekV4AttentionLayerPlan> {
    (0..config.num_hidden_layers)
        .filter_map(|i| config.attention_layer_plan(i))
        .collect()
}

#[cfg(feature = "hip")]
pub use device::HipDsv4Model;

#[cfg(feature = "hip")]
mod device {
    use std::collections::HashMap;
    use std::ffi::c_void;

    use anyhow::{Context, Result, anyhow, bail, ensure};
    use deepseek_spec::v4::{
        DeepSeekV4AttentionLayerPlan, DeepSeekV4AttentionMode, DeepSeekV4Config,
    };
    use hip_kernels::ffi;
    use hip_sys::{DeviceBuffer, Stream};

    use super::{GemvKind, LayerOffsets, gemv_kind, slot_layer_offsets};
    use crate::kv_pool::Dsv4SlotShape;
    use crate::loader::{Residency, ResidencyPlan, upload::DeviceTensor};
    use infer_gguf::deepseek4::classify_tensor;
    use infer_gguf::gguf::{GgmlType, GgufFile};

    fn check(code: i32, what: &str) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(anyhow!("{what} failed: {}", hip_kernels::KernelError(code)))
        }
    }

    struct WeightEntry {
        residency: Residency,
        ggml_type: GgmlType,
        dims: Vec<u64>,
        buffer: DeviceBuffer,
    }

    impl WeightEntry {
        fn ptr(&self) -> *mut c_void {
            self.buffer.as_ptr()
        }
        /// ne[0] = contiguous (K) dim, ne[1] = rows (N).
        fn k(&self) -> usize {
            self.dims.first().copied().unwrap_or(1) as usize
        }
        fn n(&self) -> usize {
            self.dims.get(1).copied().unwrap_or(1) as usize
        }
    }

    /// Per-step device scratch — preallocated once, reused every token
    /// (no per-step alloc). Every buffer is fully written before read.
    struct Scratch {
        token_id: DeviceBuffer,
        emb: DeviceBuffer,
        stream_a: DeviceBuffer,
        stream_b: DeviceBuffer,
        normed: DeviceBuffer,
        mixes: DeviceBuffer,
        pre: DeviceBuffer,
        post: DeviceBuffer,
        comb: DeviceBuffer,
        c_q: DeviceBuffer,
        q_raw: DeviceBuffer,
        kv_raw: DeviceBuffer,
        q_prep: DeviceBuffer,
        k_prep: DeviceBuffer,
        attn_local: DeviceBuffer,
        latent: DeviceBuffer,
        attn_out: DeviceBuffer,
        comp_kv: DeviceBuffer,
        comp_score: DeviceBuffer,
        idx_q: DeviceBuffer,
        idx_w: DeviceBuffer,
        selected: DeviceBuffer,
        router_logits: DeviceBuffer,
        gate: DeviceBuffer,
        up: DeviceBuffer,
        act: DeviceBuffer,
        expert_out: DeviceBuffer,
        moe_acc: DeviceBuffer,
        x_f32: DeviceBuffer,
        y_f32: DeviceBuffer,
        q8: DeviceBuffer,
        logits_bf16: DeviceBuffer,
        moe_zero_host: Vec<u8>,
    }

    struct SlotState {
        arena: DeviceBuffer,
        start_pos: DeviceBuffer,
        seq_len: usize,
        epoch: u64,
    }

    pub struct HipDsv4Model {
        pub config: DeepSeekV4Config,
        weights: HashMap<String, WeightEntry>,
        bias_by_layer: HashMap<usize, Vec<f32>>,
        tid2eid_by_layer: HashMap<usize, Vec<i64>>,
        layer_plans: Vec<DeepSeekV4AttentionLayerPlan>,
        shape: Dsv4SlotShape,
        offsets: Vec<LayerOffsets>,
        slots: Vec<Option<SlotState>>,
        stream: Stream,
        scratch: Scratch,
        lm_head_name: String,
    }

    fn alloc(label: &str, bytes: usize) -> Result<DeviceBuffer> {
        DeviceBuffer::alloc(bytes.max(4)).map_err(|e| anyhow!("alloc {label} ({bytes} B): {e}"))
    }

    impl HipDsv4Model {
        pub fn new(
            gguf: &GgufFile,
            config: DeepSeekV4Config,
            plan: &ResidencyPlan,
            tensors: Vec<DeviceTensor>,
            num_slots: usize,
            max_seq_len: usize,
        ) -> Result<Self> {
            // Fail-loud residency gate (no bf16 GEMV on HIP, plan §2.0).
            super::validate_matmul_residency(
                plan.tensors.iter().map(|t| {
                    let ndims = gguf.tensor(&t.name).map_or(2, |i| i.dims.len());
                    (t.name.as_str(), t.kind, t.residency, ndims)
                }),
                config.tie_word_embeddings,
            )?;

            let mut weights = HashMap::with_capacity(tensors.len());
            for dev in tensors {
                let info = gguf
                    .tensor(&dev.name)
                    .ok_or_else(|| anyhow!("uploaded tensor {} missing from GGUF", dev.name))?;
                weights.insert(
                    dev.name.clone(),
                    WeightEntry {
                        residency: dev.residency,
                        ggml_type: info.ggml_type,
                        dims: info.dims.clone(),
                        buffer: dev.buffer,
                    },
                );
            }

            // Host routing data straight from the GGUF (HostOnly residency).
            let mut bias_by_layer = HashMap::new();
            let mut tid2eid_by_layer = HashMap::new();
            for info in gguf.tensors() {
                let role = classify_tensor(&info.name)?;
                let Some(layer) = role.layer else { continue };
                match role.kind {
                    infer_gguf::deepseek4::Dsv4TensorKind::RouterBias => {
                        let n = info.element_count() as usize;
                        let bias = infer_gguf::dequant::dequantize_row_f32(
                            gguf.tensor_data(&info.name)?,
                            n,
                        )
                        .context("router bias dequant")?;
                        bias_by_layer.insert(layer, bias);
                    }
                    infer_gguf::deepseek4::Dsv4TensorKind::RouterHashTable => {
                        ensure!(
                            info.ggml_type == GgmlType::I64,
                            "tid2eid must be I64, got {:?}",
                            info.ggml_type
                        );
                        let data = gguf.tensor_data(&info.name)?;
                        let table: Vec<i64> = data
                            .as_chunks::<8>()
                            .0
                            .iter()
                            .map(|c| i64::from_le_bytes(*c))
                            .collect();
                        tid2eid_by_layer.insert(layer, table);
                    }
                    _ => {}
                }
            }

            let layer_plans = super::layer_schedule(&config);
            ensure!(
                layer_plans.len() == config.num_hidden_layers,
                "layer schedule {} != num_hidden_layers {}",
                layer_plans.len(),
                config.num_hidden_layers
            );
            let shape = Dsv4SlotShape::new(&config, max_seq_len);
            let offsets = slot_layer_offsets(&shape, (config.head_dim, config.index_head_dim));

            let h = config.hidden_size;
            let m = config.hc_mult;
            let stream_dim = h * m;
            let local_width = config.num_attention_heads * config.head_dim;
            let mix_rows = (2 + m) * m;
            // Shared expert width = n_shared * inter (it shares the
            // gate/up/act scratch with the routed experts).
            let inter = config.moe_intermediate_size * config.n_shared_experts.max(1);
            let vocab = config.vocab_size;
            let max_k = h.max(inter).max(stream_dim).max(config.q_lora_rank);
            let max_n = vocab.max(h).max(inter).max(local_width).max(mix_rows);
            let q8_bytes = max_k.div_ceil(hip_kernels::QK8_1) * hip_kernels::BLOCK_Q8_1_BYTES;
            let idx_width = config.index_n_heads * config.index_head_dim;

            let scratch = Scratch {
                token_id: alloc("token_id", 4)?,
                emb: alloc("emb", h * 2)?,
                stream_a: alloc("stream_a", stream_dim * 2)?,
                stream_b: alloc("stream_b", stream_dim * 2)?,
                normed: alloc("normed", h * 2)?,
                mixes: alloc("mixes", mix_rows * 2)?,
                pre: alloc("pre", m * 4)?,
                post: alloc("post", m * 4)?,
                comb: alloc("comb", m * m * 4)?,
                c_q: alloc("c_q", config.q_lora_rank * 2)?,
                q_raw: alloc("q_raw", local_width * 2)?,
                kv_raw: alloc("kv_raw", config.head_dim * 2)?,
                q_prep: alloc("q_prep", local_width * 2)?,
                k_prep: alloc("k_prep", config.head_dim * 2)?,
                attn_local: alloc("attn_local", local_width * 2)?,
                latent: alloc("latent", (config.o_groups * config.o_lora_rank).max(1) * 2)?,
                attn_out: alloc("attn_out", h * 2)?,
                comp_kv: alloc(
                    "comp_kv",
                    2 * config.head_dim.max(config.index_head_dim) * 2,
                )?,
                comp_score: alloc(
                    "comp_score",
                    2 * config.head_dim.max(config.index_head_dim) * 2,
                )?,
                idx_q: alloc("idx_q", idx_width.max(1) * 2)?,
                idx_w: alloc("idx_w", config.index_n_heads.max(1) * 2)?,
                selected: alloc("selected", config.index_topk.max(1) * 4)?,
                router_logits: alloc("router_logits", config.n_routed_experts * 2)?,
                gate: alloc("gate", inter * 2)?,
                up: alloc("up", inter * 2)?,
                act: alloc("act", inter * 2)?,
                expert_out: alloc("expert_out", h * 2)?,
                moe_acc: alloc("moe_acc", h * 2)?,
                x_f32: alloc("x_f32", max_k * 4)?,
                y_f32: alloc("y_f32", max_n * 4)?,
                q8: alloc("q8", q8_bytes)?,
                logits_bf16: alloc("logits_bf16", vocab * 2)?,
                moe_zero_host: vec![0u8; h * 2],
            };

            let lm_head_name = if weights.contains_key("output.weight") {
                "output.weight".to_string()
            } else {
                "token_embd.weight".to_string()
            };
            ensure!(
                weights.contains_key(&lm_head_name),
                "lm head tensor {lm_head_name} not uploaded"
            );

            Ok(Self {
                config,
                weights,
                bias_by_layer,
                tid2eid_by_layer,
                layer_plans,
                shape,
                offsets,
                slots: (0..num_slots).map(|_| None).collect(),
                stream: Stream::create().map_err(|e| anyhow!("HIP stream create: {e}"))?,
                scratch,
                lm_head_name,
            })
        }

        pub fn max_seq_len(&self) -> usize {
            self.shape.max_seq_len
        }

        /// Reset a slot whose host epoch moved (slot recycled by engine-core).
        pub fn reset_slot_if_epoch_changed(&mut self, slot: usize, epoch: u64) {
            if let Some(Some(state)) = self.slots.get(slot)
                && state.epoch != epoch
            {
                self.slots[slot] = None;
            }
        }

        fn w(&self, name: &str) -> Result<&WeightEntry> {
            self.weights
                .get(name)
                .ok_or_else(|| anyhow!("weight {name} not uploaded"))
        }

        fn lw(&self, layer: usize, suffix: &str) -> Result<&WeightEntry> {
            self.w(&format!("blk.{layer}.{suffix}"))
        }

        fn st(&self) -> *mut c_void {
            self.stream.as_ptr()
        }

        /// y[bf16; n] = W[n, k] · x[bf16; k], dispatching on the weight's
        /// kept-quantized residency; n_override caps the rows (expert slices
        /// pass the full per-expert n).
        fn gemv(
            &self,
            w: &WeightEntry,
            w_ptr: *const u8,
            n: usize,
            k: usize,
            x_bf16: *const u16,
            y_bf16: *mut u16,
        ) -> Result<()> {
            let kind = gemv_kind(w.residency)
                .ok_or_else(|| anyhow!("no HIP GEMV for residency {:?}", w.residency))?;
            let s = self.st();
            unsafe {
                match kind {
                    GemvKind::Q4K => check(
                        ffi::q4k_gemv_cuda(w_ptr, x_bf16, y_bf16, n as i32, k as i32, s),
                        "q4k_gemv",
                    ),
                    GemvKind::Q5K => check(
                        ffi::q5k_gemv_cuda(w_ptr, x_bf16, y_bf16, n as i32, k as i32, s),
                        "q5k_gemv",
                    ),
                    GemvKind::Q6K => check(
                        ffi::q6k_gemv_cuda(w_ptr, x_bf16, y_bf16, n as i32, k as i32, s),
                        "q6k_gemv",
                    ),
                    GemvKind::MmvqIq2Xxs | GemvKind::MmvqQ2K => {
                        let x_f32 = self.scratch.x_f32.as_ptr() as *mut f32;
                        let y_f32 = self.scratch.y_f32.as_ptr() as *mut f32;
                        check(
                            ffi::cast_bf16_to_f32_cuda(x_bf16, x_f32, k as i32, s),
                            "mmvq cast in",
                        )?;
                        hip_kernels::quantize_row_q8_1(
                            x_f32,
                            self.scratch.q8.as_ptr(),
                            k as i64,
                            1,
                            s,
                        )
                        .map_err(|e| anyhow!("q8_1 quantize: {e}"))?;
                        let mm = if kind == GemvKind::MmvqIq2Xxs {
                            hip_kernels::mmvq_iq2_xxs
                        } else {
                            hip_kernels::mmvq_q2_k
                        };
                        mm(
                            w_ptr.cast(),
                            self.scratch.q8.as_ptr(),
                            y_f32,
                            k as i32,
                            n as i32,
                            s,
                        )
                        .map_err(|e| anyhow!("mmvq: {e}"))?;
                        check(
                            ffi::cast_f32_to_bf16_cuda(y_f32, y_bf16, n as i32, s),
                            "mmvq cast out",
                        )
                    }
                }
            }
        }

        fn gemv_full(&self, w: &WeightEntry, x: *const u16, y: *mut u16) -> Result<()> {
            self.gemv(w, w.ptr().cast(), w.n(), w.k(), x, y)
        }

        fn rms_norm(
            &self,
            x: *const u16,
            weight: &WeightEntry,
            n: usize,
            y: *mut u16,
        ) -> Result<()> {
            unsafe {
                check(
                    ffi::rms_norm_cuda(
                        x,
                        weight.ptr().cast(),
                        y,
                        n as i32,
                        self.config.rms_norm_eps,
                        self.st(),
                    ),
                    "rms_norm",
                )
            }
        }

        fn mhc_params(&self, layer: usize, half: &str, stream: *const u16) -> Result<()> {
            let mix = self.lw(layer, &format!("hc_{half}_fn"))?;
            let base = self.lw(layer, &format!("hc_{half}_base"))?;
            let scale = self.lw(layer, &format!("hc_{half}_scale"))?;
            let stream_dim = self.config.hidden_size * self.config.hc_mult;
            ensure!(
                mix.n() * 2 <= self.scratch.mixes.len(),
                "HC mixer rows {} exceed mixes scratch",
                mix.n()
            );
            self.gemv_full(mix, stream, self.scratch.mixes.as_ptr().cast())?;
            unsafe {
                check(
                    ffi::dsv4_mhc_params_cuda(
                        stream,
                        self.scratch.mixes.as_ptr().cast(),
                        base.ptr().cast(),
                        scale.ptr().cast(),
                        self.scratch.pre.as_ptr().cast(),
                        self.scratch.post.as_ptr().cast(),
                        self.scratch.comb.as_ptr().cast(),
                        1,
                        stream_dim as i32,
                        mix.n() as i32,
                        self.config.hc_mult as i32,
                        self.config.hc_eps,
                        self.config.hc_sinkhorn_iters as i32,
                        self.st(),
                    ),
                    "dsv4_mhc_params",
                )
            }
        }

        fn mhc_pre_rms_norm(
            &self,
            stream: *const u16,
            norm: &WeightEntry,
            out: *mut u16,
        ) -> Result<()> {
            unsafe {
                check(
                    ffi::dsv4_mhc_pre_rms_norm_cuda(
                        stream,
                        self.scratch.pre.as_ptr().cast(),
                        norm.ptr().cast(),
                        out,
                        1,
                        self.config.hidden_size as i32,
                        self.config.hc_mult as i32,
                        self.config.rms_norm_eps,
                        self.st(),
                    ),
                    "dsv4_mhc_pre_rms_norm",
                )
            }
        }

        fn mhc_post(&self, new_x: *const u16, residual: *const u16, out: *mut u16) -> Result<()> {
            unsafe {
                check(
                    ffi::dsv4_mhc_post_cuda(
                        new_x,
                        residual,
                        self.scratch.post.as_ptr().cast(),
                        self.scratch.comb.as_ptr().cast(),
                        out,
                        1,
                        self.config.hidden_size as i32,
                        self.config.hc_mult as i32,
                        self.st(),
                    ),
                    "dsv4_mhc_post",
                )
            }
        }

        fn ensure_slot(&mut self, slot: usize, epoch: u64) -> Result<()> {
            ensure!(slot < self.slots.len(), "slot {slot} out of range");
            self.reset_slot_if_epoch_changed(slot, epoch);
            if self.slots[slot].is_none() {
                let bytes = self.shape.total_bytes();
                let mut arena = alloc("slot arena", bytes)?;
                // Zero once: ring/pending self-heal preconditions assume
                // unwritten regions read as zeros on a fresh slot.
                arena
                    .copy_from_host(&vec![0u8; bytes])
                    .map_err(|e| anyhow!("slot arena zero: {e}"))?;
                self.slots[slot] = Some(SlotState {
                    arena,
                    start_pos: alloc("start_pos", 4)?,
                    seq_len: 0,
                    epoch,
                });
            }
            Ok(())
        }

        /// One-token forward; returns host f32 logits. `start_pos` must equal
        /// the slot's current device seq_len (contiguous append).
        pub fn forward_token(
            &mut self,
            slot: usize,
            epoch: u64,
            token: u32,
            start_pos: usize,
        ) -> Result<Vec<f32>> {
            self.ensure_slot(slot, epoch)?;
            {
                let state = self.slots[slot].as_mut().expect("ensured");
                ensure!(
                    state.seq_len == start_pos,
                    "DSv4 HIP slot {slot} seq_len {} != start_pos {start_pos}",
                    state.seq_len
                );
                ensure!(
                    start_pos < self.shape.max_seq_len,
                    "DSv4 HIP slot {slot} exceeds max_seq_len {}",
                    self.shape.max_seq_len
                );
                state
                    .start_pos
                    .copy_from_host(&(start_pos as i32).to_le_bytes())
                    .map_err(|e| anyhow!("start_pos H2D: {e}"))?;
                self.scratch
                    .token_id
                    .copy_from_host(&(token as i32).to_le_bytes())
                    .map_err(|e| anyhow!("token H2D: {e}"))?;
            }

            let h = self.config.hidden_size;
            let hc_mult = self.config.hc_mult;
            let hc_eps = self.config.hc_eps;
            let vocab = self.config.vocab_size;
            let num_layers = self.config.num_hidden_layers;
            let s = self.st();

            let embed = self.w("token_embd.weight")?;
            let emb = self.scratch.emb.as_ptr() as *mut u16;
            unsafe {
                let tok = self.scratch.token_id.as_ptr() as *const i32;
                match gemv_kind(embed.residency) {
                    Some(GemvKind::Q4K) => check(
                        ffi::q4k_embedding_decode_cuda(embed.ptr().cast(), tok, emb, h as i32, s),
                        "q4k_embedding",
                    )?,
                    Some(GemvKind::Q5K) => check(
                        ffi::q5k_embedding_decode_cuda(embed.ptr().cast(), tok, emb, h as i32, s),
                        "q5k_embedding",
                    )?,
                    Some(GemvKind::Q6K) => check(
                        ffi::q6k_embedding_decode_cuda(embed.ptr().cast(), tok, emb, h as i32, s),
                        "q6k_embedding",
                    )?,
                    None if embed.residency == Residency::DequantBf16 => check(
                        ffi::embedding_decode_cuda(embed.ptr().cast(), tok, emb, h as i32, s),
                        "embedding",
                    )?,
                    other => bail!("token_embd residency {other:?} has no embedding kernel"),
                }
                check(
                    ffi::dsv4_mhc_expand_cuda(
                        emb,
                        self.scratch.stream_a.as_ptr().cast(),
                        1,
                        h as i32,
                        hc_mult as i32,
                        s,
                    ),
                    "dsv4_mhc_expand",
                )?;
            }

            for layer_idx in 0..num_layers {
                self.layer_forward(slot, layer_idx, token, start_pos)?;
            }

            let stream_a = self.scratch.stream_a.as_ptr() as *const u16;
            let head_mix = self.w("hc_head_fn")?;
            self.gemv_full(head_mix, stream_a, self.scratch.mixes.as_ptr().cast())?;
            unsafe {
                check(
                    ffi::dsv4_mhc_head_pre_cuda(
                        stream_a,
                        self.scratch.mixes.as_ptr().cast(),
                        self.w("hc_head_base")?.ptr().cast(),
                        self.w("hc_head_scale")?.ptr().cast(),
                        self.scratch.attn_out.as_ptr().cast(),
                        (h * hc_mult) as i32,
                        h as i32,
                        hc_mult as i32,
                        hc_eps,
                        s,
                    ),
                    "dsv4_mhc_head_pre",
                )?;
            }
            let out_norm = self.w("output_norm.weight")?;
            self.rms_norm(
                self.scratch.attn_out.as_ptr().cast(),
                out_norm,
                h,
                self.scratch.normed.as_ptr().cast(),
            )?;
            let lm = self.w(&self.lm_head_name.clone())?;
            let stage_bf16 = self.scratch.logits_bf16.as_ptr() as *mut u16;
            self.gemv(
                lm,
                lm.ptr().cast(),
                vocab,
                h,
                self.scratch.normed.as_ptr().cast(),
                stage_bf16,
            )?;
            unsafe {
                check(
                    ffi::cast_bf16_to_f32_cuda(
                        stage_bf16,
                        self.scratch.y_f32.as_ptr().cast(),
                        vocab as i32,
                        s,
                    ),
                    "logits cast",
                )?;
            }
            self.stream
                .synchronize()
                .map_err(|e| anyhow!("logits sync: {e}"))?;
            let mut host = vec![0u8; vocab * 4];
            self.scratch
                .y_f32
                .copy_to_host(&mut host)
                .map_err(|e| anyhow!("logits D2H: {e}"))?;
            let logits: Vec<f32> = host
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();

            self.slots[slot].as_mut().expect("ensured").seq_len += 1;
            Ok(logits)
        }

        fn layer_forward(
            &mut self,
            slot: usize,
            layer_idx: usize,
            token: u32,
            start_pos: usize,
        ) -> Result<()> {
            let cfg = self.config.clone();
            let plan = self.layer_plans[layer_idx].clone();
            let head_dim = cfg.head_dim;
            let local_heads = cfg.num_attention_heads;
            let rope = &cfg.rope_parameters;
            let (rope_base, orig_seq_len) = if plan.compress_ratio > 0 {
                (
                    cfg.compress_rope_theta,
                    rope.original_max_position_embeddings as i32,
                )
            } else {
                (cfg.rope_theta, 0)
            };
            let s = self.st();
            let stream_a = self.scratch.stream_a.as_ptr() as *mut u16;
            let stream_b = self.scratch.stream_b.as_ptr() as *mut u16;
            let normed = self.scratch.normed.as_ptr() as *mut u16;
            let (arena_base, start_pos_ptr) = {
                let st = self.slots[slot].as_ref().expect("slot ensured");
                (st.arena.as_ptr(), st.start_pos.as_ptr() as *const i32)
            };
            let arena = |off: usize| unsafe { (arena_base as *mut u16).add(off) };
            let off = self.offsets[layer_idx];

            self.mhc_params(layer_idx, "attn", stream_a)?;
            self.mhc_pre_rms_norm(stream_a, self.lw(layer_idx, "attn_norm.weight")?, normed)?;

            self.gemv_full(
                self.lw(layer_idx, "attn_q_a.weight")?,
                normed,
                self.scratch.c_q.as_ptr().cast(),
            )?;
            self.rms_norm(
                self.scratch.c_q.as_ptr().cast(),
                self.lw(layer_idx, "attn_q_a_norm.weight")?,
                cfg.q_lora_rank,
                self.scratch.c_q.as_ptr().cast(),
            )?;
            self.gemv_full(
                self.lw(layer_idx, "attn_q_b.weight")?,
                self.scratch.c_q.as_ptr().cast(),
                self.scratch.q_raw.as_ptr().cast(),
            )?;
            self.gemv_full(
                self.lw(layer_idx, "attn_kv_latent.weight")?,
                normed,
                self.scratch.kv_raw.as_ptr().cast(),
            )?;
            self.rms_norm(
                self.scratch.kv_raw.as_ptr().cast(),
                self.lw(layer_idx, "attn_kv_a_norm.weight")?,
                head_dim,
                self.scratch.kv_raw.as_ptr().cast(),
            )?;

            unsafe {
                check(
                    ffi::dsv4_prepare_qk_start_pos_ptr_cuda(
                        self.scratch.q_raw.as_ptr().cast(),
                        self.scratch.kv_raw.as_ptr().cast(),
                        self.scratch.q_prep.as_ptr().cast(),
                        self.scratch.k_prep.as_ptr().cast(),
                        1,
                        local_heads as i32,
                        head_dim as i32,
                        cfg.qk_rope_head_dim as i32,
                        start_pos_ptr,
                        cfg.rms_norm_eps,
                        rope_base,
                        orig_seq_len,
                        rope.factor,
                        rope.beta_fast,
                        rope.beta_slow,
                        s,
                    ),
                    "dsv4_prepare_qk",
                )?;
            }

            let sm_scale = 1.0 / (head_dim as f32).sqrt();
            let sink = self.lw(layer_idx, "attn_sinks")?;
            match plan.mode {
                DeepSeekV4AttentionMode::SlidingWindow => unsafe {
                    check(
                        ffi::dsv4_swa_attention_start_pos_ptr_cuda(
                            self.scratch.q_prep.as_ptr().cast(),
                            self.scratch.k_prep.as_ptr().cast(),
                            arena(off.sw_window),
                            sink.ptr().cast(),
                            self.scratch.attn_local.as_ptr().cast(),
                            1,
                            local_heads as i32,
                            head_dim as i32,
                            cfg.sliding_window as i32,
                            start_pos_ptr,
                            0,
                            sm_scale,
                            cfg.qk_rope_head_dim as i32,
                            rope_base,
                            orig_seq_len,
                            rope.factor,
                            rope.beta_fast,
                            rope.beta_slow,
                            1,
                            s,
                        ),
                        "dsv4_swa_attention",
                    )?;
                },
                mode => {
                    let ratio = plan.compress_ratio;
                    let overlap = ratio < 16;
                    let comp = off.compressor.expect("compressed layer has compressor");
                    self.compressor_update(
                        layer_idx,
                        "attn_compress",
                        normed,
                        arena_base,
                        start_pos_ptr,
                        comp,
                        head_dim,
                        ratio,
                        overlap,
                        cfg.qk_rope_head_dim,
                        orig_seq_len,
                    )?;
                    let selected_ptr = if mode == DeepSeekV4AttentionMode::CompressedSparse {
                        let idx = off.indexer.expect("CSA layer has indexer");
                        // Indexer keys: second compressor, no rope (official-DSA off).
                        self.compressor_update(
                            layer_idx,
                            "indexer.compress",
                            normed,
                            arena_base,
                            start_pos_ptr,
                            idx,
                            cfg.index_head_dim,
                            ratio,
                            true,
                            0,
                            0,
                        )?;
                        self.gemv_full(
                            self.lw(layer_idx, "indexer.attn_q_b.weight")?,
                            self.scratch.c_q.as_ptr().cast(),
                            self.scratch.idx_q.as_ptr().cast(),
                        )?;
                        self.gemv_full(
                            self.lw(layer_idx, "indexer.proj.weight")?,
                            normed,
                            self.scratch.idx_w.as_ptr().cast(),
                        )?;
                        let key_count = (start_pos + 1) / ratio;
                        let score_scale = (cfg.index_head_dim as f32).powf(-0.5)
                            * (cfg.index_n_heads as f32).powf(-0.5);
                        unsafe {
                            check(
                                ffi::dsv4_csa_select_start_pos_ptr_cuda(
                                    self.scratch.idx_q.as_ptr().cast(),
                                    self.scratch.idx_w.as_ptr().cast(),
                                    arena(idx.compressed),
                                    self.scratch.selected.as_ptr().cast(),
                                    1,
                                    (cfg.index_n_heads * cfg.index_head_dim) as i32,
                                    cfg.index_n_heads as i32,
                                    cfg.index_head_dim as i32,
                                    key_count as i32,
                                    ratio as i32,
                                    cfg.index_topk as i32,
                                    score_scale,
                                    start_pos_ptr,
                                    s,
                                ),
                                "dsv4_csa_select",
                            )?;
                        }
                        self.scratch.selected.as_ptr() as *const i32
                    } else {
                        std::ptr::null()
                    };
                    let compressed_count = (start_pos + 1) / ratio;
                    let comp_ptr = if compressed_count > 0 {
                        arena(comp.compressed) as *const u16
                    } else {
                        std::ptr::null()
                    };
                    let mode_int = if mode == DeepSeekV4AttentionMode::CompressedSparse {
                        1
                    } else {
                        2
                    };
                    unsafe {
                        check(
                            ffi::dsv4_hybrid_attention_start_pos_ptr_cuda(
                                self.scratch.q_prep.as_ptr().cast(),
                                self.scratch.k_prep.as_ptr().cast(),
                                arena(off.sw_window),
                                comp_ptr,
                                selected_ptr,
                                sink.ptr().cast(),
                                self.scratch.attn_local.as_ptr().cast(),
                                1,
                                local_heads as i32,
                                head_dim as i32,
                                cfg.sliding_window as i32,
                                start_pos_ptr,
                                0,
                                sm_scale,
                                cfg.qk_rope_head_dim as i32,
                                rope_base,
                                orig_seq_len,
                                rope.factor,
                                rope.beta_fast,
                                rope.beta_slow,
                                mode_int,
                                ratio as i32,
                                compressed_count as i32,
                                cfg.index_topk as i32,
                                1,
                                s,
                            ),
                            "dsv4_hybrid_attention",
                        )?;
                    }
                }
            }

            self.gemv_full(
                self.lw(layer_idx, "attn_output_a.weight")?,
                self.scratch.attn_local.as_ptr().cast(),
                self.scratch.latent.as_ptr().cast(),
            )?;
            self.gemv_full(
                self.lw(layer_idx, "attn_output_b.weight")?,
                self.scratch.latent.as_ptr().cast(),
                self.scratch.attn_out.as_ptr().cast(),
            )?;
            self.mhc_post(self.scratch.attn_out.as_ptr().cast(), stream_a, stream_b)?;
            std::mem::swap(&mut self.scratch.stream_a, &mut self.scratch.stream_b);
            let stream_a = self.scratch.stream_a.as_ptr() as *mut u16;
            let stream_b = self.scratch.stream_b.as_ptr() as *mut u16;

            // `normed` raw ptr stays valid — only stream buffers swap.
            self.mhc_params(layer_idx, "ffn", stream_a)?;
            self.mhc_pre_rms_norm(stream_a, self.lw(layer_idx, "ffn_norm.weight")?, normed)?;
            self.moe_forward(layer_idx, token, normed)?;
            self.mhc_post(self.scratch.moe_acc.as_ptr().cast(), stream_a, stream_b)?;
            std::mem::swap(&mut self.scratch.stream_a, &mut self.scratch.stream_b);
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn compressor_update(
            &self,
            layer: usize,
            prefix: &str,
            hidden: *const u16,
            arena_base: *mut c_void,
            start_pos_ptr: *const i32,
            off: super::CompressorOffsets,
            dim: usize,
            ratio: usize,
            overlap: bool,
            rope_dim: usize,
            orig_seq_len: i32,
        ) -> Result<()> {
            let cfg = &self.config;
            let width = if overlap { 2 * dim } else { dim };
            let wkv = self.lw(layer, &format!("{prefix}_kv.weight"))?;
            let wgate = self.lw(layer, &format!("{prefix}_gate.weight"))?;
            ensure!(
                wkv.n() == width && wgate.n() == width,
                "compressor {prefix} rows {}x{} != width {width}",
                wkv.n(),
                wgate.n()
            );
            self.gemv_full(wkv, hidden, self.scratch.comp_kv.as_ptr().cast())?;
            self.gemv_full(wgate, hidden, self.scratch.comp_score.as_ptr().cast())?;
            let ape = self.lw(layer, &format!("{prefix}_ape"))?;
            let norm = self.lw(layer, &format!("{prefix}_norm.weight"))?;
            let a = |o: usize| unsafe { (arena_base as *mut u16).add(o) };
            let rope = &cfg.rope_parameters;
            unsafe {
                check(
                    ffi::dsv4_compressor_update_start_pos_ptr_cuda(
                        self.scratch.comp_kv.as_ptr().cast(),
                        self.scratch.comp_score.as_ptr().cast(),
                        ape.ptr().cast(),
                        norm.ptr().cast(),
                        a(off.pending_kv),
                        a(off.pending_score),
                        a(off.prev_overlap_kv),
                        a(off.prev_overlap_score),
                        a(off.compressed),
                        1,
                        start_pos_ptr,
                        dim as i32,
                        ratio as i32,
                        width as i32,
                        i32::from(overlap),
                        cfg.rms_norm_eps,
                        rope_dim as i32,
                        cfg.compress_rope_theta,
                        orig_seq_len,
                        rope.factor,
                        rope.beta_fast,
                        rope.beta_slow,
                        self.st(),
                    ),
                    "dsv4_compressor_update",
                )
            }
        }

        /// Routed + shared MoE with HOST routing (router logits D2H per layer
        /// — the MVP's one mid-token sync).
        fn moe_forward(&mut self, layer_idx: usize, token: u32, normed: *const u16) -> Result<()> {
            let cfg = self.config.clone();
            let h = cfg.hidden_size;
            let inter = cfg.moe_intermediate_size;
            let s = self.st();
            let router = self.lw(layer_idx, "ffn_gate_inp.weight")?;
            self.gemv_full(router, normed, self.scratch.router_logits.as_ptr().cast())?;
            self.stream
                .synchronize()
                .map_err(|e| anyhow!("router sync: {e}"))?;
            let mut raw = vec![0u8; cfg.n_routed_experts * 2];
            self.scratch
                .router_logits
                .copy_to_host(&mut raw)
                .map_err(|e| anyhow!("router D2H: {e}"))?;
            let logits: Vec<f32> = raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
                .collect();
            let routes = super::route_layer_token(
                &cfg,
                layer_idx,
                token,
                &logits,
                self.bias_by_layer.get(&layer_idx).map(Vec::as_slice),
                self.tid2eid_by_layer.get(&layer_idx).map(Vec::as_slice),
            )?;

            let zero = self.scratch.moe_zero_host.clone();
            self.scratch
                .moe_acc
                .copy_from_host(&zero)
                .map_err(|e| anyhow!("moe_acc zero: {e}"))?;
            let moe_acc = self.scratch.moe_acc.as_ptr() as *mut u16;

            let up_w = self.lw(layer_idx, "ffn_up_exps.weight")?;
            let gate_w = self.lw(layer_idx, "ffn_gate_exps.weight")?;
            let down_w = self.lw(layer_idx, "ffn_down_exps.weight")?;
            let expert_bytes = |w: &WeightEntry| -> Result<usize> {
                let row = w
                    .ggml_type
                    .row_bytes(w.k())
                    .ok_or_else(|| anyhow!("unaligned expert rows"))?;
                Ok(row * w.n())
            };
            let (up_eb, gate_eb, down_eb) = (
                expert_bytes(up_w)?,
                expert_bytes(gate_w)?,
                expert_bytes(down_w)?,
            );
            for (expert, weight) in routes {
                let up_ptr = unsafe { (up_w.ptr() as *const u8).add(expert * up_eb) };
                let gate_ptr = unsafe { (gate_w.ptr() as *const u8).add(expert * gate_eb) };
                let down_ptr = unsafe { (down_w.ptr() as *const u8).add(expert * down_eb) };
                self.gemv(
                    up_w,
                    up_ptr,
                    inter,
                    h,
                    normed,
                    self.scratch.up.as_ptr().cast(),
                )?;
                self.gemv(
                    gate_w,
                    gate_ptr,
                    inter,
                    h,
                    normed,
                    self.scratch.gate.as_ptr().cast(),
                )?;
                unsafe {
                    check(
                        ffi::dsv4_swiglu_clamped_cuda(
                            self.scratch.gate.as_ptr().cast(),
                            self.scratch.up.as_ptr().cast(),
                            self.scratch.act.as_ptr().cast(),
                            inter as i32,
                            cfg.swiglu_limit,
                            s,
                        ),
                        "dsv4_swiglu_clamped",
                    )?;
                }
                self.gemv(
                    down_w,
                    down_ptr,
                    h,
                    inter,
                    self.scratch.act.as_ptr().cast(),
                    self.scratch.expert_out.as_ptr().cast(),
                )?;
                unsafe {
                    check(
                        ffi::add_scaled_row_cuda(
                            self.scratch.expert_out.as_ptr().cast(),
                            moe_acc,
                            h as i32,
                            0,
                            weight,
                            s,
                        ),
                        "add_scaled_row",
                    )?;
                }
            }

            if cfg.n_shared_experts > 0 {
                self.gemv_full(
                    self.lw(layer_idx, "ffn_up_shexp.weight")?,
                    normed,
                    self.scratch.up.as_ptr().cast(),
                )?;
                self.gemv_full(
                    self.lw(layer_idx, "ffn_gate_shexp.weight")?,
                    normed,
                    self.scratch.gate.as_ptr().cast(),
                )?;
                let sh_inter = self.lw(layer_idx, "ffn_up_shexp.weight")?.n();
                unsafe {
                    check(
                        ffi::dsv4_swiglu_clamped_cuda(
                            self.scratch.gate.as_ptr().cast(),
                            self.scratch.up.as_ptr().cast(),
                            self.scratch.act.as_ptr().cast(),
                            sh_inter as i32,
                            cfg.swiglu_limit,
                            s,
                        ),
                        "shared swiglu",
                    )?;
                }
                self.gemv_full(
                    self.lw(layer_idx, "ffn_down_shexp.weight")?,
                    self.scratch.act.as_ptr().cast(),
                    self.scratch.expert_out.as_ptr().cast(),
                )?;
                unsafe {
                    check(
                        ffi::add_assign_cuda(
                            moe_acc,
                            self.scratch.expert_out.as_ptr().cast(),
                            h as i32,
                            s,
                        ),
                        "shared add",
                    )?;
                }
            }
            Ok(())
        }
    }
}
