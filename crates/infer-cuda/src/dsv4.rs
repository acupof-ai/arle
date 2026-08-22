//! Multi-GPU only (256 FP8 experts + MLA sharding don't fit one GPU);
//! `ExpertSplit::single` is the dev/typecheck fallback.

use anyhow::{Result, anyhow, ensure};
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use deepseek_spec::{DeepSeekV4AttentionMode, DeepSeekV4Config, DeepSeekV4MoeRoutingKind};
use infer_moe::MoeConfig;
use infer_plan::SamplingParams;

use crate::moe_config::ExpertSplit;

#[path = "dsv4/budget.rs"]
mod budget;
#[path = "dsv4/decode_batch.rs"]
mod decode_batch;
#[path = "dsv4/dspark.rs"]
pub(crate) mod dspark;
#[path = "dsv4/forward_state.rs"]
mod forward_state;
#[path = "dsv4/head.rs"]
mod head;
#[path = "dsv4/layer_block.rs"]
mod layer_block;
#[path = "dsv4/load.rs"]
mod load;
#[path = "dsv4/mega_moe.rs"]
mod mega_moe;
#[path = "dsv4/mtp.rs"]
mod mtp;
#[path = "dsv4/prefill.rs"]
mod prefill;
#[path = "dsv4/probe.rs"]
mod probe;
#[path = "dsv4/slot.rs"]
mod slot;
#[path = "dsv4/slot_image.rs"]
mod slot_image;
#[path = "dsv4/spec_verify.rs"]
mod spec_verify;
#[path = "dsv4/weights.rs"]
mod weights;

pub(crate) use budget::Dsv4MlaKvArena;
pub(crate) use forward_state::*;
pub(crate) use load::load_dspark_draft;
#[cfg(all(feature = "cuda", feature = "nccl"))]
pub(crate) use mega_moe::Dsv4MegaMoeTransport;
pub(crate) use mtp::*;
use probe::Dsv4ProbeCapture;
pub(crate) use slot::*;
pub(crate) use slot_image::*;
pub(crate) use spec_verify::*;
pub(crate) use weights::*;

pub(crate) struct Dsv4Model {
    pub ctx: DeviceContext,
    pub config: DeepSeekV4Config,
    pub moe_config: MoeConfig,
    pub split: ExpertSplit,
    pub kv_arena: Dsv4MlaKvArena,
    pub embed_tokens: DeviceMatrix,
    pub lm_head: DeviceMatrix,
    pub layers: Vec<Dsv4Layer>,
    pub norm: DeviceVec,
    pub head_hc: Dsv4HyperConnection,
    pub mtp: Option<Dsv4MtpLayer>,
    /// Resolved at construction from `--spec-type mtp` / `dspark`. Per-slot
    /// construction reads this so the MTP-head load and the per-slot rollback
    /// snapshots agree on one decision.
    pub spec_decode_on: bool,
    pub tp: crate::tp::TpRuntime,
    /// Optional logit-lens probe (ARLE_PROBE_JSONL). RefCell for &self capture.
    pub probe: std::cell::RefCell<Option<Dsv4ProbeCapture>>,
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    pub(crate) mega_moe: Option<Dsv4MegaMoeTransport>,
    #[cfg(feature = "deepep")]
    pub deepep: Option<crate::deepep::DeepEpTransport>,
    /// c=1 decode graph: when true, `forward_tokens_stream_impl` skips per-step
    /// H2D copies and reuses persistent device buffers at fixed addresses.
    pub graph_mode: std::sync::atomic::AtomicBool,
    graph_token_ids: std::sync::Mutex<Option<CudaSlice<i32>>>,
    graph_token_ids_u32: std::sync::Mutex<Option<CudaSlice<u32>>>,
    graph_bufs: std::sync::Mutex<Vec<Option<HiddenStates>>>,
}

impl std::fmt::Debug for Dsv4Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dsv4Model")
            .field("layers", &self.layers.len())
            .field("hidden_size", &self.config.hidden_size)
            .field("heads", &self.config.num_attention_heads)
            .field("experts", &self.config.n_routed_experts)
            .field("experts_per_rank", &self.split.experts_per_rank)
            .field("kv_bytes_per_token", &self.kv_arena.bytes_per_token)
            .field("mtp_loaded", &self.mtp.is_some())
            .finish()
    }
}

/// Persistent-buffer indexing for the c=1 decode graph.
/// Index 0 = stream output; per-layer base = 1 + layer_idx * 7.
/// Per-layer offsets: 0=normed_attn 1=attn_out 2=attn_stream 3=normed_ffn
/// 4=moe_with_shared 5=moe_out 6=ffn_stream.
const fn graph_buf_idx_stream() -> usize {
    0
}
const fn graph_buf_idx_layer(layer_idx: usize, offset: usize) -> usize {
    1 + layer_idx * 7 + offset
}

impl Dsv4Model {
    /// Upload token IDs to the persistent graph buffer (called before capture/replay).
    pub(crate) fn graph_upload_token_ids(&self, tokens: &[u32]) -> Result<()> {
        let host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let mut buf = self.graph_token_ids.lock().unwrap();
        if buf.is_none() {
            *buf = Some(crate::ops::upload_i32(&self.ctx, &host)?);
        } else {
            self.ctx
                .stream
                .memcpy_htod(&host, buf.as_mut().unwrap())
                .map_err(|e| anyhow!("DSv4 graph token_ids H2D failed: {e}"))?;
        }
        let mut buf = self.graph_token_ids_u32.lock().unwrap();
        match buf.as_mut() {
            None => {
                *buf = Some(
                    self.ctx
                        .stream
                        .clone_htod(tokens)
                        .map_err(|e| anyhow!("DSv4 graph token_ids(u32) H2D failed: {e}"))?,
                )
            }
            Some(b) => self
                .ctx
                .stream
                .memcpy_htod(tokens, b)
                .map_err(|e| anyhow!("DSv4 graph token_ids(u32) H2D failed: {e}"))?,
        }
        Ok(())
    }

    /// Allocate (first call) or clone (subsequent calls) a persistent HiddenStates
    /// at the given buffer index. Clones share the same device memory (CudaSlice
    /// is ref-counted), so graph replay reads/writes the same addresses.
    fn graph_alloc_hidden(&self, idx: usize, dim: usize, seq_len: usize) -> Result<HiddenStates> {
        let mut bufs = self.graph_bufs.lock().unwrap();
        if bufs.len() <= idx {
            bufs.resize_with(idx + 1, || None);
        }
        if bufs[idx].is_none() {
            // SAFETY: fully written by the forward kernels before first read.
            bufs[idx] = Some(unsafe { HiddenStates::uninit(&self.ctx, dim, seq_len)? });
        }
        let b = bufs[idx].as_ref().unwrap();
        Ok(HiddenStates {
            data: b.data.clone(),
            hidden_dim: b.hidden_dim,
            seq_len: b.seq_len,
        })
    }

    /// Persistent u32 token ids for hash routing (same pre-replay upload).
    pub(crate) fn graph_token_ids_u32(&self) -> Result<CudaSlice<u32>> {
        self.graph_token_ids_u32
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| anyhow!("DSv4 graph token_ids not uploaded"))
    }

    /// Clone of the persistent stream output buffer (index 0).
    /// Only valid after the first graph-mode forward allocated it.
    pub(crate) fn graph_stream_clone(&self) -> Result<HiddenStates> {
        let bufs = self.graph_bufs.lock().unwrap();
        let b = bufs
            .first()
            .and_then(|b| b.as_ref())
            .ok_or_else(|| anyhow!("DSv4 graph stream buffer not allocated"))?;
        Ok(HiddenStates {
            data: b.data.clone(),
            hidden_dim: b.hidden_dim,
            seq_len: b.seq_len,
        })
    }

    /// True when the c=1 decode graph should use persistent buffers.
    pub(crate) fn graph_mode(&self) -> bool {
        self.graph_mode.load(std::sync::atomic::Ordering::Relaxed)
    }
}
