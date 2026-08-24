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
    graph_bufs: std::sync::Mutex<std::collections::HashMap<GraphBufKey, HiddenStates>>,
    graph_f32: std::sync::Mutex<std::collections::HashMap<GraphBufKey, CudaSlice<f32>>>,
    graph_i32: std::sync::Mutex<std::collections::HashMap<GraphBufKey, CudaSlice<i32>>>,
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

/// Persistent decode-graph buffer identity: `(layer, slot)`; `layer ==
/// usize::MAX` is the model-level slot (embedding stream, token ids).
pub(crate) type GraphBufKey = (usize, GraphSlot);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum GraphSlot {
    Stream,
    Embeddings,
    NormedAttn,
    AttnOut,
    OprojLatent,
    AttnStream,
    NormedFfn,
    MoeWithShared,
    MoeOut,
    FfnStream,
    RouterLogits,
    RouteIndices,
    RouteWeights,
    HcMixes(u8),
    HcPre(u8),
    HcPost(u8),
    HcComb(u8),
}

/// A per-step activation buffer: owned (freed on drop) or an alias of a
/// persistent graph buffer (never freed; the owner lives in `graph_bufs`).
pub(crate) enum StepBuf {
    Owned(HiddenStates),
    Alias(std::mem::ManuallyDrop<HiddenStates>),
}

impl std::ops::Deref for StepBuf {
    type Target = HiddenStates;
    fn deref(&self) -> &HiddenStates {
        match self {
            StepBuf::Owned(h) => h,
            StepBuf::Alias(h) => h,
        }
    }
}

/// Typed per-step scratch: owned or a never-freed alias (see `StepBuf`).
pub(crate) enum StepSlice<T> {
    Owned(CudaSlice<T>),
    Alias(std::mem::ManuallyDrop<CudaSlice<T>>),
}

impl<T> std::ops::Deref for StepSlice<T> {
    type Target = CudaSlice<T>;
    fn deref(&self) -> &CudaSlice<T> {
        match self {
            StepSlice::Owned(h) => h,
            StepSlice::Alias(h) => h,
        }
    }
}

impl<T> std::ops::DerefMut for StepSlice<T> {
    fn deref_mut(&mut self) -> &mut CudaSlice<T> {
        match self {
            StepSlice::Owned(h) => h,
            StepSlice::Alias(h) => h,
        }
    }
}

impl std::ops::DerefMut for StepBuf {
    fn deref_mut(&mut self) -> &mut HiddenStates {
        match self {
            StepBuf::Owned(h) => h,
            StepBuf::Alias(h) => h,
        }
    }
}

pub(crate) const GRAPH_MODEL_LAYER: usize = usize::MAX;
pub(crate) const GRAPH_MTP_LAYER: usize = usize::MAX - 1;
pub(crate) const GRAPH_DSPARK_LAYER_BASE: usize = usize::MAX - 1024;

impl Dsv4Model {
    /// Upload token IDs to the persistent graph buffer (called before capture/replay).
    pub(crate) fn graph_upload_token_ids(&self, tokens: &[u32]) -> Result<()> {
        let host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let mut buf = self
            .graph_token_ids
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if buf.is_none() {
            *buf = Some(crate::ops::upload_i32(&self.ctx, &host)?);
        } else {
            self.ctx
                .stream
                .memcpy_htod(&host, buf.as_mut().unwrap())
                .map_err(|e| anyhow!("DSv4 graph token_ids H2D failed: {e}"))?;
        }
        let mut buf = self
            .graph_token_ids_u32
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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

    /// Alias of a persistent device buffer sharing the same pointer.
    /// `CudaSlice::clone` is a D2D copy into a fresh allocation, so the capture
    /// must alias the raw pointer for replay to read/write the same memory.
    fn graph_alias_slice<T: cudarc::driver::DeviceRepr>(
        &self,
        b: &CudaSlice<T>,
    ) -> std::mem::ManuallyDrop<CudaSlice<T>> {
        use cudarc::driver::DevicePtr;
        let (ptr, _g) = b.device_ptr(&self.ctx.stream);
        // SAFETY: `ptr` is the live persistent allocation held by the model's
        // graph pools for its lifetime; the alias is never freed.
        std::mem::ManuallyDrop::new(unsafe {
            self.ctx.stream.upgrade_device_ptr::<T>(ptr, b.len())
        })
    }

    fn graph_alias(&self, b: &HiddenStates) -> StepBuf {
        let data = std::mem::ManuallyDrop::into_inner(self.graph_alias_slice(&b.data));
        StepBuf::Alias(std::mem::ManuallyDrop::new(HiddenStates {
            data,
            hidden_dim: b.hidden_dim,
            seq_len: b.seq_len,
        }))
    }

    /// Per-step activation: a persistent graph buffer alias in graph mode,
    /// else a fresh transient allocation.
    pub(crate) fn step_hidden(
        &self,
        graph_mode: bool,
        key: GraphBufKey,
        dim: usize,
        seq_len: usize,
    ) -> Result<StepBuf> {
        if !graph_mode {
            // SAFETY: fully written before first read.
            return Ok(StepBuf::Owned(unsafe {
                HiddenStates::uninit(&self.ctx, dim, seq_len)?
            }));
        }
        let mut bufs = self.graph_bufs.lock().unwrap_or_else(|e| e.into_inner());
        let b = match bufs.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                // SAFETY: fully written by the forward kernels before first read.
                e.insert(unsafe { HiddenStates::uninit(&self.ctx, dim, seq_len)? })
            }
        };
        ensure!(
            b.hidden_dim == dim && b.seq_len == seq_len,
            "DSv4 graph buffer {key:?} shape {}x{} != {dim}x{seq_len}",
            b.hidden_dim,
            b.seq_len
        );
        Ok(self.graph_alias(b))
    }

    /// Per-step f32 scratch (mHC lane weights): persistent alias in graph mode.
    pub(crate) fn step_f32(
        &self,
        graph_mode: bool,
        key: GraphBufKey,
        len: usize,
    ) -> Result<StepSlice<f32>> {
        if !graph_mode {
            // SAFETY: fully written before first read.
            return Ok(StepSlice::Owned(unsafe {
                self.ctx.stream.alloc::<f32>(len)?
            }));
        }
        let mut bufs = self.graph_f32.lock().unwrap_or_else(|e| e.into_inner());
        let b = match bufs.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(self.ctx.stream.alloc_zeros::<f32>(len)?)
            }
        };
        ensure!(
            b.len() == len,
            "DSv4 graph f32 buffer {key:?} len {} != {len}",
            b.len()
        );
        Ok(StepSlice::Alias(self.graph_alias_slice(b)))
    }

    /// Per-step i32 scratch (route indices): persistent alias in graph mode.
    pub(crate) fn step_i32(
        &self,
        graph_mode: bool,
        key: GraphBufKey,
        len: usize,
    ) -> Result<StepSlice<i32>> {
        if !graph_mode {
            return Ok(StepSlice::Owned(self.ctx.stream.alloc_zeros::<i32>(len)?));
        }
        let mut bufs = self.graph_i32.lock().unwrap_or_else(|e| e.into_inner());
        let b = match bufs.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(self.ctx.stream.alloc_zeros::<i32>(len)?)
            }
        };
        ensure!(
            b.len() == len,
            "DSv4 graph i32 buffer {key:?} len {} != {len}",
            b.len()
        );
        Ok(StepSlice::Alias(self.graph_alias_slice(b)))
    }

    /// Persistent u32 token ids for hash routing (same pre-replay upload).
    pub(crate) fn graph_token_ids_u32(&self) -> Result<StepSlice<u32>> {
        let buf = self
            .graph_token_ids_u32
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let b = buf
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 graph token_ids not uploaded"))?;
        Ok(StepSlice::Alias(self.graph_alias_slice(b)))
    }

    /// Persistent i32 token ids for the embedding lookup (pre-replay upload).
    pub(crate) fn graph_token_ids_i32(&self) -> Result<StepSlice<i32>> {
        let buf = self
            .graph_token_ids
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let b = buf
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 graph token_ids not uploaded"))?;
        Ok(StepSlice::Alias(self.graph_alias_slice(b)))
    }

    /// Alias of the persistent final-layer output stream (the last layer's
    /// ffn_stream), read by the LM head after a replay.
    pub(crate) fn graph_stream_clone(&self) -> Result<StepBuf> {
        let bufs = self.graph_bufs.lock().unwrap_or_else(|e| e.into_inner());
        let last = self
            .layers
            .len()
            .checked_sub(1)
            .ok_or_else(|| anyhow!("DSv4 no layers"))?;
        let b = bufs
            .get(&(last, GraphSlot::FfnStream))
            .ok_or_else(|| anyhow!("DSv4 graph output stream buffer not allocated"))?;
        Ok(self.graph_alias(b))
    }

    /// True when the c=1 decode graph should use persistent buffers.
    pub(crate) fn graph_mode(&self) -> bool {
        self.graph_mode.load(std::sync::atomic::Ordering::Relaxed)
    }
}
