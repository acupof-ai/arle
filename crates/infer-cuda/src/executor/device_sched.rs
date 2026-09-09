//! Device scheduling for the qwen35 executor: the paged decode-graph lane
//! and the graph-invalidation policy.
//!
//! A captured decode graph replays against FIXED device addresses, so any
//! event that changes a baked address must drop the capture. The policy has
//! two shapes:
//!
//! - **Explicit events** ([`DecodeGraphInvalidation`]) — weight offload,
//!   scratch release, LoRA re-merge, Markov-head swap, capture failure.
//!   Whole-graph drops go through [`DecodeGraphSlot::invalidate`], the ONLY
//!   writer of the field: `decode_graph` is a newtype whose inner `Option`
//!   is private to this module, so `= None`, `.take()`, or an `&mut`
//!   handoff anywhere else fails to compile. Per-slot rebuilds go through
//!   [`Qwen35CudaExecutor::rebuild_slot_graph`]; `scripts/check_repo_hygiene.py`
//!   backstops that one, since a grep can be out-written but the field cannot.
//! - **Implicit drift** — the workspace re-addresses on its own, so
//!   [`Qwen35CudaExecutor::stage_graph_step`] compares the baked pointer
//!   set each step and rebuilds the one slot that drifted. No event site
//!   can name this class, so the guard lives at the stage point.

use super::*;

/// Why a captured decode graph is being dropped. Every event that changes
/// an address baked into a capture names itself here.
#[derive(Debug)]
pub(super) enum DecodeGraphInvalidation {
    /// Capture or replay failed: eager is the permanent fallback. The caller
    /// also disarms `decode_graph_armed` and reclaims the per-slot page
    /// tables.
    CaptureFailed,
    /// Device weights were offloaded to host; captures bake the old pointers.
    WeightsOffloaded,
    /// The forward scratch (workspace + batch_decode) was released.
    ScratchReleased,
    /// A student LoRA merge replaced the projection `DeviceMatrix` buffers.
    StudentLoraRemerged,
    /// The DSpark Markov head was hot-swapped from a host snapshot.
    DsparkMarkovUpdated,
}

static QWEN35_GRAPH_CAPTURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static QWEN35_GRAPH_REPLAYS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Device addresses a slot's captured decode graph was baked against: the graph
/// replays against FIXED pointers, so replaying a stale bake reads freed memory.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Qwen35GraphBake {
    token_ids_ptr: u64,
    start_pos_ptr: u64,
    logits_ptr: u64,
    ws_epoch: u64,
}

/// Per-slot decode-graph state on a DEDICATED `seq_len == 1` workspace — the main
/// workspace re-shapes on every prefill chunk and would invalidate captures.
pub(super) struct Qwen35DecodeGraph {
    ws: crate::qwen35::Qwen35Workspace,
    pub(super) graphs: Vec<crate::graph::CudaGraphState>,
    baked: Vec<Option<Qwen35GraphBake>>,
}

impl Qwen35DecodeGraph {
    fn new(num_slots: usize, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> Self {
        Self {
            ws: crate::qwen35::Qwen35Workspace::new(),
            graphs: (0..num_slots)
                // allow_alloc_nodes: this lane's capture has never been audited
                // for alloc nodes, and a short single-GPU smoke could not get it
                // to capture. Keeping the pre-2026-08-23 warn-only behaviour
                // rather than risking a silent fallback to eager; drop the call
                // once a capture here is measured at zero.
                .map(|_| crate::graph::CudaGraphState::new(stream.clone()).allow_alloc_nodes())
                .collect(),
            baked: vec![None; num_slots],
        }
    }
}

/// The executor's decode-graph slot. The inner `Option` is private to this
/// module on purpose: a whole-graph invalidation must name its reason through
/// [`DecodeGraphSlot::invalidate`], so `= None`, `.take()`, or an `&mut`
/// handoff at any other site fails to compile.
pub(super) struct DecodeGraphSlot(Option<Qwen35DecodeGraph>);

impl DecodeGraphSlot {
    pub(super) fn none() -> Self {
        Self(None)
    }

    pub(super) fn as_ref(&self) -> Option<&Qwen35DecodeGraph> {
        self.0.as_ref()
    }

    pub(super) fn as_mut(&mut self) -> Option<&mut Qwen35DecodeGraph> {
        self.0.as_mut()
    }

    /// Lazily build the per-slot graph state on the dedicated decode stream.
    pub(super) fn get_or_init(
        &mut self,
        num_slots: usize,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> &mut Qwen35DecodeGraph {
        if self.0.is_none() {
            self.0 = Some(Qwen35DecodeGraph::new(num_slots, stream));
        }
        self.0.as_mut().expect("just initialized")
    }

    /// Drop the capture for a named reason. The ONLY writer of the field.
    pub(super) fn invalidate(&mut self, why: DecodeGraphInvalidation) {
        if self.0.is_some() {
            info!("[qwen35-decode-graph] invalidated: {why:?}");
            self.0 = None;
        }
    }
}

/// Log the first decode-graph gate miss. A miss is silent by design (the eager
/// lane is the correctness floor), which makes "armed but never captured"
/// indistinguishable from "captured fine" in a log.
fn graph_gate_miss(reason: impl FnOnce() -> String) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| info!("[qwen35-decode-graph] gate miss: {}", reason()));
}

impl Qwen35CudaExecutor {
    /// Reset one slot's capture (new occupant: the recurrent block's device
    /// addresses changed). The ONLY writer of a per-slot `graphs[slot]` /
    /// `baked[slot]` reset outside construction.
    pub(super) fn rebuild_slot_graph(&mut self, slot: usize) {
        if let Some(dg) = self.decode_graph.as_mut() {
            dg.graphs[slot] = crate::graph::CudaGraphState::new(self.model.ctx.stream.clone())
                .allow_alloc_nodes();
            dg.baked[slot] = None;
        }
    }

    /// Stage per-step device scalars into the graph workspace and drop the
    /// slot's capture when any baked address drifted (release → re-alloc).
    fn stage_graph_step(
        model: &crate::qwen35::Qwen35Model,
        dg: &mut Qwen35DecodeGraph,
        slot: usize,
        last_token: u32,
        start_pos: usize,
        label: &str,
    ) -> Result<()> {
        let Qwen35DecodeGraph { ws, graphs, baked } = dg;
        let (token_ids_ptr, start_pos_ptr) =
            model.stage_step_inputs(ws, &[last_token], start_pos)?;
        let logits_ptr = model.workspace_logits_ptr(ws)?;
        let bake = Qwen35GraphBake {
            token_ids_ptr,
            start_pos_ptr,
            logits_ptr,
            ws_epoch: ws.epoch(),
        };
        match baked[slot] {
            Some(prev) if prev != bake => {
                info!(
                    "[qwen35-decode-graph] {label}slot {slot}: workspace addresses changed; \
                     dropping stale capture and recapturing"
                );
                graphs[slot] =
                    crate::graph::CudaGraphState::new(model.ctx.stream.clone()).allow_alloc_nodes();
                baked[slot] = Some(bake);
            }
            None => baked[slot] = Some(bake),
            _ => {}
        }
        Ok(())
    }

    /// Shared graph-lane epilogue: advance the slot, bump the replay counters, then
    /// sample OUTSIDE the graph from the logits the run just wrote.
    fn finish_graph_step(
        &mut self,
        slot: usize,
        was_captured: bool,
        will_replay: bool,
        label: &str,
        row: &DecodeRow,
        position: u64,
    ) -> Result<(u32, Option<f32>)> {
        // Host-side state advance happens here — captured closure is host-state-free.
        self.slots[slot].advance_seq_len(1);
        let dg = self.decode_graph.as_ref().expect("still present");
        if !was_captured && dg.graphs[slot].is_captured() {
            let captures =
                QWEN35_GRAPH_CAPTURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let keys = dg.graphs.iter().filter(|g| g.is_captured()).count();
            info!(
                "[qwen35-decode-graph] captured {label}slot {slot} \
                 (captures_total={captures}, live_keys={keys}, max_keys={})",
                self.num_slots
            );
        }
        if will_replay {
            let replays =
                QWEN35_GRAPH_REPLAYS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if replays.is_multiple_of(100) {
                info!(
                    "[qwen35-decode-graph] {label}replay_total={replays} captures_total={}",
                    QWEN35_GRAPH_CAPTURES.load(std::sync::atomic::Ordering::Relaxed)
                );
            }
        }
        let dg = self.decode_graph.as_mut().expect("still present");
        self.model.sample_workspace_logits(
            &mut dg.ws,
            &row.params,
            position,
            penalty_of(&row.penalty_history, row.penalty_prompt_len),
        )
    }

    /// Whole-step decode graph over the PAGED pool: the growing page table is absorbed
    /// by a fixed-capacity per-slot [`crate::loader::PageMeta::persistent_decode`]
    /// refreshed outside the graph, with FA3's scheduling ceiling pinned via
    /// `seqlen_k_capture`. `Ok(None)` on any gate miss.
    pub(super) fn try_graph_decode_paged(
        &mut self,
        row: &DecodeRow,
        position: u64,
        kv_batch: &KvBatchDescriptor,
    ) -> Result<Option<(u32, Option<f32>)>> {
        // BF16 captures the FA3 lane, whose scheduling ceiling `seqlen_k_capture`
        // pins. FP8/INT8 capture the split-KV lane instead: its grid is
        // `(kv_heads * num_splits, batch, q_tiles)` and the split count comes
        // from `quant_decode_num_splits` (sm_count / (batch * kv_heads), no KV
        // length), so the grid is fixed at B=1; the true per-row length is read
        // on device from `seqused_k`, and the workspace is pool-owned. Other
        // formats have no such decode kernel and stay eager.
        let capturable = match self.full_attn_kv.as_ref().map(|p| p.format) {
            Some(KVFormat::BF16) => self.model.paged_decode_fa3_active(),
            Some(KVFormat::FP8E4M3 | KVFormat::INT8) => true,
            _ => false,
        };
        if !self.decode_graph_armed || !capturable {
            graph_gate_miss(|| {
                format!(
                    "armed={} capturable={} kv_format={:?} fa3_active={}",
                    self.decode_graph_armed,
                    capturable,
                    self.full_attn_kv.as_ref().map(|p| p.format),
                    self.model.paged_decode_fa3_active(),
                )
            });
            return Ok(None);
        }
        if row.kv_seq_len + 1 > self.model.max_seq_len() {
            graph_gate_miss(|| {
                format!(
                    "kv_seq_len {} + 1 > max_seq_len {}",
                    row.kv_seq_len,
                    self.model.max_seq_len()
                )
            });
            return Ok(None);
        }
        let slot = row.slot;
        {
            let pool = self
                .full_attn_kv
                .as_ref()
                .expect("full_attn_kv present (full_attn_paged)");
            ensure!(
                pool.seq_len(slot) == row.kv_seq_len,
                "Qwen3.6 paged decode graph: pool seq_len {} != kv_seq_len {} for slot {}",
                pool.seq_len(slot),
                row.kv_seq_len,
                slot
            );
        }
        // Idempotent, so the eager fallback may re-run it.
        self.mirror_host_slot(kv_batch, slot, row.kv_seq_len + 1)?;
        self.decode_graph
            .get_or_init(self.num_slots, &self.model.ctx.stream);
        if self.paged_decode_meta.is_empty() {
            self.paged_decode_meta = (0..self.num_slots).map(|_| None).collect();
        }
        {
            let pool = self.full_attn_kv.as_ref().expect("full_attn_kv present");
            let capacity = self.model.max_seq_len().div_ceil(pool.page_size);
            let meta = match &mut self.paged_decode_meta[slot] {
                Some(meta) => meta,
                none => none.insert(crate::loader::PageMeta::persistent_decode(
                    &self.model.ctx,
                    pool.page_size,
                    capacity,
                    pool.format,
                )?),
            };
            meta.refresh_decode(&self.model.ctx, pool, slot, row.kv_seq_len)?;
        }
        let Self {
            model,
            slots,
            decode_graph,
            paged_decode_meta,
            full_attn_kv,
            ..
        } = self;
        let dg = decode_graph
            .as_mut()
            .expect("decode_graph built above when armed");
        Self::stage_graph_step(model, dg, slot, row.last_token, row.kv_seq_len, "paged ")?;
        let Qwen35DecodeGraph { ws, graphs, .. } = dg;
        let state = &mut graphs[slot];
        let was_captured = state.is_captured();
        let will_replay = was_captured && !state.is_armed_warm();
        let slot_state = &mut slots[slot];
        let pool = full_attn_kv.as_mut().expect("full_attn_kv present");
        let meta = paged_decode_meta[slot]
            .as_ref()
            .expect("persistent meta built above");
        let mut rc = crate::qwen35::Qwen35PagedForward {
            pool,
            meta,
            cp: None,
            cp_decode: None,
        };
        let run = state.run_or_capture(|| {
            model.forward_decode_step_paged_captured(slot_state, ws, row.kv_seq_len, &mut rc)
        });
        if let Err(e) = run {
            warn!(
                "Qwen3.5 paged whole-step decode graph failed (slot {slot}), \
                 downgrading to eager forward: {e}"
            );
            self.decode_graph_armed = false;
            self.decode_graph
                .invalidate(DecodeGraphInvalidation::CaptureFailed);
            // Resource reclamation, not a dangling-pointer fix:
            // `PageMeta::persistent_decode` allocates 8 device buffers of its
            // own (upload_i32), so weight offload / scratch release never
            // dangle them; the clear pairs with the permanent disarm above.
            self.paged_decode_meta.clear();
            return Ok(None);
        }
        let out = self.finish_graph_step(slot, was_captured, will_replay, "paged ", row, position);
        out.map(Some)
    }
}
