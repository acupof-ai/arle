use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt,
    io::{BufWriter, Write},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use smallvec::SmallVec;

use crate::{
    AutogradError, Result,
    backend::DeviceHandle,
    ops,
    ops::chunk_accum::{ChunkSum, SeqAccum},
    tensor::{Dirty, TensorId, TensorStore},
};

// `Dirty` is used both by the pre-existing batched-flush filter (line ~176)
// and by the P2 device-residency gate inside `merge_grad`.

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum SavedContext {
    None,
    Tensor(TensorId),
    Tensors(SmallVec<[TensorId; 4]>),
    TensorAndScalar(TensorId, f32),
    Shape(Vec<usize>),
    AllToAllCtx {
        in_shape: Vec<usize>,
        scatter_axis: usize,
        gather_axis: usize,
    },
    MatmulCtx {
        a: TensorId,
        b: TensorId,
    },
    MatmulBTCtx {
        a: TensorId,
        b: TensorId,
        site: &'static str,
    },
    SoftmaxCtx {
        y: TensorId,
    },
    LogSoftmaxCtx {
        y: TensorId,
    },
    GatherCtx {
        indices: Vec<usize>,
        src_shape: Vec<usize>,
    },
    MoeTopKSoftmaxCtx {
        y: TensorId,
        indices: Vec<usize>,
        logits_shape: Vec<usize>,
        top_k: usize,
    },
    MoeGatherRowsCtx {
        rows: Vec<usize>,
        input_shape: Vec<usize>,
    },
    MoeWeightedScatterCtx {
        routes: Vec<ops::moe::MoeRoute>,
        values_shape: Vec<usize>,
        weights_shape: Vec<usize>,
        out_rows: usize,
    },
    MoeGroupedLinearCtx {
        input: TensorId,
        experts: Vec<ops::moe::MoeGroupedLinearExpert>,
        routes: Vec<ops::moe::MoeGroupedRoute>,
        input_kind: ops::moe::MoeGroupedLinearInput,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },
    MoeGroupedWeightedScatterCtx {
        routes: Vec<ops::moe::MoeGroupedRoute>,
        values_shape: Vec<usize>,
        weights_shape: Vec<usize>,
        out_rows: usize,
    },
    MeanCtx {
        input: TensorId,
        numel: usize,
    },
    RMSNormCtx {
        x: TensorId,
        weight: TensorId,
        inv_rms: Vec<f32>,
        eps: f32,
    },
    SiluCtx {
        x: TensorId,
    },
    SigmoidCtx {
        y: TensorId,
    },
    GeluCtx {
        x: TensorId,
    },
    RoPECtx {
        cos: TensorId,
        sin: TensorId,
    },
    ReshapeCtx {
        input_shape: Vec<usize>,
    },
    BroadcastExpandCtx {
        src_shape: Vec<usize>,
    },
    SliceCtx {
        input_shape: Vec<usize>,
        starts: Vec<usize>,
        ends: Vec<usize>,
    },
    CatHeadsCtx {
        head_counts: Vec<usize>,
    },
    CatSeqCtx {
        seq_counts: Vec<usize>,
    },
    CatCtx {
        axis: usize,
        input_shapes: Vec<Vec<usize>>,
    },
    TransposeCtx {
        axis1: usize,
        axis2: usize,
    },
    AddBroadcastCtx {
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
    },
    EmbeddingCtx {
        indices: Vec<usize>,
        table_shape: Vec<usize>,
    },
    FusedLinearDistillCtx {
        grad_hidden: Option<TensorId>,
        grad_weight: Option<TensorId>,
    },
    GeneralizedJsdCtx {
        grad_student: TensorId,
    },
    LinearAttentionCtx {
        qkv: TensorId,
        z: TensorId,
        b_proj: TensorId,
        a_proj: TensorId,
        conv1d_weight: TensorId,
        dt_bias: TensorId,
        a_log: TensorId,
        norm_weight: TensorId,
        preact: Option<TensorId>,
        qkv_conv: Option<TensorId>,
        g: Option<TensorId>,
        beta: Option<TensorId>,
        /// FlashQLA route: chunk 0 only (the state carry). Otherwise every chunk.
        chunk_state: Option<TensorId>,
        /// FlashQLA route only: the pre-norm GDN output its backward differentiates.
        raw_output: Option<TensorId>,
        /// Which forward route ran. The runtime flag can flip between calls, so
        /// the backward reads this instead of re-checking the flag.
        flashqla: bool,
        /// OPD frozen-prompt-KV carry: seeds the recurrent state + conv window
        /// from a prior (prompt) segment so the backward recompute reproduces the
        /// forward exactly. `None` for the default full-sequence path.
        initial_state: Option<TensorId>,
        initial_conv_window: Option<TensorId>,
        batch: usize,
        seq_len: usize,
        num_key_heads: usize,
        num_value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        conv_kernel: usize,
        eps: f32,
    },
    CausalSdpaRecomputeCtx {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        // q's absolute start position over the KV; 0 for square (full-seq) causal
        // attention, >0 for a context-parallel query shard over a gathered prefix.
        q_start: usize,
    },
    CheckpointCtx {
        function_id: usize,
    },
    SeqChunkedRecomputeCtx {
        function_id: usize,
        batch: usize,
        seq: usize,
        dim: usize,
        chunk: usize,
    },
    // Expert-parallel dispatch/combine plan: `src[slot]` is the source token row
    // (usize::MAX = capacity drop). `dim` is the row width. Backward applies the
    // transpose permutation (dispatch↔combine).
    EpPlanCtx {
        input: TensorId,
        src: Vec<usize>,
        num_tokens: usize,
        dim: usize,
    },
    // Expert-parallel row exchange: per-peer row counts; backward swaps them.
    EpExchangeCtx {
        input: TensorId,
        send_counts: Vec<usize>,
        recv_counts: Vec<usize>,
        dim: usize,
    },
    // Ring-attention context-parallel tile: `blocks` are (k, v, k_abs) TensorIds
    // ring-delivered in forward order; `lse`/`out` are the saved per-row logsumexp
    // and normalized output the flash-2 backward replays against. `cp_size`/`cp_rank`
    // drive the device ring's rotation + grad ring-back (1/0 = single-block host path).
    RingAttentionCtx {
        q: TensorId,
        /// `(k, v, k_pos)` per ring block; `k_pos[c]` = absolute position of the
        /// block's col c (a Vec, not a scalar base, so zigzag shards mask right).
        blocks: SmallVec<[(TensorId, TensorId, Vec<usize>); 4]>,
        lse: TensorId,
        out: TensorId,
        rows: usize,
        dim: usize,
        /// Absolute position of each local q row (Vec — zigzag rows are not contiguous).
        q_pos: Vec<usize>,
        cp_size: usize,
        cp_rank: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BackwardOp {
    Add,
    Mul,
    MulScalar,
    Exp,
    Sum,
    Matmul,
    MatmulBT,
    Softmax,
    LogSoftmax,
    Gather,
    MoeTopKSoftmax,
    MoeGatherRows,
    MoeWeightedScatter,
    MoeGroupedLinear,
    MoeGroupedWeightedScatter,
    Mean,
    RMSNorm,
    Silu,
    Sigmoid,
    Gelu,
    RoPE,
    Reshape,
    BroadcastExpand,
    Slice,
    CatHeads,
    CatSeq,
    Cat,
    Transpose,
    AddBroadcast,
    Embedding,
    FusedLinearDistill,
    GeneralizedJsd,
    LinearAttention,
    CausalSdpaRecompute,
    AllReduceSum,
    AllGatherSeq,
    ReduceScatterSum,
    AllToAll,
    EpDispatch,
    EpCombine,
    EpExchange,
    RingAttention,
    Checkpoint,
    SeqChunkedRecompute,
}

impl BackwardOp {
    pub fn name(self) -> &'static str {
        match self {
            BackwardOp::Add => "Add",
            BackwardOp::Mul => "Mul",
            BackwardOp::MulScalar => "MulScalar",
            BackwardOp::Exp => "Exp",
            BackwardOp::Sum => "Sum",
            BackwardOp::Matmul => "Matmul",
            BackwardOp::MatmulBT => "MatmulBT",
            BackwardOp::Softmax => "Softmax",
            BackwardOp::LogSoftmax => "LogSoftmax",
            BackwardOp::Gather => "Gather",
            BackwardOp::MoeTopKSoftmax => "MoeTopKSoftmax",
            BackwardOp::MoeGatherRows => "MoeGatherRows",
            BackwardOp::MoeWeightedScatter => "MoeWeightedScatter",
            BackwardOp::MoeGroupedLinear => "MoeGroupedLinear",
            BackwardOp::MoeGroupedWeightedScatter => "MoeGroupedWeightedScatter",
            BackwardOp::Mean => "Mean",
            BackwardOp::RMSNorm => "RMSNorm",
            BackwardOp::Silu => "Silu",
            BackwardOp::Sigmoid => "Sigmoid",
            BackwardOp::Gelu => "Gelu",
            BackwardOp::RoPE => "RoPE",
            BackwardOp::Reshape => "Reshape",
            BackwardOp::BroadcastExpand => "BroadcastExpand",
            BackwardOp::Slice => "Slice",
            BackwardOp::CatHeads => "CatHeads",
            BackwardOp::CatSeq => "CatSeq",
            BackwardOp::Cat => "Cat",
            BackwardOp::Transpose => "Transpose",
            BackwardOp::AddBroadcast => "AddBroadcast",
            BackwardOp::Embedding => "Embedding",
            BackwardOp::FusedLinearDistill => "FusedLinearDistill",
            BackwardOp::GeneralizedJsd => "GeneralizedJsd",
            BackwardOp::LinearAttention => "LinearAttention",
            BackwardOp::CausalSdpaRecompute => "CausalSdpaRecompute",
            BackwardOp::AllReduceSum => "AllReduceSum",
            BackwardOp::AllGatherSeq => "AllGatherSeq",
            BackwardOp::ReduceScatterSum => "ReduceScatterSum",
            BackwardOp::AllToAll => "AllToAll",
            BackwardOp::EpDispatch => "EpDispatch",
            BackwardOp::EpCombine => "EpCombine",
            BackwardOp::EpExchange => "EpExchange",
            BackwardOp::RingAttention => "RingAttention",
            BackwardOp::Checkpoint => "Checkpoint",
            BackwardOp::SeqChunkedRecompute => "SeqChunkedRecompute",
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BackwardOpProfile {
    pub count: usize,
    pub duration: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct BackwardProfile {
    pub op_totals: BTreeMap<BackwardOp, BackwardOpProfile>,
    pub site_totals: BTreeMap<(BackwardOp, &'static str), BackwardOpProfile>,
    pub merge_grad_duration: Duration,
    pub prelude_duration: Duration,
    pub total_duration: Duration,
}

impl BackwardProfile {
    fn record_op(&mut self, op: BackwardOp, duration: Duration) {
        let entry = self.op_totals.entry(op).or_default();
        entry.count += 1;
        entry.duration += duration;
    }

    fn record_site(&mut self, op: BackwardOp, site: &'static str, duration: Duration) {
        let entry = self.site_totals.entry((op, site)).or_default();
        entry.count += 1;
        entry.duration += duration;
    }

    pub fn merge(&mut self, other: &Self) {
        for (&op, &stats) in &other.op_totals {
            let entry = self.op_totals.entry(op).or_default();
            entry.count += stats.count;
            entry.duration += stats.duration;
        }
        for (&site, &stats) in &other.site_totals {
            let entry = self.site_totals.entry(site).or_default();
            entry.count += stats.count;
            entry.duration += stats.duration;
        }
        self.merge_grad_duration += other.merge_grad_duration;
        self.prelude_duration += other.prelude_duration;
        self.total_duration += other.total_duration;
    }

    pub fn total_op_duration(&self) -> Duration {
        self.op_totals
            .values()
            .fold(Duration::default(), |acc, stats| acc + stats.duration)
    }
}

#[derive(Debug, Clone)]
pub struct TapeEntry {
    pub op: BackwardOp,
    pub output_id: TensorId,
    pub input_ids: SmallVec<[TensorId; 2]>,
    pub saved: SavedContext,
}

impl TapeEntry {
    pub fn profile_site(&self) -> Option<&'static str> {
        match &self.saved {
            SavedContext::MatmulBTCtx { site, .. } => Some(*site),
            _ => None,
        }
    }

    /// The output's `requires_grad` is derived here — the OR over `input_ids` — so
    /// a call site can't forget it, or let the mark drift away from its alloc. The
    /// mark lands even on a disabled tape, which is what a `checkpoint` inner
    /// replay needs. Fast-path legal only while `set_requires_grad` leaves device
    /// residency alone.
    pub fn record(self, store: &mut TensorStore, tape: &mut Tape) -> Result<()> {
        let requires_grad = store.any_requires_grad(&self.input_ids);
        store.set_requires_grad(self.output_id, requires_grad)?;
        if tape.enabled && requires_grad {
            tape.entries.push(self);
        }
        Ok(())
    }
}

pub(crate) type GradPairs = SmallVec<[(TensorId, TensorId); 2]>;

pub(crate) type CheckpointFn =
    Arc<dyn Fn(&mut TensorStore, &mut Tape, usize, &[TensorId]) -> Result<TensorId> + Send + Sync>;

#[derive(Debug, Clone)]
struct CheckpointOpMemRecord {
    stage: &'static str,
    op_seq: Option<usize>,
    op: Option<BackwardOp>,
    site: Option<&'static str>,
    pool: Option<(u64, u64)>,
    live_tensors: usize,
}

#[derive(Debug)]
struct CheckpointOpMemScope {
    checkpoint_fn: usize,
    next_op_seq: usize,
    records: Vec<CheckpointOpMemRecord>,
}

impl CheckpointOpMemScope {
    fn new(checkpoint_fn: usize) -> Self {
        Self {
            checkpoint_fn,
            next_op_seq: 0,
            records: Vec::with_capacity(512),
        }
    }

    fn next_op_seq(&mut self) -> usize {
        let op_seq = self.next_op_seq;
        self.next_op_seq += 1;
        op_seq
    }
}

#[derive(Default)]
pub struct Tape {
    pub entries: Vec<TapeEntry>,
    pub enabled: bool,
    checkpoint_fns: Vec<CheckpointFn>,
    pub(crate) offload_checkpoints: bool,
    skip_next_checkpoint_input_offload: bool,
    checkpoint_op_mem_scope: Option<Arc<Mutex<CheckpointOpMemScope>>>,
    checkpoint_op_mem_disarmed: bool,
}

impl fmt::Debug for Tape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tape")
            .field("entries", &self.entries)
            .field("enabled", &self.enabled)
            .field("checkpoint_fns_len", &self.checkpoint_fns.len())
            .finish()
    }
}

impl Tape {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            enabled: true,
            checkpoint_fns: Vec::new(),
            offload_checkpoints: false,
            skip_next_checkpoint_input_offload: false,
            checkpoint_op_mem_scope: None,
            checkpoint_op_mem_disarmed: false,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// When enabled, `checkpoint()` offloads its saved input tensors to host RAM
    /// after the forward (re-fetched on backward replay), trading PCIe traffic
    /// for less VRAM on long training forwards.
    pub fn set_offload_checkpoints(&mut self, on: bool) {
        self.offload_checkpoints = on;
    }

    pub fn offload_checkpoints(&self) -> bool {
        self.offload_checkpoints
    }

    pub(crate) fn set_skip_next_checkpoint_input_offload(&mut self, skip: bool) {
        self.skip_next_checkpoint_input_offload = skip;
    }

    pub(crate) fn take_skip_next_checkpoint_input_offload(&mut self) -> bool {
        std::mem::take(&mut self.skip_next_checkpoint_input_offload)
    }

    pub(crate) fn register_checkpoint_fn(&mut self, checkpoint_fn: CheckpointFn) -> usize {
        let function_id = self.checkpoint_fns.len();
        self.checkpoint_fns.push(checkpoint_fn);
        function_id
    }

    /// Number of registered checkpoint replay closures (one per checkpoint
    /// group). OPD VRAM attribution only — each closure captures host-side
    /// `Arc`s (the layer stack), not per-group device tensors.
    pub(crate) fn checkpoint_fn_count(&self) -> usize {
        self.checkpoint_fns.len()
    }

    fn checkpoint_op_mem_record(
        &self,
        stage: &'static str,
        op_seq: Option<usize>,
        entry: Option<&TapeEntry>,
        store: &TensorStore,
    ) {
        let Some(scope) = &self.checkpoint_op_mem_scope else {
            return;
        };
        let mut scope = scope.lock().expect("checkpoint op memory scope poisoned");
        scope.records.push(CheckpointOpMemRecord {
            stage,
            op_seq,
            op: entry.map(|entry| entry.op),
            site: entry.and_then(TapeEntry::profile_site),
            pool: store.backend().mem_pool_stats(),
            live_tensors: store.live_tensor_count(),
        });
    }

    fn checkpoint_op_mem_begin(&self, entry: &TapeEntry, store: &TensorStore) -> Option<usize> {
        let scope = self.checkpoint_op_mem_scope.as_ref()?;
        let mut scope = scope.lock().expect("checkpoint op memory scope poisoned");
        let op_seq = scope.next_op_seq();
        scope.records.push(CheckpointOpMemRecord {
            stage: "pre_op",
            op_seq: Some(op_seq),
            op: Some(entry.op),
            site: entry.profile_site(),
            pool: store.backend().mem_pool_stats(),
            live_tensors: store.live_tensor_count(),
        });
        Some(op_seq)
    }

    pub fn backward(
        &mut self,
        loss_id: TensorId,
        store: &mut TensorStore,
    ) -> Result<HashMap<TensorId, TensorId>> {
        self.backward_impl(loss_id, store, None, true)
    }

    pub fn backward_profiled(
        &mut self,
        loss_id: TensorId,
        store: &mut TensorStore,
    ) -> Result<(HashMap<TensorId, TensorId>, BackwardProfile)> {
        let mut profile = BackwardProfile::default();
        let grads = self.backward_impl(loss_id, store, Some(&mut profile), true)?;
        Ok((grads, profile))
    }

    pub fn backward_accumulate_only(
        &mut self,
        loss_id: TensorId,
        store: &mut TensorStore,
    ) -> Result<()> {
        let targets = HashSet::new();
        self.backward_impl_seed(loss_id, None, store, None, true, Some(&targets))?;
        Ok(())
    }

    pub fn backward_accumulate_only_profiled(
        &mut self,
        loss_id: TensorId,
        store: &mut TensorStore,
    ) -> Result<BackwardProfile> {
        let mut profile = BackwardProfile::default();
        let targets = HashSet::new();
        self.backward_impl_seed(
            loss_id,
            None,
            store,
            Some(&mut profile),
            true,
            Some(&targets),
        )?;
        Ok(profile)
    }

    pub fn backward_accumulate_targets(
        &mut self,
        loss_id: TensorId,
        store: &mut TensorStore,
        target_ids: &[TensorId],
    ) -> Result<HashMap<TensorId, TensorId>> {
        let grads = self.backward_impl(loss_id, store, None, false)?;
        for &target_id in target_ids {
            if let Some(&grad_id) = grads.get(&target_id) {
                store.accumulate_grad(target_id, grad_id)?;
            }
        }
        Ok(grads)
    }

    pub fn backward_from_seed_accumulate_targets(
        &mut self,
        output_id: TensorId,
        seed_grad_id: TensorId,
        store: &mut TensorStore,
        target_ids: &[TensorId],
    ) -> Result<HashMap<TensorId, TensorId>> {
        let grads =
            self.backward_impl_seed(output_id, Some(seed_grad_id), store, None, false, None)?;
        for &target_id in target_ids {
            if let Some(&grad_id) = grads.get(&target_id) {
                store.accumulate_grad(target_id, grad_id)?;
            }
        }
        Ok(grads)
    }

    pub fn backward_from_seed_accumulate_targets_profiled(
        &mut self,
        output_id: TensorId,
        seed_grad_id: TensorId,
        store: &mut TensorStore,
        target_ids: &[TensorId],
    ) -> Result<(HashMap<TensorId, TensorId>, BackwardProfile)> {
        let mut profile = BackwardProfile::default();
        let grads = self.backward_impl_seed(
            output_id,
            Some(seed_grad_id),
            store,
            Some(&mut profile),
            false,
            None,
        )?;
        for &target_id in target_ids {
            if let Some(&grad_id) = grads.get(&target_id) {
                store.accumulate_grad(target_id, grad_id)?;
            }
        }
        Ok((grads, profile))
    }

    pub fn backward_collect(
        &mut self,
        loss_id: TensorId,
        store: &mut TensorStore,
    ) -> Result<HashMap<TensorId, TensorId>> {
        self.backward_impl(loss_id, store, None, false)
    }

    fn backward_collect_targets_only(
        &mut self,
        loss_id: TensorId,
        store: &mut TensorStore,
        target_ids: &[TensorId],
        profile: Option<&mut BackwardProfile>,
    ) -> Result<HashMap<TensorId, TensorId>> {
        let targets = target_ids.iter().copied().collect::<HashSet<_>>();
        self.backward_impl_seed(loss_id, None, store, profile, false, Some(&targets))
    }

    fn backward_impl(
        &mut self,
        loss_id: TensorId,
        store: &mut TensorStore,
        profile: Option<&mut BackwardProfile>,
        accumulate_into_store: bool,
    ) -> Result<HashMap<TensorId, TensorId>> {
        self.backward_impl_seed(loss_id, None, store, profile, accumulate_into_store, None)
    }

    fn backward_impl_seed(
        &mut self,
        loss_id: TensorId,
        seed_grad_id: Option<TensorId>,
        store: &mut TensorStore,
        mut profile: Option<&mut BackwardProfile>,
        accumulate_into_store: bool,
        return_filter: Option<&HashSet<TensorId>>,
    ) -> Result<HashMap<TensorId, TensorId>> {
        let total_started = profile.is_some().then(Instant::now);
        let was_enabled = self.enabled;
        self.enabled = false;

        let result = (|| {
            let prelude_started = profile.is_some().then(Instant::now);
            // Batch-flush all Dirty::Device tape outputs in a single
            // `mlx_eval` call before walking the backward graph. The naive
            // per-id `ensure_host` loop would call `eval` once per handle —
            // a regression with lazy `sum`, because both `y`
            // and `loss` end up Dirty::Device and each per-id eval crosses
            // the FFI boundary + grabs the shared MLX guard. MLX consumes the batch
            // as one graph realization (terminal handles share upstream
            // nodes), so subsequent per-id `readback`s are O(copy) only.
            let device_ids: Vec<TensorId> = self
                .entries
                .iter()
                .filter(|entry| {
                    store
                        .get(entry.output_id)
                        .is_some_and(|tensor| tensor.dirty == Dirty::Device)
                })
                .map(|entry| entry.output_id)
                .collect();
            // only flush all tape outputs to host upfront when the
            // backend prefers it (Metal). On CUDA this batch readback is
            // the 1 GB DtoH that per-op lazy readback avoids — strictly cheaper
            // because device-resident backward ops never need the host
            // snapshot.
            if store.backend().prefers_pre_backward_flush() {
                store.flush_to_host_batch(&device_ids)?;
            }

            let mut entry_by_output = HashMap::with_capacity(self.entries.len());
            for (index, entry) in self.entries.iter().enumerate() {
                entry_by_output.insert(entry.output_id, index);
            }

            let mut relevant_tensors = HashSet::new();
            let mut visited_outputs = HashSet::new();
            let mut post_order = Vec::new();
            collect_relevant(
                loss_id,
                &entry_by_output,
                &self.entries,
                &mut relevant_tensors,
                &mut visited_outputs,
                &mut post_order,
            );

            let mut grads = HashMap::new();
            let loss_grad_id = if let Some(seed_grad_id) = seed_grad_id {
                let expected = store.tensor(loss_id)?.shape.clone();
                let got = store.tensor(seed_grad_id)?.shape.clone();
                if expected != got {
                    return Err(AutogradError::GradientShapeMismatch {
                        tensor_id: loss_id,
                        expected,
                        got,
                    });
                }
                seed_grad_id
            } else {
                store.fill_like(loss_id, 1.0)?
            };
            // seed the backward chain with a device-resident `1.0`
            // when the backend has device residency. Without this the
            // every-op `device_path_ok` gate
            // falls through to host fallback, because `g.dirty=Host`
            // from the first step. Explicit seed gradients follow the same
            // device-residency rule so downstream ops stay on-device.
            store.ensure_device(loss_grad_id)?;
            let loss_is_tape_output = entry_by_output.contains_key(&loss_id);
            let keep_loss_grad = return_filter
                .is_none_or(|targets| targets.contains(&loss_id) || loss_is_tape_output);
            if accumulate_into_store
                && !loss_is_tape_output
                && store
                    .get(loss_id)
                    .is_some_and(|tensor| tensor.requires_grad)
            {
                store.accumulate_grad(loss_id, loss_grad_id)?;
            }
            if keep_loss_grad {
                grads.insert(loss_id, loss_grad_id);
            } else if store.get(loss_grad_id).is_some() {
                store.free(loss_grad_id)?;
            }
            if let (Some(profile), Some(started)) = (profile.as_deref_mut(), prelude_started) {
                profile.prelude_duration += started.elapsed();
            }

            let vram_profile = backward_vram_profile_enabled();
            for &entry_index in post_order.iter().rev() {
                let entry = self.entries[entry_index].clone();
                let output_grad_id = match grads.get(&entry.output_id).copied() {
                    Some(grad_id) => grad_id,
                    None => continue,
                };

                let inner_op_seq = (entry.op != BackwardOp::Checkpoint)
                    .then(|| self.checkpoint_op_mem_begin(&entry, store))
                    .flatten();
                if profile.is_some() {
                    sync_profile_boundary(store)?;
                }
                let op_started = profile.is_some().then(Instant::now);
                let input_grads = match entry.op {
                    BackwardOp::Add => ops::add_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Mul => ops::mul_backward(&entry, output_grad_id, store)?,
                    BackwardOp::MulScalar => {
                        ops::mul_scalar_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::Exp => ops::exp_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Sum => ops::sum_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Matmul => ops::matmul_backward(&entry, output_grad_id, store)?,
                    BackwardOp::MatmulBT => ops::matmul_bt_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Softmax => ops::softmax_backward(&entry, output_grad_id, store)?,
                    BackwardOp::LogSoftmax => {
                        ops::log_softmax_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::Gather => {
                        ops::gather_last_dim_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::MoeTopKSoftmax => {
                        ops::moe_topk_softmax_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::MoeGatherRows => {
                        ops::moe_gather_rows_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::MoeWeightedScatter => {
                        ops::moe_weighted_scatter_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::MoeGroupedLinear => {
                        ops::moe_grouped_linear_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::MoeGroupedWeightedScatter => {
                        ops::moe_grouped_weighted_scatter_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::Mean => ops::mean_backward(&entry, output_grad_id, store)?,
                    BackwardOp::RMSNorm => ops::rmsnorm_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Silu => ops::silu_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Sigmoid => ops::sigmoid_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Gelu => ops::gelu_backward(&entry, output_grad_id, store)?,
                    BackwardOp::RoPE => ops::rope_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Reshape => ops::reshape_backward(&entry, output_grad_id, store)?,
                    BackwardOp::BroadcastExpand => {
                        ops::broadcast_expand_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::Slice => ops::slice_backward(&entry, output_grad_id, store)?,
                    BackwardOp::CatHeads => ops::cat_heads_backward(&entry, output_grad_id, store)?,
                    BackwardOp::CatSeq => ops::cat_seq_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Cat => ops::cat_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Transpose => {
                        ops::transpose_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::AddBroadcast => {
                        ops::add_broadcast_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::Embedding => {
                        ops::embedding_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::FusedLinearDistill => {
                        ops::fused_linear_distill_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::GeneralizedJsd => {
                        ops::generalized_jsd_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::LinearAttention => {
                        ops::linear_attention_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::CausalSdpaRecompute => {
                        ops::causal_sdpa_recompute_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::AllReduceSum => {
                        ops::all_reduce_sum_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::AllGatherSeq => {
                        ops::all_gather_seq_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::ReduceScatterSum => {
                        ops::reduce_scatter_sum_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::AllToAll => {
                        ops::all_to_all_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::EpDispatch => {
                        ops::collective_ep::ep_dispatch_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::EpCombine => {
                        ops::collective_ep::ep_combine_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::EpExchange => {
                        ops::collective_ep::ep_exchange_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::RingAttention => ops::ring_attention::cp_ring_attention_backward(
                        &entry,
                        output_grad_id,
                        store,
                    )?,
                    BackwardOp::Checkpoint => self.checkpoint_backward(
                        &entry,
                        output_grad_id,
                        store,
                        profile.as_deref_mut(),
                    )?,
                    BackwardOp::SeqChunkedRecompute => self.seq_chunked_recompute_backward(
                        &entry,
                        output_grad_id,
                        store,
                        profile.as_deref_mut(),
                    )?,
                };
                if let Some(op_seq) = inner_op_seq {
                    self.checkpoint_op_mem_record("post_op", Some(op_seq), Some(&entry), store);
                }
                if vram_profile {
                    log_backward_vram_profile("after_op", entry_index, &entry, store)?;
                }
                if let (Some(profile), Some(started)) = (profile.as_deref_mut(), op_started) {
                    sync_profile_boundary(store)?;
                    let duration = started.elapsed();
                    profile.record_op(entry.op, duration);
                    if let Some(site) = entry.profile_site() {
                        profile.record_site(entry.op, site, duration);
                    }
                }

                if profile.is_some() {
                    sync_profile_boundary(store)?;
                }
                let merge_started = profile.is_some().then(Instant::now);
                let output_grad_reused = input_grads
                    .iter()
                    .any(|(_, grad_id)| *grad_id == output_grad_id);
                for (input_id, grad_id) in input_grads {
                    merge_grad(
                        &mut grads,
                        input_id,
                        grad_id,
                        store,
                        accumulate_into_store,
                        &entry_by_output,
                        return_filter,
                    )?;
                }
                let keep_output_grad =
                    return_filter.is_some_and(|targets| targets.contains(&entry.output_id));
                let release_output_grad = (accumulate_into_store || return_filter.is_some())
                    && !keep_output_grad
                    && (entry.output_id != loss_id || return_filter.is_some());
                if release_output_grad {
                    grads.remove(&entry.output_id);
                    if !output_grad_reused && store.get(output_grad_id).is_some() {
                        store.free(output_grad_id)?;
                    }
                }
                if return_filter.is_some_and(|targets| !targets.contains(&entry.output_id))
                    && store.get(entry.output_id).is_some()
                {
                    store.free(entry.output_id)?;
                }
                if let Some(op_seq) = inner_op_seq {
                    self.checkpoint_op_mem_record("post_merge", Some(op_seq), Some(&entry), store);
                }
                if vram_profile {
                    log_backward_vram_profile("after_merge", entry_index, &entry, store)?;
                }
                if let (Some(profile), Some(started)) = (profile.as_deref_mut(), merge_started) {
                    sync_profile_boundary(store)?;
                    profile.merge_grad_duration += started.elapsed();
                }
            }

            Ok(grads)
        })();

        self.enabled = was_enabled;
        if let (Some(profile), Some(started)) = (profile.take(), total_started) {
            profile.total_duration += started.elapsed();
        }
        result
    }

    fn checkpoint_backward(
        &mut self,
        entry: &TapeEntry,
        output_grad_id: TensorId,
        store: &mut TensorStore,
        profile: Option<&mut BackwardProfile>,
    ) -> Result<GradPairs> {
        let SavedContext::CheckpointCtx { function_id } = entry.saved else {
            return Err(AutogradError::TapeInvariant(
                "checkpoint backward missing saved context",
            ));
        };
        let checkpoint_fn = self
            .checkpoint_fns
            .get(function_id)
            .ok_or(AutogradError::TapeInvariant(
                "checkpoint backward missing replay function",
            ))?
            .clone();
        let selected = !self.checkpoint_op_mem_disarmed
            && selected_checkpoint_fn().is_some_and(|selected| selected == function_id);
        if selected {
            self.checkpoint_op_mem_disarmed = true;
            self.checkpoint_op_mem_scope =
                Some(Arc::new(Mutex::new(CheckpointOpMemScope::new(function_id))));
            self.checkpoint_op_mem_record("scope_enter", None, None, store);
        }

        let live_before = store.live_ids().into_iter().collect::<HashSet<_>>();
        let mut inner_profile = profile.as_ref().map(|_| BackwardProfile::default());
        let result = (|| {
            for &input_id in &entry.input_ids {
                store.ensure_checkpoint_device(input_id)?;
            }
            let mut inner_tape = Tape::new();
            inner_tape.checkpoint_op_mem_scope = self.checkpoint_op_mem_scope.clone();
            let replay_output = checkpoint_fn(store, &mut inner_tape, 0, &entry.input_ids)?;
            self.trim_after_checkpoint_replay(store)?;
            self.checkpoint_op_mem_record("post_replay", None, None, store);
            let weighted = ops::mul(replay_output, output_grad_id, store, &mut inner_tape)?;
            let loss = ops::sum(weighted, store, &mut inner_tape)?;
            let inner_result = inner_tape.backward_collect_targets_only(
                loss,
                store,
                &entry.input_ids,
                inner_profile.as_mut(),
            );
            self.checkpoint_op_mem_record("post_inner", None, None, store);
            let inner_grads = inner_result?;

            let mut grads = GradPairs::new();
            let mut keep = HashSet::new();
            for &input_id in &entry.input_ids {
                if let Some(&grad_id) = inner_grads.get(&input_id) {
                    grads.push((input_id, grad_id));
                    keep.insert(grad_id);
                }
            }
            Ok((grads, keep))
        })();

        let result = match result {
            Ok((grads, keep)) => (|| {
                store.free_new_except(&live_before, &keep)?;
                // Mirror forward offload so only one replay hidden stays resident.
                if self.offload_checkpoints
                    && let Some(&hidden_id) = entry.input_ids.first()
                    && hidden_id != entry.output_id
                {
                    store.offload_checkpoint_to_host(hidden_id)?;
                }
                self.trim_after_checkpoint_replay(store)?;
                self.checkpoint_op_mem_record("scope_exit", None, None, store);
                if let (Some(outer), Some(mut inner)) = (profile, inner_profile) {
                    // The inner wall already sits inside the Checkpoint envelope.
                    inner.total_duration = Duration::ZERO;
                    outer.merge(&inner);
                }
                Ok(grads)
            })(),
            Err(err) => {
                let _ = store.free_new_except(&live_before, &HashSet::new());
                let _ = self.trim_after_checkpoint_replay(store);
                Err(err)
            }
        };
        if selected && let Some(scope) = self.checkpoint_op_mem_scope.take() {
            flush_checkpoint_op_mem(&scope);
        }
        result
    }

    /// Replays position-wise chunks and accumulates their gradients on device.
    fn seq_chunked_recompute_backward(
        &mut self,
        entry: &TapeEntry,
        output_grad_id: TensorId,
        store: &mut TensorStore,
        _profile: Option<&mut BackwardProfile>,
    ) -> Result<GradPairs> {
        let SavedContext::SeqChunkedRecomputeCtx {
            function_id,
            batch,
            seq,
            dim,
            chunk,
        } = entry.saved
        else {
            return Err(AutogradError::TapeInvariant(
                "seq_chunked_recompute backward missing saved context",
            ));
        };
        let replay = self
            .checkpoint_fns
            .get(function_id)
            .ok_or(AutogradError::TapeInvariant(
                "seq_chunked_recompute backward missing replay function",
            ))?
            .clone();
        let input_id = *entry.input_ids.first().ok_or(AutogradError::TapeInvariant(
            "seq_chunked_recompute backward missing input",
        ))?;
        let param_ids = &entry.input_ids[1..];

        let need_input_grad = store.tensor(input_id)?.requires_grad;
        let mut d_input = need_input_grad
            .then(|| SeqAccum::new(vec![batch, seq, dim], 1, store))
            .transpose()?;
        let mut d_param: Vec<ChunkSum> = param_ids.iter().map(|_| ChunkSum::new()).collect();

        let mut start = 0;
        while start < seq {
            let end = (start + chunk).min(seq);
            let live_before = store.live_ids().into_iter().collect::<HashSet<_>>();

            // Detached chunk leaves: slice off a disabled scratch tape so the
            // sub-backward treats x_c/grad_c as inputs, not slices of the full seq.
            let mut scratch = Tape::new();
            scratch.set_enabled(false);
            store.ensure_checkpoint_device(input_id)?;
            let x_c = ops::slice(
                input_id,
                &[0, start, 0],
                &[batch, end, dim],
                store,
                &mut scratch,
            )?;
            store.set_requires_grad(x_c, need_input_grad)?;
            let grad_c = ops::slice(
                output_grad_id,
                &[0, start, 0],
                &[batch, end, dim],
                store,
                &mut scratch,
            )?;

            let mut chunk_tape = Tape::new();
            let mut chunk_inputs = vec![x_c];
            chunk_inputs.extend_from_slice(param_ids);
            let y_c = replay(store, &mut chunk_tape, start, &chunk_inputs)?;
            let weighted = ops::mul(y_c, grad_c, store, &mut chunk_tape)?;
            let loss = ops::sum(weighted, store, &mut chunk_tape)?;
            let grads =
                chunk_tape.backward_collect_targets_only(loss, store, &chunk_inputs, None)?;

            if let (Some(dest), Some(&g)) = (d_input.as_mut(), grads.get(&x_c)) {
                dest.write_rows(start, g, store)?;
            }
            // Full-seq k/v grads dominate device memory; park each over the next replay.
            for (slot, &pid) in param_ids.iter().enumerate() {
                if let Some(&g) = grads.get(&pid) {
                    d_param[slot].add(g, store)?;
                    if end < seq {
                        d_param[slot].park(store)?;
                    }
                }
            }

            let keep = d_input
                .as_ref()
                .map(SeqAccum::id)
                .into_iter()
                .chain(d_param.iter().filter_map(ChunkSum::id))
                .collect();
            store.free_new_except(&live_before, &keep)?;
            self.trim_after_checkpoint_replay(store)?;
            start = end;
        }

        let mut pairs = GradPairs::new();
        if let Some(acc) = d_input.take() {
            pairs.push((input_id, acc.finish()));
        }
        for (&pid, acc) in param_ids.iter().zip(d_param) {
            if let Some(g) = acc.finish(store)? {
                pairs.push((pid, g));
            }
        }
        Ok(pairs)
    }

    /// Trim the pool after a checkpoint replay's backward, under offload only:
    /// the replay re-fetches then re-offloads its hidden, freeing pages the pool
    /// would otherwise hoard. Trimming holds the backward high-water below the
    /// device limit (H20-96GB seq=40960: completes vs concat_axis2 OOM). No-op
    /// without offload (nothing freed to reclaim).
    fn trim_after_checkpoint_replay(&self, store: &TensorStore) -> Result<()> {
        if self.offload_checkpoints {
            store.backend().trim_memory_pool()?;
        }
        Ok(())
    }
}

fn selected_checkpoint_fn() -> Option<usize> {
    std::env::var("ARLE_OPD_OP_MEM_CHECKPOINT_FN")
        .ok()?
        .parse()
        .ok()
}

pub fn checkpoint_replay_mem_stage(tape: &Tape, store: &TensorStore, stage: &'static str) {
    tape.checkpoint_op_mem_record(stage, None, None, store);
}

fn flush_checkpoint_op_mem(scope: &Arc<Mutex<CheckpointOpMemScope>>) {
    let scope = scope.lock().expect("checkpoint op memory scope poisoned");
    let stderr = std::io::stderr();
    let mut out = BufWriter::new(stderr.lock());
    for record in &scope.records {
        let op_seq = record
            .op_seq
            .map_or_else(|| "-1".to_owned(), |seq| seq.to_string());
        let reserved = record
            .pool
            .map_or_else(|| "n/a".to_owned(), |(bytes, _)| (bytes >> 20).to_string());
        let used = record
            .pool
            .map_or_else(|| "n/a".to_owned(), |(_, bytes)| (bytes >> 20).to_string());
        let _ = writeln!(
            out,
            "[autograd-op-mem] checkpoint_fn={} stage={} op_seq={} op={} site={} pool_reserved_mib={} pool_used_current_mib={} live_tensors={}",
            scope.checkpoint_fn,
            record.stage,
            op_seq,
            record.op.map_or("-", BackwardOp::name),
            record.site.unwrap_or("-"),
            reserved,
            used,
            record.live_tensors,
        );
    }
}

fn sync_profile_boundary(store: &TensorStore) -> Result<()> {
    store.backend().eval(&[])
}

fn backward_vram_profile_enabled() -> bool {
    matches!(
        std::env::var("ARLE_OPD_BACKWARD_VRAM_PROFILE").as_deref(),
        Ok(value) if !matches!(value, "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF")
    )
}

fn log_backward_vram_profile(
    stage: &str,
    entry_index: usize,
    entry: &TapeEntry,
    store: &TensorStore,
) -> Result<()> {
    store.backend().eval(&[])?;
    if let Some((free, total)) = store.backend().device_mem_info() {
        let used_mib = total.saturating_sub(free) >> 20;
        let free_mib = free >> 20;
        let total_mib = total >> 20;
        let site = entry.profile_site().unwrap_or("-");
        let live_tensors = store.live_ids().len();
        eprintln!(
            "[autograd-backward-vram] stage={stage} index={entry_index} op={} site={} \
             output_id={} used_mib={used_mib} free_mib={free_mib} total_mib={total_mib} \
             live_tensors={live_tensors}",
            entry.op.name(),
            site,
            entry.output_id,
        );
    }
    Ok(())
}

fn collect_relevant(
    tensor_id: TensorId,
    entry_by_output: &HashMap<TensorId, usize>,
    entries: &[TapeEntry],
    relevant_tensors: &mut HashSet<TensorId>,
    visited_outputs: &mut HashSet<TensorId>,
    post_order: &mut Vec<usize>,
) {
    relevant_tensors.insert(tensor_id);
    let Some(&entry_index) = entry_by_output.get(&tensor_id) else {
        return;
    };

    let entry = &entries[entry_index];
    if !visited_outputs.insert(entry.output_id) {
        return;
    }

    for &input_id in &entry.input_ids {
        collect_relevant(
            input_id,
            entry_by_output,
            entries,
            relevant_tensors,
            visited_outputs,
            post_order,
        );
    }

    post_order.push(entry_index);
}

fn merge_grad(
    grads: &mut HashMap<TensorId, TensorId>,
    tensor_id: TensorId,
    new_grad_id: TensorId,
    store: &mut TensorStore,
    accumulate_into_store: bool,
    entry_by_output: &HashMap<TensorId, usize>,
    return_filter: Option<&HashSet<TensorId>>,
) -> Result<()> {
    let keep_in_grads = return_filter.is_none_or(|targets| {
        targets.contains(&tensor_id) || entry_by_output.contains_key(&tensor_id)
    });
    let should_store_grad = accumulate_into_store
        && !entry_by_output.contains_key(&tensor_id)
        && store
            .get(tensor_id)
            .is_some_and(|tensor| tensor.requires_grad);

    if let Some(existing_grad_id) = grads.get(&tensor_id).copied() {
        let expected = store.tensor(existing_grad_id)?.shape.clone();
        let incoming = store.tensor(new_grad_id)?.shape.clone();
        if expected != incoming {
            return Err(AutogradError::GradientShapeMismatch {
                tensor_id,
                expected,
                got: incoming,
            });
        }

        // P2 (device-resident gradient tape): if both grads are still
        // device-resident, fuse them with `add_into_device` so neither
        // side gets pulled back to host. Without this, the second
        // backward path that arrives at the same parameter would force a
        // `to_host(new_grad_id)` and the merged sum lives only in
        // `existing.data` — host-resident from then on.
        let both_on_device = {
            let existing = store.tensor(existing_grad_id)?;
            let incoming = store.tensor(new_grad_id)?;
            existing.dirty != Dirty::Host
                && existing.device_handle.is_some()
                && incoming.dirty != Dirty::Host
                && incoming.device_handle.is_some()
        };
        if both_on_device {
            // In-place accumulation avoids a THIRD full-size grad buffer
            // (add_into_device's alloc_zeros) — the OOM at long-seq writeback.
            // Safe ONLY when this buffer is uniquely owned: grads fan out by
            // Arc clone (clone_tensor / add_backward push one grad to both
            // inputs), so a shared buffer mutated in place would corrupt a live
            // sibling. Probe strong-count on the store-held handle BEFORE the
            // clone below (the clone would bump it). `Some(1)` = sole owner.
            let uniquely_owned = store
                .tensor(existing_grad_id)?
                .device_handle
                .as_ref()
                .and_then(DeviceHandle::device_buffer_strong_count)
                == Some(1);
            let existing_handle = store.device_handle(existing_grad_id)?;
            let incoming_handle = store.device_handle(new_grad_id)?;
            let sum_handle = if uniquely_owned {
                store.backend().accumulate_into_device(
                    &existing_handle,
                    &incoming_handle,
                    &expected,
                )?
            } else {
                store
                    .backend()
                    .add_into_device(&existing_handle, &incoming_handle, &expected)?
            };
            store.replace_device_handle(existing_grad_id, sum_handle)?;
        } else {
            let incoming_data = store.to_host(new_grad_id)?;
            let existing = store.tensor_mut(existing_grad_id)?;
            for (dst, src) in existing.data.iter_mut().zip(incoming_data) {
                *dst += src;
            }
        }
        if should_store_grad {
            if keep_in_grads {
                store.accumulate_grad(tensor_id, new_grad_id)?;
            } else {
                store.accumulate_grad_owned(tensor_id, new_grad_id)?;
            }
        }
        if new_grad_id != existing_grad_id && store.get(new_grad_id).is_some() {
            store.free(new_grad_id)?;
        }
    } else if keep_in_grads {
        grads.insert(tensor_id, new_grad_id);
        if should_store_grad {
            store.accumulate_grad(tensor_id, new_grad_id)?;
        }
    } else {
        if should_store_grad {
            store.accumulate_grad_owned(tensor_id, new_grad_id)?;
        }
        if store.get(new_grad_id).is_some() {
            store.free(new_grad_id)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    /// A fresh `alloc_device_tensor` defaults to `requires_grad=false`; `record`
    /// must mark it from its inputs with no help from the call site, and must do so
    /// even when the tape is off (`checkpoint`'s inner replay runs disabled).
    #[test]
    fn record_marks_output_from_inputs_even_when_disabled() {
        for enabled in [true, false] {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            tape.set_enabled(enabled);
            let x = store.alloc(Tensor::new(vec![2.0, -3.0], vec![2], true).expect("create x"));
            let handle = store.backend().zeros(&[2]).expect("device zeros");
            let out = store
                .alloc_device_tensor(vec![2], handle)
                .expect("device alloc");
            assert!(!store.tensor(out).expect("out").requires_grad);

            TapeEntry {
                op: BackwardOp::Mul,
                output_id: out,
                input_ids: smallvec::smallvec![x, x],
                saved: SavedContext::Tensors(smallvec::smallvec![x, x]),
            }
            .record(&mut store, &mut tape)
            .expect("record");

            assert!(
                store.tensor(out).expect("out").requires_grad,
                "enabled={enabled}"
            );
            assert_eq!(tape.entries.len(), usize::from(enabled));
        }
    }

    #[test]
    fn backward_on_empty_tape_does_not_panic() {
        let mut store = TensorStore::default();
        let loss = store.alloc(Tensor::new(vec![5.0], Vec::new(), true).expect("create scalar"));
        let mut tape = Tape::new();

        let grads = tape.backward(loss, &mut store).expect("backward succeeds");

        let grad_id = grads.get(&loss).copied().expect("loss grad exists");
        assert_eq!(store.to_host(grad_id).expect("copy grad"), vec![1.0]);
    }

    #[test]
    fn backward_profiled_matches_backward_and_counts_ops() {
        fn run(profiled: bool) -> (Vec<f32>, Vec<f32>, Option<BackwardProfile>) {
            let mut store = TensorStore::default();
            let x = store.alloc(Tensor::new(vec![2.0, -3.0], vec![2], true).expect("create x"));
            let mut tape = Tape::new();
            let y = ops::mul(x, x, &mut store, &mut tape).expect("x*x");
            let loss = ops::sum(y, &mut store, &mut tape).expect("sum");

            let (grads, profile) = if profiled {
                let (grads, profile) = tape
                    .backward_profiled(loss, &mut store)
                    .expect("profiled backward");
                (grads, Some(profile))
            } else {
                (
                    tape.backward(loss, &mut store).expect("plain backward"),
                    None,
                )
            };
            let grad_id = grads.get(&x).copied().expect("x grad exists");
            let stored_grad_id = store
                .get(x)
                .and_then(|tensor| tensor.grad)
                .expect("x stored grad exists");
            (
                store.to_host(grad_id).expect("x grad host"),
                store.to_host(stored_grad_id).expect("x stored grad host"),
                profile,
            )
        }

        let (plain_grad, plain_stored_grad, _) = run(false);
        let (profiled_grad, profiled_stored_grad, profile) = run(true);
        assert_eq!(plain_grad, profiled_grad);
        assert_eq!(profiled_grad, vec![4.0, -6.0]);
        assert_eq!(plain_stored_grad, profiled_grad);
        assert_eq!(profiled_stored_grad, profiled_grad);

        let profile = profile.expect("profile returned");
        assert_eq!(BackwardOp::LinearAttention.name(), "LinearAttention");
        assert_eq!(profile.op_totals[&BackwardOp::Sum].count, 1);
        assert_eq!(profile.op_totals[&BackwardOp::Mul].count, 1);
        assert!(profile.total_duration >= profile.total_op_duration());
    }

    #[test]
    fn backward_does_not_persist_intermediate_grads() {
        let mut store = TensorStore::default();
        let x = store.alloc(Tensor::new(vec![2.0, -3.0], vec![2], true).expect("create x"));
        let mut tape = Tape::new();
        let y = ops::mul(x, x, &mut store, &mut tape).expect("x*x");
        let loss = ops::sum(y, &mut store, &mut tape).expect("sum");

        let grads = tape.backward(loss, &mut store).expect("backward");

        assert!(grads.contains_key(&x), "leaf grad is returned");
        assert!(
            store.get(x).and_then(|tensor| tensor.grad).is_some(),
            "leaf grad is stored"
        );
        assert!(
            !grads.contains_key(&y),
            "intermediate grad is freed after its last consumer"
        );
        assert!(
            store.get(y).and_then(|tensor| tensor.grad).is_none(),
            "intermediate grad is not persisted on the tensor"
        );
    }

    #[test]
    fn merge_grad_sums_fanned_out_leaf_touches() {
        // A leaf used twice (`x*x`) fans its grad out to both `mul` inputs, then
        // merge_grad sums the two contributions into the leaf's grad slot — the
        // exact path the guarded in-place accumulate gates on. Pin the summed
        // value so a broken merge (in-place clobber of a shared buffer, or a
        // dropped contribution) fails here regardless of backend: d/dx Σ(x*x) = 2x.
        let mut store = TensorStore::default();
        let x = store.alloc(Tensor::new(vec![2.0, -3.0, 0.5], vec![3], true).expect("create x"));
        let mut tape = Tape::new();
        let sq = ops::mul(x, x, &mut store, &mut tape).expect("x*x");
        let loss = ops::sum(sq, &mut store, &mut tape).expect("sum");

        let grads = tape.backward(loss, &mut store).expect("backward");
        let gx = grads.get(&x).copied().expect("leaf grad present");
        let data = store.to_host(gx).expect("read grad");
        assert_eq!(data, vec![4.0, -6.0, 1.0], "Σ of both fan-out touches = 2x");
    }

    #[test]
    fn backward_collect_targets_only_drops_unrequested_leaf_grads() {
        let mut store = TensorStore::default();
        let x = store.alloc(Tensor::new(vec![2.0, -3.0], vec![2], true).expect("create x"));
        let y = store.alloc(Tensor::new(vec![4.0, 5.0], vec![2], true).expect("create y"));
        let mut tape = Tape::new();
        let prod = ops::mul(x, y, &mut store, &mut tape).expect("x*y");
        let loss = ops::sum(prod, &mut store, &mut tape).expect("sum");

        let grads = tape
            .backward_collect_targets_only(loss, &mut store, &[x], None)
            .expect("target-only collect");

        assert!(grads.contains_key(&x));
        assert!(
            !grads.contains_key(&y),
            "unrequested leaf grad should not be retained"
        );
        assert!(
            !grads.contains_key(&prod),
            "intermediate grad should be freed after its last consumer"
        );
        assert!(store.get(prod).is_none());
        let x_grad = store.to_host(grads[&x]).expect("x grad host");
        assert_eq!(x_grad, vec![4.0, 5.0]);
    }

    #[test]
    fn backward_accumulate_only_persists_leaf_grads_without_return_map() {
        let mut store = TensorStore::default();
        let x = store.alloc(Tensor::new(vec![2.0, -3.0], vec![2], true).expect("create x"));
        let y = store.alloc(Tensor::new(vec![4.0, 5.0], vec![2], true).expect("create y"));
        let mut tape = Tape::new();
        let prod = ops::mul(x, y, &mut store, &mut tape).expect("x*y");
        let sum = ops::add(prod, prod, &mut store, &mut tape).expect("prod+prod");
        let loss = ops::sum(sum, &mut store, &mut tape).expect("sum");

        tape.backward_accumulate_only(loss, &mut store)
            .expect("accumulate-only backward");

        let x_grad = store
            .get(x)
            .and_then(|tensor| tensor.grad)
            .expect("x stored grad");
        let y_grad = store
            .get(y)
            .and_then(|tensor| tensor.grad)
            .expect("y stored grad");
        assert_eq!(store.to_host(x_grad).expect("x grad host"), vec![8.0, 10.0]);
        assert_eq!(store.to_host(y_grad).expect("y grad host"), vec![4.0, -6.0]);
        assert!(
            store.get(prod).and_then(|tensor| tensor.grad).is_none(),
            "intermediate grad is not persisted"
        );
    }

    #[test]
    fn backward_profiled_counts_matmul_bt_sites() {
        let mut store = TensorStore::default();
        let a = store.alloc(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3], true).expect("create a"));
        let b = store.alloc(
            Tensor::new(vec![1.0, -1.0, 0.5, 2.0, 0.25, -0.5], vec![2, 3], true).expect("create b"),
        );
        let mut tape = Tape::new();
        let y = ops::matmul_bt_with_site(a, b, &mut store, &mut tape, "unit.matmul_bt")
            .expect("matmul_bt");
        let loss = ops::sum(y, &mut store, &mut tape).expect("sum");

        let (_grads, profile) = tape
            .backward_profiled(loss, &mut store)
            .expect("profiled backward");

        assert_eq!(profile.op_totals[&BackwardOp::MatmulBT].count, 1);
        assert_eq!(
            profile.site_totals[&(BackwardOp::MatmulBT, "unit.matmul_bt")].count,
            1
        );
    }

    #[test]
    fn backward_accumulate_targets_only_persists_requested_grads() {
        let mut store = TensorStore::default();
        let x = store.alloc(Tensor::new(vec![2.0, -3.0], vec![2], true).expect("create x"));
        let y = store.alloc(Tensor::new(vec![4.0, 5.0], vec![2], true).expect("create y"));
        let mut tape = Tape::new();
        let prod = ops::mul(x, y, &mut store, &mut tape).expect("x*y");
        let loss = ops::sum(prod, &mut store, &mut tape).expect("sum");

        let grads = tape
            .backward_accumulate_targets(loss, &mut store, &[x])
            .expect("target-only backward");

        assert!(grads.contains_key(&x));
        assert!(grads.contains_key(&y));
        let x_stored = store
            .get(x)
            .and_then(|tensor| tensor.grad)
            .expect("x stored grad");
        assert_eq!(
            store.to_host(x_stored).expect("x grad host"),
            vec![4.0, 5.0]
        );
        assert!(
            store.get(y).and_then(|tensor| tensor.grad).is_none(),
            "non-target tensor must not keep a persistent grad"
        );
    }

    #[test]
    fn backward_from_seed_accumulate_targets_uses_explicit_output_grad() {
        let mut store = TensorStore::default();
        let x = store.alloc(Tensor::new(vec![2.0, -3.0], vec![2], true).expect("create x"));
        let mut tape = Tape::new();
        let squared = ops::mul(x, x, &mut store, &mut tape).expect("x*x");
        let seed = store.alloc(Tensor::new(vec![3.0, 4.0], vec![2], false).expect("seed"));

        let grads = tape
            .backward_from_seed_accumulate_targets(squared, seed, &mut store, &[x])
            .expect("seed backward");

        let grad_id = *grads.get(&x).expect("x grad exists");
        assert_eq!(
            store.to_host(grad_id).expect("x grad host"),
            vec![12.0, -24.0]
        );
        let stored = store
            .get(x)
            .and_then(|tensor| tensor.grad)
            .expect("x stored grad");
        assert_eq!(
            store.to_host(stored).expect("stored x grad host"),
            vec![12.0, -24.0]
        );
    }

    #[test]
    fn backward_from_seed_profiled_matches_plain_and_counts_ops() {
        fn run(profiled: bool) -> (Vec<f32>, Option<BackwardProfile>) {
            let mut store = TensorStore::default();
            let x = store.alloc(Tensor::new(vec![2.0, -3.0], vec![2], true).expect("create x"));
            let mut tape = Tape::new();
            let squared = ops::mul(x, x, &mut store, &mut tape).expect("x*x");
            let seed = store.alloc(Tensor::new(vec![3.0, 4.0], vec![2], false).expect("seed"));

            let (grads, profile) = if profiled {
                let (grads, profile) = tape
                    .backward_from_seed_accumulate_targets_profiled(squared, seed, &mut store, &[x])
                    .expect("profiled seed backward");
                (grads, Some(profile))
            } else {
                (
                    tape.backward_from_seed_accumulate_targets(squared, seed, &mut store, &[x])
                        .expect("plain seed backward"),
                    None,
                )
            };
            let grad_id = *grads.get(&x).expect("x grad exists");
            (store.to_host(grad_id).expect("x grad host"), profile)
        }

        let (plain_grad, _) = run(false);
        let (profiled_grad, profile) = run(true);
        assert_eq!(plain_grad, profiled_grad);
        assert_eq!(profiled_grad, vec![12.0, -24.0]);

        let profile = profile.expect("profile returned");
        assert_eq!(profile.op_totals[&BackwardOp::Mul].count, 1);
        assert!(profile.total_duration >= profile.total_op_duration());
    }

    #[test]
    fn checkpoint_forward_records_single_entry_and_frees_segment_temporaries() {
        let mut store = TensorStore::default();
        let x = store.alloc(Tensor::new(vec![2.0, -3.0], vec![2], true).expect("create x"));
        let w = store.alloc(Tensor::new(vec![0.5, -2.0], vec![2], true).expect("create w"));
        let mut tape = Tape::new();

        let y = ops::checkpoint(vec![x, w], &mut store, &mut tape, |store, tape, inputs| {
            let prod = ops::mul(inputs[0], inputs[1], store, tape)?;
            ops::mul(prod, prod, store, tape)
        })
        .expect("checkpoint forward");

        assert_eq!(tape.entries.len(), 1);
        assert_eq!(tape.entries[0].op, BackwardOp::Checkpoint);
        assert_eq!(tape.entries[0].output_id, y);
        assert_eq!(
            store.live_tensor_count(),
            3,
            "only x, w, and checkpoint output should remain live"
        );
    }

    // The backward profile must see THROUGH checkpoint entries: inner-tape ops
    // (here Mul) must appear in the op table, else 99% of a checkpointed
    // backward reads as one opaque Checkpoint row.
    #[test]
    fn backward_profile_attributes_checkpoint_inner_ops() {
        let mut store = TensorStore::default();
        let x = store.alloc(Tensor::new(vec![2.0, -3.0], vec![2], true).expect("x"));
        let w = store.alloc(Tensor::new(vec![0.5, -2.0], vec![2], true).expect("w"));
        let mut tape = Tape::new();
        let y = ops::checkpoint(vec![x, w], &mut store, &mut tape, |store, tape, inputs| {
            let prod = ops::mul(inputs[0], inputs[1], store, tape)?;
            ops::mul(prod, prod, store, tape)
        })
        .expect("checkpoint forward");
        let loss = ops::sum(y, &mut store, &mut tape).expect("sum");
        let (_, profile) = tape.backward_profiled(loss, &mut store).expect("backward");
        assert!(
            profile.op_totals.contains_key(&BackwardOp::Checkpoint),
            "outer Checkpoint row missing"
        );
        assert!(
            profile.op_totals.contains_key(&BackwardOp::Mul),
            "inner Mul rows not merged through the checkpoint: {:?}",
            profile.op_totals.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn checkpoint_backward_matches_plain_gradients() {
        fn run(checkpointed: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
            let mut store = TensorStore::default();
            let x = store.alloc(Tensor::new(vec![2.0, -3.0], vec![2], true).expect("create x"));
            let w = store.alloc(Tensor::new(vec![0.5, -2.0], vec![2], true).expect("create w"));
            let mut tape = Tape::new();

            let y = if checkpointed {
                ops::checkpoint(vec![x, w], &mut store, &mut tape, |store, tape, inputs| {
                    let prod = ops::mul(inputs[0], inputs[1], store, tape)?;
                    ops::mul(prod, prod, store, tape)
                })
                .expect("checkpoint forward")
            } else {
                let prod = ops::mul(x, w, &mut store, &mut tape).expect("prod");
                ops::mul(prod, prod, &mut store, &mut tape).expect("square")
            };
            let loss = ops::sum(y, &mut store, &mut tape).expect("sum");
            let grads = tape.backward(loss, &mut store).expect("backward");
            let x_grad = store
                .to_host(*grads.get(&x).expect("x grad"))
                .expect("x grad host");
            let w_grad = store
                .to_host(*grads.get(&w).expect("w grad"))
                .expect("w grad host");
            let x_stored = store
                .get(x)
                .and_then(|tensor| tensor.grad)
                .expect("x stored grad");
            let w_stored = store
                .get(w)
                .and_then(|tensor| tensor.grad)
                .expect("w stored grad");
            let x_stored = store.to_host(x_stored).expect("x stored host");
            let w_stored = store.to_host(w_stored).expect("w stored host");
            (x_grad, w_grad, x_stored, w_stored)
        }

        let plain = run(false);
        let checkpointed = run(true);
        assert_eq!(checkpointed.0, plain.0);
        assert_eq!(checkpointed.1, plain.1);
        assert_eq!(checkpointed.2, plain.2);
        assert_eq!(checkpointed.3, plain.3);
        assert_eq!(checkpointed.0, vec![1.0, -24.0]);
        assert_eq!(checkpointed.1, vec![4.0, -36.0]);
    }

    #[test]
    fn checkpoint_offload_is_transparent() {
        // offload_checkpoints only moves saved inputs host<->device around the
        // backward replay; gradients must be identical with it on vs off.
        fn run(offload: bool) -> (Vec<f32>, Vec<f32>) {
            let mut store = TensorStore::default();
            let x = store.alloc(Tensor::new(vec![2.0, -3.0], vec![2], true).expect("x"));
            let w = store.alloc(Tensor::new(vec![0.5, -2.0], vec![2], true).expect("w"));
            let mut tape = Tape::new();
            tape.set_offload_checkpoints(offload);
            let y = ops::checkpoint(vec![x, w], &mut store, &mut tape, |store, tape, inputs| {
                let prod = ops::mul(inputs[0], inputs[1], store, tape)?;
                ops::mul(prod, prod, store, tape)
            })
            .expect("checkpoint forward");
            let loss = ops::sum(y, &mut store, &mut tape).expect("sum");
            let grads = tape.backward(loss, &mut store).expect("backward");
            let xg = store.to_host(*grads.get(&x).expect("x grad")).expect("xg");
            let wg = store.to_host(*grads.get(&w).expect("w grad")).expect("wg");
            (xg, wg)
        }
        assert_eq!(run(true), run(false));
    }

    #[test]
    fn checkpoint_frozen_group_offload_drops_input_hidden_device_residency() {
        // Regression for the agent-OPD writeback OOM: a FROZEN checkpoint group
        // (no trainable input → no tape entry, no backward replay) used to PIN
        // its input hidden in VRAM under `offload_checkpoints`, because the hidden
        // sits in `keep` here and then in every later group's `live_before`, so no
        // `free_new_except` ever reclaims it — +156 MiB/group, unbounded over the
        // frozen prefix. The fix drops the frozen group's input-hidden device
        // residency. Mirror the failure minimally: a frozen group whose input is
        // device-resident must NOT keep a device handle on that input afterward.
        let mut store = TensorStore::default();
        // Frozen "hidden" (requires_grad=false) made device-resident.
        let hidden = store.alloc(Tensor::new(vec![2.0, -3.0], vec![2], false).expect("h"));
        store.ensure_device(hidden).expect("upload hidden");
        store
            .replace_device_handle(
                hidden,
                store.tensor(hidden).unwrap().device_handle.clone().unwrap(),
            )
            .expect("device-authoritative");
        assert!(store.tensor(hidden).unwrap().device_handle.is_some());

        let mut tape = Tape::new();
        tape.set_offload_checkpoints(true);
        // Frozen group: the single non-param input is the hidden, requires_grad=false.
        let out = ops::checkpoint(
            vec![hidden],
            &mut store,
            &mut tape,
            |store, tape, inputs| ops::mul_scalar(inputs[0], 2.0, store, tape),
        )
        .expect("frozen checkpoint forward");

        // No tape entry recorded (frozen ⇒ no backward replay).
        assert!(
            tape.entries.is_empty(),
            "frozen group must record no checkpoint entry"
        );
        // The leak fix: the input hidden's DEVICE residency is dropped.
        assert!(
            store.tensor(hidden).unwrap().device_handle.is_none(),
            "frozen group input hidden must not retain a device handle (was the leak)"
        );
        // The forward output is still correct and usable.
        assert_eq!(store.to_host(out).expect("out"), vec![4.0, -6.0]);
    }

    #[test]
    fn checkpoint_frozen_group_default_path_keeps_device_residency() {
        // The fix is gated to `offload_checkpoints`; the DEFAULT path (offload
        // OFF) must stay byte-identical — the frozen input keeps its device handle.
        let mut store = TensorStore::default();
        let hidden = store.alloc(Tensor::new(vec![2.0, -3.0], vec![2], false).expect("h"));
        store.ensure_device(hidden).expect("upload hidden");
        store
            .replace_device_handle(
                hidden,
                store.tensor(hidden).unwrap().device_handle.clone().unwrap(),
            )
            .expect("device-authoritative");

        let mut tape = Tape::new();
        // offload_checkpoints defaults OFF.
        let _ = ops::checkpoint(
            vec![hidden],
            &mut store,
            &mut tape,
            |store, tape, inputs| ops::mul_scalar(inputs[0], 2.0, store, tape),
        )
        .expect("frozen checkpoint forward");
        assert!(
            store.tensor(hidden).unwrap().device_handle.is_some(),
            "default (offload-off) path must keep the input's device handle (byte-identical)"
        );
    }

    #[test]
    fn checkpoint_sequential_matches_per_layer() {
        use std::sync::Arc;
        // A 4-layer stack where layer i is `h -> h * w_i` (w_i trainable).
        // checkpoint_sequential with any group_size must match a plain
        // (non-checkpointed) run, both for x and every w_i gradient.
        fn run(group_size: Option<usize>) -> (Vec<f32>, Vec<Vec<f32>>) {
            let mut store = TensorStore::default();
            let x = store.alloc(Tensor::new(vec![2.0, -3.0], vec![2], true).expect("x"));
            let weights: Vec<TensorId> = (0..4)
                .map(|i| {
                    let v = i as f32 + 1.0;
                    store.alloc(Tensor::new(vec![v, -v], vec![2], true).expect("w"))
                })
                .collect();
            let mut tape = Tape::new();

            let y = if let Some(g) = group_size {
                let ws = Arc::new(weights.clone());
                let layer_fn = {
                    let ws = Arc::clone(&ws);
                    move |idx: usize, h, s: &mut TensorStore, t: &mut Tape| {
                        ops::mul(h, ws[idx], s, t)
                    }
                };
                let layer_params = |idx: usize| vec![weights[idx]];
                ops::checkpoint_sequential(
                    x,
                    weights.len(),
                    g,
                    None,
                    &mut store,
                    &mut tape,
                    layer_params,
                    layer_fn,
                )
                .expect("checkpoint_sequential")
            } else {
                let mut h = x;
                for &w in &weights {
                    h = ops::mul(h, w, &mut store, &mut tape).expect("mul");
                }
                h
            };
            let loss = ops::sum(y, &mut store, &mut tape).expect("sum");
            let grads = tape.backward(loss, &mut store).expect("backward");
            let xg = store.to_host(*grads.get(&x).expect("x grad")).expect("xg");
            let wg = weights
                .iter()
                .map(|w| store.to_host(*grads.get(w).expect("w grad")).expect("wg"))
                .collect();
            (xg, wg)
        }

        let plain = run(None);
        for g in [1, 2, 4] {
            assert_eq!(run(Some(g)), plain, "group_size {g} must match per-layer");
        }
    }
}
