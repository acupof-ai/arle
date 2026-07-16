use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};

use smallvec::SmallVec;

use crate::{
    AutogradError, Result, ops,
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
        q: Option<TensorId>,
        k: Option<TensorId>,
        v: Option<TensorId>,
        g: Option<TensorId>,
        g_cumsum: Option<TensorId>,
        beta: Option<TensorId>,
        a_inv: Option<TensorId>,
        chunk_state: Option<TensorId>,
        raw_output: Option<TensorId>,
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
    },
    CheckpointCtx {
        function_id: usize,
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
    Slice,
    CatHeads,
    CatSeq,
    Transpose,
    AddBroadcast,
    Embedding,
    FusedLinearDistill,
    GeneralizedJsd,
    LinearAttention,
    CausalSdpaRecompute,
    AllReduceSum,
    Checkpoint,
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
            BackwardOp::Slice => "Slice",
            BackwardOp::CatHeads => "CatHeads",
            BackwardOp::CatSeq => "CatSeq",
            BackwardOp::Transpose => "Transpose",
            BackwardOp::AddBroadcast => "AddBroadcast",
            BackwardOp::Embedding => "Embedding",
            BackwardOp::FusedLinearDistill => "FusedLinearDistill",
            BackwardOp::GeneralizedJsd => "GeneralizedJsd",
            BackwardOp::LinearAttention => "LinearAttention",
            BackwardOp::CausalSdpaRecompute => "CausalSdpaRecompute",
            BackwardOp::AllReduceSum => "AllReduceSum",
            BackwardOp::Checkpoint => "Checkpoint",
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
}

pub(crate) type GradPairs = SmallVec<[(TensorId, TensorId); 2]>;

pub(crate) type CheckpointFn =
    Arc<dyn Fn(&mut TensorStore, &mut Tape, &[TensorId]) -> Result<TensorId> + Send + Sync>;

#[derive(Default)]
pub struct Tape {
    pub entries: Vec<TapeEntry>,
    pub enabled: bool,
    checkpoint_fns: Vec<CheckpointFn>,
    pub(crate) offload_checkpoints: bool,
    skip_next_checkpoint_input_offload: bool,
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
        }
    }

    pub fn record(&mut self, entry: TapeEntry) {
        if self.enabled {
            self.entries.push(entry);
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
    pub fn checkpoint_fn_count(&self) -> usize {
        self.checkpoint_fns.len()
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
            // a regression once M5.3b.1 made `sum` lazy, because both `y`
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
            // P3.1: only flush all tape outputs to host upfront when the
            // backend prefers it (Metal). On CUDA this batch readback is
            // the 1 GB DtoH the M5.3b / Wave 1 / P1 / P2 / P3 milestones
            // could never kill — per-op lazy readback is strictly cheaper
            // because device-resident backward ops never need the host
            // snapshot. See
            // `docs/research/2026-05-17-cuda-training-architectural-correction.md`.
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
            // P3.1: seed the backward chain with a device-resident `1.0`
            // when the backend has device residency. Without this the
            // every-op `device_path_ok` gate in M5.3b / Wave 1 / P1 / P2 /
            // P3 falls through to host fallback, because `g.dirty=Host`
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
                    BackwardOp::Slice => ops::slice_backward(&entry, output_grad_id, store)?,
                    BackwardOp::CatHeads => ops::cat_heads_backward(&entry, output_grad_id, store)?,
                    BackwardOp::CatSeq => ops::cat_seq_backward(&entry, output_grad_id, store)?,
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
                    BackwardOp::Checkpoint => self.checkpoint_backward(
                        &entry,
                        output_grad_id,
                        store,
                        profile.as_deref_mut(),
                    )?,
                };
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

        let live_before = store.live_ids().into_iter().collect::<HashSet<_>>();
        let mut inner_profile = profile.as_ref().map(|_| BackwardProfile::default());
        let result = (|| {
            for &input_id in &entry.input_ids {
                store.ensure_checkpoint_device(input_id)?;
            }
            let mut inner_tape = Tape::new();
            let replay_output = checkpoint_fn(store, &mut inner_tape, &entry.input_ids)?;
            trim_after_checkpoint_replay(store)?;
            let weighted = ops::mul(replay_output, output_grad_id, store, &mut inner_tape)?;
            let loss = ops::sum(weighted, store, &mut inner_tape)?;
            let inner_grads = inner_tape.backward_collect_targets_only(
                loss,
                store,
                &entry.input_ids,
                inner_profile.as_mut(),
            )?;

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

        match result {
            Ok((grads, keep)) => {
                store.free_new_except(&live_before, &keep)?;
                trim_after_checkpoint_replay(store)?;
                if let (Some(outer), Some(mut inner)) = (profile, inner_profile) {
                    // The inner wall already sits inside this entry's own
                    // Checkpoint envelope — merge the attribution rows only.
                    inner.total_duration = Duration::ZERO;
                    outer.merge(&inner);
                }
                Ok(grads)
            }
            Err(err) => {
                let _ = store.free_new_except(&live_before, &HashSet::new());
                let _ = trim_after_checkpoint_replay(store);
                Err(err)
            }
        }
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

fn trim_after_checkpoint_replay(store: &TensorStore) -> Result<()> {
    if crate::runtime_flags::trim_after_checkpoint_replay() {
        store.backend().trim_memory_pool()?;
    }
    Ok(())
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
        // `existing.data` — host-resident from then on. See
        // `docs/research/2026-05-17-cuda-training-architectural-correction.md`.
        let both_on_device = {
            let existing = store.tensor(existing_grad_id)?;
            let incoming = store.tensor(new_grad_id)?;
            existing.dirty != Dirty::Host
                && existing.device_handle.is_some()
                && incoming.dirty != Dirty::Host
                && incoming.device_handle.is_some()
        };
        if both_on_device {
            let existing_handle = store
                .tensor(existing_grad_id)?
                .device_handle
                .as_ref()
                .expect("checked above")
                .clone();
            let incoming_handle = store
                .tensor(new_grad_id)?
                .device_handle
                .as_ref()
                .expect("checked above")
                .clone();
            let sum_handle =
                store
                    .backend()
                    .add_into_device(&existing_handle, &incoming_handle, &expected)?;
            store.replace_device_handle(existing_grad_id, sum_handle)?;
        } else {
            let incoming_data = store.to_host(new_grad_id)?;
            let existing = store.tensor_mut(existing_grad_id)?;
            for (dst, src) in existing.data.iter_mut().zip(incoming_data) {
                *dst += src;
            }
        }
        if should_store_grad {
            store.accumulate_grad(tensor_id, new_grad_id)?;
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
            store.accumulate_grad(tensor_id, new_grad_id)?;
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
