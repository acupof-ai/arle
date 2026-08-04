use std::{
    collections::BTreeMap,
    sync::{
        LazyLock, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    backend::{
        Device, LinearAttentionDeviceBackwardArgs, LinearAttentionDeviceBoundaryArgs,
        LinearAttentionDeviceForwardArgs, LinearAttentionDeviceParams,
        LinearAttentionScanBackwardArgs, LinearAttentionScanBackwardParams,
    },
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

#[derive(Debug, Clone, Copy)]
pub struct LinearAttentionParams {
    pub batch: usize,
    pub seq_len: usize,
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
    pub eps: f32,
}

impl From<LinearAttentionParams> for LinearAttentionDeviceParams {
    fn from(p: LinearAttentionParams) -> Self {
        Self {
            batch: p.batch,
            seq_len: p.seq_len,
            num_key_heads: p.num_key_heads,
            num_value_heads: p.num_value_heads,
            key_dim: p.key_dim,
            value_dim: p.value_dim,
            conv_kernel: p.conv_kernel,
            eps: p.eps,
        }
    }
}

struct LinearAttentionForward {
    output: Vec<f32>,
    preact: Vec<f32>,
    beta: Vec<f32>,
    exp_g: Vec<f32>,
    kv_mem: Vec<f32>,
    state_history: Vec<f32>,
    final_state: Vec<f32>,
    /// Last `conv_kernel - 1` qkv input rows, laid out `[batch, conv_kernel-1, qkv_dim]`.
    /// Captured so a follow-on segment can seed its causal conv ring instead of
    /// zero-padding absolute-position-0 (OPD frozen-prompt-KV split). Empty when
    /// `conv_kernel <= 1` (no ring to carry).
    conv_tail: Vec<f32>,
}

/// Carried recurrent context for splitting a single linear-attention sequence
/// across two `linear_attention_core` calls (prompt segment -> generated
/// segment). The gated-delta recurrence is Markovian in `state`, and the conv1d
/// is a fixed-width causal window, so seeding both reproduces the suffix exactly.
///
/// - `initial_state`: per-batch SSM state `[batch, num_value_heads, key_dim, value_dim]`
///   (a prior segment's `final_state`); `None` seeds zeros (absolute position 0).
/// - `initial_conv_window`: the prior segment's `conv_tail`
///   `[batch, conv_kernel-1, qkv_dim]`; `None` zero-pads the conv ring.
#[derive(Clone, Copy, Default)]
struct LinearAttentionCarry<'a> {
    initial_state: Option<&'a [f32]>,
    initial_conv_window: Option<&'a [f32]>,
}

impl<'a> LinearAttentionCarry<'a> {
    const NONE: Self = Self {
        initial_state: None,
        initial_conv_window: None,
    };
}

#[derive(Debug, Clone, Copy, Default)]
struct LinearAttentionSubopProfile {
    count: usize,
    duration: Duration,
}

#[derive(Debug, Clone, Default)]
struct LinearAttentionBackwardProfile {
    subops: BTreeMap<&'static str, LinearAttentionSubopProfile>,
}

impl LinearAttentionBackwardProfile {
    fn record(&mut self, subop: &'static str, duration: Duration) {
        let entry = self.subops.entry(subop).or_default();
        entry.count += 1;
        entry.duration += duration;
    }

    fn merge(&mut self, other: &Self) {
        for (&subop, &stats) in &other.subops {
            let entry = self.subops.entry(subop).or_default();
            entry.count += stats.count;
            entry.duration += stats.duration;
        }
    }

    fn total_duration(&self) -> Duration {
        self.subops
            .values()
            .fold(Duration::default(), |acc, stats| acc + stats.duration)
    }
}

static LINEAR_ATTENTION_BACKWARD_PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);
static LINEAR_ATTENTION_BACKWARD_PROFILE_TOTALS: LazyLock<Mutex<LinearAttentionBackwardProfile>> =
    LazyLock::new(|| Mutex::new(LinearAttentionBackwardProfile::default()));

fn linear_attention_backward_profile_enabled() -> bool {
    std::env::var_os("ARLE_OPD_BACKWARD_PROFILE").is_some()
}

fn subop_started(profile: &Option<LinearAttentionBackwardProfile>) -> Option<Instant> {
    profile.as_ref().map(|_| Instant::now())
}

fn record_subop(
    profile: &mut Option<LinearAttentionBackwardProfile>,
    subop: &'static str,
    duration: Duration,
) {
    if let Some(profile) = profile {
        profile.record(subop, duration);
    }
}

fn record_elapsed_subop(
    profile: &mut Option<LinearAttentionBackwardProfile>,
    subop: &'static str,
    started: Option<Instant>,
) -> Duration {
    let duration = elapsed_subop(started);
    record_subop(profile, subop, duration);
    duration
}

fn elapsed_subop(started: Option<Instant>) -> Duration {
    started.map_or(Duration::default(), |started| started.elapsed())
}

fn log_linear_attention_backward_profile(profile: &LinearAttentionBackwardProfile) {
    let call_index = LINEAR_ATTENTION_BACKWARD_PROFILE_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    print_linear_attention_backward_profile("call", call_index, profile);

    let aggregate = {
        let mut aggregate = LINEAR_ATTENTION_BACKWARD_PROFILE_TOTALS
            .lock()
            .expect("linear attention backward profile mutex poisoned");
        aggregate.merge(profile);
        aggregate.clone()
    };
    print_linear_attention_backward_profile("aggregate", call_index, &aggregate);
}

fn print_linear_attention_backward_profile(
    scope: &str,
    call_index: u64,
    profile: &LinearAttentionBackwardProfile,
) {
    let total_secs = profile.total_duration().as_secs_f64();
    let mut rows = profile
        .subops
        .iter()
        .map(|(&subop, stats)| (subop, stats.count, stats.duration.as_secs_f64()))
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(b.0)));
    for (rank, (subop, count, seconds)) in rows.iter().enumerate() {
        let pct_linear_attention = if total_secs == 0.0 {
            0.0
        } else {
            seconds / total_secs * 100.0
        };
        eprintln!(
            "opd_linear_attention_subop_profile scope={scope} calls={call_index} rank={} \
             subop={} count={} seconds={seconds:.6} pct_linear_attention={pct_linear_attention:.3}",
            rank + 1,
            subop,
            count
        );
    }
}

pub fn linear_attention_core(
    qkv: TensorId,
    z: TensorId,
    b_proj: TensorId,
    a_proj: TensorId,
    conv1d_weight: TensorId,
    dt_bias: TensorId,
    a_log: TensorId,
    norm_weight: TensorId,
    params: LinearAttentionParams,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    validate_shapes(
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        params,
        store,
    )?;

    let requires_grad = [
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
    ]
    .into_iter()
    .try_fold(false, |acc, tensor_id| {
        store
            .tensor(tensor_id)
            .map(|tensor| acc || tensor.requires_grad)
    })?;

    if let Some(output_id) = try_linear_attention_forward_device(
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        params,
        requires_grad,
        None,
        None,
        store,
        tape,
    )? {
        return Ok(output_id);
    }

    for tensor_id in [
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
    ] {
        store.ensure_host(tensor_id)?;
    }

    let qkv_tensor = store.tensor_host(qkv)?;
    let z_tensor = store.tensor_host(z)?;
    let b_tensor = store.tensor_host(b_proj)?;
    let a_tensor = store.tensor_host(a_proj)?;
    let conv_tensor = store.tensor_host(conv1d_weight)?;
    let dt_tensor = store.tensor_host(dt_bias)?;
    let a_log_tensor = store.tensor_host(a_log)?;
    let norm_tensor = store.tensor_host(norm_weight)?;

    let forward = linear_attention_forward(
        &qkv_tensor.data,
        &z_tensor.data,
        &b_tensor.data,
        &a_tensor.data,
        &conv_tensor.data,
        &conv_tensor.shape,
        &dt_tensor.data,
        &a_log_tensor.data,
        &norm_tensor.data,
        params,
        LinearAttentionCarry::NONE,
    );

    let output_shape = vec![
        params.batch,
        params.seq_len,
        params.num_value_heads * params.value_dim,
    ];
    let output_id = store.alloc(Tensor::new(forward.output, output_shape, requires_grad)?);

    TapeEntry {
        op: BackwardOp::LinearAttention,
        output_id,
        input_ids: smallvec![
            qkv,
            z,
            b_proj,
            a_proj,
            conv1d_weight,
            dt_bias,
            a_log,
            norm_weight
        ],
        saved: SavedContext::LinearAttentionCtx {
            qkv,
            z,
            b_proj,
            a_proj,
            conv1d_weight,
            dt_bias,
            a_log,
            norm_weight,
            preact: None,
            qkv_conv: None,
            g: None,
            beta: None,
            chunk_state: None,
            raw_output: None,
            initial_state: None,
            initial_conv_window: None,
            batch: params.batch,
            seq_len: params.seq_len,
            num_key_heads: params.num_key_heads,
            num_value_heads: params.num_value_heads,
            key_dim: params.key_dim,
            value_dim: params.value_dim,
            conv_kernel: params.conv_kernel,
            eps: params.eps,
        },
    }
    .record(store, tape)?;

    Ok(output_id)
}

/// Context-parallel gated-delta attention: all-to-all the sequence shard into the
/// head axis, run the full-sequence recurrence on this rank's head slice, all-to-all
/// back. Model-agnostic — parameterized by `LinearAttentionParams`, so any
/// gated-delta / Mamba-hybrid model reuses it (the sibling of `cp_causal_sdpa` for
/// full attention). `cp_size == 1` is `linear_attention_core` verbatim (byte-
/// identical single card). This mirrors Megatron's gated-delta-net CP: the fused
/// qkv rides one all-to-all per region and the packed conv weight is section-sliced
/// per rank — `linear_attention_core`'s interface is untouched.
///
/// Correctness rests on head independence: the recurrence never crosses value-heads
/// (state, conv taps, `a_log[h]`, `dt_bias[h]`, `beta[h]`, per-head rmsnorm are all
/// head-local), so running it on a 1/N head slice and concatenating == running on
/// all heads. That math is proven GPU-free by the head-split parity test; the a2a
/// transport (world>1) is pod-only (`all_to_all` world>1 is pending-remote NCCL).
///
/// `params.seq_len` is this rank's shard; the full sequence is `seq_len * cp_size`.
/// Requires `num_{value,key}_heads % cp_size == 0`.
#[allow(clippy::too_many_arguments)]
pub fn linear_attention_core_cp(
    qkv: TensorId,
    z: TensorId,
    b_proj: TensorId,
    a_proj: TensorId,
    conv1d_weight: TensorId,
    dt_bias: TensorId,
    a_log: TensorId,
    norm_weight: TensorId,
    params: LinearAttentionParams,
    cp_size: usize,
    cp_rank: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    if cp_size <= 1 {
        return linear_attention_core(
            qkv,
            z,
            b_proj,
            a_proj,
            conv1d_weight,
            dt_bias,
            a_log,
            norm_weight,
            params,
            store,
            tape,
        );
    }
    let n = cp_size;
    if !params.num_value_heads.is_multiple_of(n) || !params.num_key_heads.is_multiple_of(n) {
        return Err(AutogradError::TapeInvariant(
            "linear_attention_core_cp: num_value_heads and num_key_heads must divide cp_size",
        ));
    }
    let batch = params.batch;
    let local_seq = params.seq_len;
    let full_seq = local_seq * n;
    let (q_dim, k_dim, v_dim) = (
        params.num_key_heads * params.key_dim,
        params.num_key_heads * params.key_dim,
        params.num_value_heads * params.value_dim,
    );

    // qkv packs [q|k|v] with different head widths, so a contiguous dim/N slice
    // would cut a region. Split to q/k/v, all-to-all each to its head slice (heads
    // are outer within a region, so dim/N == this rank's head range), re-fuse.
    // z/b/a are single regions — one all-to-all each.
    let q = crate::ops::slice(qkv, &[0, 0, 0], &[batch, local_seq, q_dim], store, tape)?;
    let k = crate::ops::slice(
        qkv,
        &[0, 0, q_dim],
        &[batch, local_seq, q_dim + k_dim],
        store,
        tape,
    )?;
    let v = crate::ops::slice(
        qkv,
        &[0, 0, q_dim + k_dim],
        &[batch, local_seq, q_dim + k_dim + v_dim],
        store,
        tape,
    )?;
    let q = crate::ops::all_to_all(q, 1, 2, store, tape)?;
    let k = crate::ops::all_to_all(k, 1, 2, store, tape)?;
    let v = crate::ops::all_to_all(v, 1, 2, store, tape)?;
    let qkv = crate::ops::cat(&[q, k, v], 2, store, tape)?;
    let z = crate::ops::all_to_all(z, 1, 2, store, tape)?;
    let b_proj = crate::ops::all_to_all(b_proj, 1, 2, store, tape)?;
    let a_proj = crate::ops::all_to_all(a_proj, 1, 2, store, tape)?;

    // a2a leaves the seq blocks interleaved; the recurrence needs true global order.
    let (fwd, phys) = zigzag_block_perms(n);
    let qkv = reorder_seq_blocks(qkv, &fwd, store, tape)?;
    let z = reorder_seq_blocks(z, &fwd, store, tape)?;
    let b_proj = reorder_seq_blocks(b_proj, &fwd, store, tape)?;
    let a_proj = reorder_seq_blocks(a_proj, &fwd, store, tape)?;

    // Frozen weights sliced to this rank's head range. conv1d packs [q|k|v] on the
    // channel axis (same region surgery); dt_bias/a_log are per-value-head; norm is
    // per-value_dim (shared across heads → unsliced). Read-only slices — base
    // tensors, not LoRA, so no gradient reassembly.
    let (kh_l, vh_l) = (params.num_key_heads / n, params.num_value_heads / n);
    let conv1d_weight = slice_conv_weight_to_head(
        conv1d_weight,
        params.num_key_heads,
        params.num_value_heads,
        params.key_dim,
        params.value_dim,
        cp_rank,
        n,
        store,
        tape,
    )?;
    let dt_bias = crate::ops::slice(
        dt_bias,
        &[cp_rank * vh_l],
        &[(cp_rank + 1) * vh_l],
        store,
        tape,
    )?;
    let a_log = crate::ops::slice(
        a_log,
        &[cp_rank * vh_l],
        &[(cp_rank + 1) * vh_l],
        store,
        tape,
    )?;

    let local_params = LinearAttentionParams {
        batch,
        seq_len: full_seq,
        num_key_heads: kh_l,
        num_value_heads: vh_l,
        ..params
    };
    let out = linear_attention_core(
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        local_params,
        store,
        tape,
    )?;
    // Global order -> a2a physical layout, then restore the [b, local_seq, v_dim] shard.
    let out = reorder_seq_blocks(out, &phys, store, tape)?;
    crate::ops::all_to_all(out, 2, 1, store, tape)
}

/// `(fwd, phys)` for the `2N`-block seq permutation. Zigzag gives rank `r` global
/// chunks `r`,`2N-1-r`; a2a lays ranks in order, so physical block `2r`=chunk `r`,
/// `2r+1`=chunk `2N-1-r`. `fwd` un-interleaves to global order, `phys` inverts it.
fn zigzag_block_perms(n: usize) -> (Vec<usize>, Vec<usize>) {
    let two_n = 2 * n;
    let mut fwd = vec![0usize; two_n];
    for r in 0..n {
        fwd[r] = 2 * r;
        fwd[two_n - 1 - r] = 2 * r + 1;
    }
    let mut phys = vec![0usize; two_n];
    for (g, &p) in fwd.iter().enumerate() {
        phys[p] = g;
    }
    (fwd, phys)
}

/// Permute `x`'s seq axis by blocks: output block `i` = input block `perm[i]`.
/// Pure slice+cat, so backward reassembles the gradient blocks for free.
fn reorder_seq_blocks(
    x: TensorId,
    perm: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let shape = store.tensor(x)?.shape.clone();
    let (full_seq, block) = (shape[1], shape[1] / perm.len());
    let mut blocks = Vec::with_capacity(perm.len());
    for &src in perm {
        blocks.push(crate::ops::slice(
            x,
            &[0, src * block, 0],
            &[shape[0], (src + 1) * block, shape[2]],
            store,
            tape,
        )?);
    }
    debug_assert_eq!(full_seq, perm.len() * block);
    crate::ops::cat(&blocks, 1, store, tape)
}

/// Slice the packed `[qkv_dim, conv_kernel]` conv weight to this cp rank's head
/// range. conv_dim packs q|k|v channel regions; the rank owns a contiguous head
/// slice within each, gathered and re-fused so the local weight matches the a2a'd
/// activation layout.
#[allow(clippy::too_many_arguments)]
fn slice_conv_weight_to_head(
    conv1d_weight: TensorId,
    num_key_heads: usize,
    num_value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    cp_rank: usize,
    cp_size: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let (q_dim, k_dim) = (num_key_heads * key_dim, num_key_heads * key_dim);
    let (qk_local, v_local) = (
        num_key_heads / cp_size * key_dim,
        num_value_heads / cp_size * value_dim,
    );
    let conv_kernel = store.tensor(conv1d_weight)?.shape[1];
    // Region base offsets and this rank's local widths: [q | k | v].
    let regions = [
        (0usize, qk_local),
        (q_dim, qk_local),
        (q_dim + k_dim, v_local),
    ];
    let sliced: Vec<TensorId> = regions
        .iter()
        .map(|&(base, width)| {
            let start = base + cp_rank * width;
            crate::ops::slice(
                conv1d_weight,
                &[start, 0],
                &[start + width, conv_kernel],
                store,
                tape,
            )
        })
        .collect::<Result<_>>()?;
    crate::ops::cat(&sliced, 0, store, tape)
}

/// Host reference with recurrent and convolution carry.
#[allow(clippy::too_many_arguments)]
pub fn linear_attention_core_with_carry(
    qkv: TensorId,
    z: TensorId,
    b_proj: TensorId,
    a_proj: TensorId,
    conv1d_weight: TensorId,
    dt_bias: TensorId,
    a_log: TensorId,
    norm_weight: TensorId,
    params: LinearAttentionParams,
    initial_state: Option<TensorId>,
    initial_conv_window: Option<TensorId>,
    capture_boundary: bool,
    store: &mut TensorStore,
) -> Result<(TensorId, Option<TensorId>, Option<TensorId>)> {
    validate_shapes(
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        params,
        store,
    )?;

    for tensor_id in [
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
    ]
    .into_iter()
    .chain(initial_state)
    .chain(initial_conv_window)
    {
        store.ensure_host(tensor_id)?;
    }

    let qkv_tensor = store.tensor_host(qkv)?;
    let z_tensor = store.tensor_host(z)?;
    let b_tensor = store.tensor_host(b_proj)?;
    let a_tensor = store.tensor_host(a_proj)?;
    let conv_tensor = store.tensor_host(conv1d_weight)?;
    let dt_tensor = store.tensor_host(dt_bias)?;
    let a_log_tensor = store.tensor_host(a_log)?;
    let norm_tensor = store.tensor_host(norm_weight)?;

    let conv_window = params.conv_kernel - 1;
    let initial_state_data = initial_state.map(|id| store.tensor_host(id)).transpose()?;
    let initial_conv_data = initial_conv_window
        .map(|id| store.tensor_host(id))
        .transpose()?;
    if let Some(state_tensor) = initial_state_data.as_ref() {
        let expected = params.batch * params.num_value_heads * params.key_dim * params.value_dim;
        if state_tensor.data.len() != expected {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![
                    params.batch,
                    params.num_value_heads,
                    params.key_dim,
                    params.value_dim,
                ],
                got: state_tensor.shape.clone(),
            });
        }
    }
    if let Some(window_tensor) = initial_conv_data.as_ref() {
        let q_dim = params.num_key_heads * params.key_dim;
        let qkv_dim = q_dim * 2 + params.num_value_heads * params.value_dim;
        let expected = params.batch * conv_window * qkv_dim;
        if window_tensor.data.len() != expected {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![params.batch, conv_window, qkv_dim],
                got: window_tensor.shape.clone(),
            });
        }
    }

    let carry = LinearAttentionCarry {
        initial_state: initial_state_data.as_ref().map(|t| t.data.as_slice()),
        initial_conv_window: initial_conv_data.as_ref().map(|t| t.data.as_slice()),
    };

    let forward = linear_attention_forward(
        &qkv_tensor.data,
        &z_tensor.data,
        &b_tensor.data,
        &a_tensor.data,
        &conv_tensor.data,
        &conv_tensor.shape,
        &dt_tensor.data,
        &a_log_tensor.data,
        &norm_tensor.data,
        params,
        carry,
    );

    let output_shape = vec![
        params.batch,
        params.seq_len,
        params.num_value_heads * params.value_dim,
    ];
    let output_id = store.alloc(Tensor::new(forward.output, output_shape, false)?);

    let (final_state_id, conv_window_id) = if capture_boundary {
        let q_dim = params.num_key_heads * params.key_dim;
        let qkv_dim = q_dim * 2 + params.num_value_heads * params.value_dim;
        let final_state_id = store.alloc(Tensor::new(
            forward.final_state,
            vec![
                params.batch,
                params.num_value_heads,
                params.key_dim,
                params.value_dim,
            ],
            false,
        )?);
        let conv_window_id = store.alloc(Tensor::new(
            forward.conv_tail,
            vec![params.batch, conv_window, qkv_dim],
            false,
        )?);
        (Some(final_state_id), Some(conv_window_id))
    } else {
        (None, None)
    };

    Ok((output_id, final_state_id, conv_window_id))
}

#[allow(clippy::too_many_arguments)]
pub fn linear_attention_boundary(
    qkv: TensorId,
    b_proj: TensorId,
    a_proj: TensorId,
    conv1d_weight: TensorId,
    dt_bias: TensorId,
    a_log: TensorId,
    params: LinearAttentionParams,
    initial_state: Option<TensorId>,
    initial_conv_window: Option<TensorId>,
    store: &mut TensorStore,
) -> Result<(TensorId, TensorId)> {
    validate_boundary_shapes(
        qkv,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        params,
        store,
    )?;

    let window = params.conv_kernel - 1;
    if store.backend().device() != Device::Cuda || params.seq_len < window {
        return Err(AutogradError::TapeInvariant(
            "linear attention boundary requires CUDA and a full conv window",
        ));
    }

    let qkv_handle = store.device_handle(qkv)?;
    let b_handle = store.device_handle(b_proj)?;
    let a_handle = store.device_handle(a_proj)?;
    let conv_handle = store.device_handle(conv1d_weight)?;
    let dt_handle = store.device_handle(dt_bias)?;
    let a_log_handle = store.device_handle(a_log)?;
    let initial_state_handle = initial_state
        .map(|id| store.device_handle(id))
        .transpose()?;
    let initial_conv_handle = initial_conv_window
        .map(|id| store.device_handle(id))
        .transpose()?;
    let state = store
        .backend()
        .linear_attention_boundary_device(LinearAttentionDeviceBoundaryArgs {
            params: params.into(),
            qkv: &qkv_handle,
            b_proj: &b_handle,
            a_proj: &a_handle,
            conv1d_weight: &conv_handle,
            dt_bias: &dt_handle,
            a_log: &a_log_handle,
            initial_state: initial_state_handle.as_ref(),
            initial_conv_window: initial_conv_handle.as_ref(),
        })?
        .ok_or(AutogradError::TapeInvariant(
            "linear attention boundary unsupported",
        ))?;
    let state = store.alloc_device_tensor(
        vec![
            params.batch,
            params.num_value_heads,
            params.key_dim,
            params.value_dim,
        ],
        state,
    )?;
    let qkv_dim =
        2 * params.num_key_heads * params.key_dim + params.num_value_heads * params.value_dim;
    let conv = if window == 0 {
        store.alloc(Tensor::new(
            Vec::new(),
            vec![params.batch, 0, qkv_dim],
            false,
        )?)
    } else {
        let mut tape = Tape::new();
        tape.set_enabled(false);
        crate::ops::slice(
            qkv,
            &[0, params.seq_len - window, 0],
            &[params.batch, params.seq_len, qkv_dim],
            store,
            &mut tape,
        )?
    };
    Ok((state, conv))
}

/// Taped generated segment seeded by frozen carry.
#[allow(clippy::too_many_arguments)]
pub fn linear_attention_core_with_carry_taped(
    qkv: TensorId,
    z: TensorId,
    b_proj: TensorId,
    a_proj: TensorId,
    conv1d_weight: TensorId,
    dt_bias: TensorId,
    a_log: TensorId,
    norm_weight: TensorId,
    params: LinearAttentionParams,
    initial_state: Option<TensorId>,
    initial_conv_window: Option<TensorId>,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    validate_shapes(
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        params,
        store,
    )?;

    let requires_grad = [
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
    ]
    .into_iter()
    .try_fold(false, |acc, tensor_id| {
        store
            .tensor(tensor_id)
            .map(|tensor| acc || tensor.requires_grad)
    })?;

    // Carry is constant; device backward reuses chunk_state[0].
    if let Some(output_id) = try_linear_attention_forward_device(
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        params,
        requires_grad,
        initial_state,
        initial_conv_window,
        store,
        tape,
    )? {
        return Ok(output_id);
    }

    // Host fallback (carry-aware device kernel unsupported for this shape/dtype).
    for tensor_id in [
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
    ]
    .into_iter()
    .chain(initial_state)
    .chain(initial_conv_window)
    {
        store.ensure_host(tensor_id)?;
    }

    let qkv_tensor = store.tensor_host(qkv)?;
    let z_tensor = store.tensor_host(z)?;
    let b_tensor = store.tensor_host(b_proj)?;
    let a_tensor = store.tensor_host(a_proj)?;
    let conv_tensor = store.tensor_host(conv1d_weight)?;
    let dt_tensor = store.tensor_host(dt_bias)?;
    let a_log_tensor = store.tensor_host(a_log)?;
    let norm_tensor = store.tensor_host(norm_weight)?;

    let conv_window = params.conv_kernel.saturating_sub(1);
    let initial_state_data = initial_state.map(|id| store.tensor_host(id)).transpose()?;
    let initial_conv_data = initial_conv_window
        .map(|id| store.tensor_host(id))
        .transpose()?;
    if let Some(state_tensor) = initial_state_data.as_ref() {
        let expected = params.batch * params.num_value_heads * params.key_dim * params.value_dim;
        if state_tensor.data.len() != expected {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![
                    params.batch,
                    params.num_value_heads,
                    params.key_dim,
                    params.value_dim,
                ],
                got: state_tensor.shape.clone(),
            });
        }
    }
    if let Some(window_tensor) = initial_conv_data.as_ref() {
        let q_dim = params.num_key_heads * params.key_dim;
        let qkv_dim = q_dim * 2 + params.num_value_heads * params.value_dim;
        let expected = params.batch * conv_window * qkv_dim;
        if window_tensor.data.len() != expected {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![params.batch, conv_window, qkv_dim],
                got: window_tensor.shape.clone(),
            });
        }
    }

    let carry = LinearAttentionCarry {
        initial_state: initial_state_data.as_ref().map(|t| t.data.as_slice()),
        initial_conv_window: initial_conv_data.as_ref().map(|t| t.data.as_slice()),
    };

    let forward = linear_attention_forward(
        &qkv_tensor.data,
        &z_tensor.data,
        &b_tensor.data,
        &a_tensor.data,
        &conv_tensor.data,
        &conv_tensor.shape,
        &dt_tensor.data,
        &a_log_tensor.data,
        &norm_tensor.data,
        params,
        carry,
    );

    let output_shape = vec![
        params.batch,
        params.seq_len,
        params.num_value_heads * params.value_dim,
    ];
    let output_id = store.alloc(Tensor::new(forward.output, output_shape, requires_grad)?);

    TapeEntry {
        op: BackwardOp::LinearAttention,
        output_id,
        input_ids: smallvec![
            qkv,
            z,
            b_proj,
            a_proj,
            conv1d_weight,
            dt_bias,
            a_log,
            norm_weight
        ],
        saved: SavedContext::LinearAttentionCtx {
            qkv,
            z,
            b_proj,
            a_proj,
            conv1d_weight,
            dt_bias,
            a_log,
            norm_weight,
            preact: None,
            qkv_conv: None,
            g: None,
            beta: None,
            chunk_state: None,
            raw_output: None,
            initial_state,
            initial_conv_window,
            batch: params.batch,
            seq_len: params.seq_len,
            num_key_heads: params.num_key_heads,
            num_value_heads: params.num_value_heads,
            key_dim: params.key_dim,
            value_dim: params.value_dim,
            conv_kernel: params.conv_kernel,
            eps: params.eps,
        },
    }
    .record(store, tape)?;

    Ok(output_id)
}

/// Packed q/k/v width, one row's worth.
fn qkv_elems(p: LinearAttentionParams) -> usize {
    2 * p.num_key_heads * p.key_dim + p.num_value_heads * p.value_dim
}

/// The row kernel's taped f32 inputs: qkv, z, b_proj, a_proj. Both byte counts
/// below include it, so summing them double-counts.
fn taped_input_elems(p: LinearAttentionParams) -> usize {
    qkv_elems(p) + p.num_value_heads * p.value_dim + 2 * p.num_value_heads
}

/// Bytes retained by `try_linear_attention_forward_device`.
pub fn linear_attention_ctx_bytes(params: LinearAttentionParams) -> usize {
    let (hv, kd, vd) = (params.num_value_heads, params.key_dim, params.value_dim);
    let seq = params.seq_len;
    let f32_elems = seq * taped_input_elems(params) + seq.div_ceil(64) * hv * kd * vd;
    let bf16_elems = seq * qkv_elems(params);
    params.batch * (4 * f32_elems + 2 * bf16_elems)
}

/// Live bytes inside `..._forward_device_row`. Beside the kernel so a buffer
/// change and its count move together.
pub fn linear_attention_row_transient_bytes(params: LinearAttentionParams) -> usize {
    let (hv, kd, vd) = (params.num_value_heads, params.key_dim, params.value_dim);
    // f32: a_tril, g_cumsum, plus the taped inputs.
    let f32_elems = 64 * hv + hv + taped_input_elems(params);
    // bf16: q/k/w, v/u/v_new/raw_output, a_inv, b/a casts.
    let bf16_elems = 3 * hv * kd + 4 * hv * vd + 64 * hv + 2 * hv;
    params
        .batch
        .saturating_mul(params.seq_len)
        .saturating_mul(4 * f32_elems + 2 * bf16_elems)
}

/// Pinned so editing the row kernel's buffers forces a deliberate recount.
#[test]
fn linear_attention_byte_counts_are_pinned() {
    let p = LinearAttentionParams {
        batch: 2,
        seq_len: 65,
        num_key_heads: 2,
        num_value_heads: 3,
        key_dim: 4,
        value_dim: 5,
        conv_kernel: 4,
        eps: 1e-6,
    };
    assert_eq!(linear_attention_ctx_bytes(p), 36_060);
    assert_eq!(linear_attention_row_transient_bytes(p), 204_880);
}

fn try_linear_attention_forward_device(
    qkv: TensorId,
    z: TensorId,
    b_proj: TensorId,
    a_proj: TensorId,
    conv1d_weight: TensorId,
    dt_bias: TensorId,
    a_log: TensorId,
    norm_weight: TensorId,
    params: LinearAttentionParams,
    requires_grad: bool,
    initial_state: Option<TensorId>,
    initial_conv_window: Option<TensorId>,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<Option<TensorId>> {
    if store.backend().device() != Device::Cuda {
        return Ok(None);
    }

    let qkv_handle = store.device_handle(qkv)?;
    let z_handle = store.device_handle(z)?;
    let b_handle = store.device_handle(b_proj)?;
    let a_handle = store.device_handle(a_proj)?;
    let conv_handle = store.device_handle(conv1d_weight)?;
    let dt_handle = store.device_handle(dt_bias)?;
    let a_log_handle = store.device_handle(a_log)?;
    let norm_handle = store.device_handle(norm_weight)?;
    let initial_state_handle = initial_state
        .map(|id| store.device_handle(id))
        .transpose()?;
    let initial_conv_handle = initial_conv_window
        .map(|id| store.device_handle(id))
        .transpose()?;

    let Some(result) =
        store
            .backend()
            .linear_attention_forward_device(LinearAttentionDeviceForwardArgs {
                params: params.into(),
                qkv: &qkv_handle,
                z: &z_handle,
                b_proj: &b_handle,
                a_proj: &a_handle,
                conv1d_weight: &conv_handle,
                dt_bias: &dt_handle,
                a_log: &a_log_handle,
                norm_weight: &norm_handle,
                initial_state: initial_state_handle.as_ref(),
                initial_conv_window: initial_conv_handle.as_ref(),
            })?
    else {
        return Ok(None);
    };

    let q_dim = params.num_key_heads * params.key_dim;
    let qkv_dim = q_dim * 2 + params.num_value_heads * params.value_dim;
    let num_chunks = params.seq_len.div_ceil(64);
    let output_shape = vec![
        params.batch,
        params.seq_len,
        params.num_value_heads * params.value_dim,
    ];
    let head_shape = vec![params.batch, params.seq_len, params.num_value_heads];
    let output_id = store.alloc_device_tensor(output_shape, result.output)?;

    if requires_grad {
        let preact_id = store
            .alloc_device_tensor(vec![params.batch, params.seq_len, qkv_dim], result.preact)?;
        let qkv_conv_id = store
            .alloc_device_tensor(vec![params.batch, params.seq_len, qkv_dim], result.qkv_conv)?;
        // FlashQLA re-derives g/beta in the backward, so they never reach the tape.
        let (g_id, beta_id) = if result.flashqla {
            (None, None)
        } else {
            (
                Some(store.alloc_device_tensor(head_shape.clone(), result.g)?),
                Some(store.alloc_device_tensor(head_shape, result.beta)?),
            )
        };
        // FlashQLA keeps only chunk 0 (the state carry) — it recomputes the rest.
        let chunk_state_id = store.alloc_device_tensor(
            vec![
                params.batch,
                if result.flashqla { 1 } else { num_chunks },
                params.num_value_heads,
                params.key_dim,
                params.value_dim,
            ],
            result.chunk_state,
        )?;
        let raw_output_id = result
            .flashqla
            .then(|| {
                store.alloc_device_tensor(
                    vec![
                        params.batch,
                        params.seq_len,
                        params.num_value_heads * params.value_dim,
                    ],
                    result.raw_output,
                )
            })
            .transpose()?;
        TapeEntry {
            op: BackwardOp::LinearAttention,
            output_id,
            input_ids: smallvec![
                qkv,
                z,
                b_proj,
                a_proj,
                conv1d_weight,
                dt_bias,
                a_log,
                norm_weight
            ],
            saved: SavedContext::LinearAttentionCtx {
                qkv,
                z,
                b_proj,
                a_proj,
                conv1d_weight,
                dt_bias,
                a_log,
                norm_weight,
                preact: Some(preact_id),
                qkv_conv: Some(qkv_conv_id),
                g: g_id,
                beta: beta_id,
                chunk_state: Some(chunk_state_id),
                raw_output: raw_output_id,
                // State carry lives in chunk_state[0] → None (Some would misfire
                // needs_host_recompute). Conv carry is a real backward input → keep it.
                initial_state: None,
                initial_conv_window,
                batch: params.batch,
                seq_len: params.seq_len,
                num_key_heads: params.num_key_heads,
                num_value_heads: params.num_value_heads,
                key_dim: params.key_dim,
                value_dim: params.value_dim,
                conv_kernel: params.conv_kernel,
                eps: params.eps,
            },
        }
        .record(store, tape)?;
    }
    Ok(Some(output_id))
}

#[allow(clippy::too_many_arguments)]
fn try_linear_attention_backward_device(
    output_grad_id: TensorId,
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
    chunk_state: Option<TensorId>,
    raw_output: Option<TensorId>,
    initial_conv_window: Option<TensorId>,
    params: LinearAttentionParams,
    store: &mut TensorStore,
) -> Result<Option<GradPairs>> {
    if store.backend().device() != Device::Cuda {
        return Ok(None);
    }
    let Some(preact) = preact else {
        return Ok(None);
    };
    let Some(qkv_conv) = qkv_conv else {
        return Ok(None);
    };
    let Some(chunk_state) = chunk_state else {
        return Ok(None);
    };
    // Only FlashQLA re-derives g/beta; every other route needs them off the tape.
    if raw_output.is_none() && (g.is_none() || beta.is_none()) {
        return Ok(None);
    }

    for tensor_id in [
        output_grad_id,
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        preact,
        qkv_conv,
        chunk_state,
    ]
    .into_iter()
    .chain(g)
    .chain(beta)
    .chain(raw_output)
    {
        store.ensure_device(tensor_id)?;
    }

    let handle = |store: &TensorStore, id: TensorId| -> Result<_> {
        Ok(store
            .tensor(id)?
            .device_handle
            .as_ref()
            .ok_or(AutogradError::TapeInvariant(
                "linear attention device tensor missing handle",
            ))?
            .clone())
    };
    let upstream_handle = handle(store, output_grad_id)?;
    let qkv_handle = handle(store, qkv)?;
    let z_handle = handle(store, z)?;
    let b_handle = handle(store, b_proj)?;
    let a_handle = handle(store, a_proj)?;
    let conv_handle = handle(store, conv1d_weight)?;
    let dt_handle = handle(store, dt_bias)?;
    let a_log_handle = handle(store, a_log)?;
    let norm_handle = handle(store, norm_weight)?;
    let preact_handle = handle(store, preact)?;
    let qkv_conv_handle = handle(store, qkv_conv)?;
    let g_handle = g.map(|id| handle(store, id)).transpose()?;
    let beta_handle = beta.map(|id| handle(store, id)).transpose()?;
    let chunk_state_handle = handle(store, chunk_state)?;
    let raw_output_handle = raw_output.map(|id| handle(store, id)).transpose()?;
    let conv_tail_handle = initial_conv_window
        .map(|id| {
            store.ensure_device(id)?;
            handle(store, id)
        })
        .transpose()?;

    let Some(device_grads) =
        store
            .backend()
            .linear_attention_backward_device(LinearAttentionDeviceBackwardArgs {
                params: LinearAttentionDeviceParams {
                    batch: params.batch,
                    seq_len: params.seq_len,
                    num_key_heads: params.num_key_heads,
                    num_value_heads: params.num_value_heads,
                    key_dim: params.key_dim,
                    value_dim: params.value_dim,
                    conv_kernel: params.conv_kernel,
                    eps: params.eps,
                },
                upstream: &upstream_handle,
                qkv: &qkv_handle,
                z: &z_handle,
                b_proj: &b_handle,
                a_proj: &a_handle,
                conv1d_weight: &conv_handle,
                dt_bias: &dt_handle,
                a_log: &a_log_handle,
                norm_weight: &norm_handle,
                preact: &preact_handle,
                qkv_conv: &qkv_conv_handle,
                g: g_handle.as_ref(),
                beta: beta_handle.as_ref(),
                chunk_state: &chunk_state_handle,
                raw_output: raw_output_handle.as_ref(),
                initial_conv_window: conv_tail_handle.as_ref(),
            })?
    else {
        return Ok(None);
    };

    let q_dim = params.num_key_heads * params.key_dim;
    let qkv_dim = q_dim * 2 + params.num_value_heads * params.value_dim;
    let mut grads = GradPairs::new();
    if store.tensor(qkv)?.requires_grad {
        grads.push((
            qkv,
            store.alloc_device_tensor(
                vec![params.batch, params.seq_len, qkv_dim],
                device_grads.dqkv,
            )?,
        ));
    }
    if store.tensor(z)?.requires_grad {
        grads.push((
            z,
            store.alloc_device_tensor(
                vec![
                    params.batch,
                    params.seq_len,
                    params.num_value_heads * params.value_dim,
                ],
                device_grads.dz,
            )?,
        ));
    }
    if store.tensor(b_proj)?.requires_grad {
        grads.push((
            b_proj,
            store.alloc_device_tensor(
                vec![params.batch, params.seq_len, params.num_value_heads],
                device_grads.db,
            )?,
        ));
    }
    if store.tensor(a_proj)?.requires_grad {
        grads.push((
            a_proj,
            store.alloc_device_tensor(
                vec![params.batch, params.seq_len, params.num_value_heads],
                device_grads.da,
            )?,
        ));
    }
    if store.tensor(conv1d_weight)?.requires_grad {
        grads.push((
            conv1d_weight,
            store.alloc_device_tensor(vec![qkv_dim, params.conv_kernel], device_grads.dconv)?,
        ));
    }
    if store.tensor(dt_bias)?.requires_grad {
        grads.push((
            dt_bias,
            store.alloc_device_tensor(vec![params.num_value_heads], device_grads.ddt)?,
        ));
    }
    if store.tensor(a_log)?.requires_grad {
        grads.push((
            a_log,
            store.alloc_device_tensor(vec![params.num_value_heads], device_grads.da_log)?,
        ));
    }
    if store.tensor(norm_weight)?.requires_grad {
        grads.push((
            norm_weight,
            store.alloc_device_tensor(vec![params.value_dim], device_grads.dnorm)?,
        ));
    }
    Ok(Some(grads))
}

pub(crate) fn linear_attention_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::LinearAttentionCtx {
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        preact,
        qkv_conv,
        g,
        beta,
        chunk_state,
        raw_output,
        initial_state,
        initial_conv_window,
        batch,
        seq_len,
        num_key_heads,
        num_value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        eps,
    } = entry.saved.clone()
    else {
        return Err(AutogradError::TapeInvariant(
            "linear attention backward missing saved context",
        ));
    };

    let params = LinearAttentionParams {
        batch,
        seq_len,
        num_key_heads,
        num_value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        eps,
    };
    validate_shapes(
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        params,
        store,
    )?;

    // Only the recurrent-state carry forces the host recompute (it must rebuild the
    // full-sequence state); the conv-window carry is just a device backward input,
    // fed through below. The default path (both None) still takes the device backward.
    let needs_host_recompute = initial_state.is_some();
    if !needs_host_recompute
        && let Some(grads) = try_linear_attention_backward_device(
            output_grad_id,
            qkv,
            z,
            b_proj,
            a_proj,
            conv1d_weight,
            dt_bias,
            a_log,
            norm_weight,
            preact,
            qkv_conv,
            g,
            beta,
            chunk_state,
            raw_output,
            initial_conv_window,
            params,
            store,
        )?
    {
        return Ok(grads);
    }

    let mut profile =
        linear_attention_backward_profile_enabled().then(LinearAttentionBackwardProfile::default);
    let materialize_started = subop_started(&profile);
    for tensor_id in [
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
    ]
    .into_iter()
    .chain(initial_state)
    .chain(initial_conv_window)
    {
        store.ensure_host(tensor_id)?;
    }

    let upstream = store.tensor_host(output_grad_id)?;
    let qkv_tensor = store.tensor_host(qkv)?;
    let z_tensor = store.tensor_host(z)?;
    let b_tensor = store.tensor_host(b_proj)?;
    let a_tensor = store.tensor_host(a_proj)?;
    let conv_tensor = store.tensor_host(conv1d_weight)?;
    let dt_tensor = store.tensor_host(dt_bias)?;
    let a_log_tensor = store.tensor_host(a_log)?;
    let norm_tensor = store.tensor_host(norm_weight)?;

    let expected_shape = vec![batch, seq_len, num_value_heads * value_dim];
    if upstream.shape != expected_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_shape,
            got: upstream.shape,
        });
    }
    // Re-seed the OPD frozen-prompt-KV carry so the recompute reproduces the
    // forward's preact (conv taps) + initial recurrent state exactly — without
    // this the boundary grads of the generated segment would be wrong. The carry
    // tensors are constants (requires_grad=false); the grad scan never propagates
    // into them, so only the seeding matters here.
    let initial_state_data = initial_state.map(|id| store.tensor_host(id)).transpose()?;
    let initial_conv_data = initial_conv_window
        .map(|id| store.tensor_host(id))
        .transpose()?;
    let carry = LinearAttentionCarry {
        initial_state: initial_state_data.as_ref().map(|t| t.data.as_slice()),
        initial_conv_window: initial_conv_data.as_ref().map(|t| t.data.as_slice()),
    };
    record_elapsed_subop(&mut profile, "host_materialize", materialize_started);

    let recompute_started = subop_started(&profile);
    let forward = linear_attention_forward(
        &qkv_tensor.data,
        &z_tensor.data,
        &b_tensor.data,
        &a_tensor.data,
        &conv_tensor.data,
        &conv_tensor.shape,
        &dt_tensor.data,
        &a_log_tensor.data,
        &norm_tensor.data,
        params,
        carry,
    );
    record_elapsed_subop(&mut profile, "fwd_recompute", recompute_started);

    let grad_alloc_started = subop_started(&profile);
    let q_dim = num_key_heads * key_dim;
    let k_dim = q_dim;
    let v_offset = q_dim + k_dim;
    let mut dqkv = vec![0.0_f32; qkv_tensor.data.len()];
    let mut dz = vec![0.0_f32; z_tensor.data.len()];
    let mut db = vec![0.0_f32; b_tensor.data.len()];
    let mut da = vec![0.0_f32; a_tensor.data.len()];
    let mut ddt = vec![0.0_f32; dt_tensor.data.len()];
    let mut da_log = vec![0.0_f32; a_log_tensor.data.len()];
    let mut dnorm = vec![0.0_f32; norm_tensor.data.len()];
    record_elapsed_subop(&mut profile, "grad_alloc", grad_alloc_started);

    let scan_started = subop_started(&profile);
    let mut nested_param_grad_duration = Duration::default();
    if let Some(cuda_grads) =
        store
            .backend()
            .linear_attention_scan_backward(LinearAttentionScanBackwardArgs {
                params: LinearAttentionScanBackwardParams {
                    batch,
                    seq_len,
                    num_key_heads,
                    num_value_heads,
                    key_dim,
                    value_dim,
                    eps,
                },
                upstream: &upstream.data,
                z: &z_tensor.data,
                a_proj: &a_tensor.data,
                dt_bias: &dt_tensor.data,
                a_log: &a_log_tensor.data,
                norm_weight: &norm_tensor.data,
                preact: &forward.preact,
                beta: &forward.beta,
                exp_g: &forward.exp_g,
                kv_mem: &forward.kv_mem,
                state_history: &forward.state_history,
                final_state: &forward.final_state,
            })?
    {
        dqkv = cuda_grads.dqkv;
        dz = cuda_grads.dz;
        db = cuda_grads.db;
        da = cuda_grads.da;
        ddt = cuda_grads.ddt;
        da_log = cuda_grads.da_log;
        dnorm = cuda_grads.dnorm;
    } else {
        for batch_idx in 0..batch {
            for value_head in 0..num_value_heads {
                let key_head = value_head * num_key_heads / num_value_heads;
                let mut state = forward.final_state[state_base(
                    batch_idx,
                    value_head,
                    num_value_heads,
                    key_dim,
                    value_dim,
                )
                    ..state_base(batch_idx, value_head, num_value_heads, key_dim, value_dim)
                        + key_dim * value_dim]
                    .to_vec();
                let mut grad_state = vec![0.0_f32; key_dim * value_dim];
                let exp_a = a_log_tensor.data[value_head].exp();

                for seq_idx in (0..seq_len).rev() {
                    let preact_row = row3(
                        &forward.preact,
                        batch_idx,
                        seq_idx,
                        seq_len,
                        qkv_tensor.shape[2],
                    );
                    let q_raw =
                        silu_slice(&preact_row[key_head * key_dim..(key_head + 1) * key_dim]);
                    let k_raw = silu_slice(
                        &preact_row[q_dim + key_head * key_dim..q_dim + (key_head + 1) * key_dim],
                    );
                    let v_raw = silu_slice(
                        &preact_row[v_offset + value_head * value_dim
                            ..v_offset + (value_head + 1) * value_dim],
                    );

                    let q = l2_normalize_scaled(&q_raw, 1.0 / (key_dim as f32).sqrt());
                    let k = l2_normalize_scaled(&k_raw, 1.0);
                    let beta = forward.beta
                        [idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)];
                    let exp_g = forward.exp_g
                        [idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)];
                    let a_value = a_tensor.data
                        [idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)];
                    let softplus_input = a_value + dt_tensor.data[value_head];
                    let softplus_value = softplus_scalar(softplus_input);
                    let kv_mem = row4(
                        &forward.kv_mem,
                        batch_idx,
                        seq_idx,
                        value_head,
                        seq_len,
                        num_value_heads,
                        value_dim,
                    );
                    let delta = v_raw
                        .iter()
                        .zip(kv_mem.iter())
                        .map(|(&v_value, &kv_value)| (v_value - kv_value) * beta)
                        .collect::<Vec<_>>();

                    let gate_row = row4(
                        &z_tensor.data,
                        batch_idx,
                        seq_idx,
                        value_head,
                        seq_len,
                        num_value_heads,
                        value_dim,
                    );
                    let upstream_row = row4(
                        &upstream.data,
                        batch_idx,
                        seq_idx,
                        value_head,
                        seq_len,
                        num_value_heads,
                        value_dim,
                    );
                    let core_out = mat_t_vec(&state, &q.values, key_dim, value_dim);
                    let (normed, inv_rms) = rmsnorm_row(&core_out, &norm_tensor.data, eps);
                    let gate_silu = silu_slice(gate_row);

                    let mut dcore = vec![0.0_f32; value_dim];
                    let mut dot_beta = 0.0_f32;
                    let param_started = subop_started(&profile);
                    for value_idx in 0..value_dim {
                        dcore[value_idx] = upstream_row[value_idx] * gate_silu[value_idx];
                        let gate_grad = upstream_row[value_idx] * normed[value_idx];
                        dz[idx4(
                            batch_idx,
                            seq_idx,
                            value_head,
                            value_idx,
                            seq_len,
                            num_value_heads,
                            value_dim,
                        )] += gate_grad * silu_grad_scalar(gate_row[value_idx]);
                        dot_beta +=
                            dcore[value_idx] * core_out[value_idx] * norm_tensor.data[value_idx];
                        dnorm[value_idx] += dcore[value_idx] * core_out[value_idx] * inv_rms;
                    }
                    nested_param_grad_duration += elapsed_subop(param_started);
                    dcore = rmsnorm_backward_row(
                        &core_out,
                        &norm_tensor.data,
                        &dcore,
                        inv_rms,
                        dot_beta,
                        value_dim,
                    );

                    let dq = mat_vec(&state, &dcore, key_dim, value_dim);
                    add_outer_in_place(&mut grad_state, &q.values, &dcore, key_dim, value_dim);

                    let mut s_decay = state.clone();
                    subtract_outer_in_place(&mut s_decay, &k.values, &delta, key_dim, value_dim);

                    let mut d_delta = vec![0.0_f32; value_dim];
                    let mut dk = vec![0.0_f32; key_dim];
                    for key_idx in 0..key_dim {
                        let mut accum = 0.0_f32;
                        for value_idx in 0..value_dim {
                            let grad_value = grad_state[key_idx * value_dim + value_idx];
                            accum += grad_value * delta[value_idx];
                            d_delta[value_idx] += grad_value * k.values[key_idx];
                        }
                        dk[key_idx] += accum;
                    }

                    let mut dkv_mem = vec![0.0_f32; value_dim];
                    let v_minus_kv = v_raw
                        .iter()
                        .zip(kv_mem.iter())
                        .map(|(&v_value, &kv_value)| v_value - kv_value)
                        .collect::<Vec<_>>();
                    let mut dbeta_scalar = 0.0_f32;
                    for value_idx in 0..value_dim {
                        dbeta_scalar += d_delta[value_idx] * v_minus_kv[value_idx];
                        dkv_mem[value_idx] -= d_delta[value_idx] * beta;
                    }

                    for key_idx in 0..key_dim {
                        let mut accum = 0.0_f32;
                        for value_idx in 0..value_dim {
                            grad_state[key_idx * value_dim + value_idx] +=
                                k.values[key_idx] * dkv_mem[value_idx];
                            accum += s_decay[key_idx * value_dim + value_idx] * dkv_mem[value_idx];
                        }
                        dk[key_idx] += accum;
                    }

                    let prev_state = if seq_idx == 0 {
                        // Pre-decay state entering position 0. On the carry path this is
                        // the seeded per-head `initial_state` (matching the forward seed at
                        // ~1787), so `dexp_g` includes `Σ initial_state · dL/dS'_0`; without
                        // it the a_log/dt_bias decay-gate grads are biased. Non-carry: zeros.
                        match carry.initial_state {
                            Some(initial_state) => {
                                let base = state_base(
                                    batch_idx,
                                    value_head,
                                    num_value_heads,
                                    key_dim,
                                    value_dim,
                                );
                                initial_state[base..base + key_dim * value_dim].to_vec()
                            }
                            None => vec![0.0_f32; key_dim * value_dim],
                        }
                    } else {
                        let prev_base = state_time_base(
                            batch_idx,
                            seq_idx - 1,
                            value_head,
                            seq_len,
                            num_value_heads,
                            key_dim,
                            value_dim,
                        );
                        forward.state_history[prev_base..prev_base + key_dim * value_dim].to_vec()
                    };
                    let mut dstate_prev = vec![0.0_f32; key_dim * value_dim];
                    let mut dexp_g = 0.0_f32;
                    for idx in 0..key_dim * value_dim {
                        dexp_g += prev_state[idx] * grad_state[idx];
                        dstate_prev[idx] = grad_state[idx] * exp_g;
                    }

                    let dg = dexp_g * exp_g;
                    let softplus_grad = sigmoid_scalar(softplus_input);
                    let param_started = subop_started(&profile);
                    da[idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)] +=
                        dg * (-exp_a * softplus_grad);
                    ddt[value_head] += dg * (-exp_a * softplus_grad);
                    da_log[value_head] += dg * (-exp_a * softplus_value);
                    db[idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)] +=
                        dbeta_scalar * beta * (1.0 - beta);
                    nested_param_grad_duration += elapsed_subop(param_started);

                    let dq_raw = l2_normalize_scaled_backward(
                        &q_raw,
                        &dq,
                        q.norm,
                        1.0 / (key_dim as f32).sqrt(),
                    );
                    let dk_raw = l2_normalize_scaled_backward(&k_raw, &dk, k.norm, 1.0);
                    let dv_raw = d_delta
                        .iter()
                        .map(|&d_value| d_value * beta)
                        .collect::<Vec<_>>();

                    let param_started = subop_started(&profile);
                    for key_idx in 0..key_dim {
                        dqkv[idx3(
                            batch_idx,
                            seq_idx,
                            key_head * key_dim + key_idx,
                            seq_len,
                            qkv_tensor.shape[2],
                        )] += dq_raw[key_idx];
                        dqkv[idx3(
                            batch_idx,
                            seq_idx,
                            q_dim + key_head * key_dim + key_idx,
                            seq_len,
                            qkv_tensor.shape[2],
                        )] += dk_raw[key_idx];
                    }
                    for value_idx in 0..value_dim {
                        dqkv[idx3(
                            batch_idx,
                            seq_idx,
                            v_offset + value_head * value_dim + value_idx,
                            seq_len,
                            qkv_tensor.shape[2],
                        )] += dv_raw[value_idx];
                    }
                    nested_param_grad_duration += elapsed_subop(param_started);

                    state = prev_state;
                    grad_state = dstate_prev;
                }
            }
        }
    }
    if let Some(started) = scan_started {
        let scan_duration = started.elapsed();
        record_subop(
            &mut profile,
            "scan_state_history",
            scan_duration.saturating_sub(nested_param_grad_duration),
        );
    }
    record_subop(&mut profile, "param_grad_accum", nested_param_grad_duration);

    let param_started = subop_started(&profile);
    let (dqkv, dconv) = conv1d_backward(
        &dqkv,
        &forward.preact,
        &qkv_tensor.data,
        &conv_tensor.data,
        &conv_tensor.shape,
        params,
        carry.initial_conv_window,
    )?;
    record_elapsed_subop(&mut profile, "param_grad_accum", param_started);

    let grad_pack_started = subop_started(&profile);
    let mut grads = GradPairs::new();
    if qkv_tensor.requires_grad {
        grads.push((
            qkv,
            store.alloc(Tensor::new(dqkv, qkv_tensor.shape.clone(), false)?),
        ));
    }
    if z_tensor.requires_grad {
        grads.push((
            z,
            store.alloc(Tensor::new(dz, z_tensor.shape.clone(), false)?),
        ));
    }
    if b_tensor.requires_grad {
        grads.push((
            b_proj,
            store.alloc(Tensor::new(db, b_tensor.shape.clone(), false)?),
        ));
    }
    if a_tensor.requires_grad {
        grads.push((
            a_proj,
            store.alloc(Tensor::new(da, a_tensor.shape.clone(), false)?),
        ));
    }
    if conv_tensor.requires_grad {
        grads.push((
            conv1d_weight,
            store.alloc(Tensor::new(dconv, conv_tensor.shape.clone(), false)?),
        ));
    }
    if dt_tensor.requires_grad {
        grads.push((
            dt_bias,
            store.alloc(Tensor::new(ddt, dt_tensor.shape.clone(), false)?),
        ));
    }
    if a_log_tensor.requires_grad {
        grads.push((
            a_log,
            store.alloc(Tensor::new(da_log, a_log_tensor.shape.clone(), false)?),
        ));
    }
    if norm_tensor.requires_grad {
        grads.push((
            norm_weight,
            store.alloc(Tensor::new(dnorm, norm_tensor.shape.clone(), false)?),
        ));
    }
    record_elapsed_subop(&mut profile, "grad_pack", grad_pack_started);
    if let Some(profile) = profile {
        log_linear_attention_backward_profile(&profile);
    }
    Ok(grads)
}

fn validate_shapes(
    qkv: TensorId,
    z: TensorId,
    b_proj: TensorId,
    a_proj: TensorId,
    conv1d_weight: TensorId,
    dt_bias: TensorId,
    a_log: TensorId,
    norm_weight: TensorId,
    params: LinearAttentionParams,
    store: &TensorStore,
) -> Result<()> {
    validate_boundary_shapes(
        qkv,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        params,
        store,
    )?;
    for (tensor, expected) in [
        (
            z,
            vec![
                params.batch,
                params.seq_len,
                params.num_value_heads * params.value_dim,
            ],
        ),
        (norm_weight, vec![params.value_dim]),
    ] {
        let shape = &store.tensor(tensor)?.shape;
        if *shape != expected {
            return Err(AutogradError::ShapeMismatch {
                expected,
                got: shape.clone(),
            });
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_boundary_shapes(
    qkv: TensorId,
    b_proj: TensorId,
    a_proj: TensorId,
    conv1d_weight: TensorId,
    dt_bias: TensorId,
    a_log: TensorId,
    params: LinearAttentionParams,
    store: &TensorStore,
) -> Result<()> {
    if params.conv_kernel == 0 {
        return Err(AutogradError::TapeInvariant(
            "linear attention conv kernel is zero",
        ));
    }
    let q_dim = params.num_key_heads * params.key_dim;
    let qkv_dim = q_dim * 2 + params.num_value_heads * params.value_dim;
    let expected_rank3 = |tensor: TensorId, dim: usize| -> Result<()> {
        let shape = &store.tensor(tensor)?.shape;
        if shape != &vec![params.batch, params.seq_len, dim] {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![params.batch, params.seq_len, dim],
                got: shape.clone(),
            });
        }
        Ok(())
    };
    expected_rank3(qkv, qkv_dim)?;
    expected_rank3(b_proj, params.num_value_heads)?;
    expected_rank3(a_proj, params.num_value_heads)?;

    let conv_shape = &store.tensor(conv1d_weight)?.shape;
    let conv_ok = matches!(
        conv_shape.as_slice(),
        [channels, kernel] if *channels == qkv_dim && *kernel == params.conv_kernel
    ) || matches!(
        conv_shape.as_slice(),
        [channels, kernel, one] if *channels == qkv_dim && *kernel == params.conv_kernel && *one == 1
    );
    if !conv_ok {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![qkv_dim, params.conv_kernel],
            got: conv_shape.clone(),
        });
    }

    for tensor in [dt_bias, a_log] {
        let expected = vec![params.num_value_heads];
        let shape = &store.tensor(tensor)?.shape;
        if *shape != expected {
            return Err(AutogradError::ShapeMismatch {
                expected,
                got: shape.clone(),
            });
        }
    }
    Ok(())
}

fn linear_attention_forward(
    qkv: &[f32],
    z: &[f32],
    b_proj: &[f32],
    a_proj: &[f32],
    conv1d_weight: &[f32],
    conv1d_shape: &[usize],
    dt_bias: &[f32],
    a_log: &[f32],
    norm_weight: &[f32],
    params: LinearAttentionParams,
    carry: LinearAttentionCarry<'_>,
) -> LinearAttentionForward {
    let q_dim = params.num_key_heads * params.key_dim;
    let k_dim = q_dim;
    let v_dim = params.num_value_heads * params.value_dim;
    let qkv_dim = q_dim + k_dim + v_dim;
    let z_dim = params.num_value_heads * params.value_dim;
    let mut output = vec![0.0_f32; params.batch * params.seq_len * z_dim];
    let mut preact = vec![0.0_f32; params.batch * params.seq_len * qkv_dim];
    let mut beta = vec![0.0_f32; params.batch * params.seq_len * params.num_value_heads];
    let mut exp_g = vec![0.0_f32; params.batch * params.seq_len * params.num_value_heads];
    let mut kv_mem = vec![0.0_f32; params.batch * params.seq_len * z_dim];
    let mut state_history = vec![
        0.0_f32;
        params.batch
            * params.seq_len
            * params.num_value_heads
            * params.key_dim
            * params.value_dim
    ];
    let mut final_state =
        vec![0.0_f32; params.batch * params.num_value_heads * params.key_dim * params.value_dim];

    // Carried causal conv window: the prior segment's last `conv_kernel - 1`
    // qkv rows, laid out `[batch, conv_kernel-1, qkv_dim]`. A relative offset of
    // `r` in `-(conv_kernel-1)..=-1` maps to window row `r + (conv_kernel - 1)`.
    let conv_window = params.conv_kernel.saturating_sub(1);
    let mut conv_tail = vec![0.0_f32; params.batch * conv_window * qkv_dim];

    let conv_window_input = |batch_idx: usize, src_rel: isize, channel: usize| -> f32 {
        if src_rel >= 0 {
            let input_idx = idx3(
                batch_idx,
                src_rel as usize,
                channel,
                params.seq_len,
                qkv_dim,
            );
            qkv[input_idx]
        } else {
            // src_rel in -(conv_kernel-1)..=-1 — read the carried window (zero if absent).
            match carry.initial_conv_window {
                Some(window) => {
                    let window_row = (src_rel + conv_window as isize) as usize;
                    let window_idx = idx3(batch_idx, window_row, channel, conv_window, qkv_dim);
                    window[window_idx]
                }
                None => 0.0,
            }
        }
    };

    for batch_idx in 0..params.batch {
        let mut state = vec![0.0_f32; params.num_value_heads * params.key_dim * params.value_dim];
        if let Some(initial_state) = carry.initial_state {
            let state_len = state.len();
            let base = batch_idx * params.num_value_heads * params.key_dim * params.value_dim;
            state.copy_from_slice(&initial_state[base..base + state_len]);
        }
        for seq_idx in 0..params.seq_len {
            for channel in 0..qkv_dim {
                let mut sum = 0.0_f32;
                for tap in 0..params.conv_kernel {
                    // src position relative to this segment's start; negative => carried window.
                    let src_rel = seq_idx as isize + tap as isize + 1 - params.conv_kernel as isize;
                    sum += conv_window_input(batch_idx, src_rel, channel)
                        * conv_weight_at(conv1d_weight, conv1d_shape, channel, tap);
                }
                preact[idx3(batch_idx, seq_idx, channel, params.seq_len, qkv_dim)] = sum;
            }

            for value_head in 0..params.num_value_heads {
                let key_head = value_head * params.num_key_heads / params.num_value_heads;
                let q_raw = (0..params.key_dim)
                    .map(|offset| {
                        preact[idx3(
                            batch_idx,
                            seq_idx,
                            key_head * params.key_dim + offset,
                            params.seq_len,
                            qkv_dim,
                        )]
                    })
                    .map(silu_scalar)
                    .collect::<Vec<_>>();
                let k_raw = (0..params.key_dim)
                    .map(|offset| {
                        preact[idx3(
                            batch_idx,
                            seq_idx,
                            q_dim + key_head * params.key_dim + offset,
                            params.seq_len,
                            qkv_dim,
                        )]
                    })
                    .map(silu_scalar)
                    .collect::<Vec<_>>();
                let v_raw = (0..params.value_dim)
                    .map(|offset| {
                        preact[idx3(
                            batch_idx,
                            seq_idx,
                            q_dim + k_dim + value_head * params.value_dim + offset,
                            params.seq_len,
                            qkv_dim,
                        )]
                    })
                    .map(silu_scalar)
                    .collect::<Vec<_>>();
                let q = l2_normalize_scaled(&q_raw, 1.0 / (params.key_dim as f32).sqrt());
                let k = l2_normalize_scaled(&k_raw, 1.0);
                let beta_value = sigmoid_scalar(
                    b_proj[idx3(
                        batch_idx,
                        seq_idx,
                        value_head,
                        params.seq_len,
                        params.num_value_heads,
                    )],
                );
                beta[idx3(
                    batch_idx,
                    seq_idx,
                    value_head,
                    params.seq_len,
                    params.num_value_heads,
                )] = beta_value;
                let g = -a_log[value_head].exp()
                    * softplus_scalar(
                        a_proj[idx3(
                            batch_idx,
                            seq_idx,
                            value_head,
                            params.seq_len,
                            params.num_value_heads,
                        )] + dt_bias[value_head],
                    );
                let exp_g_value = g.exp();
                exp_g[idx3(
                    batch_idx,
                    seq_idx,
                    value_head,
                    params.seq_len,
                    params.num_value_heads,
                )] = exp_g_value;

                let base = state_head_base(value_head, params.key_dim, params.value_dim);
                for key_idx in 0..params.key_dim {
                    for value_idx in 0..params.value_dim {
                        state[base + key_idx * params.value_dim + value_idx] *= exp_g_value;
                    }
                }

                let mut kv_row = vec![0.0_f32; params.value_dim];
                for value_idx in 0..params.value_dim {
                    let mut accum = 0.0_f32;
                    for key_idx in 0..params.key_dim {
                        accum += state[base + key_idx * params.value_dim + value_idx]
                            * k.values[key_idx];
                    }
                    kv_row[value_idx] = accum;
                    kv_mem[idx4(
                        batch_idx,
                        seq_idx,
                        value_head,
                        value_idx,
                        params.seq_len,
                        params.num_value_heads,
                        params.value_dim,
                    )] = accum;
                }

                let mut core_out = vec![0.0_f32; params.value_dim];
                for value_idx in 0..params.value_dim {
                    let delta = (v_raw[value_idx] - kv_row[value_idx]) * beta_value;
                    for key_idx in 0..params.key_dim {
                        state[base + key_idx * params.value_dim + value_idx] +=
                            delta * k.values[key_idx];
                    }
                    let mut accum = 0.0_f32;
                    for key_idx in 0..params.key_dim {
                        accum += state[base + key_idx * params.value_dim + value_idx]
                            * q.values[key_idx];
                    }
                    core_out[value_idx] = accum;
                }

                let history_base = state_time_base(
                    batch_idx,
                    seq_idx,
                    value_head,
                    params.seq_len,
                    params.num_value_heads,
                    params.key_dim,
                    params.value_dim,
                );
                state_history[history_base..history_base + params.key_dim * params.value_dim]
                    .copy_from_slice(&state[base..base + params.key_dim * params.value_dim]);

                let (normed, _) = rmsnorm_row(&core_out, norm_weight, params.eps);
                for value_idx in 0..params.value_dim {
                    let gate = silu_scalar(
                        z[idx4(
                            batch_idx,
                            seq_idx,
                            value_head,
                            value_idx,
                            params.seq_len,
                            params.num_value_heads,
                            params.value_dim,
                        )],
                    );
                    output[idx4(
                        batch_idx,
                        seq_idx,
                        value_head,
                        value_idx,
                        params.seq_len,
                        params.num_value_heads,
                        params.value_dim,
                    )] = normed[value_idx] * gate;
                }
            }
        }

        let final_base = batch_idx * params.num_value_heads * params.key_dim * params.value_dim;
        final_state[final_base..final_base + state.len()].copy_from_slice(&state);

        // Capture this segment's conv tail: the last `conv_window` qkv input rows.
        // A follow-on segment seeds these at relative offsets `-(conv_window)..=-1`.
        // Short segment (seq_len < conv_window): a prior carried window is the
        // correct source for the missing earliest rows — chain it through.
        for window_row in 0..conv_window {
            // window_row r holds the row at global relative offset `r - conv_window`
            // measured from THIS segment's end (== next segment's start).
            let src_rel = params.seq_len as isize + window_row as isize - conv_window as isize;
            for channel in 0..qkv_dim {
                conv_tail[idx3(batch_idx, window_row, channel, conv_window, qkv_dim)] =
                    conv_window_input(batch_idx, src_rel, channel);
            }
        }
    }

    LinearAttentionForward {
        output,
        preact,
        beta,
        exp_g,
        kv_mem,
        state_history,
        final_state,
        conv_tail,
    }
}

fn conv1d_backward(
    grad_out: &[f32],
    preact: &[f32],
    input: &[f32],
    conv1d_weight: &[f32],
    conv1d_shape: &[usize],
    params: LinearAttentionParams,
    initial_conv_window: Option<&[f32]>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let q_dim = params.num_key_heads * params.key_dim;
    let qkv_dim = q_dim * 2 + params.num_value_heads * params.value_dim;
    if input.len() != params.batch * params.seq_len * qkv_dim {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![params.batch, params.seq_len, qkv_dim],
            got: vec![input.len()],
        });
    }

    // Carried causal-conv window `[batch, conv_kernel-1, qkv_dim]`, seeded by the
    // forward at negative `src_rel` (matching `conv_window_input` at ~1762).
    let conv_window = params.conv_kernel.saturating_sub(1);
    let mut grad_input = vec![0.0_f32; input.len()];
    let mut grad_weight = vec![0.0_f32; conv1d_weight.len()];
    for batch_idx in 0..params.batch {
        for seq_idx in 0..params.seq_len {
            for channel in 0..qkv_dim {
                let preact_idx = idx3(batch_idx, seq_idx, channel, params.seq_len, qkv_dim);
                let dpre = grad_out[preact_idx] * silu_grad_scalar(preact[preact_idx]);
                for tap in 0..params.conv_kernel {
                    // src relative to this segment's start; negative => carried window.
                    let src_rel = seq_idx as isize + tap as isize + 1 - params.conv_kernel as isize;
                    // The weight grad accumulates the tap's input value regardless of
                    // source; grad flows back INTO `grad_input` only for this-segment
                    // taps — the carried window is a frozen constant. Non-carry path:
                    // window absent => tap_input = 0.0, identical to the old skip.
                    let tap_input = if src_rel >= 0 {
                        let input_idx = idx3(
                            batch_idx,
                            src_rel as usize,
                            channel,
                            params.seq_len,
                            qkv_dim,
                        );
                        grad_input[input_idx] +=
                            dpre * conv_weight_at(conv1d_weight, conv1d_shape, channel, tap);
                        input[input_idx]
                    } else {
                        initial_conv_window.map_or(0.0, |window| {
                            let window_row = (src_rel + conv_window as isize) as usize;
                            window[idx3(batch_idx, window_row, channel, conv_window, qkv_dim)]
                        })
                    };
                    grad_weight[conv_weight_index(conv1d_shape, channel, tap)] += dpre * tap_input;
                }
            }
        }
    }
    Ok((grad_input, grad_weight))
}

struct NormalizedVec {
    values: Vec<f32>,
    norm: f32,
}

fn l2_normalize_scaled(input: &[f32], scale: f32) -> NormalizedVec {
    let norm = (input.iter().map(|value| value * value).sum::<f32>() + 1.0e-12_f32).sqrt();
    let values = input.iter().map(|value| scale * value / norm).collect();
    NormalizedVec { values, norm }
}

fn l2_normalize_scaled_backward(input: &[f32], grad: &[f32], norm: f32, scale: f32) -> Vec<f32> {
    let dot = input
        .iter()
        .zip(grad.iter())
        .map(|(&x, &g)| x * g)
        .sum::<f32>();
    let norm_cubed = norm * norm * norm;
    input
        .iter()
        .zip(grad.iter())
        .map(|(&x, &g)| scale * (g / norm - x * dot / norm_cubed))
        .collect()
}

fn rmsnorm_row(input: &[f32], weight: &[f32], eps: f32) -> (Vec<f32>, f32) {
    let inv_rms = 1.0
        / ((input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32) + eps)
            .sqrt();
    let output = input
        .iter()
        .zip(weight.iter())
        .map(|(&value, &w)| value * inv_rms * w)
        .collect();
    (output, inv_rms)
}

fn rmsnorm_backward_row(
    input: &[f32],
    weight: &[f32],
    grad: &[f32],
    inv_rms: f32,
    dot: f32,
    hidden: usize,
) -> Vec<f32> {
    let coeff = inv_rms * inv_rms * inv_rms / hidden as f32;
    input
        .iter()
        .zip(weight.iter())
        .zip(grad.iter())
        .map(|((&x, &w), &g)| g * w * inv_rms - x * coeff * dot)
        .collect()
}

fn softplus_scalar(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu_scalar(x: f32) -> f32 {
    x * sigmoid_scalar(x)
}

fn silu_grad_scalar(x: f32) -> f32 {
    let sig = sigmoid_scalar(x);
    sig * (1.0 + x * (1.0 - sig))
}

fn silu_slice(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&value| silu_scalar(value)).collect()
}

fn mat_t_vec(matrix: &[f32], vector: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; cols];
    for row in 0..rows {
        let scalar = vector[row];
        for col in 0..cols {
            output[col] += matrix[row * cols + col] * scalar;
        }
    }
    output
}

fn mat_vec(matrix: &[f32], vector: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; rows];
    for row in 0..rows {
        let mut accum = 0.0_f32;
        for col in 0..cols {
            accum += matrix[row * cols + col] * vector[col];
        }
        output[row] = accum;
    }
    output
}

fn add_outer_in_place(matrix: &mut [f32], left: &[f32], right: &[f32], rows: usize, cols: usize) {
    for row in 0..rows {
        for col in 0..cols {
            matrix[row * cols + col] += left[row] * right[col];
        }
    }
}

fn subtract_outer_in_place(
    matrix: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    cols: usize,
) {
    for row in 0..rows {
        for col in 0..cols {
            matrix[row * cols + col] -= left[row] * right[col];
        }
    }
}

fn conv_weight_at(conv1d_weight: &[f32], shape: &[usize], channel: usize, tap: usize) -> f32 {
    conv1d_weight[conv_weight_index(shape, channel, tap)]
}

fn conv_weight_index(shape: &[usize], channel: usize, tap: usize) -> usize {
    match shape {
        [_, kernel] => channel * kernel + tap,
        [_, kernel, one] if *one == 1 => channel * kernel * one + tap * one,
        _ => unreachable!("validated by shape check"),
    }
}

fn idx3(batch: usize, seq: usize, dim: usize, seq_len: usize, width: usize) -> usize {
    (batch * seq_len + seq) * width + dim
}

fn idx4(
    batch: usize,
    seq: usize,
    head: usize,
    dim: usize,
    seq_len: usize,
    heads: usize,
    width: usize,
) -> usize {
    (((batch * seq_len + seq) * heads + head) * width) + dim
}

fn row3(data: &[f32], batch: usize, seq: usize, seq_len: usize, width: usize) -> &[f32] {
    let base = idx3(batch, seq, 0, seq_len, width);
    &data[base..base + width]
}

fn row4(
    data: &[f32],
    batch: usize,
    seq: usize,
    head: usize,
    seq_len: usize,
    heads: usize,
    width: usize,
) -> &[f32] {
    let base = idx4(batch, seq, head, 0, seq_len, heads, width);
    &data[base..base + width]
}

fn state_base(batch: usize, head: usize, heads: usize, key_dim: usize, value_dim: usize) -> usize {
    ((batch * heads + head) * key_dim) * value_dim
}

fn state_head_base(head: usize, key_dim: usize, value_dim: usize) -> usize {
    head * key_dim * value_dim
}

fn state_time_base(
    batch: usize,
    seq: usize,
    head: usize,
    seq_len: usize,
    heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> usize {
    (((batch * seq_len + seq) * heads + head) * key_dim) * value_dim
}

#[cfg(test)]
mod cp_reorder_tests {
    use super::zigzag_block_perms;

    // Independent oracle: rebuild the a2a physical block layout straight from the
    // zigzag rule (rank r owns global chunks r, 2N-1-r; a2a lays ranks in order),
    // then check `fwd` un-interleaves it to 0..2N and `phys` is its inverse.
    fn a2a_physical_to_global(n: usize) -> Vec<usize> {
        let mut phys = Vec::new();
        for r in 0..n {
            phys.push(r);
            phys.push(2 * n - 1 - r);
        }
        phys
    }

    #[test]
    fn fwd_perm_restores_global_order_and_phys_inverts_it() {
        for n in [1usize, 2, 4] {
            let (fwd, phys) = zigzag_block_perms(n);
            let physical = a2a_physical_to_global(n);
            // Applying fwd to the physical layout yields ascending global chunks.
            let restored: Vec<usize> = fwd.iter().map(|&p| physical[p]).collect();
            assert_eq!(restored, (0..2 * n).collect::<Vec<_>>(), "fwd n={n}");
            // phys ∘ fwd = identity.
            for (p, &g) in fwd.iter().enumerate() {
                assert_eq!(phys[g], p, "phys not inverse of fwd at n={n}");
            }
        }
    }
}
