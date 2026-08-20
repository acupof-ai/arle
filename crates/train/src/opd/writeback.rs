//! Masked-target writeback steps (CE, batched CE, PG, GKD) over a scored trajectory.

use std::{collections::HashSet, time::Instant};

use autograd::{
    AutogradError, Tape, TensorId, TensorStore,
    ops::{
        add, embedding,
        fused_linear_distill::{
            PgStats, WeightForm, fused_linear_ce_loss_indexed, fused_linear_pg_loss_indexed,
        },
        gather_last_dim, log_softmax, matmul_bt, mul_scalar, reshape, slice,
    },
    optim::Optimizer,
};

use crate::{
    grad_clip::finite_optimizer_step,
    loss::{DEFAULT_KL_CHUNK_SIZE, KlDirection, kl_distill_loss_chunked},
    qwen35::Qwen35Model,
    teacher_infer::TeacherForward,
    trainer::cleanup_after_backward,
};

use super::{
    OpdError, Result, backward_with_optional_profile,
    loss::build_masked_loss_targets,
    map_qwen35_forward_error,
    validation::{validate_loss_value, validate_token_ids},
    windowing::masked_gkd_windows,
};

/// The 27B CE is overhead-bound (host op-dispatch, GPU ~0% util), so batching
/// amortizes the ~64-layer per-op host dispatch over B examples — near-B×
/// throughput.
///
/// The forward yields post-final-norm hidden (not logits); a per-row fused chunked
/// CE projects only the masked completion positions through `lm_head`, so the dense
/// `[B, seq, vocab]` logits tile is never materialized (was ~24 GB at B=4 / 10K-tok
/// completions). Grad-checkpoints offload to host past the length that needs it
/// (`writeback_offload_for_seq`), mirroring the single-trajectory writeback.
/// `chunk_rows` bounds the transient `[chunk, vocab]` tile — same knob as the
/// single-trajectory path's `window_size` (`--writeback-window`).
///
/// Not a superset of [`masked_writeback_step`] at B=1: every completion token is a
/// target (no `response_mask`), the reduction is the mean of per-row token-means
/// (not one global token-mean), and neither frozen-prompt-KV nor CP/DP applies.
#[allow(clippy::too_many_arguments)]
pub fn full_batch_ce_writeback_step<O: Optimizer>(
    student: &Qwen35Model,
    all_model_params: &[TensorId],
    trainable_params: &[TensorId],
    optimizer: &mut O,
    batch: &[(Vec<u32>, Vec<u32>)],
    vocab: usize,
    chunk_rows: usize,
    store: &mut TensorStore,
) -> Result<f32> {
    if batch.is_empty() {
        return Err(OpdError::InvalidInput(
            "batched writeback requires a non-empty batch".to_owned(),
        ));
    }
    let mut max_len = 0usize;
    for (i, (prompt, completion)) in batch.iter().enumerate() {
        if prompt.is_empty() || completion.is_empty() {
            return Err(OpdError::InvalidInput(
                "batched writeback: empty prompt or completion in batch".to_owned(),
            ));
        }
        validate_token_ids(
            &format!("batched writeback row {i} completion"),
            completion,
            vocab,
        )?;
        max_len = max_len.max(prompt.len() + completion.len());
    }
    let b = batch.len();

    let mut tape = Tape::new();
    tape.set_offload_checkpoints(crate::runtime_flags::writeback_offload_for_seq(b * max_len));

    // Flat [B * max_len] input, each row = prompt ++ completion ++ pad(0). Causal
    // attention + right-padding means padding never affects earlier positions, and
    // the per-row CE only targets completion positions, so padding is inert.
    let flat: Vec<usize> = batch
        .iter()
        .flat_map(|(prompt, completion)| {
            prompt
                .iter()
                .map(|&t| t as usize)
                .chain(completion.iter().map(|&t| t as usize))
                .chain(std::iter::repeat_n(
                    0usize,
                    max_len - prompt.len() - completion.len(),
                ))
        })
        .collect();
    log_writeback_vram(store, "batched", "pre forward");
    let hidden = student
        .forward_batch_hidden(&flat, b, max_len, store, &mut tape)
        .map_err(OpdError::from)?;
    log_writeback_vram(store, "batched", "post forward");

    // Per-row completion-masked CE via the fused indexed path, preserving the prior
    // mean-of-row-means reduction: each row's fused call returns that row's token-
    // mean CE (fused scales by 1/row_targets), summed and scaled by 1/b below. Row
    // i occupies flat hidden rows [i*max_len, (i+1)*max_len); predicting position
    // p = prompt.len()-1+j targets completion[j] (next-token convention, identical
    // to `next_token_sft_loss_from_logits`).
    let mut total: Option<TensorId> = None;
    for (i, (prompt, completion)) in batch.iter().enumerate() {
        let row_base = i * max_len + (prompt.len() - 1);
        let position_indices: Vec<i32> = (row_base..row_base + completion.len())
            .map(|p| p as i32)
            .collect();
        let target_tokens: Vec<i32> = completion.iter().map(|&t| t as i32).collect();
        let ce = fused_linear_ce_loss_indexed(
            hidden,
            student.lm_head_weight_id(),
            &position_indices,
            &target_tokens,
            chunk_rows,
            None,
            store,
            &mut tape,
        )
        .map_err(OpdError::from)?;
        total = Some(match total {
            None => ce,
            Some(prev) => add(prev, ce, store, &mut tape).map_err(OpdError::from)?,
        });
    }
    let total = total.expect("batch is non-empty");
    let mean = mul_scalar(total, 1.0 / b as f32, store, &mut tape).map_err(OpdError::from)?;
    let loss_value = store.to_host(mean).map_err(OpdError::from)?[0];
    validate_loss_value(loss_value)?;
    log_writeback_vram(store, "batched", "post ce");
    backward_with_optional_profile(mean, loss_value, store, &mut tape)?;
    log_writeback_vram(store, "batched", "post backward");
    finite_optimizer_step(loss_value, trainable_params, 0.0, optimizer, store)?;
    let keep_extra: HashSet<TensorId> = HashSet::new();
    cleanup_after_backward(store, &mut tape, all_model_params, &keep_extra);
    log_writeback_vram(store, "batched", "post cleanup");
    Ok(loss_value)
}

#[allow(clippy::too_many_arguments)]
#[derive(Clone, Copy)]
struct VramSample {
    free: usize,
    total: usize,
    /// Mempool hoard in MiB; `None` = probe unavailable, not a measured 0.
    hoarded_mib: Option<u64>,
}

impl VramSample {
    fn used(self) -> usize {
        self.total.saturating_sub(self.free)
    }

    fn used_mib(self) -> usize {
        self.used() >> 20
    }

    fn free_mib(self) -> usize {
        self.free >> 20
    }

    fn total_mib(self) -> usize {
        self.total >> 20
    }
}

fn writeback_vram_trace_enabled() -> bool {
    std::env::var("ARLE_OPD_VRAM_TRACE").is_ok()
}

/// `n/a` keeps an unavailable probe distinct from a measured `0MiB`.
pub fn fmt_hoarded(mib: Option<u64>) -> String {
    mib.map_or_else(|| "n/a".to_string(), |m| format!("{m}MiB"))
}

fn log_writeback_vram(store: &TensorStore, scope: &str, label: &str) -> Option<VramSample> {
    if !writeback_vram_trace_enabled() {
        return None;
    }
    match store.backend().device_mem_info() {
        Some((free, total)) => {
            let sample = VramSample {
                free,
                total,
                hoarded_mib: store.backend().hoarded_mib(),
            };
            eprintln!(
                "[opd-vram] {scope} {label}: used={}MiB free={}MiB total={}MiB hoarded={}",
                sample.used_mib(),
                sample.free_mib(),
                sample.total_mib(),
                fmt_hoarded(sample.hoarded_mib),
            );
            Some(sample)
        }
        None => {
            eprintln!("[opd-vram] {scope} {label}: device_mem_info unavailable");
            None
        }
    }
}

fn log_writeback_vram_ledger(
    scope: &str,
    base: Option<VramSample>,
    post_forward: Option<VramSample>,
    post_backward: Option<VramSample>,
    post_cleanup: Option<VramSample>,
) {
    if !writeback_vram_trace_enabled() {
        return;
    }
    let used = |sample: Option<VramSample>| sample.map(|s| s.used_mib()).unwrap_or(0);
    let hoarded = |sample: Option<VramSample>| fmt_hoarded(sample.and_then(|s| s.hoarded_mib));
    let allocator_delta = match (base, post_cleanup) {
        (Some(base), Some(cleanup)) => cleanup.used().saturating_sub(base.used()) >> 20,
        _ => 0,
    };
    eprintln!(
        "[opd-vram-ledger] {scope} base_used_mib={} post_forward_used_mib={} \
         post_backward_used_mib={} post_cleanup_used_mib={} allocator_retained_delta_mib={} \
         hoarded_fwd/bwd/clean_mib={}/{}/{}",
        used(base),
        used(post_forward),
        used(post_backward),
        used(post_cleanup),
        allocator_delta,
        hoarded(post_forward),
        hoarded(post_backward),
        hoarded(post_cleanup),
    );
}

pub enum WritebackLoss<'a> {
    Ce,
    Pg {
        rollout_logprobs: &'a [f32],
        weight: &'a [f32],
        form: WeightForm,
        kl_coef: f32,
    },
}

pub(super) struct GenSegment {
    pub(super) prompt_prefix: Vec<u32>,
    pub(super) gen_ids: Vec<u32>,
    pub(super) prompt_positions: Vec<u32>,
    pub(super) gen_positions: Vec<u32>,
}

impl GenSegment {
    pub(super) fn split(full: &[u32], prompt_len: usize) -> Self {
        let gen_start = prompt_len - 1;
        let seq_len = full.len();
        Self {
            prompt_prefix: full[0..gen_start].to_vec(),
            gen_ids: full[gen_start..].to_vec(),
            prompt_positions: (0..gen_start as u32).collect(),
            gen_positions: (gen_start as u32..seq_len as u32).collect(),
        }
    }
}

/// The third return is the pre-step global grad L2 norm (post CP/DP all-reduce,
/// pre-clip) — `None` when `step_optimizer` is false, since accumulated grads are
/// not yet reduced. Free: `finite_optimizer_step` already computes it.
#[allow(clippy::too_many_arguments)]
pub fn masked_writeback_step<O: Optimizer>(
    loss_kind: WritebackLoss,
    student: &Qwen35Model,
    all_model_params: &[TensorId],
    trainable_params: &[TensorId],
    optimizer: &mut O,
    // False accumulates gradients for a later optimizer step.
    step_optimizer: bool,
    prompt_ids: &[u32],
    response_ids: &[u32],
    response_mask: &[u8],
    vocab: usize,
    window_size: usize,
    cp: crate::context_parallel::CpContext,
    dp: crate::context_parallel::DpContext,
    store: &mut TensorStore,
) -> Result<(f32, PgStats, Option<f64>)> {
    if prompt_ids.is_empty() {
        return Err(OpdError::InvalidInput(
            "masked writeback requires a non-empty prompt".to_owned(),
        ));
    }
    if window_size == 0 {
        return Err(OpdError::InvalidInput(
            "masked writeback window_size must be > 0".to_owned(),
        ));
    }

    let prompt_len = prompt_ids.len();
    let mut full: Vec<u32> = prompt_ids
        .iter()
        .copied()
        .chain(response_ids.iter().copied())
        .collect();
    let mut seq_len = full.len();
    if seq_len > u32::MAX as usize {
        return Err(OpdError::InvalidInput(format!(
            "masked writeback trajectory length {seq_len} exceeds u32::MAX \
             position ids."
        )));
    }
    let mut positions: Vec<u32> = (0..seq_len as u32).collect();

    let loss_targets = build_masked_loss_targets(&full, prompt_len, response_mask);
    if loss_targets.is_empty() {
        eprintln!(
            "[masked-writeback] no LLM-generated targets (prompt_len={prompt_len}, \
             response_len={}, mask_ones={}); skipping (nothing to train)",
            response_ids.len(),
            response_mask.iter().filter(|&&m| m == 1).count(),
        );
        return Ok((0.0, PgStats::default(), None));
    }
    let total_targets = loss_targets.len();
    let chunk_rows = window_size; // reused: positions per fused-CE chunk.

    // inv_n override under CP or DP (single-card keeps the fused op's local default,
    // byte-identical). CP shares ONE trajectory so total_targets is already global;
    // DP sums per-replica counts. The count reduce runs over the WORLD comm, so only
    // cp rank 0 contributes — every cp rank carries the same replica-global count and
    // would over-count by cp_size (G3: losses came back exactly /world).
    let inv_n_override = if dp.is_enabled() {
        let contribution = if cp.rank == 0 { total_targets } else { 0 };
        let global =
            crate::grad_clip::dp_group_sum_count(contribution, store).map_err(OpdError::from)?;
        crate::context_parallel::global_inv_n(global)
    } else if cp.is_enabled() {
        Some(1.0_f32 / total_targets as f32)
    } else {
        None
    };

    // Frozen-prompt-KV: forward only the gen segment, rebasing masked positions
    // into it. gen_start=0 keeps positions absolute for the byte-identical full
    // path. Every masked target p >= prompt_len-1 = gen_start, so p-gen_start>=0.
    let frozen = crate::runtime_flags::writeback_frozen_prompt_kv() && prompt_len > 1;
    let gen_start = if frozen { prompt_len - 1 } else { 0 };
    let fwd_len = seq_len - gen_start;

    // The forwarded segment (gen segment when frozen, full sequence otherwise) is
    // what gets CP-sharded. Pad it to 2*cp_size once; the prompt prefix stays
    // full-rank (off-tape, every rank runs it). Loss positions map global p ->
    // fwd-local (p - gen_start) -> fwd_shard local row.
    let fwd_padded = cp.padded_seq_len(fwd_len);
    if fwd_padded != fwd_len {
        full.resize(gen_start + fwd_padded, 0);
        positions.extend(seq_len as u32..(gen_start + fwd_padded) as u32);
        seq_len = gen_start + fwd_padded;
    }
    let fwd_shard = cp.shard(fwd_padded);

    // Vocab bound-check runs over the FULL global set on every rank (not per
    // shard): a bad token must fail all ranks together, or one rank errors while
    // the others wedge in the next collective. Clear OPD error before the autograd
    // bounds check (the lm_head's [vocab, hidden] also enforces it).
    for &(p, target) in &loss_targets {
        if target >= vocab {
            return Err(OpdError::InvalidInput(format!(
                "masked writeback target token {target} at position {p} exceeds vocab={vocab}"
            )));
        }
    }
    // Map each masked target to its local hidden row. fwd_shard covers the
    // forwarded segment (gen_start..seq_len); p >= gen_start for every target,
    // so p - gen_start is a valid fwd-local index.
    let (position_indices, target_tokens): (Vec<i32>, Vec<i32>) = loss_targets
        .iter()
        .filter_map(|&(p, target)| {
            fwd_shard
                .local_of(p - gen_start)
                .map(|local| (local as i32, target as i32))
        })
        .unzip();

    let mut tape = Tape::new();
    // Offload per-layer grad-checkpoints to host RAM only past a length that needs
    // it: the H2D re-upload serializes on the host thread and starves the GPU on
    // short trajectories, but resident checkpoints OOM the allocator on long ones.
    // The forwarded segment length is fwd_padded (gen segment when frozen, full
    // sequence otherwise).
    let offload_checkpoints = crate::runtime_flags::writeback_offload_for_seq(fwd_padded);
    tape.set_offload_checkpoints(offload_checkpoints);
    tape.set_enabled(true);
    eprintln!(
        "[masked-writeback] offload_checkpoints={offload_checkpoints} fwd_len={fwd_padded} seq_len={seq_len}"
    );
    let keep_extra: HashSet<TensorId> = HashSet::new();
    eprintln!(
        "[masked-writeback] seq_len={seq_len} total_targets={total_targets} \
         chunk_rows={chunk_rows} frozen={frozen} gen_start={gen_start}"
    );

    // ONE checkpointed forward → [1, rows, hidden] (rows = seq or gen_len). No
    // per-window re-forward of the growing prefix (was O(N²)). Frozen: only the
    // gen segment is forwarded/backwarded, seeded from the off-tape prompt KV.
    let vram_base = log_writeback_vram(store, "masked-writeback", "pre forward_hidden_states");
    let t_fwd = Instant::now();
    let hidden = if frozen {
        let seg = GenSegment::split(&full, prompt_len);
        if cp.is_enabled() {
            // Frozen+CP: shard the gen segment's ids/positions; prompt prefix is
            // full-rank (off-tape). fwd_shard.local_rows() gives the gen-local
            // rows this rank owns.
            let gen_rows = fwd_shard.local_rows();
            let shard_gen_ids: Vec<u32> = gen_rows.iter().map(|&r| seg.gen_ids[r]).collect();
            let shard_gen_pos: Vec<u32> = gen_rows.iter().map(|&r| seg.gen_positions[r]).collect();
            student
                .forward_hidden_states_gen_segment(
                    store,
                    &mut tape,
                    &seg.prompt_prefix,
                    &shard_gen_ids,
                    &seg.prompt_positions,
                    &shard_gen_pos,
                    cp,
                )
                .map_err(|err| {
                    map_qwen35_forward_error(
                        "masked writeback frozen-prompt-KV CP student hidden",
                        err,
                    )
                })?
        } else {
            student
                .forward_hidden_states_gen_segment(
                    store,
                    &mut tape,
                    &seg.prompt_prefix,
                    &seg.gen_ids,
                    &seg.prompt_positions,
                    &seg.gen_positions,
                    cp,
                )
                .map_err(|err| {
                    map_qwen35_forward_error(
                        "masked writeback frozen-prompt-KV student hidden",
                        err,
                    )
                })?
        }
    } else if cp.is_enabled() {
        // CP: embed only this rank's zigzag shard rows; attention gathers the full
        // KV. positions carry ABSOLUTE ids so RoPE cos/sin match the gathered prefix.
        // local_rows() is the gather index (two chunks, front+back, in local order).
        let rows = fwd_shard.local_rows();
        let shard_ids: Vec<u32> = rows.iter().map(|&r| full[r]).collect();
        let shard_pos: Vec<u32> = rows.iter().map(|&r| positions[r]).collect();
        student
            .forward_hidden_states(store, &mut tape, &shard_ids, &shard_pos, cp)
            .map_err(|err| map_qwen35_forward_error("masked writeback CP student hidden", err))?
    } else {
        student
            .forward_hidden_states(store, &mut tape, &full, &positions, cp)
            .map_err(|err| map_qwen35_forward_error("masked writeback student hidden", err))?
    };
    let fwd_secs = t_fwd.elapsed().as_secs_f64();
    let vram_post_forward =
        log_writeback_vram(store, "masked-writeback", "post forward_hidden_states");
    eprintln!("[masked-writeback] phase=forward_hidden_states seconds={fwd_secs:.3}");

    // Chunked fused CE: per chunk computes hidden_chunk @ lm_headᵀ → logits → CE
    // → gradient, freeing each chunk. Never materializes [seq, vocab]. The loss
    // is already the mean CE per masked token and the gradient is scaled by 1/N,
    // so backward (seed 1.0) applies the per-token-mean update directly.
    let t_ce = Instant::now();
    let (loss, pg_stats) = match loss_kind {
        WritebackLoss::Ce => {
            // A CP sequence shard can legitimately own ZERO masked targets (e.g. a
            // prompt-heavy prefix shard), which the fused CE rejects. That rank must
            // still backprop through `hidden` so the forward's all_gather_seq KV
            // gather fires its reduce_scatter adjoint — else the CP group deadlocks.
            // A zero loss that depends on hidden (sum·0) yields exactly that: value
            // 0, grad 0, collectives in lockstep, and the post-backward grad
            // all-reduce sums a genuine zero contribution from this rank.
            let loss = if cp.is_enabled() && position_indices.is_empty() {
                let s = autograd::ops::sum(hidden, store, &mut tape).map_err(OpdError::from)?;
                mul_scalar(s, 0.0, store, &mut tape).map_err(OpdError::from)?
            } else {
                fused_linear_ce_loss_indexed(
                    hidden,
                    student.lm_head_weight_id(),
                    &position_indices,
                    &target_tokens,
                    chunk_rows,
                    inv_n_override,
                    store,
                    &mut tape,
                )
                .map_err(OpdError::from)?
            };
            (loss, PgStats::default())
        }
        WritebackLoss::Pg {
            rollout_logprobs,
            weight,
            form,
            kl_coef,
        } => {
            // PG under CP/DP needs per-shard rollout_logprobs/weight + a global-count
            // inv_n in the PG op — deferred (brick scope is CE writeback). CE is the
            // agent-OPD writeback loss; PG-parallel lands with the PG inv_n_override.
            if cp.is_enabled() || dp.is_enabled() {
                return Err(OpdError::InvalidInput(
                    "parallel writeback supports the CE loss only (PG-CP/DP deferred)".to_owned(),
                ));
            }
            if rollout_logprobs.len() != total_targets || weight.len() != total_targets {
                return Err(OpdError::InvalidInput(format!(
                    "masked writeback PG rollout_logprobs/weight len {}/{} != masked targets {total_targets}",
                    rollout_logprobs.len(),
                    weight.len(),
                )));
            }
            fused_linear_pg_loss_indexed(
                hidden,
                student.lm_head_weight_id(),
                &position_indices,
                &target_tokens,
                rollout_logprobs,
                weight,
                form,
                kl_coef,
                chunk_rows,
                store,
                &mut tape,
            )
            .map_err(OpdError::from)?
        }
    };

    let loss_value = store.to_host(loss).map_err(OpdError::from)?[0];
    // Reported loss: the local numerator is only this rank's partial sum (a
    // cp rank holds its sequence shard's targets, a dp replica its own data)
    // while inv_n is already 1/global_count — so the world sum of every
    // rank's partial IS the true global mean, printed identically on every
    // rank. Measured: cp=2 shards report 4.805783/6.064485, NOT identical.
    // Grads are unaffected: they already sum to the exact global mean via
    // all_reduce_cp_grads.
    let loss_value = if cp.is_enabled() || dp.is_enabled() {
        crate::grad_clip::dp_group_sum_scalar(loss_value, store).map_err(OpdError::from)?
    } else {
        loss_value
    };
    validate_loss_value(loss_value)?;
    let ce_secs = t_ce.elapsed().as_secs_f64();
    eprintln!(
        "[masked-writeback] phase=fused_ce seconds={ce_secs:.3} targets={}",
        position_indices.len()
    );
    // The forward leaves the pool hoarding ~29 GB it cannot re-cut for
    // backward's fresh grad sizes (observed: 1.28 GB alloc OOM at 131,072 with
    // free=9 MiB while hoarded=29 GB). Release it so backward allocates clean.
    if let Err(err) = store.backend().trim_memory_pool() {
        eprintln!("trim_memory_pool before backward failed (non-fatal): {err}");
    }
    log_writeback_vram(store, "masked-writeback", "pre backward");
    let t_bwd = Instant::now();
    backward_with_optional_profile(loss, loss_value, store, &mut tape)?;
    let bwd_secs = t_bwd.elapsed().as_secs_f64();
    let vram_post_backward = log_writeback_vram(store, "masked-writeback", "post backward");
    eprintln!("[masked-writeback] phase=backward seconds={bwd_secs:.3}");
    // CP replicates weights across the sequence-shard group and DP across the
    // batch-shard group; either way each rank produced only its contribution to
    // every weight grad. All-reduce-sum the trainable grads to the exact global
    // grad — MUST be after backward and before the optimizer step, and gated on
    // `step_optimizer` so PG's grad-accumulation (step_optimizer=false) doesn't
    // re-scale earlier microsteps by the group size. The collective is the same
    // sum over whatever NCCL group the backend holds (CP or DP); world==1 is a
    // no-op, so single-card stays byte-identical.
    if (cp.is_enabled() || dp.is_enabled()) && step_optimizer {
        crate::grad_clip::all_reduce_cp_grads(trainable_params, store).map_err(OpdError::from)?;
    }
    // Pre-step grad-norm telemetry. The prod `clip_grad_norm` logger sits on the
    // non-writeback OPD paths; the agentic-OPD writeback steps the optimizer here
    // directly, so surface the global L2 norm at this step too (env-gated).
    if std::env::var("ARLE_OPD_LOG_GRAD_NORM").is_ok() {
        let gn = crate::grad_clip::compute_global_norm_f64(trainable_params, store);
        eprintln!("[writeback-grad] grad_norm={gn:.6e}");
    }
    // Per-param grad norms, post-all-reduce: a global norm cannot say WHICH
    // params diverge between cp=1 and cp=2 (#85).
    if std::env::var("ARLE_OPD_DUMP_PARAM_GRADS").is_ok() {
        let names: std::collections::HashMap<TensorId, &'static str> = student
            .param_name_map()
            .into_iter()
            .chain(student.adapter_name_map())
            .map(|(name, id)| (id, name))
            .collect();
        for &param in trainable_params {
            let norm = crate::grad_clip::compute_global_norm_f64(&[param], store);
            let name = names.get(&param).copied().unwrap_or("<unnamed>");
            eprintln!("[param-grad] {name} id={param} norm={norm:.9e}");
        }
    }
    let t_opt = Instant::now();
    let grad_norm = if step_optimizer {
        Some(finite_optimizer_step(
            loss_value,
            trainable_params,
            0.0,
            optimizer,
            store,
        )?)
    } else {
        None
    };
    cleanup_after_backward(store, &mut tape, all_model_params, &keep_extra);
    let opt_secs = t_opt.elapsed().as_secs_f64();
    let vram_post_cleanup = log_writeback_vram(store, "masked-writeback", "post cleanup");
    log_writeback_vram_ledger(
        "masked-writeback",
        vram_base,
        vram_post_forward,
        vram_post_backward,
        vram_post_cleanup,
    );
    eprintln!("[masked-writeback] phase=optimizer_cleanup seconds={opt_secs:.3}");

    eprintln!("[masked-writeback] DONE loss={loss_value:.6} total_targets={total_targets}");
    Ok((loss_value, pg_stats, grad_norm))
}

/// Positions per `[rows, vocab]` logits tile in [`capture_rollout_logprobs`];
/// bounds the transient softmax tile (≈ rows·vocab·4 B) since the capture has no
/// `window_size` knob of its own.
const CAPTURE_LOGPROB_CHUNK_ROWS: usize = 256;

/// `log_softmax(logits)[target]` at each masked
/// (LLM-generated) response position, in the SAME order the fused writeback ops
/// consume (`build_masked_loss_targets`). Tape-OFF forward over `prompt ++
/// response` — call BEFORE any optimizer step so θ is still the rollout policy
/// (V0). Returns one f32 per masked target position (empty if none).
pub fn capture_rollout_logprobs(
    student: &Qwen35Model,
    prompt_ids: &[u32],
    response_ids: &[u32],
    response_mask: &[u8],
    store: &mut TensorStore,
) -> Result<Vec<f32>> {
    if prompt_ids.is_empty() {
        return Err(OpdError::InvalidInput(
            "capture_rollout_logprobs requires a non-empty prompt".to_owned(),
        ));
    }
    let prompt_len = prompt_ids.len();
    let full: Vec<u32> = prompt_ids
        .iter()
        .copied()
        .chain(response_ids.iter().copied())
        .collect();
    let seq_len = full.len();
    let loss_targets = build_masked_loss_targets(&full, prompt_len, response_mask);
    if loss_targets.is_empty() {
        return Ok(Vec::new());
    }
    let positions: Vec<u32> = (0..seq_len as u32).collect();

    // Frozen-prompt-KV: forward only the gen segment. gen_start=0 keeps the full
    // path byte-identical. Masked rows p >= prompt_len-1 = gen_start, so the
    // gathered `rows` rebase to p-gen_start (>= 0) into the [gen_len, hidden]
    // tensor; output logprobs stay in the same target order.
    let frozen = crate::runtime_flags::writeback_frozen_prompt_kv() && prompt_len > 1;
    let gen_start = if frozen { prompt_len - 1 } else { 0 };
    let hidden_rows = seq_len - gen_start;

    // Snapshot live tensors so the forward + per-chunk projection intermediates
    // (hidden, logits, logp, gathered) + the tape's checkpoints are reclaimed at
    // the end — `to_host` copies but does not free them, so trajectories leak.
    let keep_ids: HashSet<TensorId> = store.live_ids().into_iter().collect();

    // Tape ENABLED (but never backwarded) so `should_checkpoint` engages and the
    // forward frees per-layer activations instead of piling all 64 layers'
    // [seq, hidden] resident — a tape-OFF forward has no checkpointing and OOMs on
    // long trajectories atop the two resident 27B models. Checkpointing is
    // numerically exact, so π_rollout is unchanged. Needs `--gradient-checkpointing`.
    let mut tape = Tape::new();
    tape.set_enabled(true);
    // Same seq-adaptive host-offload gate as the writeback forward. Capture is
    // never backwarded, so offloaded checkpoint inputs are never re-uploaded —
    // the forward retains no device activations across layers (resident
    // checkpoints spiked 50.7→97.4 GB and OOMed at seq≈15K; offload is
    // numerically transparent, see `checkpoint_offload_is_transparent`).
    tape.set_offload_checkpoints(crate::runtime_flags::writeback_offload_for_seq(hidden_rows));
    let hidden = if frozen {
        let seg = GenSegment::split(&full, prompt_len);
        student
            .forward_hidden_states_gen_segment(
                store,
                &mut tape,
                &seg.prompt_prefix,
                &seg.gen_ids,
                &seg.prompt_positions,
                &seg.gen_positions,
                crate::context_parallel::CpContext::single(),
            )
            .map_err(|err| {
                map_qwen35_forward_error("rollout-logprob frozen-prompt-KV student hidden", err)
            })?
    } else {
        student
            .forward_hidden_states(
                store,
                &mut tape,
                &full,
                &positions,
                crate::context_parallel::CpContext::single(),
            )
            .map_err(|err| map_qwen35_forward_error("rollout-logprob student hidden", err))?
    };
    let hidden_dim = *store
        .get(hidden)
        .ok_or(AutogradError::InvalidTensorId(hidden))?
        .shape
        .last()
        .ok_or_else(|| OpdError::InvalidInput("rollout-logprob: empty hidden shape".to_owned()))?;
    let dbg = std::env::var("ARLE_OPD_LOG_DIS_STATS").is_ok();
    let shp = |store: &TensorStore, id| store.get(id).map(|t| t.shape.clone());
    if dbg {
        eprintln!(
            "[capture] seq_len={seq_len} targets={} hidden_shape={:?} hidden_dim={hidden_dim}",
            loss_targets.len(),
            shp(store, hidden),
        );
    }
    let hidden_2d =
        reshape(hidden, &[hidden_rows, hidden_dim], store, &mut tape).map_err(OpdError::from)?;
    let lm_head = student.lm_head_weight_id();

    let mut logprobs = Vec::with_capacity(loss_targets.len());
    for chunk in loss_targets.chunks(CAPTURE_LOGPROB_CHUNK_ROWS) {
        let rows: Vec<usize> = chunk.iter().map(|&(p, _)| p - gen_start).collect();
        let targets: Vec<usize> = chunk.iter().map(|&(_, t)| t).collect();
        // Never materializes [seq, vocab]. `embedding` row-gathers but emits
        // [1, chunk, hidden]; reshape to rank-2 so the bf16 matmul_bt forward
        // (which rejects rank-3) accepts it.
        let rows_hidden_3d =
            embedding(hidden_2d, &rows, store, &mut tape).map_err(OpdError::from)?;
        let rows_hidden = reshape(rows_hidden_3d, &[rows.len(), hidden_dim], store, &mut tape)
            .map_err(OpdError::from)?;
        if dbg {
            eprintln!(
                "[capture] chunk rows={} rows_hidden={:?} lm_head={:?}",
                rows.len(),
                shp(store, rows_hidden),
                shp(store, lm_head),
            );
        }
        let logits = matmul_bt(rows_hidden, lm_head, store, &mut tape).map_err(OpdError::from)?;
        let logp = log_softmax(logits, store, &mut tape).map_err(OpdError::from)?;
        let gathered = gather_last_dim(logp, &targets, store, &mut tape).map_err(OpdError::from)?;
        logprobs.extend_from_slice(&store.to_host(gathered).map_err(OpdError::from)?);
    }
    store.retain_ids(&keep_ids);
    Ok(logprobs)
}

/// GKD (per-token teacher-KL) writeback for agent-OPD replay — the distillation
/// sibling of [`masked_writeback_step`]. Instead of hard next-token CE on the
/// passing trajectory tokens, it distils a TEACHER's per-position distribution on
/// the SAME masked (LLM-generated) positions via forward-KL, so the signal is
/// dense (every vocab logit, not one hard target) rather than pure reproduction.
///
/// Mirrors the CE path's structure: ONE gradient-checkpointed
/// `forward_hidden_states` over `prompt ++ response` → `[1, seq, hidden]`, then
/// per masked window `[ws, we)` computes the student logits
/// `hidden[ws..we] @ lm_headᵀ` → `[1, w, vocab]`, forwards the frozen TEACHER over
/// the same window → `[1, w, vocab]`, and reuses
/// [`kl_distill_loss_chunked`](crate::loss::kl_distill_loss_chunked) (forward KL,
/// `batchmean`) — chunking the softmax intermediates. Windows track contiguous
/// runs of the `response_mask` so tool/environment positions never receive loss.
/// The returned scalar is the target-count-weighted mean KL per masked token.
///
/// `window_size` (reuse `--writeback-window`) bounds each `[1, w, vocab]` tile;
/// unlike the CE path a full-vocab logit tile is inherent to KL, so prefer a
/// smaller window here than for masked CE.
#[allow(clippy::too_many_arguments)]
pub fn masked_gkd_writeback_step<O: Optimizer, T: TeacherForward + ?Sized>(
    student: &Qwen35Model,
    teacher: &T,
    all_model_params: &[TensorId],
    trainable_params: &[TensorId],
    optimizer: &mut O,
    prompt_ids: &[u32],
    response_ids: &[u32],
    response_mask: &[u8],
    vocab: usize,
    window_size: usize,
    temperature: f32,
    store: &mut TensorStore,
) -> Result<f32> {
    if prompt_ids.is_empty() {
        return Err(OpdError::InvalidInput(
            "GKD writeback requires a non-empty prompt".to_owned(),
        ));
    }
    if window_size == 0 {
        return Err(OpdError::InvalidInput(
            "GKD writeback window_size must be > 0".to_owned(),
        ));
    }
    if teacher.vocab_size() != vocab {
        return Err(OpdError::InvalidInput(format!(
            "GKD writeback teacher vocab_size {} != student vocab {vocab}",
            teacher.vocab_size()
        )));
    }

    let prompt_len = prompt_ids.len();
    let full: Vec<u32> = prompt_ids
        .iter()
        .copied()
        .chain(response_ids.iter().copied())
        .collect();
    let seq_len = full.len();
    if seq_len > u32::MAX as usize {
        return Err(OpdError::InvalidInput(format!(
            "GKD writeback trajectory length {seq_len} exceeds u32::MAX position ids."
        )));
    }
    let positions: Vec<u32> = (0..seq_len as u32).collect();

    let loss_targets = build_masked_loss_targets(&full, prompt_len, response_mask);
    if loss_targets.is_empty() {
        eprintln!(
            "[gkd-writeback] no LLM-generated targets (prompt_len={prompt_len}, \
             response_len={}, mask_ones={}); skipping (nothing to train)",
            response_ids.len(),
            response_mask.iter().filter(|&&m| m == 1).count(),
        );
        return Ok(0.0);
    }
    let masked_positions: Vec<usize> = loss_targets.iter().map(|&(p, _)| p).collect();
    let total_targets = masked_positions.len();
    let windows = masked_gkd_windows(&masked_positions, window_size);

    let mut tape = Tape::new();
    let offload_checkpoints = crate::runtime_flags::writeback_offload_for_seq(seq_len);
    tape.set_offload_checkpoints(offload_checkpoints);
    tape.set_enabled(true);
    // Retain the teacher's params across the post-backward `retain_ids` prune:
    // the EMA adapter (and any teacher-only weights) are NOT in the student's
    // `all_model_params`, so without this they would be freed and the next step's
    // teacher forward / EMA update would read dropped tensors.
    let keep_extra: HashSet<TensorId> = teacher.parameter_ids().iter().copied().collect();
    eprintln!(
        "[gkd-writeback] seq_len={seq_len} total_targets={total_targets} \
         windows={} temperature={temperature} offload_checkpoints={offload_checkpoints}",
        windows.len()
    );

    // ONE checkpointed forward over prompt++response → [1, seq, hidden]; the
    // student logits for each window are hidden[ws..we] @ lm_headᵀ (never the full
    // [seq, vocab]). The teacher scores the same window separately (frozen).
    let hidden = student
        .forward_hidden_states(
            store,
            &mut tape,
            &full,
            &positions,
            crate::context_parallel::CpContext::single(),
        )
        .map_err(|err| map_qwen35_forward_error("GKD writeback student hidden", err))?;
    let hidden_dim = *store
        .get(hidden)
        .ok_or(AutogradError::InvalidTensorId(hidden))?
        .shape
        .last()
        .ok_or_else(|| OpdError::InvalidInput("GKD writeback: empty hidden shape".to_owned()))?;
    let lm_head = student.lm_head_weight_id();

    let mut total_loss: Option<TensorId> = None;
    for window in &windows {
        let w = window.end - window.start;
        let hidden_slice = slice(
            hidden,
            &[0, window.start, 0],
            &[1, window.end, hidden_dim],
            store,
            &mut tape,
        )
        .map_err(OpdError::from)?;
        let hidden_2d =
            reshape(hidden_slice, &[w, hidden_dim], store, &mut tape).map_err(OpdError::from)?;
        let student_logits_2d =
            matmul_bt(hidden_2d, lm_head, store, &mut tape).map_err(OpdError::from)?;
        let student_logits =
            reshape(student_logits_2d, &[1, w, vocab], store, &mut tape).map_err(OpdError::from)?;

        let teacher_logits = teacher
            .forward_logits_window_device(&full, &positions, *window, store, &mut tape)
            .map_err(|err| OpdError::InvalidInput(format!("GKD teacher window forward: {err}")))?;

        let kl = kl_distill_loss_chunked(
            student_logits,
            teacher_logits.tensor_id,
            w,
            DEFAULT_KL_CHUNK_SIZE,
            temperature,
            KlDirection::Forward,
            store,
            &mut tape,
        )
        .map_err(OpdError::from)?;
        // Target-count weight so the accumulated loss is the mean KL per masked
        // token (each window contributes its share of the total masked positions).
        let weighted = mul_scalar(kl, w as f32 / total_targets as f32, store, &mut tape)
            .map_err(OpdError::from)?;
        total_loss = Some(match total_loss {
            Some(prev) => add(prev, weighted, store, &mut tape).map_err(OpdError::from)?,
            None => weighted,
        });
    }

    let loss = total_loss.ok_or_else(|| {
        OpdError::InvalidInput(
            "GKD writeback produced no windows despite non-empty targets".to_owned(),
        )
    })?;
    let loss_value = store.to_host(loss).map_err(OpdError::from)?[0];
    validate_loss_value(loss_value)?;
    backward_with_optional_profile(loss, loss_value, store, &mut tape)?;
    finite_optimizer_step(loss_value, trainable_params, 0.0, optimizer, store)?;
    cleanup_after_backward(store, &mut tape, all_model_params, &keep_extra);

    eprintln!("[gkd-writeback] DONE mean_kl={loss_value:.6} total_targets={total_targets}");
    Ok(loss_value)
}
