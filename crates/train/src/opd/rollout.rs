//! Student rollout generation: greedy/sampled decode, device argmax, and the step's rollout phase.

use std::{collections::HashSet, sync::LazyLock, time::Instant};

use autograd::{AutogradError, Device, Tape, TensorId, TensorStore};
use infer_plan::{SamplingParams, sample_token};

use crate::{
    causal_lm::live_tensor_ids,
    qwen35::{
        Qwen35KvCache, Qwen35Model, forward_rollout_cached, forward_rollout_cached_device_token,
    },
};

#[cfg(feature = "cuda")]
use super::InferRolloutCtx;
#[cfg(feature = "cuda")]
use super::log_free_vram;
use super::{
    EngineOffloadMode, OpdError, OpdStepConfig, OpdStepProfile, Result, log_opd_step_trace,
    map_qwen35_forward_error, record_profile,
    validation::{validate_forced_rollout, validate_rollout_shape, validate_token_ids},
};

fn greedy_next_token(
    logits_id: TensorId,
    seq_len: usize,
    vocab: usize,
    store: &mut TensorStore,
    sampling: Option<&SamplingParams>,
    position: u64,
) -> Result<u32> {
    let host = store.to_host(logits_id)?;
    if seq_len == 0 || vocab == 0 {
        return Err(OpdError::InvalidInput(format!(
            "OPD rollout cannot sample next token with seq_len={seq_len}, vocab={vocab}. \
             Hint: pass a non-empty prompt and a Qwen35Config with vocab_size > 0."
        )));
    }
    let expected_len = seq_len.checked_mul(vocab).ok_or_else(|| {
        OpdError::InvalidInput(format!(
            "OPD rollout logits shape overflow for seq_len={seq_len}, vocab={vocab}. \
             Hint: check the prompt length and Qwen35Config::vocab_size before calling opd_step."
        ))
    })?;
    if host.len() != expected_len {
        return Err(OpdError::InvalidInput(format!(
            "OPD rollout logits length mismatch: logits_len={}, expected exactly \
             seq_len * vocab = {expected_len} ({seq_len} * {vocab}). Hint: check \
             Qwen35Model::forward output shape and Qwen35Config::vocab_size.",
            host.len()
        )));
    }
    let last_row_start = (seq_len - 1) * vocab;
    let row = &host[last_row_start..last_row_start + vocab];
    if let Some(params) = sampling {
        for (i, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(OpdError::InvalidInput(format!(
                    "OPD rollout logits contain non-finite value at last-row vocab index {i}: {v}. \
                     Hint: check student forward numerics, checkpoint dtype conversion, and \
                     learning-rate stability before sampling the next token."
                )));
            }
        }
        return Ok(sample_token(row, params, position));
    }

    let mut best_idx: usize = 0;
    let mut best_val: f32 = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if !v.is_finite() {
            return Err(OpdError::InvalidInput(format!(
                "OPD rollout logits contain non-finite value at last-row vocab index {i}: {v}. \
                 Hint: check student forward numerics, checkpoint dtype conversion, and \
                 learning-rate stability before sampling the next token."
            )));
        }
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(best_idx as u32)
}

fn device_argmax_token(
    logits_id: TensorId,
    vocab: usize,
    store: &mut TensorStore,
    sampling: Option<&SamplingParams>,
    position: u64,
) -> Result<TensorId> {
    if vocab == 0 {
        return Err(OpdError::InvalidInput(
            "OPD rollout cannot sample next token with vocab=0. Hint: verify \
             Qwen35Config::vocab_size before calling opd_step."
                .to_owned(),
        ));
    }
    let shape = store
        .get(logits_id)
        .ok_or(AutogradError::InvalidTensorId(logits_id))?
        .shape
        .clone();
    let last_dim = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim != vocab {
        return Err(OpdError::InvalidInput(format!(
            "OPD rollout logits last dim mismatch: got {last_dim}, expected \
             vocab={vocab}. Hint: check Qwen35Model::forward output shape and \
             Qwen35Config::vocab_size."
        )));
    }
    let total = shape.iter().product::<usize>();
    let rows = total / vocab;
    if rows != 1 {
        return Err(OpdError::InvalidInput(format!(
            "OPD device rollout expects exactly one logits row, got {rows}. \
             Hint: rollout KV cache should return only the final next-token \
            logits row."
        )));
    }
    if let Some(params) = sampling {
        let host = store.to_host(logits_id)?;
        if host.len() != vocab {
            return Err(OpdError::InvalidInput(format!(
                "OPD sampled device rollout logits length mismatch: got {}, expected vocab={vocab}. \
                 Hint: rollout KV cache should return exactly one final next-token row.",
                host.len()
            )));
        }
        for (i, &v) in host.iter().enumerate() {
            if !v.is_finite() {
                return Err(OpdError::InvalidInput(format!(
                    "OPD sampled device rollout logits contain non-finite value at vocab index {i}: {v}. \
                     Hint: check student forward numerics before sampling the next token."
                )));
            }
        }
        let sampled = sample_token(&host, params, position);
        let token_id = store.from_slice(&[sampled as f32], &[1])?;
        store.ensure_device(token_id)?;
        return Ok(token_id);
    }

    store.ensure_device(logits_id)?;
    let logits_handle = store
        .get(logits_id)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "device_argmax_token: logits missing device handle",
        ))?;
    let token_handle = store.backend().argmax_last_dim(&logits_handle, &shape)?;
    Ok(store.alloc_device_tensor(vec![rows], token_handle)?)
}

fn write_rollout_token(
    buffer_id: TensorId,
    token_id: TensorId,
    rollout_len: usize,
    step: usize,
    store: &mut TensorStore,
) -> Result<TensorId> {
    store.ensure_device(buffer_id)?;
    store.ensure_device(token_id)?;
    let buffer_handle = store
        .get(buffer_id)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "write_rollout_token: rollout buffer missing device handle",
        ))?;
    let token_handle = store
        .get(token_id)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "write_rollout_token: token missing device handle",
        ))?;
    let next_handle =
        store
            .backend()
            .write_scalar_at(&buffer_handle, &token_handle, rollout_len, step)?;
    Ok(store.alloc_device_tensor(vec![rollout_len], next_handle)?)
}

fn read_generated_rollout_tokens(
    buffer_id: TensorId,
    rollout_len: usize,
    vocab: usize,
    store: &mut TensorStore,
) -> Result<Vec<u32>> {
    let host = store.to_host(buffer_id)?;
    if host.len() != rollout_len {
        return Err(OpdError::InvalidInput(format!(
            "OPD generated rollout token buffer length mismatch: got {}, \
             expected {rollout_len}. Hint: device argmax rollout buffer shape \
             should match cfg.rollout_len.",
            host.len()
        )));
    }
    let out: Vec<u32> = host
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if !value.is_finite() {
                return Err(OpdError::InvalidInput(format!(
                    "OPD generated rollout token at index {index} is non-finite ({value}). \
                     Hint: check CUDA argmax output and student forward numerics."
                )));
            }
            let rounded = value.round();
            if (value - rounded).abs() > 0.0 {
                return Err(OpdError::InvalidInput(format!(
                    "OPD generated rollout token at index {index} is not an exact \
                     integer id ({value}). Hint: CUDA argmax should write exact \
                     f32 token ids."
                )));
            }
            if rounded < 0.0 || rounded as usize >= vocab {
                return Err(OpdError::InvalidInput(format!(
                    "OPD generated rollout token id {rounded} at index {index} is \
                     outside student.config().vocab_size={vocab}. Hint: check the \
                     argmax kernel bounds and model vocab size."
                )));
            }
            Ok(rounded as u32)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(out)
}

fn use_device_rollout_argmax(store: &TensorStore, rollout_len: usize, vocab: usize) -> bool {
    matches!(store.backend().device(), Device::Cuda) && (rollout_len >= 4 || vocab >= 65_536)
}

static ROLLOUT_PROGRESS_ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("ARLE_OPD_ROLLOUT_PROGRESS").is_some());

fn should_retain_rollout_step(step: usize, rollout_len: usize) -> bool {
    let completed_steps = step + 1;
    completed_steps.is_multiple_of(crate::runtime_flags::rollout_retain_interval())
        || completed_steps == rollout_len
}

fn maybe_log_rollout_progress(path: &str, step: usize, rollout_len: usize, started: &Instant) {
    if !*ROLLOUT_PROGRESS_ENABLED {
        return;
    }
    let completed_steps = step + 1;
    if completed_steps.is_multiple_of(crate::runtime_flags::rollout_progress_interval())
        || completed_steps == rollout_len
    {
        eprintln!(
            "opd_rollout_progress path={path} step={completed_steps}/{rollout_len} \
             elapsed_seconds={:.3} retain_interval={}",
            started.elapsed().as_secs_f64(),
            crate::runtime_flags::rollout_retain_interval()
        );
    }
}

fn retain_rollout_step_tensors(
    store: &mut TensorStore,
    base_keep: &HashSet<TensorId>,
    rollout_cache: &Qwen35KvCache,
    current_device_token: Option<TensorId>,
    generated_tokens: Option<TensorId>,
) {
    let mut keep = base_keep.clone();
    rollout_cache.extend_tensor_ids(&mut keep);
    if let Some(token_id) = current_device_token {
        keep.insert(token_id);
    }
    if let Some(buffer_id) = generated_tokens {
        keep.insert(buffer_id);
    }
    store.retain_ids(&keep);
}

fn rollout_full_forward(
    student: &Qwen35Model,
    rollout: &mut Vec<u32>,
    rollout_len: usize,
    vocab: usize,
    sampling: Option<&SamplingParams>,
    store: &mut TensorStore,
    tape: &mut Tape,
    base_keep: &HashSet<TensorId>,
) -> Result<()> {
    let rollout_started = Instant::now();
    for step in 0..rollout_len {
        let positions = (0..rollout.len() as u32).collect::<Vec<_>>();
        let logits = student
            .forward(store, tape, rollout, &positions)
            .map_err(|err| map_qwen35_forward_error("student rollout", err))?;
        let next = greedy_next_token(
            logits,
            rollout.len(),
            vocab,
            store,
            sampling,
            rollout.len() as u64,
        )?;
        rollout.push(next);
        if should_retain_rollout_step(step, rollout_len) {
            store.retain_ids(base_keep);
        }
        maybe_log_rollout_progress("full-forward", step, rollout_len, &rollout_started);
    }
    store.retain_ids(base_keep);
    Ok(())
}

pub fn student_rollout_only(
    student: &Qwen35Model,
    prompt_ids: &[u32],
    rollout_len: usize,
    sampling: Option<&SamplingParams>,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<Vec<u32>> {
    let vocab = student.config().vocab_size;
    if prompt_ids.is_empty() {
        return Err(OpdError::InvalidInput(
            "student_rollout_only requires a non-empty prompt_ids slice. Hint: \
             pass at least one BOS/chat token; the rollout helper does not \
             synthesize prompts."
                .to_owned(),
        ));
    }
    if vocab == 0 {
        return Err(OpdError::InvalidInput(
            "student_rollout_only requires student.config().vocab_size > 0.".to_owned(),
        ));
    }
    validate_rollout_shape(prompt_ids.len(), rollout_len, vocab)?;
    validate_token_ids("prompt_ids", prompt_ids, vocab)?;
    let rollout_keep_base = live_tensor_ids(store);
    student_rollout_only_with_keep(
        student,
        prompt_ids,
        rollout_len,
        sampling,
        &rollout_keep_base,
        store,
        tape,
    )
}

fn student_rollout_only_with_keep(
    student: &Qwen35Model,
    prompt_ids: &[u32],
    rollout_len: usize,
    sampling: Option<&SamplingParams>,
    rollout_keep_base: &HashSet<TensorId>,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<Vec<u32>> {
    tape.entries.clear();
    tape.set_enabled(false);
    let mut rollout: Vec<u32> = prompt_ids.to_vec();
    let vocab = student.config().vocab_size;
    let use_rollout_kv_cache = student.supports_rollout_kv_cache();

    if use_rollout_kv_cache && use_device_rollout_argmax(store, rollout_len, vocab) {
        let mut rollout_cache = Qwen35KvCache::new(student, prompt_ids.len() + rollout_len);
        let mut generated_tokens = if rollout_len == 0 {
            None
        } else {
            let handle = store.backend().zeros(&[rollout_len])?;
            Some(store.alloc_device_tensor(vec![rollout_len], handle)?)
        };
        let mut current_device_token: Option<TensorId> = None;
        let rollout_started = Instant::now();
        for step in 0..rollout_len {
            let logits = if step == 0 {
                let positions = (0..prompt_ids.len() as u32).collect::<Vec<_>>();
                forward_rollout_cached(
                    student,
                    store,
                    tape,
                    prompt_ids,
                    &positions,
                    &mut rollout_cache,
                )
                .map_err(|err| map_qwen35_forward_error("student rollout", err))?
            } else {
                let token_id = current_device_token.ok_or_else(|| {
                    OpdError::InvalidInput(
                        "OPD rollout cache cannot decode from an empty rollout. Hint: pass a \
                         non-empty prompt before calling student_rollout_only."
                            .to_owned(),
                    )
                })?;
                let position = (prompt_ids.len() + step - 1) as u32;
                forward_rollout_cached_device_token(
                    student,
                    store,
                    tape,
                    token_id,
                    position,
                    &mut rollout_cache,
                )
                .map_err(|err| map_qwen35_forward_error("student rollout", err))?
            };
            let next_token = device_argmax_token(
                logits,
                vocab,
                store,
                sampling,
                (prompt_ids.len() + step) as u64,
            )?;
            if let Some(buffer_id) = generated_tokens {
                generated_tokens = Some(write_rollout_token(
                    buffer_id,
                    next_token,
                    rollout_len,
                    step,
                    store,
                )?);
            }
            current_device_token = Some(next_token);
            if should_retain_rollout_step(step, rollout_len) {
                retain_rollout_step_tensors(
                    store,
                    rollout_keep_base,
                    &rollout_cache,
                    current_device_token,
                    generated_tokens,
                );
            }
            maybe_log_rollout_progress("device-token", step, rollout_len, &rollout_started);
        }
        if let Some(buffer_id) = generated_tokens {
            rollout.extend(read_generated_rollout_tokens(
                buffer_id,
                rollout_len,
                vocab,
                store,
            )?);
        }
        store.retain_ids(rollout_keep_base);
    } else if use_rollout_kv_cache {
        let mut rollout_cache = Qwen35KvCache::new(student, prompt_ids.len() + rollout_len);
        let rollout_started = Instant::now();
        for step in 0..rollout_len {
            let (input_ids, positions, logits_seq_len) = if step == 0 {
                (
                    rollout.clone(),
                    (0..rollout.len() as u32).collect::<Vec<_>>(),
                    1,
                )
            } else {
                let last = *rollout.last().ok_or_else(|| {
                    OpdError::InvalidInput(
                        "OPD rollout cache cannot decode from an empty rollout. Hint: pass a \
                         non-empty prompt before calling student_rollout_only."
                            .to_owned(),
                    )
                })?;
                let position = (rollout.len() - 1) as u32;
                (vec![last], vec![position], 1)
            };
            let logits = forward_rollout_cached(
                student,
                store,
                tape,
                &input_ids,
                &positions,
                &mut rollout_cache,
            )
            .map_err(|err| map_qwen35_forward_error("student rollout", err))?;
            let next = greedy_next_token(
                logits,
                logits_seq_len,
                vocab,
                store,
                sampling,
                rollout.len() as u64,
            )?;
            rollout.push(next);
            if should_retain_rollout_step(step, rollout_len) {
                retain_rollout_step_tensors(store, rollout_keep_base, &rollout_cache, None, None);
            }
            maybe_log_rollout_progress("host-token", step, rollout_len, &rollout_started);
        }
        store.retain_ids(rollout_keep_base);
    } else {
        rollout_full_forward(
            student,
            &mut rollout,
            rollout_len,
            vocab,
            sampling,
            store,
            tape,
            rollout_keep_base,
        )?;
    }

    debug_assert!(
        tape.entries.is_empty(),
        "student_rollout_only must keep rollout candidates off the backward tape"
    );
    Ok(rollout)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(super) fn rollout_phase(
    student: &Qwen35Model,
    prompt_ids: &[u32],
    cfg: &OpdStepConfig,
    vocab: usize,
    forced_rollout: Option<&[u32]>,
    rollout_keep_base: &HashSet<TensorId>,
    engine_offload: EngineOffloadMode,
    total_started: Instant,
    store: &mut TensorStore,
    tape: &mut Tape,
    profile: &mut Option<&mut OpdStepProfile>,
    #[cfg(feature = "cuda")] infer_rollout: Option<&InferRolloutCtx<'_>>,
) -> Result<Vec<u32>> {
    let phase_started = Instant::now();
    store.retain_ids(rollout_keep_base);
    let rollout = if let Some(forced_rollout) = forced_rollout {
        tape.entries.clear();
        tape.set_enabled(false);
        log_opd_step_trace(total_started, "forced_rollout_start", "");
        validate_forced_rollout(forced_rollout, prompt_ids, cfg.rollout_len, vocab)?;
        store.retain_ids(rollout_keep_base);
        forced_rollout.to_vec()
    } else {
        let rollout_sampling = cfg.rollout_sampling.as_ref();
        // Infer-engine rollout: mirror the train LoRA into the infer student
        // once per step, then decode via the infer engine. Same `rollout`
        // output as the train-crate helper.
        #[cfg(feature = "cuda")]
        if let Some(ctx) = infer_rollout {
            log_opd_step_trace(total_started, "infer_rollout_reload_start", "");
            // OPD engine time-share: the rollout student may have been
            // offloaded to host RAM during the previous step's backward.
            // Reload it before the LoRA sync (which re-merges resident base
            // weights) and the rollout decode. Fence the train backend first
            // so the previous step's optimizer/cleanup pool ops are ordered
            // ahead of the reload's pool allocations (same cross-context
            // ordering reason as the teacher reload fence).
            if engine_offload.offloads_student() {
                store
                    .backend()
                    .device_synchronize()
                    .map_err(OpdError::from)?;
                ctx.student.reload_engine_weights().map_err(|err| {
                    OpdError::InvalidInput(format!("infer student reload failed: {err}"))
                })?;
            }
            log_free_vram(store, "after_student_reload");
            log_opd_step_trace(total_started, "infer_rollout_sync_lora_start", "");
            ctx.student
                .sync_lora_from_store(
                    store,
                    &student.adapter_name_map(),
                    &student.param_name_map(),
                    ctx.lora_config,
                )
                .map_err(|err| {
                    OpdError::InvalidInput(format!("infer student LoRA sync failed: {err}"))
                })?;
            log_opd_step_trace(total_started, "infer_rollout_generate_start", "");
            let rollout = ctx
                .student
                .generate_rollout(prompt_ids, cfg.rollout_len, rollout_sampling)
                .map_err(|err| {
                    OpdError::InvalidInput(format!("infer student rollout failed: {err}"))
                })?;
            log_opd_step_trace(
                total_started,
                "infer_rollout_generate_done",
                format!("actual_rollout_len={}", rollout.len()),
            );
            log_free_vram(store, "after_rollout_generate");
            store.retain_ids(rollout_keep_base);
            // NB: the infer student engine is idle after the rollout, but we do
            // NOT offload it here. Offloading mid-step (before the teacher
            // forward) churns the shared device memory pool while the teacher
            // forward allocates from it, racing the async frees → illegal
            // address. Instead both idle engines are offloaded together AFTER
            // the teacher scores, just before the student backward (see
            // `backward_chunked_kl`), on a quiesced device.
            rollout
        } else {
            log_opd_step_trace(total_started, "train_rollout_start", "");
            student_rollout_only(
                student,
                prompt_ids,
                cfg.rollout_len,
                rollout_sampling,
                store,
                tape,
            )?
        }
        #[cfg(not(feature = "cuda"))]
        {
            log_opd_step_trace(total_started, "train_rollout_start", "");
            student_rollout_only(
                student,
                prompt_ids,
                cfg.rollout_len,
                rollout_sampling,
                store,
                tape,
            )?
        }
    };
    record_profile(profile, |profile| {
        profile.student_rollout_seconds += phase_started.elapsed().as_secs_f64();
    });
    log_opd_step_trace(
        total_started,
        "student_rollout_done",
        format!("actual_rollout_len={}", rollout.len()),
    );
    Ok(rollout)
}
