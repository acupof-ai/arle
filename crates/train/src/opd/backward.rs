//! Backward routes over the scored rollout: chunked full-prefix KL and the windowed GKD loop.

use std::{collections::HashSet, time::Instant};

use autograd::{
    AutogradError, Tape, TensorId, TensorStore,
    ops::{mul_scalar, slice},
    tensor::Dirty,
};

use crate::{
    loss::{
        KlDirection, fused_linear_distill_loss, generalized_beta_jsd_loss_chunked,
        kl_distill_loss_chunked,
    },
    qwen35::{Qwen35Model, SequenceWindow},
    teacher_infer::TeacherForward,
    trainer::{cleanup_after_backward, extend_keep_with_params_and_grads},
};

use super::{
    EngineOffloadMode, GkdLossConfig, GkdSftAnchor, OpdError, OpdKlMask, OpdStepProfile, Result,
    backward_with_optional_profile, log_opd_window_trace,
    loss::{kl_distill_loss_for_config, next_token_sft_loss_from_logits},
    map_qwen35_forward_error, map_teacher_forward_error, opd_backward_profile_enabled,
    print_backward_profile, record_profile,
    validation::{validate_logits_shape, validate_loss_value},
    windowing::{kl_logit_range, sequence_windows, sequence_windows_for_range},
};

#[allow(clippy::too_many_arguments)]
pub(super) fn backward_chunked_kl_rollout<T: TeacherForward + ?Sized>(
    student: &Qwen35Model,
    teacher: &T,
    rollout: &[u32],
    prompt_len: usize,
    vocab: usize,
    chunk_size: usize,
    kl_mask: OpdKlMask,
    kl_direction: KlDirection,
    kl_temperature: f32,
    kl_beta: Option<f32>,
    student_model_params: &[TensorId],
    keep_extra: &HashSet<TensorId>,
    store: &mut TensorStore,
    tape: &mut Tape,
    profile: &mut Option<&mut OpdStepProfile>,
    engine_offload: EngineOffloadMode,
    student_engine_offload: Option<&dyn Fn() -> Result<usize>>,
    student_engine_reload: Option<&dyn Fn() -> Result<()>>,
) -> Result<f32> {
    let kl_range = kl_logit_range(kl_mask, prompt_len, rollout.len())?;
    if kl_range.start >= kl_range.end {
        return Ok(0.0);
    }

    // Forward teacher + student over the scored prefix exactly ONCE. Causal
    // attention makes position p's logits independent of tokens after p, so a
    // single full-prefix forward yields the same per-position logits the old
    // per-chunk loop produced by re-forwarding `[0..seq_end_k]` from token 0
    // for every chunk — but without the O(n^2) redundant recompute that
    // dominated backward (each chunk re-ran the full growing prefix through
    // the dense base model). Chunking now happens only at the loss/softmax
    // level via `kl_distill_loss_chunked`, matching the already-correct
    // `kl_distill_loss_for_config` path used by the non-rollout KL callers.
    let seq_end = kl_range.end;
    let positions = (0..seq_end as u32).collect::<Vec<_>>();
    let prefix = &rollout[..seq_end];

    tape.entries.clear();
    tape.set_enabled(false);
    // OPD engine time-share: the teacher may have been offloaded to host RAM
    // during the previous step's student backward. Reload it before scoring.
    // In `Student` mode the teacher stays resident, so this is skipped.
    //
    // Fence the train backend first. The reload re-allocates the teacher's
    // weight buffers (incl. the W4A8 Marlin packed side buffers) from the
    // shared async pool on the infer scheduler thread; without this barrier
    // the previous step's still-draining train pool ops race those allocs
    // (event tracking is disabled) and the reloaded Marlin side buffer is
    // dropped → "missing W4A8 Marlin-packed side buffer".
    if engine_offload.offloads_teacher() {
        store
            .backend()
            .device_synchronize()
            .map_err(OpdError::from)?;
        teacher
            .reload_engine_weights()
            .map_err(|err| map_teacher_forward_error("teacher reload", err))?;
    }
    let phase_started = Instant::now();
    let teacher_logits = teacher
        .forward_logits_device(prefix, &positions, store, tape)
        .map_err(|err| map_teacher_forward_error("teacher scoring", err))?;
    record_profile(profile, |profile| {
        profile.teacher_forward_seconds += phase_started.elapsed().as_secs_f64();
    });
    let expected_shape = vec![1, seq_end, vocab];
    if teacher_logits.shape != expected_shape {
        return Err(OpdError::InvalidInput(format!(
            "OPD chunked teacher logits shape mismatch: got {:?}, expected {:?}. \
             Hint: the TeacherForward implementation must return \
             [batch=1, seq_len, vocab] logits for the exact prefix being scored.",
            teacher_logits.shape, expected_shape
        )));
    }

    // KL over the completion region only. The teacher is a fixed distillation
    // target — no gradient ever flows into it — so slice its completion region
    // with the tape DISABLED. This is load-bearing for VRAM: a tape-enabled
    // slice registers a `slice_bwd` node that pre-allocates a grad buffer the
    // size of the *full* teacher logits (`[1, seq_end, vocab]` ≈ 142 MiB at
    // 144×248320), which was the rollout-128 backward OOM site. With the tape
    // off, only the small `[1, kl_range, vocab]` slice is materialized. The
    // teacher runs through the infer runtime (full logits — the windowed
    // `forward_logits_window_device` is only supported by the in-process
    // Qwen35 teacher), so we then free the full-prefix teacher logits: with the
    // tape disabled it has no backward dependency, reclaiming ~142 MiB to widen
    // headroom for the student backward.
    let phase_started = Instant::now();
    let starts = [0, kl_range.start, 0];
    let ends = [1, kl_range.end, vocab];
    tape.set_enabled(false);
    let teacher_kl =
        slice(teacher_logits.tensor_id, &starts, &ends, store, tape).map_err(OpdError::from)?;
    store
        .free(teacher_logits.tensor_id)
        .map_err(OpdError::from)?;

    // OPD engine time-share: the teacher's KL target is now materialized in the
    // train store and the rollout is complete, so BOTH idle infer engines
    // (rollout student + scoring teacher) can be offloaded to host RAM to free
    // VRAM for the student backward. They are offloaded together here — on a
    // device quiesced after teacher scoring — rather than the student earlier,
    // to avoid racing the teacher forward's allocations against the student's
    // async pool frees. Reloaded at the next step (student before rollout,
    // teacher before scoring).
    //
    // Fence the train backend before the infer-thread offload frees the
    // pool blocks, same cross-context ordering reason as the reload fence.
    if engine_offload.is_enabled() {
        store
            .backend()
            .device_synchronize()
            .map_err(OpdError::from)?;
    }
    if engine_offload.offloads_student()
        && let Some(offload_student) = student_engine_offload
    {
        let freed = offload_student()?;
        eprintln!(
            "opd_engine_offload student_offloaded freed_bytes={freed} freed_mib={:.1}",
            freed as f64 / (1024.0 * 1024.0)
        );
    }
    if engine_offload.offloads_teacher() {
        let freed = teacher
            .offload_engine_weights()
            .map_err(|err| map_teacher_forward_error("teacher offload", err))?;
        eprintln!(
            "opd_engine_offload teacher_offloaded freed_bytes={freed} freed_mib={:.1}",
            freed as f64 / (1024.0 * 1024.0)
        );
    }

    tape.set_enabled(true);
    let student_phase_started = Instant::now();
    // VRAM-fit lever (2026-05-29): run the student lm_head over the KL window
    // ONLY, not the full scored prefix. The hidden-state forward still covers
    // `[0..seq_end]` (causal attention needs the full prefix), but the
    // vocab-wide lm_head projection — and its backward grad buffers — are the
    // dominant transient at seq×vocab=144×248320. `forward_logits_window`
    // slices the cheap 1024-wide hidden to the window then projects, so the
    // full `[1, seq_end, vocab]` student logits tensor (≈142 MiB) and the
    // `slice_bwd` grad buffer of the same size never materialize. Numerically
    // identical: causal logits at position p are independent of tokens after p,
    // so windowed lm_head == full lm_head sliced.
    let kl_window = SequenceWindow {
        start: kl_range.start,
        end: kl_range.end,
    };
    let student_kl = student
        .forward_logits_window(store, tape, prefix, &positions, kl_window)
        .map_err(|err| map_qwen35_forward_error("student chunk KL", err))?;
    record_profile(profile, |profile| {
        profile.student_forward_seconds += student_phase_started.elapsed().as_secs_f64();
    });

    let loss = if let Some(beta) = kl_beta {
        generalized_beta_jsd_loss_chunked(
            student_kl,
            teacher_kl,
            kl_range.len(),
            chunk_size,
            kl_temperature,
            beta,
            store,
            tape,
        )
    } else {
        kl_distill_loss_chunked(
            student_kl,
            teacher_kl,
            kl_range.len(),
            chunk_size,
            kl_temperature,
            kl_direction,
            store,
            tape,
        )
    }
    .map_err(OpdError::from)?;
    let loss_value = store.to_host(loss).map_err(OpdError::from)?[0];
    record_profile(profile, |profile| {
        profile.kl_loss_seconds += phase_started.elapsed().as_secs_f64();
    });

    let phase_started = Instant::now();
    backward_with_optional_profile(loss, loss_value, store, tape)?;
    record_profile(profile, |profile| {
        profile.backward_seconds += phase_started.elapsed().as_secs_f64();
    });
    cleanup_after_backward(store, tape, student_model_params, keep_extra);

    // OPD engine time-share: the heavy student-backward transients are now
    // freed, so reload the engines we offloaded BEFORE returning. This keeps
    // the offload window strictly inside the backward and leaves both engines
    // resident for whatever the caller does next — the inter-step KL eval and
    // checkpoint save both run a teacher/student forward, which would hit an
    // offloaded (placeholder) weight and fail (W4A8 "missing Marlin-packed
    // side buffer"). Reload is idempotent, so the next step's pre-rollout /
    // pre-scoring reloads become no-ops. Fence the train backend first so the
    // backward's pool ops are ordered ahead of the reload's allocations.
    if engine_offload.is_enabled() {
        store
            .backend()
            .device_synchronize()
            .map_err(OpdError::from)?;
    }
    if engine_offload.offloads_teacher() {
        teacher
            .reload_engine_weights()
            .map_err(|err| map_teacher_forward_error("teacher reload (post-backward)", err))?;
    }
    if engine_offload.offloads_student()
        && let Some(reload_student) = student_engine_reload
    {
        reload_student()?;
    }

    Ok(loss_value)
}

fn backward_weighted_window_loss(
    loss: TensorId,
    weight: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
    profile: &mut Option<&mut OpdStepProfile>,
) -> Result<f32> {
    if !weight.is_finite() || weight < 0.0 {
        return Err(OpdError::InvalidInput(format!(
            "OPD window loss weight must be finite and non-negative, got {weight}. \
             Hint: verify lambda and window/target counts before Route B backward."
        )));
    }
    let loss_started = Instant::now();
    let weighted_loss = if (weight - 1.0).abs() < f32::EPSILON {
        loss
    } else {
        mul_scalar(loss, weight, store, tape)?
    };
    let loss_value = store.to_host(weighted_loss)?[0];
    validate_loss_value(loss_value)?;
    record_profile(profile, |profile| {
        profile.kl_loss_seconds += loss_started.elapsed().as_secs_f64();
    });

    let phase_started = Instant::now();
    backward_with_optional_profile(weighted_loss, loss_value, store, tape)?;
    record_profile(profile, |profile| {
        profile.backward_seconds += phase_started.elapsed().as_secs_f64();
    });
    Ok(loss_value)
}

fn cleanup_window_backward_retaining_base(
    store: &mut TensorStore,
    tape: &mut Tape,
    base_tape_len: usize,
    base_keep: &HashSet<TensorId>,
    student_model_params: &[TensorId],
    keep_extra: &HashSet<TensorId>,
    keep_grads: &[TensorId],
) {
    tape.entries.truncate(base_tape_len);
    tape.set_enabled(true);
    let mut keep = base_keep.clone();
    keep.extend(keep_extra.iter().copied());
    for &grad_id in keep_grads {
        keep.insert(grad_id);
    }
    extend_keep_with_params_and_grads(&mut keep, student_model_params.iter().copied(), store);
    store.retain_ids(&keep);
}

fn accumulate_tensor_grad(
    store: &mut TensorStore,
    existing_grad_id: TensorId,
    incoming_grad_id: TensorId,
) -> Result<()> {
    let expected = store
        .get(existing_grad_id)
        .ok_or(AutogradError::InvalidTensorId(existing_grad_id))?
        .shape
        .clone();
    let incoming = store
        .get(incoming_grad_id)
        .ok_or(AutogradError::InvalidTensorId(incoming_grad_id))?
        .shape
        .clone();
    if expected != incoming {
        return Err(AutogradError::GradientShapeMismatch {
            tensor_id: existing_grad_id,
            expected,
            got: incoming,
        }
        .into());
    }

    let both_on_device = {
        let existing = store
            .get(existing_grad_id)
            .ok_or(AutogradError::InvalidTensorId(existing_grad_id))?;
        let incoming = store
            .get(incoming_grad_id)
            .ok_or(AutogradError::InvalidTensorId(incoming_grad_id))?;
        existing.dirty != Dirty::Host
            && existing.device_handle.is_some()
            && incoming.dirty != Dirty::Host
            && incoming.device_handle.is_some()
    };
    if both_on_device {
        let existing_handle = store
            .get(existing_grad_id)
            .and_then(|tensor| tensor.device_handle.as_ref())
            .ok_or(AutogradError::InvalidTensorId(existing_grad_id))?
            .clone();
        let incoming_handle = store
            .get(incoming_grad_id)
            .and_then(|tensor| tensor.device_handle.as_ref())
            .ok_or(AutogradError::InvalidTensorId(incoming_grad_id))?
            .clone();
        let sum_handle =
            store
                .backend()
                .add_into_device(&existing_handle, &incoming_handle, &incoming)?;
        store.replace_device_handle(existing_grad_id, sum_handle)?;
        return Ok(());
    }

    let incoming_data = store.to_host(incoming_grad_id)?;
    let existing = store
        .get_mut(existing_grad_id)
        .ok_or(AutogradError::InvalidTensorId(existing_grad_id))?;
    for (dst, src) in existing.data.iter_mut().zip(incoming_data) {
        *dst += src;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn backward_windowed_pure_kl_cached_student_hidden<T: TeacherForward + ?Sized>(
    student: &Qwen35Model,
    teacher: &T,
    prompt_ids: &[u32],
    rollout: &[u32],
    positions: &[u32],
    vocab: usize,
    gkd_config: GkdLossConfig<'_>,
    window_size: usize,
    student_target_params: &[TensorId],
    student_model_params: &[TensorId],
    keep_extra: &HashSet<TensorId>,
    store: &mut TensorStore,
    tape: &mut Tape,
    engine_offload: EngineOffloadMode,
    profile: &mut Option<&mut OpdStepProfile>,
) -> Result<f32> {
    let kl_range = kl_logit_range(gkd_config.kl_mask, prompt_ids.len(), rollout.len())?;
    let windows = sequence_windows_for_range(kl_range, window_size)?;
    if windows.is_empty() {
        return Err(OpdError::InvalidInput(
            "OPD windowed Route B built zero KL windows. Hint: verify \
             prompt length, rollout length, and --logits-window-size."
                .to_owned(),
        ));
    }

    let mut teacher_tape = Tape::new();
    teacher_tape.set_enabled(false);
    if engine_offload.offloads_teacher() {
        let reload_started = Instant::now();
        log_opd_window_trace("kl", "teacher_reload_start", 0, reload_started, "");
        store
            .backend()
            .device_synchronize()
            .map_err(OpdError::from)?;
        teacher
            .reload_engine_weights()
            .map_err(|err| map_teacher_forward_error("teacher reload (windowed KL)", err))?;
        log_opd_window_trace("kl", "teacher_reload_done", 0, reload_started, "");
    }
    let teacher_started = Instant::now();
    log_opd_window_trace(
        "kl",
        "teacher_full_forward_start",
        0,
        teacher_started,
        format!("seq_len={}", rollout.len()),
    );
    let teacher_full_logits = teacher
        .forward_logits_device(rollout, positions, store, &mut teacher_tape)
        .map_err(|err| map_teacher_forward_error("teacher full KL cache", err))?;
    log_opd_window_trace(
        "kl",
        "teacher_full_forward_done",
        0,
        teacher_started,
        format!("shape={:?}", teacher_full_logits.shape),
    );
    let expected_teacher_shape = vec![1, rollout.len(), vocab];
    if teacher_full_logits.shape != expected_teacher_shape {
        return Err(OpdError::InvalidInput(format!(
            "OPD cached teacher logits shape mismatch: got {:?}, expected {:?}. \
             Hint: the TeacherForward implementation must return full \
             [batch=1, seq_len, vocab] logits when the windowed pure-KL path \
             caches teacher logits for reuse.",
            teacher_full_logits.shape, expected_teacher_shape
        )));
    }
    record_profile(profile, |profile| {
        profile.teacher_forward_seconds += teacher_started.elapsed().as_secs_f64();
    });

    if engine_offload.offloads_teacher() {
        store
            .backend()
            .device_synchronize()
            .map_err(OpdError::from)?;
        let freed = teacher
            .offload_engine_weights()
            .map_err(|err| map_teacher_forward_error("teacher offload (windowed KL)", err))?;
        eprintln!(
            "opd_engine_offload teacher_offloaded freed_bytes={freed} freed_mib={:.1}",
            freed as f64 / (1024.0 * 1024.0)
        );
    }

    tape.entries.clear();
    tape.set_enabled(true);
    let student_started = Instant::now();
    log_opd_window_trace(
        "kl",
        "student_hidden_forward_start",
        0,
        student_started,
        format!("seq_len={}", rollout.len()),
    );
    let student_hidden = student
        .forward_hidden_states(
            store,
            tape,
            rollout,
            positions,
            crate::context_parallel::CpContext::single(),
        )
        .map_err(|err| map_qwen35_forward_error("student windowed KL hidden", err))?;
    log_opd_window_trace(
        "kl",
        "student_hidden_forward_done",
        0,
        student_started,
        format!("tensor_id={student_hidden}"),
    );
    record_profile(profile, |profile| {
        profile.student_forward_seconds += student_started.elapsed().as_secs_f64();
    });

    let base_tape_len = tape.entries.len();
    let base_keep = store.live_ids().into_iter().collect::<HashSet<_>>();

    let mut total_loss = 0.0f32;
    let mut hidden_grad: Option<TensorId> = None;
    for (window_offset, window) in windows.into_iter().enumerate() {
        let window_index = window_offset + 1;
        let window_started = Instant::now();
        log_opd_window_trace(
            "kl",
            "start",
            window_index,
            window_started,
            format!(
                "start={} end={} len={} cached_hidden=true",
                window.start,
                window.end,
                window.len()
            ),
        );
        tape.entries.truncate(base_tape_len);
        tape.set_enabled(true);
        let mut window_tape = Tape::new();
        window_tape.set_enabled(false);

        let phase_started = Instant::now();
        log_opd_window_trace(
            "kl",
            "teacher_forward_start",
            window_index,
            window_started,
            "",
        );
        let teacher_tensor_id = slice(
            teacher_full_logits.tensor_id,
            &[0, window.start, 0],
            &[1, window.end, vocab],
            store,
            &mut window_tape,
        )?;
        let teacher_shape = vec![1, window.len(), vocab];
        log_opd_window_trace(
            "kl",
            "teacher_forward_done",
            window_index,
            window_started,
            format!("shape={teacher_shape:?} cached_full=true"),
        );
        record_profile(profile, |profile| {
            profile.teacher_forward_seconds += phase_started.elapsed().as_secs_f64();
        });
        validate_logits_shape("teacher windowed KL", &teacher_shape, window.len(), vocab)?;

        window_tape.set_enabled(true);
        let phase_started = Instant::now();
        let kl_loss = if gkd_config.kl_beta.is_some() || !gkd_config.fused_distill {
            let logits_started = Instant::now();
            log_opd_window_trace(
                "kl",
                "student_logits_window_start",
                window_index,
                window_started,
                format!(
                    "kl_beta={} fused_distill={}",
                    gkd_config.kl_beta.is_some(),
                    gkd_config.fused_distill
                ),
            );
            let student_logits = student
                .logits_from_hidden_window(store, &mut window_tape, student_hidden, window)
                .map_err(|err| map_qwen35_forward_error("student cached-hidden KL beta", err))?;
            record_profile(profile, |profile| {
                profile.student_forward_seconds += logits_started.elapsed().as_secs_f64();
            });
            let loss_started = Instant::now();
            let loss = kl_distill_loss_for_config(
                student_logits,
                teacher_tensor_id,
                window.len(),
                gkd_config.kl_chunk_size,
                gkd_config.kl_direction,
                gkd_config.kl_temperature,
                gkd_config.kl_beta,
                store,
                &mut window_tape,
            )?;
            record_profile(profile, |profile| {
                profile.kl_loss_seconds += loss_started.elapsed().as_secs_f64();
            });
            loss
        } else {
            log_opd_window_trace(
                "kl",
                "fused_linear_distill_start",
                window_index,
                window_started,
                format!(
                    "hidden_tensor_id={student_hidden} lm_head_tensor_id={}",
                    student.lm_head_weight_id()
                ),
            );
            // Opt-in fused path (--fused-distill). NOT default: H20 A/B measured it
            // ~205 s/step (lm_head on HOST, GPU 0%) vs ~3.9 s dense (GPU 98%) on the
            // 27B; loss matches dense to ~5e-4. Only worth it for windows too large to
            // materialize [window, vocab]. errors/2026-06-23-opd-fused-distill-default-host-bound.
            let loss = fused_linear_distill_loss(
                student_hidden,
                student.lm_head_weight_id(),
                teacher_tensor_id,
                window.start,
                window.len(),
                gkd_config.kl_chunk_size.unwrap_or(window.len()),
                gkd_config.kl_temperature,
                gkd_config.kl_direction,
                store,
                &mut window_tape,
            )?;
            record_profile(profile, |profile| {
                profile.kl_loss_seconds += phase_started.elapsed().as_secs_f64();
            });
            loss
        };
        log_opd_window_trace(
            "kl",
            "kl_loss_done",
            window_index,
            window_started,
            format!("tensor_id={kl_loss}"),
        );
        let weight = window.len() as f32 / kl_range.len() as f32;
        log_opd_window_trace("kl", "backward_start", window_index, window_started, "");
        let loss_started = Instant::now();
        let weighted_loss = if (weight - 1.0).abs() < f32::EPSILON {
            kl_loss
        } else {
            mul_scalar(kl_loss, weight, store, &mut window_tape)?
        };
        let loss_value = store.to_host(weighted_loss)?[0];
        validate_loss_value(loss_value)?;
        total_loss += loss_value;
        record_profile(profile, |profile| {
            profile.kl_loss_seconds += loss_started.elapsed().as_secs_f64();
        });

        let phase_started = Instant::now();
        let window_grads = window_tape.backward_collect(weighted_loss, store)?;
        let window_hidden_grad = *window_grads
            .get(&student_hidden)
            .ok_or(AutogradError::MissingGradient(student_hidden))?;
        for &target_id in student_target_params {
            if let Some(&grad_id) = window_grads.get(&target_id) {
                store.accumulate_grad(target_id, grad_id)?;
            }
        }
        let keep_grads: Vec<_> = hidden_grad
            .into_iter()
            .chain([window_hidden_grad])
            .collect();
        cleanup_window_backward_retaining_base(
            store,
            tape,
            base_tape_len,
            &base_keep,
            student_model_params,
            keep_extra,
            &keep_grads,
        );
        if let Some(existing_hidden_grad) = hidden_grad {
            accumulate_tensor_grad(store, existing_hidden_grad, window_hidden_grad)?;
        } else {
            hidden_grad = Some(window_hidden_grad);
        }
        record_profile(profile, |profile| {
            profile.backward_seconds += phase_started.elapsed().as_secs_f64();
        });
        log_opd_window_trace(
            "kl",
            "backward_done",
            window_index,
            window_started,
            format!("loss_accum={total_loss:.12e}"),
        );
        log_opd_window_trace("kl", "cleanup_start", window_index, window_started, "");
        cleanup_window_backward_retaining_base(
            store,
            tape,
            base_tape_len,
            &base_keep,
            student_model_params,
            keep_extra,
            hidden_grad.as_slice(),
        );
        log_opd_window_trace("kl", "done", window_index, window_started, "");
    }

    store
        .free(teacher_full_logits.tensor_id)
        .map_err(OpdError::from)?;
    let hidden_grad = hidden_grad.ok_or(AutogradError::MissingGradient(student_hidden))?;
    let base_backward_started = Instant::now();
    log_opd_window_trace(
        "kl",
        "base_backward_start",
        0,
        base_backward_started,
        format!("seed_grad_id={hidden_grad}"),
    );
    tape.entries.truncate(base_tape_len);
    tape.set_enabled(false);
    if opd_backward_profile_enabled() {
        let (_, backward_profile) = tape.backward_from_seed_accumulate_targets_profiled(
            student_hidden,
            hidden_grad,
            store,
            student_target_params,
        )?;
        print_backward_profile("base", 0, total_loss, &backward_profile);
    } else {
        tape.backward_from_seed_accumulate_targets(
            student_hidden,
            hidden_grad,
            store,
            student_target_params,
        )?;
    }
    log_opd_window_trace("kl", "base_backward_done", 0, base_backward_started, "");
    record_profile(profile, |profile| {
        profile.backward_seconds += base_backward_started.elapsed().as_secs_f64();
    });
    cleanup_after_backward(store, tape, student_model_params, keep_extra);

    Ok(total_loss)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn backward_windowed_gkd_loss<T: TeacherForward + ?Sized>(
    student: &Qwen35Model,
    teacher: &T,
    prompt_ids: &[u32],
    rollout: &[u32],
    positions: &[u32],
    vocab: usize,
    gkd_config: GkdLossConfig<'_>,
    window_size: usize,
    student_target_params: &[TensorId],
    student_model_params: &[TensorId],
    keep_extra: &HashSet<TensorId>,
    store: &mut TensorStore,
    tape: &mut Tape,
    engine_offload: EngineOffloadMode,
    profile: &mut Option<&mut OpdStepProfile>,
) -> Result<f32> {
    let mut total_loss = 0.0f32;
    let mut backward_windows = 0usize;

    if gkd_config.lambda == 0.0 {
        return backward_windowed_pure_kl_cached_student_hidden(
            student,
            teacher,
            prompt_ids,
            rollout,
            positions,
            vocab,
            gkd_config,
            window_size,
            student_target_params,
            student_model_params,
            keep_extra,
            store,
            tape,
            engine_offload,
            profile,
        );
    }

    if gkd_config.lambda < 1.0 {
        let kl_range = kl_logit_range(gkd_config.kl_mask, prompt_ids.len(), rollout.len())?;
        for window in sequence_windows_for_range(kl_range, window_size)? {
            let window_index = backward_windows + 1;
            let window_started = Instant::now();
            log_opd_window_trace(
                "kl",
                "start",
                window_index,
                window_started,
                format!(
                    "start={} end={} len={}",
                    window.start,
                    window.end,
                    window.len()
                ),
            );
            tape.entries.clear();
            tape.set_enabled(false);

            let phase_started = Instant::now();
            log_opd_window_trace(
                "kl",
                "teacher_forward_start",
                window_index,
                window_started,
                "",
            );
            let teacher_logits = teacher
                .forward_logits_window_device(rollout, positions, window, store, tape)
                .map_err(|err| map_teacher_forward_error("teacher windowed KL", err))?;
            log_opd_window_trace(
                "kl",
                "teacher_forward_done",
                window_index,
                window_started,
                format!("shape={:?}", teacher_logits.shape),
            );
            record_profile(profile, |profile| {
                profile.teacher_forward_seconds += phase_started.elapsed().as_secs_f64();
            });
            validate_logits_shape(
                "teacher windowed KL",
                &teacher_logits.shape,
                window.len(),
                vocab,
            )?;

            tape.set_enabled(true);
            let phase_started = Instant::now();
            log_opd_window_trace(
                "kl",
                "student_forward_start",
                window_index,
                window_started,
                "",
            );
            let student_logits = student
                .forward_logits_window(store, tape, rollout, positions, window)
                .map_err(|err| map_qwen35_forward_error("student windowed KL", err))?;
            log_opd_window_trace(
                "kl",
                "student_forward_done",
                window_index,
                window_started,
                format!("tensor_id={student_logits}"),
            );
            let student_shape = store
                .get(student_logits)
                .ok_or(AutogradError::InvalidTensorId(student_logits))?
                .shape
                .clone();
            validate_logits_shape("student windowed KL", &student_shape, window.len(), vocab)?;
            record_profile(profile, |profile| {
                profile.student_forward_seconds += phase_started.elapsed().as_secs_f64();
            });

            let phase_started = Instant::now();
            log_opd_window_trace("kl", "kl_loss_start", window_index, window_started, "");
            let kl_loss = kl_distill_loss_for_config(
                student_logits,
                teacher_logits.tensor_id,
                window.len(),
                gkd_config.kl_chunk_size,
                gkd_config.kl_direction,
                gkd_config.kl_temperature,
                gkd_config.kl_beta,
                store,
                tape,
            )?;
            log_opd_window_trace(
                "kl",
                "kl_loss_done",
                window_index,
                window_started,
                format!("tensor_id={kl_loss}"),
            );
            record_profile(profile, |profile| {
                profile.kl_loss_seconds += phase_started.elapsed().as_secs_f64();
            });
            let weight = (1.0 - gkd_config.lambda) * (window.len() as f32 / kl_range.len() as f32);
            log_opd_window_trace("kl", "backward_start", window_index, window_started, "");
            total_loss += backward_weighted_window_loss(kl_loss, weight, store, tape, profile)?;
            log_opd_window_trace(
                "kl",
                "backward_done",
                window_index,
                window_started,
                format!("loss_accum={total_loss:.12e}"),
            );
            backward_windows += 1;
            log_opd_window_trace("kl", "cleanup_start", window_index, window_started, "");
            cleanup_after_backward(store, tape, student_model_params, keep_extra);
            log_opd_window_trace("kl", "done", window_index, window_started, "");
        }
    }

    if gkd_config.lambda > 0.0 {
        match gkd_config.sft_anchor {
            GkdSftAnchor::StudentRollout => {
                let target_count = rollout.len().checked_sub(1).ok_or_else(|| {
                    OpdError::InvalidInput(
                        "GKD student-rollout SFT anchor requires at least two rollout \
                         tokens so logits can be trained against next-token labels."
                            .to_owned(),
                    )
                })?;
                if target_count == 0 {
                    return Err(OpdError::InvalidInput(
                        "GKD student-rollout SFT anchor requires at least two rollout \
                         tokens so logits can be trained against next-token labels."
                            .to_owned(),
                    ));
                }
                for target_window in sequence_windows(target_count, window_size)? {
                    tape.entries.clear();
                    tape.set_enabled(true);

                    let logits_window = target_window;
                    let phase_started = Instant::now();
                    let student_logits = student
                        .forward_logits_window(store, tape, rollout, positions, logits_window)
                        .map_err(|err| {
                            map_qwen35_forward_error("student windowed rollout SFT", err)
                        })?;
                    let student_shape = store
                        .get(student_logits)
                        .ok_or(AutogradError::InvalidTensorId(student_logits))?
                        .shape
                        .clone();
                    validate_logits_shape(
                        "student windowed rollout SFT",
                        &student_shape,
                        logits_window.len(),
                        vocab,
                    )?;
                    record_profile(profile, |profile| {
                        profile.student_forward_seconds += phase_started.elapsed().as_secs_f64();
                    });

                    let target_tokens = &rollout[target_window.start + 1..target_window.end + 1];
                    let phase_started = Instant::now();
                    let sft_loss = next_token_sft_loss_from_logits(
                        student_logits,
                        logits_window.len(),
                        0,
                        target_tokens,
                        vocab,
                        store,
                        tape,
                    )?;
                    record_profile(profile, |profile| {
                        profile.kl_loss_seconds += phase_started.elapsed().as_secs_f64();
                    });
                    let weight =
                        gkd_config.lambda * (target_tokens.len() as f32 / target_count as f32);
                    total_loss +=
                        backward_weighted_window_loss(sft_loss, weight, store, tape, profile)?;
                    backward_windows += 1;
                    cleanup_after_backward(store, tape, student_model_params, keep_extra);
                }
            }
            GkdSftAnchor::CorpusTruth => {
                let corpus_tokens = gkd_config.corpus_tokens.ok_or_else(|| {
                    OpdError::InvalidInput(
                        "GKD corpus-truth SFT anchor requires corpus completion \
                         tokens. Hint: add completion/target fields to \
                         --prompts-file."
                            .to_owned(),
                    )
                })?;
                if prompt_ids.is_empty() || corpus_tokens.is_empty() {
                    return Err(OpdError::InvalidInput(
                        "GKD corpus-truth SFT anchor requires non-empty prompt and \
                         completion tokens."
                            .to_owned(),
                    ));
                }
                let total_len = prompt_ids
                    .len()
                    .checked_add(corpus_tokens.len())
                    .ok_or_else(|| {
                        OpdError::InvalidInput(
                            "GKD corpus-truth SFT prompt+completion length overflowed. \
                             Hint: reduce prompt/completion max tokens."
                                .to_owned(),
                        )
                    })?;
                if total_len > u32::MAX as usize {
                    return Err(OpdError::InvalidInput(format!(
                        "GKD corpus-truth SFT sequence length {total_len} exceeds \
                         u32::MAX RoPE position range. Hint: reduce prompt/completion \
                         max tokens."
                    )));
                }
                let sft_sequence: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(corpus_tokens.iter().copied())
                    .collect();
                let sft_positions = (0..total_len as u32).collect::<Vec<_>>();
                for target_window in sequence_windows(corpus_tokens.len(), window_size)? {
                    let logits_window = SequenceWindow {
                        start: prompt_ids.len() - 1 + target_window.start,
                        end: prompt_ids.len() - 1 + target_window.end,
                    };

                    tape.entries.clear();
                    tape.set_enabled(true);
                    let phase_started = Instant::now();
                    let student_logits = student
                        .forward_logits_window(
                            store,
                            tape,
                            &sft_sequence,
                            &sft_positions,
                            logits_window,
                        )
                        .map_err(|err| {
                            map_qwen35_forward_error("student windowed corpus SFT", err)
                        })?;
                    let student_shape = store
                        .get(student_logits)
                        .ok_or(AutogradError::InvalidTensorId(student_logits))?
                        .shape
                        .clone();
                    validate_logits_shape(
                        "student windowed corpus SFT",
                        &student_shape,
                        logits_window.len(),
                        vocab,
                    )?;
                    record_profile(profile, |profile| {
                        profile.student_forward_seconds += phase_started.elapsed().as_secs_f64();
                    });

                    let target_tokens = &corpus_tokens[target_window.start..target_window.end];
                    let phase_started = Instant::now();
                    let sft_loss = next_token_sft_loss_from_logits(
                        student_logits,
                        logits_window.len(),
                        0,
                        target_tokens,
                        vocab,
                        store,
                        tape,
                    )?;
                    record_profile(profile, |profile| {
                        profile.kl_loss_seconds += phase_started.elapsed().as_secs_f64();
                    });
                    let weight = gkd_config.lambda
                        * (target_tokens.len() as f32 / corpus_tokens.len() as f32);
                    total_loss +=
                        backward_weighted_window_loss(sft_loss, weight, store, tape, profile)?;
                    backward_windows += 1;
                    cleanup_after_backward(store, tape, student_model_params, keep_extra);
                }
            }
        }
    }

    if backward_windows == 0 {
        return Err(OpdError::InvalidInput(
            "OPD windowed Route B built zero backward windows. Hint: verify \
             lambda, prompt length, rollout length, and --logits-window-size."
                .to_owned(),
        ));
    }
    Ok(total_loss)
}
