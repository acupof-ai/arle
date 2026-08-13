//! The OPD step driver: validation, rollout, route selection, backward, optimizer step.

use std::{collections::HashSet, time::Instant};

use autograd::{Tape, TensorId, TensorStore, optim::Optimizer};

use crate::{
    grad_clip::finite_optimizer_step,
    qwen35::Qwen35Model,
    teacher_infer::{InProcessTeacher, TeacherForward},
    trainer::{cleanup_after_backward, retained_param_and_grad_ids},
};

use super::{
    EngineOffloadMode, GkdLossConfig, OpdError, OpdStepConfig, OpdStepOutcome, OpdStepProfile,
    Result,
    backward::{backward_chunked_kl_rollout, backward_windowed_gkd_loss},
    backward_with_optional_profile, log_opd_step_trace,
    loss::{gkd_sft_loss, kl_distill_loss_for_config, mix_gkd_losses},
    map_qwen35_forward_error, map_teacher_forward_error, record_profile,
    rollout::run_opd_rollout_phase,
    validation::{
        validate_gkd_loss_config, validate_loss_value, validate_rollout_shape,
        validate_step_config, validate_student_param_ownership, validate_student_params,
        validate_teacher_params, validate_token_ids,
    },
    windowing::{kl_logit_range, slice_logits_for_kl},
};
#[cfg(feature = "cuda")]
use super::{InferRolloutCtx, engine_offload_mode};

pub fn opd_step<O: Optimizer>(
    student: &Qwen35Model,
    teacher: &Qwen35Model,
    prompt_ids: &[u32],
    cfg: OpdStepConfig,
    student_params: &[TensorId],
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
    forced_rollout: Option<&[u32]>,
) -> Result<OpdStepOutcome> {
    let teacher = InProcessTeacher::new(teacher);
    opd_step_with_teacher(
        OpdStepInputs::new(student, &teacher, prompt_ids, cfg, student_params)
            .forced_rollout(forced_rollout),
        optimizer,
        store,
        tape,
        None,
    )
}

/// Holds the `&`/Copy fields common to both GKD backward routes;
/// `&mut` store/tape/optimizer/profile stay explicit.
struct GkdRouteCtx<'a, T: TeacherForward + ?Sized> {
    student: &'a Qwen35Model,
    teacher: &'a T,
    prompt_ids: &'a [u32],
    cfg: &'a OpdStepConfig,
    gkd_config: GkdLossConfig<'a>,
    student_params: &'a [TensorId],
    student_model_params: &'a [TensorId],
    keep_extra: &'a HashSet<TensorId>,
    engine_offload: EngineOffloadMode,
    total_started: Instant,
    vocab: usize,
    rollout: &'a [u32],
    positions: &'a [u32],
    #[cfg(feature = "cuda")]
    infer_rollout: Option<&'a InferRolloutCtx<'a>>,
}

/// Route B — windowed scoring/backward: the memory path for long rollouts /
/// large vocabs. `kl_chunk_size` chunks softmax inside each window.
fn run_windowed_gkd_route<O: Optimizer, T: TeacherForward + ?Sized>(
    rt: &GkdRouteCtx<'_, T>,
    window_size: usize,
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
    profile: &mut Option<&mut OpdStepProfile>,
) -> Result<OpdStepOutcome> {
    log_opd_step_trace(
        rt.total_started,
        "windowed_route_start",
        format!("window_size={window_size}"),
    );
    let phase_started = Instant::now();
    optimizer.zero_grad(store, rt.student_params);
    record_profile(profile, |profile| {
        profile.optimizer_zero_grad_seconds += phase_started.elapsed().as_secs_f64();
    });

    #[cfg(feature = "cuda")]
    let student_engine_offloaded = if rt.engine_offload.offloads_student() {
        if let Some(ctx) = rt.infer_rollout {
            log_opd_step_trace(rt.total_started, "infer_rollout_offload_start", "");
            store
                .backend()
                .device_synchronize()
                .map_err(OpdError::from)?;
            let freed = ctx.student.offload_engine_weights().map_err(|err| {
                OpdError::InvalidInput(format!("infer student offload failed: {err}"))
            })?;
            eprintln!(
                "opd_engine_offload student_offloaded freed_bytes={freed} freed_mib={:.1}",
                freed as f64 / (1024.0 * 1024.0)
            );
            true
        } else {
            false
        }
    } else {
        false
    };

    log_opd_step_trace(rt.total_started, "windowed_backward_start", "");
    let loss_result = backward_windowed_gkd_loss(
        rt.student,
        rt.teacher,
        rt.prompt_ids,
        rt.rollout,
        rt.positions,
        rt.vocab,
        rt.gkd_config,
        window_size,
        rt.student_params,
        rt.student_model_params,
        rt.keep_extra,
        store,
        tape,
        rt.engine_offload,
        profile,
    );
    log_opd_step_trace(rt.total_started, "windowed_backward_done", "");

    #[cfg(feature = "cuda")]
    if student_engine_offloaded {
        // Keep the rollout engine offloaded until the next rollout.
        // Reloading here happens before `cleanup_after_backward` prunes
        // the long-sequence autograd tape and can OOM on 2K-token OPD
        // steps. The next-step reload path above runs after cleanup and
        // before LoRA sync/generation, which is the first point where the
        // infer student must be resident again.
        log_opd_step_trace(
            rt.total_started,
            "infer_rollout_reload_deferred_until_next_step",
            "",
        );
    }

    let loss_value = loss_result?;

    let phase_started = Instant::now();
    finite_optimizer_step(
        loss_value,
        rt.student_params,
        rt.cfg.grad_clip,
        optimizer,
        store,
    )?;
    record_profile(profile, |profile| {
        profile.grad_clip_seconds += phase_started.elapsed().as_secs_f64();
    });
    log_opd_step_trace(rt.total_started, "optimizer_step_done", "");

    Ok(OpdStepOutcome {
        loss: loss_value,
        rollout_len: rt.rollout.len(),
    })
}

fn run_chunked_kl_route<O: Optimizer, T: TeacherForward + ?Sized>(
    rt: &GkdRouteCtx<'_, T>,
    chunk_size: usize,
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
    profile: &mut Option<&mut OpdStepProfile>,
) -> Result<OpdStepOutcome> {
    let phase_started = Instant::now();
    optimizer.zero_grad(store, rt.student_params);
    record_profile(profile, |profile| {
        profile.optimizer_zero_grad_seconds += phase_started.elapsed().as_secs_f64();
    });

    // Build the rollout-student offload/reload hooks (cuda-only). The
    // offload is invoked inside the backward right after the teacher
    // scores (idle engines offloaded on a quiesced device); the reload
    // runs at the end of the backward so the student is resident again
    // for the inter-step eval / checkpoint / next rollout.
    #[cfg(feature = "cuda")]
    let student_offload_fn: Option<Box<dyn Fn() -> Result<usize>>> =
        if rt.engine_offload.offloads_student() {
            rt.infer_rollout.map(|ctx| {
                let student = ctx.student;
                Box::new(move || {
                    student.offload_engine_weights().map_err(|err| {
                        OpdError::InvalidInput(format!("infer student offload failed: {err}"))
                    })
                }) as Box<dyn Fn() -> Result<usize>>
            })
        } else {
            None
        };
    #[cfg(not(feature = "cuda"))]
    let student_offload_fn: Option<Box<dyn Fn() -> Result<usize>>> = None;

    #[cfg(feature = "cuda")]
    let student_reload_fn: Option<Box<dyn Fn() -> Result<()>>> =
        if rt.engine_offload.offloads_student() {
            rt.infer_rollout.map(|ctx| {
                let student = ctx.student;
                Box::new(move || {
                    student.reload_engine_weights().map_err(|err| {
                        OpdError::InvalidInput(format!(
                            "infer student reload (post-backward) failed: {err}"
                        ))
                    })
                }) as Box<dyn Fn() -> Result<()>>
            })
        } else {
            None
        };
    #[cfg(not(feature = "cuda"))]
    let student_reload_fn: Option<Box<dyn Fn() -> Result<()>>> = None;

    let loss_value = backward_chunked_kl_rollout(
        rt.student,
        rt.teacher,
        rt.rollout,
        rt.prompt_ids.len(),
        rt.vocab,
        chunk_size,
        rt.gkd_config.kl_mask,
        rt.gkd_config.kl_direction,
        rt.gkd_config.kl_temperature,
        rt.gkd_config.kl_beta,
        rt.student_model_params,
        rt.keep_extra,
        store,
        tape,
        profile,
        rt.engine_offload,
        student_offload_fn.as_deref(),
        student_reload_fn.as_deref(),
    )?;

    let phase_started = Instant::now();
    finite_optimizer_step(
        loss_value,
        rt.student_params,
        rt.cfg.grad_clip,
        optimizer,
        store,
    )?;
    record_profile(profile, |profile| {
        profile.grad_clip_seconds += phase_started.elapsed().as_secs_f64();
    });

    Ok(OpdStepOutcome {
        loss: loss_value,
        rollout_len: rt.rollout.len(),
    })
}

/// Participants, step inputs and objective config for one OPD step. Optional
/// fields carry their defaults from `new`; override them with the setters.
pub struct OpdStepInputs<'a, T: TeacherForward + ?Sized> {
    pub student: &'a Qwen35Model,
    pub teacher: &'a T,
    pub prompt_ids: &'a [u32],
    pub cfg: OpdStepConfig,
    pub student_params: &'a [TensorId],
    pub gkd: GkdLossConfig<'a>,
    pub forced_rollout: Option<&'a [u32]>,
    #[cfg(feature = "cuda")]
    pub infer_rollout: Option<InferRolloutCtx<'a>>,
}

impl<'a, T: TeacherForward + ?Sized> OpdStepInputs<'a, T> {
    pub fn new(
        student: &'a Qwen35Model,
        teacher: &'a T,
        prompt_ids: &'a [u32],
        cfg: OpdStepConfig,
        student_params: &'a [TensorId],
    ) -> Self {
        Self {
            student,
            teacher,
            prompt_ids,
            cfg,
            student_params,
            gkd: GkdLossConfig::default(),
            forced_rollout: None,
            #[cfg(feature = "cuda")]
            infer_rollout: None,
        }
    }

    pub fn gkd(mut self, gkd: GkdLossConfig<'a>) -> Self {
        self.gkd = gkd;
        self
    }

    pub fn forced_rollout(mut self, forced_rollout: Option<&'a [u32]>) -> Self {
        self.forced_rollout = forced_rollout;
        self
    }

    #[cfg(feature = "cuda")]
    pub fn infer_rollout(mut self, infer_rollout: Option<InferRolloutCtx<'a>>) -> Self {
        self.infer_rollout = infer_rollout;
        self
    }
}

pub fn opd_step_with_teacher<O: Optimizer, T: TeacherForward + ?Sized>(
    inputs: OpdStepInputs<'_, T>,
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
    profile: Option<&mut OpdStepProfile>,
) -> Result<OpdStepOutcome> {
    let student = inputs.student;
    let teacher = inputs.teacher;
    let prompt_ids = inputs.prompt_ids;
    let student_params = inputs.student_params;
    let gkd_config = inputs.gkd;
    let forced_rollout = inputs.forced_rollout;
    let cfg = inputs.cfg;
    #[cfg(feature = "cuda")]
    let infer_rollout = inputs.infer_rollout;
    let mut profile = profile;
    if let Some(profile) = profile.as_deref_mut() {
        *profile = OpdStepProfile::default();
    }
    let total_started = Instant::now();
    log_opd_step_trace(
        total_started,
        "start",
        format!(
            "prompt_len={} rollout_len={}",
            prompt_ids.len(),
            cfg.rollout_len
        ),
    );
    validate_step_config(&cfg)?;
    validate_gkd_loss_config(gkd_config)?;
    if let Some(teacher_topk) = gkd_config.teacher_topk {
        return Err(OpdError::InvalidInput(format!(
            "OPD teacher_topk={teacher_topk} requires Piece A engine-side \
             top-k teacher targets on H20/CUDA. Local Piece C only wires the \
             config gate; omit --teacher-topk to keep the dense full-vocab path."
        )));
    }
    let vocab = student.config().vocab_size;
    if prompt_ids.is_empty() {
        return Err(OpdError::InvalidInput(
            "OPD step requires a non-empty prompt_ids slice. Hint: pass at least \
             one BOS/chat token; the OPD substrate does not synthesize prompts."
                .to_owned(),
        ));
    }
    if vocab == 0 {
        return Err(OpdError::InvalidInput(
            "OPD step requires student.config().vocab_size > 0. Hint: verify the \
             loaded Qwen35Config before constructing the student model."
                .to_owned(),
        ));
    }
    validate_rollout_shape(prompt_ids.len(), cfg.rollout_len, vocab)?;
    let teacher_vocab = teacher.vocab_size();
    if teacher_vocab != vocab {
        return Err(OpdError::InvalidInput(format!(
            "OPD requires teacher/student vocab_size to match, got \
             teacher.vocab_size()={teacher_vocab} and \
             student.config().vocab_size={vocab}. Hint: use model directories \
             that share the same tokenizer before running OPD."
        )));
    }
    validate_token_ids("prompt_ids", prompt_ids, vocab)?;
    let teacher_params = teacher.parameter_ids().to_vec();
    let student_model_params = student.all_parameter_ids();
    if !teacher_params.is_empty() {
        validate_teacher_params(&teacher_params, store)?;
    }
    validate_student_params(student_params, store)?;
    validate_student_param_ownership(student_params, &student_model_params, &teacher_params)?;
    let keep_extra = retained_param_and_grad_ids(&teacher_params, store);
    let mut rollout_keep_base = retained_param_and_grad_ids(&student_model_params, store);
    rollout_keep_base.extend(keep_extra.iter().copied());

    let result = (|| {
        #[cfg(feature = "cuda")]
        let engine_offload = engine_offload_mode();
        #[cfg(not(feature = "cuda"))]
        let engine_offload = EngineOffloadMode::Off;

        let rollout = run_opd_rollout_phase(
            student,
            prompt_ids,
            &cfg,
            vocab,
            forced_rollout,
            &rollout_keep_base,
            engine_offload,
            total_started,
            store,
            tape,
            &mut profile,
            #[cfg(feature = "cuda")]
            infer_rollout.as_ref(),
        )?;

        let positions: Vec<u32> = (0..rollout.len() as u32).collect();
        let rt = GkdRouteCtx {
            student,
            teacher,
            prompt_ids,
            cfg: &cfg,
            gkd_config,
            student_params,
            student_model_params: &student_model_params,
            keep_extra: &keep_extra,
            engine_offload,
            total_started,
            vocab,
            rollout: &rollout,
            positions: &positions,
            #[cfg(feature = "cuda")]
            infer_rollout: infer_rollout.as_ref(),
        };
        if let Some(window_size) = gkd_config.logits_window_size {
            return run_windowed_gkd_route(&rt, window_size, optimizer, store, tape, &mut profile);
        }
        if let Some(chunk_size) = gkd_config.kl_chunk_size
            && gkd_config.lambda == 0.0
        {
            return run_chunked_kl_route(&rt, chunk_size, optimizer, store, tape, &mut profile);
        }

        // Tape is still disabled by the rollout phase here; teacher params are
        // `requires_grad = false` anyway, so this only defends against rogue
        // grad-bearing weights.
        let phase_started = Instant::now();
        let teacher_logits = teacher
            .forward_logits_device(&rollout, &positions, store, tape)
            .map_err(|err| map_teacher_forward_error("teacher scoring", err))?;
        record_profile(&mut profile, |profile| {
            profile.teacher_forward_seconds += phase_started.elapsed().as_secs_f64();
        });
        let expected_teacher_shape = vec![1, rollout.len(), vocab];
        if teacher_logits.shape != expected_teacher_shape {
            return Err(OpdError::InvalidInput(format!(
                "OPD teacher logits shape mismatch: got {:?}, expected {:?}. \
                 Hint: the TeacherForward implementation must return \
                 [batch=1, seq_len, vocab] logits for the exact rollout \
                 scored by the student.",
                teacher_logits.shape, expected_teacher_shape
            )));
        }
        let mut keep_teacher_logits = rollout_keep_base.clone();
        keep_teacher_logits.insert(teacher_logits.tensor_id);
        store.retain_ids(&keep_teacher_logits);

        tape.set_enabled(true);
        let phase_started = Instant::now();
        let student_logits = student
            .forward(store, tape, &rollout, &positions)
            .map_err(|err| map_qwen35_forward_error("student KL", err))?;
        record_profile(&mut profile, |profile| {
            profile.student_forward_seconds += phase_started.elapsed().as_secs_f64();
        });

        let phase_started = Instant::now();
        let kl_range = kl_logit_range(gkd_config.kl_mask, prompt_ids.len(), rollout.len())?;
        let (student_kl_logits, teacher_kl_logits) =
            if kl_range.start == 0 && kl_range.end == rollout.len() {
                (student_logits, teacher_logits.tensor_id)
            } else {
                (
                    slice_logits_for_kl(student_logits, kl_range, vocab, store, tape)?,
                    slice_logits_for_kl(teacher_logits.tensor_id, kl_range, vocab, store, tape)?,
                )
            };
        let kl_loss = kl_distill_loss_for_config(
            student_kl_logits,
            teacher_kl_logits,
            kl_range.len(),
            gkd_config.kl_chunk_size,
            gkd_config.kl_direction,
            gkd_config.kl_temperature,
            gkd_config.kl_beta,
            store,
            tape,
        )?;
        let loss = if gkd_config.lambda == 0.0 {
            kl_loss
        } else {
            let sft_loss = gkd_sft_loss(
                gkd_config,
                student,
                prompt_ids,
                student_logits,
                &rollout,
                vocab,
                store,
                tape,
            )?;
            mix_gkd_losses(kl_loss, sft_loss, gkd_config.lambda, store, tape)?
        };
        let loss_value = store.to_host(loss)?[0];
        validate_loss_value(loss_value)?;
        record_profile(&mut profile, |profile| {
            profile.kl_loss_seconds += phase_started.elapsed().as_secs_f64();
        });

        let phase_started = Instant::now();
        optimizer.zero_grad(store, student_params);
        record_profile(&mut profile, |profile| {
            profile.optimizer_zero_grad_seconds += phase_started.elapsed().as_secs_f64();
        });
        let phase_started = Instant::now();
        backward_with_optional_profile(loss, loss_value, store, tape)?;
        record_profile(&mut profile, |profile| {
            profile.backward_seconds += phase_started.elapsed().as_secs_f64();
        });
        let phase_started = Instant::now();
        finite_optimizer_step(loss_value, student_params, cfg.grad_clip, optimizer, store)?;
        record_profile(&mut profile, |profile| {
            profile.grad_clip_seconds += phase_started.elapsed().as_secs_f64();
        });

        Ok(OpdStepOutcome {
            loss: loss_value,
            rollout_len: rollout.len(),
        })
    })();

    // Prune rollout/teacher/student forward temporaries on both success and
    // failure. Retain the full student model (not just optimizer targets)
    // because LoRA-only OPD needs frozen base weights for the next forward.
    let phase_started = Instant::now();
    cleanup_after_backward(store, tape, &student_model_params, &keep_extra);
    record_profile(&mut profile, |profile| {
        profile.post_step_cleanup_seconds += phase_started.elapsed().as_secs_f64();
        profile.total_seconds = total_started.elapsed().as_secs_f64();
    });
    result
}
