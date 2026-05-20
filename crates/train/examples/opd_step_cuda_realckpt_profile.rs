#![cfg_attr(
    not(all(feature = "cuda", not(feature = "no-cuda"))),
    allow(dead_code, unused_imports)
)]

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
mod app {
    use std::{
        collections::{BTreeMap, HashSet},
        env,
        path::PathBuf,
        sync::Arc,
        time::{Duration, Instant},
    };

    use autograd::{
        backend_cuda::CudaBackend,
        optim::{AdamW, Optimizer},
        tensor::Dirty,
        AutogradError, BackwardProfile, Tape, TensorId, TensorStore,
    };
    use train::{
        grad_clip::clip_grad_norm,
        loss::kl_distill_loss,
        opd::{OpdStepConfig, OpdStepOutcome},
        qwen35::{
            forward_rollout_cached, forward_rollout_cached_device_token, Qwen35KvCache, Qwen35Model,
        },
        qwen35_loader::{load_qwen35_from_hf_dir, load_qwen35_trainable_from_hf_dir},
        trainer::{cleanup_after_backward, retained_param_and_grad_ids},
    };

    const DEFAULT_MODEL_DIR: &str = "/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B";
    const DEFAULT_ROLLOUT_LEN: usize = 8;
    const PROMPT_IDS: &[u32] = &[1, 872, 198, 3456];
    const LR: f32 = 5.0e-5;
    const GRAD_CLIP: f32 = 1.0;
    const PERTURB_SCALE: f32 = 1.0e-3;
    const PERTURB_SEED: u64 = 0x0f0d_cafe_2026_0521;
    const HOST_MIRROR_DROP_MIN_ELEMENTS: usize = 1_000_000;

    type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

    #[derive(Debug, Default, Clone)]
    struct PhaseTotals {
        durations: BTreeMap<&'static str, Duration>,
    }

    impl PhaseTotals {
        fn add(&mut self, phase: &'static str, duration: Duration) {
            *self.durations.entry(phase).or_default() += duration;
        }

        fn seconds(&self, phase: &'static str) -> f64 {
            self.durations
                .get(phase)
                .copied()
                .unwrap_or_default()
                .as_secs_f64()
        }
    }

    fn timed<T>(
        totals: &mut PhaseTotals,
        phase: &'static str,
        f: impl FnOnce() -> AnyResult<T>,
    ) -> AnyResult<T> {
        let started = Instant::now();
        let value = f()?;
        totals.add(phase, started.elapsed());
        Ok(value)
    }

    fn env_flag(name: &str) -> bool {
        env::var_os(name).is_some_and(|value| {
            let value = value.to_string_lossy();
            value == "1" || value.eq_ignore_ascii_case("true")
        })
    }

    pub fn main() -> AnyResult<()> {
        let model_dir = resolve_model_dir()?;
        let rollout_len = resolve_rollout_len()?;
        println!(
            "config backend=cuda model_dir={} prompt={PROMPT_IDS:?} rollout_len={rollout_len} lr={LR} grad_clip={GRAD_CLIP} perturb_scale={PERTURB_SCALE}",
            model_dir.display()
        );

        let cuda_backend = Arc::new(CudaBackend::new(0)?);
        let mut store = TensorStore::with_backend(cuda_backend.clone());
        let mut tape = Tape::new();

        let teacher_load_started = Instant::now();
        let teacher = load_qwen35_from_hf_dir(&model_dir, &mut store)?;
        let teacher_load_seconds = teacher_load_started.elapsed().as_secs_f64();
        let student_load_started = Instant::now();
        let student = load_qwen35_trainable_from_hf_dir(&model_dir, &mut store)?;
        let student_load_seconds = student_load_started.elapsed().as_secs_f64();

        let teacher_params = teacher.all_parameter_ids();
        let student_model_params = student.all_parameter_ids();
        let student_trainable_params = trainable_params(&student, &store);
        let teacher_param_elements = param_element_count(&teacher_params, &store);
        let student_model_elements = param_element_count(&student_model_params, &store);
        let student_trainable_elements = param_element_count(&student_trainable_params, &store);
        perturb_params(
            &student_trainable_params,
            &mut store,
            PERTURB_SEED,
            PERTURB_SCALE,
        );
        let mut optimizer = AdamW::new_with_device(LR, (0.9, 0.999), 1.0e-8, 0.0, cuda_backend);
        let step_cfg = OpdStepConfig {
            rollout_len,
            grad_clip: GRAD_CLIP,
        };

        println!(
            "model config hidden={} intermediate={} layers={} vocab={} num_heads={} num_kv_heads={} head_dim={} tie_word_embeddings={} rope_theta={} teacher_param_elements={} student_model_elements={} student_trainable_elements={} teacher_load_seconds={teacher_load_seconds:.6} student_load_seconds={student_load_seconds:.6}",
            student.config().hidden_size,
            student.config().intermediate_size,
            student.config().num_hidden_layers,
            student.config().vocab_size,
            student.config().num_attention_heads,
            student.config().num_key_value_heads,
            student.config().head_dim,
            student.config().tie_word_embeddings,
            student.config().rope_theta,
            teacher_param_elements,
            student_model_elements,
            student_trainable_elements
        );

        let warmup_started = Instant::now();
        let warmup_loss = warmup_loss_probe(
            &student,
            &teacher,
            PROMPT_IDS,
            step_cfg,
            &student_model_params,
            &mut store,
            &mut tape,
        )?;
        println!(
            "warmup_summary loss={warmup_loss:.12e} seconds={:.6}",
            warmup_started.elapsed().as_secs_f64()
        );
        let (host_rollout, device_rollout) = rollout_equivalence_probe(
            &student,
            PROMPT_IDS,
            rollout_len,
            &teacher_params,
            &student_model_params,
            &mut store,
            &mut tape,
        )?;
        println!(
            "rollout_equivalence host={host_rollout:?} device={device_rollout:?} match={}",
            host_rollout == device_rollout
        );
        if host_rollout != device_rollout {
            return Err("device rollout token sequence differs from host greedy reference".into());
        }
        if env_flag("ARLE_OPD_REALCKPT_PROFILE_DROP_HOST_MIRRORS") {
            let mirror_stats =
                drop_large_host_mirrors(&teacher_params, &student_model_params, &mut store);
            println!(
                "host_mirror_control enabled=true min_elements={} tensors_cleared={} bytes_cleared={} mib_cleared={:.3}",
                HOST_MIRROR_DROP_MIN_ELEMENTS,
                mirror_stats.tensors_cleared,
                mirror_stats.bytes_cleared,
                mirror_stats.bytes_cleared as f64 / (1024.0 * 1024.0)
            );
        } else {
            println!("host_mirror_control enabled=false");
        }

        let (outcome, totals, backward_profile) = profiled_opd_step(
            &student,
            &teacher,
            PROMPT_IDS,
            step_cfg,
            &student_model_params,
            &student_trainable_params,
            &mut optimizer,
            &mut store,
            &mut tape,
        )?;

        print_profile(outcome, &totals, &backward_profile);
        Ok(())
    }

    fn resolve_model_dir() -> AnyResult<PathBuf> {
        let path = env::var_os("ARLE_OPD_REALCKPT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
        if !path.join("config.json").is_file() || !path.join("model.safetensors").is_file() {
            return Err(format!(
                "{} is not a complete Qwen3-0.6B ModelScope checkpoint directory",
                path.display()
            )
            .into());
        }
        Ok(path)
    }

    fn resolve_rollout_len() -> AnyResult<usize> {
        let Some(raw) = env::var_os("ARLE_OPD_REALCKPT_PROFILE_ROLLOUT_LEN") else {
            return Ok(DEFAULT_ROLLOUT_LEN);
        };
        let value: usize = raw
            .to_string_lossy()
            .parse()
            .map_err(|err| format!("invalid ARLE_OPD_REALCKPT_PROFILE_ROLLOUT_LEN: {err}"))?;
        if value == 0 {
            return Err("ARLE_OPD_REALCKPT_PROFILE_ROLLOUT_LEN must be > 0".into());
        }
        Ok(value)
    }

    fn profiled_opd_step<O: Optimizer>(
        student: &Qwen35Model,
        teacher: &Qwen35Model,
        prompt_ids: &[u32],
        cfg: OpdStepConfig,
        student_model_params: &[TensorId],
        student_trainable_params: &[TensorId],
        optimizer: &mut O,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> AnyResult<(OpdStepOutcome, PhaseTotals, BackwardProfile)> {
        let total_started = Instant::now();
        let mut totals = PhaseTotals::default();
        let vocab = student.config().vocab_size;

        let keep_extra = timed(&mut totals, "keep_extra_build", || {
            let teacher_params = teacher.all_parameter_ids();
            Ok(retained_param_and_grad_ids(&teacher_params, store))
        })?;

        timed(&mut totals, "rollout_tape_disable", || {
            tape.entries.clear();
            tape.set_enabled(false);
            Ok(())
        })?;

        let mut rollout = prompt_ids.to_vec();
        let mut rollout_cache = Qwen35KvCache::new(student);
        let mut generated_tokens = if cfg.rollout_len == 0 {
            None
        } else {
            Some(timed(&mut totals, "rollout_token_buffer_alloc", || {
                let handle = store.backend().zeros(&[cfg.rollout_len])?;
                Ok(store.alloc_device_tensor(vec![cfg.rollout_len], handle)?)
            })?)
        };
        let mut current_device_token: Option<TensorId> = None;
        for step in 0..cfg.rollout_len {
            let logits = if step == 0 {
                let positions = timed(&mut totals, "rollout_positions", || {
                    Ok((0..prompt_ids.len() as u32).collect::<Vec<_>>())
                })?;
                timed(&mut totals, "rollout_student_forward", || {
                    Ok(forward_rollout_cached(
                        student,
                        store,
                        tape,
                        prompt_ids,
                        &positions,
                        &mut rollout_cache,
                    )?)
                })?
            } else {
                let position = timed(&mut totals, "rollout_positions", || {
                    Ok((prompt_ids.len() + step - 1) as u32)
                })?;
                let token_id = current_device_token
                    .ok_or("rollout cache cannot decode without a previous device token")?;
                timed(&mut totals, "rollout_student_forward", || {
                    Ok(forward_rollout_cached_device_token(
                        student,
                        store,
                        tape,
                        token_id,
                        position,
                        &mut rollout_cache,
                    )?)
                })?
            };
            let next_token = timed(&mut totals, "rollout_argmax_device", || {
                device_argmax_token(logits, vocab, store)
            })?;
            if let Some(buffer_id) = generated_tokens {
                generated_tokens = Some(timed(&mut totals, "rollout_token_write", || {
                    write_rollout_token(buffer_id, next_token, cfg.rollout_len, step, store)
                })?);
            }
            current_device_token = Some(next_token);
        }
        if let Some(buffer_id) = generated_tokens {
            let generated = timed(&mut totals, "rollout_tokens_readback", || {
                read_generated_rollout_tokens(buffer_id, cfg.rollout_len, vocab, store)
            })?;
            rollout.extend(generated);
        }

        let positions = timed(&mut totals, "full_positions", || {
            Ok((0..rollout.len() as u32).collect::<Vec<_>>())
        })?;

        let teacher_logits = timed(&mut totals, "teacher_forward", || {
            Ok(teacher.forward(store, tape, &rollout, &positions)?)
        })?;

        timed(&mut totals, "student_tape_enable", || {
            tape.set_enabled(true);
            Ok(())
        })?;

        let student_logits = timed(&mut totals, "student_forward", || {
            Ok(student.forward(store, tape, &rollout, &positions)?)
        })?;

        let loss = timed(&mut totals, "kl_distill_loss", || {
            Ok(kl_distill_loss(
                student_logits,
                teacher_logits,
                rollout.len(),
                store,
                tape,
            )?)
        })?;
        let loss_value = timed(&mut totals, "loss_readback", || Ok(store.to_host(loss)?[0]))?;

        timed(&mut totals, "optimizer_zero_grad", || {
            optimizer.zero_grad(store, student_trainable_params);
            Ok(())
        })?;
        let backward_profile = timed(&mut totals, "backward", || {
            let (_, profile) = tape.backward_profiled(loss, store)?;
            Ok(profile)
        })?;
        timed(&mut totals, "grad_clip", || {
            clip_grad_norm(student_trainable_params, cfg.grad_clip, store);
            Ok(())
        })?;
        timed(&mut totals, "optimizer_step", || {
            optimizer.step(store, student_trainable_params)?;
            Ok(())
        })?;
        timed(&mut totals, "post_step_cleanup", || {
            cleanup_after_backward(store, tape, student_model_params, &keep_extra);
            Ok(())
        })?;

        totals.add("total_step", total_started.elapsed());

        Ok((
            OpdStepOutcome {
                loss: loss_value,
                rollout_len: rollout.len(),
            },
            totals,
            backward_profile,
        ))
    }

    fn warmup_loss_probe(
        student: &Qwen35Model,
        teacher: &Qwen35Model,
        prompt_ids: &[u32],
        cfg: OpdStepConfig,
        student_model_params: &[TensorId],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> AnyResult<f32> {
        let vocab = student.config().vocab_size;
        let teacher_params = teacher.all_parameter_ids();
        let keep_extra = retained_param_and_grad_ids(&teacher_params, store);

        tape.entries.clear();
        tape.set_enabled(false);

        let mut rollout = prompt_ids.to_vec();
        let mut rollout_cache = Qwen35KvCache::new(student);
        let mut generated_tokens = if cfg.rollout_len == 0 {
            None
        } else {
            let handle = store.backend().zeros(&[cfg.rollout_len])?;
            Some(store.alloc_device_tensor(vec![cfg.rollout_len], handle)?)
        };
        let mut current_device_token: Option<TensorId> = None;
        for step in 0..cfg.rollout_len {
            let logits = if step == 0 {
                let positions = (0..prompt_ids.len() as u32).collect::<Vec<_>>();
                forward_rollout_cached(
                    student,
                    store,
                    tape,
                    prompt_ids,
                    &positions,
                    &mut rollout_cache,
                )?
            } else {
                let token_id = current_device_token
                    .ok_or("rollout cache cannot decode without a previous device token")?;
                forward_rollout_cached_device_token(
                    student,
                    store,
                    tape,
                    token_id,
                    (prompt_ids.len() + step - 1) as u32,
                    &mut rollout_cache,
                )?
            };
            let next_token = device_argmax_token(logits, vocab, store)?;
            if let Some(buffer_id) = generated_tokens {
                generated_tokens = Some(write_rollout_token(
                    buffer_id,
                    next_token,
                    cfg.rollout_len,
                    step,
                    store,
                )?);
            }
            current_device_token = Some(next_token);
        }
        if let Some(buffer_id) = generated_tokens {
            rollout.extend(read_generated_rollout_tokens(
                buffer_id,
                cfg.rollout_len,
                vocab,
                store,
            )?);
        }

        let positions = (0..rollout.len() as u32).collect::<Vec<_>>();
        let teacher_logits = teacher.forward(store, tape, &rollout, &positions)?;
        tape.set_enabled(true);
        let student_logits = student.forward(store, tape, &rollout, &positions)?;
        let loss = kl_distill_loss(student_logits, teacher_logits, rollout.len(), store, tape)?;
        let loss_value = store.to_host(loss)?[0];
        cleanup_after_backward(store, tape, student_model_params, &keep_extra);
        Ok(loss_value)
    }

    fn greedy_next_token(
        logits_id: TensorId,
        seq_len: usize,
        vocab: usize,
        store: &mut TensorStore,
    ) -> AnyResult<u32> {
        let host = store.to_host(logits_id)?;
        let last_row_start = (seq_len - 1) * vocab;
        let row = &host[last_row_start..last_row_start + vocab];
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &value) in row.iter().enumerate() {
            if value > best_val {
                best_val = value;
                best_idx = idx;
            }
        }
        Ok(best_idx as u32)
    }

    fn device_argmax_token(
        logits_id: TensorId,
        vocab: usize,
        store: &mut TensorStore,
    ) -> AnyResult<TensorId> {
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
            return Err(format!(
                "rollout logits last dim mismatch: got {last_dim}, expected vocab={vocab}"
            )
            .into());
        }
        let total = shape.iter().product::<usize>();
        let rows = total / vocab;
        if rows != 1 {
            return Err(format!("device rollout expected one logits row, got {rows}").into());
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
    ) -> AnyResult<TensorId> {
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
    ) -> AnyResult<Vec<u32>> {
        let host = store.to_host(buffer_id)?;
        if host.len() != rollout_len {
            return Err(format!(
                "generated rollout token buffer length mismatch: got {}, expected {rollout_len}",
                host.len()
            )
            .into());
        }
        let mut out = Vec::with_capacity(rollout_len);
        for (index, &value) in host.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "generated rollout token at index {index} is non-finite ({value})"
                )
                .into());
            }
            let rounded = value.round();
            if (value - rounded).abs() > 0.0 {
                return Err(format!(
                    "generated rollout token at index {index} is not an exact integer ({value})"
                )
                .into());
            }
            if rounded < 0.0 || rounded as usize >= vocab {
                return Err(format!(
                    "generated rollout token id {rounded} at index {index} is outside vocab={vocab}"
                )
                .into());
            }
            out.push(rounded as u32);
        }
        Ok(out)
    }

    fn rollout_equivalence_probe(
        student: &Qwen35Model,
        prompt_ids: &[u32],
        rollout_len: usize,
        teacher_params: &[TensorId],
        student_model_params: &[TensorId],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> AnyResult<(Vec<u32>, Vec<u32>)> {
        let keep_extra = retained_param_and_grad_ids(teacher_params, store);
        let host = host_rollout_reference(student, prompt_ids, rollout_len, store, tape)?;
        cleanup_after_backward(store, tape, student_model_params, &keep_extra);
        let device = device_rollout_reference(student, prompt_ids, rollout_len, store, tape)?;
        cleanup_after_backward(store, tape, student_model_params, &keep_extra);
        Ok((host, device))
    }

    fn host_rollout_reference(
        student: &Qwen35Model,
        prompt_ids: &[u32],
        rollout_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> AnyResult<Vec<u32>> {
        let vocab = student.config().vocab_size;
        tape.entries.clear();
        tape.set_enabled(false);

        let mut rollout = prompt_ids.to_vec();
        let mut rollout_cache = Qwen35KvCache::new(student);
        for step in 0..rollout_len {
            let (input_ids, positions) = if step == 0 {
                (
                    prompt_ids.to_vec(),
                    (0..prompt_ids.len() as u32).collect::<Vec<_>>(),
                )
            } else {
                let Some(&last) = rollout.last() else {
                    return Err("rollout cache cannot decode from an empty rollout".into());
                };
                (vec![last], vec![(rollout.len() - 1) as u32])
            };
            let logits = forward_rollout_cached(
                student,
                store,
                tape,
                &input_ids,
                &positions,
                &mut rollout_cache,
            )?;
            let next = greedy_next_token(logits, 1, vocab, store)?;
            rollout.push(next);
        }
        Ok(rollout)
    }

    fn device_rollout_reference(
        student: &Qwen35Model,
        prompt_ids: &[u32],
        rollout_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> AnyResult<Vec<u32>> {
        let vocab = student.config().vocab_size;
        tape.entries.clear();
        tape.set_enabled(false);

        let mut rollout = prompt_ids.to_vec();
        let mut rollout_cache = Qwen35KvCache::new(student);
        let mut generated_tokens = if rollout_len == 0 {
            None
        } else {
            let handle = store.backend().zeros(&[rollout_len])?;
            Some(store.alloc_device_tensor(vec![rollout_len], handle)?)
        };
        let mut current_device_token: Option<TensorId> = None;
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
                )?
            } else {
                let token_id = current_device_token
                    .ok_or("rollout cache cannot decode without a previous device token")?;
                forward_rollout_cached_device_token(
                    student,
                    store,
                    tape,
                    token_id,
                    (prompt_ids.len() + step - 1) as u32,
                    &mut rollout_cache,
                )?
            };
            let next_token = device_argmax_token(logits, vocab, store)?;
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
        }
        if let Some(buffer_id) = generated_tokens {
            rollout.extend(read_generated_rollout_tokens(
                buffer_id,
                rollout_len,
                vocab,
                store,
            )?);
        }
        Ok(rollout)
    }

    fn trainable_params(model: &Qwen35Model, store: &TensorStore) -> Vec<TensorId> {
        model
            .all_parameter_ids()
            .into_iter()
            .filter(|id| store.get(*id).is_some_and(|tensor| tensor.requires_grad))
            .collect()
    }

    fn param_element_count(params: &[TensorId], store: &TensorStore) -> usize {
        params
            .iter()
            .filter_map(|id| store.get(*id).map(|tensor| tensor.size))
            .sum()
    }

    fn perturb_params(params: &[TensorId], store: &mut TensorStore, seed: u64, scale: f32) {
        let mut rng = Lcg::new(seed);
        for &id in params {
            if let Some(tensor) = store.get_mut(id) {
                for value in &mut tensor.data {
                    *value += rng.next_f32() * scale;
                }
            }
        }
    }

    #[derive(Debug, Default)]
    struct HostMirrorDropStats {
        tensors_cleared: usize,
        bytes_cleared: usize,
    }

    fn drop_large_host_mirrors(
        teacher_params: &[TensorId],
        student_params: &[TensorId],
        store: &mut TensorStore,
    ) -> HostMirrorDropStats {
        let mut seen = HashSet::new();
        let mut stats = HostMirrorDropStats::default();
        for id in teacher_params.iter().chain(student_params).copied() {
            if !seen.insert(id) {
                continue;
            }
            let Some(Some(tensor)) = store.tensors.get_mut(id) else {
                continue;
            };
            if tensor.size < HOST_MIRROR_DROP_MIN_ELEMENTS
                || tensor.device_handle.is_none()
                || tensor.dirty == Dirty::Host
                || tensor.data.is_empty()
            {
                continue;
            }
            stats.tensors_cleared += 1;
            stats.bytes_cleared += tensor.data.len() * std::mem::size_of::<f32>();
            tensor.data.clear();
            tensor.dirty = Dirty::Device;
        }
        stats
    }

    #[derive(Debug)]
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_f32(&mut self) -> f32 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((self.state >> 32) as u32) as f32 / (u32::MAX as f32);
            unit * 2.0 - 1.0
        }
    }

    fn print_profile(
        outcome: OpdStepOutcome,
        totals: &PhaseTotals,
        backward_profile: &BackwardProfile,
    ) {
        let total_step_secs = totals.seconds("total_step");
        println!(
            "step_summary loss={:.12e} rollout_len={} total_step_seconds={total_step_secs:.6}",
            outcome.loss, outcome.rollout_len
        );

        let mut phase_rows: Vec<(&'static str, f64)> = totals
            .durations
            .iter()
            .filter_map(|(&phase, duration)| {
                (phase != "total_step").then_some((phase, duration.as_secs_f64()))
            })
            .collect();
        phase_rows.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));

        for (rank, (phase, seconds)) in phase_rows.iter().enumerate() {
            let pct_total = if total_step_secs == 0.0 {
                0.0
            } else {
                seconds / total_step_secs * 100.0
            };
            println!(
                "phase_summary rank={} phase={} seconds={:.6} pct_total={:.3}",
                rank + 1,
                phase,
                seconds,
                pct_total
            );
        }

        let backward_total_secs = backward_profile.total_duration.as_secs_f64();
        let backward_op_secs = backward_profile.total_op_duration().as_secs_f64();
        let backward_merge_secs = backward_profile.merge_grad_duration.as_secs_f64();
        let backward_prelude_secs = backward_profile.prelude_duration.as_secs_f64();
        let backward_unattributed_secs =
            (backward_total_secs - backward_op_secs - backward_merge_secs - backward_prelude_secs)
                .max(0.0);
        println!(
            "backward_profile_summary total_seconds={backward_total_secs:.6} op_seconds={backward_op_secs:.6} merge_grad_seconds={backward_merge_secs:.6} prelude_seconds={backward_prelude_secs:.6} unattributed_seconds={backward_unattributed_secs:.6}"
        );

        let mut backward_rows = backward_profile
            .op_totals
            .iter()
            .map(|(&op, stats)| (op, stats.count, stats.duration.as_secs_f64()))
            .collect::<Vec<_>>();
        backward_rows.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
        for (rank, (op, count, seconds)) in backward_rows.iter().enumerate() {
            let pct_backward = if backward_total_secs == 0.0 {
                0.0
            } else {
                seconds / backward_total_secs * 100.0
            };
            let pct_total = if total_step_secs == 0.0 {
                0.0
            } else {
                seconds / total_step_secs * 100.0
            };
            println!(
                "backward_op_summary rank={} op={:?} count={} seconds={:.6} pct_backward={:.3} pct_total={:.3}",
                rank + 1,
                op,
                count,
                seconds,
                pct_backward,
                pct_total
            );
        }
    }
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    app::main()
}

#[cfg(not(all(feature = "cuda", not(feature = "no-cuda"))))]
fn main() {
    eprintln!(
        "opd_step_cuda_realckpt_profile requires CUDA. Run with: \
         cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda"
    );
    std::process::exit(1);
}
