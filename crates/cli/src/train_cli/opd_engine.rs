#[cfg(feature = "cuda")]
use {
    super::{
        cc_eval::agent_opd_eval_out_dir,
        opd_runtime::{apply_tape_dtype, build_opd_store, log_opd_vram, trainable_param_ids},
    },
    crate::args::TrainAgentOpdArgs,
    anyhow::{Context, Result, anyhow, bail},
    autograd::TensorId,
    qwen35_spec::Qwen35Config,
    std::{
        path::{Path, PathBuf},
        time::Instant,
    },
};

/// Install the `--rollout-engine` selection (CUDA only; inert on CPU, which has
/// no infer engine). Unset → `infer` default.
#[cfg(feature = "cuda")]
pub(super) fn apply_opd_rollout_engine(engine: Option<crate::args::OpdRolloutEngineArg>) {
    if let Some(engine) = engine {
        train::opd::set_infer_rollout_override(engine == crate::args::OpdRolloutEngineArg::Infer);
    }
}

#[cfg(not(feature = "cuda"))]
pub(super) fn apply_opd_rollout_engine(_engine: Option<crate::args::OpdRolloutEngineArg>) {}

#[cfg(feature = "cuda")]
pub(super) fn load_opd_infer_student(
    student_dir: &Path,
    max_seq_len: usize,
    train_backend: std::sync::Arc<dyn autograd::Backend>,
    vocab_size: usize,
    runtime: &crate::args::OpdRuntimeArgs,
    memory_budget_bytes: Option<usize>,
) -> Result<Option<train::infer_student::InferStudent>> {
    if !train::opd::infer_rollout_flag_enabled() {
        return Ok(None);
    }

    use std::sync::{Arc, Mutex};

    use infer_api::{EngineLoadConfig, LoadedInferenceEngine};

    let max_seq_len = max_seq_len.max(128);
    eprintln!(
        "[arle train opd] loading infer rollout student from {} (max_seq_len={max_seq_len})",
        student_dir.display()
    );
    let engine = LoadedInferenceEngine::load_with_config(
        student_dir
            .to_str()
            .ok_or_else(|| anyhow!("student model path is not valid UTF-8"))?,
        true,
        EngineLoadConfig {
            dspark_draft_model: runtime.dspark_draft_model.clone(),
            dspark_sps_bias_ms: runtime.dspark_sps_bias_ms,
            dspark_sps_row_ms: runtime.dspark_sps_row_ms,
            mem_fraction_static: runtime.rollout_mem_fraction,
            memory_budget_bytes,
            // Whole-step decode graph for the rollout: eager per-token decode is
            // host-launch-bound (~156 ms/token), the OPD step's dominant cost.
            cuda: runtime.cuda_runtime_flags(),
            world_size: None,
            ..EngineLoadConfig::single_sequence(max_seq_len)
        },
    )
    .with_context(|| format!("load infer rollout student from {}", student_dir.display()))?;

    Ok(Some(train::infer_student::InferStudent::new(
        Arc::new(Mutex::new(engine)),
        train_backend,
        vocab_size,
    )))
}

#[cfg(feature = "cuda")]
pub(super) fn maybe_preoffload_infer_student_before_teacher(
    infer_student: &Option<train::infer_student::InferStudent>,
    train_backend: &std::sync::Arc<dyn autograd::Backend>,
) -> Result<()> {
    if !train::opd::engine_offload_mode().offloads_student() {
        return Ok(());
    }
    let Some(student) = infer_student.as_ref() else {
        return Ok(());
    };

    train_backend
        .device_synchronize()
        .context("synchronize train backend before infer rollout student pre-teacher offload")?;
    let freed = student
        .offload_engine_weights()
        .context("offload infer rollout student before infer teacher load")?;
    eprintln!(
        "opd_engine_offload student_pre_teacher_offloaded freed_bytes={freed} freed_mib={:.1}",
        freed as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

/// Borrow the rollout engine's resident FP8 base pointers for
/// `--share-frozen-base` and map them onto the loader's backend-agnostic table,
/// so the autograd student's frozen FP8 base projections import a NON-OWNING
/// view instead of allocating their own ~27 GB copy.
#[cfg(feature = "cuda")]
pub(super) fn shared_frozen_base_entries(
    engine: &infer_api::LoadedInferenceEngine,
    label: &str,
) -> Result<Vec<train::qwen35_loader::SharedFrozenBaseEntry>> {
    use train::qwen35_loader::SharedFrozenBaseEntry;

    let table = engine
        .frozen_base_fp8_pointers()
        .context("borrow rollout-engine FP8 base pointers for --share-frozen-base")?;
    eprintln!(
        "[arle train {label}] --share-frozen-base: borrowing {} resident FP8 base projections from the rollout engine (zero-copy)",
        table.len()
    );
    Ok(table
        .into_iter()
        .map(|p| SharedFrozenBaseEntry {
            layer_idx: p.layer_idx,
            proj_suffix: p.proj_suffix,
            weight_ptr: p.weight_ptr,
            scale_ptr: p.scale_ptr,
            rows: p.rows,
            cols: p.cols,
            block_m: p.block_m,
            block_k: p.block_k,
        })
        .collect())
}

/// Rollout engine + cc serve + autograd student for one agent-opd rank. Every
/// mesh rank loads this (the cp fleet serves one group's samples in parallel);
/// rank 0 additionally owns the harness, filtering, and saves.
#[cfg(feature = "cuda")]
pub(super) struct AgentOpdServeStudent {
    pub(super) store: autograd::TensorStore,
    pub(super) train_backend: std::sync::Arc<dyn autograd::Backend>,
    pub(super) vocab: usize,
    pub(super) infer_student: train::infer_student::InferStudent,
    pub(super) student: train::qwen35::Qwen35Model,
    pub(super) serve_thread: infer_api::ServeThread,
    pub(super) dump_dir: PathBuf,
    pub(super) cc_model_id: String,
    pub(super) all_params: Vec<TensorId>,
    pub(super) trainable: Vec<TensorId>,
}

#[cfg(feature = "cuda")]
pub(super) fn load_agent_opd_serve_student(
    args: &TrainAgentOpdArgs,
    lora: train::lora::LoraConfig,
    target_set: train::lora::LoraTargetSet,
    serve_port: u16,
) -> Result<AgentOpdServeStudent> {
    use std::sync::{Arc, Mutex};

    use infer_api::{EngineLoadConfig, LoadedInferenceEngine};
    use train::{
        infer_student::InferStudent,
        qwen35_checkpoint::load_qwen35_lora_adapters,
        qwen35_loader::{SharedFrozenBaseEntry, load_qwen35_lora_from_hf_dir_with_shared_base},
    };

    let student_dir = args.student_model.as_path();
    let (mut store, train_backend, _backend_label) = build_opd_store(args.backend)?;
    apply_tape_dtype(&mut store, args.tape_dtype)?;
    // Vocab from the checkpoint config (not the autograd student) so the rollout
    // engine can load BEFORE the autograd student when `--share-frozen-base` is
    // set. Same value `Qwen35Model::config()` exposes.
    let hf_config = Qwen35Config::from_json_file(student_dir.join("config.json"))
        .with_context(|| format!("read config.json from {}", student_dir.display()))?;
    let vocab = hf_config.vocab_size;
    let eval_out_dir = agent_opd_eval_out_dir(args);
    // Rollout engine (student) doubles as the cc serve. KV budget for cc
    // traffic: K concurrent cc streams × the measured session ceiling, +25%
    // headroom, page_size 16.
    use train::cc_harness::{CC_MAX_SESSION_TOKENS, CC_SESSION_TOKENS};
    // Sessions THIS rank serves at once. Round-robin restarts per group
    // (`sample % base_urls.len()`), so every concurrent group hands this rank
    // its own ceil(K/cp) — G × ceil(K/cp), not ceil(G×K/cp).
    let width = args.prompts_per_update.max(1)
        * args
            .samples_per_prompt
            .max(1)
            .div_ceil(train::context_parallel::CpContext::from_env().size.max(1));
    // Pool = K typical streams, but never smaller than one full long-horizon
    // session (+25% headroom each) so a 200K session stays schedulable.
    let cc_pages = (width * CC_SESSION_TOKENS)
        .max(CC_MAX_SESSION_TOKENS)
        .div_ceil(16);
    let cc_total_pages = cc_pages + cc_pages / 4;

    // Shared frozen-base pointers alias engine weight buffers that a student
    // offload frees mid-step — refuse the combination at load time.
    if !args.no_share_frozen_base && train::opd::engine_offload_mode().offloads_student() {
        bail!(
            "--engine-offload student is incompatible with frozen-base sharing; \
             pass --no-share-frozen-base"
        );
    }

    // Load order: default loads the autograd student FIRST (engine then sees
    // post-student free VRAM — byte-identical). --share-frozen-base loads the
    // engine FIRST so the student can import (zero-copy) its resident FP8 base.
    let prebuilt_student = if !args.no_share_frozen_base {
        None
    } else {
        eprintln!(
            "[arle train agent-opd] loading student from {}",
            student_dir.display()
        );
        Some(
            load_qwen35_lora_from_hf_dir_with_shared_base(
                student_dir,
                lora,
                target_set,
                args.lora_layer_start,
                args.lora_skip_experts,
                None,
                &mut store,
            )
            .with_context(|| format!("load LoRA student from {}", student_dir.display()))?,
        )
    };

    eprintln!(
        "[arle train agent-opd] loading rollout engine from {} (slots={width} pages={cc_total_pages})",
        student_dir.display()
    );
    let student_engine = LoadedInferenceEngine::load_with_config(
        student_dir
            .to_str()
            .ok_or_else(|| anyhow!("student path is not valid UTF-8"))?,
        // agent-OPD: decode CUDA-graph default-OFF. Its captured workspace (~30 GB
        // on the 27B MoE, captured during the rollout's decode) would co-reside
        // with the masked-CE writeback and OOM it (post-rollout engine ~87 GB vs
        // ~55 GB no-graph). A measured ~26 GB headroom made it worth exposing as
        // --qwen35-decode-graph; the default flip waits for the co-residency license.
        args.runtime.qwen35_decode_graph,
        EngineLoadConfig {
            num_slots: width,
            page_size: 16,
            total_pages: cc_total_pages,
            // Request caps are POOL-derived, not per-session: capping at
            // CC_SESSION_TOKENS silently aborted cc mid-conversation (a
            // tripwire — sidecars showed prompt>22K → gen 0). The pool bounds
            // memory and the engine preempts under KV pressure (#162).
            max_prompt_tokens: cc_total_pages * 16 - 256,
            max_total_tokens: cc_total_pages * 16,
            chunked_prefill_size: Some(CC_SESSION_TOKENS),
            mem_fraction_static: args.runtime.rollout_mem_fraction,
            dspark_draft_model: args.runtime.dspark_draft_model.clone(),
            dspark_sps_bias_ms: args.runtime.dspark_sps_bias_ms,
            dspark_sps_row_ms: args.runtime.dspark_sps_row_ms,
            // MTP GPU gate passed 2026-07-17 (1.21×); default-on waits for the
            // depth sweep + an in-loop A/B.
            mtp_draft_tokens: args.runtime.mtp_draft_tokens,
            cuda: args.runtime.cuda_runtime_flags(),
            world_size: None,
            ..EngineLoadConfig::default()
        },
    )
    .with_context(|| format!("load rollout engine from {}", student_dir.display()))?;
    log_opd_vram(
        "after rollout engine load (KV pool alloc'd)",
        &train_backend,
    );

    // cc serve over THIS engine (same engine thread, same KV pool): install the
    // dump sink BEFORE any traffic, then serve the router on a background
    // thread. Token-only shutdown (`serve_thread.shutdown()` at run end).
    // Every fleet rank shares ONE dump dir (filenames are pid-tagged).
    let dump_dir = eval_out_dir.join("dumps");
    infer_api::set_messages_dump_dir(&dump_dir)
        .with_context(|| format!("create cc dump dir {}", dump_dir.display()))?;
    let cc_model_id = infer_api::InferenceEngine::model_id(&student_engine).to_owned();
    // Rollout = flag temperature (>0 keeps behavior logprobs non-empty) +
    // model nucleus from generation_config (truncates the tail — no salad).
    let mut sampling_defaults = infer_api::SamplingDefaults::from_generation_config(student_dir);
    sampling_defaults.temperature = args.rollout_temperature;
    infer_api::set_sampling_defaults(sampling_defaults);
    let serve_thread = infer_api::serve_router_on_thread(
        student_engine.local_router(0)?, // thinking unbounded — the serve default
        "127.0.0.1",
        serve_port,
    )?;
    eprintln!(
        "[arle train agent-opd] cc serve on http://127.0.0.1:{} (model={cc_model_id}, dumps={})",
        serve_port,
        dump_dir.display()
    );

    // Train-infer FP8 weight sharing (`--share-frozen-base`): borrow the rollout
    // engine's resident FP8 base pointers and pass them into the autograd student
    // load so its frozen FP8 base projections import a NON-OWNING view.
    let shared_base_entries: Vec<SharedFrozenBaseEntry> = if !args.no_share_frozen_base {
        shared_frozen_base_entries(&student_engine, "agent-opd")?
    } else {
        Vec::new()
    };
    let shared_base = if !args.no_share_frozen_base {
        Some(shared_base_entries.as_slice())
    } else {
        None
    };

    let student = match prebuilt_student {
        Some(s) => s,
        None => {
            eprintln!(
                "[arle train agent-opd] loading student from {}",
                student_dir.display()
            );
            load_qwen35_lora_from_hf_dir_with_shared_base(
                student_dir,
                lora,
                target_set,
                args.lora_layer_start,
                args.lora_skip_experts,
                shared_base,
                &mut store,
            )
            .with_context(|| format!("load LoRA student from {}", student_dir.display()))?
        }
    };

    // Resume: overlay a saved adapter onto the fresh student (both load branches
    // merge here) BEFORE the handoff fence, so its A/B uploads drain with the base.
    if let Some(dir) = args.lora_adapters.as_deref() {
        load_qwen35_lora_adapters(&student, &mut store, dir)
            .with_context(|| format!("resume LoRA adapter from {}", dir.display()))?;
        eprintln!("[agent-opd] resumed adapter from {}", dir.display());
    }

    // Shared-base bytes alias the engine's resident FP8 — drain the autograd
    // backend's OWN in-flight uploads before the first autograd forward
    // (cross-stream handoff fence). MUST be a stream-scoped sync, NOT a
    // context-wide `cuCtxSynchronize`: the co-resident rollout engine shares this
    // device primary context but runs its streams with cudarc event-tracking
    // DISABLED and idle-parked between scheduler steps, so a `cuCtxSynchronize`
    // here blocks forever in poll() draining the engine's never-host-progressed
    // streams (measured deadlock — main thread poll(nfds=2,-1), GPU flat, all
    // 851 student tensors already materialized). The engine's resident FP8 base
    // weights are fully written by its own load+warmup before this point, so the
    // borrow only needs the student's own upload stream drained.
    if !args.no_share_frozen_base {
        let opd_load_trace = std::env::var("ARLE_OPD_LOAD_TRACE").is_ok();
        if opd_load_trace {
            eprintln!("[opd-load-trace] pre stream-sync (share-frozen-base handoff fence)");
        }
        train_backend
            .stream_synchronize()
            .context("stream sync before first shared-base autograd forward")?;
        if opd_load_trace {
            eprintln!("[opd-load-trace] post stream-sync OK; building InferStudent");
        }
    }

    let all_params: Vec<TensorId> = student.all_parameter_ids();
    let trainable = trainable_param_ids(&all_params, &store);
    if trainable.is_empty() {
        bail!("agent-opd student has no trainable (LoRA) parameters; check --lora-target-set");
    }

    let infer_student = InferStudent::new(
        Arc::new(Mutex::new(student_engine)),
        train_backend.clone(),
        vocab,
    )
    .with_lora_merge_fp8(args.lora_merge_fp8);
    log_opd_vram(
        "after autograd student load (resident floor)",
        &train_backend,
    );

    Ok(AgentOpdServeStudent {
        store,
        train_backend,
        vocab,
        infer_student,
        student,
        serve_thread,
        dump_dir,
        cc_model_id,
        all_params,
        trainable,
    })
}

/// Drain the serve engine before the weight re-merge / KV-pool drop. The
/// group's cc children have exited by the time this runs, so any request still
/// in flight is an orphan by definition (its client is dead) — cancel them all,
/// then REQUIRE active_requests == 0: a live request past this point reads
/// stale or freed engine state (an engine-thread panic).
#[cfg(feature = "cuda")]
fn quiesce_serve(
    engine: &std::sync::Arc<std::sync::Mutex<infer_api::LoadedInferenceEngine>>,
) -> Result<()> {
    use infer_api::InferenceEngine as _;

    let lock = || {
        engine
            .lock()
            .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))
    };
    let cancelled = lock()?.quiesce_admissions()?;
    if cancelled > 0 {
        eprintln!("[agent-opd] quiesce: cancelled {cancelled} orphaned request(s)");
    }
    let started = Instant::now();
    let mut next_warn = 10u64;
    loop {
        let active = lock()?.telemetry().active_requests;
        if active == 0 {
            return Ok(());
        }
        let elapsed = started.elapsed().as_secs();
        anyhow::ensure!(
            elapsed < 60,
            "quiesce: {active} request(s) still active {elapsed}s after cancellation — \
             refusing to proceed (a live request during weight re-merge / KV drop \
             corrupts the engine)"
        );
        if elapsed >= next_warn {
            eprintln!("[agent-opd] quiesce: {active} request(s) still active after {elapsed}s");
            next_warn += 10;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// Every mesh rank runs this exact sequence before a mirrored update — the cp
/// collectives stay aligned by construction, not by mirrored edits. Quiesce
/// first (the group's cc children exited with run_group, so this only drains a
/// straggler request); then drop the scratch + DEAD rollout KV pool — the two
/// headroom levers (scratch otherwise OOMs the logits alloc; the writeback's
/// fresh autograd forward never reads the engine KV).
#[cfg(feature = "cuda")]
pub(super) fn quiesce_and_release_engines(
    infer_student: &train::infer_student::InferStudent,
) -> Result<()> {
    train::aopd_profile::time_try("quiesce", train::aopd_profile::WALL, || {
        quiesce_serve(infer_student.engine())
    })?;
    if let Err(err) = infer_student.release_inference_scratch() {
        eprintln!("[agent-opd] release inference scratch failed: {err}");
    }
    if let Err(err) = infer_student.release_kv_pool() {
        eprintln!("[agent-opd] release KV pool failed: {err}");
    }
    Ok(())
}

/// Group-end twin of [`quiesce_and_release_engines`]: re-merge the trained
/// LoRA into this rank's engine when the leader synced, then re-acquire the KV
/// pool. Returns the sync wall (0.0 when not synced).
#[cfg(feature = "cuda")]
pub(super) fn sync_and_restore_engines(
    infer_student: &train::infer_student::InferStudent,
    store: &mut autograd::TensorStore,
    adapter_map: &std::collections::HashMap<&'static str, TensorId>,
    param_name_map: &std::collections::HashMap<&'static str, TensorId>,
    lora: train::lora::LoraConfig,
    synced: bool,
) -> Result<f64> {
    let mut sync_secs = 0.0;
    if synced {
        let started = Instant::now();
        infer_student
            .sync_lora_from_store(store, adapter_map, param_name_map, lora)
            .context("sync trained LoRA into rollout engine")?;
        sync_secs = started.elapsed().as_secs_f64();
        train::aopd_profile::record("sync_lora", train::aopd_profile::GPU, sync_secs);
    }
    infer_student
        .ensure_kv_pool_and_resume_admissions()
        .context("restore rollout engine after group")?;
    Ok(sync_secs)
}
