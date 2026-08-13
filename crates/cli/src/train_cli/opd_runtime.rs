use anyhow::{Context, Result, bail};
use autograd::{TensorId, TensorStore};
use indicatif::ProgressStyle;
use infer_plan::SamplingParams;
use serde::Serialize;

use crate::args::{
    KlDirectionArg, LrScheduleArg, OpdBackendArg, OpdKlMaskArg, OpdSftAnchorArg, TapeDtypeArg,
};

pub(super) fn opd_step_profile_enabled() -> bool {
    std::env::var_os("ARLE_OPD_STEP_PROFILE").is_some()
}

pub(super) fn opd_logits_window_size_arg(value: usize) -> Option<usize> {
    (value > 0).then_some(value)
}

pub(super) fn print_opd_step_profile(step: usize, profile: &train::opd::OpdStepProfile) {
    println!(
        "opd_step_profile step={step} total_seconds={:.6} student_rollout_seconds={:.6} \
         teacher_forward_seconds={:.6} student_forward_seconds={:.6} kl_loss_seconds={:.6} \
         backward_seconds={:.6} optimizer_zero_grad_seconds={:.6} grad_clip_seconds={:.6} \
         optimizer_step_seconds={:.6} post_step_cleanup_seconds={:.6}",
        profile.total_seconds,
        profile.student_rollout_seconds,
        profile.teacher_forward_seconds,
        profile.student_forward_seconds,
        profile.kl_loss_seconds,
        profile.backward_seconds,
        profile.optimizer_zero_grad_seconds,
        profile.grad_clip_seconds,
        profile.optimizer_step_seconds,
        profile.post_step_cleanup_seconds,
    );
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OpdStepMetric {
    pub(super) step: usize,
    pub(super) loss: f32,
    pub(super) lr: f32,
    pub(super) grad_norm: f32,
    pub(super) rollout_len: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OpdSummary {
    pub(super) step_count: usize,
    pub(super) final_loss: Option<f32>,
    pub(super) mean_loss: Option<f32>,
    pub(super) min_loss: Option<f32>,
    pub(super) max_loss: Option<f32>,
}

pub(super) fn opd_summary(step_metrics: &[OpdStepMetric]) -> OpdSummary {
    let final_loss = step_metrics.last().map(|metric| metric.loss);
    let mean_loss = if step_metrics.is_empty() {
        None
    } else {
        Some(step_metrics.iter().map(|metric| metric.loss).sum::<f32>() / step_metrics.len() as f32)
    };
    let min_loss = step_metrics
        .iter()
        .map(|metric| metric.loss)
        .min_by(f32::total_cmp);
    let max_loss = step_metrics
        .iter()
        .map(|metric| metric.loss)
        .max_by(f32::total_cmp);
    OpdSummary {
        step_count: step_metrics.len(),
        final_loss,
        mean_loss,
        min_loss,
        max_loss,
    }
}

pub(super) fn opd_progress_style() -> Result<ProgressStyle> {
    ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} \
         avg_loss={msg} eta={eta_precise}",
    )
    .map(|style| style.progress_chars("=>-"))
    .context("build OPD progress style")
}

pub(super) fn current_grad_norm(params: &[TensorId], store: &TensorStore) -> f32 {
    let mut total_sq_norm = 0.0_f64;
    for &param_id in params {
        let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) else {
            continue;
        };
        let Some(grad) = store.get(grad_id) else {
            continue;
        };
        total_sq_norm += grad
            .data
            .iter()
            .map(|&value| {
                let value = f64::from(value);
                value * value
            })
            .sum::<f64>();
    }
    total_sq_norm.sqrt() as f32
}

#[derive(Debug, Clone)]
pub(super) struct PromptSampler {
    state: u64,
}

impl PromptSampler {
    pub(super) fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    pub(super) fn next_index(&mut self, len: usize) -> usize {
        debug_assert!(len > 0);
        if len <= 1 {
            return 0;
        }
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 32) as usize) % len
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(super) fn next_unit(&mut self) -> f64 {
        self.next_index(1 << 24) as f64 / (1 << 24) as f64
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct OpdLrSchedule {
    mode: LrScheduleArg,
    base_lr: f32,
    min_lr: f32,
    warmup_steps: u64,
    total_steps: u64,
}

impl OpdLrSchedule {
    pub(super) fn new(
        mode: LrScheduleArg,
        base_lr: f32,
        warmup_steps: Option<usize>,
        total_steps: usize,
    ) -> Result<Self> {
        if mode == LrScheduleArg::Cosine && (!base_lr.is_finite() || base_lr < 0.0) {
            bail!("--lr-schedule cosine requires a finite non-negative --lr, got {base_lr}");
        }
        let warmup_steps = warmup_steps.unwrap_or_else(|| default_cosine_warmup_steps(total_steps));
        Ok(Self {
            mode,
            base_lr,
            min_lr: base_lr * 0.1,
            warmup_steps: warmup_steps as u64,
            total_steps: total_steps as u64,
        })
    }

    fn lr_at_step(self, step: u64) -> f32 {
        match self.mode {
            LrScheduleArg::Fixed => self.base_lr,
            LrScheduleArg::Cosine => self.cosine_lr_at_step(step),
        }
    }

    pub(super) fn apply_to_optimizer(self, optimizer: &mut autograd::AdamW, step: u64) -> f32 {
        let lr = self.lr_at_step(step);
        if self.mode == LrScheduleArg::Cosine {
            autograd::Optimizer::set_lr(optimizer, lr);
        }
        lr
    }

    fn cosine_lr_at_step(self, step: u64) -> f32 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return self.base_lr * (step as f32 / self.warmup_steps as f32);
        }
        if self.total_steps <= self.warmup_steps {
            return self.base_lr;
        }
        if self.total_steps > 0 && step >= self.total_steps - 1 {
            return self.min_lr;
        }
        let decay_span = self.total_steps - self.warmup_steps - 1;
        if decay_span == 0 {
            return self.min_lr;
        }
        let progress = (step - self.warmup_steps) as f32 / decay_span as f32;
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        self.min_lr + (self.base_lr - self.min_lr) * cosine
    }
}

fn default_cosine_warmup_steps(total_steps: usize) -> usize {
    if total_steps == 0 {
        0
    } else {
        total_steps.saturating_mul(3).div_ceil(100).max(1)
    }
}

#[allow(unused_variables)]
pub(super) fn build_opd_store(
    arg: OpdBackendArg,
) -> Result<(
    autograd::TensorStore,
    std::sync::Arc<dyn autograd::Backend>,
    &'static str,
)> {
    #[cfg(feature = "cuda")]
    {
        use std::sync::Arc;
        let want_cuda = matches!(arg, OpdBackendArg::Cuda | OpdBackendArg::Auto);
        if want_cuda {
            // Mesh env comes from the launcher; both axes size<=1 keeps the
            // single-card new(0) byte-identical.
            let cp = train::context_parallel::CpContext::from_env();
            let dp = train::context_parallel::DpContext::from_env();
            let backend = if cp.is_enabled() || dp.is_enabled() {
                #[cfg(feature = "nccl")]
                {
                    let ordinal = std::env::var("INFER_CUDA_DEVICE")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    let uid = infer_api::nccl_unique_id_from_env()
                        .context("CP/DP: read INFER_NCCL_UNIQUE_ID")?;
                    let world_rank = train::context_parallel::world_rank(cp, dp);
                    let seq_group =
                        (cp.is_enabled() && dp.is_enabled()).then_some((dp.rank, cp.size, cp.rank));
                    let backend = autograd::backend_cuda::CudaBackend::new_with_mesh(
                        ordinal,
                        uid,
                        cp.size * dp.size,
                        world_rank,
                        seq_group,
                    )
                    .context("init CUDA+NCCL backend for the CP×DP mesh")?;
                    if !backend.has_collective() {
                        bail!(
                            "multi-rank mesh (cp={}×dp={}) got a CUDA backend without an NCCL \
                             communicator; refusing the silent host-transport fallback",
                            cp.size,
                            dp.size
                        );
                    }
                    backend
                }
                #[cfg(not(feature = "nccl"))]
                {
                    bail!(
                        "context/data parallelism (ARLE_TRAIN_CP_SIZE>1 or ARLE_TRAIN_DP_SIZE>1) \
                         requires the nccl feature; rebuild with --features cuda,nccl"
                    );
                }
            } else {
                autograd::backend_cuda::CudaBackend::new(0).context("init CUDA backend (GPU 0)")?
            };
            let backend: Arc<dyn autograd::Backend> = Arc::new(backend);
            let label = match (cp.is_enabled(), dp.is_enabled()) {
                (true, true) => "cuda:cp×dp",
                (true, false) => "cuda:cp",
                (false, true) => "cuda:dp",
                (false, false) => "cuda:0",
            };
            return Ok((
                autograd::TensorStore::with_backend(backend.clone()),
                backend,
                label,
            ));
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        if matches!(arg, OpdBackendArg::Cuda) {
            bail!(
                "arle was built without the cuda feature; rebuild with \
                 `cargo build --release --features cuda` to use --backend cuda"
            );
        }
    }
    let backend: std::sync::Arc<dyn autograd::Backend> = std::sync::Arc::new(autograd::CpuBackend);
    Ok((
        autograd::TensorStore::with_backend(backend.clone()),
        backend,
        "cpu",
    ))
}

/// Bf16 tape is CUDA-only; bail instead of silently training on an f32 tape.
pub(super) fn apply_tape_dtype(store: &mut TensorStore, requested: TapeDtypeArg) -> Result<()> {
    store.set_tape_dtype(requested.as_tape_dtype());
    if requested == TapeDtypeArg::Bf16 && store.backend().tape_dtype() != autograd::TapeDtype::Bf16
    {
        bail!(
            "--tape-dtype bf16 is a no-op on the {:?} backend; use --backend cuda",
            store.backend().device()
        );
    }
    Ok(())
}

pub(super) fn parse_lora_target_set(raw: &str) -> Result<train::lora::LoraTargetSet> {
    use train::lora::LoraTargetSet;
    match raw {
        "attention-qv" | "attention_qv" | "qv" => Ok(LoraTargetSet::AttentionQv),
        "attention-full" | "attention_full" | "full-attn" => Ok(LoraTargetSet::AttentionFull),
        "all-linear" | "all_linear" | "all" => Ok(LoraTargetSet::AllLinear),
        other => bail!(
            "unknown --lora-target-set {other:?} (expected attention-qv, attention-full, or all-linear)"
        ),
    }
}

pub(super) fn kl_direction_arg(arg: KlDirectionArg) -> train::loss::KlDirection {
    match arg {
        KlDirectionArg::Forward => train::loss::KlDirection::Forward,
        KlDirectionArg::Reverse => train::loss::KlDirection::Reverse,
    }
}

pub(super) fn kl_mask_arg(arg: OpdKlMaskArg) -> train::opd::OpdKlMask {
    match arg {
        OpdKlMaskArg::Completion => train::opd::OpdKlMask::CompletionOnly,
        OpdKlMaskArg::Full => train::opd::OpdKlMask::Full,
    }
}

pub(super) fn opd_sft_anchor_arg(arg: OpdSftAnchorArg) -> train::opd::GkdSftAnchor {
    match arg {
        OpdSftAnchorArg::StudentRollout => train::opd::GkdSftAnchor::StudentRollout,
        OpdSftAnchorArg::CorpusTruth => train::opd::GkdSftAnchor::CorpusTruth,
    }
}

pub(super) fn rollout_sampling_params(
    temperature: f32,
    top_p: f32,
    top_k: i32,
    seed: Option<u64>,
) -> Option<SamplingParams> {
    if temperature == 0.0 {
        None
    } else {
        Some(SamplingParams {
            temperature,
            top_p,
            top_k,
            seed,
            ..SamplingParams::default()
        })
    }
}

/// Reject advertised-but-unimplemented distillation objectives before any
/// model/store is loaded, so a no-op flag fails fast instead of silently
/// running a different objective mid-training.
pub(super) fn reject_unimplemented_gkd_objectives(
    gkd_entropy_weight: f32,
    teacher_topk: Option<usize>,
) -> Result<()> {
    if gkd_entropy_weight != 0.0 {
        bail!(
            "--gkd-entropy-weight {gkd_entropy_weight} is not implemented: the per-position \
             entropy-weighted objective (AEPO) does not exist yet. Omit the flag (0.0) to run \
             unweighted KL."
        );
    }
    if teacher_topk.is_some() {
        bail!(
            "--teacher-topk requires an engine-side top-k teacher-logprob producer (H20 Piece A) \
             that is not wired. Omit the flag to run dense KL."
        );
    }
    Ok(())
}

pub(super) fn validate_train_opd_gkd_args(
    gkd_lambda: f32,
    sft_anchor: OpdSftAnchorArg,
) -> Result<()> {
    if !(0.0..=1.0).contains(&gkd_lambda) || !gkd_lambda.is_finite() {
        bail!("--gkd-lambda must be finite and in [0.0, 1.0], got {gkd_lambda}");
    }
    match sft_anchor {
        OpdSftAnchorArg::StudentRollout => {
            if gkd_lambda == 0.0 {
                Ok(())
            } else {
                bail!(
                    "`arle train opd` keeps the student-rollout anchor pure-KL; \
                     --gkd-lambda must be 0.0 unless --sft-anchor corpus-truth is used, \
                     got {gkd_lambda}"
                )
            }
        }
        OpdSftAnchorArg::CorpusTruth => {
            if gkd_lambda > 0.0 {
                Ok(())
            } else {
                bail!(
                    "--sft-anchor corpus-truth requires --gkd-lambda > 0.0; \
                     use --gkd-lambda 1.0 for off-policy SFT on teacher completions"
                )
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub(super) fn validate_online_rollout_temperature(
    preset: train::update_strategy::UpdatePreset,
    strategy: crate::args::UpdateStrategyArg,
    temperature: f32,
) -> Result<()> {
    if preset.needs().behavior_logprobs && temperature <= 0.0 {
        bail!(
            "--rollout-temperature must be > 0 for ratio-weighted --update-strategy {strategy:?}; \
             greedy rollout does not produce behavior logprobs (rejection-ce permits greedy)"
        );
    }
    Ok(())
}

/// Log device VRAM `(used / free / total MiB)` at an agent-OPD milestone when
/// `ARLE_OPD_VRAM_TRACE` is set (default off — zero overhead on the hot path).
/// Attributes the resident-floor + writeback transient breakdown without
/// `nvidia-smi`. Uses the train backend's bound CUDA context (the single-GPU
/// agent-OPD device), so `device_mem_info` reflects the whole-process resident.
#[cfg(feature = "cuda")]
pub(super) fn log_opd_vram(label: &str, backend: &std::sync::Arc<dyn autograd::Backend>) {
    if std::env::var("ARLE_OPD_VRAM_TRACE").is_err() {
        return;
    }
    match backend.device_mem_info() {
        Some((free, total)) => {
            let used = total.saturating_sub(free);
            eprintln!(
                "[opd-vram] {label}: used={}MiB free={}MiB total={}MiB hoarded={}",
                used >> 20,
                free >> 20,
                total >> 20,
                train::opd::fmt_hoarded(backend.hoarded_mib()),
            );
        }
        None => eprintln!("[opd-vram] {label}: device_mem_info unavailable"),
    }
}

/// The subset of a model's parameter ids the optimizer will actually update.
pub(super) fn trainable_param_ids(all_params: &[TensorId], store: &TensorStore) -> Vec<TensorId> {
    all_params
        .iter()
        .copied()
        .filter(|id| store.get(*id).is_some_and(|tensor| tensor.requires_grad))
        .collect()
}
