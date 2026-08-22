use autograd::{Tape, TensorId, TensorStore};
#[cfg(feature = "cuda")]
use {
    anyhow::{Context, Result, anyhow},
    std::path::Path,
};

pub(super) enum OpdCliTeacher<'a> {
    InProcess(train::teacher_infer::InProcessTeacher<'a>),
    #[cfg(feature = "cuda")]
    Infer(train::teacher_infer::InferTeacher),
    Api(train::teacher_infer::ApiTeacher),
    CorpusSftOnly {
        vocab_size: usize,
    },
}

impl train::teacher_infer::TeacherForward for OpdCliTeacher<'_> {
    fn forward_logits_device(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> std::result::Result<
        train::teacher_infer::DeviceLogits,
        train::teacher_infer::TeacherForwardError,
    > {
        match self {
            Self::InProcess(teacher) => {
                train::teacher_infer::TeacherForward::forward_logits_device(
                    teacher, input_ids, positions, store, tape,
                )
            }
            #[cfg(feature = "cuda")]
            Self::Infer(teacher) => train::teacher_infer::TeacherForward::forward_logits_device(
                teacher, input_ids, positions, store, tape,
            ),
            Self::Api(teacher) => train::teacher_infer::TeacherForward::forward_logits_device(
                teacher, input_ids, positions, store, tape,
            ),
            Self::CorpusSftOnly { .. } => {
                Err(train::teacher_infer::TeacherForwardError::InvalidInput(
                    "corpus-truth SFT-only teacher was asked to score KL logits; \
                     this path is only valid with --sft-anchor corpus-truth --gkd-lambda 1.0"
                        .to_owned(),
                ))
            }
        }
    }

    fn forward_logits_window_device(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        window: train::qwen35::SequenceWindow,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> std::result::Result<
        train::teacher_infer::DeviceLogits,
        train::teacher_infer::TeacherForwardError,
    > {
        match self {
            Self::InProcess(teacher) => {
                train::teacher_infer::TeacherForward::forward_logits_window_device(
                    teacher, input_ids, positions, window, store, tape,
                )
            }
            #[cfg(feature = "cuda")]
            Self::Infer(teacher) => {
                train::teacher_infer::TeacherForward::forward_logits_window_device(
                    teacher, input_ids, positions, window, store, tape,
                )
            }
            Self::Api(teacher) => {
                train::teacher_infer::TeacherForward::forward_logits_window_device(
                    teacher, input_ids, positions, window, store, tape,
                )
            }
            Self::CorpusSftOnly { .. } => {
                Err(train::teacher_infer::TeacherForwardError::InvalidInput(
                    "corpus-truth SFT-only teacher was asked to score windowed KL logits; \
                     this path is only valid with --sft-anchor corpus-truth --gkd-lambda 1.0"
                        .to_owned(),
                ))
            }
        }
    }

    fn forward_hidden_device(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> std::result::Result<TensorId, train::teacher_infer::TeacherForwardError> {
        match self {
            Self::InProcess(teacher) => {
                train::teacher_infer::TeacherForward::forward_hidden_device(
                    teacher, input_ids, positions, store, tape,
                )
            }
            _ => Err(train::teacher_infer::TeacherForwardError::InvalidInput(
                "forward_hidden_device only supported by in-process teacher".to_owned(),
            )),
        }
    }

    fn logits_from_hidden_window_device(
        &self,
        hidden: TensorId,
        window: train::qwen35::SequenceWindow,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> std::result::Result<
        train::teacher_infer::DeviceLogits,
        train::teacher_infer::TeacherForwardError,
    > {
        match self {
            Self::InProcess(teacher) => {
                train::teacher_infer::TeacherForward::logits_from_hidden_window_device(
                    teacher, hidden, window, store, tape,
                )
            }
            _ => Err(train::teacher_infer::TeacherForwardError::InvalidInput(
                "logits_from_hidden_window_device only supported by in-process teacher".to_owned(),
            )),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::InProcess(teacher) => train::teacher_infer::TeacherForward::vocab_size(teacher),
            #[cfg(feature = "cuda")]
            Self::Infer(teacher) => train::teacher_infer::TeacherForward::vocab_size(teacher),
            Self::Api(teacher) => train::teacher_infer::TeacherForward::vocab_size(teacher),
            Self::CorpusSftOnly { vocab_size } => *vocab_size,
        }
    }

    fn parameter_ids(&self) -> &[TensorId] {
        match self {
            Self::InProcess(teacher) => {
                train::teacher_infer::TeacherForward::parameter_ids(teacher)
            }
            #[cfg(feature = "cuda")]
            Self::Infer(teacher) => train::teacher_infer::TeacherForward::parameter_ids(teacher),
            Self::Api(teacher) => train::teacher_infer::TeacherForward::parameter_ids(teacher),
            Self::CorpusSftOnly { .. } => &[],
        }
    }

    fn offload_engine_weights(
        &self,
    ) -> std::result::Result<usize, train::teacher_infer::TeacherForwardError> {
        match self {
            Self::InProcess(teacher) => {
                train::teacher_infer::TeacherForward::offload_engine_weights(teacher)
            }
            #[cfg(feature = "cuda")]
            Self::Infer(teacher) => {
                train::teacher_infer::TeacherForward::offload_engine_weights(teacher)
            }
            Self::Api(teacher) => {
                train::teacher_infer::TeacherForward::offload_engine_weights(teacher)
            }
            Self::CorpusSftOnly { .. } => Ok(0),
        }
    }

    fn reload_engine_weights(
        &self,
    ) -> std::result::Result<(), train::teacher_infer::TeacherForwardError> {
        match self {
            Self::InProcess(teacher) => {
                train::teacher_infer::TeacherForward::reload_engine_weights(teacher)
            }
            #[cfg(feature = "cuda")]
            Self::Infer(teacher) => {
                train::teacher_infer::TeacherForward::reload_engine_weights(teacher)
            }
            Self::Api(teacher) => {
                train::teacher_infer::TeacherForward::reload_engine_weights(teacher)
            }
            Self::CorpusSftOnly { .. } => Ok(()),
        }
    }
}

#[cfg(feature = "cuda")]
pub(super) fn load_opd_infer_teacher(
    teacher_dir: &Path,
    max_seq_len: usize,
    mem_fraction_static: f64,
    train_backend: std::sync::Arc<dyn autograd::Backend>,
    vocab_size: usize,
    memory_budget_bytes: Option<usize>,
) -> Result<train::teacher_infer::InferTeacher> {
    use std::sync::{Arc, Mutex};

    use infer_api::{EngineLoadConfig, LoadedInferenceEngine};

    let max_seq_len = max_seq_len.max(128);
    eprintln!(
        "[arle train opd] loading infer teacher from {} (max_seq_len={max_seq_len})",
        teacher_dir.display()
    );
    let engine = LoadedInferenceEngine::load_with_config(
        teacher_dir
            .to_str()
            .ok_or_else(|| anyhow!("teacher model path is not valid UTF-8"))?,
        EngineLoadConfig {
            // Scoring is single-sequence; the default 0.9 fraction lets the teacher
            // pool starve a co-resident student on one GPU.
            mem_fraction_static,
            memory_budget_bytes,
            world_size: None,
            ..EngineLoadConfig::single_sequence(max_seq_len)
        },
    )
    .with_context(|| format!("load infer teacher from {}", teacher_dir.display()))?;

    Ok(train::teacher_infer::InferTeacher::new(
        Arc::new(Mutex::new(engine)),
        train_backend,
        vocab_size,
    ))
}

#[cfg(feature = "cuda")]
pub(super) fn maybe_preoffload_infer_teacher_before_steps(
    teacher: &OpdCliTeacher<'_>,
    train_backend: &std::sync::Arc<dyn autograd::Backend>,
) -> Result<()> {
    if !train::opd::engine_offload_mode().offloads_teacher() {
        return Ok(());
    }

    train_backend
        .device_synchronize()
        .context("synchronize train backend before infer teacher pre-step offload")?;
    let freed = train::teacher_infer::TeacherForward::offload_engine_weights(teacher)
        .map_err(|err| anyhow!("offload infer teacher before first OPD step failed: {err}"))?;
    eprintln!(
        "opd_engine_offload teacher_pre_step_offloaded freed_bytes={freed} freed_mib={:.1}",
        freed as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}
