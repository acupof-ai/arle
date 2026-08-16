#![cfg(all(feature = "cuda", not(feature = "no-cuda")))]

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use autograd::{Backend, Tape, TensorStore, backend_cuda::CudaBackend};
use infer_api::{EngineLoadConfig, LoadedInferenceEngine};
use train::{
    qwen35::SequenceWindow,
    qwen35_loader::load_qwen35_from_hf_dir,
    teacher_infer::{InProcessTeacher, InferTeacher, TeacherForward},
};

const DEFAULT_QWEN35_08B_DIR: &str = "/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base";
const DOMINANT_TOP_K: usize = 64;
const DOMINANT_RELERR_GATE: f32 = 5.0e-2;

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

fn resolve_qwen35_08b_dir() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("ARLE_PARITY_QWEN35_08B_DIR") {
        let path = PathBuf::from(explicit);
        if path.is_dir() {
            return Some(path);
        }
    }
    let path = PathBuf::from(DEFAULT_QWEN35_08B_DIR);
    if path.is_dir() && path.join("config.json").is_file() {
        return Some(path);
    }
    None
}

#[test]
fn infer_teacher_matches_in_process_on_dominant_logits() -> TestResult {
    let Some(model_dir) = resolve_qwen35_08b_dir() else {
        eprintln!(
            "infer_teacher_matches_in_process_on_dominant_logits: skipping; \
             set ARLE_PARITY_QWEN35_08B_DIR or populate {DEFAULT_QWEN35_08B_DIR}"
        );
        return Ok(());
    };

    let backend: Arc<dyn Backend> = Arc::new(CudaBackend::new(0)?);
    let mut store = TensorStore::with_backend(backend.clone());
    let mut tape = Tape::new();
    tape.set_enabled(false);

    let train_model = load_qwen35_from_hf_dir(&model_dir, &mut store)?;
    let in_process = InProcessTeacher::new(&train_model);
    let infer_engine = load_infer_engine(&model_dir)?;
    let infer_teacher = InferTeacher::new(
        Arc::new(Mutex::new(infer_engine)),
        backend,
        train_model.config().vocab_size,
    );

    let input_ids = [9419u32];
    let positions = [0u32];
    let in_process_logits =
        in_process.forward_logits_device(&input_ids, &positions, &mut store, &mut tape)?;
    let infer_logits =
        infer_teacher.forward_logits_device(&input_ids, &positions, &mut store, &mut tape)?;

    assert_eq!(infer_logits.shape, in_process_logits.shape);
    let in_process_host = store.to_host(in_process_logits.tensor_id)?;
    let infer_host = store.to_host(infer_logits.tensor_id)?;
    assert_eq!(infer_host.len(), in_process_host.len());

    let relerr = dominant_relerr(&in_process_host, &infer_host, DOMINANT_TOP_K);
    eprintln!(
        "infer_teacher_matches_in_process_on_dominant_logits top_k={DOMINANT_TOP_K} relerr={relerr:e}"
    );
    assert!(
        relerr <= DOMINANT_RELERR_GATE,
        "InferTeacher vs InProcessTeacher top-{DOMINANT_TOP_K} relerr {relerr:e} exceeds \
         BF16-realistic gate {DOMINANT_RELERR_GATE:e}"
    );

    let window_err = infer_teacher
        .forward_logits_window_device(
            &[9419u32, 9450],
            &[0u32, 1],
            SequenceWindow { start: 1, end: 2 },
            &mut store,
            &mut tape,
        )
        .expect_err("InferTeacher must not fake windowed logits via full raw logits");
    assert!(
        window_err
            .to_string()
            .contains("does not support true windowed logits"),
        "unexpected InferTeacher window error: {window_err}"
    );

    Ok(())
}

fn load_infer_engine(model_dir: &Path) -> anyhow::Result<LoadedInferenceEngine> {
    // Single-forward logits-parity check (not serving): one slot, a small KV
    // budget. (Migrated off the legacy ServerRuntimeConfig knobs to the rewrite's
    // EngineLoadConfig; 8 pages x 16 = 128 token capacity matches the old 128 cap.)
    LoadedInferenceEngine::load_with_config(
        model_dir
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?,
        false, // enable_cuda_graph
        EngineLoadConfig {
            num_slots: 1,
            total_pages: 8,
            page_size: 16,
            tp_size: None,
            ..EngineLoadConfig::default()
        },
    )
}

fn dominant_relerr(reference: &[f32], candidate: &[f32], top_k: usize) -> f32 {
    if top_k == 0 {
        return 0.0;
    }
    let mut indices: Vec<usize> = (0..reference.len()).collect();
    indices.sort_unstable_by(|&a, &b| reference[b].abs().total_cmp(&reference[a].abs()));
    indices
        .into_iter()
        .take(top_k.min(reference.len()))
        .map(|index| {
            (candidate[index] - reference[index]).abs() / reference[index].abs().max(1.0e-6)
        })
        .fold(0.0f32, f32::max)
}
