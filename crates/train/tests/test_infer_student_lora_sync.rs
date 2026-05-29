#![cfg(all(feature = "cuda", not(feature = "no-cuda")))]

//! OPD Phase P2 canary: per-step student LoRA sync correctness.
//!
//! Validates the in-memory re-merge sync (`InferStudent::sync_lora_from_store`
//! -> infer `remerge_student_lora`) against the train-crate student's own
//! greedy argmax on the same sequence.
//!
//! - **Step A (numeric floor)**: zero LoRA (B zero-init -> student == base).
//!   Greedy rollout through BOTH paths; token agreement % is the
//!   BF16(infer)-vs-F32(train) floor — argmax flips here are pure numeric
//!   rounding, not a sync bug.
//! - **Step B (sync correctness)**: set a small non-zero LoRA B in the train
//!   student, `sync_lora_from_store`, rollout BOTH paths again.
//!
//! PASS: Step B agreement >= 90% (sync is correct; infer-student tracks the
//! train-student). KILL-investigate 60-90%; KILL (sync bug) < 60%.
//!
//! Run: `cargo test --release -p train --features cuda \
//!   --test test_infer_student_lora_sync -- --ignored --nocapture`

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use autograd::{Backend, Tape, TensorStore, backend_cuda::CudaBackend};
use infer::server_engine::{InferenceEngineOptions, LoadedInferenceEngine, ServerRuntimeConfig};
use safetensors::tensor::{Dtype, View};
use std::borrow::Cow;
use train::{
    LoraConfig, infer_student::InferStudent, lora::LoraTargetSet,
    qwen35_loader::load_qwen35_lora_from_hf_dir,
};

const DEFAULT_QWEN35_08B_DIR: &str = "/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base";
const ROLLOUT_TOKENS: usize = 64;
const MAX_SEQ_LEN: usize = 256;
const LORA_RANK: usize = 8;
const LORA_ALPHA: f32 = 16.0;

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

/// A raw F32 safetensors matrix view, used to write the adapter the infer
/// engine loads at bootstrap.
struct F32Tensor {
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl F32Tensor {
    fn new(shape: Vec<usize>, values: &[f32]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self { shape, data }
    }
}

impl View for &F32Tensor {
    fn dtype(&self) -> Dtype {
        Dtype::F32
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Map a train adapter name (`...layers.7.self_attn.q_proj.weight.lora_a`) to
/// the PEFT on-disk name infer's loader expects.
fn peft_name(internal: &str) -> Option<String> {
    let (base, suffix) = if let Some(b) = internal.strip_suffix(".lora_a") {
        (b, "lora_A")
    } else if let Some(b) = internal.strip_suffix(".lora_b") {
        (b, "lora_B")
    } else {
        return None;
    };
    let base = base.strip_suffix(".weight")?;
    Some(format!("base_model.model.{base}.{suffix}.weight"))
}

/// Write the train student's current q/v adapter (raw A/B) to a PEFT adapter
/// dir so the infer engine snapshots the pristine base at load and seeds the
/// matching adapter.
fn write_adapter_dir(
    dir: &Path,
    store: &mut TensorStore,
    adapter_map: &std::collections::HashMap<&'static str, autograd::TensorId>,
) -> TestResult {
    let mut tensors: BTreeMap<String, F32Tensor> = BTreeMap::new();
    for (&name, &id) in adapter_map {
        let Some(peft) = peft_name(name) else {
            continue;
        };
        // Only q/v adapters (AttentionQv target set) are relevant.
        if !(name.contains(".q_proj.") || name.contains(".v_proj.")) {
            continue;
        }
        let shape = store.get(id).expect("adapter tensor").shape.clone();
        let values = store.to_host(id)?;
        tensors.insert(peft, F32Tensor::new(shape, &values));
    }
    assert!(
        !tensors.is_empty(),
        "no q/v adapters to write; check target set"
    );
    let views: BTreeMap<String, &F32Tensor> = tensors.iter().map(|(k, v)| (k.clone(), v)).collect();
    safetensors::serialize_to_file(views, None, &dir.join("adapter_model.safetensors"))?;
    fs::write(
        dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "r": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "target_modules": ["q_proj", "v_proj"],
        }))?,
    )?;
    Ok(())
}

/// Greedy argmax over the last position of the train student's F32 logits.
fn train_next_token(
    model: &train::qwen35::Qwen35Model,
    store: &mut TensorStore,
    sequence: &[u32],
    vocab: usize,
    keep: &std::collections::HashSet<autograd::TensorId>,
) -> TestResult2<u32> {
    let mut tape = Tape::new();
    let seq_usize: Vec<usize> = sequence.iter().map(|&t| t as usize).collect();
    let logits = model.forward_tokens(&seq_usize, store, &mut tape)?;
    let host = store.to_host(logits)?;
    let last = &host[(sequence.len() - 1) * vocab..sequence.len() * vocab];
    let token = argmax(last) as u32;
    // Free per-step forward intermediates so the O(n^2) reference rollout does
    // not exhaust VRAM alongside the in-process infer engine.
    store.retain_ids(keep);
    Ok(token)
}

type TestResult2<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = v[0];
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

/// Run a `ROLLOUT_TOKENS` greedy rollout through both the infer student and
/// the train student (each appending its own argmax) and return token
/// agreement % over the generated tail.
fn rollout_agreement(
    student: &InferStudent,
    train_model: &train::qwen35::Qwen35Model,
    store: &mut TensorStore,
    prompt: &[u32],
    vocab: usize,
    keep: &std::collections::HashSet<autograd::TensorId>,
) -> TestResult2<(f64, Vec<u32>, Vec<u32>)> {
    let mut infer_seq = prompt.to_vec();
    let mut train_seq = prompt.to_vec();
    let mut agree = 0usize;
    for _ in 0..ROLLOUT_TOKENS {
        let positions: Vec<u32> = (0..infer_seq.len() as u32).collect();
        let infer_tok = student.decode_next_token(&infer_seq, &positions)?;
        let train_tok = train_next_token(train_model, store, &train_seq, vocab, keep)?;
        if infer_tok == train_tok {
            agree += 1;
        }
        infer_seq.push(infer_tok);
        train_seq.push(train_tok);
    }
    let pct = 100.0 * agree as f64 / ROLLOUT_TOKENS as f64;
    Ok((
        pct,
        infer_seq[prompt.len()..].to_vec(),
        train_seq[prompt.len()..].to_vec(),
    ))
}

fn load_infer_engine(model_dir: &Path) -> anyhow::Result<LoadedInferenceEngine> {
    let mut runtime = ServerRuntimeConfig {
        engine: InferenceEngineOptions {
            enable_cuda_graph: false,
        },
        max_seq_len: Some(MAX_SEQ_LEN),
        ..ServerRuntimeConfig::default()
    };
    runtime.scheduler.max_slots = 1;
    runtime.scheduler.chunked_prefill_size = MAX_SEQ_LEN;
    runtime.scheduler.max_num_batched_tokens = MAX_SEQ_LEN;
    runtime.scheduler.max_prefill_tokens = MAX_SEQ_LEN;
    runtime.scheduler.long_prefill_token_threshold = MAX_SEQ_LEN;
    runtime.scheduler.prefill_max_requests = Some(1);
    runtime.scheduler.mem_fraction_static = 0.05;
    runtime.scheduler.kv_pool_fallback_bytes = 128 * 1024 * 1024;
    LoadedInferenceEngine::load_with_runtime_config(
        model_dir
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?,
        runtime,
    )
}

#[test]
#[ignore = "GPU + Qwen3.5-0.8B weights; run explicitly with --ignored"]
fn infer_student_lora_sync_tracks_train_student() -> TestResult {
    let Some(model_dir) = resolve_qwen35_08b_dir() else {
        eprintln!(
            "infer_student_lora_sync_tracks_train_student: skipping; \
             set ARLE_PARITY_QWEN35_08B_DIR or populate {DEFAULT_QWEN35_08B_DIR}"
        );
        return Ok(());
    };

    let lora_config = LoraConfig {
        rank: LORA_RANK,
        alpha: LORA_ALPHA,
    };

    // Train-side LoRA student (AttentionQv target set -> q/v adapters on the
    // 6 full-attention layers only). B is zero-init at this point.
    let backend: Arc<dyn Backend> = Arc::new(CudaBackend::new(0)?);
    let mut store = TensorStore::with_backend(backend.clone());
    let train_model = load_qwen35_lora_from_hf_dir(
        &model_dir,
        lora_config,
        LoraTargetSet::AttentionQv,
        &mut store,
    )?;
    let vocab = train_model.config().vocab_size;
    let adapter_map = train_model.adapter_name_map();
    eprintln!(
        "train LoRA student: {} adapter tensors (q/v across full-attn layers)",
        adapter_map.len()
    );

    // Seed the infer engine with the matching (zero-B) adapter so it caches the
    // pristine base at load. Then both paths share base weights for Step A.
    let adapter_dir = tempfile::tempdir()?;
    write_adapter_dir(adapter_dir.path(), &mut store, &adapter_map)?;
    // SAFETY: single-threaded test setup before the engine spawns.
    unsafe {
        std::env::set_var("INFER_LORA_PATH", adapter_dir.path());
    }
    let infer_engine = load_infer_engine(&model_dir)?;
    unsafe {
        std::env::remove_var("INFER_LORA_PATH");
    }
    let student = InferStudent::new(Arc::new(Mutex::new(infer_engine)), backend, vocab);

    let prompt: Vec<u32> = vec![
        9419, 374, 264, 1273, 9934, 369, 279, 4128, 1614, 13, 5651, 752, 911, 432, 25, 220,
    ];
    assert!(prompt.iter().all(|&t| (t as usize) < vocab));

    // Stable model-parameter id set so the reference rollout can free its
    // O(n^2) forward intermediates without dropping weights/adapters.
    let keep: std::collections::HashSet<autograd::TensorId> =
        train_model.all_parameter_ids().into_iter().collect();

    // === Step A: zero-LoRA numeric floor ===
    let (floor_pct, infer_tail_a, train_tail_a) =
        rollout_agreement(&student, &train_model, &mut store, &prompt, vocab, &keep)?;
    eprintln!("=== Step A: zero-LoRA floor (BF16 infer vs F32 train) ===");
    eprintln!("agreement: {floor_pct:.1}% over {ROLLOUT_TOKENS} tokens");
    eprintln!(
        "infer tail: {:?}",
        &infer_tail_a[infer_tail_a.len().saturating_sub(8)..]
    );
    eprintln!(
        "train tail: {:?}",
        &train_tail_a[train_tail_a.len().saturating_sub(8)..]
    );

    // === Step B: set a small non-zero LoRA B, sync, re-roll ===
    // B is [out_features, rank]; fill with a small deterministic pattern so the
    // adapter is a real (non-identity) perturbation but stays numerically tame.
    for (&name, &id) in &adapter_map {
        if !name.ends_with(".lora_b") {
            continue;
        }
        if !(name.contains(".q_proj.") || name.contains(".v_proj.")) {
            continue;
        }
        let tensor = store.get_mut(id).expect("lora_b tensor");
        for (i, slot) in tensor.data.iter_mut().enumerate() {
            // Small, varied, deterministic values.
            *slot = 0.01 * (((i % 7) as f32) - 3.0);
        }
    }

    student.sync_lora_from_store(&mut store, &adapter_map, lora_config)?;

    let (sync_pct, infer_tail_b, train_tail_b) =
        rollout_agreement(&student, &train_model, &mut store, &prompt, vocab, &keep)?;
    eprintln!("=== Step B: non-zero LoRA sync correctness ===");
    eprintln!("agreement: {sync_pct:.1}% over {ROLLOUT_TOKENS} tokens");
    eprintln!(
        "infer tail: {:?}",
        &infer_tail_b[infer_tail_b.len().saturating_sub(8)..]
    );
    eprintln!(
        "train tail: {:?}",
        &train_tail_b[train_tail_b.len().saturating_sub(8)..]
    );

    let verdict = if sync_pct >= 90.0 {
        "PASS (sync correct)"
    } else if sync_pct >= 60.0 {
        "KILL-INVESTIGATE (numeric divergence)"
    } else {
        "KILL (sync bug)"
    };
    eprintln!("=== VERDICT: {verdict} | floor={floor_pct:.1}% sync={sync_pct:.1}% ===");

    // The non-zero adapter must actually change the train student's output vs
    // the zero-LoRA tail; otherwise the sync test is vacuous.
    assert_ne!(
        train_tail_a, train_tail_b,
        "non-zero LoRA B did not change the train student's rollout; test is vacuous"
    );

    assert!(
        sync_pct >= 60.0,
        "Step B agreement {sync_pct:.1}% < 60% indicates a LoRA-sync bug, not BF16 rounding"
    );

    Ok(())
}
