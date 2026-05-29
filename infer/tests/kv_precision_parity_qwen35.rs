#![cfg(feature = "cuda")]

//! KV precision parity audit — Qwen3.5 hybrid (linear + full attention) variant.
//!
//! Mirrors `kv_precision_parity.rs` but loads `Qwen35Model` instead of
//! `Qwen3Model` so the audit can run on hardware that only has Qwen3.5
//! family weights cached (V100 box, 2026-05-26 onward).
//!
//! The Qwen3.5 architecture is a hybrid stack: only the
//! `num_full_attention_layers` (8 of 36 for the 4B variant) consume the
//! paged KV pool that the FP8 / INT8 / TurboQuant precisions affect. The
//! linear-attention layers run their own per-layer recurrent state in
//! `Qwen35State` and are unaffected by the `KVFormat` setting. Audit
//! interpretation therefore changes: a divergence here is dominated by
//! the full-attention layers' KV path, not the whole model. Even so, the
//! 2026-05-26 FP8 step-1 catastrophic divergence on Qwen3-4B should still
//! reproduce here for FP8 if the bug is in the shared paged-pool KV
//! quantize / decode dispatch — and confirming reproduction across model
//! families is the strongest signal we'd get without an integration
//! instrumentation pass.
//!
//! Run: `cargo test --release -p infer --features cuda --test
//! kv_precision_parity_qwen35 -- --nocapture --test-threads=1`. Knobs
//! (same as the dense variant): `KV_PARITY_PROMPTS`,
//! `KV_PARITY_MAX_TOKENS`, `KV_PARITY_MAX_SEQ_LEN`,
//! `INFER_TEST_MODEL_PATH` (must point at a Qwen3.5 dir),
//! `INFER_TEST_CUDA_GRAPH=0`. Skipped if the model dir is missing.

use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use log::info;
use tokio::sync::mpsc;

use infer::metrics::ServerMetrics;
use infer::model::qwen35::{Qwen35Model, Qwen35RuntimeConfig};
use infer::model::{KVCacheDtype, KVFormat};
use infer::sampler::SamplingParams;
use infer::scheduler::{IncomingRequest, RequestPriority, Scheduler, SchedulerConfig};
use infer::server_engine::CompletionStreamDelta;
use infer::tokenizer::Tokenizer;

/// On V100 the cached weights live in the modelscope hub; override via
/// `INFER_TEST_MODEL_PATH`. The default below is what the Qwen3.5 box
/// uses out of the box.
const DEFAULT_MODEL_HINT: &str = "models/Qwen3.5-4B";

const DEFAULT_PROMPTS: &[&str] = &[
    "Explain in detail how a transformer language model performs causal attention during decode. \
     Cover: KV caching, the role of the attention mask, why the past keys and values are reused, \
     and how rotary position embedding interacts with cached positions. Begin step by step.",
    "Write a Rust function that computes the n-th Fibonacci number using iterative dynamic \
     programming. Use only u128, return Option<u128> to signal overflow, and add inline doc \
     comments explaining the invariant maintained across the loop. Then explain the time and \
     space complexity in big-O terms.",
    "Summarize the differences between supervised fine-tuning, reinforcement learning from human \
     feedback, and on-policy distillation for large language model post-training. For each \
     approach explain: what data the model sees, how the loss is computed, and where the most \
     common failure modes appear in practice.",
    "Walk through the dynamics of training a small ResNet on CIFAR-10 with SGD + momentum. \
     Describe the typical curve of train loss vs validation loss over the first 30 epochs, \
     when overfitting becomes visible, what hyperparameters most influence the regime, and \
     why batch normalization changes the picture compared to plain GroupNorm.",
];

#[derive(Clone, Copy, Debug)]
struct PrecisionCase {
    name: &'static str,
    dtype: KVCacheDtype,
    format: KVFormat,
    gate_trajectory: Option<f32>,
}

fn precision_matrix() -> Vec<PrecisionCase> {
    let mut cases = vec![
        PrecisionCase {
            name: "bf16",
            dtype: KVCacheDtype::BF16,
            format: KVFormat::BF16,
            gate_trajectory: Some(1.0),
        },
        PrecisionCase {
            name: "int8",
            dtype: KVCacheDtype::INT8,
            format: KVFormat::INT8,
            gate_trajectory: Some(0.99),
        },
        PrecisionCase {
            name: "fp8",
            dtype: KVCacheDtype::BF16,
            format: KVFormat::FP8E4M3,
            gate_trajectory: None,
        },
        PrecisionCase {
            name: "tq4",
            dtype: KVCacheDtype::BF16,
            format: KVFormat::TurboQuant {
                key_bits: 4,
                val_bits: 4,
            },
            gate_trajectory: None,
        },
        // INT4 + KIVI per-channel K PoC (2026-05-27). Parallel to TQ4 in
        // memory footprint (4-bit packed) but uses per-channel K
        // calibration instead of Hadamard rotation for outlier handling.
        PrecisionCase {
            name: "int4",
            dtype: KVCacheDtype::BF16,
            format: KVFormat::INT4,
            gate_trajectory: None,
        },
    ];
    if matches!(
        std::env::var("KV_PARITY_INCLUDE_TQ23").as_deref(),
        Ok("1" | "true" | "TRUE")
    ) {
        cases.push(PrecisionCase {
            name: "tq3",
            dtype: KVCacheDtype::BF16,
            format: KVFormat::TurboQuant {
                key_bits: 3,
                val_bits: 3,
            },
            gate_trajectory: None,
        });
        cases.push(PrecisionCase {
            name: "tq2",
            dtype: KVCacheDtype::BF16,
            format: KVFormat::TurboQuant {
                key_bits: 2,
                val_bits: 2,
            },
            gate_trajectory: None,
        });
    }
    cases
}

#[derive(Debug)]
struct PrecisionResult {
    name: &'static str,
    sequences: Vec<Vec<u32>>,
    elapsed_secs: f64,
}

#[derive(Debug)]
struct DiffRow {
    name: &'static str,
    per_prompt_match: Vec<f32>,
    mean_match: f32,
    first_diverging_prompt: Option<usize>,
    first_diverging_step: Option<usize>,
    gate: Option<f32>,
    gate_passed: Option<bool>,
    elapsed_secs: f64,
}

fn get_model_path() -> String {
    std::env::var("INFER_TEST_MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_HINT.to_string())
}

fn max_tokens() -> usize {
    // Default tuned for the snappy iteration grid (4 prompts × 4 tokens).
    // Override with KV_PARITY_MAX_TOKENS=16 for the stress grid, 256+ for
    // long-horizon trajectory tracking.
    std::env::var("KV_PARITY_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
}

fn num_prompts() -> usize {
    std::env::var("KV_PARITY_PROMPTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PROMPTS.len())
}

fn max_seq_len_override() -> usize {
    std::env::var("KV_PARITY_MAX_SEQ_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5120)
}

fn init_logging() {
    static LOGGER: OnceLock<()> = OnceLock::new();
    LOGGER.get_or_init(|| {
        infer::logging::init_stderr("info");
    });
}

fn gpu_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn make_request(
    prompt: &str,
    tokens: usize,
) -> (
    IncomingRequest,
    mpsc::UnboundedReceiver<CompletionStreamDelta>,
) {
    let (tx, rx) = mpsc::unbounded_channel();
    let req = IncomingRequest {
        prompt: prompt.to_string(),
        prompt_tokens: None,
        max_tokens: tokens,
        sampling: SamplingParams::default(),
        stop: None,
        speculative: None,
        priority: RequestPriority::default(),
        session_id: None,
        ingress_numa_node: None,
        delta_tx: tx,
        trace_context: None,
        distributed: None,
    };
    (req, rx)
}

fn collect_tokens(rx: &mut mpsc::UnboundedReceiver<CompletionStreamDelta>) -> Vec<u32> {
    let mut ids = Vec::new();
    while let Some(delta) = rx.blocking_recv() {
        if !delta.token_ids.is_empty() {
            ids.extend(delta.token_ids);
        }
        if delta.finish_reason.is_some() {
            break;
        }
    }
    ids
}

fn run_precision(
    case: PrecisionCase,
    model_path: &str,
    prompts: &[&str],
    tokens: usize,
) -> Result<PrecisionResult> {
    let _guard = gpu_test_lock().lock().unwrap();
    let started = Instant::now();

    info!(
        "kv-parity-qwen35: booting precision={} dtype={:?} format={:?}",
        case.name, case.dtype, case.format
    );

    // SAFETY: set before any scheduler worker thread is spawned; the
    // gpu_test_lock serializes precisions.
    unsafe {
        std::env::set_var("INFER_DETERMINISTIC", "1");
    }

    let enable_cuda_graph = !matches!(
        std::env::var("INFER_TEST_CUDA_GRAPH").as_deref(),
        Ok("0" | "false" | "FALSE" | "off" | "OFF")
    );
    let model = Qwen35Model::from_safetensors_with_runtime(
        model_path,
        Qwen35RuntimeConfig {
            enable_cuda_graph,
            ..Qwen35RuntimeConfig::default()
        },
    )
    .context("failed to load Qwen3.5 model")?;
    let tokenizer = Tokenizer::from_file(model_path).context("failed to load tokenizer")?;

    let num_slots = 1;
    let config = SchedulerConfig::runtime_defaults(num_slots);
    let (scheduler, handle) = Scheduler::with_config(
        model,
        tokenizer,
        "kv-parity-qwen35",
        42,
        ServerMetrics::new("kv-parity-qwen35"),
        config,
        Some(max_seq_len_override()),
        case.dtype,
        case.format,
        None,
    )
    .context("failed to construct scheduler")?;

    let scheduler_thread = std::thread::spawn(move || scheduler.run());

    let mut sequences = Vec::with_capacity(prompts.len());
    for (idx, prompt) in prompts.iter().enumerate() {
        let (req, mut rx) = make_request(prompt, tokens);
        handle.submit(req).context("submit failed")?;
        let ids = collect_tokens(&mut rx);
        info!(
            "kv-parity-qwen35: precision={} prompt[{}/{}] tokens={}",
            case.name,
            idx + 1,
            prompts.len(),
            ids.len()
        );
        sequences.push(ids);
    }

    drop(handle);
    scheduler_thread
        .join()
        .map_err(|_| anyhow::anyhow!("scheduler thread panicked"))?;

    Ok(PrecisionResult {
        name: case.name,
        sequences,
        elapsed_secs: started.elapsed().as_secs_f64(),
    })
}

fn diff_against_reference(
    reference: &PrecisionResult,
    candidate: &PrecisionResult,
    gate: Option<f32>,
    tokens: usize,
) -> DiffRow {
    assert_eq!(reference.sequences.len(), candidate.sequences.len());

    // Dump first-N token IDs per (precision, prompt) so we can decode and
    // discriminate noise-fidelity from real quality drift. See
    // docs/experience/wins/2026-05-27-v100-kv-precision-parity-qwen35-4b.md
    // for the diagnostic rationale.
    for (idx, (ref_seq, cand_seq)) in reference
        .sequences
        .iter()
        .zip(candidate.sequences.iter())
        .enumerate()
    {
        let take = ref_seq.len().min(cand_seq.len()).min(16);
        let ref_head: Vec<u32> = ref_seq.iter().copied().take(take).collect();
        let cand_head: Vec<u32> = cand_seq.iter().copied().take(take).collect();
        eprintln!(
            "kv-parity-qwen35: {:<5} prompt{} first{} tokens: ref={:?} cand={:?}",
            candidate.name, idx, take, ref_head, cand_head
        );
    }

    let mut per_prompt_match = Vec::with_capacity(reference.sequences.len());
    let mut first_diverging_prompt = None;
    let mut first_diverging_step = None;
    for (idx, (ref_seq, cand_seq)) in reference
        .sequences
        .iter()
        .zip(candidate.sequences.iter())
        .enumerate()
    {
        let mut common = 0usize;
        while common < ref_seq.len()
            && common < cand_seq.len()
            && ref_seq[common] == cand_seq[common]
        {
            common += 1;
        }
        let denom = ref_seq.len().max(1);
        per_prompt_match.push(common as f32 / denom as f32);
        if first_diverging_prompt.is_none() && common < ref_seq.len().min(cand_seq.len()) {
            first_diverging_prompt = Some(idx);
            first_diverging_step = Some(common);
        }
        if first_diverging_prompt.is_none() && ref_seq.len() != cand_seq.len() {
            first_diverging_prompt = Some(idx);
            first_diverging_step = Some(ref_seq.len().min(cand_seq.len()));
        }
    }

    let mean_match = if per_prompt_match.is_empty() {
        0.0
    } else {
        per_prompt_match.iter().sum::<f32>() / per_prompt_match.len() as f32
    };
    let gate_passed = gate.map(|g| mean_match >= g - 1e-6);
    let _ = tokens;

    DiffRow {
        name: candidate.name,
        per_prompt_match,
        mean_match,
        first_diverging_prompt,
        first_diverging_step,
        gate,
        gate_passed,
        elapsed_secs: candidate.elapsed_secs,
    }
}

fn write_json_report(
    model_path: &str,
    tokens: usize,
    prompts: &[&str],
    rows: &[DiffRow],
) -> Result<std::path::PathBuf> {
    let basename = Path::new(model_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("target");
    std::fs::create_dir_all(&dir).ok();
    let out = dir.join(format!("kv-parity-qwen35-{basename}-{unix}.json"));

    let mut buf = String::new();
    buf.push_str("{\n");
    buf.push_str(&format!("  \"model\": \"{basename}\",\n"));
    buf.push_str(&format!("  \"unix_ts\": {unix},\n"));
    buf.push_str(&format!("  \"max_tokens\": {tokens},\n"));
    buf.push_str(&format!("  \"num_prompts\": {},\n", prompts.len()));
    buf.push_str("  \"precisions\": [\n");
    for (i, row) in rows.iter().enumerate() {
        let trailing = if i + 1 == rows.len() { "" } else { "," };
        buf.push_str("    {\n");
        buf.push_str(&format!("      \"name\": \"{}\",\n", row.name));
        buf.push_str(&format!("      \"mean_match\": {:.6},\n", row.mean_match));
        buf.push_str(&format!(
            "      \"per_prompt_match\": [{}],\n",
            row.per_prompt_match
                .iter()
                .map(|v| format!("{v:.4}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        match (row.first_diverging_prompt, row.first_diverging_step) {
            (Some(p), Some(s)) => {
                buf.push_str(&format!("      \"first_diverging_prompt\": {p},\n"));
                buf.push_str(&format!("      \"first_diverging_step\": {s},\n"));
            }
            _ => {
                buf.push_str("      \"first_diverging_prompt\": null,\n");
                buf.push_str("      \"first_diverging_step\": null,\n");
            }
        }
        match (row.gate, row.gate_passed) {
            (Some(g), Some(p)) => {
                buf.push_str(&format!("      \"gate\": {g:.4},\n"));
                buf.push_str(&format!("      \"gate_passed\": {p},\n"));
            }
            _ => {
                buf.push_str("      \"gate\": null,\n");
                buf.push_str("      \"gate_passed\": null,\n");
            }
        }
        buf.push_str(&format!(
            "      \"elapsed_secs\": {:.2}\n",
            row.elapsed_secs
        ));
        buf.push_str(&format!("    }}{trailing}\n"));
    }
    buf.push_str("  ]\n}\n");
    std::fs::write(&out, buf).context("write kv-parity-qwen35 report")?;
    Ok(out)
}

#[test]
fn kv_precision_parity_audit_qwen35() -> Result<()> {
    init_logging();

    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!(
            "skipping kv_precision_parity_audit_qwen35: model path missing ({})",
            model_path
        );
        return Ok(());
    }

    let tokens = max_tokens();
    let prompts: Vec<&str> = DEFAULT_PROMPTS
        .iter()
        .take(num_prompts())
        .copied()
        .collect();
    let cases = precision_matrix();

    info!(
        "kv-parity-qwen35: model={} tokens/prompt={} prompts={} precisions={}",
        model_path,
        tokens,
        prompts.len(),
        cases.len()
    );

    let bf16_case = cases
        .iter()
        .find(|c| c.name == "bf16")
        .copied()
        .expect("bf16 must be in matrix");
    let reference = run_precision(bf16_case, &model_path, &prompts, tokens)?;

    let mut rows = Vec::with_capacity(cases.len());
    rows.push(diff_against_reference(
        &reference,
        &reference,
        bf16_case.gate_trajectory,
        tokens,
    ));

    for case in cases.iter().filter(|c| c.name != "bf16") {
        match run_precision(*case, &model_path, &prompts, tokens) {
            Ok(result) => {
                rows.push(diff_against_reference(
                    &reference,
                    &result,
                    case.gate_trajectory,
                    tokens,
                ));
            }
            Err(err) => {
                eprintln!(
                    "kv-parity-qwen35: precision={} failed to boot or run: {err:#}",
                    case.name
                );
                rows.push(DiffRow {
                    name: case.name,
                    per_prompt_match: vec![],
                    mean_match: 0.0,
                    first_diverging_prompt: Some(0),
                    first_diverging_step: Some(0),
                    gate: case.gate_trajectory,
                    gate_passed: Some(false),
                    elapsed_secs: 0.0,
                });
            }
        }
    }

    let report = write_json_report(&model_path, tokens, &prompts, &rows)?;
    eprintln!("kv-parity-qwen35 report: {}", report.display());
    for row in &rows {
        eprintln!(
            "kv-parity-qwen35: {:<6} mean_match={:.4} first_div={:?}/{:?} gate={:?} passed={:?} elapsed={:.1}s",
            row.name,
            row.mean_match,
            row.first_diverging_prompt,
            row.first_diverging_step,
            row.gate,
            row.gate_passed,
            row.elapsed_secs,
        );
    }

    let failed: Vec<&DiffRow> = rows
        .iter()
        .filter(|r| matches!(r.gate_passed, Some(false)))
        .collect();
    if !failed.is_empty() {
        let mut msg = String::from("KV precision parity gate failures (Qwen3.5):\n");
        for row in &failed {
            msg.push_str(&format!(
                "  - {}: mean_match={:.4} < gate={:.4} (first divergence prompt={:?} step={:?})\n",
                row.name,
                row.mean_match,
                row.gate.unwrap_or(0.0),
                row.first_diverging_prompt,
                row.first_diverging_step,
            ));
        }
        msg.push_str(&format!("Full report: {}\n", report.display()));
        panic!("{msg}");
    }

    Ok(())
}
