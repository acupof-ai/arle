#![cfg(feature = "cuda")]

use std::path::PathBuf;

use anyhow::{Context, Result};
use cuda_kernels::TokenKVPool;
use infer::model::{
    GenerationState, KVFormat, ModelForward, PrefillBatchRequest, Qwen35Model, Qwen35RuntimeConfig,
};
use infer::model_arch::ModelArchInfo;
use tokenizers::Tokenizer;

fn model_dir() -> PathBuf {
    std::env::var("ARLE_PROBE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base")
        })
}

fn prompt() -> String {
    let reps = std::env::var("ARLE_PROBE_REPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10);
    format!("{}The capital of France is", "Hello world. ".repeat(reps))
}

fn probe_kv_format() -> KVFormat {
    match std::env::var("ARLE_PROBE_KV_FORMAT")
        .unwrap_or_else(|_| "bf16".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "fp8" | "fp8e4m3" => KVFormat::FP8E4M3,
        "bf16" => KVFormat::BF16,
        other => panic!("unsupported ARLE_PROBE_KV_FORMAT={other:?}; use bf16 or fp8e4m3"),
    }
}

fn topk(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    indexed.truncate(k);
    indexed
}

fn decode_token(tokenizer: &Tokenizer, token_id: usize) -> String {
    tokenizer
        .decode(&[token_id as u32], false)
        .unwrap_or_else(|_| "<decode-error>".to_string())
}

fn print_topk(label: &str, logits: &[f32], tokenizer: &Tokenizer) {
    let top = topk(logits, 8);
    println!("{label}:");
    for (rank, (token, value)) in top.iter().enumerate() {
        println!(
            "  #{rank}: id={token} logit={value:.6} text={:?}",
            decode_token(tokenizer, *token)
        );
    }
}

fn max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn sequential_logits(model: &Qwen35Model, tokens: &[u32]) -> Result<Vec<f32>> {
    let mut state = model.create_state()?;
    state.set_max_seq_len(tokens.len().max(1));
    let mut logits = Vec::new();
    for &token in tokens {
        let (_, current) = model.forward_with_logits(&[token], &mut state)?;
        logits = current.to_host(model.device_context())?;
    }
    Ok(logits)
}

fn contiguous_prefill_logits(model: &Qwen35Model, tokens: &[u32]) -> Result<Vec<f32>> {
    let mut state = model.create_state()?;
    state.set_max_seq_len(tokens.len().max(1));
    model.forward_prefill(tokens, &mut state)?;
    state.logits().to_host(model.device_context())
}

fn paged_prefill_logits(model: &Qwen35Model, tokens: &[u32]) -> Result<Vec<f32>> {
    let mut states = vec![model.create_state()?];
    states[0].set_max_seq_len(tokens.len().max(1));
    let kv_format = probe_kv_format();
    let budget = TokenKVPool::budget_bytes_for_tokens(
        model.num_kv_layers(),
        model.num_kv_heads(),
        model.head_dim(),
        tokens.len().max(1),
        kv_format,
    );
    let mut pool = TokenKVPool::with_format(
        model.device_context(),
        model.num_kv_layers(),
        model.num_kv_heads(),
        model.head_dim(),
        1,
        budget,
        kv_format,
    )?;
    model.forward_prefill_batch(
        &[PrefillBatchRequest {
            slot_idx: 0,
            tokens,
            start_pos: 0,
            total_tokens: tokens.len(),
        }],
        &mut states,
        Some(&mut pool),
    )?;
    states[0].logits().to_host(model.device_context())
}

fn main() -> Result<()> {
    infer::logging::init_stderr("warn");
    let model_dir = model_dir();
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|err| anyhow::anyhow!("load tokenizer: {err}"))?;
    let text = prompt();
    let encoding = tokenizer
        .encode(text.as_str(), false)
        .map_err(|err| anyhow::anyhow!("encode prompt: {err}"))?;
    let tokens = encoding.get_ids().to_vec();

    println!("model_dir={}", model_dir.display());
    println!("prompt_tokens={}", tokens.len());
    println!("prompt={text:?}");

    let model = Qwen35Model::from_safetensors_with_runtime(
        model_dir
            .to_str()
            .context("probe model path is not valid UTF-8")?,
        Qwen35RuntimeConfig {
            enable_cuda_graph: false,
            ..Qwen35RuntimeConfig::default()
        },
    )?;

    let sequential = sequential_logits(&model, &tokens)?;
    let contiguous = contiguous_prefill_logits(&model, &tokens)?;
    let paged = paged_prefill_logits(&model, &tokens)?;

    print_topk("sequential_decode", &sequential, &tokenizer);
    print_topk("contiguous_prefill", &contiguous, &tokenizer);
    print_topk("paged_prefill_batch", &paged, &tokenizer);
    println!(
        "delta sequential_vs_contiguous max_abs={:.6}",
        max_abs_delta(&sequential, &contiguous)
    );
    println!(
        "delta contiguous_vs_paged max_abs={:.6}",
        max_abs_delta(&contiguous, &paged)
    );

    Ok(())
}
