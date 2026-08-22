#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("metal_kv_memory_probe requires `--features metal,no-cuda`");
    std::process::exit(2);
}

#[cfg(feature = "metal")]
fn main() -> anyhow::Result<()> {
    use infer_api::{
        CompletionRequest, EngineLoadConfig, InferenceEngine, KvCacheDtype, LoadedInferenceEngine,
        SamplingParams,
    };
    use serde_json::json;
    use std::time::Instant;

    let mut model = "mlx-community/Qwen3.6-35B-A3B-4bit".to_string();
    let mut dtype = KvCacheDtype::Auto;
    let mut target_prompt_tokens = 4096usize;
    let mut max_tokens = 8usize;
    let mut total_pages = EngineLoadConfig::default().total_pages;
    let mut chunked_prefill_size = 64usize;
    let mut memory_budget_bytes: Option<usize> = None;
    let mut low_impact = true;
    let mut warmup_runs = 0usize;
    let mut repeat = 1usize;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" | "--model-path" => {
                model = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))?;
            }
            "--kv-cache-dtype" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))?;
                dtype = match value.as_str() {
                    "auto" => KvCacheDtype::Auto,
                    "bf16" => KvCacheDtype::Bf16,
                    "int8" => KvCacheDtype::Int8,
                    other => anyhow::bail!("unsupported --kv-cache-dtype {other}"),
                };
            }
            "--prompt-tokens" => {
                target_prompt_tokens = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))?
                    .parse()?;
            }
            "--max-tokens" => {
                max_tokens = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))?
                    .parse()?;
            }
            "--total-pages" => {
                total_pages = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))?
                    .parse()?;
            }
            "--chunked-prefill-size" => {
                chunked_prefill_size = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))?
                    .parse()?;
            }
            "--memory-budget-gib" => {
                let gib: usize = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))?
                    .parse()?;
                memory_budget_bytes = Some(gib.saturating_mul(1 << 30));
            }
            "--low-impact" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("{arg} needs true|false"))?;
                low_impact = match value.as_str() {
                    "1" | "true" => true,
                    "0" | "false" => false,
                    other => anyhow::bail!("unsupported --low-impact {other}; use true|false"),
                };
            }
            "--warmup-runs" => {
                warmup_runs = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))?
                    .parse()?;
            }
            "--repeat" => {
                repeat = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))?
                    .parse()?;
            }
            "--help" | "-h" => {
                println!(
                    "usage: cargo run -p infer-api --example metal_kv_memory_probe --release --no-default-features --features metal,no-cuda -- [--model MODEL] [--kv-cache-dtype auto|bf16|int8] [--prompt-tokens N] [--max-tokens N] [--memory-budget-gib N] [--low-impact true|false] [--warmup-runs N] [--repeat N]"
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    anyhow::ensure!(repeat >= 1, "repeat must be >= 1");

    let mut config = EngineLoadConfig {
        num_slots: 1,
        total_pages,
        chunked_prefill_size: Some(chunked_prefill_size),
        max_prompt_tokens: target_prompt_tokens.saturating_add(1024),
        max_total_tokens: target_prompt_tokens
            .saturating_add(max_tokens)
            .saturating_add(1024),
        kv_cache_dtype: dtype,
        memory_budget_bytes,
        low_impact,
        ..EngineLoadConfig::default()
    };
    let min_pages = config.max_total_tokens.div_ceil(config.page_size);
    if config.total_pages < min_pages {
        config.total_pages = min_pages;
    }

    let mut engine = LoadedInferenceEngine::load_with_config(&model, config)?;
    infer_metal::clear_metal_cache();
    let after_load = infer_metal::allocator_memory();

    let seed = "Metal int8 KV memory probe shared long-context sentence. ";
    let mut prompt = String::new();
    loop {
        prompt.push_str(seed);
        let len = engine.tokenize(&prompt)?.len();
        if len >= target_prompt_tokens {
            break;
        }
    }
    let prompt_tokens = engine.tokenize(&prompt)?.len();
    let mut runs = Vec::with_capacity(repeat);
    let mut last_text_prefix = String::new();
    let mut last_completion_tokens = 0usize;
    let mut measured_wall_ms_total = 0.0f64;
    let mut measured_completion_tokens_total = 0usize;
    for run_idx in 0..warmup_runs.saturating_add(repeat) {
        let is_warmup = run_idx < warmup_runs;
        let hit_before = infer_metal::paged_kv_read_hits();
        let fallback_before = infer_metal::paged_kv_read_fallbacks();
        let start = Instant::now();
        let output = engine.complete(CompletionRequest {
            prompt: prompt.clone(),
            max_tokens,
            sampling: SamplingParams::default(),
            stop: None,
        })?;
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let after_request = infer_metal::allocator_memory();
        let completion_tokens = output.usage.completion_tokens;
        last_completion_tokens = completion_tokens;
        last_text_prefix = output.text.chars().take(80).collect::<String>();
        if !is_warmup {
            let measured_idx = run_idx - warmup_runs;
            let ms_per_completion_token =
                (completion_tokens > 0).then_some(wall_ms / completion_tokens as f64);
            measured_wall_ms_total += wall_ms;
            measured_completion_tokens_total =
                measured_completion_tokens_total.saturating_add(completion_tokens);
            runs.push(json!({
                "run": measured_idx,
                "wall_ms": wall_ms,
                "completion_tokens": completion_tokens,
                "ms_per_completion_token": ms_per_completion_token,
                "paged_kv_read_hits": infer_metal::paged_kv_read_hits().saturating_sub(hit_before),
                "paged_kv_read_fallbacks": infer_metal::paged_kv_read_fallbacks().saturating_sub(fallback_before),
                "allocator_after_request": {
                    "active_bytes": after_request.active_bytes,
                    "peak_bytes": after_request.peak_bytes,
                    "cache_bytes": after_request.cache_bytes,
                }
            }));
        }
    }
    infer_metal::clear_metal_cache();
    let after_clear = infer_metal::allocator_memory();
    let avg_wall_ms = measured_wall_ms_total / repeat as f64;
    let avg_ms_per_completion_token = (measured_completion_tokens_total > 0)
        .then_some(measured_wall_ms_total / measured_completion_tokens_total as f64);

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "model": model,
            "kv_cache_dtype": format!("{:?}", dtype).to_lowercase(),
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "memory_budget_gib": memory_budget_bytes.map(|bytes| bytes / (1 << 30)),
            "low_impact": low_impact,
            "warmup_runs": warmup_runs,
            "repeat": repeat,
            "completion_tokens": last_completion_tokens,
            "text_prefix": last_text_prefix,
            "summary": {
                "avg_wall_ms": avg_wall_ms,
                "avg_ms_per_completion_token": avg_ms_per_completion_token,
                "measured_completion_tokens_total": measured_completion_tokens_total,
                "paged_kv_read_hits_total": infer_metal::paged_kv_read_hits(),
                "paged_kv_read_fallbacks_total": infer_metal::paged_kv_read_fallbacks(),
            },
            "runs": runs,
            "allocator": {
                "after_load": {
                    "active_bytes": after_load.active_bytes,
                    "peak_bytes": after_load.peak_bytes,
                    "cache_bytes": after_load.cache_bytes,
                },
                "after_clear": {
                    "active_bytes": after_clear.active_bytes,
                    "peak_bytes": after_clear.peak_bytes,
                    "cache_bytes": after_clear.cache_bytes,
                }
            }
        }))?
    );
    Ok(())
}
