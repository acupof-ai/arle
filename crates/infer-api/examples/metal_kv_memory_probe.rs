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

    let mut model = "mlx-community/Qwen3.6-35B-A3B-4bit".to_string();
    let mut dtype = KvCacheDtype::Auto;
    let mut target_prompt_tokens = 4096usize;
    let mut max_tokens = 8usize;
    let mut total_pages = EngineLoadConfig::default().total_pages;
    let mut chunked_prefill_size = 64usize;

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
            "--help" | "-h" => {
                println!(
                    "usage: cargo run -p infer-api --example metal_kv_memory_probe --release --no-default-features --features metal,no-cuda -- [--model MODEL] [--kv-cache-dtype auto|bf16|int8] [--prompt-tokens N] [--max-tokens N]"
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
    }

    let mut config = EngineLoadConfig {
        num_slots: 1,
        total_pages,
        chunked_prefill_size,
        max_prompt_tokens: target_prompt_tokens.saturating_add(1024),
        max_total_tokens: target_prompt_tokens
            .saturating_add(max_tokens)
            .saturating_add(1024),
        kv_cache_dtype: dtype,
        low_impact: true,
        ..EngineLoadConfig::default()
    };
    let min_pages = config.max_total_tokens.div_ceil(config.page_size);
    if config.total_pages < min_pages {
        config.total_pages = min_pages;
    }

    let mut engine = LoadedInferenceEngine::load_with_config(&model, true, config)?;
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
    let output = engine.complete(CompletionRequest {
        prompt,
        max_tokens,
        sampling: SamplingParams::default(),
        stop: None,
        logprobs: false,
        session_id: None,
        trace_context: None,
        cancel: None,
    })?;
    let after_request = infer_metal::allocator_memory();
    infer_metal::clear_metal_cache();
    let after_clear = infer_metal::allocator_memory();

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "model": model,
            "kv_cache_dtype": format!("{:?}", dtype).to_lowercase(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output.usage.completion_tokens,
            "text_prefix": output.text.chars().take(80).collect::<String>(),
            "allocator": {
                "after_load": {
                    "active_bytes": after_load.active_bytes,
                    "peak_bytes": after_load.peak_bytes,
                    "cache_bytes": after_load.cache_bytes,
                },
                "after_request": {
                    "active_bytes": after_request.active_bytes,
                    "peak_bytes": after_request.peak_bytes,
                    "cache_bytes": after_request.cache_bytes,
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
