//! Metal backend builder (Apple Silicon, MLX). The service layer stays
//! backend-neutral: Metal-specific model resolution, executor construction, and
//! KV-pool sizing happen here. DeepSeek-OCR VLM checkpoints dispatch to the
//! diffusion executor inside the same builder.

use std::path::Path;

use anyhow::Result;
use infer_api::EngineLoadConfig;
use infer_seam::{BufferedDiffusionExecutor, HostPagedKvPool};
use infer_server::{OpenAiTokenizer, ServeHandle, ServeShutdown};
use infer_metal::{
    MetalDeepseekOcrModel, MetalExecutor, MetalKvCacheDtype, MetalKvPool,
    MetalResourceRequest, MetalWeightOnlyResourceRequest, plan_resource_budget,
    plan_weight_only_resource_budget,
};

pub(crate) fn build(
    model_path: &str,
    config: &EngineLoadConfig,
    shutdown: ServeShutdown,
) -> Result<(ServeHandle, OpenAiTokenizer, String)> {
    if config.mtp_enabled() {
        anyhow::bail!("MTP speculative decode is only supported by the CUDA backend");
    }
    // Flags land in the statics before executor construction (spec-decode
    // resolver + pipeline/warmup/paged-read/sampling gates).
    infer_metal::apply_runtime_flags(&config.metal);
    let metal_kv_dtype = MetalKvCacheDtype::resolve(config.kv_cache_dtype)?;
    let resolved = infer_metal::resolve_model_path(model_path)?;
    if infer_metal::model_dir_is_deepseek_ocr(&resolved) {
        return deepseek_ocr_serve_handle(model_path, &resolved, config, shutdown);
    }

    // Tokenizer loading (~190ms) is independent of resource planning and
    // engine startup — run it in parallel.
    let tokenizer_path = resolved.clone();
    let tokenizer_handle =
        std::thread::spawn(move || OpenAiTokenizer::from_model_dir(&tokenizer_path));
    let model_id = infer_api::model_id_from_path(model_path);

    let model_source = resolved.to_string_lossy().to_string();
    let mut scheduler = config.scheduler_config();
    let num_slots = config.hot_workspace_slots();
    let page_size = config.page_size;
    let low_impact = config.low_impact;
    let resource_plan = plan_resource_budget(
        &resolved,
        MetalResourceRequest {
            kv_cache_dtype: metal_kv_dtype,
            num_slots,
            total_pages: config.total_pages,
            page_size,
            low_impact,
            memory_budget_bytes: config.memory_budget_bytes,
            system_reserve_bytes: config.system_reserve_bytes,
            allow_swap: config.allow_swap,
            mem_fraction_static: config.mem_fraction_static,
        },
    )?;
    let total_pages = resource_plan.planned_total_pages;
    let planned_capacity_tokens = resource_plan.capacity_tokens;
    if planned_capacity_tokens < scheduler.max_total_tokens {
        log::warn!(
            "Metal resource guard clamps max_total_tokens {} -> {}",
            scheduler.max_total_tokens,
            planned_capacity_tokens
        );
        scheduler.max_total_tokens = planned_capacity_tokens.max(1);
    }
    if scheduler.max_prompt_tokens > scheduler.max_total_tokens {
        log::warn!(
            "Metal resource guard clamps max_prompt_tokens {} -> {}",
            scheduler.max_prompt_tokens,
            scheduler.max_total_tokens
        );
        scheduler.max_prompt_tokens = scheduler.max_total_tokens;
    }
    // Opt-in L3 NVMe spill (`--kv-disk`): attached inside the builder —
    // at construction, like every other tier knob — never post-spawn.
    // Metal serves single-process, so the deployment-total cap is the
    // per-rank cap (world = 1).
    let kv_ssd = config.kv_ssd_spill(1, infer_metal::default_t2_budget_bytes)?;
    let serve = ServeHandle::spawn_with_engine_builder_and_shutdown(
        move || {
            let mut executor =
                MetalExecutor::from_model_path_with_kv_cache_dtype_and_resource_plan(
                    &model_source,
                    metal_kv_dtype,
                    resource_plan,
                )?;
            if let Some((root, budget)) = kv_ssd {
                // Namespace tag: a different model or KV dtype must not
                // serve this store's pages.
                let epoch = format!(
                    "{:016x}",
                    infer_seam::prefix_block_content_key(
                        0,
                        &format!("{model_source}|{metal_kv_dtype:?}")
                            .bytes()
                            .map(u32::from)
                            .collect::<Vec<_>>(),
                    )
                );
                anyhow::ensure!(
                    executor.set_kv_tier_disk(root, budget, page_size, &epoch),
                    "--kv-disk: the loaded Metal model has no usable \
                     page-addressable KV tier store (a budget below one \
                     page also lands here; raise --kv-disk-limit)"
                );
            }
            let kv = MetalKvPool::new(num_slots, total_pages, page_size);
            if low_impact {
                let governor = infer_seam::CooperativeGovernor::new(infer_seam::StepBudget {
                    max_tokens: scheduler.chunked_prefill_size.max(1),
                    max_micros: 20_000,
                })
                .with_yield_every_ticks(8);
                infer_core::Engine::with_config_and_governor(
                    Box::new(executor),
                    Box::new(kv),
                    scheduler,
                    Box::new(governor),
                )
            } else {
                infer_core::Engine::with_config(Box::new(executor), Box::new(kv), scheduler)
            }
        },
        shutdown,
    )?;
    let tokenizer = tokenizer_handle
        .join()
        .expect("tokenizer thread panicked")?;
    Ok((serve, tokenizer, model_id))
}

/// Metal DeepSeek-OCR VLM builder. The DeepEncoder + DeepSeek-MoE MLX bridge
/// owns generation and is adapted to the shared autoregressive engine by a
/// buffered executor (single image, 1024x1024 base view).
fn deepseek_ocr_serve_handle(
    model_path: &str,
    resolved: &Path,
    config: &EngineLoadConfig,
    shutdown: ServeShutdown,
) -> Result<(ServeHandle, OpenAiTokenizer, String)> {
    anyhow::ensure!(
        !config.kv_ssd_requested(),
        "--kv-disk: DeepSeek-OCR Metal owns no page-addressable KV tier store"
    );

    let mut tokenizer = OpenAiTokenizer::from_model_dir(resolved)?;
    // DeepSeek-OCR's tokenizer.json ships a byte-level BPE vocab but a
    // mismatched decoder, leaking `Ġ`/`Ċ` glyphs into the OCR text. Force a
    // byte-level decoder so the output is real UTF-8.
    tokenizer.force_byte_level_decoder();
    let model_id = infer_api::model_id_from_path(model_path);
    let model_source = resolved.to_string_lossy().to_string();
    let mut scheduler = config.scheduler_config();
    scheduler.num_slots = 1;
    scheduler.max_prompt_tokens = scheduler.max_prompt_tokens.min(scheduler.max_total_tokens);
    let page_size = config.page_size.max(1);
    let total_pages = config.total_pages.max(1);
    let low_impact = config.low_impact;
    let resource_plan = plan_weight_only_resource_budget(
        resolved,
        MetalWeightOnlyResourceRequest {
            low_impact,
            memory_budget_bytes: config.memory_budget_bytes,
            system_reserve_bytes: config.system_reserve_bytes,
            allow_swap: config.allow_swap,
        },
    )?;
    let cancel = shutdown.cancel_flag();
    let max_denoising_steps = config.diffusion_max_denoising_steps.filter(|&s| s > 0);

    let serve = ServeHandle::spawn_with_engine_builder_and_shutdown(
        move || {
            let loaded = MetalDeepseekOcrModel::load_with_resource_plan(
                Path::new(&model_source),
                Some(resource_plan),
            )?;
            log::info!(
                "DeepSeek-OCR VLM loaded: image_token_id={}; Metal DeepEncoder soft-token bridge enabled",
                loaded.image_token_id
            );
            let mut generation = loaded.generation;
            if let Some(steps) = max_denoising_steps {
                generation.max_denoising_steps = steps;
            }
            let executor =
                BufferedDiffusionExecutor::new_with_cancel(loaded.model, generation, cancel);
            let kv = HostPagedKvPool::new(1, total_pages, page_size);
            if low_impact {
                let governor = infer_seam::CooperativeGovernor::new(infer_seam::StepBudget {
                    max_tokens: scheduler.chunked_prefill_size.max(1),
                    max_micros: 20_000,
                })
                .with_yield_every_ticks(8);
                infer_core::Engine::with_config_and_governor(
                    Box::new(executor),
                    Box::new(kv),
                    scheduler,
                    Box::new(governor),
                )
            } else {
                infer_core::Engine::with_config(Box::new(executor), Box::new(kv), scheduler)
            }
        },
        shutdown,
    )?;
    Ok((serve, tokenizer, model_id))
}
