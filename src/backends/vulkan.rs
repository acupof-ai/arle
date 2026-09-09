//! Vulkan backend builder (cross-vendor, Qwen3 GGUF). Resolves the `.gguf`
//! checkpoint + sibling `tokenizer.json`, then spawns the `ServeHandle` over
//! [`infer_vulkan::load_qwen3_gguf`]. The numeric Vulkan forward remains
//! pending AIPC/on-box validation and fails loud inside the backend.

use std::path::PathBuf;

use anyhow::Result;
use infer_api::EngineLoadConfig;
use infer_server::{OpenAiTokenizer, ServeHandle, ServeShutdown};

pub(crate) fn build(
    model_path: &str,
    config: &EngineLoadConfig,
    shutdown: ServeShutdown,
) -> Result<(ServeHandle, OpenAiTokenizer, String)> {
    if config.mtp_enabled() {
        anyhow::bail!("MTP speculative decode is only supported by the CUDA backend");
    }
    if let Some(cap) = config.vulkan_submit_cap {
        infer_vulkan::forward::set_submit_cap(cap);
    }
    anyhow::ensure!(
        !config.kv_ssd_requested(),
        "--kv-disk: the Vulkan backend has no KV tier store"
    );
    let gguf_path = super::resolve_gguf_path(model_path, "Vulkan")?;
    let tokenizer_dir = gguf_path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .map_or_else(|| PathBuf::from("."), std::path::Path::to_path_buf);
    let tokenizer = OpenAiTokenizer::from_model_dir(&tokenizer_dir)?;
    let model_id = infer_api::model_id_from_path(model_path);

    let mut scheduler = config.scheduler_config();
    scheduler.num_slots = 1;
    let num_slots = 1;
    let max_seq_len = config.max_total_tokens;
    let serve = ServeHandle::spawn_with_engine_builder_and_shutdown(
        move || {
            let (executor, kv) = infer_vulkan::load_qwen3_gguf(&gguf_path, num_slots, max_seq_len)?;
            infer_core::Engine::with_config(Box::new(executor), Box::new(kv), scheduler)
        },
        shutdown,
    )?;
    Ok((serve, tokenizer, model_id))
}
