//! HIP backend builder (AMD ROCm, DSv4 GGUF). Resolves the `.gguf` checkpoint
//! + sibling `tokenizer.json`, then spawns the `ServeHandle` over
//! [`infer_hip::load_dsv4_gguf`], which returns the matched executor + host KV
//! pool pair. DSv4 is a slot-arena model: per-slot depth is the per-request
//! `max_total_tokens` budget.

use std::path::PathBuf;

use anyhow::Result;
use infer_api::EngineLoadConfig;
use infer_server::{OpenAiTokenizer, ServeHandle, ServeShutdown};

pub(crate) fn build(
    model_path: &str,
    config: &EngineLoadConfig,
    shutdown: ServeShutdown,
) -> Result<(ServeHandle, OpenAiTokenizer, String)> {
    let gguf_path = super::resolve_gguf_path(model_path, "HIP")?;
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
            let (executor, kv) = infer_hip::load_dsv4_gguf(&gguf_path, num_slots, max_seq_len)?;
            infer_core::Engine::with_config(Box::new(executor), Box::new(kv), scheduler)
        },
        shutdown,
    )?;
    Ok((serve, tokenizer, model_id))
}
