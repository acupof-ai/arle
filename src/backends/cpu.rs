//! Portable CPU backend builder: the placeholder `MetalExecutor` over the real
//! backend-neutral host paged KV pool (no MLX, no CUDA). Smoke / CI.

use anyhow::Result;
use infer_api::EngineLoadConfig;
use infer_metal::MetalExecutor;
use infer_seam::HostPagedKvPool;
use infer_server::{OpenAiTokenizer, ServeHandle, ServeShutdown};

pub(crate) fn build(
    model_path: &str,
    config: &EngineLoadConfig,
    shutdown: ServeShutdown,
) -> Result<(ServeHandle, OpenAiTokenizer, String)> {
    // CPU smoke: placeholder executor over a real host KV pool; still
    // needs a tokenizer dir for encode/decode.
    let tokenizer = OpenAiTokenizer::from_model_dir(model_path)?;
    let model_id = infer_api::model_id_from_path(model_path);
    let executor = MetalExecutor::new();
    let kv = HostPagedKvPool::new(
        config.hot_workspace_slots(),
        config.total_pages,
        config.page_size,
    );
    let serve = ServeHandle::spawn_with_shutdown(
        Box::new(executor),
        Box::new(kv),
        config.scheduler_config(),
        shutdown,
    );
    Ok((serve, tokenizer, model_id))
}
