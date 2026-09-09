//! CUDA backend builder. Wraps `infer_api::build_cuda_engine` (the single-GPU
//! engine constructor infer-api keeps for the OPD/multiproc surface) with HF
//! path resolution + tokenizer load, matching the other backend builders.

use anyhow::Result;
use infer_api::EngineLoadConfig;
use infer_server::{OpenAiTokenizer, ServeHandle, ServeShutdown};
use infer_util::hf_hub::resolve_model_path;

pub(crate) fn build(
    model_path: &str,
    config: &EngineLoadConfig,
    shutdown: ServeShutdown,
) -> Result<(ServeHandle, OpenAiTokenizer, String)> {
    // Resolve HF id -> local cache dir, downloading if absent. Mirrors the
    // Metal path's `infer_metal::resolve_model_path` so `arle serve
    // --model-path Qwen/Qwen3.5-4B` works on CUDA without a pre-download.
    let resolved = resolve_model_path(model_path)?;
    let resolved_str = resolved.to_string_lossy().to_string();
    let tokenizer = OpenAiTokenizer::from_model_dir(&resolved)?;
    let model_id = infer_api::model_id_from_path(&resolved_str);

    let model_source = resolved_str;
    let engine_config = config.clone();
    let serve = ServeHandle::spawn_with_engine_builder_and_shutdown(
        move || infer_api::build_cuda_engine(&model_source, &engine_config),
        shutdown,
    )?;
    Ok((serve, tokenizer, model_id))
}
