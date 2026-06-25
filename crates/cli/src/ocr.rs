//! `arle ocr <image>` — one-shot image OCR with DeepSeek-OCR.
//!
//! Resolves the DeepSeek-OCR model (local-first, auto-downloading the default
//! `sahilchachra/unlimited-ocr-mxfp8-mlx` on first use), loads it through the
//! shared `LoadedInferenceEngine`, and runs the multimodal chat path on a single
//! image. Metal/Apple Silicon only — DeepSeek-OCR has no CUDA backend.

use anyhow::{Context, Result};
use infer_api::{
    ChatPromptMessage, InferenceEngine, LoadedInferenceEngine, MultimodalChatRequest, SamplingParams,
};

use crate::args::OcrArgs;
use crate::model_catalog::DEEPSEEK_OCR_MODEL_ID;

/// Generous default OCR budget: full-page markdown can run long, and the BOS-fix
/// + EOS handling means the model stops on its own well before this.
const DEFAULT_OCR_MAX_TOKENS: usize = 8192;

/// Run the `ocr` subcommand: load DeepSeek-OCR, read one image, print the text.
pub(crate) fn run(args: &OcrArgs) -> Result<()> {
    let model_source = match &args.model_path {
        Some(path) => path.clone(),
        None => resolve_or_download_ocr_model()?,
    };

    let mut engine = LoadedInferenceEngine::load(&model_source, /*cuda_graph=*/ false)
        .with_context(|| format!("failed to load DeepSeek-OCR model from `{model_source}`"))?;

    let prompt = args
        .prompt
        .clone()
        .unwrap_or_else(|| args.mode.prompt().to_string());

    let image = crate::repl::load_cli_image(&args.image)
        .with_context(|| format!("failed to load image `{}`", args.image))?;

    let max_tokens = if args.max_tokens == 0 {
        DEFAULT_OCR_MAX_TOKENS
    } else {
        args.max_tokens
    };

    let request = MultimodalChatRequest {
        messages: vec![ChatPromptMessage::user_with_images(prompt, vec![image])],
        max_tokens,
        sampling: SamplingParams::default(),
    };

    let output = engine
        .complete_multimodal_chat(request)
        .context("DeepSeek-OCR inference failed")?;

    if args.json {
        let doc = serde_json::json!({
            "text": output.text,
            "model": engine.model_id(),
            "usage": {
                "prompt_tokens": output.usage.prompt_tokens,
                "completion_tokens": output.usage.completion_tokens,
            },
        });
        println!("{}", serde_json::to_string(&doc)?);
    } else {
        println!("{}", output.text);
    }
    Ok(())
}

/// Resolve the default DeepSeek-OCR model to a local path, downloading it from
/// HuggingFace on first use (with a progress bar). Mirrors the model picker's
/// download flow but never prompts — `arle ocr` is meant to "just work".
fn resolve_or_download_ocr_model() -> Result<String> {
    if let Some(path) = infer_util::hf_hub::resolve_local_weighted_model_path(DEEPSEEK_OCR_MODEL_ID)
    {
        return Ok(path.to_string_lossy().into_owned());
    }
    eprintln!(
        "[ocr] DeepSeek-OCR model not found locally — downloading {DEEPSEEK_OCR_MODEL_ID} (~3.6 GB, first run only)…"
    );
    let path = crate::download::download_model_with_progress(DEEPSEEK_OCR_MODEL_ID)
        .with_context(|| format!("failed to download `{DEEPSEEK_OCR_MODEL_ID}`"))?;
    Ok(path.to_string_lossy().into_owned())
}
