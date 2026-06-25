//! `arle ocr <image>` — one-shot image OCR with DeepSeek-OCR.
//!
//! Resolves the DeepSeek-OCR model (local-first, auto-downloading the default
//! `sahilchachra/unlimited-ocr-mxfp8-mlx` on first use), loads it through the
//! shared `LoadedInferenceEngine`, and runs the multimodal chat path on a single
//! image. Metal/Apple Silicon only — DeepSeek-OCR has no CUDA backend.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, ensure};
use infer_api::{
    ChatPromptMessage, InferenceEngine, LoadedInferenceEngine, MultimodalChatRequest, SamplingParams,
};

use crate::args::OcrArgs;
use crate::model_catalog::DEEPSEEK_OCR_MODEL_ID;

/// Generous default OCR budget: full-page markdown can run long, and the BOS-fix
/// + EOS handling means the model stops on its own well before this.
const DEFAULT_OCR_MAX_TOKENS: usize = 8192;
const PDF_RENDERER_ENV: &str = "ARLE_OCR_PDFTOPPM";

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

    let image = load_ocr_input(&args.image)
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

fn load_ocr_input(source: &str) -> Result<infer_api::ChatPromptImage> {
    if !is_pdf_source(source) {
        return crate::repl::load_cli_image(source);
    }
    let pdf_path = materialize_pdf_source(source)?;
    let page_png = render_pdf_first_page(&pdf_path)?;
    crate::repl::load_cli_image(&page_png.to_string_lossy())
}

fn is_pdf_source(source: &str) -> bool {
    source
        .trim()
        .rsplit('.')
        .next()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
}

fn materialize_pdf_source(source: &str) -> Result<PathBuf> {
    let source = source.trim();
    ensure!(!source.is_empty(), "pdf source must not be empty");
    if source.starts_with("http://") || source.starts_with("https://") {
        return download_pdf_to_temp(source);
    }
    let path = source.strip_prefix("file://").unwrap_or(source);
    let path = Path::new(path);
    let meta = std::fs::metadata(path)
        .with_context(|| format!("stat pdf {} failed", path.display()))?;
    ensure!(meta.is_file(), "pdf {} is not a regular file", path.display());
    Ok(path.to_path_buf())
}

fn download_pdf_to_temp(source: &str) -> Result<PathBuf> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("arle-cli/0.1")
        .build()
        .context("build HTTP client failed")?;
    let response = client
        .get(source)
        .send()
        .with_context(|| format!("fetch pdf {source} failed"))?
        .error_for_status()
        .with_context(|| format!("fetch pdf {source} returned an error status"))?;
    let data = response
        .bytes()
        .with_context(|| format!("read pdf {source} response failed"))?;
    let dir = std::env::temp_dir().join(format!("arle-ocr-pdf-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create temp dir {} failed", dir.display()))?;
    let path = dir.join("input.pdf");
    std::fs::write(&path, &data)
        .with_context(|| format!("write temp pdf {} failed", path.display()))?;
    Ok(path)
}

fn render_pdf_first_page(pdf_path: &Path) -> Result<PathBuf> {
    let renderer = std::env::var(PDF_RENDERER_ENV).unwrap_or_else(|_| "pdftoppm".to_string());
    let dir = std::env::temp_dir().join(format!("arle-ocr-page-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create temp dir {} failed", dir.display()))?;
    let out_prefix = dir.join("page");
    let output = Command::new(&renderer)
        .arg("-png")
        .arg("-f")
        .arg("1")
        .arg("-l")
        .arg("1")
        .arg("-singlefile")
        .arg(pdf_path)
        .arg(&out_prefix)
        .output()
        .with_context(|| {
            format!(
                "spawn pdf renderer `{renderer}` failed; install poppler (`pdftoppm`) or set {PDF_RENDERER_ENV}"
            )
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "pdf render failed via `{renderer}`: {}",
            stderr.lines().last().unwrap_or("unknown error")
        );
    }
    let png_path = out_prefix.with_extension("png");
    ensure!(
        png_path.exists(),
        "pdf renderer `{renderer}` produced no image at {}",
        png_path.display()
    );
    Ok(png_path)
}

#[cfg(test)]
mod tests {
    use super::{PDF_RENDERER_ENV, is_pdf_source, render_pdf_first_page};
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn detects_pdf_sources() {
        assert!(is_pdf_source("a.pdf"));
        assert!(is_pdf_source("A.PDF"));
        assert!(is_pdf_source("https://x/y/report.pdf"));
        assert!(!is_pdf_source("a.png"));
    }

    #[test]
    fn pdf_render_error_mentions_renderer() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pdf = dir.path().join("doc.pdf");
        std::fs::write(&pdf, b"%PDF-1.4").expect("write pdf");
        unsafe { std::env::set_var(PDF_RENDERER_ENV, "definitely-not-installed-pdftoppm"); }
        let err = render_pdf_first_page(&pdf).expect_err("missing renderer should fail");
        let msg = err.to_string();
        assert!(msg.contains("pdftoppm") || msg.contains(PDF_RENDERER_ENV));
        unsafe { std::env::remove_var(PDF_RENDERER_ENV); }
    }

    #[test]
    fn pdf_render_uses_override_renderer() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pdf = dir.path().join("doc.pdf");
        std::fs::write(&pdf, b"%PDF-1.4").expect("write pdf");
        let script = dir.path().join("fake-pdftoppm.sh");
        let mut file = std::fs::File::create(&script).expect("create script");
        writeln!(
            file,
            "#!/bin/sh\nout=\"$8.png\"\nprintf 'fakepng' > \"$out\"\n"
        )
        .expect("write script");
        let mut perms = std::fs::metadata(&script).expect("meta").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script, perms).expect("chmod");
        unsafe { std::env::set_var(PDF_RENDERER_ENV, &script); }
        let out = render_pdf_first_page(&pdf).expect("fake renderer works");
        assert!(out.ends_with("page.png"));
        assert!(out.exists());
        unsafe { std::env::remove_var(PDF_RENDERER_ENV); }
    }
}
