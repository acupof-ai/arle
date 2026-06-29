//! `arle ocr <image>` — one-shot image OCR with DeepSeek-OCR.
//!
//! Resolves the DeepSeek-OCR model (local-first, auto-downloading the default
//! `sahilchachra/unlimited-ocr-mxfp8-mlx` on first use), loads it through the
//! shared `LoadedInferenceEngine`, and runs the multimodal chat path on a single
//! image. Metal/Apple Silicon only — DeepSeek-OCR has no CUDA backend.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, ensure};
use infer_api::{
    ChatPromptImage, ChatPromptMessage, InferenceEngine, LoadedInferenceEngine,
    MultimodalChatRequest, SamplingParams,
};

use crate::args::OcrArgs;
use crate::model_catalog::DEEPSEEK_OCR_MODEL_ID;

const DEFAULT_OCR_MAX_TOKENS: usize = 32_768;
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

    let page_selection = parse_page_selection(args.pages.as_deref())?;
    let inputs = load_ocr_inputs(&args.image, page_selection.as_deref())
        .with_context(|| format!("failed to load image `{}`", args.image))?;
    let max_tokens = resolve_ocr_max_tokens(&model_source, args.max_tokens);
    let page_count = inputs.len();
    if page_count > 1 {
        eprintln!("[ocr] OCRing {} pages...", page_count);
    }

    let mut pages = Vec::with_capacity(page_count);
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    let started_at = Instant::now();
    for (index, image) in inputs.into_iter().enumerate() {
        if page_count > 1 {
            let done = index;
            if done == 0 {
                eprintln!("[ocr] page {}/{}", index + 1, page_count);
            } else {
                let elapsed = started_at.elapsed().as_secs_f64();
                let avg = elapsed / done as f64;
                let remain = avg * (page_count - done) as f64;
                eprintln!(
                    "[ocr] page {}/{} · elapsed {:.0}s · eta {:.0}s",
                    index + 1,
                    page_count,
                    elapsed,
                    remain
                );
            }
        }
        let request = MultimodalChatRequest {
            messages: vec![ChatPromptMessage::user_with_images(
                prompt.clone(),
                vec![image],
            )],
            max_tokens,
            sampling: SamplingParams::default(),
        };
        let output = engine
            .complete_multimodal_chat(request)
            .with_context(|| format!("DeepSeek-OCR inference failed on page {}", index + 1))?;
        prompt_tokens += output.usage.prompt_tokens;
        completion_tokens += output.usage.completion_tokens;
        pages.push(output.text);
    }

    let text = format_pages(&pages);
    if args.json {
        let doc = serde_json::json!({
            "text": text,
            "pages": pages,
            "model": engine.model_id(),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        });
        write_ocr_output(args.output.as_deref(), &serde_json::to_string(&doc)?)?;
    } else {
        write_ocr_output(args.output.as_deref(), &text)?;
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

fn resolve_ocr_max_tokens(model_source: &str, requested: usize) -> usize {
    if requested > 0 {
        return requested;
    }
    read_ocr_max_context(model_source).unwrap_or(DEFAULT_OCR_MAX_TOKENS)
}

fn read_ocr_max_context(model_source: &str) -> Option<usize> {
    let path = Path::new(model_source);
    let raw = std::fs::read_to_string(path.join("config.json")).ok()?;
    let cfg = serde_json::from_str::<serde_json::Value>(&raw).ok()?;
    cfg.get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .filter(|&n| n > 0)
}

fn load_ocr_inputs(source: &str, page_selection: Option<&[usize]>) -> Result<Vec<ChatPromptImage>> {
    if !is_pdf_source(source) {
        return Ok(vec![crate::repl::load_cli_image(source)?]);
    }
    let pdf_path = materialize_pdf_source(source)?;
    let page_pngs = render_pdf_pages(&pdf_path, page_selection)?;
    page_pngs
        .into_iter()
        .map(|page_png| crate::repl::load_cli_image(&page_png.to_string_lossy()))
        .collect()
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
    let meta =
        std::fs::metadata(path).with_context(|| format!("stat pdf {} failed", path.display()))?;
    ensure!(
        meta.is_file(),
        "pdf {} is not a regular file",
        path.display()
    );
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

fn render_pdf_pages(pdf_path: &Path, page_selection: Option<&[usize]>) -> Result<Vec<PathBuf>> {
    let renderer = std::env::var(PDF_RENDERER_ENV).unwrap_or_else(|_| "pdftoppm".to_string());
    let dir = std::env::temp_dir().join(format!("arle-ocr-page-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create temp dir {} failed", dir.display()))?;
    let out_prefix = dir.join("page");
    if let Some(page_selection) = page_selection {
        eprintln!(
            "[ocr] Rendering {} selected PDF pages...",
            page_selection.len()
        );
        let pngs = page_selection
            .iter()
            .map(|page| {
                let prefix = dir.join(format!("page-{page}"));
                run_pdftoppm(&renderer, pdf_path, &prefix, Some(*page))
                    .with_context(|| format!("render pdf page {page} failed"))?;
                let png = prefix.with_extension("png");
                ensure!(
                    png.exists(),
                    "pdf renderer `{renderer}` produced no image for page {page}"
                );
                Ok(png)
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(pngs);
    }
    eprintln!("[ocr] Rendering all PDF pages...");
    run_pdftoppm(&renderer, pdf_path, &out_prefix, None)?;
    let mut pngs = std::fs::read_dir(&dir)
        .with_context(|| format!("read rendered pages in {} failed", dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
        })
        .collect::<Vec<_>>();
    pngs.sort_by_key(|path| pdf_page_sort_key(path.as_path()));
    ensure!(
        !pngs.is_empty(),
        "pdf renderer `{renderer}` produced no images under {}",
        dir.display()
    );
    eprintln!("[ocr] Rendered {} PDF pages.", pngs.len());
    Ok(pngs)
}

fn run_pdftoppm(
    renderer: &str,
    pdf_path: &Path,
    out_prefix: &Path,
    page: Option<usize>,
) -> Result<()> {
    let mut cmd = Command::new(renderer);
    cmd.arg("-png");
    if let Some(page) = page {
        cmd.arg("-f")
            .arg(page.to_string())
            .arg("-l")
            .arg(page.to_string())
            .arg("-singlefile");
    }
    let output = cmd
        .arg(pdf_path)
        .arg(out_prefix)
        .output()
        .with_context(|| {
            format!(
                "spawn pdf renderer `{renderer}` failed; install poppler (`pdftoppm`) or set {PDF_RENDERER_ENV}"
            )
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "pdf render failed via `{renderer}`: {}",
            stderr.lines().last().unwrap_or("unknown error")
        ));
    }
    Ok(())
}

fn pdf_page_sort_key(path: &Path) -> usize {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| stem.rsplit('-').next())
        .and_then(|n| n.parse::<usize>().ok())
        .unwrap_or(usize::MAX)
}

fn format_pages(pages: &[String]) -> String {
    if pages.len() <= 1 {
        return pages.first().cloned().unwrap_or_default();
    }
    pages
        .iter()
        .enumerate()
        .map(|(i, page)| format!("## Page {}\n{}", i + 1, page.trim()))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn parse_page_selection(input: Option<&str>) -> Result<Option<Vec<usize>>> {
    let Some(input) = input.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    let mut pages = Vec::new();
    for part in input.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if let Some((start, end)) = part.split_once('-') {
            let start = parse_page_num(start)?;
            let end = parse_page_num(end)?;
            ensure!(start <= end, "invalid page range `{part}`: start > end");
            pages.extend(start..=end);
        } else {
            pages.push(parse_page_num(part)?);
        }
    }
    pages.sort_unstable();
    pages.dedup();
    ensure!(!pages.is_empty(), "page selection is empty");
    Ok(Some(pages))
}

fn parse_page_num(input: &str) -> Result<usize> {
    let page = input
        .trim()
        .parse::<usize>()
        .with_context(|| format!("invalid page number `{input}`"))?;
    ensure!(page > 0, "page numbers are 1-based");
    Ok(page)
}

fn write_ocr_output(path: Option<&Path>, text: &str) -> Result<()> {
    if let Some(path) = path {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create output dir {} failed", parent.display()))?;
        }
        std::fs::write(path, text)
            .with_context(|| format!("write output {} failed", path.display()))?;
    } else {
        println!("{text}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_OCR_MAX_TOKENS, PDF_RENDERER_ENV, format_pages, is_pdf_source,
        parse_page_selection, pdf_page_sort_key, read_ocr_max_context, render_pdf_pages,
        write_ocr_output,
    };
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

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
        unsafe {
            std::env::set_var(PDF_RENDERER_ENV, "definitely-not-installed-pdftoppm");
        }
        let err = render_pdf_pages(&pdf, None).expect_err("missing renderer should fail");
        let msg = err.to_string();
        assert!(msg.contains("pdftoppm") || msg.contains(PDF_RENDERER_ENV));
        unsafe {
            std::env::remove_var(PDF_RENDERER_ENV);
        }
    }

    #[test]
    fn pdf_render_uses_override_renderer_for_all_pages() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pdf = dir.path().join("doc.pdf");
        std::fs::write(&pdf, b"%PDF-1.4").expect("write pdf");
        let script = dir.path().join("fake-pdftoppm.sh");
        let mut file = std::fs::File::create(&script).expect("create script");
        writeln!(
            file,
            "#!/bin/sh\nfor last in \"$@\"; do :; done\nprefix=\"$last\"\nprintf 'fakepng1' > \"${{prefix}}-1.png\"\nprintf 'fakepng2' > \"${{prefix}}-2.png\"\n"
        )
        .expect("write script");
        let mut perms = std::fs::metadata(&script).expect("meta").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script, perms).expect("chmod");
        unsafe {
            std::env::set_var(PDF_RENDERER_ENV, &script);
        }
        let out = render_pdf_pages(&pdf, None).expect("fake renderer works");
        assert_eq!(out.len(), 2);
        assert!(out[0].ends_with("page-1.png"));
        assert!(out[1].ends_with("page-2.png"));
        assert!(out[0].exists());
        assert!(out[1].exists());
        unsafe {
            std::env::remove_var(PDF_RENDERER_ENV);
        }
    }

    #[test]
    fn pdf_render_selected_pages_use_singlefile() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pdf = dir.path().join("doc.pdf");
        std::fs::write(&pdf, b"%PDF-1.4").expect("write pdf");
        let script = dir.path().join("fake-pdftoppm.sh");
        let mut file = std::fs::File::create(&script).expect("create script");
        writeln!(
            file,
            "#!/bin/sh\nfor last in \"$@\"; do :; done\nprintf 'fakepng' > \"${{last}}.png\"\n"
        )
        .expect("write script");
        let mut perms = std::fs::metadata(&script).expect("meta").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script, perms).expect("chmod");
        unsafe {
            std::env::set_var(PDF_RENDERER_ENV, &script);
        }
        let out = render_pdf_pages(&pdf, Some(&[3, 1])).expect("selected pages render");
        assert_eq!(out.len(), 2);
        assert!(out[0].ends_with("page-3.png"));
        assert!(out[1].ends_with("page-1.png"));
        unsafe {
            std::env::remove_var(PDF_RENDERER_ENV);
        }
    }

    #[test]
    fn page_sort_key_orders_numerically() {
        let mut paths = [
            PathBuf::from("page-10.png"),
            PathBuf::from("page-2.png"),
            PathBuf::from("page-1.png"),
        ];
        paths.sort_by_key(|p| pdf_page_sort_key(p));
        assert_eq!(paths[0], PathBuf::from("page-1.png"));
        assert_eq!(paths[1], PathBuf::from("page-2.png"));
        assert_eq!(paths[2], PathBuf::from("page-10.png"));
    }

    #[test]
    fn format_pages_single_and_multi() {
        assert_eq!(format_pages(&[String::from("one")]), "one");
        assert_eq!(
            format_pages(&[String::from("one"), String::from("two")]),
            "## Page 1\none\n\n## Page 2\ntwo"
        );
    }

    #[test]
    fn reads_ocr_max_context_from_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.json"),
            r#"{ "max_position_embeddings": 32768, "model_type": "deepseekocr" }"#,
        )
        .expect("write config");
        assert_eq!(
            read_ocr_max_context(dir.path().to_str().expect("utf8 path")),
            Some(32_768)
        );
        assert_eq!(read_ocr_max_context("/definitely/missing"), None);
        assert_eq!(DEFAULT_OCR_MAX_TOKENS, 32_768);
    }

    #[test]
    fn parses_page_selection() {
        assert_eq!(parse_page_selection(None).expect("none"), None);
        assert_eq!(
            parse_page_selection(Some("1,3,5-7")).expect("pages"),
            Some(vec![1, 3, 5, 6, 7])
        );
        assert_eq!(
            parse_page_selection(Some("7,3,3,1")).expect("dedup"),
            Some(vec![1, 3, 7])
        );
    }

    #[test]
    fn rejects_invalid_page_selection() {
        assert!(parse_page_selection(Some("0")).is_err());
        assert!(parse_page_selection(Some("3-1")).is_err());
        assert!(parse_page_selection(Some("a")).is_err());
    }

    #[test]
    fn writes_output_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("nested").join("ocr.txt");
        write_ocr_output(Some(&out), "hello").expect("write output");
        assert_eq!(std::fs::read_to_string(out).expect("read output"), "hello");
    }
}
