use anyhow::{Result, anyhow};
use infer_plan::MultimodalImage;

use crate::schema::{ChatContent, ChatMessage};

/// The `<image>` placeholder the DeepSeek-OCR chat/template path emits.
const DEEPSEEK_OCR_IMAGE_MARKER: &str = "<image>";
/// DeepSeek-OCR begin-of-sentence string (encodes to token id 0 even with
/// `add_special_tokens=false`). The reference processor always prepends BOS;
/// the server's `encode` never adds special tokens, so the manually-built
/// prompt must carry it explicitly or the decoder runs without position-0 BOS.
const DEEPSEEK_OCR_BOS_MARKER: &str = "<｜begin▁of▁sentence｜>";
const DEEPSEEK_OCR_BASE_SIZE: usize = 1024;
const DEEPSEEK_OCR_PATCH_SIZE: usize = 16;
const DEEPSEEK_OCR_DOWNSAMPLE: usize = 4;

/// Soft-token count for the 1024x1024 base view: a 16x16 grid + one newline per
/// row + one trailing separator = (q+1)*q + 1 where q = 1024/16/4 = 16.
fn deepseek_ocr_soft_token_count() -> usize {
    let q = DEEPSEEK_OCR_BASE_SIZE / DEEPSEEK_OCR_PATCH_SIZE / DEEPSEEK_OCR_DOWNSAMPLE; // 16
    (q + 1) * q + 1
}

/// Preprocess one image into the DeepSeek-OCR base view: aspect-preserving pad
/// to 1024x1024 (gray 0.5 fill, centered), normalized `(v/255 - 0.5)/0.5`,
/// channel-first `[3,1024,1024]`. `soft_token_count` is the number of `<image>`
/// placeholders the prompt must carry.
pub fn preprocess_deepseek_ocr_image(bytes: &[u8]) -> Result<MultimodalImage> {
    let image = image::load_from_memory(bytes)
        .map_err(|err| anyhow!("invalid image data: {err}"))?
        .to_rgb8();
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        anyhow::bail!("image dimensions must be non-zero");
    }
    let target = DEEPSEEK_OCR_BASE_SIZE as u32;
    // Aspect-preserving scale to fit inside target x target (ImageOps.pad).
    let scale = (target as f64 / width as f64).min(target as f64 / height as f64);
    let new_w = ((width as f64 * scale).round() as u32).clamp(1, target);
    let new_h = ((height as f64 * scale).round() as u32).clamp(1, target);
    let resized = image::imageops::resize(
        &image,
        new_w,
        new_h,
        image::imageops::FilterType::CatmullRom,
    );
    let off_x = ((target - new_w) / 2) as usize;
    let off_y = ((target - new_h) / 2) as usize;
    let side = target as usize;
    let plane = side * side;
    // Gray (0.5) fill in normalized space -> padded regions are 0.0 after
    // (0.5 - 0.5)/0.5; initialize every channel to 0.0 (the normalized mean).
    let mut pixels = vec![0.0f32; 3 * plane];
    for y in 0..new_h as usize {
        for x in 0..new_w as usize {
            let pixel = resized.get_pixel(x as u32, y as u32).0;
            let base = (off_y + y) * side + (off_x + x);
            pixels[base] = (f32::from(pixel[0]) / 255.0 - 0.5) / 0.5;
            pixels[plane + base] = (f32::from(pixel[1]) / 255.0 - 0.5) / 0.5;
            pixels[2 * plane + base] = (f32::from(pixel[2]) / 255.0 - 0.5) / 0.5;
        }
    }
    Ok(MultimodalImage {
        pixels,
        channels: 3,
        height: side,
        width: side,
        soft_token_count: deepseek_ocr_soft_token_count(),
    })
}

/// Expand each `<image>` marker the chat template emits into
/// `soft_token_count` repeated `<image>` placeholders (the DeepEncoder splices
/// vision embeddings onto exactly those token positions).
pub fn expand_deepseek_ocr_image_markers(
    prompt: &str,
    images: &[MultimodalImage],
) -> Result<String> {
    let mut output = String::with_capacity(prompt.len() + images.len() * 4096);
    let mut rest = prompt;
    for image in images {
        let Some(pos) = rest.find(DEEPSEEK_OCR_IMAGE_MARKER) else {
            anyhow::bail!("chat template did not emit enough DeepSeek-OCR image markers");
        };
        output.push_str(&rest[..pos]);
        for _ in 0..image.soft_token_count {
            output.push_str(DEEPSEEK_OCR_IMAGE_MARKER);
        }
        rest = &rest[pos + DEEPSEEK_OCR_IMAGE_MARKER.len()..];
    }
    output.push_str(rest);
    Ok(output)
}

pub(crate) fn extract_images(
    messages: &[crate::schema::ChatMessage],
    kind: Option<infer_plan::MultimodalKind>,
) -> Result<Vec<infer_plan::MultimodalImage>, crate::schema::ApiError> {
    let mut images = Vec::new();
    for message in messages {
        let Some(crate::schema::ChatContent::Parts(parts)) = &message.content else {
            continue;
        };
        for part in parts {
            match part.normalized_kind() {
                "image" => {
                    let url = image_data_url(part)?;
                    let bytes = decode_image_data_url(url)?;
                    let preprocessed = match kind {
                        Some(infer_plan::MultimodalKind::DeepseekOcr) => {
                            preprocess_deepseek_ocr_image(&bytes)
                        }
                        None => {
                            return Err(crate::schema::ApiError::bad_request(
                                "image content requires a VLM backend (DeepSeek-OCR)",
                            ));
                        }
                    };
                    images.push(preprocessed.map_err(|err| {
                        crate::schema::ApiError::bad_request(format!("invalid image data: {err}"))
                    })?);
                }
                "audio" | "video" => {
                    return Err(crate::schema::ApiError::bad_request(
                        "audio/video content parts are not supported by this VLM",
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(images)
}

pub(crate) fn expand_image_markers(
    prompt: &str,
    images: &[infer_plan::MultimodalImage],
    kind: Option<infer_plan::MultimodalKind>,
) -> Result<String, crate::schema::ApiError> {
    let expanded = match kind {
        Some(infer_plan::MultimodalKind::DeepseekOcr) => {
            expand_deepseek_ocr_image_markers(prompt, images)
        }
        None => {
            return Err(crate::schema::ApiError::bad_request(
                "image content requires a VLM backend (DeepSeek-OCR)",
            ));
        }
    };
    expanded.map_err(|err| crate::schema::ApiError::bad_request(format!("{err}")))
}

fn image_data_url(part: &crate::schema::ChatContentPart) -> Result<&str, crate::schema::ApiError> {
    fn value_url(value: &serde_json::Value) -> Option<&str> {
        value
            .as_str()
            .or_else(|| value.get("url").and_then(serde_json::Value::as_str))
    }
    part.image_url
        .as_ref()
        .and_then(value_url)
        .or_else(|| part.input_image.as_ref().and_then(value_url))
        .or_else(|| part.extra.get("image_url").and_then(value_url))
        .or_else(|| part.extra.get("url").and_then(serde_json::Value::as_str))
        .ok_or_else(|| {
            crate::schema::ApiError::bad_request("image content part must include image_url.url")
        })
}

fn decode_image_data_url(url: &str) -> Result<Vec<u8>, crate::schema::ApiError> {
    use base64::Engine;
    let Some((header, payload)) = url.split_once(',') else {
        return Err(crate::schema::ApiError::bad_request(
            "image_url must be a data:image/...;base64 URL",
        ));
    };
    let header = header.to_ascii_lowercase();
    if !header.starts_with("data:image/") || !header.ends_with(";base64") {
        return Err(crate::schema::ApiError::bad_request(
            "image_url must be a data:image/...;base64 URL",
        ));
    }
    base64::engine::general_purpose::STANDARD
        .decode(payload.as_bytes())
        .map_err(|err| {
            crate::schema::ApiError::bad_request(format!("invalid base64 image data: {err}"))
        })
}

/// Build a DeepSeek-OCR prompt from chat messages in the reference processor's
/// `<image>\n<text>` layout: BOS, then per message the image marker(s) first and
/// the text after (newline-separated). The model's chat template is too trivial
/// to render image parts, and crucially never emits BOS — so we prepend
/// [`DEEPSEEK_OCR_BOS_MARKER`] explicitly (it encodes to id 0 even with
/// `add_special_tokens=false`).
///
/// Two failure modes it guards against: (1) no BOS → the decoder runs without a
/// position-0 token and degenerates into a non-stopping loop that burns the
/// whole token budget; (2) text-before-image ordering → garbled output, since
/// the model was trained with the image soft tokens ahead of the instruction.
pub fn build_deepseek_ocr_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::from(DEEPSEEK_OCR_BOS_MARKER);
    for message in messages {
        match &message.content {
            Some(ChatContent::Text(text)) => prompt.push_str(text),
            Some(ChatContent::Parts(parts)) => {
                // Image markers first (reference `<image>\n<text>` SFT layout),
                // regardless of the order parts arrive in.
                let mut image_count = 0usize;
                for part in parts {
                    if part.normalized_kind() == "image" {
                        prompt.push_str(DEEPSEEK_OCR_IMAGE_MARKER);
                        image_count += 1;
                    }
                }
                let mut wrote_text = false;
                for part in parts {
                    if part.normalized_kind() == "text"
                        && let Some(text) = &part.text
                    {
                        if image_count > 0 && !wrote_text {
                            prompt.push('\n');
                        }
                        prompt.push_str(text);
                        wrote_text = true;
                    }
                }
            }
            None => {}
        }
    }
    prompt
}
