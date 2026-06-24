use anyhow::{Result, anyhow};
use infer_plan::MultimodalImage;

pub const GEMMA4_IMAGE_MARKER: &str = "<|image|>";
pub const GEMMA4_BOI_MARKER: &str = "<|image>";
pub const GEMMA4_EOI_MARKER: &str = "<image|>";
const GEMMA4_PATCH_SIZE: usize = 16;
const GEMMA4_POOLING_KERNEL: usize = 3;
const GEMMA4_MAX_SOFT_TOKENS: usize = 280;

pub fn preprocess_gemma4_image(bytes: &[u8]) -> Result<MultimodalImage> {
    let image = image::load_from_memory(bytes)
        .map_err(|err| anyhow!("invalid image data: {err}"))?
        .to_rgb8();
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        anyhow::bail!("image dimensions must be non-zero");
    }
    let (target_width, target_height) = gemma4_resize_shape(width as usize, height as usize)?;
    let resized = image::imageops::resize(
        &image,
        target_width as u32,
        target_height as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let mut pixels = vec![0.0f32; 3 * target_height * target_width];
    for y in 0..target_height {
        for x in 0..target_width {
            let pixel = resized.get_pixel(x as u32, y as u32).0;
            let base = y * target_width + x;
            pixels[base] = f32::from(pixel[0]) / 255.0;
            pixels[target_height * target_width + base] = f32::from(pixel[1]) / 255.0;
            pixels[2 * target_height * target_width + base] = f32::from(pixel[2]) / 255.0;
        }
    }
    let patches = (target_height / GEMMA4_PATCH_SIZE) * (target_width / GEMMA4_PATCH_SIZE);
    let soft_token_count = patches / (GEMMA4_POOLING_KERNEL * GEMMA4_POOLING_KERNEL);
    Ok(MultimodalImage {
        pixels,
        channels: 3,
        height: target_height,
        width: target_width,
        soft_token_count,
    })
}

pub fn expand_gemma4_image_markers(prompt: &str, images: &[MultimodalImage]) -> Result<String> {
    let mut output = String::with_capacity(prompt.len() + images.len() * 4096);
    let mut rest = prompt;
    for image in images {
        let Some(pos) = rest.find(GEMMA4_IMAGE_MARKER) else {
            anyhow::bail!("chat template did not emit enough Gemma4 image markers");
        };
        output.push_str(&rest[..pos]);
        output.push_str(GEMMA4_BOI_MARKER);
        for _ in 0..image.soft_token_count {
            output.push_str(GEMMA4_IMAGE_MARKER);
        }
        output.push_str(GEMMA4_EOI_MARKER);
        rest = &rest[pos + GEMMA4_IMAGE_MARKER.len()..];
    }
    output.push_str(rest);
    if output.matches(GEMMA4_IMAGE_MARKER).count()
        != images
            .iter()
            .map(|image| image.soft_token_count)
            .sum::<usize>()
    {
        anyhow::bail!("chat template emitted more Gemma4 image markers than provided images");
    }
    Ok(output)
}

fn gemma4_resize_shape(width: usize, height: usize) -> Result<(usize, usize)> {
    let side_multiple = GEMMA4_POOLING_KERNEL * GEMMA4_PATCH_SIZE;
    let max_patches = GEMMA4_MAX_SOFT_TOKENS * GEMMA4_POOLING_KERNEL * GEMMA4_POOLING_KERNEL;
    let target_pixels = (max_patches * GEMMA4_PATCH_SIZE * GEMMA4_PATCH_SIZE) as f64;
    let factor = (target_pixels / (width * height) as f64).sqrt();
    let quantize = |side: usize| -> usize {
        (((factor * side as f64) / side_multiple as f64).floor() as usize) * side_multiple
    };
    let mut target_width = quantize(width);
    let mut target_height = quantize(height);
    if target_width == 0 && target_height == 0 {
        anyhow::bail!("image is too small for Gemma4 patch preprocessing");
    }
    let max_side_length =
        (max_patches / (GEMMA4_POOLING_KERNEL * GEMMA4_POOLING_KERNEL)) * side_multiple;
    if target_width == 0 {
        target_width = side_multiple;
        target_height = (((height as f64 / width as f64) * side_multiple as f64).floor() as usize)
            .min(max_side_length)
            .max(side_multiple);
    }
    if target_height == 0 {
        target_height = side_multiple;
        target_width = (((width as f64 / height as f64) * side_multiple as f64).floor() as usize)
            .min(max_side_length)
            .max(side_multiple);
    }
    Ok((target_width, target_height))
}

// ── DeepSeek-OCR (deepseekocr) ─────────────────────────────────────────────
//
// Base-view-only preprocessing for the 1024x1024 global view (single image, no
// dynamic crop tiling). The DeepEncoder compresses the 64x64 patch grid to
// 16x16 = 256 soft tokens; the processor lays them out row-major with one
// `image_newline` per row (16x17 = 272) plus a trailing `view_separator`, so the
// chat prompt carries exactly (16+1)*16 + 1 = 273 `<image>` placeholders.

/// The `<image>` placeholder the DeepSeek-OCR chat/template path emits.
pub const DEEPSEEK_OCR_IMAGE_MARKER: &str = "<image>";
/// DeepSeek-OCR begin-of-sentence string (encodes to token id 0 even with
/// `add_special_tokens=false`). The reference processor always prepends BOS;
/// the server's `encode` never adds special tokens, so the manually-built
/// prompt must carry it explicitly or the decoder runs without position-0 BOS.
pub const DEEPSEEK_OCR_BOS_MARKER: &str = "<｜begin▁of▁sentence｜>";
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
