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
