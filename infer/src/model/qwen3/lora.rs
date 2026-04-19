//! HuggingFace PEFT LoRA adapters for Qwen3 — loader + types.
//!
//! Phase 1 of M2b: forward-only CUDA LoRA adapter loaded from an on-disk
//! PEFT safetensors directory. Phase 2 wires these adapters into
//! `ops/linear.rs` and the Qwen3 attention/MLP forward path. This file
//! defines the types and the loader only — no hot-path changes.
//!
//! The PEFT on-disk layout we target:
//!
//! ```text
//! <lora_path>/adapter_config.json   (r, lora_alpha, target_modules)
//! <lora_path>/adapter_model.safetensors
//! ```
//!
//! Tensor keys follow HuggingFace PEFT's convention for a causal-LM
//! wrapped in `PeftModelForCausalLM`:
//!
//! ```text
//! base_model.model.model.layers.<i>.self_attn.<m>.lora_A.weight   [r, in_features]
//! base_model.model.model.layers.<i>.self_attn.<m>.lora_B.weight   [out_features, r]
//! base_model.model.model.layers.<i>.mlp.<m>.lora_A.weight         (gate/up/down)
//! base_model.model.model.layers.<i>.mlp.<m>.lora_B.weight
//! ```
//!
//! The loader is tolerant: it accepts both `base_model.model.model.` and
//! `base_model.model.` prefixes (different transformers versions and
//! different peft export scripts land on slightly different conventions).
//!
//! `scale = lora_alpha / r` is pre-computed at load time so the forward
//! path can emit `y += scale * B @ (A @ x)` without repeating the
//! division per token.

use anyhow::{Context, Result, anyhow};
use half::bf16;
use log::{debug, info, warn};
use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use cuda_kernels::prelude::{DeviceContext, DeviceMatrix};

/// A single LoRA adapter: `delta = scale * B @ A`.
///
/// Shapes follow the PEFT on-disk convention:
/// - `a`: `[r, in_features]`
/// - `b`: `[out_features, r]`
///
/// `scale = lora_alpha / r` is baked in at load time. Both matrices are
/// bf16 on-device (converted from f32 at load if the adapter was saved
/// in f32, which is the PEFT default).
pub struct LoRAAdapter {
    pub a: DeviceMatrix,
    pub b: DeviceMatrix,
    pub scale: f32,
}

/// Per-layer LoRA slot — one optional adapter per projection the base
/// model exposes. The fields mirror the names peft uses in the key path,
/// so `target_modules: ["q_proj", "v_proj"]` maps directly to which
/// slots get populated.
#[derive(Default)]
pub struct LayerLoRA {
    pub q_proj: Option<LoRAAdapter>,
    pub k_proj: Option<LoRAAdapter>,
    pub v_proj: Option<LoRAAdapter>,
    pub o_proj: Option<LoRAAdapter>,
    pub gate_proj: Option<LoRAAdapter>,
    pub up_proj: Option<LoRAAdapter>,
    pub down_proj: Option<LoRAAdapter>,
}

/// Full Qwen3 LoRA bundle — one `LayerLoRA` per transformer block.
pub struct Qwen3LoRA {
    pub layers: Vec<LayerLoRA>,
}

/// Load a PEFT-format LoRA adapter directory and build a `Qwen3LoRA`
/// keyed by layer index.
///
/// `num_layers` comes from the base model's `Config::num_hidden_layers`
/// — we only populate slots whose `layer_idx < num_layers`, silently
/// dropping any extra that the adapter might carry (it shouldn't).
/// Missing adapters are `None` — not an error — so a LoRA that targets
/// only `q_proj` and `v_proj` (the PEFT default) leaves the other five
/// slots empty.
///
/// Supported source dtypes: `F32` (peft default) and `BF16`. F32 is
/// converted to bf16 on upload so the forward path can use the same
/// bf16 GEMM primitives as the base weights.
pub fn load_peft_lora(
    ctx: &DeviceContext,
    lora_path: &str,
    num_layers: usize,
) -> Result<Qwen3LoRA> {
    let dir = Path::new(lora_path);
    if !dir.is_dir() {
        return Err(anyhow!(
            "LoRA path '{}' is not a directory (expected a PEFT adapter dir)",
            lora_path
        ));
    }

    let cfg_path = dir.join("adapter_config.json");
    let cfg_raw =
        fs::read_to_string(&cfg_path).with_context(|| format!("reading {}", cfg_path.display()))?;
    let cfg: serde_json::Value = serde_json::from_str(&cfg_raw)
        .with_context(|| format!("parsing {}", cfg_path.display()))?;

    let r = cfg
        .get("r")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| anyhow!("adapter_config.json missing `r`"))? as usize;
    let lora_alpha =
        cfg.get("lora_alpha")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| anyhow!("adapter_config.json missing `lora_alpha`"))? as f32;
    if r == 0 {
        return Err(anyhow!("adapter_config.json has r=0 (invalid)"));
    }
    let scale = lora_alpha / r as f32;

    let st_path = dir.join("adapter_model.safetensors");
    let file =
        fs::File::open(&st_path).with_context(|| format!("opening {}", st_path.display()))?;
    // SAFETY: Mmap is kept alive for the duration of this function; the
    // SafeTensors view borrows from it, and we consume all tensors before
    // the mmap is dropped.
    let mmap =
        unsafe { Mmap::map(&file) }.with_context(|| format!("mmapping {}", st_path.display()))?;
    let st = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("parsing safetensors header in {}", st_path.display()))?;

    // Bucket every (layer_idx, module, A_or_B) triple that matches the
    // peft naming convention. Unknown keys are ignored with a debug log
    // so the loader tolerates adapter_model.safetensors files that also
    // contain embed_tokens / lm_head deltas (we only support per-layer
    // attention + MLP slots in Phase 1).
    let mut buckets: HashMap<(usize, LoraModule), LoraAB> = HashMap::new();
    for (name, view) in st.iter() {
        if let Some((layer_idx, module, which)) = parse_peft_key(name) {
            if layer_idx >= num_layers {
                debug!(
                    "lora: skipping {} (layer_idx {} >= num_layers {})",
                    name, layer_idx, num_layers
                );
                continue;
            }
            let entry = buckets.entry((layer_idx, module)).or_default();
            match which {
                Which::A => entry.a = Some(view),
                Which::B => entry.b = Some(view),
            }
        } else {
            debug!("lora: unrecognized tensor key '{}', skipping", name);
        }
    }

    // Materialise each populated bucket into a LoRAAdapter on-device.
    let mut layers: Vec<LayerLoRA> = (0..num_layers).map(|_| LayerLoRA::default()).collect();
    let mut loaded_count = 0usize;
    for ((layer_idx, module), ab) in buckets {
        let Some(a_view) = ab.a else {
            warn!(
                "lora: layer {} {:?}: lora_A present without lora_B (or vice versa); skipping",
                layer_idx, module
            );
            continue;
        };
        let Some(b_view) = ab.b else {
            warn!(
                "lora: layer {} {:?}: lora_B missing; skipping",
                layer_idx, module
            );
            continue;
        };

        let a_shape = a_view.shape();
        let b_shape = b_view.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(anyhow!(
                "lora: layer {} {:?}: expected rank-2 A/B, got {:?} / {:?}",
                layer_idx,
                module,
                a_shape,
                b_shape
            ));
        }
        if a_shape[0] != r {
            return Err(anyhow!(
                "lora: layer {} {:?}: lora_A rows {} != config r {}",
                layer_idx,
                module,
                a_shape[0],
                r
            ));
        }
        if b_shape[1] != r {
            return Err(anyhow!(
                "lora: layer {} {:?}: lora_B cols {} != config r {}",
                layer_idx,
                module,
                b_shape[1],
                r
            ));
        }

        let a = upload_as_bf16(ctx, &a_view, a_shape[0], a_shape[1], None)
            .with_context(|| format!("lora: layer {} {:?} A", layer_idx, module))?;
        // B is pre-scaled by `scale = lora_alpha / r` at load time so the
        // hot path can emit `y += B_scaled @ (A @ x)` without a runtime
        // axpy. Adapter.scale retains the original value for diagnostics.
        let b = upload_as_bf16(ctx, &b_view, b_shape[0], b_shape[1], Some(scale))
            .with_context(|| format!("lora: layer {} {:?} B", layer_idx, module))?;
        let adapter = LoRAAdapter { a, b, scale };
        match module {
            LoraModule::QProj => layers[layer_idx].q_proj = Some(adapter),
            LoraModule::KProj => layers[layer_idx].k_proj = Some(adapter),
            LoraModule::VProj => layers[layer_idx].v_proj = Some(adapter),
            LoraModule::OProj => layers[layer_idx].o_proj = Some(adapter),
            LoraModule::GateProj => layers[layer_idx].gate_proj = Some(adapter),
            LoraModule::UpProj => layers[layer_idx].up_proj = Some(adapter),
            LoraModule::DownProj => layers[layer_idx].down_proj = Some(adapter),
        }
        loaded_count += 1;
    }

    info!(
        "lora: loaded {} adapters across {} layers from {} (r={}, alpha={}, scale={:.4})",
        loaded_count, num_layers, lora_path, r, lora_alpha, scale
    );
    Ok(Qwen3LoRA { layers })
}

/// Which projection an adapter targets. The seven variants cover the
/// Qwen3 attention + MLP surface; anything else in the safetensors file
/// is ignored at parse time.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
enum LoraModule {
    QProj,
    KProj,
    VProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
}

impl LoraModule {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "q_proj" => Some(Self::QProj),
            "k_proj" => Some(Self::KProj),
            "v_proj" => Some(Self::VProj),
            "o_proj" => Some(Self::OProj),
            "gate_proj" => Some(Self::GateProj),
            "up_proj" => Some(Self::UpProj),
            "down_proj" => Some(Self::DownProj),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum Which {
    A,
    B,
}

#[derive(Default)]
struct LoraAB<'a> {
    a: Option<TensorView<'a>>,
    b: Option<TensorView<'a>>,
}

/// Parse a PEFT key of the form
/// `<prefix>layers.<i>.(self_attn|mlp).<module>.lora_(A|B).weight` into
/// `(layer_idx, module, A_or_B)`. Returns `None` for keys that don't
/// match the expected shape (lm_head, embed_tokens, scalar metadata,
/// etc.) — those are skipped with a debug log upstream.
fn parse_peft_key(name: &str) -> Option<(usize, LoraModule, Which)> {
    // Split on '.' and scan for `layers.<i>`. Works uniformly across
    // `base_model.model.layers.<i>...`, `base_model.model.model.layers.<i>...`,
    // or any other prefix peft might emit — we only care about what
    // comes after `layers.<i>`.
    let parts: Vec<&str> = name.split('.').collect();
    let layers_pos = parts.iter().position(|p| *p == "layers")?;
    let layer_idx: usize = parts.get(layers_pos + 1)?.parse().ok()?;
    // After layers.<i> we expect: (self_attn|mlp) . <module> . lora_(A|B) . weight
    let kind = *parts.get(layers_pos + 2)?;
    if kind != "self_attn" && kind != "mlp" {
        return None;
    }
    let module = LoraModule::from_str(parts.get(layers_pos + 3)?)?;
    let which = match *parts.get(layers_pos + 4)? {
        "lora_A" => Which::A,
        "lora_B" => Which::B,
        _ => return None,
    };
    // Some exporters drop the trailing `.weight`; accept both.
    match parts.get(layers_pos + 5) {
        Some(&"weight") | None => {}
        _ => return None,
    }
    Some((layer_idx, module, which))
}

/// Upload a 2D safetensors view as a bf16 `DeviceMatrix`, converting
/// from f32 if needed. PEFT adapters default to f32 on disk; we convert
/// once at load time so the inference path only deals with bf16.
///
/// When `scale` is `Some`, each element is multiplied by it before the
/// bf16 rounding step — used on the B matrix so the forward path avoids
/// a runtime scalar multiply. For BF16-on-disk sources, scaling requires
/// a host-side f32 round-trip (bf16 → f32 → scale → bf16) which is a
/// small one-time cost at load.
fn upload_as_bf16(
    ctx: &DeviceContext,
    view: &TensorView<'_>,
    rows: usize,
    cols: usize,
    scale: Option<f32>,
) -> Result<DeviceMatrix> {
    let bytes = view.data();
    let expected_elems = rows * cols;
    match view.dtype() {
        Dtype::BF16 => {
            if bytes.len() != expected_elems * 2 {
                return Err(anyhow!(
                    "bf16 tensor: expected {} bytes, got {}",
                    expected_elems * 2,
                    bytes.len()
                ));
            }
            match scale {
                None => DeviceMatrix::from_safetensors(ctx, bytes, rows, cols),
                Some(s) => {
                    let bf16_src: &[bf16] = unsafe {
                        std::slice::from_raw_parts(bytes.as_ptr().cast::<bf16>(), expected_elems)
                    };
                    let scaled: Vec<bf16> = bf16_src
                        .iter()
                        .map(|x| bf16::from_f32(x.to_f32() * s))
                        .collect();
                    DeviceMatrix::from_host(ctx, &scaled, rows, cols)
                }
            }
        }
        Dtype::F32 => {
            if bytes.len() != expected_elems * 4 {
                return Err(anyhow!(
                    "f32 tensor: expected {} bytes, got {}",
                    expected_elems * 4,
                    bytes.len()
                ));
            }
            // NOTE: assumes little-endian host (standard for all Tier-1
            // Rust targets; same assumption as `DeviceMatrix::from_safetensors`).
            let f32_slice: &[f32] =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), expected_elems) };
            let s = scale.unwrap_or(1.0);
            let bf16_host: Vec<bf16> = f32_slice.iter().map(|x| bf16::from_f32(x * s)).collect();
            DeviceMatrix::from_host(ctx, &bf16_host, rows, cols)
        }
        other => Err(anyhow!(
            "unsupported LoRA dtype {:?} (only F32 and BF16 are supported)",
            other
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_peft_key_double_model_prefix() {
        let got = parse_peft_key("base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight");
        match got {
            Some((3, LoraModule::QProj, Which::A)) => {}
            other => panic!("expected (3, QProj, A), got {:?}", other),
        }
    }

    #[test]
    fn parse_peft_key_single_model_prefix() {
        let got = parse_peft_key("base_model.model.layers.0.mlp.down_proj.lora_B.weight");
        match got {
            Some((0, LoraModule::DownProj, Which::B)) => {}
            other => panic!("expected (0, DownProj, B), got {:?}", other),
        }
    }

    #[test]
    fn parse_peft_key_rejects_embed() {
        assert!(parse_peft_key("base_model.model.embed_tokens.weight").is_none());
        assert!(parse_peft_key("base_model.model.lm_head.weight").is_none());
    }

    #[test]
    fn parse_peft_key_rejects_unknown_module() {
        assert!(
            parse_peft_key("base_model.model.layers.0.self_attn.foo_proj.lora_A.weight").is_none()
        );
    }
}
