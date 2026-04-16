//! Quantization format detection and metadata parsing.
//!
//! Reads quantization configuration from model directories (config.json,
//! quantize_config.json, etc.) and returns a strongly-typed [`QuantMeta`].
//!
//! All parsing is pure CPU — no GPU / CUDA required.
//!
//! # Supported formats
//!
//! | Format | Config source | Notes |
//! |--------|--------------|-------|
//! | GPTQ   | `quantize_config.json` | AutoGPTQ / AutoAWQ GPTQ backend |
//! | AWQ    | `quant_config.json` | AutoAWQ |
//! | FP8    | `config.json` → `quantization_config` | Compressed-Tensors / Modelopt |
//! | INT8   | `config.json` → `quantization_config` | SmoothQuant style |
//! | GGUF   | file extension `.gguf` in directory | llama.cpp format |
//! | None   | (no config found) | BF16 / FP16 weights |

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

// ============================================================================
// QuantFormat enum
// ============================================================================

/// High-level quantization format identifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum QuantFormat {
    None,
    Gptq,
    Awq,
    Fp8,
    Int8,
    Gguf,
    TurboQuant,
}

impl QuantFormat {
    pub fn is_quantized(self) -> bool {
        !matches!(self, Self::None)
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::None => "none (BF16/FP16)",
            Self::Gptq => "GPTQ (INT4)",
            Self::Awq => "AWQ (INT4)",
            Self::Fp8 => "FP8 (E4M3)",
            Self::Int8 => "INT8 (W8A8)",
            Self::Gguf => "GGUF",
            Self::TurboQuant => "TurboQuant (TQ3)",
        }
    }
}

impl std::fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.display_name())
    }
}

// ============================================================================
// Per-format config structs
// ============================================================================

/// GPTQ quantization parameters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GptqConfig {
    /// Bit-width (typically 4 or 8).
    pub bits: u8,
    /// Group size for per-group quantization (-1 = per-column).
    pub group_size: i32,
    /// Activation order (desc_act) — reorders channels by decreasing Hessian importance.
    pub desc_act: bool,
    /// Symmetric quantization (no zero-point).
    pub sym: bool,
    /// Checkpoint format string (e.g. "gptq" or "marlin").
    pub checkpoint_format: Option<String>,
}

impl Default for GptqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            desc_act: false,
            sym: true,
            checkpoint_format: None,
        }
    }
}

/// AWQ quantization parameters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AwqConfig {
    /// Bit-width (typically 4).
    pub bits: u8,
    /// Group size.
    pub group_size: usize,
    /// Whether zero-point is stored (true = asymmetric).
    pub zero_point: bool,
    /// Backend/kernel variant.
    pub version: AwqVersion,
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            zero_point: true,
            version: AwqVersion::Gemm,
        }
    }
}

/// AWQ kernel variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AwqVersion {
    /// Standard GEMM path.
    Gemm,
    /// GEMV-optimised path (small batch sizes).
    Gemv,
    /// Marlin mixed-precision GEMM (fastest on A100+).
    Marlin,
}

impl AwqVersion {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "gemv" => Self::Gemv,
            "marlin" => Self::Marlin,
            _ => Self::Gemm,
        }
    }
}

/// FP8 quantization parameters (H100 / Hopper+).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fp8Config {
    /// Whether activation scales are computed dynamically per-token or are static per-tensor.
    pub activation_scheme: Fp8ActivationScheme,
    /// Whether weights use FP8 storage (E4M3). Currently always true.
    pub weight_fp8: bool,
}

impl Default for Fp8Config {
    fn default() -> Self {
        Self {
            activation_scheme: Fp8ActivationScheme::Dynamic,
            weight_fp8: true,
        }
    }
}

/// FP8 activation scaling strategy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fp8ActivationScheme {
    /// Per-token dynamic scale (more accurate, slightly slower).
    Dynamic,
    /// Per-tensor static scale from calibration data.
    Static,
}

/// INT8 (SmoothQuant / W8A8) quantization config.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Int8Config {
    /// Whether SmoothQuant migration was applied (channel-wise scale baked in).
    pub is_smoothquant: bool,
    /// Per-channel weight scaling.
    pub per_channel: bool,
}

impl Default for Int8Config {
    fn default() -> Self {
        Self {
            is_smoothquant: false,
            per_channel: true,
        }
    }
}

/// TurboQuant weight quantization config (Hadamard rotation + Lloyd-Max).
#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantWeightConfig {
    /// Bit-width (2, 3, or 4).
    pub bits: u8,
    /// Group size for per-group quantization.
    pub group_size: usize,
    /// Rotation type ("hadamard" or "full").
    pub rotation: String,
}

impl Default for TurboQuantWeightConfig {
    fn default() -> Self {
        Self {
            bits: 3,
            group_size: 128,
            rotation: "hadamard".to_string(),
        }
    }
}

/// GGUF file info.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufConfig {
    /// Path to the .gguf file.
    pub file_path: String,
    /// GGUF file type enum value (e.g. 15 = Q4_K_M, 7 = Q8_0).
    pub file_type: u32,
}

// ============================================================================
// QuantMeta union
// ============================================================================

/// Fully parsed quantization metadata.
#[derive(Clone, Debug)]
pub enum QuantMeta {
    None,
    Gptq(GptqConfig),
    Awq(AwqConfig),
    Fp8(Fp8Config),
    Int8(Int8Config),
    Gguf(GgufConfig),
    TurboQuant(TurboQuantWeightConfig),
}

impl QuantMeta {
    pub fn format(&self) -> QuantFormat {
        match self {
            Self::None => QuantFormat::None,
            Self::Gptq(_) => QuantFormat::Gptq,
            Self::Awq(_) => QuantFormat::Awq,
            Self::Fp8(_) => QuantFormat::Fp8,
            Self::Int8(_) => QuantFormat::Int8,
            Self::Gguf(_) => QuantFormat::Gguf,
            Self::TurboQuant(_) => QuantFormat::TurboQuant,
        }
    }

    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::None)
    }
}

// ============================================================================
// Serde helpers (private) — raw JSON shapes from quantize_config.json etc.
// ============================================================================

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawGptqConfig {
    bits: Option<u8>,
    group_size: Option<i32>,
    desc_act: Option<bool>,
    sym: Option<bool>,
    checkpoint_format: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawAwqConfig {
    bits: Option<u8>,
    group_size: Option<u64>,
    zero_point: Option<bool>,
    version: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawQuantizationConfig {
    quant_type: Option<String>,
    #[serde(rename = "type")]
    type_field: Option<String>,
    activation_scheme: Option<String>,
    is_smoothquant: Option<bool>,
    // AWQ / GPTQ sub-fields sometimes appear at top level
    bits: Option<u8>,
    group_size: Option<u64>,
    zero_point: Option<bool>,
    version: Option<String>,
    desc_act: Option<bool>,
    sym: Option<bool>,
    checkpoint_format: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawModelConfig {
    quantization_config: Option<RawQuantizationConfig>,
    #[serde(rename = "model_type")]
    _model_type: Option<String>,
}

// ============================================================================
// Detection logic
// ============================================================================

/// Detect the quantization format for a model directory without parsing full config.
///
/// Fast path — just looks at which files are present.
pub fn detect_quant_format(model_path: &str) -> QuantFormat {
    load_quant_meta(model_path).map_or(QuantFormat::None, |m| m.format())
}

/// Fully parse and return quantization metadata for a model directory.
pub fn load_quant_meta(model_path: &str) -> Result<QuantMeta> {
    let dir = Path::new(model_path);

    // 1. GGUF: look for a .gguf file
    if let Some(meta) = try_load_gguf(dir) {
        return Ok(meta);
    }

    // 2. TurboQuant: `turboquant_config.json`
    let tq_path = dir.join("turboquant_config.json");
    if tq_path.exists() {
        return load_turboquant_from_file(&tq_path);
    }

    // 3. GPTQ: AutoGPTQ writes `quantize_config.json`
    let gptq_path = dir.join("quantize_config.json");
    if gptq_path.exists() {
        return load_gptq_from_file(&gptq_path);
    }

    // 3. AWQ: AutoAWQ writes `quant_config.json`
    let awq_path = dir.join("quant_config.json");
    if awq_path.exists() {
        return load_awq_from_file(&awq_path);
    }

    // 4. Fall back to config.json `quantization_config` field
    let config_path = dir.join("config.json");
    if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)
            .with_context(|| format!("reading {}", config_path.display()))?;
        if let Some(meta) = try_parse_config_json(&content)? {
            return Ok(meta);
        }
    }

    Ok(QuantMeta::None)
}

/// Parse quantization metadata directly from a `config.json` string.
///
/// Useful for unit testing without disk I/O.
pub fn parse_quant_meta_from_config_json(json: &str) -> Result<QuantMeta> {
    try_parse_config_json(json).map(|opt| opt.unwrap_or(QuantMeta::None))
}

fn try_load_gguf(dir: &Path) -> Option<QuantMeta> {
    let rd = std::fs::read_dir(dir).ok()?;
    for entry in rd.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.ends_with(".gguf") {
            return Some(QuantMeta::Gguf(GgufConfig {
                file_path: entry.path().to_string_lossy().into_owned(),
                file_type: 0, // actual value requires reading GGUF header
            }));
        }
    }
    None
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawTurboQuantConfig {
    bits: Option<u8>,
    group_size: Option<u64>,
    rotation: Option<String>,
}

fn load_turboquant_from_file(path: &Path) -> Result<QuantMeta> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let raw: RawTurboQuantConfig =
        serde_json::from_str(&content).with_context(|| "parsing turboquant_config.json")?;
    Ok(QuantMeta::TurboQuant(TurboQuantWeightConfig {
        bits: raw.bits.unwrap_or(3),
        group_size: raw.group_size.unwrap_or(128) as usize,
        rotation: raw.rotation.unwrap_or_else(|| "hadamard".to_string()),
    }))
}

fn load_gptq_from_file(path: &Path) -> Result<QuantMeta> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let raw: RawGptqConfig =
        serde_json::from_str(&content).with_context(|| "parsing quantize_config.json")?;
    Ok(QuantMeta::Gptq(GptqConfig {
        bits: raw.bits.unwrap_or(4),
        group_size: raw.group_size.unwrap_or(128),
        desc_act: raw.desc_act.unwrap_or(false),
        sym: raw.sym.unwrap_or(true),
        checkpoint_format: raw.checkpoint_format,
    }))
}

fn load_awq_from_file(path: &Path) -> Result<QuantMeta> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let raw: RawAwqConfig =
        serde_json::from_str(&content).with_context(|| "parsing quant_config.json")?;
    Ok(QuantMeta::Awq(AwqConfig {
        bits: raw.bits.unwrap_or(4),
        group_size: raw.group_size.unwrap_or(128) as usize,
        zero_point: raw.zero_point.unwrap_or(true),
        version: raw
            .version
            .as_deref()
            .map_or(AwqVersion::Gemm, AwqVersion::from_str),
    }))
}

fn try_parse_config_json(json: &str) -> Result<Option<QuantMeta>> {
    let raw: RawModelConfig = serde_json::from_str(json).context("parsing config.json")?;
    let Some(qc) = raw.quantization_config else {
        return Ok(None);
    };

    // Determine quant type from field
    let qtype = qc
        .quant_type
        .as_deref()
        .or(qc.type_field.as_deref())
        .unwrap_or("")
        .to_lowercase();

    let meta = match qtype.as_str() {
        "gptq" => QuantMeta::Gptq(GptqConfig {
            bits: qc.bits.unwrap_or(4),
            group_size: qc.group_size.map_or(128, |g| g as i32),
            desc_act: qc.desc_act.unwrap_or(false),
            sym: qc.sym.unwrap_or(true),
            checkpoint_format: qc.checkpoint_format,
        }),
        "awq" => QuantMeta::Awq(AwqConfig {
            bits: qc.bits.unwrap_or(4),
            group_size: qc.group_size.unwrap_or(128) as usize,
            zero_point: qc.zero_point.unwrap_or(true),
            version: qc
                .version
                .as_deref()
                .map_or(AwqVersion::Gemm, AwqVersion::from_str),
        }),
        "fp8" | "float8" | "fp8_e4m3" => QuantMeta::Fp8(Fp8Config {
            activation_scheme: match qc.activation_scheme.as_deref().unwrap_or("dynamic") {
                "static" => Fp8ActivationScheme::Static,
                _ => Fp8ActivationScheme::Dynamic,
            },
            weight_fp8: true,
        }),
        "int8" | "smoothquant" | "w8a8" => QuantMeta::Int8(Int8Config {
            is_smoothquant: qc.is_smoothquant.unwrap_or(qtype == "smoothquant"),
            per_channel: true,
        }),
        "turboquant" | "tq" => QuantMeta::TurboQuant(TurboQuantWeightConfig {
            bits: qc.bits.unwrap_or(3),
            group_size: qc.group_size.unwrap_or(128) as usize,
            rotation: "hadamard".to_string(),
        }),
        _ => return Ok(None),
    };
    Ok(Some(meta))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn gptq_config_json() -> &'static str {
        r#"{"quantization_config":{"quant_type":"gptq","bits":4,"group_size":128,"desc_act":false,"sym":true}}"#
    }

    fn awq_config_json() -> &'static str {
        r#"{"quantization_config":{"quant_type":"awq","bits":4,"group_size":128,"zero_point":true,"version":"gemm"}}"#
    }

    fn fp8_dynamic_json() -> &'static str {
        r#"{"quantization_config":{"quant_type":"fp8","activation_scheme":"dynamic"}}"#
    }

    fn fp8_static_json() -> &'static str {
        r#"{"quantization_config":{"quant_type":"fp8","activation_scheme":"static"}}"#
    }

    fn int8_smoothquant_json() -> &'static str {
        r#"{"quantization_config":{"quant_type":"smoothquant","is_smoothquant":true}}"#
    }

    fn no_quant_json() -> &'static str {
        r#"{"architectures":["LlamaForCausalLM"],"hidden_size":4096}"#
    }

    #[test]
    fn parse_gptq_from_config_json() {
        let meta = parse_quant_meta_from_config_json(gptq_config_json()).unwrap();
        assert_eq!(meta.format(), QuantFormat::Gptq);
        if let QuantMeta::Gptq(c) = meta {
            assert_eq!(c.bits, 4);
            assert_eq!(c.group_size, 128);
            assert!(!c.desc_act);
            assert!(c.sym);
        } else {
            panic!("expected Gptq");
        }
    }

    #[test]
    fn parse_awq_from_config_json() {
        let meta = parse_quant_meta_from_config_json(awq_config_json()).unwrap();
        assert_eq!(meta.format(), QuantFormat::Awq);
        if let QuantMeta::Awq(c) = meta {
            assert_eq!(c.bits, 4);
            assert_eq!(c.group_size, 128);
            assert!(c.zero_point);
            assert_eq!(c.version, AwqVersion::Gemm);
        } else {
            panic!("expected Awq");
        }
    }

    #[test]
    fn parse_fp8_dynamic() {
        let meta = parse_quant_meta_from_config_json(fp8_dynamic_json()).unwrap();
        assert_eq!(meta.format(), QuantFormat::Fp8);
        if let QuantMeta::Fp8(c) = meta {
            assert_eq!(c.activation_scheme, Fp8ActivationScheme::Dynamic);
        } else {
            panic!("expected Fp8");
        }
    }

    #[test]
    fn parse_fp8_static() {
        let meta = parse_quant_meta_from_config_json(fp8_static_json()).unwrap();
        if let QuantMeta::Fp8(c) = meta {
            assert_eq!(c.activation_scheme, Fp8ActivationScheme::Static);
        } else {
            panic!("expected Fp8");
        }
    }

    #[test]
    fn parse_int8_smoothquant() {
        let meta = parse_quant_meta_from_config_json(int8_smoothquant_json()).unwrap();
        assert_eq!(meta.format(), QuantFormat::Int8);
        if let QuantMeta::Int8(c) = meta {
            assert!(c.is_smoothquant);
        } else {
            panic!("expected Int8");
        }
    }

    #[test]
    fn no_quant_returns_none() {
        let meta = parse_quant_meta_from_config_json(no_quant_json()).unwrap();
        assert_eq!(meta.format(), QuantFormat::None);
        assert!(!meta.is_quantized());
    }

    #[test]
    fn gptq_from_quantize_config_json() {
        // Test the load_gptq_from_file path using a temp file
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("quantize_config.json");
        std::fs::write(
            &file,
            r#"{"bits":4,"group_size":64,"desc_act":true,"sym":false}"#,
        )
        .unwrap();

        let meta = load_quant_meta(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(meta.format(), QuantFormat::Gptq);
        if let QuantMeta::Gptq(c) = meta {
            assert_eq!(c.bits, 4);
            assert_eq!(c.group_size, 64);
            assert!(c.desc_act);
            assert!(!c.sym);
        } else {
            panic!("expected Gptq");
        }
    }

    #[test]
    fn awq_from_quant_config_json() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("quant_config.json");
        std::fs::write(
            &file,
            r#"{"bits":4,"group_size":128,"zero_point":true,"version":"marlin"}"#,
        )
        .unwrap();

        let meta = load_quant_meta(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(meta.format(), QuantFormat::Awq);
        if let QuantMeta::Awq(c) = meta {
            assert_eq!(c.version, AwqVersion::Marlin);
        } else {
            panic!("expected Awq");
        }
    }

    #[test]
    fn gguf_detected_by_extension() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.Q4_K_M.gguf"), b"fake").unwrap();

        let meta = load_quant_meta(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(meta.format(), QuantFormat::Gguf);
    }

    #[test]
    fn turboquant_from_config_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("turboquant_config.json"),
            r#"{"quant_type":"turboquant","bits":3,"group_size":128,"rotation":"hadamard"}"#,
        )
        .unwrap();

        let meta = load_quant_meta(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(meta.format(), QuantFormat::TurboQuant);
        if let QuantMeta::TurboQuant(c) = meta {
            assert_eq!(c.bits, 3);
            assert_eq!(c.group_size, 128);
            assert_eq!(c.rotation, "hadamard");
        } else {
            panic!("expected TurboQuant");
        }
    }

    #[test]
    fn turboquant_from_config_json() {
        let meta = parse_quant_meta_from_config_json(
            r#"{"quantization_config":{"quant_type":"turboquant","bits":3,"group_size":64}}"#,
        )
        .unwrap();
        assert_eq!(meta.format(), QuantFormat::TurboQuant);
        if let QuantMeta::TurboQuant(c) = meta {
            assert_eq!(c.bits, 3);
            assert_eq!(c.group_size, 64);
        } else {
            panic!("expected TurboQuant");
        }
    }

    #[test]
    fn quant_format_display() {
        assert_eq!(QuantFormat::Gptq.to_string(), "GPTQ (INT4)");
        assert_eq!(QuantFormat::Fp8.to_string(), "FP8 (E4M3)");
        assert_eq!(QuantFormat::None.to_string(), "none (BF16/FP16)");
    }

    #[test]
    fn quant_format_is_quantized() {
        assert!(QuantFormat::Gptq.is_quantized());
        assert!(QuantFormat::Awq.is_quantized());
        assert!(!QuantFormat::None.is_quantized());
    }
}
