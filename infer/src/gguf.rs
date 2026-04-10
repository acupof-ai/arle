//! Minimal GGUF file parser — zero external dependencies.
//!
//! Reads the GGUF binary header, metadata KV pairs, and tensor directory.
//! Provides random-access tensor reading with dequantization to BF16.
//!
//! # Supported tensor types
//!
//! | Type | Bits | Dequant | Notes |
//! |------|------|---------|-------|
//! | F32  | 32   | Truncate to BF16 | Lossless for typical weight ranges |
//! | F16  | 16   | Cast to BF16 | Near-lossless |
//! | BF16 | 16   | Direct | No conversion needed |
//! | Q8_0 | 8    | scale × int8 → BF16 | Block size 32 |
//! | Q4_0 | 4    | scale × int4 → BF16 | Block size 32 |
//! | Q4_K_M | 4  | Multi-scale blocks → BF16 | Block size 256, most common GGUF format |
//!
//! # GGUF tensor name mapping
//!
//! GGUF uses llama.cpp naming (`blk.N.attn_q.weight`).
//! [`map_gguf_name`] converts to HuggingFace naming (`model.layers.N.self_attn.q_proj.weight`).

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

use anyhow::{Result, anyhow, bail};
use half::{bf16, f16};

// ── GGUF Constants ──

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as little-endian u32 (bytes: 47 47 55 46)

/// GGUF tensor element types (from ggml.h).
/// Names follow llama.cpp convention (Q4_K_M, not Q4Km).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q8_0 = 7,
    I8 = 8,
    I16 = 9,
    Q2_K = 10,
    Q3_K_S = 11,
    Q3_K_M = 12,
    Q3_K_L = 13,
    Q4_K_S = 14,
    Q4_K_M = 15,
    Q5_K_S = 16,
    Q5_K_M = 17,
    Q6_K = 18,
    BF16 = 30,
}

impl GgmlType {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q8_0),
            8 => Ok(Self::I8),
            9 => Ok(Self::I16),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K_S),
            12 => Ok(Self::Q3_K_M),
            13 => Ok(Self::Q3_K_L),
            14 => Ok(Self::Q4_K_S),
            15 => Ok(Self::Q4_K_M),
            16 => Ok(Self::Q5_K_S),
            17 => Ok(Self::Q5_K_M),
            18 => Ok(Self::Q6_K),
            30 => Ok(Self::BF16),
            _ => bail!("unsupported GGML type: {v}"),
        }
    }

    /// Bytes per block for this type. Each block contains `block_size()` elements.
    fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 => 1,
            Self::Q8_0 => 2 + 32, // f16 scale + 32 × i8 = 34
            Self::Q4_0 => 2 + 16, // f16 scale + 16 bytes = 18
            Self::Q3_K_S | Self::Q3_K_M | Self::Q3_K_L => 110, // hmask(32)+qs(64)+scales(12)+d(2)
            Self::Q4_K_S | Self::Q4_K_M => 144, // d(2)+dmin(2)+scales(12)+qs(128)
            Self::Q5_K_S | Self::Q5_K_M => 176, // d(2)+dmin(2)+scales(12)+qh(32)+qs(128)
            Self::Q6_K => 210,    // ql(128)+qh(64)+scales(16)+d(2)
            _ => 0,
        }
    }

    /// Elements per block.
    fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::I8 | Self::I16 => 1,
            Self::Q8_0 | Self::Q4_0 => 32,
            Self::Q3_K_S
            | Self::Q3_K_M
            | Self::Q3_K_L
            | Self::Q4_K_S
            | Self::Q4_K_M
            | Self::Q5_K_S
            | Self::Q5_K_M
            | Self::Q6_K => 256,
            _ => 1,
        }
    }
}

// ── GGUF Metadata Value Types ──

#[derive(Debug, Clone, Copy)]
#[repr(u32)]
#[allow(dead_code)]
enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

/// A parsed GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    Bool(bool),
    Str(String),
    Array(Vec<GgufValue>),
    Other,
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
}

// ── Tensor Info ──

/// A tensor entry from the GGUF directory (not yet loaded).
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgmlType,
    /// Byte offset from the start of the data section.
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }

    /// Total bytes in the file for this tensor.
    pub fn size_bytes(&self) -> usize {
        let n = self.numel();
        let bs = self.dtype.block_size();
        let num_blocks = (n + bs - 1) / bs;
        num_blocks * self.dtype.block_bytes()
    }
}

// ── GGUF File ──

/// A parsed GGUF file with metadata and tensor directory.
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    /// Byte offset where tensor data starts in the file.
    data_offset: u64,
    path: String,
}

impl GgufFile {
    /// Open and parse a GGUF file. Reads header + metadata + tensor directory
    /// but does NOT load tensor data into memory.
    pub fn open(path: &str) -> Result<Self> {
        let mut f =
            File::open(path).map_err(|e| anyhow!("Failed to open GGUF file '{}': {}", path, e))?;

        // Header
        let magic = read_u32(&mut f)?;
        if magic != GGUF_MAGIC {
            bail!(
                "Not a GGUF file: magic={:#x} (expected {:#x})",
                magic,
                GGUF_MAGIC
            );
        }
        let version = read_u32(&mut f)?;
        if version < 2 || version > 3 {
            bail!("Unsupported GGUF version: {version} (expected 2 or 3)");
        }
        let tensor_count = read_u64(&mut f)?;
        let metadata_kv_count = read_u64(&mut f)?;

        // Metadata KV pairs
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(&mut f)?;
            let value = read_value(&mut f)?;
            metadata.insert(key, value);
        }

        // Tensor directory
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = read_string(&mut f)?;
            let n_dims = read_u32(&mut f)?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(read_u64(&mut f)?);
            }
            let dtype = GgmlType::from_u32(read_u32(&mut f)?)?;
            let offset = read_u64(&mut f)?;
            tensors.insert(
                name.clone(),
                GgufTensorInfo {
                    name,
                    shape,
                    dtype,
                    offset,
                },
            );
        }

        // Data section starts after header, aligned to 32 bytes
        let cur_pos = f.stream_position()?;
        let data_offset = (cur_pos + 31) & !31;

        log::info!(
            "GGUF: v{}, {} tensors, {} metadata keys, data@{:#x}",
            version,
            tensors.len(),
            metadata.len(),
            data_offset
        );

        Ok(Self {
            version,
            metadata,
            tensors,
            data_offset,
            path: path.to_string(),
        })
    }

    /// Read raw tensor bytes from the file.
    pub fn read_tensor_raw(&self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found in GGUF", name))?;
        let size = info.size_bytes();
        let abs_offset = self.data_offset + info.offset;

        let mut f = File::open(&self.path)?;
        f.seek(SeekFrom::Start(abs_offset))?;
        let mut buf = vec![0u8; size];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Check if a tensor is quantized (not F32/F16/BF16).
    pub fn is_quantized(&self, name: &str) -> bool {
        self.tensors
            .get(name)
            .map(|t| !matches!(t.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16))
            .unwrap_or(false)
    }

    /// Read Q8_0 tensor in packed format: split into (qweight: Vec<i8>, scales: Vec<bf16>).
    ///
    /// Returns data suitable for `DeviceMatrix::from_quantized_int8`:
    ///   - qweight: `[numel]` i8 values (one per element)
    ///   - scales: `[rows × num_groups]` bf16 (one per 32-element block)
    pub fn read_tensor_q8_packed(&self, name: &str) -> Result<(Vec<i8>, Vec<bf16>, usize)> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found in GGUF", name))?;
        if info.dtype != GgmlType::Q8_0 {
            bail!("Expected Q8_0, got {:?}", info.dtype);
        }
        let raw = self.read_tensor_raw(name)?;
        let numel = info.numel();
        const BLOCK_SIZE: usize = 32;
        let num_blocks = numel / BLOCK_SIZE;

        let mut qweight = Vec::with_capacity(numel);
        let mut scales = Vec::with_capacity(num_blocks);

        for b in 0..num_blocks {
            let base = b * (2 + BLOCK_SIZE); // f16 scale + 32 × i8
            let scale = f16::from_le_bytes([raw[base], raw[base + 1]]).to_f32();
            scales.push(bf16::from_f32(scale));
            for i in 0..BLOCK_SIZE {
                qweight.push(raw[base + 2 + i] as i8);
            }
        }
        Ok((qweight, scales, BLOCK_SIZE))
    }

    /// Read and dequantize a tensor to BF16.
    pub fn read_tensor_bf16(&self, name: &str) -> Result<Vec<bf16>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found in GGUF", name))?;
        let raw = self.read_tensor_raw(name)?;
        let numel = info.numel();
        dequant_to_bf16(&raw, info.dtype, numel)
    }

    /// Get the architecture string from metadata (e.g., "llama", "gemma").
    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture")?.as_str()
    }

    /// Get a u32 metadata value.
    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key)?.as_u32()
    }

    /// Get a string metadata value.
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key)?.as_str()
    }

    /// Get a f32 metadata value.
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key)? {
            GgufValue::F32(v) => Some(*v),
            _ => None,
        }
    }

    /// Extract model config from GGUF metadata.
    ///
    /// Returns a generic config struct that can be used by any model.
    /// Field names follow GGUF convention: `{arch}.{field}` where arch
    /// is from `general.architecture` (e.g., "qwen3", "llama", "gemma").
    pub fn extract_model_config(&self) -> Result<GgufModelConfig> {
        let arch = self
            .architecture()
            .ok_or_else(|| anyhow!("GGUF missing general.architecture"))?
            .to_string();
        let p = |field: &str| format!("{arch}.{field}");

        Ok(GgufModelConfig {
            architecture: arch.clone(),
            vocab_size: self.meta_u32(&p("vocab_size")).unwrap_or(0) as usize,
            hidden_size: self
                .meta_u32(&p("embedding_length"))
                .ok_or_else(|| anyhow!("GGUF missing {}.embedding_length", arch))?
                as usize,
            num_hidden_layers: self
                .meta_u32(&p("block_count"))
                .ok_or_else(|| anyhow!("GGUF missing {}.block_count", arch))?
                as usize,
            num_attention_heads: self
                .meta_u32(&p("attention.head_count"))
                .ok_or_else(|| anyhow!("GGUF missing {}.attention.head_count", arch))?
                as usize,
            num_key_value_heads: self.meta_u32(&p("attention.head_count_kv")).unwrap_or(0) as usize,
            head_dim: self.meta_u32(&p("attention.key_length")).unwrap_or(128) as usize,
            intermediate_size: self.meta_u32(&p("feed_forward_length")).unwrap_or(0) as usize,
            rms_norm_eps: self
                .meta_f32(&p("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-6),
            rope_theta: self.meta_f32(&p("rope.freq_base")).unwrap_or(1_000_000.0),
            context_length: self.meta_u32(&p("context_length")).unwrap_or(4096) as usize,
        })
    }

    /// Check if this GGUF embeds a HuggingFace tokenizer JSON blob.
    pub fn has_embedded_tokenizer(&self) -> bool {
        self.metadata.contains_key("tokenizer.huggingface.json")
            || self.metadata.contains_key("tokenizer.ggml.tokens")
    }

    /// Extract embedded HuggingFace tokenizer JSON, if present.
    pub fn extract_tokenizer_json(&self) -> Option<&str> {
        self.meta_str("tokenizer.huggingface.json")
    }
}

/// Generic model config extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct GgufModelConfig {
    pub architecture: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub context_length: usize,
}

// ── Dequantization ──

/// Dequantize raw GGUF tensor bytes to BF16.
fn dequant_to_bf16(raw: &[u8], dtype: GgmlType, numel: usize) -> Result<Vec<bf16>> {
    match dtype {
        GgmlType::BF16 => {
            // Direct reinterpret
            assert_eq!(raw.len(), numel * 2);
            let slice = unsafe { std::slice::from_raw_parts(raw.as_ptr().cast::<bf16>(), numel) };
            Ok(slice.to_vec())
        }
        GgmlType::F16 => {
            assert_eq!(raw.len(), numel * 2);
            let f16s = unsafe { std::slice::from_raw_parts(raw.as_ptr().cast::<f16>(), numel) };
            Ok(f16s.iter().map(|v| bf16::from_f32(v.to_f32())).collect())
        }
        GgmlType::I8 => {
            assert_eq!(raw.len(), numel);
            Ok(raw
                .iter()
                .map(|&v| bf16::from_f32(v as i8 as f32))
                .collect())
        }
        GgmlType::I16 => {
            assert_eq!(raw.len(), numel * 2);
            let i16s = unsafe { std::slice::from_raw_parts(raw.as_ptr().cast::<i16>(), numel) };
            Ok(i16s.iter().map(|&v| bf16::from_f32(v as f32)).collect())
        }
        GgmlType::F32 => {
            assert_eq!(raw.len(), numel * 4);
            let f32s = unsafe { std::slice::from_raw_parts(raw.as_ptr().cast::<f32>(), numel) };
            Ok(f32s.iter().map(|v| bf16::from_f32(*v)).collect())
        }
        GgmlType::Q8_0 => dequant_q8_0(raw, numel),
        GgmlType::Q4_0 => dequant_q4_0(raw, numel),
        GgmlType::Q3_K_S | GgmlType::Q3_K_M | GgmlType::Q3_K_L => dequant_q3_k(raw, numel),
        GgmlType::Q4_K_M | GgmlType::Q4_K_S => dequant_q4_k(raw, numel),
        GgmlType::Q5_K_S | GgmlType::Q5_K_M => dequant_q5_k(raw, numel),
        GgmlType::Q6_K => dequant_q6_k(raw, numel),
        _ => bail!("Dequant not yet implemented for {:?}", dtype),
    }
}

/// Q8_0: 32 elements per block. Layout: f16 scale + 32× i8.
fn dequant_q8_0(raw: &[u8], numel: usize) -> Result<Vec<bf16>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + BLOCK_SIZE; // f16 + 32 × i8
    let num_blocks = numel / BLOCK_SIZE;
    assert_eq!(raw.len(), num_blocks * BLOCK_BYTES);

    let mut out = Vec::with_capacity(numel);
    for b in 0..num_blocks {
        let base = b * BLOCK_BYTES;
        let scale = f16::from_le_bytes([raw[base], raw[base + 1]]).to_f32();
        for i in 0..BLOCK_SIZE {
            let val = raw[base + 2 + i] as i8;
            out.push(bf16::from_f32(val as f32 * scale));
        }
    }
    Ok(out)
}

/// Q4_0: 32 elements per block. Layout: f16 scale + 16 bytes (2 nibbles each).
fn dequant_q4_0(raw: &[u8], numel: usize) -> Result<Vec<bf16>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 16; // f16 + 16 bytes packed nibbles
    let num_blocks = numel / BLOCK_SIZE;
    assert_eq!(raw.len(), num_blocks * BLOCK_BYTES);

    let mut out = Vec::with_capacity(numel);
    for b in 0..num_blocks {
        let base = b * BLOCK_BYTES;
        let scale = f16::from_le_bytes([raw[base], raw[base + 1]]).to_f32();
        for i in 0..16 {
            let byte = raw[base + 2 + i];
            let lo = (byte & 0x0F) as i8 - 8; // Q4_0: unsigned → signed offset by 8
            let hi = ((byte >> 4) & 0x0F) as i8 - 8;
            out.push(bf16::from_f32(lo as f32 * scale));
            out.push(bf16::from_f32(hi as f32 * scale));
        }
    }
    Ok(out)
}

/// Q4_K (Q4_K_M/Q4_K_S): 256 elements per superblock.
///
/// Layout per superblock (144 bytes):
///   - d: f16 (2 bytes) — super-block scale
///   - dmin: f16 (2 bytes) — super-block minimum
///   - scales: 12 bytes — packed 6-bit scale/min for 8 sub-blocks
///   - qs: 128 bytes — 256 × 4-bit quantized values (2 per byte)
fn dequant_q4_k(raw: &[u8], numel: usize) -> Result<Vec<bf16>> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 144; // 2+2+12+128
    let num_blocks = numel / QK_K;
    if raw.len() < num_blocks * BLOCK_BYTES {
        bail!(
            "Q4_K: expected {} bytes for {} elements, got {}",
            num_blocks * BLOCK_BYTES,
            numel,
            raw.len()
        );
    }

    let mut out = Vec::with_capacity(numel);

    for b in 0..num_blocks {
        let base = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([raw[base], raw[base + 1]]).to_f32();
        let dmin = f16::from_le_bytes([raw[base + 2], raw[base + 3]]).to_f32();
        let scales_raw = &raw[base + 4..base + 16]; // 12 bytes of packed scales
        let qs = &raw[base + 16..base + 144]; // 128 bytes of packed nibbles

        // Decode 6-bit scales: 8 sub-block scales + 8 sub-block mins from 12 bytes
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        // First 4 sub-blocks: scales in lower 6 bits of bytes 0-3, mins in bytes 4-7
        for i in 0..4 {
            sc[i] = scales_raw[i] & 63;
            mn[i] = scales_raw[i + 4] & 63;
        }
        // Last 4 sub-blocks: upper 4 bits split across bytes 8-11.
        // Lower 2 bits from scales_raw[i] >> 6, upper 4 bits from bytes 8-11.
        // llama.cpp: sc[4+i] = (scales_raw[i] >> 6) | ((scales_raw[8+i] & 0x0F) << 2)
        for i in 0..4 {
            sc[4 + i] = (scales_raw[i] >> 6) | ((scales_raw[8 + i] & 0x0F) << 2);
            mn[4 + i] = (scales_raw[i + 4] >> 6) | ((scales_raw[8 + i] >> 4) << 2);
        }

        // Dequantize 8 sub-blocks of 32 elements each
        for j in 0..8 {
            let sub_scale = d * sc[j] as f32;
            let sub_min = dmin * mn[j] as f32;
            let qs_offset = j * 16; // 32 nibbles = 16 bytes
            for i in 0..16 {
                let byte = qs[qs_offset + i];
                let lo = (byte & 0x0F) as f32;
                let hi = ((byte >> 4) & 0x0F) as f32;
                out.push(bf16::from_f32(lo * sub_scale - sub_min));
                out.push(bf16::from_f32(hi * sub_scale - sub_min));
            }
        }
    }
    Ok(out)
}

/// Q3_K: 256 elements per superblock (110 bytes).
///
/// Layout: hmask(32B) + qs(64B, 2-bit packed) + scales(12B, 6-bit packed) + d(f16, 2B)
fn dequant_q3_k(raw: &[u8], numel: usize) -> Result<Vec<bf16>> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 110;
    let num_blocks = numel / QK_K;
    if raw.len() < num_blocks * BLOCK_BYTES {
        bail!(
            "Q3_K: expected {} bytes, got {}",
            num_blocks * BLOCK_BYTES,
            raw.len()
        );
    }

    let mut out = Vec::with_capacity(numel);
    for b in 0..num_blocks {
        let base = b * BLOCK_BYTES;
        let hmask = &raw[base..base + 32];
        let qs = &raw[base + 32..base + 96];
        let scales_raw = &raw[base + 96..base + 108];
        let d = f16::from_le_bytes([raw[base + 108], raw[base + 109]]).to_f32();

        // Decode 6-bit scales for 16 sub-blocks (each 16 elements)
        let mut scales = [0i8; 16];
        for i in 0..8 {
            scales[i] = (scales_raw[i] & 0x0F) as i8 - 8;
            scales[i + 8] = ((scales_raw[i] >> 4) & 0x0F) as i8 - 8;
        }
        // Upper bits from last 4 bytes
        for i in 0..4 {
            scales[i] |= ((scales_raw[8 + i] & 0x03) as i8) << 4;
            scales[i + 4] |= (((scales_raw[8 + i] >> 2) & 0x03) as i8) << 4;
            scales[i + 8] |= (((scales_raw[8 + i] >> 4) & 0x03) as i8) << 4;
            scales[i + 12] |= (((scales_raw[8 + i] >> 6) & 0x03) as i8) << 4;
        }

        for j in 0..QK_K {
            // 2-bit value from qs (4 per byte)
            let q2 = (qs[j / 4] >> ((j % 4) * 2)) & 0x03;
            // High bit from hmask
            let hbit = ((hmask[j / 8] >> (j % 8)) & 1) as u8;
            // 3-bit value
            let q3 = q2 | (hbit << 2);
            let sc = scales[j / 16] as f32;
            out.push(bf16::from_f32(d * sc * (q3 as f32 - 4.0)));
        }
    }
    Ok(out)
}

/// Q5_K: 256 elements per superblock (176 bytes).
///
/// Layout: d(f16) + dmin(f16) + scales(12B) + qh(32B, high bits) + qs(128B, low 4 bits)
fn dequant_q5_k(raw: &[u8], numel: usize) -> Result<Vec<bf16>> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 176;
    let num_blocks = numel / QK_K;
    if raw.len() < num_blocks * BLOCK_BYTES {
        bail!(
            "Q5_K: expected {} bytes, got {}",
            num_blocks * BLOCK_BYTES,
            raw.len()
        );
    }

    let mut out = Vec::with_capacity(numel);
    for b in 0..num_blocks {
        let base = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([raw[base], raw[base + 1]]).to_f32();
        let dmin = f16::from_le_bytes([raw[base + 2], raw[base + 3]]).to_f32();
        let scales_raw = &raw[base + 4..base + 16];
        let qh = &raw[base + 16..base + 48];
        let qs = &raw[base + 48..base + 176];

        // Decode scales (same format as Q4_K)
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for i in 0..4 {
            sc[i] = scales_raw[i] & 63;
            mn[i] = scales_raw[i + 4] & 63;
        }
        for i in 0..4 {
            sc[4 + i] = (scales_raw[i] >> 6) | ((scales_raw[8 + i] & 0x0F) << 2);
            mn[4 + i] = (scales_raw[i + 4] >> 6) | ((scales_raw[8 + i] >> 4) << 2);
        }

        for j in 0..8 {
            let sub_scale = d * sc[j] as f32;
            let sub_min = dmin * mn[j] as f32;
            for i in 0..32 {
                let idx = j * 32 + i;
                // Low 4 bits from qs
                let qs_byte = qs[idx / 2];
                let lo = if idx % 2 == 0 {
                    qs_byte & 0x0F
                } else {
                    qs_byte >> 4
                };
                // High bit from qh
                let hbit = ((qh[idx / 8] >> (idx % 8)) & 1) as u8;
                let q5 = lo | (hbit << 4);
                out.push(bf16::from_f32(q5 as f32 * sub_scale - sub_min));
            }
        }
    }
    Ok(out)
}

/// Q6_K: 256 elements per superblock.
///
/// Layout per superblock (210 bytes):
///   - ql: 128 bytes (lower 4 bits of 6-bit values, packed as nibbles)
///   - qh: 64 bytes (upper 2 bits of 6-bit values, packed 4 per byte)
///   - scales: 16 bytes (i8 per 16-element sub-block)
///   - d: f16 (2 bytes) — super-block scale
fn dequant_q6_k(raw: &[u8], numel: usize) -> Result<Vec<bf16>> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 210; // 128+64+16+2
    let num_blocks = numel / QK_K;
    if raw.len() < num_blocks * BLOCK_BYTES {
        bail!(
            "Q6_K: expected {} bytes, got {}",
            num_blocks * BLOCK_BYTES,
            raw.len()
        );
    }

    let mut out = Vec::with_capacity(numel);

    for b in 0..num_blocks {
        let base = b * BLOCK_BYTES;
        let ql = &raw[base..base + 128];
        let qh = &raw[base + 128..base + 192];
        let scales = &raw[base + 192..base + 208];
        let d = f16::from_le_bytes([raw[base + 208], raw[base + 209]]).to_f32();

        for j in 0..QK_K {
            // Lower 4 bits from ql
            let ql_byte = ql[j / 2];
            let ql_val = if j % 2 == 0 {
                ql_byte & 0x0F
            } else {
                ql_byte >> 4
            };
            // Upper 2 bits from qh
            let qh_byte = qh[j / 4];
            let qh_shift = (j % 4) * 2;
            let qh_val = (qh_byte >> qh_shift) & 0x03;
            // 6-bit value
            let q = ((qh_val << 4) | ql_val) as i8 - 32; // offset by 32 for signed
            // Scale from sub-block (16 elements per scale)
            let sc = scales[j / 16] as i8;
            out.push(bf16::from_f32(d * sc as f32 * q as f32));
        }
    }
    Ok(out)
}

// ── Name Mapping ──

/// Map GGUF tensor name (llama.cpp convention) to HuggingFace convention.
///
/// Examples:
/// - `token_embd.weight` → `model.embed_tokens.weight`
/// - `blk.0.attn_q.weight` → `model.layers.0.self_attn.q_proj.weight`
/// - `blk.0.ffn_gate.weight` → `model.layers.0.mlp.gate_proj.weight`
/// - `output_norm.weight` → `model.norm.weight`
/// - `output.weight` → `lm_head.weight`
/// Map GGUF tensor name to HuggingFace name with a configurable prefix.
///
/// `prefix` is typically `"model"` for Qwen3/Llama or `"model.language_model"` for Qwen3.5.
pub fn map_gguf_name_with_prefix(gguf_name: &str, prefix: &str) -> String {
    // Token embeddings
    if gguf_name == "token_embd.weight" {
        return format!("{prefix}.embed_tokens.weight");
    }
    // Output norm
    if gguf_name == "output_norm.weight" {
        return format!("{prefix}.norm.weight");
    }
    // LM head
    if gguf_name == "output.weight" {
        return "lm_head.weight".to_string();
    }

    // Layer tensors: blk.N.<suffix>
    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        if let Some((layer_str, suffix)) = rest.split_once('.') {
            let hf_suffix = match suffix {
                // Full attention (Qwen3, Llama, Gemma)
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "attn_norm.weight" => "input_layernorm.weight",
                "attn_q_norm.weight" => "self_attn.q_norm.weight",
                "attn_k_norm.weight" => "self_attn.k_norm.weight",
                // Qwen3.5 linear attention (SSM/GDR)
                "attn_qkv.weight" => "linear_attn.in_proj_qkv.weight",
                "attn_gate.weight" => "linear_attn.in_proj_z.weight",
                "ssm_alpha.weight" => "linear_attn.in_proj_a.weight",
                "ssm_beta.weight" => "linear_attn.in_proj_b.weight",
                "ssm_conv1d.weight" => "linear_attn.conv1d.weight",
                "ssm_out.weight" => "linear_attn.out_proj.weight",
                "ssm_dt.bias" => "linear_attn.dt_bias",
                "ssm_a" => "linear_attn.a_log",
                "ssm_norm.weight" => "linear_attn.norm.weight",
                // MLP
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                "post_attention_norm.weight" => "post_attention_layernorm.weight",
                // Fallthrough
                other => other,
            };
            return format!("{prefix}.layers.{layer_str}.{hf_suffix}");
        }
    }

    // Fallback: return as-is
    gguf_name.to_string()
}

/// Map GGUF tensor name to HuggingFace name (default prefix "model").
pub fn map_gguf_name(gguf_name: &str) -> String {
    map_gguf_name_with_prefix(gguf_name, "model")
}

// ── Binary Readers ──

fn read_u32(f: &mut File) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(f: &mut File) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i32(f: &mut File) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32(f: &mut File) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_string(f: &mut File) -> io::Result<String> {
    let len = read_u64(f)? as usize;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_value(f: &mut File) -> io::Result<GgufValue> {
    let vtype = read_u32(f)?;
    match vtype {
        0 => {
            let mut b = [0u8; 1];
            f.read_exact(&mut b)?;
            Ok(GgufValue::U32(b[0] as u32))
        }
        1 => {
            let mut b = [0u8; 1];
            f.read_exact(&mut b)?;
            Ok(GgufValue::I32(b[0] as i8 as i32))
        }
        2 => {
            let mut b = [0u8; 2];
            f.read_exact(&mut b)?;
            Ok(GgufValue::U32(u16::from_le_bytes(b) as u32))
        }
        3 => {
            let mut b = [0u8; 2];
            f.read_exact(&mut b)?;
            Ok(GgufValue::I32(i16::from_le_bytes(b) as i32))
        }
        4 => Ok(GgufValue::U32(read_u32(f)?)),
        5 => Ok(GgufValue::I32(read_i32(f)?)),
        6 => Ok(GgufValue::F32(read_f32(f)?)),
        7 => {
            let mut b = [0u8; 1];
            f.read_exact(&mut b)?;
            Ok(GgufValue::Bool(b[0] != 0))
        }
        8 => Ok(GgufValue::Str(read_string(f)?)),
        9 => {
            let elem_type = read_u32(f)?;
            let len = read_u64(f)? as usize;
            let mut arr = Vec::with_capacity(len.min(1024));
            for _ in 0..len {
                // For arrays, each element has the same type (no nested type tag)
                let val = read_value_of_type(f, elem_type)?;
                arr.push(val);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::U64(read_u64(f)?)),
        11 => {
            let v = read_u64(f)?;
            Ok(GgufValue::I32(v as i64 as i32))
        }
        12 => {
            let mut b = [0u8; 8];
            f.read_exact(&mut b)?;
            Ok(GgufValue::F32(f64::from_le_bytes(b) as f32))
        }
        _ => bail_io(&format!("Unknown GGUF value type: {vtype}")),
    }
}

fn read_value_of_type(f: &mut File, vtype: u32) -> io::Result<GgufValue> {
    match vtype {
        0 => {
            let mut b = [0u8; 1];
            f.read_exact(&mut b)?;
            Ok(GgufValue::U32(b[0] as u32))
        }
        4 => Ok(GgufValue::U32(read_u32(f)?)),
        5 => Ok(GgufValue::I32(read_i32(f)?)),
        6 => Ok(GgufValue::F32(read_f32(f)?)),
        8 => Ok(GgufValue::Str(read_string(f)?)),
        10 => Ok(GgufValue::U64(read_u64(f)?)),
        _ => {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            Ok(GgufValue::Other)
        }
    }
}

fn bail_io(msg: &str) -> io::Result<GgufValue> {
    Err(io::Error::new(io::ErrorKind::InvalidData, msg))
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_mapping() {
        assert_eq!(
            map_gguf_name("token_embd.weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(map_gguf_name("output_norm.weight"), "model.norm.weight");
        assert_eq!(map_gguf_name("output.weight"), "lm_head.weight");
        assert_eq!(
            map_gguf_name("blk.0.attn_q.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            map_gguf_name("blk.12.ffn_gate.weight"),
            "model.layers.12.mlp.gate_proj.weight"
        );
        assert_eq!(
            map_gguf_name("blk.5.attn_norm.weight"),
            "model.layers.5.input_layernorm.weight"
        );
    }

    #[test]
    fn test_dequant_q8_0_roundtrip() {
        // Manually construct a Q8_0 block: scale=0.5, values=[0,1,2,...,31]
        let scale_f16 = f16::from_f32(0.5);
        let scale_bytes = scale_f16.to_le_bytes();
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bytes);
        for i in 0..32u8 {
            block.push(i);
        }

        let result = dequant_q8_0(&block, 32).unwrap();
        assert_eq!(result.len(), 32);
        // First element: 0 * 0.5 = 0.0
        assert_eq!(result[0], bf16::from_f32(0.0));
        // Last element: 31 * 0.5 = 15.5
        assert!((result[31].to_f32() - 15.5).abs() < 0.1);
    }

    #[test]
    fn test_dequant_q4_0_roundtrip() {
        // Q4_0 block: scale=1.0, 16 bytes of packed nibbles
        let scale_f16 = f16::from_f32(1.0);
        let scale_bytes = scale_f16.to_le_bytes();
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bytes);
        // Pack: lo=8 (→0 after -8 offset), hi=9 (→1 after -8)
        for _ in 0..16 {
            block.push(0x98); // hi=9, lo=8
        }

        let result = dequant_q4_0(&block, 32).unwrap();
        assert_eq!(result.len(), 32);
        // lo = (8 - 8) * 1.0 = 0.0
        assert_eq!(result[0], bf16::from_f32(0.0));
        // hi = (9 - 8) * 1.0 = 1.0
        assert!((result[1].to_f32() - 1.0).abs() < 0.1);
    }

    #[test]
    #[ignore = "requires downloaded GGUF model"]
    fn test_parse_real_gguf() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
        );
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping: {path} not found");
            return;
        }

        let gguf = GgufFile::open(path).expect("Failed to parse GGUF");
        assert!(gguf.version >= 2);
        assert!(!gguf.tensors.is_empty());

        // Check architecture metadata
        let arch = gguf.architecture().expect("Missing architecture metadata");
        assert!(
            arch == "qwen3" || arch == "qwen2" || arch == "llama",
            "Unexpected architecture: {arch}"
        );

        // Check tensor count
        eprintln!("Tensors: {}", gguf.tensors.len());
        eprintln!("Architecture: {arch}");

        // Verify name mapping works for first few tensors
        for (name, info) in gguf.tensors.iter().take(5) {
            let hf_name = map_gguf_name(name);
            eprintln!("  {name} → {hf_name} ({:?}, {:?})", info.dtype, info.shape);
        }

        // Try reading and dequantizing one tensor
        let first_name = gguf.tensors.keys().next().unwrap();
        let bf16_data = gguf
            .read_tensor_bf16(first_name)
            .expect("Failed to dequant");
        let info = &gguf.tensors[first_name];
        assert_eq!(bf16_data.len(), info.numel());
        eprintln!(
            "Dequantized '{}': {} elements, first={}, last={}",
            first_name,
            bf16_data.len(),
            bf16_data[0],
            bf16_data[bf16_data.len() - 1]
        );
    }
}
