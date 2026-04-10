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
            Self::Q8_0 => 2 + 32, // f16 scale + 32 × i8
            Self::Q4_0 => 2 + 16, // f16 scale + 32 nibbles in 16 bytes
            Self::Q4_K_M => 144,  // complex: 256 elements per superblock
            _ => 0,               // unsupported — caught at dequant time
        }
    }

    /// Elements per block.
    fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q8_0 | Self::Q4_0 => 32,
            Self::Q4_K_M | Self::Q4_K_S => 256,
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
        _ => bail!(
            "Dequant not yet implemented for {:?}. Supported: F32, F16, BF16, Q8_0, Q4_0",
            dtype
        ),
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

// ── Name Mapping ──

/// Map GGUF tensor name (llama.cpp convention) to HuggingFace convention.
///
/// Examples:
/// - `token_embd.weight` → `model.embed_tokens.weight`
/// - `blk.0.attn_q.weight` → `model.layers.0.self_attn.q_proj.weight`
/// - `blk.0.ffn_gate.weight` → `model.layers.0.mlp.gate_proj.weight`
/// - `output_norm.weight` → `model.norm.weight`
/// - `output.weight` → `lm_head.weight`
pub fn map_gguf_name(gguf_name: &str) -> String {
    // Token embeddings
    if gguf_name == "token_embd.weight" {
        return "model.embed_tokens.weight".to_string();
    }
    // Output norm
    if gguf_name == "output_norm.weight" {
        return "model.norm.weight".to_string();
    }
    // LM head
    if gguf_name == "output.weight" {
        return "lm_head.weight".to_string();
    }

    // Layer tensors: blk.N.<suffix>.weight
    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        if let Some((layer_str, suffix)) = rest.split_once('.') {
            let hf_suffix = match suffix {
                // Attention
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "attn_norm.weight" => "input_layernorm.weight",
                "attn_q_norm.weight" => "self_attn.q_norm.weight",
                "attn_k_norm.weight" => "self_attn.k_norm.weight",
                // MLP
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                // Gemma post-attention norm
                "post_attention_norm.weight" => "post_attention_layernorm.weight",
                // Fallthrough
                other => other,
            };
            return format!("model.layers.{layer_str}.{hf_suffix}");
        }
    }

    // Fallback: return as-is
    gguf_name.to_string()
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
