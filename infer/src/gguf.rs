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
use std::fs;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use half::{bf16, f16};
use qwen35_spec::{LayerType as Qwen35LayerType, Qwen35Config};

// ── GGUF Constants ──

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" as little-endian u32 (bytes: 47 47 55 46)

/// GGUF tensor element types. Values match llama.cpp `enum ggml_type` exactly
/// — see `ggml.h`. S / M / L in names like "Q4_K_M" are *file-level recipe
/// tiers* that mix several per-tensor dtypes (mostly Q4_K + some Q6_K); they
/// are NOT distinct per-tensor types and do NOT appear in GGUF tensor headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4,5 = deprecated Q4_2/Q4_3
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
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
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            30 => Ok(Self::BF16),
            _ => bail!("unsupported GGML type: {v}"),
        }
    }

    /// Bytes per block for this type. Each block contains `block_size()` elements.
    fn block_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 => 1,
            Self::Q8_0 => 2 + 32,           // f16 scale + 32 × i8 = 34
            Self::Q8_1 => 4 + 32,           // 2×f16 scale + 32 × i8 = 36
            Self::Q4_0 => 2 + 16,           // f16 scale + 16 bytes = 18
            Self::Q4_1 => 4 + 16,           // 2×f16 scale + 16 bytes = 20
            Self::Q5_0 => 2 + 4 + 16,       // f16 + qh(4) + qs(16) = 22
            Self::Q5_1 => 4 + 4 + 16,       // 2×f16 + qh(4) + qs(16) = 24
            Self::Q2_K => 16 + 64 + 2 + 2,  // scales(16)+qs(64)+d(2)+dmin(2) = 84
            Self::Q3_K => 110,              // hmask(32)+qs(64)+scales(12)+d(2)
            Self::Q4_K => 144,              // d(2)+dmin(2)+scales(12)+qs(128)
            Self::Q5_K => 176,              // d(2)+dmin(2)+scales(12)+qh(32)+qs(128)
            Self::Q6_K => 210,              // ql(128)+qh(64)+scales(16)+d(2)
            Self::Q8_K => 4 + 256 + 16 * 2, // d(4)+qs(256)+bsums(32) = 292
        }
    }

    /// Elements per block.
    fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::I8 | Self::I16 | Self::I32 => 1,
            Self::Q8_0 | Self::Q8_1 | Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
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
        let num_blocks = n.div_ceil(bs);
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
        if !(2..=3).contains(&version) {
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
            .is_some_and(|t| !matches!(t.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16))
    }

    /// Read Q8_0 tensor in packed format: split into (`qweight: Vec<i8>`, `scales: Vec<bf16>`).
    ///
    /// Returns data suitable for `DeviceMatrix::from_quantized_int8`:
    ///   - qweight: `[numel]` i8 values (one per element)
    ///   - scales: `[rows × num_groups]` bf16 (one per 32-element block)
    pub fn read_tensor_q8_packed(&self, name: &str) -> Result<(Vec<i8>, Vec<bf16>, usize)> {
        const BLOCK_SIZE: usize = 32;
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found in GGUF", name))?;
        if info.dtype != GgmlType::Q8_0 {
            bail!("Expected Q8_0, got {:?}", info.dtype);
        }
        let raw = self.read_tensor_raw(name)?;
        let numel = info.numel();
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

    /// Read a Q6_K tensor in packed form (raw 210-byte superblock layout).
    ///
    /// Layout per superblock: `ql(128) | qh(64) | scales(16 × i8) | d:f16(2)`.
    /// Element dequant: `w = d * scales[j/16] * ((low4 | high2<<4) - 32)`.
    pub fn read_tensor_q6k_packed(&self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found in GGUF", name))?;
        if info.dtype != GgmlType::Q6_K {
            bail!("Expected Q6_K, got {:?}", info.dtype);
        }
        let numel = info.numel();
        if numel % 256 != 0 {
            bail!(
                "Q6_K tensor '{}' numel {} is not a multiple of 256",
                name,
                numel
            );
        }
        let expected = (numel / 256) * 210;
        let raw = self.read_tensor_raw(name)?;
        if raw.len() != expected {
            bail!(
                "Q6_K tensor '{}': expected {} bytes ({} superblocks), got {}",
                name,
                expected,
                numel / 256,
                raw.len()
            );
        }
        Ok(raw)
    }

    /// Read a Q3_K tensor in packed form (raw 110-byte superblock layout).
    ///
    /// Layout per superblock: `hmask(32) | qs(64, 2-bit) | scales(12, 6-bit signed) | d:f16(2)`.
    pub fn read_tensor_q3k_packed(&self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found in GGUF", name))?;
        if info.dtype != GgmlType::Q3_K {
            bail!("Expected Q3_K, got {:?}", info.dtype);
        }
        let numel = info.numel();
        if numel % 256 != 0 {
            bail!(
                "Q3_K tensor '{}' numel {} is not a multiple of 256",
                name,
                numel
            );
        }
        let expected = (numel / 256) * 110;
        let raw = self.read_tensor_raw(name)?;
        if raw.len() != expected {
            bail!(
                "Q3_K tensor '{}': expected {} bytes ({} superblocks), got {}",
                name,
                expected,
                numel / 256,
                raw.len()
            );
        }
        Ok(raw)
    }

    /// Read a Q4_K_M / Q4_K_S tensor in packed form (raw 144-byte superblock layout).
    ///
    /// Unlike `read_tensor_bf16`, this returns the GGUF bytes verbatim so they can be
    /// uploaded directly to the GPU without a BF16 intermediate. Used by the native
    /// Q4_K GEMV kernel. Validates dtype and element/block count.
    ///
    /// Returned layout, per row `[N, K]`:
    ///   - `K/256` superblocks per row, 144 bytes each
    ///   - Superblock: `d:f16(2) | dmin:f16(2) | scales_packed(12) | qs(128)`
    ///
    /// Callers that need the K-dim contiguous-per-row reinterpretation (same trick
    /// as `read_tensor_q8_packed`) should treat the output as `[ne1 * ne0/256 * 144]`.
    pub fn read_tensor_q4k_packed(&self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found in GGUF", name))?;
        if info.dtype != GgmlType::Q4_K {
            bail!("Expected Q4_K, got {:?}", info.dtype);
        }
        let numel = info.numel();
        if numel % 256 != 0 {
            bail!(
                "Q4_K tensor '{}' numel {} is not a multiple of 256",
                name,
                numel
            );
        }
        let expected = (numel / 256) * 144;
        let raw = self.read_tensor_raw(name)?;
        if raw.len() != expected {
            bail!(
                "Q4_K tensor '{}': expected {} bytes ({} superblocks), got {}",
                name,
                expected,
                numel / 256,
                raw.len()
            );
        }
        Ok(raw)
    }

    /// Read a Q5_K tensor in packed form (raw 176-byte superblock layout).
    ///
    /// Layout per superblock:
    /// `d:f16(2) | dmin:f16(2) | scales_packed(12) | qh(32) | qs(128)`.
    pub fn read_tensor_q5k_packed(&self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found in GGUF", name))?;
        if info.dtype != GgmlType::Q5_K {
            bail!("Expected Q5_K, got {:?}", info.dtype);
        }
        let numel = info.numel();
        if numel % 256 != 0 {
            bail!(
                "Q5_K tensor '{}' numel {} is not a multiple of 256",
                name,
                numel
            );
        }
        let expected = (numel / 256) * 176;
        let raw = self.read_tensor_raw(name)?;
        if raw.len() != expected {
            bail!(
                "Q5_K tensor '{}': expected {} bytes ({} superblocks), got {}",
                name,
                expected,
                numel / 256,
                raw.len()
            );
        }
        Ok(raw)
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

    /// Read and dequantize a tensor to F32.
    ///
    /// Keeps native F32 tensors lossless and promotes every other supported
    /// storage type through the existing BF16 dequant path.
    pub fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found in GGUF", name))?;
        let raw = self.read_tensor_raw(name)?;
        if info.dtype == GgmlType::F32 {
            return Ok(raw
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect());
        }
        Ok(dequant_to_bf16(&raw, info.dtype, info.numel())?
            .into_iter()
            .map(bf16::to_f32)
            .collect())
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

    pub fn extract_qwen35_config(&self) -> Result<Qwen35Config> {
        let common = self.extract_model_config()?;
        let full_attention_interval = self
            .meta_u32("qwen35.full_attention_interval")
            .unwrap_or(1)
            .max(1) as usize;
        let rotary_dim = self
            .meta_u32("qwen35.rope.dimension_count")
            .context("GGUF missing qwen35.rope.dimension_count")? as usize;
        let linear_num_key_heads =
            self.meta_u32("qwen35.ssm.group_count")
                .context("GGUF missing qwen35.ssm.group_count")? as usize;
        let linear_key_head_dim =
            self.meta_u32("qwen35.ssm.state_size")
                .context("GGUF missing qwen35.ssm.state_size")? as usize;
        let linear_inner_size =
            self.meta_u32("qwen35.ssm.inner_size")
                .context("GGUF missing qwen35.ssm.inner_size")? as usize;
        anyhow::ensure!(
            linear_key_head_dim > 0 && linear_inner_size.is_multiple_of(linear_key_head_dim),
            "invalid Qwen3.5 GGUF SSM metadata: inner_size={linear_inner_size}, state_size={linear_key_head_dim}"
        );

        let eos_token_id = self
            .meta_u32("tokenizer.ggml.eos_token_id")
            .unwrap_or(151_645);
        let bos_token_id = self.meta_u32("tokenizer.ggml.bos_token_id");
        let config = Qwen35Config {
            hidden_size: common.hidden_size,
            intermediate_size: common.intermediate_size,
            num_hidden_layers: common.num_hidden_layers,
            vocab_size: common.vocab_size,
            rms_norm_eps: common.rms_norm_eps,
            stop_token_ids: vec![eos_token_id],
            bos_token_id,
            eos_token_id,
            tie_word_embeddings: true,
            num_attention_heads: common.num_attention_heads,
            num_key_value_heads: common.num_key_value_heads,
            head_dim: common.head_dim,
            linear_num_key_heads,
            linear_key_head_dim,
            linear_num_value_heads: linear_inner_size / linear_key_head_dim,
            linear_value_head_dim: linear_key_head_dim,
            linear_conv_kernel_dim: self.meta_u32("qwen35.ssm.conv_kernel").unwrap_or(4) as usize,
            rope_theta: common.rope_theta,
            partial_rotary_factor: rotary_dim as f32 / common.head_dim as f32,
            rotary_dim,
            rope_cache_len_hint: Some(common.context_length),
            layer_types: qwen35_layer_types_from_interval(
                common.num_hidden_layers,
                full_attention_interval,
            ),
            num_experts: 0,
            num_experts_per_tok: 0,
            decoder_sparse_step: 1,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            norm_topk_prob: true,
            mlp_only_layers: Vec::new(),
        };
        config.validate().map_err(anyhow::Error::from)?;
        Ok(config)
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

/// Reverse llama.cpp's `_reorder_v_heads` permutation for BF16 1-D data.
///
/// GGUF stores V heads tiled by V-within-K-group; the HF/Qwen runtime paths
/// expect them grouped by K head. `head_dim=1` covers scalar tensors like
/// `dt_bias`.
pub fn reverse_v_reorder(
    data: &mut [bf16],
    num_k_heads: usize,
    num_v_per_k: usize,
    head_dim: usize,
) {
    if num_v_per_k <= 1 {
        return;
    }
    let src = data.to_vec();
    for k in 0..num_k_heads {
        for v in 0..num_v_per_k {
            for d in 0..head_dim {
                let gguf_idx = (v * num_k_heads + k) * head_dim + d;
                let hf_idx = (k * num_v_per_k + v) * head_dim + d;
                data[hf_idx] = src[gguf_idx];
            }
        }
    }
}

/// Same V-head reorder reversal as [`reverse_v_reorder`], but for `f32`.
pub fn reverse_v_reorder_f32(
    data: &mut [f32],
    num_k_heads: usize,
    num_v_per_k: usize,
    head_dim: usize,
) {
    if num_v_per_k <= 1 {
        return;
    }
    let src = data.to_vec();
    for k in 0..num_k_heads {
        for v in 0..num_v_per_k {
            for d in 0..head_dim {
                let gguf_idx = (v * num_k_heads + k) * head_dim + d;
                let hf_idx = (k * num_v_per_k + v) * head_dim + d;
                data[hf_idx] = src[gguf_idx];
            }
        }
    }
}

/// Reverse llama.cpp's V-head tiling for BF16 2-D row-major matrices where
/// the reordered dimension lives on rows.
pub fn reverse_v_reorder_rows(
    data: &mut [bf16],
    rows: usize,
    cols: usize,
    num_k_heads: usize,
    num_v_per_k: usize,
    head_dim: usize,
) {
    if num_v_per_k <= 1 {
        return;
    }
    let src = data.to_vec();
    let _ = rows;
    for k in 0..num_k_heads {
        for v in 0..num_v_per_k {
            let gguf_head = v * num_k_heads + k;
            let hf_head = k * num_v_per_k + v;
            let src_start = gguf_head * head_dim * cols;
            let dst_start = hf_head * head_dim * cols;
            let size = head_dim * cols;
            data[dst_start..dst_start + size].copy_from_slice(&src[src_start..src_start + size]);
        }
    }
}

/// Reverse llama.cpp's V-head tiling for BF16 2-D row-major matrices where
/// the reordered dimension lives on columns.
pub fn reverse_v_reorder_cols(
    data: &mut [bf16],
    rows: usize,
    cols: usize,
    num_k_heads: usize,
    num_v_per_k: usize,
    head_dim: usize,
) {
    if num_v_per_k <= 1 {
        return;
    }
    let src = data.to_vec();
    for r in 0..rows {
        for k in 0..num_k_heads {
            for v in 0..num_v_per_k {
                let gguf_head = v * num_k_heads + k;
                let hf_head = k * num_v_per_k + v;
                for d in 0..head_dim {
                    data[r * cols + hf_head * head_dim + d] =
                        src[r * cols + gguf_head * head_dim + d];
                }
            }
        }
    }
}

/// Resolve a HuggingFace tensor name to the actual GGUF tensor name.
///
/// Tries direct lookup first, then reverse-maps llama.cpp names using the
/// Qwen/Qwen3.5 prefixes we support today.
pub fn find_tensor_name(gguf: &GgufFile, hf_name: &str) -> Result<String> {
    if gguf.tensors.contains_key(hf_name) {
        return Ok(hf_name.to_string());
    }

    let prefixes = ["model", "model.language_model"];
    for gguf_name in gguf.tensors.keys() {
        for prefix in &prefixes {
            if map_gguf_name_with_prefix(gguf_name, prefix) == hf_name {
                return Ok(gguf_name.clone());
            }
        }
    }

    bail!(
        "Tensor '{}' not found in GGUF (tried {} tensor names with prefixes {:?})",
        hf_name,
        gguf.tensors.len(),
        prefixes
    )
}

/// Open a GGUF file from either a direct `.gguf` path or a directory
/// containing at least one `.gguf`.
pub fn try_open(model_path: &str) -> Option<GgufFile> {
    let path = Path::new(model_path);
    if path.is_file()
        && path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        return GgufFile::open(path.to_str()?).ok();
    }
    if !path.is_dir() {
        return None;
    }
    for entry in fs::read_dir(path).ok()? {
        let entry = entry.ok()?;
        if !entry.file_type().ok()?.is_file() {
            continue;
        }
        let candidate = entry.path();
        if candidate
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            match GgufFile::open(candidate.to_str()?) {
                Ok(gguf) => return Some(gguf),
                Err(err) => {
                    log::warn!("Failed to parse GGUF {}: {}", candidate.display(), err);
                }
            }
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct HostTensor<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

impl<T> HostTensor<T> {
    fn vector(data: Vec<T>) -> Self {
        let len = data.len();
        Self {
            data,
            shape: vec![len],
        }
    }

    fn matrix(data: Vec<T>, rows: usize, cols: usize) -> Self {
        Self {
            data,
            shape: vec![rows, cols],
        }
    }

    fn with_shape(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
}

fn gguf_vector_len(gguf: &GgufFile, hf_name: &str) -> Result<usize> {
    let gguf_name = find_tensor_name(gguf, hf_name)?;
    let info = &gguf.tensors[&gguf_name];
    anyhow::ensure!(
        info.shape.len() == 1,
        "expected 1D GGUF tensor for '{hf_name}', got {}D",
        info.shape.len()
    );
    Ok(info.shape[0] as usize)
}

fn gguf_matrix_dims(gguf: &GgufFile, hf_name: &str) -> Result<(usize, usize)> {
    let gguf_name = find_tensor_name(gguf, hf_name)?;
    let info = &gguf.tensors[&gguf_name];
    anyhow::ensure!(
        info.shape.len() == 2,
        "expected 2D GGUF tensor for '{hf_name}', got {}D",
        info.shape.len()
    );
    Ok((info.shape[1] as usize, info.shape[0] as usize))
}

pub fn load_vector_bf16_host(gguf: &GgufFile, hf_name: &str) -> Result<HostTensor<bf16>> {
    let gguf_name = find_tensor_name(gguf, hf_name)?;
    let data = gguf.read_tensor_bf16(&gguf_name)?;
    let len = gguf_vector_len(gguf, hf_name)?;
    anyhow::ensure!(
        data.len() == len,
        "unexpected element count for '{hf_name}': got {}, expected {len}",
        data.len()
    );
    Ok(HostTensor::vector(data))
}

pub fn load_vector_offset_norm_bf16_host(
    gguf: &GgufFile,
    hf_name: &str,
) -> Result<HostTensor<bf16>> {
    let mut tensor = load_vector_bf16_host(gguf, hf_name)?;
    for value in &mut tensor.data {
        *value = bf16::from_f32(value.to_f32() - 1.0);
    }
    Ok(tensor)
}

pub fn load_vector_v_reorder_bf16_host(
    gguf: &GgufFile,
    hf_name: &str,
    num_k_heads: usize,
    num_v_per_k: usize,
    head_dim: usize,
) -> Result<HostTensor<bf16>> {
    let mut tensor = load_vector_bf16_host(gguf, hf_name)?;
    reverse_v_reorder(&mut tensor.data, num_k_heads, num_v_per_k, head_dim);
    Ok(tensor)
}

pub fn load_vector_f32_host(gguf: &GgufFile, hf_name: &str) -> Result<HostTensor<f32>> {
    let gguf_name = find_tensor_name(gguf, hf_name)?;
    let data = gguf.read_tensor_f32(&gguf_name)?;
    let len = gguf_vector_len(gguf, hf_name)?;
    anyhow::ensure!(
        data.len() == len,
        "unexpected element count for '{hf_name}': got {}, expected {len}",
        data.len()
    );
    Ok(HostTensor::vector(data))
}

pub fn load_matrix_bf16_host(gguf: &GgufFile, hf_name: &str) -> Result<HostTensor<bf16>> {
    let gguf_name = find_tensor_name(gguf, hf_name)?;
    let data = gguf.read_tensor_bf16(&gguf_name)?;
    let (rows, cols) = gguf_matrix_dims(gguf, hf_name)?;
    anyhow::ensure!(
        data.len() == rows * cols,
        "unexpected element count for '{hf_name}': got {}, expected {}",
        data.len(),
        rows * cols
    );
    Ok(HostTensor::matrix(data, rows, cols))
}

pub fn load_matrix_v_reorder_rows_bf16_host(
    gguf: &GgufFile,
    hf_name: &str,
    num_k_heads: usize,
    num_v_per_k: usize,
    head_dim: usize,
) -> Result<HostTensor<bf16>> {
    let mut tensor = load_matrix_bf16_host(gguf, hf_name)?;
    let rows = tensor.shape[0];
    let cols = tensor.shape[1];
    reverse_v_reorder_rows(
        &mut tensor.data,
        rows,
        cols,
        num_k_heads,
        num_v_per_k,
        head_dim,
    );
    Ok(tensor)
}

pub fn load_matrix_v_reorder_cols_bf16_host(
    gguf: &GgufFile,
    hf_name: &str,
    num_k_heads: usize,
    num_v_per_k: usize,
    head_dim: usize,
) -> Result<HostTensor<bf16>> {
    let mut tensor = load_matrix_bf16_host(gguf, hf_name)?;
    let rows = tensor.shape[0];
    let cols = tensor.shape[1];
    reverse_v_reorder_cols(
        &mut tensor.data,
        rows,
        cols,
        num_k_heads,
        num_v_per_k,
        head_dim,
    );
    Ok(tensor)
}

pub fn load_qwen35_qkv_matrix_bf16_host(
    gguf: &GgufFile,
    hf_name: &str,
    num_k_heads: usize,
    num_v_per_k: usize,
    key_head_dim: usize,
    value_head_dim: usize,
) -> Result<HostTensor<bf16>> {
    let mut tensor = load_matrix_bf16_host(gguf, hf_name)?;
    let cols = tensor.shape[1];
    let q_rows = num_k_heads * key_head_dim;
    let k_rows = num_k_heads * key_head_dim;
    let v_rows = num_k_heads * num_v_per_k * value_head_dim;
    let expected_rows = q_rows + k_rows + v_rows;
    anyhow::ensure!(
        tensor.shape[0] == expected_rows,
        "unexpected Qwen3.5 QKV rows for '{hf_name}': got {}, expected {expected_rows}",
        tensor.shape[0]
    );
    let v_start = (q_rows + k_rows) * cols;
    reverse_v_reorder_rows(
        &mut tensor.data[v_start..v_start + v_rows * cols],
        v_rows,
        cols,
        num_k_heads,
        num_v_per_k,
        value_head_dim,
    );
    Ok(tensor)
}

pub fn load_qwen35_conv1d_bf16_host(
    gguf: &GgufFile,
    hf_name: &str,
    num_key_heads: usize,
    key_head_dim: usize,
    num_value_heads: usize,
    value_head_dim: usize,
    kernel_dim: usize,
) -> Result<HostTensor<bf16>> {
    let gguf_name = find_tensor_name(gguf, hf_name)?;
    let mut data = gguf.read_tensor_bf16(&gguf_name)?;
    let qk_channels = key_head_dim * num_key_heads * 2;
    let v_channels = num_value_heads * value_head_dim;
    let expected = (qk_channels + v_channels) * kernel_dim;
    anyhow::ensure!(
        data.len() == expected,
        "unexpected conv1d weight size for '{hf_name}': got {}, expected {expected}",
        data.len()
    );
    let v_start = qk_channels * kernel_dim;
    reverse_v_reorder_rows(
        &mut data[v_start..v_start + v_channels * kernel_dim],
        v_channels,
        kernel_dim,
        num_key_heads,
        num_value_heads / num_key_heads,
        value_head_dim,
    );
    Ok(HostTensor::with_shape(
        data,
        vec![qk_channels + v_channels, kernel_dim, 1],
    ))
}

pub fn load_qwen35_a_log_f32_host(
    gguf: &GgufFile,
    hf_name: &str,
    num_k_heads: usize,
    num_v_per_k: usize,
) -> Result<HostTensor<f32>> {
    let mut tensor = load_vector_f32_host(gguf, hf_name)?;
    for value in &mut tensor.data {
        let abs_a = value.abs().max(1e-10);
        *value = abs_a.ln();
    }
    reverse_v_reorder_f32(&mut tensor.data, num_k_heads, num_v_per_k, 1);
    Ok(tensor)
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

fn qwen35_layer_types_from_interval(
    num_hidden_layers: usize,
    full_attention_interval: usize,
) -> Vec<Qwen35LayerType> {
    (0..num_hidden_layers)
        .map(|idx| {
            if (idx + 1).is_multiple_of(full_attention_interval.max(1)) {
                Qwen35LayerType::FullAttention
            } else {
                Qwen35LayerType::LinearAttention
            }
        })
        .collect()
}

// ── Dequantization ──

/// Dequantize raw GGUF tensor bytes to BF16.
pub(crate) fn dequant_to_bf16(raw: &[u8], dtype: GgmlType, numel: usize) -> Result<Vec<bf16>> {
    match dtype {
        GgmlType::BF16 => {
            assert_eq!(raw.len(), numel * 2);
            Ok(raw
                .chunks_exact(2)
                .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect())
        }
        GgmlType::F16 => {
            assert_eq!(raw.len(), numel * 2);
            Ok(raw
                .chunks_exact(2)
                .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
                .map(|v| bf16::from_f32(v.to_f32()))
                .collect())
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
            Ok(raw
                .chunks_exact(2)
                .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                .map(|v| bf16::from_f32(v as f32))
                .collect())
        }
        GgmlType::F32 => {
            assert_eq!(raw.len(), numel * 4);
            Ok(raw
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .map(bf16::from_f32)
                .collect())
        }
        GgmlType::Q8_0 => Ok(dequant_q8_0(raw, numel)),
        GgmlType::Q4_0 => Ok(dequant_q4_0(raw, numel)),
        GgmlType::Q3_K => dequant_q3_k(raw, numel),
        GgmlType::Q4_K => dequant_q4_k(raw, numel),
        GgmlType::Q5_K => dequant_q5_k(raw, numel),
        GgmlType::Q6_K => dequant_q6_k(raw, numel),
        _ => bail!("Dequant not yet implemented for {:?}", dtype),
    }
}

/// Q8_0: 32 elements per block. Layout: f16 scale + 32× i8.
fn dequant_q8_0(raw: &[u8], numel: usize) -> Vec<bf16> {
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
    out
}

/// Q4_0: 32 elements per block. Layout: f16 scale + 16 bytes (2 nibbles each).
fn dequant_q4_0(raw: &[u8], numel: usize) -> Vec<bf16> {
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
    out
}

/// Decode one packed `(scale, min)` pair from the 12-byte K-quants header.
///
/// This mirrors ggml's `get_scale_min_k4()` bit layout exactly.
fn decode_scale_min_k4(scales: &[u8], index: usize) -> (u8, u8) {
    debug_assert!(scales.len() >= 12);
    debug_assert!(index < 8);
    if index < 4 {
        (scales[index] & 0x3F, scales[index + 4] & 0x3F)
    } else {
        let scale = (scales[index + 4] & 0x0F) | ((scales[index - 4] >> 6) << 4);
        let min = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
        (scale, min)
    }
}

/// Q4_K (Q4_K_M/Q4_K_S): 256 elements per superblock (144 bytes). Mirrors
/// llama.cpp's `dequantize_row_q4_K` exactly.
///
/// Element layout:
///
/// NOT the naive "2 elements per ql byte" interpretation!
/// The superblock is 4 outer iterations of 64 elements. Each iteration reads
/// 32 contiguous ql bytes and splits them into two 32-element halves:
///   - first half  uses `ql[l].low  nibble` with scale `sc[2*iter + 0]`
///   - second half uses `ql[l].high nibble` with scale `sc[2*iter + 1]`
///     Then ql advances by 32 bytes → next outer iteration.
///
/// Block layout:
///   d(f16) | dmin(f16) | scales(12B) | qs(128B)
fn dequant_q4_k(raw: &[u8], numel: usize) -> Result<Vec<bf16>> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 144;
    let num_blocks = numel / QK_K;
    if raw.len() < num_blocks * BLOCK_BYTES {
        bail!(
            "Q4_K: expected {} bytes for {} elements, got {}",
            num_blocks * BLOCK_BYTES,
            numel,
            raw.len()
        );
    }

    let mut out = vec![bf16::ZERO; numel];

    for b in 0..num_blocks {
        let base = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([raw[base], raw[base + 1]]).to_f32();
        let dmin = f16::from_le_bytes([raw[base + 2], raw[base + 3]]).to_f32();
        let scales_raw = &raw[base + 4..base + 16];
        let qs = &raw[base + 16..base + 144];

        let out_base = b * QK_K;
        // 4 outer iterations of 64 elements (2 sub-blocks each).
        for iter in 0..4 {
            let j_lo = iter * 2;
            let j_hi = j_lo + 1;
            let (sc_lo, mn_lo) = decode_scale_min_k4(scales_raw, j_lo);
            let (sc_hi, mn_hi) = decode_scale_min_k4(scales_raw, j_hi);
            let d1 = d * sc_lo as f32;
            let m1 = dmin * mn_lo as f32;
            let d2 = d * sc_hi as f32;
            let m2 = dmin * mn_hi as f32;
            let ql_slice = &qs[iter * 32..iter * 32 + 32];
            for l in 0..32 {
                let byte = ql_slice[l];
                let lo = (byte & 0x0F) as f32;
                let hi = (byte >> 4) as f32;
                out[out_base + j_lo * 32 + l] = bf16::from_f32(lo * d1 - m1);
                out[out_base + j_hi * 32 + l] = bf16::from_f32(hi * d2 - m2);
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

        // Decode 16 sub-block scales. Each is a 6-bit UNSIGNED value in 0..63,
        // packed across `scales_raw`:
        //   * low 4 bits come from the low/high nibble of scales_raw[0..8]
        //     (i<8 → low nibble of raw[i]; i≥8 → high nibble of raw[i-8])
        //   * high 2 bits come from scales_raw[8..12] at positions determined
        //     by (i mod 4) / (i / 4)
        //
        // The signed scale is the 6-bit unsigned value minus 32 (range -32..31).
        //
        // NOTE: must combine the 6 bits BEFORE subtracting 32 — if you subtract
        // first and then OR in the top 2 bits, sign-extension corrupts the
        // result whenever the low-4-bit scale already landed in -8..-1, because
        // bit 4 is already set from the sign extension and the OR is a no-op.
        let mut scales = [0i8; 16];
        let low_nibble = |i: usize| -> u8 {
            if i < 8 {
                scales_raw[i] & 0x0F
            } else {
                (scales_raw[i - 8] >> 4) & 0x0F
            }
        };
        let high_bits = |i: usize| -> u8 {
            // sub-block index 0..16 → which byte/shift in scales_raw[8..12]
            let byte = 8 + (i & 3);
            let shift = 2 * (i / 4);
            (scales_raw[byte] >> shift) & 0x03
        };
        for (i, scale) in scales.iter_mut().enumerate() {
            let u6: u8 = low_nibble(i) | (high_bits(i) << 4);
            *scale = (u6 as i32 - 32) as i8;
        }

        for j in 0..QK_K {
            // 2-bit value from qs (4 per byte)
            let q2 = (qs[j / 4] >> ((j % 4) * 2)) & 0x03;
            // High bit from hmask
            let hbit = (hmask[j / 8] >> (j % 8)) & 1;
            // 3-bit value
            let q3 = q2 | (hbit << 2);
            let sc = scales[j / 16] as f32;
            out.push(bf16::from_f32(d * sc * (q3 as f32 - 4.0)));
        }
    }
    Ok(out)
}

/// Q5_K: 256 elements per superblock (176 bytes). Mirrors llama.cpp's
/// `dequantize_row_q5_K` — the 8 sub-blocks of 32 elements are laid out in 4
/// outer iterations of 64 elements, where each iteration:
///   - shares one `ql[iter*32 + l]` byte for the low 4 bits of BOTH its halves
///   - the FIRST half (sub-block `iter*2`) takes the low nibble + `qh[l]` bit `iter*2`
///   - the SECOND half (sub-block `iter*2+1`) takes the high nibble + `qh[l]` bit `iter*2+1`
///
/// Layout: d(f16) | dmin(f16) | scales(12B) | qh(32B, high bits) | qs(128B, low 4 bits)
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

    let mut out = vec![bf16::ZERO; numel];
    for b in 0..num_blocks {
        let base = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([raw[base], raw[base + 1]]).to_f32();
        let dmin = f16::from_le_bytes([raw[base + 2], raw[base + 3]]).to_f32();
        let scales_raw = &raw[base + 4..base + 16];
        let qh = &raw[base + 16..base + 48];
        let qs = &raw[base + 48..base + 176];

        let out_base = b * QK_K;
        // 4 outer iterations of 64 elements, 2 sub-blocks each.
        for iter in 0..4 {
            let j_lo = iter * 2;
            let j_hi = j_lo + 1;
            let (sc_lo, mn_lo) = decode_scale_min_k4(scales_raw, j_lo);
            let (sc_hi, mn_hi) = decode_scale_min_k4(scales_raw, j_hi);
            let d1 = d * sc_lo as f32;
            let m1 = dmin * mn_lo as f32;
            let d2 = d * sc_hi as f32;
            let m2 = dmin * mn_hi as f32;
            let ql_slice = &qs[iter * 32..iter * 32 + 32];
            for l in 0..32 {
                let ql_byte = ql_slice[l];
                // First half: low nibble + qh[l] bit (2*iter + 0)
                let lo_nib = ql_byte & 0x0F;
                let hbit_lo = (qh[l] >> j_lo) & 1;
                let q5_lo = lo_nib | (hbit_lo << 4);
                out[out_base + j_lo * 32 + l] = bf16::from_f32(q5_lo as f32 * d1 - m1);
                // Second half: high nibble + qh[l] bit (2*iter + 1)
                let hi_nib = ql_byte >> 4;
                let hbit_hi = (qh[l] >> j_hi) & 1;
                let q5_hi = hi_nib | (hbit_hi << 4);
                out[out_base + j_hi * 32 + l] = bf16::from_f32(q5_hi as f32 * d2 - m2);
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
///
/// Q6_K: 256 elements per superblock (210 bytes). Mirrors llama.cpp's
/// `dequantize_row_q6_K` exactly.
///
/// The layout is NOT the naive "low nibble then high nibble of each ql byte".
/// Each half of 128 elements interleaves across four 32-element quadrants drawn
/// from `ql[l]`/`ql[l+32]` low/high nibbles and `qh[l]` bit pairs.
///
/// Block layout:
///   ql:      128 bytes (lower 4 bits of each weight, 2 per byte)
///   qh:       64 bytes (upper 2 bits of each weight, 4 per byte)
///   scales:   16 × i8  (one per 16-element sub-block)
///   d:        f16
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

    let mut out = vec![bf16::ZERO; numel];

    for b in 0..num_blocks {
        let base = b * BLOCK_BYTES;
        let ql_all = &raw[base..base + 128];
        let qh_all = &raw[base + 128..base + 192];
        let scales_all = &raw[base + 192..base + 208];
        let d = f16::from_le_bytes([raw[base + 208], raw[base + 209]]).to_f32();
        let out_base = b * QK_K;

        // Two halves of 128 elements each.
        for half in 0..2 {
            let ql = &ql_all[half * 64..(half + 1) * 64];
            let qh = &qh_all[half * 32..(half + 1) * 32];
            let sc = &scales_all[half * 8..(half + 1) * 8];
            let y_off = out_base + half * 128;

            for l in 0..32 {
                let is = l / 16;
                let q1 = (((ql[l] & 0x0F) | ((qh[l] & 0x03) << 4)) as i32 - 32) as i8;
                let q2 = (((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) as i32 - 32) as i8;
                let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i32 - 32) as i8;
                let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i32 - 32) as i8;
                out[y_off + l] = bf16::from_f32(d * sc[is] as i8 as f32 * q1 as f32);
                out[y_off + l + 32] = bf16::from_f32(d * sc[is + 2] as i8 as f32 * q2 as f32);
                out[y_off + l + 64] = bf16::from_f32(d * sc[is + 4] as i8 as f32 * q3 as f32);
                out[y_off + l + 96] = bf16::from_f32(d * sc[is + 6] as i8 as f32 * q4 as f32);
            }
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
///
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
                "ffn_norm.weight" | "post_attention_norm.weight" => {
                    "post_attention_layernorm.weight"
                }
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

        let result = dequant_q8_0(&block, 32);
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
        block.extend(std::iter::repeat_n(0x98u8, 16)); // hi=9, lo=8 × 16 bytes

        let result = dequant_q4_0(&block, 32);
        assert_eq!(result.len(), 32);
        // lo = (8 - 8) * 1.0 = 0.0
        assert_eq!(result[0], bf16::from_f32(0.0));
        // hi = (9 - 8) * 1.0 = 1.0
        assert!((result[1].to_f32() - 1.0).abs() < 0.1);
    }

    #[test]
    fn parse_only_gguf_quant_types_fail_explicitly_in_dequant_path() {
        for dtype in [
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q8_1,
            GgmlType::Q2_K,
            GgmlType::Q8_K,
        ] {
            let err = dequant_to_bf16(&[], dtype, 0).unwrap_err().to_string();
            assert!(
                err.contains("Dequant not yet implemented"),
                "unexpected error for {dtype:?}: {err}"
            );
        }
    }

    #[test]
    fn test_decode_scale_min_k4_matches_ggml_layout() {
        let mut scales = [0u8; 12];
        scales[0] = 0b1000_0000;
        scales[1] = 0b0010_1010;
        scales[4] = 0b1100_0000;
        scales[5] = 0b0001_0001;
        scales[8] = 0x75;

        assert_eq!(decode_scale_min_k4(&scales, 1), (42, 17));
        assert_eq!(decode_scale_min_k4(&scales, 4), (37, 55));
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
