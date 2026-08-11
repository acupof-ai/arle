//! GGUF v2/v3 reader over memmap.
//!
//! Format pinned against llama.cpp's `gguf.h` (lines 1-46). The header is
//! magic "GGUF" (4 bytes), version u32, tensor count i64, kv count i64.
//! Strings are u64 length + bytes; enums i32; bool i8. Arrays are elem
//! type i32 + count u64 + payload. Tensor info is name + n_dims u32 +
//! dims i64 each + ggml type i32 + data offset u64. The data blob is
//! aligned to `general.alignment` (default 32, gguf.h:44-46). v2 and v3
//! share this layout (v3 only adds big-endian variants, which we reject
//! by magic); only the pre-v2 v1 used u32 lengths and is unsupported.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};

pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// GGML tensor dtypes, ids pinned against `enum ggml_type`
/// (llama.cpp ggml.h:389-433).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    I8,
    I16,
    I32,
    I64,
    F64,
    Iq1M,
    Bf16,
    Tq1_0,
    Tq2_0,
    Mxfp4,
    Nvfp4,
    Q1_0,
}

impl GgmlType {
    pub fn from_id(id: u32) -> Result<Self> {
        Ok(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            19 => Self::Iq1S,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::Iq1M,
            30 => Self::Bf16,
            34 => Self::Tq1_0,
            35 => Self::Tq2_0,
            39 => Self::Mxfp4,
            40 => Self::Nvfp4,
            41 => Self::Q1_0,
            other => bail!("unknown ggml type id {other}"),
        })
    }

    /// Elements per block (ggml-common.h block defs; QK_K=256 at line 89).
    pub fn block_size(self) -> usize {
        match self {
            Self::F32
            | Self::F16
            | Self::Bf16
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::F64 => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::Iq4Nl
            | Self::Mxfp4 => 32,
            Self::Nvfp4 => 64,
            _ => 256,
        }
    }

    /// Bytes per block, pinned against vendor/llama.cpp/ggml-common.h
    /// static_asserts (q8_0 l.246, q2_K l.299, q4_K l.328, q5_K l.346,
    /// q6_K l.358, iq2_xxs l.375, q3_K l.311, q8_1 l.259, q8_K l.366).
    /// `None` = layout not pinned here; tensor_data refuses to slice it.
    pub fn type_size(self) -> Option<usize> {
        Some(match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::Bf16 | Self::I16 => 2,
            Self::I8 => 1,
            Self::I64 | Self::F64 => 8,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::Iq2Xxs => 66,
            _ => return None,
        })
    }

    /// Bytes of one row of `ncols` elements; `None` if the layout is
    /// unpinned or `ncols` is not block-aligned.
    pub fn row_bytes(self, ncols: usize) -> Option<usize> {
        let ts = self.type_size()?;
        let bs = self.block_size();
        if ncols == 0 || !ncols.is_multiple_of(bs) {
            return None;
        }
        Some(ncols / bs * ts)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(u64::from(v)),
            Self::U16(v) => Some(u64::from(v)),
            Self::U32(v) => Some(u64::from(v)),
            Self::U64(v) => Some(v),
            Self::I8(v) if v >= 0 => Some(v as u64),
            Self::I16(v) if v >= 0 => Some(v as u64),
            Self::I32(v) if v >= 0 => Some(v as u64),
            Self::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            Self::F64(v) => Some(v as f32),
            _ => self.as_u64().map(|v| v as f32),
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ne[0..n_dims] in GGUF order: ne[0] is the contiguous (column) dim.
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Offset into the data blob (already alignment-padded per spec).
    pub offset: u64,
}

impl TensorInfo {
    pub fn element_count(&self) -> u64 {
        self.dims.iter().product()
    }

    pub fn byte_len(&self) -> Option<u64> {
        let ncols = usize::try_from(self.dims.first().copied().unwrap_or(1)).ok()?;
        let row = self.ggml_type.row_bytes(ncols)? as u64;
        Some(self.dims.iter().skip(1).product::<u64>() * row)
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| anyhow!("GGUF truncated at byte {} (+{n})", self.pos))?;
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let len = usize::try_from(self.u64()?).map_err(|_| anyhow!("GGUF string len overflow"))?;
        ensure!(len <= self.buf.len(), "GGUF string len {len} exceeds file");
        Ok(String::from_utf8_lossy(self.take(len)?).into_owned())
    }

    fn value(&mut self, type_id: u32, depth: u32) -> Result<GgufValue> {
        Ok(match type_id {
            0 => GgufValue::U8(self.take(1)?[0]),
            1 => GgufValue::I8(self.take(1)?[0] as i8),
            2 => GgufValue::U16(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            3 => GgufValue::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            4 => GgufValue::U32(self.u32()?),
            5 => GgufValue::I32(self.u32()? as i32),
            6 => GgufValue::F32(f32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            7 => GgufValue::Bool(self.take(1)?[0] != 0),
            8 => GgufValue::String(self.string()?),
            9 => {
                ensure!(depth == 0, "GGUF nested arrays unsupported");
                let elem_type = self.u32()?;
                let count = self.u64()?;
                ensure!(
                    count <= self.buf.len() as u64,
                    "GGUF array count {count} exceeds file"
                );
                let out = (0..count)
                    .map(|_| self.value(elem_type, depth + 1))
                    .collect::<Result<Vec<_>>>()?;
                GgufValue::Array(out)
            }
            10 => GgufValue::U64(self.u64()?),
            11 => GgufValue::I64(self.u64()? as i64),
            12 => GgufValue::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            other => bail!("unknown GGUF value type {other}"),
        })
    }
}

#[derive(Debug)]
pub struct GgufFile {
    mmap: memmap2::Mmap,
    pub version: u32,
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
    pub alignment: u64,
    pub data_start: usize,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file =
            std::fs::File::open(path).with_context(|| format!("open GGUF {}", path.display()))?;
        // SAFETY: read-only mmap of an immutable model artifact file;
        // the file outlives the returned `Self` (caller owns both).
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("mmap GGUF {}", path.display()))?;
        Self::from_mmap(mmap)
    }

    fn from_mmap(mmap: memmap2::Mmap) -> Result<Self> {
        let mut r = Reader { buf: &mmap, pos: 0 };
        ensure!(r.take(4)? == b"GGUF", "not a GGUF file (bad magic)");
        let version = r.u32()?;
        ensure!(
            version == 2 || version == 3,
            "unsupported GGUF version {version} (v2/v3 only; v1 used u32 lengths)"
        );
        let tensor_count = r.u64()?;
        let kv_count = r.u64()?;
        ensure!(
            tensor_count <= mmap.len() as u64 && kv_count <= mmap.len() as u64,
            "GGUF counts exceed file size"
        );

        let mut metadata = HashMap::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = r.string()?;
            let type_id = r.u32()?;
            let value = r.value(type_id, 0)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        let mut index = HashMap::with_capacity(tensor_count as usize);
        for i in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()?;
            ensure!(n_dims <= 4, "tensor {name}: {n_dims} dims > GGML_MAX_DIMS");
            let dims = (0..n_dims).map(|_| r.u64()).collect::<Result<Vec<_>>>()?;
            let ggml_type = GgmlType::from_id(r.u32()?)?;
            let offset = r.u64()?;
            index.insert(name.clone(), i as usize);
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(GgufValue::as_u64)
            .filter(|&a| a > 0 && a.is_power_of_two())
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
        let data_start = r.pos.div_ceil(alignment as usize) * alignment as usize;
        ensure!(
            data_start <= mmap.len(),
            "GGUF data blob start beyond file end"
        );

        Ok(Self {
            mmap,
            version,
            metadata,
            tensors,
            index,
            alignment,
            data_start,
        })
    }

    pub fn metadata(&self) -> &HashMap<String, GgufValue> {
        &self.metadata
    }

    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(GgufValue::as_u64)
    }

    pub fn get_usize(&self, key: &str) -> Option<usize> {
        self.get_u64(key).and_then(|v| usize::try_from(v).ok())
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key).and_then(GgufValue::as_f32)
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(GgufValue::as_bool)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(GgufValue::as_str)
    }

    /// Scalar or first-of-array f32 (llama.cpp `get_key_or_arr` shape for
    /// per-layer keys like `swiglu_clamp_exp`).
    pub fn get_f32_scalar_or_first(&self, key: &str) -> Option<f32> {
        match self.get(key)? {
            GgufValue::Array(items) => items.first().and_then(GgufValue::as_f32),
            v => v.as_f32(),
        }
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensor(name)
            .ok_or_else(|| anyhow!("tensor {name} not in GGUF"))?;
        let len = info.byte_len().ok_or_else(|| {
            anyhow!(
                "tensor {name}: unpinned dtype {:?} or unaligned ne0",
                info.ggml_type
            )
        })?;
        let start = self
            .data_start
            .checked_add(usize::try_from(info.offset).map_err(|_| anyhow!("offset overflow"))?)
            .ok_or_else(|| anyhow!("tensor {name}: offset overflow"))?;
        let end = start
            .checked_add(usize::try_from(len).map_err(|_| anyhow!("len overflow"))?)
            .filter(|&e| e <= self.mmap.len())
            .ok_or_else(|| anyhow!("tensor {name}: data out of file bounds"))?;
        Ok(&self.mmap[start..end])
    }
}

/// Synthetic-GGUF writer for unit tests in this crate and its consumers
/// (`infer-hip` / `infer-vulkan` test builds reach it cross-crate, so it
/// cannot be `#[cfg(test)]`).
pub mod test_writer {
    use super::GGUF_DEFAULT_ALIGNMENT;

    pub enum V {
        U32(u32),
        F32(f32),
        Bool(bool),
        Str(&'static str),
        ArrF32(Vec<f32>),
    }

    pub struct T {
        pub name: String,
        pub dims: Vec<u64>,
        pub type_id: u32,
        pub data: Vec<u8>,
    }

    fn put_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    pub fn write(kvs: &[(&str, V)], tensors: &[T]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (key, value) in kvs {
            put_str(&mut buf, key);
            match value {
                V::U32(v) => {
                    buf.extend_from_slice(&4u32.to_le_bytes());
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                V::F32(v) => {
                    buf.extend_from_slice(&6u32.to_le_bytes());
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                V::Bool(v) => {
                    buf.extend_from_slice(&7u32.to_le_bytes());
                    buf.push(u8::from(*v));
                }
                V::Str(v) => {
                    buf.extend_from_slice(&8u32.to_le_bytes());
                    put_str(&mut buf, v);
                }
                V::ArrF32(items) => {
                    buf.extend_from_slice(&9u32.to_le_bytes());
                    buf.extend_from_slice(&6u32.to_le_bytes());
                    buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for item in items {
                        buf.extend_from_slice(&item.to_le_bytes());
                    }
                }
            }
        }
        let align = GGUF_DEFAULT_ALIGNMENT as usize;
        let mut offset = 0usize;
        for t in tensors {
            put_str(&mut buf, &t.name);
            buf.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&t.type_id.to_le_bytes());
            buf.extend_from_slice(&(offset as u64).to_le_bytes());
            offset += t.data.len().div_ceil(align) * align;
        }
        while !buf.len().is_multiple_of(align) {
            buf.push(0);
        }
        for t in tensors {
            buf.extend_from_slice(&t.data);
            while !buf.len().is_multiple_of(align) {
                buf.push(0);
            }
        }
        buf
    }

    pub fn write_to_temp(kvs: &[(&str, V)], tensors: &[T], tag: &str) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("arle-infer-hip-{tag}-{}.gguf", std::process::id()));
        std::fs::write(&path, write(kvs, tensors)).unwrap();
        path
    }
}

#[cfg(test)]
mod tests {
    use super::test_writer::{T, V, write_to_temp};
    use super::*;

    #[test]
    fn synthetic_roundtrip_metadata_and_tensors() {
        let f32_data = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let q8_block = vec![0u8; 34];
        let path = write_to_temp(
            &[
                ("general.architecture", V::Str("deepseek4")),
                ("deepseek4.block_count", V::U32(2)),
                ("deepseek4.rope.freq_base", V::F32(10000.0)),
                ("deepseek4.expert_weights_norm", V::Bool(true)),
                ("deepseek4.swiglu_clamp_exp", V::ArrF32(vec![10.0, 12.0])),
            ],
            &[
                T {
                    name: "a".into(),
                    dims: vec![8],
                    type_id: 0,
                    data: f32_data,
                },
                T {
                    name: "b".into(),
                    dims: vec![32],
                    type_id: 8,
                    data: q8_block,
                },
            ],
            "roundtrip",
        );
        let g = GgufFile::open(&path).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.alignment, 32);
        assert_eq!(g.get_str("general.architecture"), Some("deepseek4"));
        assert_eq!(g.get_usize("deepseek4.block_count"), Some(2));
        assert_eq!(g.get_f32("deepseek4.rope.freq_base"), Some(10000.0));
        assert_eq!(g.get_bool("deepseek4.expert_weights_norm"), Some(true));
        assert_eq!(
            g.get_f32_scalar_or_first("deepseek4.swiglu_clamp_exp"),
            Some(10.0)
        );

        let a = g.tensor("a").unwrap();
        assert_eq!(a.dims, vec![8]);
        assert_eq!(a.ggml_type, GgmlType::F32);
        let data = g.tensor_data("a").unwrap();
        assert_eq!(data.len(), 32);
        assert_eq!(f32::from_le_bytes(data[4..8].try_into().unwrap()), 1.0);

        let b = g.tensor("b").unwrap();
        assert_eq!(b.ggml_type, GgmlType::Q8_0);
        assert_eq!(g.tensor_data("b").unwrap().len(), 34);
        assert!(g.tensor("missing").is_none());
        assert!(g.tensor_data("missing").is_err());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn rejects_bad_magic_and_version() {
        let dir = std::env::temp_dir();
        let bad_magic = dir.join(format!("arle-hip-badmagic-{}.gguf", std::process::id()));
        std::fs::write(&bad_magic, b"GGLAxxxxxxxxxxxxxxxxxxxx").unwrap();
        assert!(GgufFile::open(&bad_magic).is_err());
        std::fs::remove_file(bad_magic).ok();

        let v1 = dir.join(format!("arle-hip-v1-{}.gguf", std::process::id()));
        let mut buf = b"GGUF".to_vec();
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 16]);
        std::fs::write(&v1, buf).unwrap();
        assert!(
            GgufFile::open(&v1)
                .unwrap_err()
                .to_string()
                .contains("version 1")
        );
        std::fs::remove_file(v1).ok();
    }

    #[test]
    fn row_bytes_reject_unaligned() {
        assert_eq!(GgmlType::Q4K.row_bytes(256), Some(144));
        assert_eq!(GgmlType::Q4K.row_bytes(255), None);
        assert_eq!(GgmlType::Iq2Xxs.row_bytes(512), Some(132));
        assert_eq!(GgmlType::F32.row_bytes(3), Some(12));
        assert_eq!(GgmlType::Iq4Nl.row_bytes(32), None);
    }

    #[test]
    fn parses_real_qwen35_gguf_if_present() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("skip: {} not present", path.display());
            return;
        }
        let g = GgufFile::open(&path).unwrap();
        assert!(g.version == 2 || g.version == 3);
        assert!(g.get_str("general.architecture").is_some());
        assert!(!g.tensors().is_empty());
        let embd = g
            .tensors()
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .expect("token_embd.weight present");
        assert_eq!(embd.dims.len(), 2);
        let data = g.tensor_data("token_embd.weight").unwrap();
        assert_eq!(data.len() as u64, embd.byte_len().unwrap());
    }
}
