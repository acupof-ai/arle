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
use std::path::{Path, PathBuf};

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
            // block_mxfp4 { uint8_t e /* E8M0 */; uint8_t qs[QK_MXFP4/2] }
            // with QK_MXFP4 = 32 (ggml-common.h l.205-209 static_assert).
            // Unsloth's "UD-Q*_XL" dynamic quants put the routed experts in
            // MXFP4, so this is 90% of a 122B-A10B checkpoint's elements, not a
            // corner case.
            Self::Mxfp4 => 17,
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
/// One physical `.gguf` file's weight blob. A single-file checkpoint has
/// exactly one of these; a split checkpoint has `split.count` of them.
struct Shard {
    mmap: memmap2::Mmap,
    /// Where this shard's tensor data begins, after its own header + padding.
    /// Split shards each carry a full GGUF header, so this is per-shard and
    /// NOT a property of the logical model.
    data_start: usize,
}

/// Sibling path for part `no` (0-based) of a `count`-way split.
///
/// llama.cpp's naming is `<base>-<NNNNN>-of-<NNNNN>.gguf`, 1-based and
/// zero-padded to five digits. Rewriting only the matched suffix keeps any
/// other hyphens in the base name (`...-UD-Q4_K_XL-00001-of-00003.gguf`)
/// intact. Returns `None` when the name does not carry the suffix, which the
/// caller reports rather than guessing at a filename.
fn split_part_path(first: &Path, no: u64, count: u64) -> Option<PathBuf> {
    let name = first.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf")?;
    // Expect the LAST 17 chars to be "-NNNNN-of-NNNNN".
    let (base, suffix) = stem.split_at(stem.len().checked_sub(15)?);
    let (lhs, rhs) = suffix.split_once("-of-")?;
    let lhs = lhs.strip_prefix('-')?;
    if lhs.len() != 5
        || rhs.len() != 5
        || !lhs.bytes().chain(rhs.bytes()).all(|b| b.is_ascii_digit())
    {
        return None;
    }
    Some(first.with_file_name(format!("{base}-{:05}-of-{:05}.gguf", no + 1, count)))
}

pub struct GgufFile {
    /// Shard 0 is always the file the caller named. For a split checkpoint the
    /// rest follow in `split.no` order.
    shards: Vec<Shard>,
    pub version: u32,
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<TensorInfo>,
    /// Which shard each entry of `tensors` lives in, parallel to `tensors`.
    /// Kept beside `TensorInfo` rather than inside it: the offset in
    /// `TensorInfo` is what the file declares, and callers that reason about
    /// GGUF layout should not have to know the model was split.
    tensor_shard: Vec<usize>,
    index: HashMap<String, usize>,
    pub alignment: u64,
    /// Shard 0's data start, retained as public API. For a split checkpoint
    /// shard 0 typically holds NO tensors (llama.cpp writes all metadata to
    /// part 1 and the weights to parts 2..N), so this is a header offset, not
    /// a useful base for tensor addressing — use [`GgufFile::tensor_data`].
    pub data_start: usize,
}

impl GgufFile {
    /// Open a checkpoint. If `path` is part of a SPLIT GGUF, the sibling parts
    /// are opened too and the result behaves as one logical model.
    ///
    /// llama.cpp's `gguf-split` writes `<base>-00001-of-0000N.gguf`, putting
    /// every KV pair in part 1 and every tensor in parts 2..N — part 1 has
    /// `tensor_count == 0`. Reading only the named file therefore yields a
    /// model with full metadata and no weights, which surfaces far downstream
    /// as a confusing "GGUF has neither <arch>.vocab_size nor
    /// token_embd.weight/output.weight dims". Follow the split here instead.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut model = Self::open_one(path)?;

        let shard_count = model
            .get("split.count")
            .and_then(GgufValue::as_u64)
            .unwrap_or(0);
        if shard_count <= 1 {
            return Ok(model);
        }
        let named_no = model
            .get("split.no")
            .and_then(GgufValue::as_u64)
            .unwrap_or(0);
        ensure!(
            named_no == 0,
            "open the FIRST part of a split GGUF ({}): this is part {} of {}, and only              part 1 carries the model metadata",
            path.display(),
            named_no + 1,
            shard_count
        );

        let expected_tensors = model
            .get("split.tensors.count")
            .and_then(GgufValue::as_u64)
            .unwrap_or(0);
        for no in 1..shard_count {
            let sibling = split_part_path(path, no, shard_count).with_context(|| {
                format!(
                    "{} declares split.count={shard_count} but its name does not match the                      -00001-of-{shard_count:05} convention, so the sibling parts cannot be located",
                    path.display()
                )
            })?;
            let part = Self::open_one(&sibling)
                .with_context(|| format!("open split part {} of {shard_count}", no + 1))?;
            model.absorb_shard(part, &sibling)?;
        }
        if expected_tensors > 0 {
            let got = model.tensors.len() as u64;
            ensure!(
                got == expected_tensors,
                "split GGUF {}: assembled {got} tensors but split.tensors.count says                  {expected_tensors}",
                path.display()
            );
        }
        Ok(model)
    }

    /// Open exactly one `.gguf` file, following no split.
    fn open_one(path: &Path) -> Result<Self> {
        let file =
            std::fs::File::open(path).with_context(|| format!("open GGUF {}", path.display()))?;
        // SAFETY: read-only mmap of an immutable model artifact file;
        // the file outlives the returned `Self` (caller owns both).
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("mmap GGUF {}", path.display()))?;
        Self::from_mmap(mmap)
    }

    /// Fold `part`'s tensors (and its mmap) into `self` as a new shard.
    ///
    /// Metadata is NOT merged: the trailing parts carry only the three
    /// `split.*` keys, and part 1 is authoritative for everything else.
    fn absorb_shard(&mut self, part: Self, path: &Path) -> Result<()> {
        let shard = self.shards.len();
        let Self {
            shards: part_shards,
            tensors: part_tensors,
            data_start: part_data_start,
            ..
        } = part;
        self.shards.push(Shard {
            mmap: part_shards
                .into_iter()
                .next()
                .expect("a parsed GgufFile always has shard 0")
                .mmap,
            data_start: part_data_start,
        });
        for tensor in part_tensors {
            let name = tensor.name.clone();
            if let Some(&prev) = self.index.get(&name) {
                bail!(
                    "split GGUF: tensor {name} appears in shard {} and again in {}",
                    self.tensor_shard[prev],
                    path.display()
                );
            }
            self.index.insert(name, self.tensors.len());
            self.tensors.push(tensor);
            self.tensor_shard.push(shard);
        }
        Ok(())
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
            shards: vec![Shard { mmap, data_start }],
            version,
            metadata,
            tensor_shard: vec![0; tensors.len()],
            tensors,
            index,
            alignment,
            data_start,
        })
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
        let idx = *self
            .index
            .get(name)
            .ok_or_else(|| anyhow!("tensor {name} not in GGUF"))?;
        let info = &self.tensors[idx];
        let len = info.byte_len().ok_or_else(|| {
            anyhow!(
                "tensor {name}: unpinned dtype {:?} or unaligned ne0",
                info.ggml_type
            )
        })?;
        // `info.offset` is relative to the data blob of the shard the tensor
        // was declared in, not to the model as a whole.
        let shard = &self.shards[self.tensor_shard[idx]];
        let start = shard
            .data_start
            .checked_add(usize::try_from(info.offset).map_err(|_| anyhow!("offset overflow"))?)
            .ok_or_else(|| anyhow!("tensor {name}: offset overflow"))?;
        let end = start
            .checked_add(usize::try_from(len).map_err(|_| anyhow!("len overflow"))?)
            .filter(|&e| e <= shard.mmap.len())
            .ok_or_else(|| anyhow!("tensor {name}: data out of file bounds"))?;
        Ok(&shard.mmap[start..end])
    }
}

#[cfg(test)]
mod split_tests {
    use super::split_part_path;
    use std::path::{Path, PathBuf};

    /// The real on-box name: the base carries its own hyphens (`-UD-Q4_K_XL`),
    /// so only the trailing `-NNNNN-of-NNNNN` may be rewritten.
    #[test]
    fn rewrites_only_the_split_suffix() {
        let first = Path::new(r"C:\models\Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf");
        assert_eq!(
            split_part_path(first, 1, 3),
            Some(PathBuf::from(
                r"C:\models\Qwen3.5-122B-A10B-UD-Q4_K_XL-00002-of-00003.gguf"
            ))
        );
        assert_eq!(
            split_part_path(first, 2, 3),
            Some(PathBuf::from(
                r"C:\models\Qwen3.5-122B-A10B-UD-Q4_K_XL-00003-of-00003.gguf"
            ))
        );
    }

    /// A single-file checkpoint must not be mistaken for part 1 of a split —
    /// returning `None` makes the caller report the mismatch instead of
    /// silently probing for a file that was never written.
    #[test]
    fn rejects_names_without_the_suffix() {
        for name in [
            "model.gguf",
            "Qwen3.8-27B-Q4_K_M.gguf",
            "weird-1-of-3.gguf",         // not zero-padded to five
            "weird-00001-of-3.gguf",     // right-hand side too short
            "weird-0000X-of-00003.gguf", // non-digit
        ] {
            assert_eq!(
                split_part_path(Path::new(name), 1, 3),
                None,
                "{name} should not parse as a split part"
            );
        }
    }
}
