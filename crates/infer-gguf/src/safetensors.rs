//! Multi-shard safetensors reader, shaped like [`crate::gguf::GgufFile`].
//!
//! Format (huggingface/safetensors `README.md`, "Format" section): 8-byte
//! little-endian header length `N`, then `N` bytes of UTF-8 JSON, then the
//! data blob. The JSON is one object whose keys are tensor names mapping to
//! `{"dtype": .., "shape": [..], "data_offsets": [start, end]}`, with the
//! offsets RELATIVE to the first byte of the blob (i.e. to `8 + N`). The
//! reserved key `__metadata__` is a string->string map, not a tensor.
//!
//! # Dimension order
//!
//! safetensors `shape` is row-major `[outer, .., inner]`, so the LAST entry is
//! the contiguous one. GGUF's `ne` is the opposite: `ne[0]` is contiguous.
//! This reader normalises to the **GGUF convention** — [`SafeTensorInfo::dims`]
//! is the header `shape` REVERSED — so that a Vulkan uploader written against
//! `GgufFile` can consume either source without a per-format special case. For
//! `lm_head.weight` that means `dims == [hidden, vocab]` while the header says
//! `shape == [vocab, hidden]`; `real_checkpoint_pins_lm_head_dimension_order`
//! nails this against the on-box checkpoint.
//!
//! # No `serde_json`, no `safetensors` crate
//!
//! `infer-gguf` depends on neither, and its `Cargo.toml` was outside the scope
//! this module was written under, so the header is parsed here. The scanner
//! below is a full JSON reader for the shapes the format allows, not a
//! substring hack — safetensors headers are machine-written by `serde_json`
//! (or Python `json`), so escapes and key order are the writer's choice, not
//! ours.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::gguf::GgmlType;

/// A tensor as declared by one shard's header.
#[derive(Debug, Clone)]
pub struct SafeTensorInfo {
    pub name: String,
    /// GGUF order: `dims[0]` is the contiguous (innermost) dim. This is the
    /// header `shape` reversed — see the module docs.
    pub dims: Vec<u64>,
    /// The dtype string exactly as the header spells it (`"BF16"`, `"U8"`,
    /// `"F8_E4M3"`, ...). Kept verbatim so callers can act on dtypes that have
    /// no `GgmlType` twin instead of receiving a guess.
    pub dtype: String,
    /// `Some` only for dtypes with an exact 1:1 `GgmlType`. NVFP4 expert
    /// planes arrive as `U8` and land here as `None`.
    pub ggml_type: Option<GgmlType>,
    /// Offset into this tensor's shard's data blob, i.e. relative to `8 + N`.
    pub offset: u64,
    /// Byte length as declared by `data_offsets`, not recomputed from `shape`.
    pub len: u64,
}

impl SafeTensorInfo {
    /// Product of `dims`; `1` for a rank-0 scalar, matching `shape: []`.
    pub fn element_count(&self) -> u64 {
        self.dims.iter().product()
    }
}

/// Map a safetensors dtype string to its `GgmlType`.
///
/// Only exact, same-width, same-signedness twins are listed. Everything else —
/// `U8`/`U16`/`U32`/`U64`, `BOOL`, `F8_E5M2`, the sub-byte `F4`/`E*M*` families
/// — returns `None` and keeps its verbatim string. In particular `U8` is the
/// carrier for this checkpoint's NVFP4 expert planes: the byte plane and its
/// separate `weight_scale` tensor only become a `GgmlType::Nvfp4` block after
/// something re-interleaves them, which is not this reader's job.
fn ggml_type_for(dtype: &str) -> Option<GgmlType> {
    Some(match dtype {
        "BF16" => GgmlType::Bf16,
        "F16" => GgmlType::F16,
        "F32" => GgmlType::F32,
        "F64" => GgmlType::F64,
        "F8_E4M3" => GgmlType::F8E4M3,
        "I8" => GgmlType::I8,
        "I16" => GgmlType::I16,
        "I32" => GgmlType::I32,
        "I64" => GgmlType::I64,
        _ => return None,
    })
}

/// Bytes per declared element, for the dtypes safetensors stores a whole number
/// of bytes at a time. `None` means there is no shape-to-length invariant to
/// check: a sub-byte dtype (`F4`, `F6_E2M3`, `E8M0`, ...), where `shape` and
/// byte length are not proportional, or a spelling this reader has never seen.
///
/// Mapped dtypes take their width from `GgmlType` so the two tables cannot
/// drift; the literals below are only the dtypes with no `GgmlType` twin.
///
/// `U8` is the entry that matters. It carries the whole NVFP4 expert tier of the
/// `qwen4_exp` checkpoint — 56.25 GiB of its 126, measured over all 206 shard
/// headers — and it has the strictest invariant of any dtype in the file:
/// exactly one byte per declared element. The 2-values-per-byte packing lives in
/// the SHAPE (`[640, 1280]` for a `[640, 2560]` matrix), not in a shape-to-bytes
/// ratio, so the cross-check below applies to it unchanged.
fn bytes_per_element(dtype: &str, ggml_type: Option<GgmlType>) -> Option<u64> {
    if let Some(ty) = ggml_type
        && ty.block_size() == 1
        && let Some(elem) = ty.type_size()
    {
        return Some(elem as u64);
    }
    Some(match dtype {
        "U8" | "BOOL" | "F8_E5M2" => 1,
        "U16" => 2,
        "U32" => 4,
        "U64" => 8,
        _ => return None,
    })
}

/// Cap on `{`/`[` nesting while skipping an unknown value. A conforming header
/// nests three deep (`{tensor: {shape: [..]}}`); the cap only exists so a
/// malformed file cannot recurse the parser off the stack.
const MAX_JSON_DEPTH: u32 = 64;

struct Json<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Json<'_> {
    fn skip_ws(&mut self) {
        while let Some(&b) = self.buf.get(self.pos) {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// Next non-whitespace byte without consuming it.
    fn peek(&mut self) -> Result<u8> {
        self.skip_ws();
        self.buf
            .get(self.pos)
            .copied()
            .ok_or_else(|| anyhow!("safetensors header ends mid-value at byte {}", self.pos))
    }

    fn bump(&mut self) -> Result<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Ok(b)
    }

    fn expect(&mut self, want: u8) -> Result<()> {
        let got = self.bump()?;
        ensure!(
            got == want,
            "safetensors header: expected '{}' at byte {}, found '{}'",
            want as char,
            self.pos - 1,
            got.escape_ascii()
        );
        Ok(())
    }

    /// One `\uXXXX` escape, `\u` already consumed. Combines a surrogate pair
    /// with the `\uXXXX` that must follow it (RFC 8259 s7).
    fn unicode_escape(&mut self) -> Result<char> {
        let unit = self.hex4()?;
        let scalar = match unit {
            0xD800..=0xDBFF => {
                ensure!(
                    self.buf.get(self.pos..self.pos + 2) == Some(b"\\u".as_slice()),
                    "safetensors header: lone high surrogate at byte {}",
                    self.pos
                );
                self.pos += 2;
                let low = self.hex4()?;
                ensure!(
                    (0xDC00..=0xDFFF).contains(&low),
                    "safetensors header: high surrogate not followed by a low one at byte {}",
                    self.pos
                );
                0x1_0000 + ((u32::from(unit) - 0xD800) << 10) + (u32::from(low) - 0xDC00)
            }
            _ => u32::from(unit),
        };
        char::from_u32(scalar)
            .ok_or_else(|| anyhow!("safetensors header: \\u{scalar:04X} is not a scalar value"))
    }

    fn hex4(&mut self) -> Result<u16> {
        let digits = self
            .buf
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| anyhow!("safetensors header truncated in \\u escape"))?;
        let text = std::str::from_utf8(digits)
            .map_err(|_| anyhow!("safetensors header: non-ASCII \\u escape"))?;
        self.pos += 4;
        u16::from_str_radix(text, 16)
            .with_context(|| format!("safetensors header: bad \\u escape \\u{text}"))
    }

    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut out: Vec<u8> = Vec::new();
        loop {
            // Copy the whole run up to the next '"' or '\' at once: tensor
            // names are plain ASCII in every writer we have seen, so this is
            // the path that actually runs for ~300k names.
            let start = self.pos;
            while self
                .buf
                .get(self.pos)
                .is_some_and(|b| !matches!(b, b'"' | b'\\'))
            {
                self.pos += 1;
            }
            out.extend_from_slice(&self.buf[start..self.pos]);
            let b = *self.buf.get(self.pos).ok_or_else(|| {
                anyhow!("safetensors header: unterminated string at byte {start}")
            })?;
            self.pos += 1;
            if b == b'"' {
                return String::from_utf8(out).map_err(|_| {
                    anyhow!("safetensors header: string at byte {start} is not UTF-8")
                });
            }
            let esc = *self
                .buf
                .get(self.pos)
                .ok_or_else(|| anyhow!("safetensors header truncated after '\\'"))?;
            self.pos += 1;
            match esc {
                b'"' => out.push(b'"'),
                b'\\' => out.push(b'\\'),
                b'/' => out.push(b'/'),
                b'b' => out.push(0x08),
                b'f' => out.push(0x0C),
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b't' => out.push(b'\t'),
                b'u' => {
                    let mut utf8 = [0u8; 4];
                    out.extend_from_slice(self.unicode_escape()?.encode_utf8(&mut utf8).as_bytes());
                }
                other => bail!(
                    "safetensors header: unknown escape '\\{}' at byte {}",
                    other.escape_ascii(),
                    self.pos - 1
                ),
            }
        }
    }

    /// A non-negative JSON integer. Shapes and offsets are the only numbers a
    /// header may carry, and both are counts; `1.0` or `1e9` is rejected
    /// rather than silently truncated to something that still slices.
    fn u64(&mut self) -> Result<u64> {
        self.skip_ws();
        let start = self.pos;
        while self.buf.get(self.pos).is_some_and(u8::is_ascii_digit) {
            self.pos += 1;
        }
        ensure!(
            self.pos > start,
            "safetensors header: expected a non-negative integer at byte {start}"
        );
        ensure!(
            !self
                .buf
                .get(self.pos)
                .is_some_and(|b| matches!(b, b'.' | b'e' | b'E')),
            "safetensors header: non-integer number at byte {start}"
        );
        let text = std::str::from_utf8(&self.buf[start..self.pos]).expect("ASCII digits");
        text.parse()
            .with_context(|| format!("safetensors header: integer {text} overflows u64"))
    }

    fn u64_array(&mut self) -> Result<Vec<u64>> {
        self.expect(b'[')?;
        let mut out = Vec::new();
        if self.peek()? == b']' {
            self.pos += 1;
            return Ok(out);
        }
        loop {
            out.push(self.u64()?);
            match self.bump()? {
                b',' => {}
                b']' => return Ok(out),
                other => bail!(
                    "safetensors header: expected ',' or ']' at byte {}, found '{}'",
                    self.pos - 1,
                    other.escape_ascii()
                ),
            }
        }
    }

    /// Consume one value of any kind, discarding it.
    fn skip_value(&mut self, depth: u32) -> Result<()> {
        ensure!(
            depth < MAX_JSON_DEPTH,
            "safetensors header nests deeper than {MAX_JSON_DEPTH} levels"
        );
        match self.peek()? {
            b'"' => {
                self.string()?;
            }
            b'{' | b'[' => {
                let open = self.bump()?;
                let close = if open == b'{' { b'}' } else { b']' };
                if self.peek()? == close {
                    self.pos += 1;
                    return Ok(());
                }
                loop {
                    if open == b'{' {
                        self.string()?;
                        self.expect(b':')?;
                    }
                    self.skip_value(depth + 1)?;
                    let sep = self.bump()?;
                    if sep == close {
                        return Ok(());
                    }
                    ensure!(
                        sep == b',',
                        "safetensors header: expected ',' or '{}' at byte {}, found '{}'",
                        close as char,
                        self.pos - 1,
                        sep.escape_ascii()
                    );
                }
            }
            _ => {
                // A number, `true`, `false` or `null`: run to the next
                // structural byte. Nothing downstream reads these.
                let start = self.pos;
                while self.buf.get(self.pos).is_some_and(|b| {
                    !matches!(b, b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r')
                }) {
                    self.pos += 1;
                }
                ensure!(
                    self.pos > start,
                    "safetensors header: empty value at byte {start}"
                );
            }
        }
        Ok(())
    }
}

/// Parse one shard's JSON header.
///
/// `blob_len` is the number of bytes after the header, used to reject a
/// `data_offsets` pair that would slice past the file. Returns the tensors in
/// header order plus the `__metadata__` map (empty when the key is absent).
fn parse_header(
    json: &[u8],
    blob_len: u64,
) -> Result<(Vec<SafeTensorInfo>, HashMap<String, String>)> {
    let mut j = Json { buf: json, pos: 0 };
    let mut tensors = Vec::new();
    let mut metadata = HashMap::new();

    j.expect(b'{')?;
    if j.peek()? == b'}' {
        return Ok((tensors, metadata));
    }
    loop {
        let name = j.string()?;
        j.expect(b':')?;
        if name == "__metadata__" {
            metadata = parse_metadata(&mut j)?;
        } else {
            tensors.push(parse_tensor(&mut j, name, blob_len)?);
        }
        match j.bump()? {
            b',' => {}
            b'}' => break,
            other => bail!(
                "safetensors header: expected ',' or '}}' at byte {}, found '{}'",
                j.pos - 1,
                other.escape_ascii()
            ),
        }
    }
    Ok((tensors, metadata))
}

fn parse_metadata(j: &mut Json<'_>) -> Result<HashMap<String, String>> {
    let mut out = HashMap::new();
    j.expect(b'{')?;
    if j.peek()? == b'}' {
        j.pos += 1;
        return Ok(out);
    }
    loop {
        let key = j.string()?;
        j.expect(b':')?;
        // The spec types this map as string->string, but a non-string value
        // here is no reason to refuse a 126 GiB checkpoint: drop it.
        if j.peek()? == b'"' {
            let value = j.string()?;
            out.insert(key, value);
        } else {
            j.skip_value(0)?;
        }
        match j.bump()? {
            b',' => {}
            b'}' => return Ok(out),
            other => bail!(
                "safetensors __metadata__: expected ',' or '}}' at byte {}, found '{}'",
                j.pos - 1,
                other.escape_ascii()
            ),
        }
    }
}

fn parse_tensor(j: &mut Json<'_>, name: String, blob_len: u64) -> Result<SafeTensorInfo> {
    let mut dtype: Option<String> = None;
    let mut shape: Option<Vec<u64>> = None;
    let mut offsets: Option<Vec<u64>> = None;

    j.expect(b'{')
        .with_context(|| format!("tensor {name}: entry is not an object"))?;
    if j.peek()? != b'}' {
        loop {
            let key = j.string()?;
            j.expect(b':')?;
            match key.as_str() {
                "dtype" => dtype = Some(j.string()?),
                "shape" => shape = Some(j.u64_array()?),
                "data_offsets" => offsets = Some(j.u64_array()?),
                // Forward compatibility: an unknown key is skipped, not fatal.
                _ => j.skip_value(0)?,
            }
            match j.bump()? {
                b',' => {}
                b'}' => break,
                other => bail!(
                    "tensor {name}: expected ',' or '}}' at byte {}, found '{}'",
                    j.pos - 1,
                    other.escape_ascii()
                ),
            }
        }
    } else {
        j.pos += 1;
    }

    let dtype = dtype.ok_or_else(|| anyhow!("tensor {name}: header entry has no \"dtype\""))?;
    let shape = shape.ok_or_else(|| anyhow!("tensor {name}: header entry has no \"shape\""))?;
    let offsets =
        offsets.ok_or_else(|| anyhow!("tensor {name}: header entry has no \"data_offsets\""))?;
    ensure!(
        offsets.len() == 2,
        "tensor {name}: data_offsets has {} entries, want 2",
        offsets.len()
    );
    let (start, end) = (offsets[0], offsets[1]);
    ensure!(
        start <= end && end <= blob_len,
        "tensor {name}: data_offsets [{start}, {end}] outside a {blob_len}-byte blob"
    );
    let len = end - start;

    let ggml_type = ggml_type_for(&dtype);
    // Cross-check the declared length against shape x element size: the cheap
    // guard that catches a mis-declared header at open time instead of letting a
    // short slice reach a shader. It pins the ELEMENT COUNT, so it catches a
    // plane declared at its logical width instead of its packed one, a dropped
    // or invented axis, and a truncated span — but not a pure transpose, whose
    // product is unchanged. Axis ORDER is pinned by the callers' geometry
    // asserts, not here.
    if let Some(elem) = bytes_per_element(&dtype, ggml_type) {
        let elems = shape.iter().try_fold(1u64, |acc, &d| acc.checked_mul(d));
        let want = elems.and_then(|n| n.checked_mul(elem));
        ensure!(
            want == Some(len),
            "tensor {name}: dtype {dtype} shape {shape:?} implies {want:?} bytes but \
             data_offsets declare {len}"
        );
    }

    // See the module docs: safetensors is row-major (contiguous dim LAST),
    // GGUF's ne is contiguous-FIRST.
    let dims = shape.into_iter().rev().collect();
    Ok(SafeTensorInfo {
        name,
        dims,
        dtype,
        ggml_type,
        offset: start,
        len,
    })
}

/// One physical `.safetensors` file.
struct Shard {
    mmap: memmap2::Mmap,
    /// `8 + header_len`; `data_offsets` are relative to this, per shard.
    blob_start: usize,
    path: PathBuf,
    metadata: HashMap<String, String>,
}

/// A directory of `.safetensors` files read as one logical checkpoint.
///
/// Mirrors [`crate::gguf::GgufFile`]: N mmaps, a name -> (shard, offset, len)
/// index, and [`Self::tensor_data`] handing back a borrowed slice. Nothing is
/// copied — the on-box `qwen4_exp` checkpoint's PLE table alone is 47.68 GiB,
/// so a `to_vec()` anywhere on this path is fatal, not merely wasteful.
pub struct SafeTensorsDir {
    shards: Vec<Shard>,
    tensors: Vec<SafeTensorInfo>,
    /// Which shard each entry of `tensors` lives in, parallel to `tensors`.
    /// Kept beside `SafeTensorInfo` rather than inside it, matching how
    /// `GgufFile` keeps `tensor_shard` out of `TensorInfo`: the offset a
    /// tensor declares is a fact about its file, and callers reasoning about
    /// layout should not have to know the checkpoint was split.
    tensor_shard: Vec<usize>,
    index: HashMap<String, usize>,
    /// The checkpoint directory, when opened through [`Self::open_dir`]
    /// (`None` for an explicit file list). Consumers that need an OWNING
    /// second view of the same checkpoint — the n-gram gather pool holds an
    /// `Arc` where the model only borrows — reopen from here; the page cache
    /// makes the second mmap free.
    root: Option<PathBuf>,
}

impl SafeTensorsDir {
    /// Open every `*.safetensors` in `dir` as one checkpoint.
    ///
    /// No index file is consulted, and none is required. The on-box
    /// `qwen4_exp` checkpoint does ship a complete `model.safetensors.index.json`
    /// (measured: 296,475 entries over all 206 shards), but every shard header
    /// already carries the same `dtype`/`shape`/`data_offsets` the index would
    /// repeat, so reading the directory keeps the one authority — the file the
    /// bytes live in — and works on the sibling-shard layouts that ship without
    /// an index at all.
    pub fn open_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("read dir {}", dir.display()))?
            .map(|entry| Ok(entry?.path()))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        // read_dir order is filesystem-defined; sort so shard indices, and any
        // error message quoting one, are reproducible.
        paths.sort();
        ensure!(
            !paths.is_empty(),
            "no *.safetensors in {} — is this a checkpoint directory?",
            dir.display()
        );
        let mut opened = Self::open_files(&paths)?;
        opened.root = Some(dir.to_path_buf());
        Ok(opened)
    }

    /// The directory this was opened from, if [`Self::open_dir`] was used.
    #[must_use]
    pub fn root(&self) -> Option<&Path> {
        self.root.as_deref()
    }

    /// Open an explicit, ordered list of shards. Shard indices follow `paths`.
    pub fn open_files(paths: &[PathBuf]) -> Result<Self> {
        let mut out = Self {
            shards: Vec::with_capacity(paths.len()),
            tensors: Vec::new(),
            tensor_shard: Vec::new(),
            index: HashMap::new(),
            root: None,
        };
        for path in paths {
            out.absorb_shard(path)
                .with_context(|| format!("open safetensors shard {}", path.display()))?;
        }
        Ok(out)
    }

    fn absorb_shard(&mut self, path: &Path) -> Result<()> {
        let file = std::fs::File::open(path)?;
        // SAFETY: read-only mmap of an immutable model artifact file; the
        // `File` may be dropped here because the mapping keeps the underlying
        // object alive, and the mmap lives as long as the returned `Self`.
        let mmap = unsafe { memmap2::Mmap::map(&file) }?;

        let header_len = u64::from_le_bytes(
            mmap.get(..8)
                .ok_or_else(|| anyhow!("file is {} bytes, too short for a header", mmap.len()))?
                .try_into()
                .expect("8 bytes"),
        );
        let blob_start = usize::try_from(header_len)
            .ok()
            .and_then(|n| n.checked_add(8))
            .filter(|&s| s <= mmap.len())
            .ok_or_else(|| {
                anyhow!(
                    "header length {header_len} does not fit in the {}-byte file",
                    mmap.len()
                )
            })?;
        let blob_len = (mmap.len() - blob_start) as u64;
        let (tensors, metadata) = parse_header(&mmap[8..blob_start], blob_len)?;

        let shard = self.shards.len();
        self.shards.push(Shard {
            mmap,
            blob_start,
            path: path.to_path_buf(),
            metadata,
        });
        self.tensors.reserve(tensors.len());
        for tensor in tensors {
            if let Some(&prev) = self.index.get(&tensor.name) {
                bail!(
                    "tensor {} appears in {} and again here",
                    tensor.name,
                    self.shards[self.tensor_shard[prev]].path.display()
                );
            }
            self.index.insert(tensor.name.clone(), self.tensors.len());
            self.tensors.push(tensor);
            self.tensor_shard.push(shard);
        }
        Ok(())
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn shard_path(&self, shard: usize) -> Option<&Path> {
        self.shards.get(shard).map(|s| s.path.as_path())
    }

    /// One shard's `__metadata__`. Per shard, not merged: the expert shards
    /// each carry their own `layer` / `expert_start` / `expert_end`.
    pub fn shard_metadata(&self, shard: usize) -> Option<&HashMap<String, String>> {
        self.shards.get(shard).map(|s| &s.metadata)
    }

    pub fn tensors(&self) -> &[SafeTensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&SafeTensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    /// Which shard holds `name`, for callers that want to stage a whole file.
    pub fn tensor_shard(&self, name: &str) -> Option<usize> {
        self.index.get(name).map(|&i| self.tensor_shard[i])
    }

    /// The tensor's bytes, borrowed straight out of the mmap. No copy.
    pub fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        let idx = *self
            .index
            .get(name)
            .ok_or_else(|| anyhow!("tensor {name} not in checkpoint"))?;
        let info = &self.tensors[idx];
        let shard = &self.shards[self.tensor_shard[idx]];
        let start = usize::try_from(info.offset)
            .ok()
            .and_then(|o| shard.blob_start.checked_add(o))
            .ok_or_else(|| anyhow!("tensor {name}: offset overflow"))?;
        let end = usize::try_from(info.len)
            .ok()
            .and_then(|l| start.checked_add(l))
            .filter(|&e| e <= shard.mmap.len())
            .ok_or_else(|| anyhow!("tensor {name}: data out of file bounds"))?;
        Ok(&shard.mmap[start..end])
    }
}

#[cfg(test)]
mod header_tests {
    use super::*;

    /// Build a header the way the reference writer does: 8-byte LE length,
    /// then the JSON.
    fn header(json: &str) -> Vec<u8> {
        let mut out = (json.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(json.as_bytes());
        out
    }

    /// The whole point of the module: safetensors' contiguous dim is LAST,
    /// GGUF's is FIRST. Reversing is the entire conversion, and getting it
    /// wrong transposes every weight into plausible garbage.
    #[test]
    fn shape_is_reversed_into_gguf_ne_order() {
        let json = r#"{"lm_head.weight":{"dtype":"BF16","shape":[248320,2560],"data_offsets":[0,1271398400]}}"#;
        let (tensors, meta) = parse_header(json.as_bytes(), 1_271_398_400).expect("parse");
        assert!(meta.is_empty());
        assert_eq!(tensors.len(), 1);
        // ne[0] = 2560 = hidden = the contiguous dim; ne[1] = 248320 = vocab.
        assert_eq!(tensors[0].dims, vec![2560, 248320]);
        assert_eq!(tensors[0].ggml_type, Some(GgmlType::Bf16));
        assert_eq!(tensors[0].len, 1_271_398_400);
        assert_eq!(tensors[0].element_count(), 248_320 * 2560);
    }

    /// `__metadata__` is not a tensor, and the expert shards' `layer` /
    /// `expert_start` keys are worth keeping.
    #[test]
    fn metadata_key_is_not_a_tensor() {
        let json = r#"{"__metadata__":{"format":"pt","layer":"0","expert_start":"0"},
                       "w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let (tensors, meta) = parse_header(json.as_bytes(), 8).expect("parse");
        assert_eq!(tensors.len(), 1, "__metadata__ leaked in as a tensor");
        assert_eq!(tensors[0].name, "w");
        assert_eq!(meta.get("layer").map(String::as_str), Some("0"));
        assert_eq!(meta.get("expert_start").map(String::as_str), Some("0"));
    }

    /// A rank-0 scalar (`shape: []`) is 1 element, not 0 — the expert shards
    /// carry 147,456 of them as `input_scale` / `weight_scale_2`.
    #[test]
    fn rank_zero_scalar_is_one_element() {
        let json = r#"{"s":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#;
        let (tensors, _) = parse_header(json.as_bytes(), 4).expect("parse");
        assert!(tensors[0].dims.is_empty());
        assert_eq!(tensors[0].element_count(), 1);
        assert_eq!(tensors[0].len, 4);
    }

    /// NVFP4 planes arrive as `U8`. Guessing a `GgmlType` for them would
    /// invent a block layout the file does not have.
    #[test]
    fn unmapped_dtypes_keep_their_string() {
        let json = r#"{"q":{"dtype":"U8","shape":[4,8],"data_offsets":[0,32]},
                       "f8":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[32,36]},
                       "i":{"dtype":"I64","shape":[3],"data_offsets":[36,60]}}"#;
        let (tensors, _) = parse_header(json.as_bytes(), 60).expect("parse");
        let by = |n: &str| tensors.iter().find(|t| t.name == n).expect(n);
        assert_eq!(by("q").ggml_type, None);
        assert_eq!(by("q").dtype, "U8");
        assert_eq!(by("f8").ggml_type, Some(GgmlType::F8E4M3));
        assert_eq!(by("i").ggml_type, Some(GgmlType::I64));
    }

    #[test]
    fn rejects_offsets_past_the_blob() {
        let json = r#"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        let err = parse_header(json.as_bytes(), 8).expect_err("must reject");
        assert!(
            err.to_string().contains("outside a 8-byte blob"),
            "unexpected error: {err}"
        );
    }

    /// The shape/length cross-check: a header claiming 4 f32s in 8 bytes is a
    /// corrupt or mis-parsed entry, and must not become a short slice.
    #[test]
    fn rejects_length_that_contradicts_the_shape() {
        let json = r#"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}"#;
        let err = parse_header(json.as_bytes(), 8).expect_err("must reject");
        assert!(
            err.to_string().contains("implies"),
            "unexpected error: {err}"
        );
    }

    /// The tier the cross-check used to skip. A `[640, 2560]` NVFP4 plane is
    /// stored `[640, 1280]` — two FP4 values per byte — so a header that
    /// declares the LOGICAL width over the PACKED byte span is off by exactly
    /// 2x. That is the mistake a repacker or a hand-edited header makes, it is
    /// 56.25 GiB of this checkpoint, and while `U8` was exempt it was accepted
    /// in silence and handed to a GEMV as a half-length row.
    #[test]
    fn rejects_u8_plane_whose_shape_contradicts_its_byte_span() {
        let bad =
            r#"{"e.gate_proj.weight":{"dtype":"U8","shape":[640,2560],"data_offsets":[0,819200]}}"#;
        let err = parse_header(bad.as_bytes(), 819_200).expect_err("must reject");
        assert!(
            err.to_string().contains(
                "dtype U8 shape [640, 2560] implies Some(1638400) bytes but \
                 data_offsets declare 819200"
            ),
            "unexpected error: {err}"
        );

        // The same plane declared at its packed width is what the real shards
        // carry, and still opens.
        let good =
            r#"{"e.gate_proj.weight":{"dtype":"U8","shape":[640,1280],"data_offsets":[0,819200]}}"#;
        let (tensors, _) = parse_header(good.as_bytes(), 819_200).expect("packed shape parses");
        assert_eq!(tensors[0].dims, vec![1280, 640]);
        assert_eq!(tensors[0].ggml_type, None, "still not guessed into a block");
    }

    /// The rest of the whole-byte tier, none of which has a `GgmlType` twin.
    /// Each row is the correct element count against a deliberately wrong span.
    #[test]
    fn cross_check_covers_the_unsigned_and_bool_tiers() {
        for (dtype, elems, wrong_len) in [
            ("U16", 4u64, 4u64),
            ("U32", 4, 8),
            ("U64", 4, 16),
            ("BOOL", 4, 2),
            ("F8_E5M2", 4, 8),
        ] {
            let json = format!(
                r#"{{"w":{{"dtype":"{dtype}","shape":[{elems}],"data_offsets":[0,{wrong_len}]}}}}"#
            );
            assert!(
                parse_header(json.as_bytes(), wrong_len).is_err(),
                "{dtype}: a {wrong_len}-byte span for {elems} elements must be rejected"
            );
        }
    }

    /// The exemption that must survive: a sub-byte dtype packs several elements
    /// into a byte, so `shape x 1` is not its length and asserting it would
    /// refuse a valid file. `bytes_per_element` returns `None` for these.
    #[test]
    fn sub_byte_dtypes_stay_exempt_from_the_cross_check() {
        let json = r#"{"q":{"dtype":"F4","shape":[64],"data_offsets":[0,32]}}"#;
        let (tensors, _) = parse_header(json.as_bytes(), 32).expect("F4 must not be rejected");
        assert_eq!(tensors[0].ggml_type, None);
        assert_eq!(tensors[0].len, 32);
        assert_eq!(tensors[0].element_count(), 64);
    }

    /// Names come out of a JSON string, so escapes are the writer's choice.
    #[test]
    fn decodes_string_escapes_in_names() {
        let json = r#"{"a\/b\u00e9\n":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let (tensors, _) = parse_header(json.as_bytes(), 4).expect("parse");
        assert_eq!(tensors[0].name, "a/bé\n");
    }

    /// Unknown keys inside a tensor entry must not break the reader.
    #[test]
    fn skips_unknown_entry_keys() {
        let json = r#"{"w":{"extra":{"nested":[1,2,{"deep":true}]},"dtype":"F32",
                       "shape":[1],"data_offsets":[0,4],"trailing":null}}"#;
        let (tensors, _) = parse_header(json.as_bytes(), 4).expect("parse");
        assert_eq!(tensors[0].name, "w");
        assert_eq!(tensors[0].len, 4);
    }

    /// Two shards, each with its own header and its own blob-relative
    /// offsets. Shard 1's tensor starts at offset 0 of ITS blob, so a reader
    /// that kept one global base would hand back shard 0's bytes.
    #[test]
    fn two_shard_directory_resolves_per_shard_offsets() {
        let dir = std::env::temp_dir().join(format!(
            "arle-st-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");

        let mut a = header(
            r#"{"__metadata__":{"shard":"a"},"a.w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
        );
        a.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let mut b = header(r#"{"b.w":{"dtype":"I8","shape":[3],"data_offsets":[0,3]}}"#);
        b.extend_from_slice(&[9, 10, 11]);
        std::fs::write(dir.join("00-a.safetensors"), &a).expect("write a");
        std::fs::write(dir.join("01-b.safetensors"), &b).expect("write b");
        // Must be ignored by discovery even though the stem matches.
        std::fs::write(dir.join("model.safetensors.index.json"), b"{}").expect("write index");

        let st = SafeTensorsDir::open_dir(&dir).expect("open dir");
        assert_eq!(st.shard_count(), 2, "index.json was picked up as a shard");
        assert_eq!(st.tensors().len(), 2);
        assert_eq!(
            st.tensor_data("a.w").expect("a.w"),
            &[1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(st.tensor_data("b.w").expect("b.w"), &[9, 10, 11]);
        assert_eq!(st.tensor_shard("b.w"), Some(1));
        assert_eq!(
            st.shard_metadata(0).and_then(|m| m.get("shard")),
            Some(&"a".to_string())
        );
        assert!(st.shard_metadata(1).is_some_and(HashMap::is_empty));
        assert!(st.tensor_data("nope").is_err());

        std::fs::remove_dir_all(&dir).expect("clean temp dir");
    }
}

#[cfg(test)]
mod on_box_tests {
    use super::*;

    /// Measured on this box, 2026-08: 206 `*.safetensors` (192 expert shards +
    /// 10 `model-plefp8-*` + 4 `model-bf16-*`) holding 296,475 tensors.
    const CHECKPOINT: &str = r"C:\Users\Asus\models\qwen3.8-flash-next-nvfp4";
    const SHARDS: usize = 206;
    const TENSORS: usize = 296_475;

    /// Not `#[ignore]`d on purpose: a dimension-order convention that only a
    /// never-run test pins is not pinned at all. It costs nothing off-box —
    /// the directory is absent and the test returns immediately.
    fn open() -> Option<SafeTensorsDir> {
        let dir = std::env::var("INFER_SAFETENSORS_TEST_DIR").unwrap_or_else(|_| CHECKPOINT.into());
        let path = std::path::Path::new(&dir);
        if !path.is_dir() {
            eprintln!(
                "skip: {} not present (set INFER_SAFETENSORS_TEST_DIR)",
                path.display()
            );
            return None;
        }
        Some(SafeTensorsDir::open_dir(path).expect("open checkpoint"))
    }

    #[test]
    fn real_checkpoint_indexes_every_shard() {
        let Some(st) = open() else { return };
        assert_eq!(st.shard_count(), SHARDS);
        assert_eq!(st.tensors().len(), TENSORS);

        // The expert shards carry their own routing metadata; the bf16 shards
        // do not. Shard 0 sorts first, so it is layer 0's experts 0..128.
        let meta = st.shard_metadata(0).expect("shard 0");
        assert_eq!(meta.get("layer").map(String::as_str), Some("0"));
        assert_eq!(meta.get("expert_start").map(String::as_str), Some("0"));
        assert_eq!(meta.get("expert_end").map(String::as_str), Some("128"));
    }

    /// PIN: safetensors says `shape == [248320, 2560]` (`[out, in]`,
    /// row-major). This reader reports GGUF `ne` order, so `dims[0]` is the
    /// contiguous input width and `dims[1]` is the vocab. A caller asking
    /// "which axis is vocab?" must find it at `dims[1]`, exactly where
    /// llama.cpp puts `output.weight = {n_embd, n_vocab}`.
    #[test]
    fn real_checkpoint_pins_lm_head_dimension_order() {
        let Some(st) = open() else { return };
        const HIDDEN: u64 = 2560; // config.json text_config.hidden_size
        const VOCAB: u64 = 248_320; // config.json text_config.vocab_size

        for name in ["lm_head.weight", "model.language_model.embed_tokens.weight"] {
            let t = st.tensor(name).unwrap_or_else(|| panic!("{name} missing"));
            assert_eq!(t.dims, vec![HIDDEN, VOCAB], "{name} dimension order");
            assert_eq!(t.ggml_type, Some(GgmlType::Bf16), "{name} dtype");
            assert_eq!(t.len, HIDDEN * VOCAB * 2, "{name} byte length");
        }

        // A non-square weight where the two axes cannot be confused: q_proj is
        // 24 heads x 256 head_dim = 6144 out, 2560 in.
        let q = st
            .tensor("model.language_model.layers.0.self_attn.q_proj.weight")
            .or_else(|| st.tensor("mtp.layers.0.self_attn.q_proj.weight"))
            .expect("a q_proj");
        assert_eq!(q.dims[0], HIDDEN, "q_proj ne[0] must be the input width");
    }

    /// Read a real tensor's bytes and check the values, not just the length —
    /// a wrong `blob_start` or a shard mix-up yields a right-sized slice of
    /// the wrong bytes.
    #[test]
    fn real_checkpoint_reads_bytes_without_copying() {
        let Some(st) = open() else { return };
        let name = "model.language_model.layers.1.ple.ple_embedding.layer_multipliers";
        let info = st.tensor(name).expect("layer_multipliers");
        assert_eq!(info.dtype, "I64");
        assert_eq!(info.dims, vec![3]);

        let bytes = st.tensor_data(name).expect("read");
        // Captured from the on-box file: three i64 n-gram hash multipliers.
        assert_eq!(
            bytes,
            &[
                137, 215, 14, 235, 142, 21, 0, 0, //
                53, 255, 48, 2, 74, 18, 0, 0, //
                167, 59, 235, 246, 82, 7, 0, 0,
            ]
        );

        // The 47.68 GiB PLE table must be reachable as a borrowed slice; the
        // assert is on the pointer's range, so nothing is read or copied.
        let ple = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight";
        let table = st.tensor_data(ple).expect("read PLE shard 0");
        assert_eq!(table.len(), 2_500_012 * 160);
        assert_eq!(
            st.tensor(ple).and_then(|t| t.ggml_type),
            Some(GgmlType::F8E4M3)
        );
    }

    /// The routed experts are NVFP4, which safetensors spells as a `U8` byte
    /// plane plus a separate `F8_E4M3` scale. Both must survive as themselves.
    #[test]
    fn real_checkpoint_leaves_nvfp4_planes_unmapped() {
        let Some(st) = open() else { return };
        let plane = st
            .tensor("model.language_model.layers.0.mlp.experts.0.gate_proj.weight")
            .expect("expert gate_proj weight");
        assert_eq!(plane.dtype, "U8");
        assert_eq!(
            plane.ggml_type, None,
            "a U8 NVFP4 plane must not be guessed into a GgmlType"
        );

        let scale = st
            .tensor("model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale")
            .expect("expert gate_proj weight_scale");
        assert_eq!(scale.ggml_type, Some(GgmlType::F8E4M3));
    }

    /// Sizes the tier the cross-check now guards. `open()` succeeding is itself
    /// the evidence that every U8 plane satisfies `shape x 1 == byte span`, so
    /// what is worth asserting here is HOW MUCH that covers — a future reader
    /// that quietly re-exempts `U8` would leave this many bytes unchecked.
    #[test]
    fn real_checkpoint_u8_tier_is_the_bulk_of_the_file() {
        let Some(st) = open() else { return };
        let mut planes = 0usize;
        let mut bytes = 0u64;
        for t in st.tensors() {
            if t.dtype != "U8" {
                continue;
            }
            planes += 1;
            bytes += t.len;
            assert_eq!(
                t.element_count(),
                t.len,
                "{}: U8 is exactly one byte per element",
                t.name
            );
        }
        // Measured 2026-08 over all 206 shard headers: 48 layers x 512 experts
        // x 3 projections, and 56.25 GiB — 45% of the checkpoint's 126 GiB.
        assert_eq!(planes, 73_728, "U8 NVFP4 planes");
        assert_eq!(bytes, 60_397_977_600, "U8 bytes");
    }
}
