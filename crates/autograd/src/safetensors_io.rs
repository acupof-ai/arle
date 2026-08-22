use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use half::bf16;
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors, serialize_to_file};

use crate::{AutogradError, Result, Tensor, TensorId, TensorStore};

pub struct SafetensorsRegistry {
    map: HashMap<String, TensorId>,
}

impl SafetensorsRegistry {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: impl Into<String>, id: TensorId) {
        self.map.insert(name.into(), id);
    }

    pub fn get(&self, name: &str) -> Option<TensorId> {
        self.map.get(name).copied()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(String::as_str)
    }

    pub fn load_into(&mut self, store: &mut TensorStore, path: &Path) -> Result<()> {
        self.load_into_impl(store, path, false)
    }

    /// Like [`load_into`], but fails if any tensor currently registered in
    /// `self` is missing from the file — resume paths where a partial or
    /// mismatched checkpoint must not silently hybridize with base-model
    /// weights.
    pub fn load_into_strict(&mut self, store: &mut TensorStore, path: &Path) -> Result<()> {
        self.load_into_impl(store, path, true)
    }

    fn load_into_impl(&mut self, store: &mut TensorStore, path: &Path, strict: bool) -> Result<()> {
        let file = File::open(path).map_err(|err| {
            tape_invariant(format!(
                "failed to open safetensors file {}: {err}",
                path.display()
            ))
        })?;
        // SAFETY: Mmap::map reads the file contents; the file is kept open by `file`
        // for the lifetime of the mmap, so no use-after-close.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|err| {
            tape_invariant(format!(
                "failed to memory-map safetensors file {}: {err}",
                path.display()
            ))
        })?;
        let tensors = SafeTensors::deserialize(&mmap[..])
            .map_err(|err| tape_invariant(format!("failed to deserialize safetensors: {err}")))?;

        // Strict mode: remember which registered names the file covered;
        // error on any missing after the loop. Unknown names still
        // auto-register (same as non-strict).
        let mut seen = if strict {
            Some(std::collections::HashSet::<String>::new())
        } else {
            None
        };

        for (name, view) in tensors.iter() {
            let shape = view.shape().to_vec();
            let data = tensor_view_to_f32(&view)?;

            if let Some(id) = self.map.get(name).copied() {
                let expected = store.tensor(id)?.shape.clone();
                if expected != shape {
                    return Err(AutogradError::ShapeMismatch {
                        expected,
                        got: shape,
                    });
                }
                let tensor = store.tensor_mut(id)?;
                tensor.data = data;
                if let Some(set) = seen.as_mut() {
                    set.insert(name.to_owned());
                }
            } else {
                let id = store.alloc(Tensor::new(data, shape, true)?);
                self.insert(name.to_owned(), id);
                if let Some(set) = seen.as_mut() {
                    set.insert(name.to_owned());
                }
            }
        }

        if let Some(seen) = seen {
            let missing: Vec<&str> = self
                .map
                .keys()
                .filter(|k| !seen.contains(k.as_str()))
                .map(String::as_str)
                .collect();
            if !missing.is_empty() {
                let mut sorted = missing;
                sorted.sort_unstable();
                let shown: Vec<&str> = sorted.iter().take(5).copied().collect();
                let suffix = if sorted.len() > 5 {
                    format!(" (+{} more)", sorted.len() - 5)
                } else {
                    String::new()
                };
                return Err(tape_invariant(format!(
                    "safetensors {} is missing {} registered tensor(s): {}{}",
                    path.display(),
                    sorted.len(),
                    shown.join(", "),
                    suffix,
                )));
            }
        }

        Ok(())
    }

    pub fn save_from(&self, store: &mut TensorStore, path: &Path) -> Result<()> {
        let data = self
            .map
            .iter()
            .map(|(name, id)| -> Result<_> {
                let shape = store.tensor(*id)?.shape.clone();
                let host = store.to_host(*id)?;
                let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
                Ok((name.clone(), TensorFileView { shape, bytes }))
            })
            .collect::<Result<Vec<_>>>()?;

        serialize_to_file(data, None, path)
            .map_err(|err| tape_invariant(format!("failed to serialize safetensors: {err}")))?;
        Ok(())
    }

    // infer's `DeviceMatrix::from_safetensors` reinterprets the file bytes as
    // `&[bf16]` and rejects anything else. The f32 path stays bit-exact for
    // training-side roundtrip tests; this bf16 path is the one infer consumes.
    pub fn save_from_bf16(&self, store: &mut TensorStore, path: &Path) -> Result<()> {
        let data = self
            .map
            .iter()
            .map(|(name, id)| -> Result<_> {
                let shape = store.tensor(*id)?.shape.clone();
                let host = store.to_host(*id)?;
                let bytes: Vec<u8> = host
                    .iter()
                    .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
                    .collect();
                Ok((name.clone(), TensorFileBf16View { shape, bytes }))
            })
            .collect::<Result<Vec<_>>>()?;

        serialize_to_file(data, None, path).map_err(|err| {
            tape_invariant(format!("failed to serialize bf16 safetensors: {err}"))
        })?;
        Ok(())
    }
}

impl Default for SafetensorsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

struct TensorFileView {
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl safetensors::View for TensorFileView {
    fn dtype(&self) -> Dtype {
        Dtype::F32
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.bytes.as_slice())
    }

    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

struct TensorFileBf16View {
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl safetensors::View for TensorFileBf16View {
    fn dtype(&self) -> Dtype {
        Dtype::BF16
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.bytes.as_slice())
    }

    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

/// Widen a little-endian BF16 payload to f32.
///
/// The obvious `chunks_exact(2).map(u16::from_le_bytes)` is one scalar
/// unaligned read per element; `train::qwen35_loader` measured that shape into
/// a watchdog kill on `embed_tokens`/`lm_head` (1.27 B elements each on the
/// 27B checkpoints). Bulk-copy first, then widen with a shift.
#[must_use]
pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let mut bits = vec![0u16; bytes.len() / 2];
    // SAFETY: `bits` owns `bits.len()` u16 = `bytes.len() & !1` contiguous
    // bytes, and u16 has no invalid bit patterns.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            bits.as_mut_ptr().cast::<u8>(),
            bits.len() * 2,
        );
    }
    if cfg!(target_endian = "big") {
        for b in &mut bits {
            *b = b.swap_bytes();
        }
    }
    bits.into_iter()
        .map(|b| f32::from_bits(u32::from(b) << 16))
        .collect()
}

fn tensor_view_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    let data = view.data();
    match view.dtype() {
        Dtype::F32 => Ok(data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()),
        Dtype::BF16 => Ok(bf16_bytes_to_f32(data)),
        Dtype::F16 => Ok(data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| half::f16::from_le_bytes(*chunk).to_f32())
            .collect()),
        dtype => Err(tape_invariant(format!("unsupported dtype: {dtype}"))),
    }
}

fn tape_invariant(message: String) -> AutogradError {
    AutogradError::TapeInvariant(Box::leak(message.into_boxed_str()))
}

#[cfg(all(test, feature = "safetensors"))]
mod tests {
    use super::*;

    use safetensors::{Dtype, serialize_to_file};
    use tempfile::tempdir;

    #[test]
    fn roundtrip_f32() -> Result<()> {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("roundtrip.safetensors");

        let mut source_store = TensorStore::default();
        let first = source_store.alloc(Tensor::new(vec![1.0, -2.5, 3.25, 0.0], vec![2, 2], true)?);
        let second = source_store.alloc(Tensor::new(vec![4.5, -5.0, 6.75], vec![3], true)?);

        let mut source_registry = SafetensorsRegistry::new();
        source_registry.insert("layer.weight", first);
        source_registry.insert("layer.bias", second);
        source_registry.save_from(&mut source_store, &path)?;

        let mut loaded_store = TensorStore::default();
        let mut loaded_registry = SafetensorsRegistry::new();
        loaded_registry.load_into(&mut loaded_store, &path)?;

        assert_eq!(loaded_registry.len(), 2);
        assert_f32_bits_eq(
            &source_store.to_host(first)?,
            &loaded_store.to_host(loaded_registry.get("layer.weight").expect("weight id"))?,
        );
        assert_eq!(
            loaded_store
                .tensor(loaded_registry.get("layer.weight").expect("weight id"))?
                .shape,
            vec![2, 2]
        );
        assert_f32_bits_eq(
            &source_store.to_host(second)?,
            &loaded_store.to_host(loaded_registry.get("layer.bias").expect("bias id"))?,
        );
        assert_eq!(
            loaded_store
                .tensor(loaded_registry.get("layer.bias").expect("bias id"))?
                .shape,
            vec![3]
        );

        Ok(())
    }

    #[test]
    fn overwrite_existing() -> Result<()> {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("overwrite.safetensors");

        let mut source_store = TensorStore::default();
        let original_id =
            source_store.alloc(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)?);
        let mut source_registry = SafetensorsRegistry::new();
        source_registry.insert("weight", original_id);
        source_registry.save_from(&mut source_store, &path)?;

        let mut target_store = TensorStore::default();
        let existing_id =
            target_store.alloc(Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2], true)?);
        let mut target_registry = SafetensorsRegistry::new();
        target_registry.insert("weight", existing_id);

        target_registry.load_into(&mut target_store, &path)?;

        assert_eq!(target_registry.get("weight"), Some(existing_id));
        assert_f32_bits_eq(
            &source_store.to_host(original_id)?,
            &target_store.to_host(existing_id)?,
        );

        Ok(())
    }

    #[test]
    fn bf16_widens_to_f32() -> Result<()> {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("bf16.safetensors");

        let source = [1.0_f32, 2.5, -3.0, 0.0];
        let narrowed: Vec<bf16> = source.iter().copied().map(bf16::from_f32).collect();
        let bytes: Vec<u8> = narrowed
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let data = vec![(
            "weight".to_owned(),
            TestTensorView {
                dtype: Dtype::BF16,
                shape: vec![2, 2],
                bytes,
            },
        )];
        serialize_to_file(data, None, &path).map_err(|err| {
            tape_invariant(format!("failed to serialize bf16 test tensor: {err}"))
        })?;

        let mut store = TensorStore::default();
        let mut registry = SafetensorsRegistry::new();
        registry.load_into(&mut store, &path)?;

        let loaded = store.to_host(registry.get("weight").expect("weight id"))?;
        let expected: Vec<f32> = narrowed.iter().map(|value| value.to_f32()).collect();
        for (got, want) in loaded.iter().zip(expected.iter()) {
            assert!((got - want).abs() <= 1e-2, "got {got}, want {want}");
        }

        Ok(())
    }

    #[test]
    fn roundtrip_bf16_via_save() -> Result<()> {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("roundtrip-bf16.safetensors");

        // Pick values that actually survive bf16's 7-bit mantissa; we still
        // assert with a relative tolerance because bf16 is lossy.
        let source_values = vec![1.0_f32, -2.5, 3.25, 0.0];
        let mut source_store = TensorStore::default();
        let weight_id = source_store.alloc(Tensor::new(source_values.clone(), vec![2, 2], true)?);
        let mut source_registry = SafetensorsRegistry::new();
        source_registry.insert("weight", weight_id);
        source_registry.save_from_bf16(&mut source_store, &path)?;

        let mut loaded_store = TensorStore::default();
        let mut loaded_registry = SafetensorsRegistry::new();
        loaded_registry.load_into(&mut loaded_store, &path)?;

        let loaded = loaded_store.to_host(loaded_registry.get("weight").expect("weight id"))?;
        assert_eq!(loaded.len(), source_values.len());
        for (got, want) in loaded.iter().zip(source_values.iter()) {
            let tol = want.abs().max(1.0) * 1e-2;
            assert!(
                (got - want).abs() <= tol,
                "bf16 roundtrip drift: got {got}, want {want}"
            );
        }
        assert_eq!(
            loaded_store
                .tensor(loaded_registry.get("weight").expect("weight id"))?
                .shape,
            vec![2, 2]
        );

        Ok(())
    }

    #[test]
    fn shape_mismatch_errors() -> Result<()> {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("shape-mismatch.safetensors");

        let mut source_store = TensorStore::default();
        let source_id = source_store.alloc(Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            true,
        )?);
        let mut source_registry = SafetensorsRegistry::new();
        source_registry.insert("weight", source_id);
        source_registry.save_from(&mut source_store, &path)?;

        let mut target_store = TensorStore::default();
        let target_id = target_store.alloc(Tensor::new(
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![3, 2],
            true,
        )?);
        let mut target_registry = SafetensorsRegistry::new();
        target_registry.insert("weight", target_id);

        let err = target_registry
            .load_into(&mut target_store, &path)
            .expect_err("shape mismatch");
        assert!(matches!(err, AutogradError::ShapeMismatch { .. }));

        Ok(())
    }

    struct TestTensorView {
        dtype: Dtype,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    impl safetensors::View for TestTensorView {
        fn dtype(&self) -> Dtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(self.bytes.as_slice())
        }

        fn data_len(&self) -> usize {
            self.bytes.len()
        }
    }

    fn assert_f32_bits_eq(lhs: &[f32], rhs: &[f32]) {
        assert_eq!(lhs.len(), rhs.len());
        for (left, right) in lhs.iter().zip(rhs.iter()) {
            assert_eq!(left.to_bits(), right.to_bits());
        }
    }
}
