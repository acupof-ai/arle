use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors, serialize_to_file};

use crate::{AutogradError, GpuTensor, Result, TensorId, TensorStore};

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
        let file = File::open(path).map_err(|err| {
            tape_invariant(format!(
                "failed to open safetensors file {}: {err}",
                path.display()
            ))
        })?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|err| {
            tape_invariant(format!(
                "failed to memory-map safetensors file {}: {err}",
                path.display()
            ))
        })?;
        let tensors = SafeTensors::deserialize(&mmap[..])
            .map_err(|err| tape_invariant(format!("failed to deserialize safetensors: {err}")))?;

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
            } else {
                let id = store.alloc(GpuTensor::new(data, shape, true)?);
                self.insert(name.to_owned(), id);
            }
        }

        Ok(())
    }

    pub fn save_from(&self, store: &mut TensorStore, path: &Path) -> Result<()> {
        let mut data = Vec::with_capacity(self.map.len());
        for (name, id) in &self.map {
            let shape = store.tensor(*id)?.shape.clone();
            let host = store.to_host(*id)?;
            let bytes: Vec<u8> = host.iter().flat_map(|value| value.to_le_bytes()).collect();
            data.push((name.clone(), TensorFileView { shape, bytes }));
        }

        serialize_to_file(data, None, path)
            .map_err(|err| tape_invariant(format!("failed to serialize safetensors: {err}")))?;
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

fn tensor_view_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    let data = view.data();
    match view.dtype() {
        Dtype::F32 => Ok(data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()),
        Dtype::BF16 => Ok(data
            .chunks_exact(2)
            .map(|chunk| f32::from_bits((u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16))
            .collect()),
        Dtype::F16 => Ok(data
            .chunks_exact(2)
            .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
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

    use half::bf16;
    use safetensors::{Dtype, serialize_to_file};
    use tempfile::tempdir;

    #[test]
    fn roundtrip_f32() -> Result<()> {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("roundtrip.safetensors");

        let mut source_store = TensorStore::default();
        let first = source_store.alloc(GpuTensor::new(
            vec![1.0, -2.5, 3.25, 0.0],
            vec![2, 2],
            true,
        )?);
        let second = source_store.alloc(GpuTensor::new(vec![4.5, -5.0, 6.75], vec![3], true)?);

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
            source_store.alloc(GpuTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)?);
        let mut source_registry = SafetensorsRegistry::new();
        source_registry.insert("weight", original_id);
        source_registry.save_from(&mut source_store, &path)?;

        let mut target_store = TensorStore::default();
        let existing_id =
            target_store.alloc(GpuTensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2], true)?);
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
    fn shape_mismatch_errors() -> Result<()> {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("shape-mismatch.safetensors");

        let mut source_store = TensorStore::default();
        let source_id = source_store.alloc(GpuTensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            true,
        )?);
        let mut source_registry = SafetensorsRegistry::new();
        source_registry.insert("weight", source_id);
        source_registry.save_from(&mut source_store, &path)?;

        let mut target_store = TensorStore::default();
        let target_id = target_store.alloc(GpuTensor::new(
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
