use crate::{
    AutogradError, Result,
    backend::{Backend, CpuBackend, Device, DeviceHandle},
};
use std::{collections::HashSet, fmt, path::PathBuf, sync::Arc};

const CHECKPOINT_L3_NS: u64 = 0x43;
const CHECKPOINT_L3_CHUNK_NS: u64 = 0x44;

pub type TensorId = usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dirty {
    /// Host data is authoritative; there is no usable device mirror.
    Host,
    /// Device data is authoritative; host `data` must be empty.
    Device,
    /// Host and device mirrors are both current.
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckpointResidency {
    None,
    Host,
    L3,
    /// Backend-owned pinned host slot; `data` is empty — the buffer is the storage.
    Pinned(u32),
}

#[derive(Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub size: usize,
    pub requires_grad: bool,
    pub grad: Option<TensorId>,
    pub device_handle: Option<DeviceHandle>,
    pub dirty: Dirty,
    checkpoint_residency: CheckpointResidency,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        assert!(
            matches!(
                self.checkpoint_residency,
                CheckpointResidency::None | CheckpointResidency::Host
            ),
            "restore an off-host checkpoint before cloning"
        );
        // P3.1 (post-Wave-1): mirror the `TensorStore::clone_tensor`
        // discipline at the `Tensor` level — when the source is
        // `Dirty::Device` (host data is empty/stale, the truth lives on
        // device), preserve the `Arc`-shared `DeviceHandle` (ref-count
        // bump, no extra alloc) instead of panicking. Callers that need
        // host data are expected to `ensure_host` first; callers that
        // only need shape/metadata or that route the clone back through
        // a device path (which is the entire post-P3.1 backward chain)
        // see a clone that still has device residency.
        //
        // Pre-P3.1 this asserted `dirty != Dirty::Device`. The
        // assertion was load-bearing for the *post* M5.3b world where
        // `Tape::backward`'s `flush_to_host_batch` made every tape output
        // `Dirty::Both` before any clone. Once P3.1 dropped that flush
        // on CUDA (it was the 1 GB DtoH), assorted backward ops still
        // doing `store.tensor(x)?.clone()` (rope, rms_norm, silu, ...)
        // hit the assert. Preserving the handle is the right fix; the
        // host-only fallback paths inside those ops already
        // `ensure_host(x)` before they touch `.data`.
        Self {
            data: if self.dirty == Dirty::Device {
                Vec::new()
            } else {
                self.data.clone()
            },
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            size: self.size,
            requires_grad: self.requires_grad,
            grad: self.grad,
            device_handle: if self.dirty == Dirty::Host {
                None
            } else {
                self.device_handle.clone()
            },
            dirty: self.dirty.clone(),
            checkpoint_residency: CheckpointResidency::None,
        }
    }
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Result<Self> {
        let size = shape_size(&shape);
        if data.len() != size {
            return Err(AutogradError::DataLengthMismatch {
                len: data.len(),
                shape,
                size,
            });
        }

        let strides = contiguous_strides(&shape);
        Ok(Self {
            data,
            shape,
            strides,
            size,
            requires_grad,
            grad: None,
            device_handle: None,
            dirty: Dirty::Host,
            checkpoint_residency: CheckpointResidency::None,
        })
    }

    /// Metadata-only tensor slot for checkpoint loaders.
    ///
    /// The shape/grad metadata is available immediately, but no host or device
    /// value exists until the loader installs a real `DeviceHandle` or replaces
    /// the slot with materialized host data. Treat it as `Dirty::Device` so any
    /// accidental pre-load use fails through the existing missing-handle guard
    /// instead of silently reading an empty host buffer.
    pub fn unmaterialized(shape: Vec<usize>, requires_grad: bool) -> Result<Self> {
        let size = shape_size(&shape);
        let strides = contiguous_strides(&shape);
        Ok(Self {
            data: Vec::new(),
            shape,
            strides,
            size,
            requires_grad,
            grad: None,
            device_handle: None,
            dirty: Dirty::Device,
            checkpoint_residency: CheckpointResidency::None,
        })
    }
}

/// Idle host buffers kept for reuse across checkpoint offloads. Bounded: OPD
/// trajectory lengths span ~5K–30K tokens, so first-fit reuse with unbounded
/// retention accumulated one buffer per size class ever seen (449-task run:
/// summed host RssAnon stepped 208 → 306 GiB). The pool only has to absorb
/// churn between a release and the next take.
#[derive(Debug)]
struct CheckpointOffloadPool {
    free: Vec<Vec<f32>>,
    pooled_bytes: usize,
    cap_bytes: usize,
}

impl Default for CheckpointOffloadPool {
    fn default() -> Self {
        Self {
            free: Vec::new(),
            pooled_bytes: 0,
            cap_bytes: CHECKPOINT_OFFLOAD_POOL_CAP_BYTES,
        }
    }
}

const CHECKPOINT_OFFLOAD_POOL_CAP_BYTES: usize = 4 << 30;

struct CheckpointL3 {
    store: kv_native_sys::KvTierStore,
    host_cap_bytes: usize,
    host_bytes: usize,
}

impl fmt::Debug for CheckpointL3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CheckpointL3")
            .field("host_cap_bytes", &self.host_cap_bytes)
            .field("host_bytes", &self.host_bytes)
            .finish_non_exhaustive()
    }
}

impl CheckpointL3 {
    fn from_env() -> Option<Self> {
        let root = std::env::var_os("ARLE_CHECKPOINT_L3_ROOT").map(PathBuf::from)?;
        let host_cap_bytes = parse_env_bytes("ARLE_CHECKPOINT_HOST_CAP_BYTES")?;
        let disk_cap_bytes = parse_env_bytes("ARLE_CHECKPOINT_L3_CAP_BYTES")?;
        Self::try_new(root, host_cap_bytes, disk_cap_bytes)
    }

    fn try_new(root: PathBuf, host_cap_bytes: usize, disk_cap_bytes: usize) -> Option<Self> {
        if host_cap_bytes == 0 || disk_cap_bytes == 0 {
            return None;
        }
        let mut store = kv_native_sys::KvTierStore::with_budget(0, kv_native_sys::BLOB_CHUNK_BYTES);
        if !store.set_disk(root, disk_cap_bytes, kv_native_sys::BLOB_CHUNK_BYTES) {
            return None;
        }
        Some(Self {
            store,
            host_cap_bytes,
            host_bytes: 0,
        })
    }

    fn should_spill(&self, bytes: usize) -> bool {
        self.host_bytes.saturating_add(bytes) > self.host_cap_bytes
    }

    fn insert(&mut self, id: TensorId, host: &[f32]) -> bool {
        let Ok(key) = u64::try_from(id) else {
            return false;
        };
        if key >> (kv_native_sys::TIER_NS_SHIFT - kv_native_sys::CHUNK_IDX_BITS) != 0 {
            return false;
        }
        self.store.insert_chunked(
            CHECKPOINT_L3_NS,
            CHECKPOINT_L3_CHUNK_NS,
            key,
            bytemuck::cast_slice(host),
        )
    }

    fn restore(&mut self, id: TensorId, expected_len: usize) -> Result<Vec<f32>> {
        let bytes = self
            .store
            .read_chunked(CHECKPOINT_L3_NS, CHECKPOINT_L3_CHUNK_NS, id as u64)
            .map_err(|_| AutogradError::TapeInvariant("checkpoint L3 restore failed"))?;
        if bytes.len() != expected_len.saturating_mul(std::mem::size_of::<f32>()) {
            return Err(AutogradError::TapeInvariant(
                "checkpoint L3 payload length mismatch",
            ));
        }
        let host = bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect();
        self.store
            .remove_chunked(CHECKPOINT_L3_NS, CHECKPOINT_L3_CHUNK_NS, id as u64);
        Ok(host)
    }

    fn remove(&mut self, id: TensorId) {
        self.store
            .remove_chunked(CHECKPOINT_L3_NS, CHECKPOINT_L3_CHUNK_NS, id as u64);
    }
}

fn parse_env_bytes(name: &str) -> Option<usize> {
    let raw = std::env::var(name).ok()?;
    match raw.parse::<usize>() {
        Ok(value) if value > 0 => Some(value),
        _ => {
            log::warn!("ignoring invalid {name}={raw:?}");
            None
        }
    }
}

impl CheckpointOffloadPool {
    fn take(&mut self, len: usize) -> Vec<f32> {
        // Best fit, not first fit: a short trajectory must not claim the
        // largest buffer and force the next long one to allocate.
        let pick = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, buf)| buf.capacity() >= len)
            .min_by_key(|(_, buf)| buf.capacity())
            .map(|(index, _)| index);
        match pick {
            Some(index) => {
                let mut buf = self.free.swap_remove(index);
                self.pooled_bytes = self
                    .pooled_bytes
                    .saturating_sub(buf.capacity() * std::mem::size_of::<f32>());
                buf.resize(len, 0.0);
                buf
            }
            None => vec![0.0; len],
        }
    }

    fn recycle(&mut self, mut buf: Vec<f32>) {
        let bytes = buf.capacity() * std::mem::size_of::<f32>();
        if bytes == 0 {
            return;
        }
        buf.clear();
        self.free.push(buf);
        self.pooled_bytes += bytes;
        // Smallest-first eviction: a large buffer serves any smaller take, so
        // capacity coverage is what the cap should buy.
        while self.pooled_bytes > self.cap_bytes && self.free.len() > 1 {
            let Some(index) = self
                .free
                .iter()
                .enumerate()
                .min_by_key(|(_, buf)| buf.capacity())
                .map(|(index, _)| index)
            else {
                break;
            };
            let dropped = self.free.swap_remove(index);
            self.pooled_bytes = self
                .pooled_bytes
                .saturating_sub(dropped.capacity() * std::mem::size_of::<f32>());
        }
    }
}

/// Storage dtype for activation tensors saved on the tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TapeDtype {
    #[default]
    F32,
    Bf16,
}

impl TapeDtype {
    /// NVRTC prelude injected ahead of the kernel sources to bind `T`.
    pub fn nvrtc_prelude(self) -> &'static str {
        match self {
            TapeDtype::F32 => "using T = float;\n",
            TapeDtype::Bf16 => "#include <cuda_bf16.h>\nusing T = __nv_bfloat16;\n",
        }
    }
}

#[derive(Debug)]
pub struct TensorStore {
    pub tensors: Vec<Option<Tensor>>,
    pub free_ids: Vec<TensorId>,
    backend: Arc<dyn Backend>,
    checkpoint_offload_pool: CheckpointOffloadPool,
    checkpoint_l3: Option<CheckpointL3>,
    tape_dtype: TapeDtype,
}

impl Default for TensorStore {
    fn default() -> Self {
        Self::with_backend(Arc::new(CpuBackend))
    }
}

impl TensorStore {
    pub fn with_backend(backend: Arc<dyn Backend>) -> Self {
        Self {
            tensors: Vec::new(),
            free_ids: Vec::new(),
            backend,
            checkpoint_offload_pool: CheckpointOffloadPool::default(),
            checkpoint_l3: CheckpointL3::from_env(),
            tape_dtype: TapeDtype::F32,
        }
    }

    pub fn set_tape_dtype(&mut self, dtype: TapeDtype) {
        self.tape_dtype = dtype;
        self.backend.set_tape_dtype(dtype);
    }

    pub fn tape_dtype(&self) -> TapeDtype {
        self.tape_dtype
    }

    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    pub fn set_backend(&mut self, backend: Arc<dyn Backend>) {
        self.backend = backend;
    }

    fn discard_checkpoint_copy(&mut self, id: TensorId) -> Result<()> {
        let (residency, bytes) = {
            let tensor = self.tensor(id)?;
            (
                tensor.checkpoint_residency,
                tensor.data.len() * std::mem::size_of::<f32>(),
            )
        };
        if let CheckpointResidency::Pinned(slot) = residency {
            self.backend().checkpoint_pin_release(slot);
        }
        if let Some(l3) = &mut self.checkpoint_l3 {
            match residency {
                CheckpointResidency::Host => {
                    l3.host_bytes = l3.host_bytes.saturating_sub(bytes);
                }
                CheckpointResidency::L3 => l3.remove(id),
                CheckpointResidency::None | CheckpointResidency::Pinned(_) => {}
            }
        }
        self.raw_tensor_mut(id)?.checkpoint_residency = CheckpointResidency::None;
        Ok(())
    }

    pub fn alloc(&mut self, tensor: Tensor) -> TensorId {
        if let Some(id) = self.free_ids.pop() {
            self.tensors[id] = Some(tensor);
            id
        } else {
            let id = self.tensors.len();
            self.tensors.push(Some(tensor));
            id
        }
    }

    pub fn free(&mut self, id: TensorId) -> Result<()> {
        let tensor = {
            let slot = self
                .tensors
                .get_mut(id)
                .ok_or(AutogradError::InvalidTensorId(id))?;
            slot.take().ok_or(AutogradError::InvalidTensorId(id))?
        };
        match tensor.checkpoint_residency {
            CheckpointResidency::Host => {
                if let Some(l3) = &mut self.checkpoint_l3 {
                    l3.host_bytes = l3
                        .host_bytes
                        .saturating_sub(tensor.data.len() * std::mem::size_of::<f32>());
                }
                self.checkpoint_offload_pool.recycle(tensor.data);
            }
            CheckpointResidency::L3 => {
                if let Some(l3) = &mut self.checkpoint_l3 {
                    l3.remove(id);
                }
            }
            CheckpointResidency::Pinned(slot) => self.backend.checkpoint_pin_release(slot),
            CheckpointResidency::None => {}
        }
        self.free_ids.push(id);
        Ok(())
    }

    pub fn live_ids(&self) -> Vec<TensorId> {
        self.tensors
            .iter()
            .enumerate()
            .filter_map(|(id, slot)| slot.as_ref().map(|_| id))
            .collect()
    }

    pub fn live_tensor_count(&self) -> usize {
        self.tensors.iter().filter(|slot| slot.is_some()).count()
    }

    pub fn live_host_bytes(&self) -> usize {
        self.tensors
            .iter()
            .filter_map(Option::as_ref)
            .map(|tensor| tensor.data.len() * std::mem::size_of::<f32>())
            .sum()
    }

    pub fn free_new_except(
        &mut self,
        live_before: &HashSet<TensorId>,
        keep: &HashSet<TensorId>,
    ) -> Result<usize> {
        let to_free = self
            .live_ids()
            .into_iter()
            .filter(|id| !live_before.contains(id) && !keep.contains(id))
            .collect::<Vec<_>>();
        let freed = to_free.len();
        for id in to_free {
            self.free(id)?;
        }
        Ok(freed)
    }

    pub fn retain_ids(&mut self, keep: &HashSet<TensorId>) {
        for (id, slot) in self.tensors.iter_mut().enumerate() {
            if keep.contains(&id) || slot.is_none() {
                continue;
            }
            if let Some(tensor) = slot.take() {
                match tensor.checkpoint_residency {
                    CheckpointResidency::Host => {
                        if let Some(l3) = &mut self.checkpoint_l3 {
                            l3.host_bytes = l3
                                .host_bytes
                                .saturating_sub(tensor.data.len() * std::mem::size_of::<f32>());
                        }
                        self.checkpoint_offload_pool.recycle(tensor.data);
                    }
                    CheckpointResidency::L3 => {
                        if let Some(l3) = &mut self.checkpoint_l3 {
                            l3.remove(id);
                        }
                    }
                    CheckpointResidency::Pinned(slot) => {
                        self.backend.checkpoint_pin_release(slot);
                    }
                    CheckpointResidency::None => {}
                }
            }
            self.free_ids.push(id);
        }
    }

    pub fn get(&self, id: TensorId) -> Option<&Tensor> {
        self.tensors.get(id).and_then(Option::as_ref)
    }

    pub fn get_mut(&mut self, id: TensorId) -> Option<&mut Tensor> {
        if self
            .tensors
            .get(id)
            .and_then(Option::as_ref)
            .is_some_and(|tensor| {
                tensor.dirty == Dirty::Device
                    || matches!(
                        tensor.checkpoint_residency,
                        CheckpointResidency::L3 | CheckpointResidency::Pinned(_)
                    )
            })
        {
            self.ensure_host(id)
                .expect("ensure_host before mutable tensor access");
        }

        let checkpoint_host_bytes = self
            .tensors
            .get(id)
            .and_then(Option::as_ref)
            .filter(|tensor| tensor.checkpoint_residency == CheckpointResidency::Host)
            .map_or(0, |tensor| tensor.data.len() * std::mem::size_of::<f32>());
        if let Some(l3) = &mut self.checkpoint_l3 {
            l3.host_bytes = l3.host_bytes.saturating_sub(checkpoint_host_bytes);
        }
        let tensor = self.tensors.get_mut(id).and_then(Option::as_mut)?;
        tensor.device_handle = None;
        tensor.dirty = Dirty::Host;
        tensor.checkpoint_residency = CheckpointResidency::None;
        Some(tensor)
    }

    pub fn from_slice(&mut self, data: &[f32], shape: &[usize]) -> Result<TensorId> {
        let tensor = Tensor::new(data.to_vec(), shape.to_vec(), false)?;
        Ok(self.alloc(tensor))
    }

    pub fn ensure_host(&mut self, id: TensorId) -> Result<()> {
        if let CheckpointResidency::Pinned(slot) = self.tensor(id)?.checkpoint_residency {
            let size = self.tensor(id)?.size;
            let mut host = self.checkpoint_offload_pool.take(size);
            self.backend().checkpoint_pin_readback(slot, &mut host)?;
            let tensor = self.raw_tensor_mut(id)?;
            tensor.data = host;
            tensor.dirty = Dirty::Host;
            tensor.checkpoint_residency = CheckpointResidency::Host;
            return Ok(());
        }
        if self.tensor(id)?.checkpoint_residency == CheckpointResidency::L3 {
            let expected_len = self.tensor(id)?.size;
            let host = self
                .checkpoint_l3
                .as_mut()
                .ok_or(AutogradError::TapeInvariant(
                    "L3 checkpoint missing storage backend",
                ))?
                .restore(id, expected_len)?;
            let bytes = host.len() * std::mem::size_of::<f32>();
            let tensor = self.raw_tensor_mut(id)?;
            tensor.data = host;
            tensor.dirty = Dirty::Host;
            tensor.checkpoint_residency = CheckpointResidency::Host;
            self.checkpoint_l3
                .as_mut()
                .expect("storage backend checked above")
                .host_bytes += bytes;
            return Ok(());
        }
        if self.tensor(id)?.dirty != Dirty::Device {
            return Ok(());
        }

        let handle = self
            .tensor(id)?
            .device_handle
            .as_ref()
            .ok_or(AutogradError::TapeInvariant(
                "device-resident tensor missing device handle",
            ))?
            .clone();
        self.backend().eval(&[&handle])?;
        let host = self.backend().readback(&handle)?;
        let tensor = self.raw_tensor_mut(id)?;
        tensor.data = host;
        tensor.dirty = Dirty::Both;
        tensor.checkpoint_residency = CheckpointResidency::None;
        Ok(())
    }

    /// Flush a batch of `Dirty::Device` tensors to host using **one**
    /// backend `eval` call, then per-id `readback`. Equivalent to calling
    /// `ensure_host` for each id, but collapses N FFI eval boundaries
    /// into 1 — meaningful on Metal where each `mlx_eval` round-trip
    /// dominates at small shapes (see `tape::backward` flush loop).
    /// Tensors not currently `Dirty::Device` are silently skipped, so the
    /// caller can pass the entire output set without pre-filtering.
    pub fn flush_to_host_batch(&mut self, ids: &[TensorId]) -> Result<()> {
        let mut to_flush: Vec<(TensorId, DeviceHandle)> = Vec::with_capacity(ids.len());
        for &id in ids {
            let tensor = self.tensor(id)?;
            if tensor.dirty != Dirty::Device {
                continue;
            }
            let handle = tensor
                .device_handle
                .as_ref()
                .ok_or(AutogradError::TapeInvariant(
                    "device-resident tensor missing device handle",
                ))?
                .clone();
            to_flush.push((id, handle));
        }
        if to_flush.is_empty() {
            return Ok(());
        }
        let handle_refs: Vec<&DeviceHandle> = to_flush.iter().map(|(_, h)| h).collect();
        self.backend().eval(&handle_refs)?;
        for (id, handle) in to_flush {
            let host = self.backend().readback(&handle)?;
            let tensor = self.raw_tensor_mut(id)?;
            tensor.data = host;
            tensor.dirty = Dirty::Both;
            tensor.checkpoint_residency = CheckpointResidency::None;
        }
        Ok(())
    }

    pub fn ensure_device(&mut self, id: TensorId) -> Result<()> {
        if let CheckpointResidency::Pinned(slot) = self.tensor(id)?.checkpoint_residency {
            let shape = self.tensor(id)?.shape.clone();
            let handle = self.backend().checkpoint_pin_reload(slot, &shape)?;
            let tensor = self.raw_tensor_mut(id)?;
            tensor.device_handle = Some(handle);
            tensor.dirty = Dirty::Device;
            tensor.checkpoint_residency = CheckpointResidency::None;
            return Ok(());
        }
        let was_l3 = self.tensor(id)?.checkpoint_residency == CheckpointResidency::L3;
        if was_l3 {
            self.ensure_host(id)?;
        }
        let (dirty, has_handle) = {
            let tensor = self.tensor(id)?;
            (tensor.dirty.clone(), tensor.device_handle.is_some())
        };

        if dirty == Dirty::Device && !has_handle {
            return Err(AutogradError::TapeInvariant(
                "device-resident tensor missing device handle",
            ));
        }

        if has_handle && dirty != Dirty::Host {
            return Ok(());
        }

        let (shape, checkpoint_residency) = {
            let tensor = self.tensor(id)?;
            (tensor.shape.clone(), tensor.checkpoint_residency)
        };
        let handle = {
            let tensor = self.tensor(id)?;
            self.backend().upload(&tensor.data, &shape)?
        };
        let recycled = if checkpoint_residency == CheckpointResidency::Host {
            let tensor = self.raw_tensor_mut(id)?;
            std::mem::take(&mut tensor.data)
        } else {
            Vec::new()
        };
        if checkpoint_residency == CheckpointResidency::Host {
            if let Some(l3) = &mut self.checkpoint_l3 {
                l3.host_bytes = l3
                    .host_bytes
                    .saturating_sub(recycled.len() * std::mem::size_of::<f32>());
            }
            if !was_l3 {
                self.checkpoint_offload_pool.recycle(recycled);
            }
        }
        let tensor = self.raw_tensor_mut(id)?;
        tensor.device_handle = Some(handle);
        tensor.dirty = if checkpoint_residency == CheckpointResidency::Host {
            Dirty::Device
        } else {
            Dirty::Both
        };
        tensor.checkpoint_residency = CheckpointResidency::None;
        Ok(())
    }

    pub(crate) fn checkpoint_parked(&self, id: TensorId) -> Result<bool> {
        Ok(!matches!(
            self.tensor(id)?.checkpoint_residency,
            CheckpointResidency::None
        ))
    }

    /// Restore an offloaded checkpoint ahead of its backward replay. L3/Pinned
    /// unconditionally (the payload is not in `data`); `Host` only under
    /// `--checkpoint-reload-device`, since the replay ops otherwise fall to host
    /// `Vec` paths and re-upload a repacked copy instead of one HtoD up front.
    pub(crate) fn ensure_checkpoint_device(&mut self, id: TensorId) -> Result<()> {
        let reload = match self.tensor(id)?.checkpoint_residency {
            CheckpointResidency::L3 | CheckpointResidency::Pinned(_) => true,
            CheckpointResidency::Host => crate::runtime_flags::checkpoint_reload_device(),
            CheckpointResidency::None => false,
        };
        if reload {
            self.ensure_device(id)?;
        }
        Ok(())
    }

    pub fn to_host(&mut self, id: TensorId) -> Result<Vec<f32>> {
        self.ensure_host(id)?;
        Ok(self.tensor(id)?.data.clone())
    }

    pub fn alloc_device_tensor(
        &mut self,
        shape: Vec<usize>,
        handle: DeviceHandle,
    ) -> Result<TensorId> {
        let tensor = Tensor {
            data: Vec::new(),
            shape: shape.clone(),
            strides: contiguous_strides(&shape),
            size: shape_size(&shape),
            requires_grad: false,
            grad: None,
            device_handle: Some(handle),
            dirty: Dirty::Device,
            checkpoint_residency: CheckpointResidency::None,
        };
        Ok(self.alloc(tensor))
    }

    /// Replace an existing tensor's device handle with a fresh one, marking
    /// the host copy stale (`Dirty::Device`) and clearing the cached host
    /// `data` buffer.
    ///
    /// Used by the device-backed AdamW path (M5.3b.10): the optimizer
    /// produces a new `DeviceHandle` for each updated parameter and we must
    /// install it without going through `get_mut` — `get_mut` auto-triggers
    /// `ensure_host`, which would re-download the old pre-update values and
    /// then mark the tensor `Dirty::Host`, forcing a full re-upload on the
    /// next forward pass. That is the exact churn this path exists to kill.
    pub fn replace_device_handle(&mut self, id: TensorId, handle: DeviceHandle) -> Result<()> {
        self.discard_checkpoint_copy(id)?;
        let tensor = self.raw_tensor_mut(id)?;
        tensor.device_handle = Some(handle);
        tensor.dirty = Dirty::Device;
        tensor.data.clear();
        tensor.checkpoint_residency = CheckpointResidency::None;
        Ok(())
    }

    /// Re-store a frozen leaf as bf16 device-resident, converting the host f32
    /// data to bf16 bits on the CPU before upload. Used by w2s to reload the
    /// 27B base weights (54 GB in bf16) without materializing the 108 GB f32
    /// copy on GPU.
    pub fn upload_frozen_bf16_from_host(&mut self, id: TensorId) -> Result<()> {
        if self.backend().device() != Device::Cuda {
            self.ensure_device(id)?;
            return Ok(());
        }
        let (data, shape, dirty, has_handle, residency) = {
            let tensor = self.tensor(id)?;
            (
                tensor.data.clone(),
                tensor.shape.clone(),
                tensor.dirty.clone(),
                tensor.device_handle.is_some(),
                tensor.checkpoint_residency,
            )
        };
        if data.is_empty() {
            eprintln!(
                "[upload_frozen_bf16_from_host] id={id} shape={shape:?} data_len=0 \
                 dirty={dirty:?} has_handle={has_handle} residency={residency:?}"
            );
        }
        let bf16_bits: Vec<u16> = data
            .iter()
            .map(|&f| crate::backend::f32_to_bf16_bits(f))
            .collect();
        let handle = self.backend().upload_bf16_bits(&bf16_bits, &shape)?;
        self.replace_device_handle(id, handle)
    }

    /// Re-store a frozen leaf as bf16 device-resident (`--tape-precision bf16`).
    /// No-op unless the flag is set and the backend is CUDA. Frozen leaves only:
    /// the bf16 handle drops the f32 host mirror, so a later backward needing the
    /// exact f32 value would read the widened (lossy) copy.
    pub fn quantize_frozen_bf16(&mut self, id: TensorId) -> Result<()> {
        if !crate::runtime_flags::tape_bf16() || self.backend().device() != Device::Cuda {
            return Ok(());
        }
        self.ensure_device(id)?;
        let (handle, shape) = {
            let tensor = self.tensor(id)?;
            let handle = tensor
                .device_handle
                .as_ref()
                .ok_or(AutogradError::TapeInvariant(
                    "quantize_frozen_bf16: tensor has no device handle",
                ))?
                .clone();
            (handle, tensor.shape.clone())
        };
        let bf16 = self.backend().quantize_frozen_to_bf16(&handle, &shape)?;
        self.replace_device_handle(id, bf16)
    }

    /// Drop the cached host mirror for a tensor after ensuring a device handle
    /// exists. This keeps model weights device-authoritative on memory-tight
    /// CUDA hosts; callers that later need host data can still use `to_host`.
    pub fn evict_host_mirror(&mut self, id: TensorId) -> Result<usize> {
        if matches!(
            self.tensor(id)?.checkpoint_residency,
            CheckpointResidency::L3 | CheckpointResidency::Pinned(_)
        ) {
            self.ensure_host(id)?;
        }
        self.discard_checkpoint_copy(id)?;
        let (dirty, has_handle) = {
            let tensor = self.tensor(id)?;
            (tensor.dirty.clone(), tensor.device_handle.is_some())
        };
        if dirty == Dirty::Host || !has_handle {
            let handle = {
                let tensor = self.tensor(id)?;
                let handle = self.backend().upload(&tensor.data, &tensor.shape)?;
                self.backend().eval(&[&handle])?;
                handle
            };
            self.raw_tensor_mut(id)?.device_handle = Some(handle);
        }
        let tensor = self.raw_tensor_mut(id)?;
        if tensor.device_handle.is_none() {
            return Err(AutogradError::TapeInvariant(
                "evict_host_mirror: tensor has no device handle after upload",
            ));
        }
        let evicted_bytes = tensor.data.capacity() * std::mem::size_of::<f32>();
        tensor.data = Vec::new();
        tensor.dirty = Dirty::Device;
        tensor.checkpoint_residency = CheckpointResidency::None;
        Ok(evicted_bytes)
    }

    /// Inverse of [`Self::evict_host_mirror`]: ensure the host copy is current,
    /// then drop the device handle to free VRAM. The tensor becomes host-only
    /// (`Dirty::Host`); a later `ensure_device` (e.g. checkpoint backward replay)
    /// re-uploads it. Used to offload grad-checkpoints to host RAM during a long
    /// training forward so they don't pin ~30 GB of VRAM. Returns bytes freed.
    pub fn offload_to_host(&mut self, id: TensorId) -> Result<usize> {
        let (size, dirty, has_handle, residency) = {
            let t = self.tensor(id)?;
            (
                t.size,
                t.dirty.clone(),
                t.device_handle.is_some(),
                t.checkpoint_residency,
            )
        };
        // Fast path: a live device buffer read back into a pooled host buffer,
        // reused across parks instead of a fresh Vec each call (#191). The
        // buffer returns to the pool on the next `ensure_device` (residency ==
        // Host branch there). Sole caller — chunk-accumulator park — is always
        // device-resident and never L3, so the fallback below is defensive.
        if has_handle && dirty == Dirty::Device && residency == CheckpointResidency::None {
            let handle = self
                .tensor(id)?
                .device_handle
                .as_ref()
                .expect("has_handle")
                .clone();
            self.backend().eval(&[&handle])?;
            let mut host = self.checkpoint_offload_pool.take(size);
            self.backend().readback_into(&handle, &mut host)?;
            let bytes = size * std::mem::size_of::<f32>();
            let old = {
                let t = self.raw_tensor_mut(id)?;
                t.device_handle = None;
                t.dirty = Dirty::Host;
                t.checkpoint_residency = CheckpointResidency::Host;
                std::mem::replace(&mut t.data, host)
            };
            if let Some(l3) = &mut self.checkpoint_l3 {
                l3.host_bytes = l3.host_bytes.saturating_add(bytes);
            }
            self.checkpoint_offload_pool.recycle(old);
            return Ok(bytes);
        }
        self.ensure_host(id)?;
        self.discard_checkpoint_copy(id)?;
        let tensor = self.raw_tensor_mut(id)?;
        let freed = if tensor.device_handle.is_some() {
            tensor.size * std::mem::size_of::<f32>()
        } else {
            0
        };
        tensor.device_handle = None;
        tensor.dirty = Dirty::Host;
        tensor.checkpoint_residency = CheckpointResidency::None;
        Ok(freed)
    }

    /// Checkpoint-only host offload with a reusable host buffer and a minimum
    /// tensor size threshold. Returns device bytes freed.
    pub fn offload_checkpoint_to_host(&mut self, id: TensorId) -> Result<usize> {
        let min_bytes = crate::runtime_flags::checkpoint_offload_min_bytes();
        let (size, bytes, handle) = {
            let tensor = self.tensor(id)?;
            let bytes = tensor.size.saturating_mul(std::mem::size_of::<f32>());
            if bytes < min_bytes
                || tensor.checkpoint_residency != CheckpointResidency::None
                || tensor.dirty == Dirty::Host
                || tensor.device_handle.is_none()
            {
                return Ok(0);
            }
            (
                tensor.size,
                bytes,
                tensor
                    .device_handle
                    .as_ref()
                    .expect("checked above")
                    .clone(),
            )
        };

        // Pinned first (unless L3 is configured — the two spill mechanisms never mix).
        if self.checkpoint_l3.is_none()
            && let Some(slot) = self.backend().checkpoint_pin_offload(&handle, size)?
        {
            let old_host = {
                let tensor = self.raw_tensor_mut(id)?;
                tensor.device_handle = None;
                tensor.dirty = Dirty::Host;
                tensor.checkpoint_residency = CheckpointResidency::Pinned(slot);
                std::mem::take(&mut tensor.data)
            };
            self.checkpoint_offload_pool.recycle(old_host);
            return Ok(bytes);
        }

        let mut host = self.checkpoint_offload_pool.take(size);
        self.backend().eval(&[&handle])?;
        self.backend().readback_into(&handle, &mut host)?;
        let spilled = self
            .checkpoint_l3
            .as_mut()
            .is_some_and(|l3| l3.should_spill(bytes) && l3.insert(id, &host));
        if spilled {
            let tensor = self.raw_tensor_mut(id)?;
            tensor.device_handle = None;
            tensor.data = Vec::new();
            tensor.dirty = Dirty::Host;
            tensor.checkpoint_residency = CheckpointResidency::L3;
            return Ok(bytes);
        }
        let old_host = {
            let tensor = self.raw_tensor_mut(id)?;
            tensor.device_handle = None;
            tensor.dirty = Dirty::Host;
            tensor.checkpoint_residency = CheckpointResidency::Host;
            std::mem::replace(&mut tensor.data, host)
        };
        if let Some(l3) = &mut self.checkpoint_l3 {
            l3.host_bytes = l3.host_bytes.saturating_add(bytes);
        }
        self.checkpoint_offload_pool.recycle(old_host);
        Ok(bytes)
    }

    /// Drop a tensor's DEVICE residency (free VRAM) WITHOUT a host readback,
    /// when the value is provably dead — used by `checkpoint` to reclaim a frozen
    /// group's input hidden, which has no backward replay and is referenced by
    /// nothing once the group's output exists. Unlike `offload_to_host` this does
    /// NOT copy to host (no PCIe cost): a frozen-group hidden is never read again.
    /// The slot becomes `Dirty::Host` with empty `data`, so any erroneous later
    /// access fails loud through the missing-data guards rather than reading
    /// stale memory. Returns bytes freed. No-op if already host-resident.
    pub fn drop_device_residency(&mut self, id: TensorId) -> Result<usize> {
        self.discard_checkpoint_copy(id)?;
        let tensor = self.raw_tensor_mut(id)?;
        let freed = if tensor.device_handle.is_some() {
            tensor.size * std::mem::size_of::<f32>()
        } else {
            0
        };
        tensor.device_handle = None;
        tensor.dirty = Dirty::Host;
        tensor.checkpoint_residency = CheckpointResidency::None;
        Ok(freed)
    }

    /// Flip a tensor's `requires_grad` flag in place (host-side metadata only;
    /// does not touch device residency). Used to freeze a parameter — e.g. the
    /// SOPD EMA adapter, which is a frozen teacher param updated host-side, not
    /// through autograd.
    pub fn set_requires_grad(&mut self, id: TensorId, requires_grad: bool) -> Result<()> {
        self.raw_tensor_mut(id)?.requires_grad = requires_grad;
        Ok(())
    }

    /// An op's output requires grad iff any input does. Freed inputs read as
    /// false, matching a checkpoint replay whose saved set is already reclaimed.
    pub fn any_requires_grad(&self, ids: &[TensorId]) -> bool {
        ids.iter()
            .any(|&id| self.get(id).is_some_and(|tensor| tensor.requires_grad))
    }

    pub fn detach(&mut self, id: TensorId) -> Result<TensorId> {
        let detached = self.clone_tensor(id)?;
        self.set_requires_grad(detached, false)?;
        self.set_grad(detached, None)?;
        Ok(detached)
    }

    pub(crate) fn set_grad(&mut self, id: TensorId, grad: Option<TensorId>) -> Result<()> {
        self.raw_tensor_mut(id)?.grad = grad;
        Ok(())
    }

    fn raw_tensor_mut(&mut self, id: TensorId) -> Result<&mut Tensor> {
        self.tensors
            .get_mut(id)
            .and_then(Option::as_mut)
            .ok_or(AutogradError::InvalidTensorId(id))
    }

    pub fn zeros_like(&mut self, id: TensorId) -> Result<TensorId> {
        let source = self.tensor(id)?;
        let tensor = Tensor::new(vec![0.0; source.size], source.shape.clone(), false)?;
        Ok(self.alloc(tensor))
    }

    pub fn accumulate_grad(&mut self, param_id: TensorId, grad_id: TensorId) -> Result<()> {
        self.accumulate_grad_inner(param_id, grad_id, false)
    }

    pub(crate) fn accumulate_grad_owned(
        &mut self,
        param_id: TensorId,
        grad_id: TensorId,
    ) -> Result<()> {
        self.accumulate_grad_inner(param_id, grad_id, true)
    }

    fn accumulate_grad_inner(
        &mut self,
        param_id: TensorId,
        grad_id: TensorId,
        owned: bool,
    ) -> Result<()> {
        let (requires_grad, shape, existing_grad) = {
            let tensor = self.tensor(param_id)?;
            (tensor.requires_grad, tensor.shape.clone(), tensor.grad)
        };
        if !requires_grad {
            return Ok(());
        }

        let grad_shape = self.tensor(grad_id)?.shape.clone();
        if shape != grad_shape {
            return Err(AutogradError::GradientShapeMismatch {
                tensor_id: param_id,
                expected: shape,
                got: grad_shape,
            });
        }

        match existing_grad {
            Some(existing_id) => {
                // `owned` only claims no lifetime alias (from `keep_in_grads`),
                // not that the device buffer is unshared — grads fan out by Arc
                // clone (clone_tensor / add_backward), so in-place would corrupt
                // a live sibling. Probe the store-held handle's strong-count
                // BEFORE the clone below (the clone bumps it); `Some(1)` proves
                // sole ownership. Mirrors the tape.rs accumulate path.
                let (both_on_device, existing_uniquely_owned) = {
                    let existing = self.tensor(existing_id)?;
                    let incoming = self.tensor(grad_id)?;
                    let both = existing.dirty != Dirty::Host
                        && existing.device_handle.is_some()
                        && incoming.dirty != Dirty::Host
                        && incoming.device_handle.is_some();
                    let unique = existing
                        .device_handle
                        .as_ref()
                        .and_then(DeviceHandle::device_buffer_strong_count)
                        == Some(1);
                    (both, unique)
                };
                if both_on_device {
                    let existing_handle = self
                        .tensor(existing_id)?
                        .device_handle
                        .as_ref()
                        .expect("checked above")
                        .clone();
                    let incoming_handle = self
                        .tensor(grad_id)?
                        .device_handle
                        .as_ref()
                        .expect("checked above")
                        .clone();
                    let sum_handle = if owned && existing_uniquely_owned {
                        self.backend().accumulate_into_device(
                            &existing_handle,
                            &incoming_handle,
                            &shape,
                        )?
                    } else {
                        self.backend().add_into_device(
                            &existing_handle,
                            &incoming_handle,
                            &shape,
                        )?
                    };
                    self.replace_device_handle(existing_id, sum_handle)?;
                } else {
                    let incoming = self.to_host(grad_id)?;
                    let existing = self
                        .get_mut(existing_id)
                        .ok_or(AutogradError::InvalidTensorId(existing_id))?;
                    for (dst, src) in existing.data.iter_mut().zip(incoming) {
                        *dst += src;
                    }
                }
            }
            None => {
                let cloned_grad_id = self.clone_tensor(grad_id)?;
                self.set_grad(param_id, Some(cloned_grad_id))?;
            }
        }

        Ok(())
    }

    pub(crate) fn tensor(&self, id: TensorId) -> Result<&Tensor> {
        self.get(id).ok_or(AutogradError::InvalidTensorId(id))
    }

    pub(crate) fn tensor_mut(&mut self, id: TensorId) -> Result<&mut Tensor> {
        self.get_mut(id).ok_or(AutogradError::InvalidTensorId(id))
    }

    /// P3.1: Read a tensor by id with `ensure_host` applied first, then
    /// return an owned clone. This is the "I want host data" version
    /// that op-backward host fallbacks should call instead of
    /// `store.tensor(x)?.clone()`. Pre-P3.1, every `tensor(x)?.clone()`
    /// was safe because `Tape::backward`'s `flush_to_host_batch` made
    /// every tape output `Dirty::Both` before any backward op ran. With
    /// the flush gone on CUDA, host-only backward ops have to be
    /// explicit about their host residency requirement.
    pub fn tensor_host(&mut self, id: TensorId) -> Result<Tensor> {
        self.ensure_host(id)?;
        Ok(self.tensor(id)?.clone())
    }

    /// The device-side `tensor_host`. Bumps the refcount — a caller that needs the
    /// pre-clone count must read `tensor` directly.
    pub fn device_handle(&mut self, id: TensorId) -> Result<DeviceHandle> {
        self.ensure_device(id)?;
        Ok(self
            .tensor(id)?
            .device_handle
            .as_ref()
            .expect("ensure_device")
            .clone())
    }

    pub(crate) fn clone_tensor(&mut self, id: TensorId) -> Result<TensorId> {
        // Wave 1 (post-M5.3b nsys attribution): preserve the device
        // handle on `Dirty::Device` tensors so the post-backward grad map
        // doesn't force a host readback for tensors that subsequent
        // backward ops will consume on-device. A plain `ensure_host` here
        // would, on the `[B, S, V] ≈ 1 GB`
        // log_softmax grad, trigger the same readback the pre-backward
        // flush used to. The device-aware backward overrides on `CudaBackend`
        // (and the dispatchers in `ops::softmax::log_softmax_backward` /
        // `ops::gather::gather_last_dim_backward`) keep the chain
        // device-resident as long as the grad never gets touched through
        // a host-only path.
        //
        // `DeviceHandle` is `Arc`-shared (`CudaStorage`'s `Arc<CudaSlice<f32>>`,
        // `MlxHandle`'s `Arc<MlxHandleInner>`), so cloning the handle is a
        // ref-count bump — no extra device allocation. Grads are write-once
        // (each backward emits a fresh handle), so aliasing the storage is
        // sound: nothing in the autograd graph mutates a tape entry's
        // output buffer in place.
        let dirty = self.tensor(id)?.dirty.clone();
        if dirty == Dirty::Device {
            let source = self.tensor(id)?;
            let device_handle = source.device_handle.clone();
            let cloned = Tensor {
                data: Vec::new(),
                shape: source.shape.clone(),
                strides: source.strides.clone(),
                size: source.size,
                requires_grad: source.requires_grad,
                grad: source.grad,
                device_handle,
                dirty: Dirty::Device,
                checkpoint_residency: CheckpointResidency::None,
            };
            return Ok(self.alloc(cloned));
        }

        self.ensure_host(id)?;
        let tensor = self.tensor(id)?.clone();
        Ok(self.alloc(tensor))
    }

    pub(crate) fn fill_like(&mut self, id: TensorId, value: f32) -> Result<TensorId> {
        let source = self.tensor(id)?;
        let tensor = Tensor::new(vec![value; source.size], source.shape.clone(), false)?;
        Ok(self.alloc(tensor))
    }
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for (index, dim) in shape.iter().enumerate().rev() {
        strides[index] = stride;
        stride *= *dim;
    }
    strides
}

fn shape_size(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}
