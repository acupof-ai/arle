//! Host pinned pool for the tiered KV cache T1 tier.
//!
//! The backing storage now lives in `kv-native-sys` as a Zig-managed,
//! allocation-stable host arena. Rust keeps the existing safe wrapper shape
//! (`HostPinnedRegion`, `SharedHostPinnedPool`) and, on the CUDA lane, pins the
//! arena once with `cuMemHostRegister_v2` so scheduler/coordinator call sites do
//! not need to change.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use anyhow::{Result, anyhow};

/// Reservation handed back by [`HostPinnedPool::reserve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HostPinnedRegion {
    pub offset: u64,
    pub len: usize,
}

/// Cloneable handle for sharing a single allocation-stable host pinned pool
/// across the scheduler and coordinator.
#[derive(Clone, Debug)]
pub struct SharedHostPinnedPool {
    inner: Arc<Mutex<HostPinnedPool>>,
}

impl SharedHostPinnedPool {
    pub fn new(pool: HostPinnedPool) -> Self {
        Self {
            inner: Arc::new(Mutex::new(pool)),
        }
    }

    pub fn from_arc(inner: Arc<Mutex<HostPinnedPool>>) -> Self {
        Self { inner }
    }

    pub fn as_arc(&self) -> Arc<Mutex<HostPinnedPool>> {
        Arc::clone(&self.inner)
    }

    pub fn lock(&self) -> Result<MutexGuard<'_, HostPinnedPool>> {
        self.inner
            .lock()
            .map_err(|err: PoisonError<MutexGuard<'_, HostPinnedPool>>| {
                anyhow!("SharedHostPinnedPool poisoned: {err}")
            })
    }

    pub fn read_region(&self, region: HostPinnedRegion) -> Result<Vec<u8>> {
        let pool = self.lock()?;
        let offset = pool.validate_live_region(region)?;
        Ok(
            unsafe { std::slice::from_raw_parts(pool.arena.base_ptr.add(offset), region.len) }
                .to_vec(),
        )
    }

    pub fn write_region(&self, region: HostPinnedRegion, bytes: &[u8]) -> Result<()> {
        if bytes.len() != region.len {
            return Err(anyhow!(
                "SharedHostPinnedPool region length mismatch: region={} bytes={}",
                region.len,
                bytes.len()
            ));
        }
        let pool = self.lock()?;
        let offset = pool.validate_live_region(region)?;
        unsafe { std::slice::from_raw_parts_mut(pool.arena.base_ptr.add(offset), region.len) }
            .copy_from_slice(bytes);
        Ok(())
    }

    pub fn release_region(&self, region: HostPinnedRegion) -> Result<()> {
        let mut pool = self.lock()?;
        pool.release(region)
    }
}

impl PartialEq for SharedHostPinnedPool {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for SharedHostPinnedPool {}

#[derive(Debug)]
struct NativeArena {
    handle: *mut kv_native_sys::KvHostArenaHandle,
    base_ptr: *mut u8,
    #[cfg(feature = "cuda")]
    cuda_registered: bool,
}

/// Allocation-stable host pool.
///
/// The arena is created once, never moves, and is internally sub-allocated by
/// the Zig substrate. Rust exposes typed regions and slice views over that
/// single stable base pointer.
#[derive(Debug)]
pub struct HostPinnedPool {
    capacity_bytes: usize,
    arena: NativeArena,
    live_regions: HashSet<HostPinnedRegion>,
}

unsafe impl Send for HostPinnedPool {}
unsafe impl Sync for HostPinnedPool {}

impl HostPinnedPool {
    pub fn new(capacity_bytes: usize) -> Result<Self> {
        if capacity_bytes == 0 {
            return Err(anyhow!("HostPinnedPool capacity must be > 0"));
        }

        let (handle, base_ptr) = kv_native_sys::host_arena_create(capacity_bytes)
            .map_err(|err| anyhow!("HostPinnedPool native arena create failed: {err}"))?;

        #[cfg(feature = "cuda")]
        let cuda_registered = {
            let status = unsafe {
                cudarc::driver::sys::cuMemHostRegister_v2(
                    base_ptr.cast::<std::ffi::c_void>(),
                    capacity_bytes,
                    0,
                )
            };
            if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                let _ = unsafe { kv_native_sys::host_arena_destroy(handle) };
                return Err(anyhow!(
                    "HostPinnedPool cuMemHostRegister_v2 failed: {status:?}"
                ));
            }
            true
        };

        Ok(Self {
            capacity_bytes,
            arena: NativeArena {
                handle,
                base_ptr,
                #[cfg(feature = "cuda")]
                cuda_registered,
            },
            live_regions: HashSet::new(),
        })
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub fn reserved_bytes(&self) -> usize {
        unsafe { kv_native_sys::host_arena_reserved_bytes(self.arena.handle.cast_const()) }
            .expect("HostPinnedPool reserved_bytes native query must succeed")
    }

    pub fn remaining_bytes(&self) -> usize {
        self.capacity_bytes.saturating_sub(self.reserved_bytes())
    }

    pub fn reserve(&mut self, len: usize) -> Option<HostPinnedRegion> {
        let region = unsafe { kv_native_sys::host_arena_reserve(self.arena.handle, len) }
            .expect("HostPinnedPool reserve native query must succeed")?;
        let region = HostPinnedRegion {
            offset: region.offset,
            len: region.len,
        };
        assert!(
            self.live_regions.insert(region),
            "HostPinnedPool reserve returned a duplicate live region: offset={} len={}",
            region.offset,
            region.len
        );
        Some(region)
    }

    pub fn release(&mut self, region: HostPinnedRegion) -> Result<()> {
        self.validate_live_region(region)?;
        unsafe {
            kv_native_sys::host_arena_release(
                self.arena.handle,
                kv_native_sys::KvHostArenaRegion {
                    offset: region.offset,
                    len: region.len,
                },
            )
        }
        .map_err(|err| {
            anyhow!(
                "HostPinnedPool release native call failed for offset={} len={}: {err}",
                region.offset,
                region.len
            )
        })?;
        let removed = self.live_regions.remove(&region);
        debug_assert!(removed, "validated live region disappeared before removal");
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        unsafe { kv_native_sys::host_arena_reset(self.arena.handle) }
            .map_err(|err| anyhow!("HostPinnedPool reset native call failed: {err}"))?;
        self.live_regions.clear();
        Ok(())
    }

    pub fn as_slice(&self, region: HostPinnedRegion) -> &[u8] {
        let offset = self
            .validate_live_region(region)
            .expect("HostPinnedPool as_slice requires a live region");
        unsafe { std::slice::from_raw_parts(self.arena.base_ptr.add(offset), region.len) }
    }

    pub fn as_mut_slice(&mut self, region: HostPinnedRegion) -> &mut [u8] {
        let offset = self
            .validate_live_region(region)
            .expect("HostPinnedPool as_mut_slice requires a live region");
        unsafe { std::slice::from_raw_parts_mut(self.arena.base_ptr.add(offset), region.len) }
    }

    pub fn host_ptr(&self, region: HostPinnedRegion) -> u64 {
        let offset = self
            .validate_live_region(region)
            .expect("HostPinnedPool host_ptr requires a live region");
        (self.arena.base_ptr as usize + offset) as u64
    }

    fn validate_live_region(&self, region: HostPinnedRegion) -> Result<usize> {
        let offset = region.offset as usize;
        let region_end = offset.checked_add(region.len).ok_or_else(|| {
            anyhow!(
                "HostPinnedRegion overflow: offset={} len={}",
                region.offset,
                region.len
            )
        })?;
        if region_end > self.capacity_bytes {
            return Err(anyhow!(
                "HostPinnedRegion out of bounds: offset={} len={} cap={}",
                region.offset,
                region.len,
                self.capacity_bytes
            ));
        }
        if !self.live_regions.contains(&region) {
            return Err(anyhow!(
                "HostPinnedRegion is not live in this pool: offset={} len={}",
                region.offset,
                region.len
            ));
        }
        Ok(offset)
    }
}

impl Drop for HostPinnedPool {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if self.arena.cuda_registered && !self.arena.base_ptr.is_null() {
            unsafe {
                let _ = cudarc::driver::sys::cuMemHostUnregister(
                    self.arena.base_ptr.cast::<std::ffi::c_void>(),
                );
            }
        }

        let _ = unsafe { kv_native_sys::host_arena_destroy(self.arena.handle) };
        self.arena.handle = std::ptr::null_mut();
        self.arena.base_ptr = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_advances_offset_without_reallocating() {
        let mut pool = HostPinnedPool::new(1024).unwrap();
        let a = pool.reserve(128).unwrap();
        let b = pool.reserve(256).unwrap();
        assert_eq!(a.offset, 0);
        assert_eq!(a.len, 128);
        assert_eq!(b.offset, 128);
        assert_eq!(b.len, 256);
        assert_eq!(pool.reserved_bytes(), 384);
    }

    #[test]
    fn release_and_reuse_via_free_list_first_fit() {
        let mut pool = HostPinnedPool::new(1024).unwrap();
        let a = pool.reserve(128).unwrap();
        let _b = pool.reserve(128).unwrap();
        pool.release(a).unwrap();
        let c = pool.reserve(128).unwrap();
        assert_eq!(c.offset, 0);
        assert_eq!(pool.reserved_bytes(), 256);
    }

    #[test]
    fn release_splits_larger_free_hole_on_reuse() {
        let mut pool = HostPinnedPool::new(1024).unwrap();
        let a = pool.reserve(256).unwrap();
        pool.release(a).unwrap();
        let b = pool.reserve(64).unwrap();
        assert_eq!(b.offset, 0);
        assert_eq!(b.len, 64);
        let c = pool.reserve(192).unwrap();
        assert_eq!(c.offset, 64);
        assert_eq!(c.len, 192);
    }

    #[test]
    fn reserve_returns_none_on_exhaustion() {
        let mut pool = HostPinnedPool::new(128).unwrap();
        assert!(pool.reserve(128).is_some());
        assert!(pool.reserve(1).is_none());
    }

    #[test]
    fn reset_rewinds_allocator_and_clears_free_list() {
        let mut pool = HostPinnedPool::new(1024).unwrap();
        let a = pool.reserve(128).unwrap();
        pool.release(a).unwrap();
        pool.reserve(64).unwrap();
        pool.reset().unwrap();
        let region = pool.reserve(128).unwrap();
        assert_eq!(region.offset, 0);
        assert_eq!(pool.reserved_bytes(), 128);
    }

    #[test]
    fn as_slice_round_trips_bytes_via_mut_write() {
        let mut pool = HostPinnedPool::new(64).unwrap();
        let region = pool.reserve(32).unwrap();
        pool.as_mut_slice(region).copy_from_slice(&[7u8; 32]);
        assert_eq!(pool.as_slice(region), &[7u8; 32]);
    }

    #[test]
    fn host_ptr_stays_stable_across_subsequent_reservations() {
        let mut pool = HostPinnedPool::new(1024).unwrap();
        let a = pool.reserve(128).unwrap();
        let ptr_before = pool.host_ptr(a);
        let _b = pool.reserve(128).unwrap();
        let ptr_after = pool.host_ptr(a);
        assert_eq!(ptr_before, ptr_after);
    }

    #[test]
    fn release_rejects_double_free() {
        let mut pool = HostPinnedPool::new(256).unwrap();
        let region = pool.reserve(64).unwrap();
        pool.release(region).unwrap();

        let err = pool.release(region).unwrap_err();
        assert!(err.to_string().contains("not live"));
    }

    #[test]
    fn reset_invalidates_old_regions() {
        let mut pool = HostPinnedPool::new(256).unwrap();
        let region = pool.reserve(64).unwrap();
        pool.reset().unwrap();

        let err = pool.release(region).unwrap_err();
        assert!(err.to_string().contains("not live"));
    }

    #[test]
    fn shared_pool_read_rejects_released_region() {
        let shared = SharedHostPinnedPool::new(HostPinnedPool::new(256).unwrap());
        let region = {
            let mut pool = shared.lock().unwrap();
            let region = pool.reserve(64).unwrap();
            pool.release(region).unwrap();
            region
        };

        let err = shared.read_region(region).unwrap_err();
        assert!(err.to_string().contains("not live"));
    }
}
