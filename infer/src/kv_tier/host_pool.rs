//! Host pinned pool for the tiered KV cache T1 tier.
//!
//! The backing storage lives in `kv-native-sys` as an allocation-stable host
//! arena. Rust keeps the existing safe wrapper shape (`HostPinnedRegion`,
//! `SharedHostPinnedPool`) and, on the CUDA lane, pins the arena once with
//! `cuMemHostRegister_v2` so scheduler/coordinator call sites do not need to
//! change.

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

    /// Acquire the inner pool guard.
    ///
    /// **Poison policy: propagate as `anyhow!`, do not auto-recover.**
    ///
    /// A poisoned mutex means a previous holder panicked mid-operation, which
    /// for the host pinned pool can leave the arena's free-list / live-set
    /// in an inconsistent state (e.g. a `reserve` that bumped the bump pointer
    /// but never inserted into `live_regions`, or a `release` that ran the
    /// native call but never updated the Rust-side mirror). Continuing past
    /// that with `into_inner` risks double-frees or silent metadata drift, so
    /// the data-plane API surfaces poisoning as a hard error.
    ///
    /// Some callers (notably `coordinator.rs`) currently use
    /// `PoisonError::into_inner` to recover at scheduler boundaries; that is a
    /// deliberate trade-off in the coordinator's bookkeeping path (the
    /// coordinator can fail-fast the offending ticket without corrupting the
    /// pool itself) and is **not** the policy for direct pool I/O. If you need
    /// to soft-recover here, do it at the call site after demonstrating that
    /// the protected invariants are still intact.
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

    /// Run `f` over a borrowed slice of the region while the pool lock is
    /// held. Avoids the `Vec<u8>` allocation that [`Self::read_region`]
    /// requires for callers that only need to consume the bytes.
    ///
    /// **Caution:** `f` runs while the pool mutex is held. Do **not** perform
    /// blocking I/O (disk writes, network send, etc.) inside `f` — that
    /// stalls every other reservation/release on the same pool. Use this only
    /// for short, CPU-bound consumers (e.g. `memcpy_htod`, hashing,
    /// `Arc::from(slice)`). For blocking consumers, fall back to
    /// [`Self::read_region`] and pay the one-shot allocation.
    pub fn with_region_slice<R>(
        &self,
        region: HostPinnedRegion,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<R> {
        let pool = self.lock()?;
        let offset = pool.validate_live_region(region)?;
        let slice =
            unsafe { std::slice::from_raw_parts(pool.arena.base_ptr.add(offset), region.len) };
        Ok(f(slice))
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
/// `kv-native-sys`. Rust exposes typed regions and slice views over that single
/// stable base pointer.
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

    pub fn reserved_bytes(&self) -> Result<usize> {
        unsafe { kv_native_sys::host_arena_reserved_bytes(self.arena.handle.cast_const()) }
            .map_err(|err| anyhow!("HostPinnedPool reserved_bytes native query failed: {err}"))
    }

    pub fn remaining_bytes(&self) -> Result<usize> {
        Ok(self.capacity_bytes.saturating_sub(self.reserved_bytes()?))
    }

    pub fn reserve(&mut self, len: usize) -> Result<Option<HostPinnedRegion>> {
        let region = unsafe { kv_native_sys::host_arena_reserve(self.arena.handle, len) }
            .map_err(|err| anyhow!("HostPinnedPool reserve native query failed: {err}"))?;
        let Some(region) = region else {
            return Ok(None);
        };
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
        Ok(Some(region))
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
        let a = pool.reserve(128).unwrap().unwrap();
        let b = pool.reserve(256).unwrap().unwrap();
        assert_eq!(a.offset, 0);
        assert_eq!(a.len, 128);
        assert_eq!(b.offset, 128);
        assert_eq!(b.len, 256);
        assert_eq!(pool.reserved_bytes().unwrap(), 384);
    }

    #[test]
    fn release_and_reuse_via_free_list_first_fit() {
        let mut pool = HostPinnedPool::new(1024).unwrap();
        let a = pool.reserve(128).unwrap().unwrap();
        let _b = pool.reserve(128).unwrap().unwrap();
        pool.release(a).unwrap();
        let c = pool.reserve(128).unwrap().unwrap();
        assert_eq!(c.offset, 0);
        assert_eq!(pool.reserved_bytes().unwrap(), 256);
    }

    #[test]
    fn release_splits_larger_free_hole_on_reuse() {
        let mut pool = HostPinnedPool::new(1024).unwrap();
        let a = pool.reserve(256).unwrap().unwrap();
        pool.release(a).unwrap();
        let b = pool.reserve(64).unwrap().unwrap();
        assert_eq!(b.offset, 0);
        assert_eq!(b.len, 64);
        let c = pool.reserve(192).unwrap().unwrap();
        assert_eq!(c.offset, 64);
        assert_eq!(c.len, 192);
    }

    #[test]
    fn reserve_returns_none_on_exhaustion() {
        let mut pool = HostPinnedPool::new(128).unwrap();
        assert!(pool.reserve(128).unwrap().is_some());
        assert!(pool.reserve(1).unwrap().is_none());
    }

    #[test]
    fn reset_rewinds_allocator_and_clears_free_list() {
        let mut pool = HostPinnedPool::new(1024).unwrap();
        let a = pool.reserve(128).unwrap().unwrap();
        pool.release(a).unwrap();
        pool.reserve(64).unwrap().unwrap();
        pool.reset().unwrap();
        let region = pool.reserve(128).unwrap().unwrap();
        assert_eq!(region.offset, 0);
        assert_eq!(pool.reserved_bytes().unwrap(), 128);
    }

    #[test]
    fn as_slice_round_trips_bytes_via_mut_write() {
        let mut pool = HostPinnedPool::new(64).unwrap();
        let region = pool.reserve(32).unwrap().unwrap();
        pool.as_mut_slice(region).copy_from_slice(&[7u8; 32]);
        assert_eq!(pool.as_slice(region), &[7u8; 32]);
    }

    #[test]
    fn host_ptr_stays_stable_across_subsequent_reservations() {
        let mut pool = HostPinnedPool::new(1024).unwrap();
        let a = pool.reserve(128).unwrap().unwrap();
        let ptr_before = pool.host_ptr(a);
        let _b = pool.reserve(128).unwrap().unwrap();
        let ptr_after = pool.host_ptr(a);
        assert_eq!(ptr_before, ptr_after);
    }

    #[test]
    fn release_rejects_double_free() {
        let mut pool = HostPinnedPool::new(256).unwrap();
        let region = pool.reserve(64).unwrap().unwrap();
        pool.release(region).unwrap();

        let err = pool.release(region).unwrap_err();
        assert!(err.to_string().contains("not live"));
    }

    #[test]
    fn reset_invalidates_old_regions() {
        let mut pool = HostPinnedPool::new(256).unwrap();
        let region = pool.reserve(64).unwrap().unwrap();
        pool.reset().unwrap();

        let err = pool.release(region).unwrap_err();
        assert!(err.to_string().contains("not live"));
    }

    #[test]
    fn shared_pool_read_rejects_released_region() {
        let shared = SharedHostPinnedPool::new(HostPinnedPool::new(256).unwrap());
        let region = {
            let mut pool = shared.lock().unwrap();
            let region = pool.reserve(64).unwrap().unwrap();
            pool.release(region).unwrap();
            region
        };

        let err = shared.read_region(region).unwrap_err();
        assert!(err.to_string().contains("not live"));
    }
}
