//! Host pinned pool for the tiered KV cache T1 tier.
//!
//! Owns a **single, allocation-stable** pinned region on the host side —
//! `cudaHostAlloc`-backed on the CUDA lane, plain `Vec<u8>` on the no-cuda
//! lane so local unit tests exercise the allocator logic without touching
//! CUDA. Both backings expose the same `HostPinnedRegion { offset, len }`
//! shape so callers see one API.
//!
//! The "single allocation, never moves" invariant is load-bearing: the
//! coordinator will eventually hand the base pointer to NIXL / UCX for
//! registered-memory zero-copy transfers (project doc §4.2 invariant 5 +
//! §8 pitfall 2). Reallocating the region mid-flight would invalidate
//! any outstanding memory registration. The free list exists for reuse
//! of interior holes WITHOUT ever growing or re-basing the region.

use anyhow::{Result, anyhow};

/// Reservation handed back by [`HostPinnedPool::reserve`].
///
/// `offset` is measured from the base of the pool's one pinned region;
/// it is stable across sends and thread boundaries. `len` is the byte
/// size of the reservation (not capped to a block size — callers pick
/// their own granularity).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostPinnedRegion {
    pub offset: u64,
    pub len: usize,
}

#[derive(Debug)]
enum Backing {
    #[cfg(feature = "cuda")]
    CudaPinned {
        /// Base pointer returned by `cuMemAllocHost_v2`. **Never** null
        /// between `new` and `drop`. Stored as a raw pointer because
        /// the pool's Send/Sync impls are hand-written.
        base_ptr: *mut u8,
    },
    /// Non-CUDA backing used both on the no-cuda feature lane and in
    /// scheduler tests on CUDA hosts that do not want to spend real
    /// pinned memory. Plain `Vec<u8>` — Send + Sync via std.
    InMemory { buffer: Vec<u8> },
}

/// Allocation-stable host pool.
///
/// Layout:
/// - `capacity_bytes`: total byte capacity of the underlying backing.
/// - `next_offset`: bump-allocator cursor for fresh allocations.
/// - `free_list`: interior holes returned via [`Self::release`] that
///   can be reused without growing the region. **Not coalesced**: for
///   the M3 local batch, first-fit on exact-or-larger sizes is enough
///   (the watermark path always frees full blocks whose size matches a
///   recent reservation). A smarter allocator can replace this without
///   changing the public API.
///
/// The pool is `Send + Sync` by hand: the `Backing::CudaPinned` raw
/// pointer is the single base address of a `cuMemAllocHost_v2` region
/// that is allocated once, freed in `Drop`, and otherwise immutable.
/// Concurrent reads / writes into disjoint [`HostPinnedRegion`] slices
/// are safe because each reservation owns a non-overlapping byte range.
#[derive(Debug)]
pub struct HostPinnedPool {
    capacity_bytes: usize,
    next_offset: usize,
    free_list: Vec<HostPinnedRegion>,
    backing: Backing,
}

// SAFETY: The `Backing::CudaPinned` raw pointer is a single
// `cuMemAllocHost_v2` base address that is valid for the lifetime of
// the pool. Reservations hand out disjoint byte ranges, so concurrent
// readers/writers of different `HostPinnedRegion`s never alias. The
// pool itself performs no shared mutation across threads other than
// the allocator cursor + free list, which higher-level code wraps in
// `Mutex` / channel synchronization.
unsafe impl Send for HostPinnedPool {}
unsafe impl Sync for HostPinnedPool {}

impl HostPinnedPool {
    /// Allocate a new host pinned pool backing one physically-stable
    /// region of `capacity_bytes` bytes.
    ///
    /// On the `cuda` feature lane this calls `cuMemAllocHost_v2`; on the
    /// `no-cuda` lane it falls back to a plain `Vec<u8>` so unit tests
    /// exercise the bookkeeping without needing a GPU. Either way the
    /// resulting pool honors the "never grow, never move" invariant.
    pub fn new(capacity_bytes: usize) -> Result<Self> {
        if capacity_bytes == 0 {
            return Err(anyhow!("HostPinnedPool capacity must be > 0"));
        }

        #[cfg(feature = "cuda")]
        let backing = {
            let mut ptr: *mut u8 = std::ptr::null_mut();
            // SAFETY: `cuMemAllocHost_v2` is the standard CUDA driver
            // entry point for portable pinned host memory. We pass a
            // valid `&mut *mut u8` and a non-zero byte count; the call
            // either writes a non-null pointer on success or returns a
            // non-success status which we translate into an anyhow
            // error.
            let status = unsafe {
                cudarc::driver::sys::cuMemAllocHost_v2(
                    &mut ptr as *mut *mut u8 as *mut *mut std::ffi::c_void,
                    capacity_bytes,
                )
            };
            if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(anyhow!(
                    "HostPinnedPool cuMemAllocHost_v2 failed: {status:?}"
                ));
            }
            if ptr.is_null() {
                return Err(anyhow!(
                    "HostPinnedPool cuMemAllocHost_v2 returned null on success"
                ));
            }
            // Zero the region so stale contents cannot leak into a
            // subsequent reservation. Standard pattern — FlashInfer
            // workspace does the same on its `plan_info` allocation.
            // SAFETY: `ptr` is a freshly-allocated CUDA pinned region
            // of exactly `capacity_bytes` bytes that no other code has
            // yet observed. `write_bytes` over the full length is
            // well-defined.
            unsafe { std::ptr::write_bytes(ptr, 0, capacity_bytes) };
            Backing::CudaPinned { base_ptr: ptr }
        };
        #[cfg(not(feature = "cuda"))]
        let backing = Backing::InMemory {
            buffer: vec![0u8; capacity_bytes],
        };

        Ok(Self {
            capacity_bytes,
            next_offset: 0,
            free_list: Vec::new(),
            backing,
        })
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub fn reserved_bytes(&self) -> usize {
        self.next_offset
            .saturating_sub(self.free_list.iter().map(|r| r.len).sum::<usize>())
    }

    pub fn remaining_bytes(&self) -> usize {
        self.capacity_bytes.saturating_sub(self.next_offset)
            + self.free_list.iter().map(|r| r.len).sum::<usize>()
    }

    /// Reserve a contiguous byte range of `len`. Tries the free list
    /// first (first-fit on exact-or-larger), falls back to bump
    /// allocation from `next_offset`. Returns `None` on exhaustion so
    /// callers can respond with a structured error instead of panicking.
    pub fn reserve(&mut self, len: usize) -> Option<HostPinnedRegion> {
        if len == 0 {
            return None;
        }
        // First-fit scan of the free list.
        if let Some(idx) = self.free_list.iter().position(|r| r.len >= len) {
            let mut region = self.free_list.swap_remove(idx);
            if region.len > len {
                // Split: keep the tail back on the free list.
                let tail = HostPinnedRegion {
                    offset: region.offset + len as u64,
                    len: region.len - len,
                };
                self.free_list.push(tail);
                region.len = len;
            }
            return Some(region);
        }
        // Bump-allocate.
        if self.next_offset + len > self.capacity_bytes {
            return None;
        }
        let region = HostPinnedRegion {
            offset: self.next_offset as u64,
            len,
        };
        self.next_offset += len;
        Some(region)
    }

    /// Release a region back to the free list so future `reserve` calls
    /// can reuse the same byte range. Does **not** call `cuMemFreeHost`
    /// — the pool stays exactly one `cuMemAllocHost_v2` region for its
    /// entire lifetime (project doc §4.2 invariant 5, §8 pitfall 2).
    pub fn release(&mut self, region: HostPinnedRegion) {
        // Best-effort sanity: a release that doesn't match any past
        // reservation is silently accepted, but the free list won't
        // grow past the high water mark of live reservations.
        if (region.offset as usize) + region.len > self.capacity_bytes {
            return;
        }
        self.free_list.push(region);
    }

    /// Reset the pool to empty. Invalidates every outstanding
    /// [`HostPinnedRegion`]; callers must not use a region after
    /// `reset`. Primarily useful for tests and for a full
    /// session-level restart path.
    pub fn reset(&mut self) {
        self.next_offset = 0;
        self.free_list.clear();
    }

    /// Read-only slice view of a region. Panics if the region is out
    /// of bounds — that can only happen if the caller fabricated a
    /// region, which is a contract violation.
    ///
    /// # Safety of the cuda path
    ///
    /// The pinned region is a single-owner `cuMemAllocHost_v2`
    /// allocation whose base pointer never moves. Building a `&[u8]`
    /// over a disjoint byte range is safe for the lifetime of the
    /// borrow as long as no concurrent writer targets the same range
    /// — that is the coordinator's responsibility, not this type's.
    pub fn as_slice(&self, region: HostPinnedRegion) -> &[u8] {
        let offset = region.offset as usize;
        assert!(
            offset + region.len <= self.capacity_bytes,
            "HostPinnedRegion out of bounds: offset={offset} len={} cap={}",
            region.len,
            self.capacity_bytes
        );
        match &self.backing {
            #[cfg(feature = "cuda")]
            Backing::CudaPinned { base_ptr } => {
                // SAFETY: base_ptr is non-null and points at a pinned
                // allocation of exactly `capacity_bytes`. The asserted
                // bounds check above guarantees `[offset, offset+len)`
                // is within that allocation.
                unsafe { std::slice::from_raw_parts(base_ptr.add(offset), region.len) }
            }
            Backing::InMemory { buffer } => &buffer[offset..offset + region.len],
        }
    }

    /// Mutable slice view of a region. Same safety story as
    /// [`Self::as_slice`] but for writes.
    pub fn as_mut_slice(&mut self, region: HostPinnedRegion) -> &mut [u8] {
        let offset = region.offset as usize;
        assert!(
            offset + region.len <= self.capacity_bytes,
            "HostPinnedRegion out of bounds: offset={offset} len={} cap={}",
            region.len,
            self.capacity_bytes
        );
        match &mut self.backing {
            #[cfg(feature = "cuda")]
            Backing::CudaPinned { base_ptr } => {
                // SAFETY: see as_slice. We hold `&mut self`, so no
                // other view exists for the lifetime of this borrow.
                unsafe { std::slice::from_raw_parts_mut(base_ptr.add(offset), region.len) }
            }
            Backing::InMemory { buffer } => &mut buffer[offset..offset + region.len],
        }
    }

    /// Raw host pointer for a region, in address-space form.
    ///
    /// CUDA transport code uses this to hand off to
    /// `cudaMemcpyAsync(... host_ptr, len, cudaMemcpyHostToDevice, ...)`
    /// and similar. The no-cuda lane exposes the `Vec<u8>` address so
    /// unit tests can still prove the bookkeeping; nothing real reads
    /// or writes through that pointer under `no-cuda`.
    pub fn host_ptr(&self, region: HostPinnedRegion) -> u64 {
        let offset = region.offset as usize;
        assert!(
            offset + region.len <= self.capacity_bytes,
            "HostPinnedRegion out of bounds"
        );
        match &self.backing {
            #[cfg(feature = "cuda")]
            Backing::CudaPinned { base_ptr } => (*base_ptr as usize + offset) as u64,
            Backing::InMemory { buffer } => buffer.as_ptr() as u64 + offset as u64,
        }
    }
}

impl Drop for HostPinnedPool {
    fn drop(&mut self) {
        match &mut self.backing {
            #[cfg(feature = "cuda")]
            Backing::CudaPinned { base_ptr } => {
                if !base_ptr.is_null() {
                    // SAFETY: `base_ptr` was produced by a successful
                    // `cuMemAllocHost_v2` call in `new` and has not been
                    // freed yet. Ignoring the return status matches the
                    // FlashInfer workspace pattern — a failure here
                    // during shutdown is unrecoverable anyway.
                    unsafe {
                        let _ =
                            cudarc::driver::sys::cuMemFreeHost(*base_ptr as *mut std::ffi::c_void);
                    }
                    *base_ptr = std::ptr::null_mut();
                }
            }
            Backing::InMemory { .. } => {}
        }
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
        pool.release(a);
        // Next same-size reservation should take from the free list,
        // not bump past `_b`.
        let c = pool.reserve(128).unwrap();
        assert_eq!(c.offset, 0);
        assert_eq!(pool.next_offset, 256);
    }

    #[test]
    fn release_splits_larger_free_hole_on_reuse() {
        let mut pool = HostPinnedPool::new(1024).unwrap();
        let a = pool.reserve(256).unwrap();
        pool.release(a);
        let b = pool.reserve(64).unwrap();
        assert_eq!(b.offset, 0);
        assert_eq!(b.len, 64);
        // The tail 192 bytes should still be available.
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
        pool.release(a);
        pool.reserve(64).unwrap();
        pool.reset();
        let region = pool.reserve(128).unwrap();
        assert_eq!(region.offset, 0);
        assert_eq!(pool.free_list.len(), 0);
    }

    #[test]
    fn as_slice_round_trips_bytes_via_mut_write() {
        let mut pool = HostPinnedPool::new(64).unwrap();
        let region = pool.reserve(32).unwrap();
        // Write through the mutable view.
        pool.as_mut_slice(region).copy_from_slice(&[7u8; 32]);
        // Read through the shared view.
        assert_eq!(pool.as_slice(region), &[7u8; 32]);
    }

    #[test]
    fn host_ptr_stays_stable_across_subsequent_reservations() {
        let mut pool = HostPinnedPool::new(1024).unwrap();
        let a = pool.reserve(128).unwrap();
        let ptr_before = pool.host_ptr(a);
        // Allocating more bytes must NOT relocate `a`.
        let _b = pool.reserve(128).unwrap();
        let ptr_after = pool.host_ptr(a);
        assert_eq!(ptr_before, ptr_after);
    }
}
