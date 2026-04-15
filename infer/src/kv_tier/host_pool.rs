//! Host pinned pool skeleton for the tiered KV cache M3a transport path.
//!
//! This file intentionally ships only the allocation-stable bookkeeping that is
//! locally verifiable on macOS / no-cuda. Real `cudaHostAlloc` ownership and
//! async copy validation land on the remote CUDA lane.

use anyhow::{Result, anyhow};

/// Reservation returned by [`HostPinnedPool::reserve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostPinnedRegion {
    pub offset: u64,
    pub len: usize,
}

/// Allocation-stable host pool descriptor.
///
/// The eventual CUDA implementation will back this with one `cudaHostAlloc`
/// region that never moves. The local lane only verifies the layout and
/// reservation arithmetic.
#[derive(Debug)]
pub struct HostPinnedPool {
    capacity_bytes: usize,
    next_offset: usize,
}

impl HostPinnedPool {
    pub fn new(capacity_bytes: usize) -> Result<Self> {
        if capacity_bytes == 0 {
            return Err(anyhow!("HostPinnedPool capacity must be > 0"));
        }
        Ok(Self {
            capacity_bytes,
            next_offset: 0,
        })
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub fn reserved_bytes(&self) -> usize {
        self.next_offset
    }

    pub fn remaining_bytes(&self) -> usize {
        self.capacity_bytes.saturating_sub(self.next_offset)
    }

    pub fn reserve(&mut self, len: usize) -> Option<HostPinnedRegion> {
        if len == 0 || len > self.remaining_bytes() {
            return None;
        }
        let region = HostPinnedRegion {
            offset: self.next_offset as u64,
            len,
        };
        self.next_offset += len;
        Some(region)
    }

    pub fn reset(&mut self) {
        self.next_offset = 0;
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
        assert_eq!(b.offset, 128);
        assert_eq!(pool.reserved_bytes(), 384);
        assert_eq!(pool.remaining_bytes(), 640);
    }

    #[test]
    fn reset_rewinds_allocator_to_the_front() {
        let mut pool = HostPinnedPool::new(1024).unwrap();
        pool.reserve(512).unwrap();
        pool.reset();
        let region = pool.reserve(128).unwrap();
        assert_eq!(region.offset, 0);
        assert_eq!(pool.reserved_bytes(), 128);
    }
}
