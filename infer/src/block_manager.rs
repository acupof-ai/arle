//! Batch-scheduler-local paged KV block accounting helper.
//!
//! # Overview
//!
//! This module is used by the standalone CPU accounting scheduler in
//! [`crate::scheduler::batch`]. It tracks fixed-size KV blocks and can
//! describe swap plans, but it is **not** the production multilayer KV path.
//! The serving/runtime path uses:
//! - [`crate::prefix_cache::RadixCache`] for block metadata and reuse
//! - [`crate::kv_tier`] for tier transitions
//! - `crates/cuda-kernels/src/paged_kv.rs` for the live CUDA pool
//!
//! # Block lifecycle
//!
//! ```text
//! allocate_gpu(n) → Vec<BlockId>      // reserve n GPU blocks
//! swap_out(blocks)                    // GPU → CPU plan
//! swap_in(blocks)                     // CPU → GPU plan
//! free_gpu(blocks)                    // return GPU blocks to free list
//! free_cpu(blocks)                    // return CPU blocks to free list
//! ```
//!
//! The manager is pure accounting — it tracks which blocks are free or
//! occupied but does NOT perform the actual memory copies.
//!
//! # Shared-block accounting
//!
//! The batch scheduler can pin the same block table entries across multiple
//! requests. This module tracks refcounts, but it does not own any production
//! decode-time COW path.

use std::collections::{HashMap, VecDeque};

// ============================================================================
// Types
// ============================================================================

/// Physical block identifier.
///
/// Re-exported from [`crate::types::BlockId`] — the canonical single
/// `u32` type after the 2026-04-15 tiered-kv-cache revision. See
/// `docs/projects/tiered-kv-cache.md` §5.1 for the rationale.
pub use crate::types::BlockId;

/// Where a block resides.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockLocation {
    Gpu,
    Cpu,
}

/// A single managed KV block.
#[derive(Debug)]
struct Block {
    #[allow(dead_code)]
    id: BlockId,
    location: BlockLocation,
    /// Number of requests (or cache nodes) currently pinning this block.
    ref_count: u32,
}

/// Result of a swap operation requested by the block manager.
#[derive(Debug)]
pub struct SwapPlan {
    /// Blocks to copy from GPU → CPU.
    pub gpu_to_cpu: Vec<(BlockId, BlockId)>, // (src_gpu_block, dst_cpu_block)
    /// Blocks to copy from CPU → GPU.
    pub cpu_to_gpu: Vec<(BlockId, BlockId)>, // (src_cpu_block, dst_gpu_block)
}

// ============================================================================
// BlockManager
// ============================================================================

/// Manages allocation and lifecycle of paged KV cache blocks.
pub struct BlockManager {
    blocks: HashMap<BlockId, Block>,
    free_gpu: VecDeque<BlockId>,
    free_cpu: VecDeque<BlockId>,
    block_size: usize, // tokens per block
    #[allow(dead_code)]
    next_id: u32,
}

impl BlockManager {
    /// Create a new block manager.
    ///
    /// - `num_gpu_blocks`: total GPU KV blocks.
    /// - `num_cpu_blocks`: total CPU KV blocks (swap buffer).
    /// - `block_size`: tokens per block (must match the radix cache and kernel).
    pub fn new(num_gpu_blocks: usize, num_cpu_blocks: usize, block_size: usize) -> Self {
        assert!(block_size > 0);
        let mut blocks = HashMap::new();
        let mut free_gpu = VecDeque::new();
        let mut free_cpu = VecDeque::new();
        let mut next_id = 0u32;

        for _ in 0..num_gpu_blocks {
            let id = BlockId(next_id);
            next_id += 1;
            blocks.insert(
                id,
                Block {
                    id,
                    location: BlockLocation::Gpu,
                    ref_count: 0,
                },
            );
            free_gpu.push_back(id);
        }

        for _ in 0..num_cpu_blocks {
            let id = BlockId(next_id);
            next_id += 1;
            blocks.insert(
                id,
                Block {
                    id,
                    location: BlockLocation::Cpu,
                    ref_count: 0,
                },
            );
            free_cpu.push_back(id);
        }

        Self {
            blocks,
            free_gpu,
            free_cpu,
            block_size,
            next_id,
        }
    }

    // -----------------------------------------------------------------------
    // Metrics
    // -----------------------------------------------------------------------

    /// Number of free GPU blocks.
    pub fn free_gpu_blocks(&self) -> usize {
        self.free_gpu.len()
    }

    /// Number of free CPU blocks.
    pub fn free_cpu_blocks(&self) -> usize {
        self.free_cpu.len()
    }

    /// Total GPU blocks (free + allocated).
    pub fn total_gpu_blocks(&self) -> usize {
        self.blocks
            .values()
            .filter(|b| b.location == BlockLocation::Gpu)
            .count()
    }

    /// Total CPU blocks (free + allocated).
    pub fn total_cpu_blocks(&self) -> usize {
        self.blocks
            .values()
            .filter(|b| b.location == BlockLocation::Cpu)
            .count()
    }

    /// Block size (tokens per block).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// GPU memory utilization in [0.0, 1.0].
    pub fn gpu_utilization(&self) -> f32 {
        let total = self.total_gpu_blocks();
        if total == 0 {
            return 0.0;
        }
        let used = total - self.free_gpu_blocks();
        used as f32 / total as f32
    }

    // -----------------------------------------------------------------------
    // Allocation
    // -----------------------------------------------------------------------

    /// Allocate `n` GPU blocks. Returns `Err` if insufficient free GPU blocks.
    pub fn allocate_gpu(&mut self, n: usize) -> Result<Vec<BlockId>, AllocationError> {
        if self.free_gpu.len() < n {
            return Err(AllocationError::InsufficientGpuBlocks {
                requested: n,
                available: self.free_gpu.len(),
            });
        }

        let mut allocated = Vec::with_capacity(n);
        for _ in 0..n {
            let id = self.free_gpu.pop_front().expect("checked above");
            self.blocks
                .get_mut(&id)
                .expect("block must exist")
                .ref_count = 1;
            allocated.push(id);
        }
        Ok(allocated)
    }

    /// Allocate `n` CPU blocks. Returns `Err` if insufficient free CPU blocks.
    pub fn allocate_cpu(&mut self, n: usize) -> Result<Vec<BlockId>, AllocationError> {
        if self.free_cpu.len() < n {
            return Err(AllocationError::InsufficientCpuBlocks {
                requested: n,
                available: self.free_cpu.len(),
            });
        }

        let mut allocated = Vec::with_capacity(n);
        for _ in 0..n {
            let id = self.free_cpu.pop_front().expect("checked above");
            self.blocks
                .get_mut(&id)
                .expect("block must exist")
                .ref_count = 1;
            allocated.push(id);
        }
        Ok(allocated)
    }

    // -----------------------------------------------------------------------
    // Free
    // -----------------------------------------------------------------------

    /// Decrement ref_count for each block. Blocks with ref_count == 0 after
    /// decrement are returned to the appropriate free list.
    pub fn free(&mut self, blocks: &[BlockId]) {
        for &id in blocks {
            let block = self.blocks.get_mut(&id).expect("block must be managed");
            if block.ref_count > 0 {
                block.ref_count -= 1;
            }
            if block.ref_count == 0 {
                match block.location {
                    BlockLocation::Gpu => self.free_gpu.push_back(id),
                    BlockLocation::Cpu => self.free_cpu.push_back(id),
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Reference counting
    // -----------------------------------------------------------------------

    /// Increment ref_count for the given blocks (e.g. when sharing with prefix cache).
    pub fn pin(&mut self, blocks: &[BlockId]) {
        for &id in blocks {
            self.blocks
                .get_mut(&id)
                .expect("block must be managed")
                .ref_count += 1;
        }
    }

    /// Get the current ref_count for a block.
    pub fn ref_count(&self, id: BlockId) -> u32 {
        self.blocks.get(&id).map_or(0, |b| b.ref_count)
    }

    // -----------------------------------------------------------------------
    // Swap (preemption support)
    // -----------------------------------------------------------------------

    /// Plan a swap-out for the given GPU blocks (GPU → CPU).
    ///
    /// Allocates CPU blocks and returns a `SwapPlan`. The caller schedules
    /// the actual memory copy, then calls `commit_swap_out` to finalize.
    ///
    /// Returns `Err` if insufficient CPU blocks.
    pub fn plan_swap_out(&mut self, gpu_blocks: &[BlockId]) -> Result<SwapPlan, AllocationError> {
        let cpu_blocks = self.allocate_cpu(gpu_blocks.len())?;
        let plan = SwapPlan {
            gpu_to_cpu: gpu_blocks
                .iter()
                .copied()
                .zip(cpu_blocks.iter().copied())
                .collect(),
            cpu_to_gpu: vec![],
        };
        Ok(plan)
    }

    /// Commit a completed swap-out: move block metadata from GPU to CPU.
    ///
    /// After this call, the CPU blocks in `plan.gpu_to_cpu` are "hot" and
    /// the GPU blocks are freed.
    pub fn commit_swap_out(&mut self, plan: &SwapPlan) {
        for &(gpu_id, cpu_id) in &plan.gpu_to_cpu {
            // Mark GPU block as free.
            {
                let gpu_block = self.blocks.get_mut(&gpu_id).expect("gpu block must exist");
                gpu_block.ref_count = 0;
            }
            self.free_gpu.push_back(gpu_id);

            // CPU block inherits the ref_count from the GPU block (1 from allocate_cpu).
            let _ = cpu_id; // already allocated via allocate_cpu above
        }
    }

    /// Plan a swap-in for the given CPU blocks (CPU → GPU).
    ///
    /// Returns `Err` if insufficient GPU blocks.
    pub fn plan_swap_in(&mut self, cpu_blocks: &[BlockId]) -> Result<SwapPlan, AllocationError> {
        let gpu_blocks = self.allocate_gpu(cpu_blocks.len())?;
        let plan = SwapPlan {
            gpu_to_cpu: vec![],
            cpu_to_gpu: cpu_blocks
                .iter()
                .copied()
                .zip(gpu_blocks.iter().copied())
                .collect(),
        };
        Ok(plan)
    }

    /// Commit a completed swap-in: move block metadata from CPU to GPU.
    pub fn commit_swap_in(&mut self, plan: &SwapPlan) {
        for &(cpu_id, gpu_id) in &plan.cpu_to_gpu {
            // Free the CPU block.
            {
                let cpu_block = self.blocks.get_mut(&cpu_id).expect("cpu block must exist");
                cpu_block.ref_count = 0;
            }
            self.free_cpu.push_back(cpu_id);

            // GPU block is already marked allocated by allocate_gpu.
            let _ = gpu_id;
        }
    }

    // -----------------------------------------------------------------------
    // How many blocks are needed for a given token count
    // -----------------------------------------------------------------------

    /// Number of blocks needed to hold `num_tokens` tokens.
    pub fn blocks_for_tokens(&self, num_tokens: usize) -> usize {
        num_tokens.div_ceil(self.block_size)
    }
}

// ============================================================================
// Error type
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocationError {
    InsufficientGpuBlocks { requested: usize, available: usize },
    InsufficientCpuBlocks { requested: usize, available: usize },
}

impl std::fmt::Display for AllocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientGpuBlocks {
                requested,
                available,
            } => {
                write!(f, "OOM: need {requested} GPU blocks, only {available} free")
            }
            Self::InsufficientCpuBlocks {
                requested,
                available,
            } => {
                write!(f, "OOM: need {requested} CPU blocks, only {available} free")
            }
        }
    }
}

impl std::error::Error for AllocationError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn manager() -> BlockManager {
        BlockManager::new(8, 4, 16)
    }

    #[test]
    fn initial_state() {
        let m = manager();
        assert_eq!(m.free_gpu_blocks(), 8);
        assert_eq!(m.free_cpu_blocks(), 4);
        assert_eq!(m.block_size(), 16);
        assert!((m.gpu_utilization() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn allocate_and_free_gpu() {
        let mut m = manager();
        let blocks = m.allocate_gpu(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(m.free_gpu_blocks(), 5);

        m.free(&blocks);
        assert_eq!(m.free_gpu_blocks(), 8);
    }

    #[test]
    fn allocate_too_many_gpu_returns_err() {
        let mut m = manager();
        let result = m.allocate_gpu(9);
        assert!(result.is_err());
        match result {
            Err(AllocationError::InsufficientGpuBlocks {
                requested,
                available,
            }) => {
                assert_eq!(requested, 9);
                assert_eq!(available, 8);
            }
            _ => panic!("wrong error"),
        }
    }

    #[test]
    fn allocate_and_free_cpu() {
        let mut m = manager();
        let blocks = m.allocate_cpu(2).unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(m.free_cpu_blocks(), 2);

        m.free(&blocks);
        assert_eq!(m.free_cpu_blocks(), 4);
    }

    #[test]
    fn pin_increments_ref_count() {
        let mut m = manager();
        let blocks = m.allocate_gpu(1).unwrap();
        assert_eq!(m.ref_count(blocks[0]), 1);

        m.pin(&blocks);
        assert_eq!(m.ref_count(blocks[0]), 2);

        // Free once — still ref_count 1, not returned to free list.
        m.free(&blocks);
        assert_eq!(m.ref_count(blocks[0]), 1);
        assert_eq!(m.free_gpu_blocks(), 7); // still allocated

        // Free again — ref_count 0, returned to free list.
        m.free(&blocks);
        assert_eq!(m.ref_count(blocks[0]), 0);
        assert_eq!(m.free_gpu_blocks(), 8);
    }

    #[test]
    fn swap_out_and_in() {
        let mut m = manager();
        let gpu_blocks = m.allocate_gpu(2).unwrap();
        assert_eq!(m.free_gpu_blocks(), 6);
        assert_eq!(m.free_cpu_blocks(), 4);

        // Plan swap-out.
        let plan = m.plan_swap_out(&gpu_blocks).unwrap();
        assert_eq!(plan.gpu_to_cpu.len(), 2);
        assert_eq!(m.free_cpu_blocks(), 2); // CPU blocks allocated

        // Commit swap-out.
        m.commit_swap_out(&plan);
        assert_eq!(m.free_gpu_blocks(), 8); // GPU blocks freed

        // Plan swap-in.
        let cpu_blocks: Vec<BlockId> = plan.gpu_to_cpu.iter().map(|&(_, c)| c).collect();
        let plan2 = m.plan_swap_in(&cpu_blocks).unwrap();
        assert_eq!(plan2.cpu_to_gpu.len(), 2);
        assert_eq!(m.free_gpu_blocks(), 6); // GPU blocks re-allocated

        m.commit_swap_in(&plan2);
        assert_eq!(m.free_cpu_blocks(), 4); // CPU blocks freed
    }

    #[test]
    fn blocks_for_tokens() {
        let m = BlockManager::new(4, 2, 16);
        assert_eq!(m.blocks_for_tokens(0), 0);
        assert_eq!(m.blocks_for_tokens(1), 1);
        assert_eq!(m.blocks_for_tokens(16), 1);
        assert_eq!(m.blocks_for_tokens(17), 2);
        assert_eq!(m.blocks_for_tokens(32), 2);
        assert_eq!(m.blocks_for_tokens(33), 3);
    }

    #[test]
    fn gpu_utilization() {
        let mut m = BlockManager::new(4, 0, 16);
        assert!((m.gpu_utilization() - 0.0).abs() < 1e-6);
        let b = m.allocate_gpu(2).unwrap();
        assert!((m.gpu_utilization() - 0.5).abs() < 1e-6);
        m.free(&b);
        assert!((m.gpu_utilization() - 0.0).abs() < 1e-6);
    }
}
