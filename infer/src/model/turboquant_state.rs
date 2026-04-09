//! TurboQuant per-layer state: rotation matrices and Lloyd-Max codebook.
//!
//! Generated deterministically at model load time from a seed. The rotation
//! matrix is unique per layer (seed + layer_idx), while the codebook is shared
//! across all layers (depends only on head_dim and bits).
//!
//! All GPU allocations happen once at init — no per-request allocation.

use anyhow::{Result, anyhow};
use cudarc::driver::CudaSlice;

use crate::ffi;
use crate::tensor::DeviceContext;

/// Packed bytes per head for a given head_dim and bit width.
pub fn packed_bytes_per_head(head_dim: usize, bits: u8) -> usize {
    let effective_bits = if bits == 3 { 4 } else { bits as usize };
    (head_dim * effective_bits + 7) / 8
}

/// Shared codebook (centroids + boundaries) for a given (head_dim, bits) pair.
/// Allocated once, shared across all layers.
pub struct TurboQuantCodebook {
    /// Lloyd-Max centroids: `[2^bits]` f32, device memory.
    pub centroids: CudaSlice<f32>,
    /// Lloyd-Max boundaries: `[2^bits + 1]` f32 (including endpoints -1, +1), device memory.
    pub boundaries: CudaSlice<f32>,
    pub num_levels: usize,
    pub bits: u8,
    pub head_dim: usize,
}

impl TurboQuantCodebook {
    /// Compute Lloyd-Max codebook on CPU, upload to GPU.
    pub fn new(ctx: &DeviceContext, head_dim: usize, bits: u8) -> Result<Self> {
        let num_levels = 1usize << bits;
        let max_iters = 200;

        let mut centroids_host = vec![0.0f32; num_levels];
        let mut boundaries_host = vec![0.0f32; num_levels + 1];

        unsafe {
            ffi::turboquant_lloyd_max(
                centroids_host.as_mut_ptr(),
                boundaries_host.as_mut_ptr(),
                num_levels as i32,
                head_dim as i32,
                max_iters,
            );
        }

        let centroids = ctx
            .stream
            .clone_htod(&centroids_host)
            .map_err(|e| anyhow!("TQ codebook H2D: {e}"))?;
        let boundaries = ctx
            .stream
            .clone_htod(&boundaries_host)
            .map_err(|e| anyhow!("TQ boundaries H2D: {e}"))?;

        Ok(Self {
            centroids,
            boundaries,
            num_levels,
            bits,
            head_dim,
        })
    }
}

/// Per-layer rotation matrix.
pub struct TurboQuantRotation {
    /// Random orthogonal matrix `Pi`: `[head_dim, head_dim]` f32, device memory.
    pub matrix: CudaSlice<f32>,
    pub head_dim: usize,
}

impl TurboQuantRotation {
    /// Generate a deterministic random orthogonal rotation matrix on CPU, upload to GPU.
    pub fn new(ctx: &DeviceContext, head_dim: usize, seed: u64) -> Result<Self> {
        let n = head_dim * head_dim;
        let mut pi_host = vec![0.0f32; n];

        unsafe {
            ffi::turboquant_generate_rotation(pi_host.as_mut_ptr(), head_dim as i32, seed);
        }

        let matrix = ctx
            .stream
            .clone_htod(&pi_host)
            .map_err(|e| anyhow!("TQ rotation H2D: {e}"))?;

        Ok(Self { matrix, head_dim })
    }
}

/// Complete TurboQuant state for one KV type (K or V) across all layers.
pub struct TurboQuantLayerState {
    /// Per-layer rotation matrices.
    pub rotations: Vec<TurboQuantRotation>,
    /// Shared codebook (same for all layers with same head_dim + bits).
    pub codebook: TurboQuantCodebook,
    /// Packed bytes per head.
    pub packed_per_head: usize,
}

impl TurboQuantLayerState {
    /// Initialize TurboQuant state for all layers.
    ///
    /// Each layer gets a unique rotation matrix derived from `base_seed + layer_idx * 7`.
    /// The codebook is shared.
    pub fn new(
        ctx: &DeviceContext,
        num_layers: usize,
        head_dim: usize,
        bits: u8,
        base_seed: u64,
    ) -> Result<Self> {
        let codebook = TurboQuantCodebook::new(ctx, head_dim, bits)?;

        let mut rotations = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let seed = base_seed.wrapping_add((layer_idx as u64).wrapping_mul(7));
            rotations.push(TurboQuantRotation::new(ctx, head_dim, seed)?);
        }

        let packed_per_head = packed_bytes_per_head(head_dim, bits);

        Ok(Self {
            rotations,
            codebook,
            packed_per_head,
        })
    }
}
