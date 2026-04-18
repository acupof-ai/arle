//! TurboQuant per-layer state: rotation transforms and Lloyd-Max codebook.
//!
//! Supports two rotation modes:
//! - **Full rotation** (O(D²)): random orthogonal matrix via QR. Higher quality,
//!   useful for reference/validation.
//! - **Hadamard** (O(D log D)): random sign flip + Fast Walsh-Hadamard Transform.
//!   18x fewer FMAs at D=128 (896 vs 16384). Default for production.
//!
//! Generated deterministically at model load time from a seed. The codebook is
//! shared across all layers (depends only on head_dim and bits).
//!
//! All GPU allocations happen once at init — no per-request allocation.

use anyhow::{Result, anyhow};
use cudarc::driver::CudaSlice;

use crate::ffi;
use crate::tensor::DeviceContext;

/// Packed bytes per head for a given head_dim and bit width.
pub fn packed_bytes_per_head(head_dim: usize, bits: u8) -> usize {
    let effective_bits = if bits == 3 { 4 } else { bits as usize };
    (head_dim * effective_bits).div_ceil(8)
}

/// Rotation mode for TurboQuant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RotationMode {
    /// Full random orthogonal matrix (D×D matmul, O(D²)).
    Full,
    /// Randomized Hadamard: sign flip + FWHT (O(D log D)). Default.
    #[default]
    Hadamard,
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

/// Per-layer rotation transform — either a full D×D matrix or Hadamard signs.
pub enum TurboQuantRotation {
    /// Full random orthogonal matrix: `[D, D]` f32, device memory.
    Full { matrix: CudaSlice<f32> },
    /// Hadamard signs: `[D]` i8 ∈ {-1, +1}, device memory.
    Hadamard { signs: CudaSlice<i8> },
}

impl TurboQuantRotation {
    /// Generate a full rotation matrix (QR of Gaussian, O(D²) init).
    pub fn new_full(ctx: &DeviceContext, head_dim: usize, seed: u64) -> Result<Self> {
        let n = head_dim * head_dim;
        let mut pi_host = vec![0.0f32; n];

        unsafe {
            ffi::turboquant_generate_rotation(pi_host.as_mut_ptr(), head_dim as i32, seed);
        }

        let matrix = ctx
            .stream
            .clone_htod(&pi_host)
            .map_err(|e| anyhow!("TQ rotation H2D: {e}"))?;

        Ok(Self::Full { matrix })
    }

    /// Generate Hadamard signs (O(D) init, O(D log D) runtime).
    pub fn new_hadamard(ctx: &DeviceContext, head_dim: usize, seed: u64) -> Result<Self> {
        let mut signs_host = vec![0i8; head_dim];

        unsafe {
            ffi::turboquant_generate_signs(signs_host.as_mut_ptr(), head_dim as i32, seed);
        }

        let signs = ctx
            .stream
            .clone_htod(&signs_host)
            .map_err(|e| anyhow!("TQ signs H2D: {e}"))?;

        Ok(Self::Hadamard { signs })
    }

    /// Get the full rotation matrix pointer (panics if Hadamard mode).
    pub fn full_matrix_ptr(&self) -> &CudaSlice<f32> {
        match self {
            Self::Full { matrix } => matrix,
            Self::Hadamard { .. } => panic!("full_matrix_ptr called on Hadamard rotation"),
        }
    }

    /// Get the Hadamard signs pointer (panics if Full mode).
    pub fn hadamard_signs_ptr(&self) -> &CudaSlice<i8> {
        match self {
            Self::Hadamard { signs } => signs,
            Self::Full { .. } => panic!("hadamard_signs_ptr called on Full rotation"),
        }
    }

    /// Whether this is Hadamard mode.
    #[allow(dead_code)]
    pub fn is_hadamard(&self) -> bool {
        matches!(self, Self::Hadamard { .. })
    }
}

/// Complete TurboQuant state for one KV type (K or V) across all layers.
pub struct TurboQuantLayerState {
    /// Per-layer rotation transforms.
    pub rotations: Vec<TurboQuantRotation>,
    /// Shared codebook (same for all layers with same head_dim + bits).
    pub codebook: TurboQuantCodebook,
    /// Packed bytes per head.
    pub packed_per_head: usize,
    /// Rotation mode.
    pub mode: RotationMode,
}

impl TurboQuantLayerState {
    /// Initialize TurboQuant state for all layers.
    ///
    /// Each layer gets a unique rotation derived from `base_seed + layer_idx * 7`.
    /// Default mode is Hadamard (O(D log D)).
    pub fn new(
        ctx: &DeviceContext,
        num_layers: usize,
        head_dim: usize,
        bits: u8,
        base_seed: u64,
    ) -> Result<Self> {
        Self::with_mode(
            ctx,
            num_layers,
            head_dim,
            bits,
            base_seed,
            RotationMode::default(),
        )
    }

    /// Initialize with explicit rotation mode.
    pub fn with_mode(
        ctx: &DeviceContext,
        num_layers: usize,
        head_dim: usize,
        bits: u8,
        base_seed: u64,
        mode: RotationMode,
    ) -> Result<Self> {
        // Hadamard requires head_dim to be a power of 2.
        if mode == RotationMode::Hadamard {
            assert!(
                head_dim.is_power_of_two(),
                "Hadamard rotation requires power-of-2 head_dim, got {head_dim}"
            );
        }

        let codebook = TurboQuantCodebook::new(ctx, head_dim, bits)?;

        let mut rotations = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let seed = base_seed.wrapping_add((layer_idx as u64).wrapping_mul(7));
            let rotation = match mode {
                RotationMode::Full => TurboQuantRotation::new_full(ctx, head_dim, seed)?,
                RotationMode::Hadamard => TurboQuantRotation::new_hadamard(ctx, head_dim, seed)?,
            };
            rotations.push(rotation);
        }

        let packed_per_head = packed_bytes_per_head(head_dim, bits);

        Ok(Self {
            rotations,
            codebook,
            packed_per_head,
            mode,
        })
    }
}
