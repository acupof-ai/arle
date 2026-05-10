//! Recurrent state for Qwen3.5 linear attention layers.
//!
//! Each linear attention layer maintains:
//! - Recurrent state: [num_value_heads, key_head_dim, value_head_dim] f32, V contiguous ([H,K,V])
//! - Conv state: [qkv_dim × (conv_kernel_dim - 1)] bf16

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;

use super::config::Config35;
use cuda_kernels::prelude::{DeviceContext, DeviceVec};

/// Per-layer recurrent state for a single linear attention layer.
pub(crate) struct LayerRecurrentState {
    /// Recurrent state matrix: [num_value_heads * key_head_dim * value_head_dim] f32
    /// Stored as f32 per mamba_ssm_dtype="float32" in config.
    pub(crate) state: CudaSlice<f32>,
    /// Conv1d state buffer: [qkv_dim * (conv_kernel_dim - 1)] bf16
    /// Stores the last (kernel_dim - 1) inputs for causal conv1d.
    pub(crate) conv_state: DeviceVec,
}

/// Snapshot of one linear attention layer's state (for prefix cache restore).
struct LayerSnapshot {
    state: CudaSlice<f32>,
    conv_state: CudaSlice<bf16>,
}

/// Post-prefill snapshot of all linear attention layers.
/// Captured after prefill completes; restored on full prefix cache hit.
struct RecurrentSnapshot {
    layers: Vec<LayerSnapshot>,
    seq_len: usize,
}

/// Recurrent state for all linear attention layers.
pub(crate) struct RecurrentState {
    pub(crate) layers: Vec<LayerRecurrentState>,
    /// Number of tokens processed so far (for prefill/decode tracking).
    pub(crate) seq_len: usize,
    /// Post-prefill snapshot for prefix cache reuse.
    /// Saved after prefill, restored on full prefix hit to avoid decode contamination.
    snapshot: Option<RecurrentSnapshot>,
}

impl RecurrentState {
    /// Allocate zeroed recurrent state for all linear attention layers.
    pub(crate) fn new(ctx: &DeviceContext, config: &Config35) -> Result<Self> {
        let num_linear_layers = config.num_hidden_layers - config.num_full_attention_layers();

        let state_size = config.linear_num_value_heads
            * config.linear_key_head_dim
            * config.linear_value_head_dim;
        let qkv_dim = config.linear_attn_qkv_dim();
        let conv_state_size = qkv_dim * (config.linear_conv_kernel_dim - 1);

        let mut layers = Vec::with_capacity(num_linear_layers);
        for _ in 0..num_linear_layers {
            let state: CudaSlice<f32> = ctx
                .stream
                .alloc_zeros(state_size)
                .map_err(|e| anyhow::anyhow!("Alloc recurrent state failed: {}", e))?;
            layers.push(LayerRecurrentState {
                state,
                conv_state: DeviceVec::zeros(ctx, conv_state_size)?,
            });
        }

        Ok(Self {
            layers,
            seq_len: 0,
            snapshot: None,
        })
    }

    /// Reset all state to zeros for a new generation.
    pub(crate) fn reset(&mut self, ctx: &DeviceContext) -> Result<()> {
        self.seq_len = 0;
        for layer in &mut self.layers {
            ctx.stream
                .memset_zeros(&mut layer.state)
                .map_err(|e| anyhow::anyhow!("memset recurrent state failed: {}", e))?;
            ctx.stream
                .memset_zeros(&mut layer.conv_state.data)
                .map_err(|e| anyhow::anyhow!("memset conv state failed: {}", e))?;
        }
        Ok(())
    }

    /// Save a snapshot of current recurrent state (GPU → GPU copy).
    ///
    /// Called after prefill completes, before decode begins. On a subsequent
    /// full prefix cache hit, `restore_snapshot()` brings the state back to
    /// this clean post-prefill point, avoiding decode-token contamination.
    ///
    /// Cost: ~49 MB GPU memcpy for Qwen3.5-4B (24 layers × ~2 MB each).
    pub(crate) fn save_snapshot(&mut self, ctx: &DeviceContext) -> Result<()> {
        self.snapshot = Some(self.clone_to_snapshot(ctx)?);
        Ok(())
    }

    fn clone_to_snapshot(&self, ctx: &DeviceContext) -> Result<RecurrentSnapshot> {
        let mut snap_layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let state_copy: CudaSlice<f32> = ctx
                .stream
                .clone_dtod(&layer.state)
                .map_err(|e| anyhow::anyhow!("snapshot recurrent state D2D failed: {}", e))?;
            let conv_copy: CudaSlice<bf16> = ctx
                .stream
                .clone_dtod(&layer.conv_state.data)
                .map_err(|e| anyhow::anyhow!("snapshot conv state D2D failed: {}", e))?;
            snap_layers.push(LayerSnapshot {
                state: state_copy,
                conv_state: conv_copy,
            });
        }
        Ok(RecurrentSnapshot {
            layers: snap_layers,
            seq_len: self.seq_len,
        })
    }

    fn restore_layers_from_snapshot(
        ctx: &DeviceContext,
        layers: &mut [LayerRecurrentState],
        snap: &RecurrentSnapshot,
    ) -> Result<()> {
        for (i, snap_layer) in snap.layers.iter().enumerate() {
            ctx.stream
                .memcpy_dtod(&snap_layer.state, &mut layers[i].state)
                .map_err(|e| anyhow::anyhow!("restore recurrent state D2D failed: {}", e))?;
            ctx.stream
                .memcpy_dtod(&snap_layer.conv_state, &mut layers[i].conv_state.data)
                .map_err(|e| anyhow::anyhow!("restore conv state D2D failed: {}", e))?;
        }
        Ok(())
    }

    fn copy_layers_to_snapshot(
        ctx: &DeviceContext,
        layers: &[LayerRecurrentState],
        snap: &mut RecurrentSnapshot,
        seq_len: usize,
    ) -> Result<()> {
        for (i, layer) in layers.iter().enumerate() {
            ctx.stream
                .memcpy_dtod(&layer.state, &mut snap.layers[i].state)
                .map_err(|e| anyhow::anyhow!("snapshot recurrent state D2D failed: {}", e))?;
            ctx.stream
                .memcpy_dtod(&layer.conv_state.data, &mut snap.layers[i].conv_state)
                .map_err(|e| anyhow::anyhow!("snapshot conv state D2D failed: {}", e))?;
        }
        snap.seq_len = seq_len;
        Ok(())
    }

    /// Restore recurrent state from snapshot. Returns true if restored.
    ///
    /// Called on full prefix cache hit to revert decode-token contamination.
    /// The live state is overwritten with the clean post-prefill snapshot.
    pub(crate) fn restore_snapshot(&mut self, ctx: &DeviceContext) -> Result<bool> {
        let Some(snap) = &self.snapshot else {
            return Ok(false);
        };
        Self::restore_layers_from_snapshot(ctx, &mut self.layers, snap)?;
        self.seq_len = snap.seq_len;
        Ok(true)
    }
}

/// Prototype benchmark for Medusa Phase 1.B-Qwen3.5 snapshot-ring rollback.
///
/// Ring slots are allocated before timing. The measured section copies the live
/// recurrent state into each preallocated slot and restores from the middle slot.
/// The stream is synchronized before and after so the returned duration includes
/// the D2D copy work, not just enqueue overhead.
#[allow(dead_code)]
pub(crate) fn bench_snapshot_ring_overhead(
    ctx: &DeviceContext,
    state: &mut RecurrentState,
    k_plus_1: usize,
) -> Result<std::time::Duration> {
    anyhow::ensure!(k_plus_1 > 0, "k_plus_1 must be non-zero");

    let mut ring = Vec::with_capacity(k_plus_1);
    for _ in 0..k_plus_1 {
        ring.push(state.clone_to_snapshot(ctx)?);
    }

    ctx.sync()?;
    let start = std::time::Instant::now();
    for snap in &mut ring {
        RecurrentState::copy_layers_to_snapshot(ctx, &state.layers, snap, state.seq_len)?;
    }
    RecurrentState::restore_layers_from_snapshot(ctx, &mut state.layers, &ring[k_plus_1 / 2])?;
    state.seq_len = ring[k_plus_1 / 2].seq_len;
    ctx.sync()?;
    Ok(start.elapsed())
}

#[cfg(all(test, feature = "cuda", not(feature = "no-cuda")))]
mod tests {
    use super::*;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3.5-4B");

    #[test]
    #[ignore = "CUDA micro-bench; prints Qwen3.5 recurrent snapshot-ring timing"]
    fn qwen35_recurrent_snapshot_ring_bench_k6() {
        let ctx = DeviceContext::new().expect("create CUDA device context");
        let config = Config35::from_file(MODEL_PATH).expect("load Qwen3.5-4B config");
        let mut state = RecurrentState::new(&ctx, &config).expect("allocate recurrent state");
        let k_plus_1 = 6;
        let elapsed = bench_snapshot_ring_overhead(&ctx, &mut state, k_plus_1)
            .expect("bench snapshot ring overhead");

        let state_size = config.linear_num_value_heads
            * config.linear_key_head_dim
            * config.linear_value_head_dim;
        let conv_state_size = config.linear_attn_qkv_dim() * (config.linear_conv_kernel_dim - 1);
        let per_snapshot_bytes = state.layers.len()
            * (state_size * std::mem::size_of::<f32>()
                + conv_state_size * std::mem::size_of::<bf16>());
        let ring_mib = (per_snapshot_bytes * k_plus_1) as f64 / (1024.0 * 1024.0);
        let total_ms = elapsed.as_secs_f64() * 1_000.0;
        println!(
            "qwen35_snapshot_ring_bench k_plus_1={} total_ms={:.3} per_snapshot_ms={:.3} estimated_ring_memory_delta_mib={:.1}",
            k_plus_1,
            total_ms,
            total_ms / k_plus_1 as f64,
            ring_mib
        );
    }
}
