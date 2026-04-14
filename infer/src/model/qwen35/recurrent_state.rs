//! Recurrent state for Qwen3.5 linear attention layers.
//!
//! Each linear attention layer maintains:
//! - Recurrent state: [num_value_heads, key_head_dim, value_head_dim] f32, V contiguous ([H,K,V])
//! - Conv state: [qkv_dim × (conv_kernel_dim - 1)] bf16

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;

use super::config::Config35;
use infer_cuda_kernels::prelude::{DeviceContext, DeviceVec};

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
        self.snapshot = Some(RecurrentSnapshot {
            layers: snap_layers,
            seq_len: self.seq_len,
        });
        Ok(())
    }

    /// Restore recurrent state from snapshot. Returns true if restored.
    ///
    /// Called on full prefix cache hit to revert decode-token contamination.
    /// The live state is overwritten with the clean post-prefill snapshot.
    pub(crate) fn restore_snapshot(&mut self, ctx: &DeviceContext) -> Result<bool> {
        let snap = match &self.snapshot {
            Some(s) => s,
            None => return Ok(false),
        };
        for (i, snap_layer) in snap.layers.iter().enumerate() {
            ctx.stream
                .memcpy_dtod(&snap_layer.state, &mut self.layers[i].state)
                .map_err(|e| anyhow::anyhow!("restore recurrent state D2D failed: {}", e))?;
            ctx.stream
                .memcpy_dtod(&snap_layer.conv_state, &mut self.layers[i].conv_state.data)
                .map_err(|e| anyhow::anyhow!("restore conv state D2D failed: {}", e))?;
        }
        self.seq_len = snap.seq_len;
        Ok(true)
    }
}
