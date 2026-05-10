//! Recurrent state for Qwen3.5 linear attention layers.
//!
//! Each linear attention layer maintains:
//! - Recurrent state: [num_value_heads, key_head_dim, value_head_dim] f32, V contiguous ([H,K,V])
//! - Conv state: [qkv_dim × (conv_kernel_dim - 1)] bf16

use anyhow::{Context, Result};
use cudarc::driver::CudaSlice;
use cudarc::driver::safe::CudaGraph;
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
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
    /// Medusa verifier rollback ring. Captured lazily when speculation first
    /// needs recurrent-state rollback for this slot.
    #[allow(dead_code)]
    spec_ring: Option<RecurrentSnapshotRing>,
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
            spec_ring: None,
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

    /// Ensure the Medusa verifier rollback ring exists for `k_plus_1` slots.
    #[allow(dead_code)]
    pub(crate) fn ensure_spec_ring(&mut self, ctx: &DeviceContext, k_plus_1: usize) -> Result<()> {
        let needs_capture = self
            .spec_ring
            .as_ref()
            .is_none_or(|ring| ring.slot_count() != k_plus_1);
        if needs_capture {
            self.spec_ring = Some(RecurrentSnapshotRing::capture(ctx, self, k_plus_1)?);
        }
        Ok(())
    }

    /// Save the current recurrent state into a verifier rollback slot.
    #[allow(dead_code)]
    pub(crate) fn push_ring_slot(
        &mut self,
        ctx: &DeviceContext,
        slot_idx: usize,
        k_plus_1: usize,
    ) -> Result<()> {
        self.ensure_spec_ring(ctx, k_plus_1)?;
        self.spec_ring
            .as_mut()
            .context("spec ring missing after capture")?
            .push_slot(slot_idx, self.seq_len)
    }

    /// Restore recurrent state from a verifier rollback slot.
    #[allow(dead_code)]
    pub(crate) fn restore_from_ring(&mut self, slot_idx: usize) -> Result<()> {
        let ring = self
            .spec_ring
            .as_ref()
            .context("spec ring restore requested before capture")?;
        self.seq_len = ring.restore_slot(slot_idx)?;
        Ok(())
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

#[allow(dead_code)]
pub(crate) struct RecurrentSnapshotRing {
    slots: Vec<RecurrentSnapshot>,
    snap_graphs: Vec<CudaGraph>,
    restore_graphs: Vec<CudaGraph>,
}

impl RecurrentSnapshotRing {
    #[allow(dead_code)]
    pub(crate) fn capture_for_bench(
        ctx: &DeviceContext,
        state: &mut RecurrentState,
        k_plus_1: usize,
    ) -> Result<Self> {
        Self::capture(ctx, state, k_plus_1)
    }

    pub(crate) fn capture(
        ctx: &DeviceContext,
        state: &mut RecurrentState,
        k_plus_1: usize,
    ) -> Result<Self> {
        anyhow::ensure!(k_plus_1 > 0, "k_plus_1 must be non-zero");

        let mut slots = Vec::with_capacity(k_plus_1);
        for _ in 0..k_plus_1 {
            slots.push(state.clone_to_snapshot(ctx)?);
        }
        ctx.sync()?;

        let mut snap_graphs = Vec::with_capacity(k_plus_1);
        for slot in &mut slots {
            ctx.stream
                .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(|e| anyhow::anyhow!("begin snapshot graph capture failed: {e}"))?;
            RecurrentState::copy_layers_to_snapshot(ctx, &state.layers, slot, state.seq_len)?;
            let graph = ctx
                .stream
                .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                .map_err(|e| anyhow::anyhow!("end snapshot graph capture failed: {e}"))?
                .context("snapshot graph capture returned no graph")?;
            snap_graphs.push(graph);
        }

        let mut restore_graphs = Vec::with_capacity(k_plus_1);
        for slot in &slots {
            ctx.stream
                .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(|e| anyhow::anyhow!("begin restore graph capture failed: {e}"))?;
            RecurrentState::restore_layers_from_snapshot(ctx, &mut state.layers, slot)?;
            let graph = ctx
                .stream
                .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                .map_err(|e| anyhow::anyhow!("end restore graph capture failed: {e}"))?
                .context("restore graph capture returned no graph")?;
            restore_graphs.push(graph);
        }
        ctx.sync()?;

        Ok(Self {
            slots,
            snap_graphs,
            restore_graphs,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn push_slot(&mut self, slot_idx: usize, seq_len: usize) -> Result<()> {
        anyhow::ensure!(
            slot_idx < self.slots.len(),
            "snapshot ring slot out of range"
        );
        self.slots[slot_idx].seq_len = seq_len;
        self.snap_graphs[slot_idx]
            .launch()
            .map_err(|e| anyhow::anyhow!("snapshot ring push launch failed: {e}"))
    }

    #[allow(dead_code)]
    pub(crate) fn restore_slot(&self, slot_idx: usize) -> Result<usize> {
        anyhow::ensure!(
            slot_idx < self.slots.len(),
            "snapshot ring slot out of range"
        );
        self.restore_graphs[slot_idx]
            .launch()
            .map_err(|e| anyhow::anyhow!("snapshot ring restore launch failed: {e}"))?;
        Ok(self.slots[slot_idx].seq_len)
    }

    #[allow(dead_code)]
    pub(crate) fn bench_launch_once(
        &self,
        ctx: &DeviceContext,
        restore_slot: usize,
    ) -> Result<std::time::Duration> {
        anyhow::ensure!(
            restore_slot < self.restore_graphs.len(),
            "restore_slot out of range"
        );

        ctx.sync()?;
        let start = std::time::Instant::now();
        for graph in &self.snap_graphs {
            graph
                .launch()
                .map_err(|e| anyhow::anyhow!("snapshot graph launch failed: {e}"))?;
        }
        self.restore_graphs[restore_slot]
            .launch()
            .map_err(|e| anyhow::anyhow!("restore graph launch failed: {e}"))?;
        ctx.sync()?;
        Ok(start.elapsed())
    }

    #[allow(dead_code)]
    pub(crate) fn slot_count(&self) -> usize {
        self.slots.len()
    }
}

#[allow(dead_code)]
pub(crate) fn bench_snapshot_ring_graph_overhead(
    ctx: &DeviceContext,
    state: &mut RecurrentState,
    k_plus_1: usize,
) -> Result<std::time::Duration> {
    let ring = RecurrentSnapshotRing::capture_for_bench(ctx, state, k_plus_1)?;
    ring.bench_launch_once(ctx, k_plus_1 / 2)
}

#[cfg(all(test, feature = "cuda", not(feature = "no-cuda")))]
mod tests {
    use super::*;
    use qwen35_spec::LayerType;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3.5-4B");

    fn tiny_config() -> Config35 {
        Config35 {
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            vocab_size: 32,
            rms_norm_eps: 1e-6,
            stop_token_ids: vec![0],
            bos_token_id: Some(1),
            eos_token_id: 0,
            tie_word_embeddings: true,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 8,
            linear_num_key_heads: 1,
            linear_key_head_dim: 4,
            linear_num_value_heads: 1,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 3,
            rope_theta: 10_000.0,
            rope_scaling: None,
            partial_rotary_factor: 1.0,
            rotary_dim: 8,
            rope_cache_len_hint: Some(16),
            layer_types: vec![LayerType::LinearAttention, LayerType::LinearAttention],
            num_experts: 0,
            num_experts_per_tok: 0,
            decoder_sparse_step: 1,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            norm_topk_prob: true,
            mlp_only_layers: Vec::new(),
        }
    }

    fn fill_recurrent_state(
        ctx: &DeviceContext,
        state: &mut RecurrentState,
        marker: usize,
    ) -> Result<()> {
        state.seq_len = marker;
        for (layer_idx, layer) in state.layers.iter_mut().enumerate() {
            let recurrent: Vec<f32> = (0..layer.state.len())
                .map(|idx| marker as f32 * 1000.0 + layer_idx as f32 * 100.0 + idx as f32)
                .collect();
            ctx.stream
                .memcpy_htod(&recurrent, &mut layer.state)
                .map_err(|e| anyhow::anyhow!("fill recurrent state failed: {e}"))?;

            let conv: Vec<bf16> = (0..layer.conv_state.len)
                .map(|idx| bf16::from_f32(marker as f32 * 10.0 + layer_idx as f32 + idx as f32))
                .collect();
            ctx.stream
                .memcpy_htod(&conv, &mut layer.conv_state.data)
                .map_err(|e| anyhow::anyhow!("fill conv state failed: {e}"))?;
        }
        Ok(())
    }

    fn read_recurrent_state(
        ctx: &DeviceContext,
        state: &RecurrentState,
    ) -> Result<(usize, Vec<Vec<f32>>, Vec<Vec<bf16>>)> {
        let recurrent = state
            .layers
            .iter()
            .map(|layer| {
                ctx.stream
                    .clone_dtoh(&layer.state)
                    .map_err(|e| anyhow::anyhow!("read recurrent state failed: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let conv = state
            .layers
            .iter()
            .map(|layer| {
                ctx.stream
                    .clone_dtoh(&layer.conv_state.data)
                    .map_err(|e| anyhow::anyhow!("read conv state failed: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((state.seq_len, recurrent, conv))
    }

    #[test]
    #[ignore = "CUDA micro-test; verifies Medusa recurrent snapshot-ring restore idempotence"]
    fn qwen35_recurrent_snapshot_ring_restore_idempotent() {
        let ctx = DeviceContext::new().expect("create CUDA device context");
        let config = tiny_config();
        let mut state = RecurrentState::new(&ctx, &config).expect("allocate recurrent state");
        let k_plus_1 = 6;
        let mut expected = Vec::with_capacity(k_plus_1);

        for slot_idx in 0..k_plus_1 {
            fill_recurrent_state(&ctx, &mut state, slot_idx + 1).expect("fill recurrent state");
            state
                .push_ring_slot(&ctx, slot_idx, k_plus_1)
                .expect("push snapshot ring slot");
            ctx.sync().expect("sync snapshot ring push");
            expected.push(read_recurrent_state(&ctx, &state).expect("read expected state"));
        }

        for slot_idx in (0..k_plus_1).rev() {
            fill_recurrent_state(&ctx, &mut state, slot_idx + 100)
                .expect("overwrite recurrent state");
            state
                .restore_from_ring(slot_idx)
                .expect("restore snapshot ring slot");
            ctx.sync().expect("sync snapshot ring restore");
            let actual = read_recurrent_state(&ctx, &state).expect("read restored state");
            assert_eq!(actual, expected[slot_idx]);
        }
    }

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

    #[test]
    #[ignore = "CUDA micro-bench; prints Qwen3.5 recurrent snapshot-ring CUDA Graph timing"]
    fn qwen35_recurrent_snapshot_ring_bench_k6_graph() {
        let ctx = DeviceContext::new().expect("create CUDA device context");
        let config = Config35::from_file(MODEL_PATH).expect("load Qwen3.5-4B config");
        let mut state = RecurrentState::new(&ctx, &config).expect("allocate recurrent state");
        let k_plus_1 = 6;
        let ring = RecurrentSnapshotRing::capture_for_bench(&ctx, &mut state, k_plus_1)
            .expect("capture recurrent snapshot ring graphs");
        assert_eq!(ring.slot_count(), k_plus_1);

        let samples: Vec<f64> = (0..3)
            .map(|_| {
                let elapsed = ring
                    .bench_launch_once(&ctx, k_plus_1 / 2)
                    .expect("bench snapshot ring graph overhead");
                elapsed.as_secs_f64() * 1_000.0
            })
            .collect();
        let mean_ms = samples.iter().sum::<f64>() / samples.len() as f64;
        let sigma_ms = (samples
            .iter()
            .map(|sample| {
                let delta = sample - mean_ms;
                delta * delta
            })
            .sum::<f64>()
            / samples.len() as f64)
            .sqrt();
        let state_size = config.linear_num_value_heads
            * config.linear_key_head_dim
            * config.linear_value_head_dim;
        let conv_state_size = config.linear_attn_qkv_dim() * (config.linear_conv_kernel_dim - 1);
        let per_snapshot_bytes = state.layers.len()
            * (state_size * std::mem::size_of::<f32>()
                + conv_state_size * std::mem::size_of::<bf16>());
        let ring_mib = (per_snapshot_bytes * k_plus_1) as f64 / (1024.0 * 1024.0);
        println!(
            "qwen35_snapshot_ring_graph_bench k_plus_1={} samples_ms={:?} mean_ms={:.3} sigma_ms={:.3} per_snapshot_mean_ms={:.3} estimated_ring_memory_delta_mib={:.1}",
            k_plus_1,
            samples,
            mean_ms,
            sigma_ms,
            mean_ms / k_plus_1 as f64,
            ring_mib
        );
    }
}
