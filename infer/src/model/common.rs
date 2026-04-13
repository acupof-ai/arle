//! Shared building blocks for transformer model implementations.
//!
//! Contains types and functions used by both Qwen3 and Qwen3.5:
//! - `MLP` — SwiGLU MLP weights with optional merged gate+up projection
//! - `get_embeddings_batch` — token ID → hidden state embedding lookup
//! - `compute_logits_batch` — last hidden → final norm → LM head projection
//! - `output_projection` — resolve tied vs untied LM head

use anyhow::Result;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;

use crate::ops;
use crate::tensor::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use crate::weight_loader::{
    load_shard_info, load_shard_info_fixed, load_tensor_2d, load_tensor_2d_maybe_quantized,
    mmap_shards,
};

// ─── MLP weights ─────────────────────────────────────────────────────────────

/// SwiGLU MLP layer weights. Shared by all transformer models in this crate.
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct MLP {
    pub(crate) gate_proj: DeviceMatrix,
    pub(crate) up_proj: DeviceMatrix,
    /// Merged gate+up projection: `[2*inter_dim, hidden_dim]`.
    /// Pre-computed for the batched decode path (one GEMM instead of two).
    /// `None` when the model variant doesn't use merged projections.
    pub(crate) gate_up_proj: Option<DeviceMatrix>,
    pub(crate) down_proj: DeviceMatrix,
}

impl MLP {
    /// Load MLP weights from safetensors.
    ///
    /// `prefix` should be e.g. `"model.layers.0.mlp"`.
    /// When `merge_gate_up` is true, a fused `[gate; up]` matrix is also created.
    pub(crate) fn load(
        ctx: &DeviceContext,
        shards: &[SafeTensors],
        weight_map: &HashMap<String, usize>,
        prefix: &str,
        merge_gate_up: bool,
    ) -> Result<Self> {
        Self::load_with_quant(ctx, shards, weight_map, prefix, merge_gate_up, 0)
    }

    pub(crate) fn load_with_quant(
        ctx: &DeviceContext,
        shards: &[SafeTensors],
        weight_map: &HashMap<String, usize>,
        prefix: &str,
        merge_gate_up: bool,
        quant_group_size: usize,
    ) -> Result<Self> {
        let load_w = |name: &str| -> Result<DeviceMatrix> {
            if quant_group_size > 0 {
                load_tensor_2d_maybe_quantized(ctx, shards, weight_map, name, quant_group_size)
            } else {
                load_tensor_2d(ctx, shards, weight_map, name)
            }
        };
        let gate_proj = load_w(&format!("{}.gate_proj.weight", prefix))?;
        let up_proj = load_w(&format!("{}.up_proj.weight", prefix))?;
        let gate_up_proj = if merge_gate_up {
            Some(DeviceMatrix::concat_rows(ctx, &[&gate_proj, &up_proj])?)
        } else {
            None
        };
        let down_proj = load_w(&format!("{}.down_proj.weight", prefix))?;
        Ok(Self {
            gate_proj,
            up_proj,
            gate_up_proj,
            down_proj,
        })
    }
}

// ─── Embedding ───────────────────────────────────────────────────────────────

/// Look up embeddings for a batch of token IDs.
///
/// Returns `HiddenStates` of shape `[hidden_dim, seq_len]`.
pub(crate) fn get_embeddings_batch(
    ctx: &DeviceContext,
    embed_tokens: &DeviceMatrix,
    token_ids: &[u32],
    hidden_dim: usize,
) -> Result<HiddenStates> {
    let seq_len = token_ids.len();
    let token_ids_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
    let token_ids_gpu = ctx
        .stream
        .clone_htod(&token_ids_i32)
        .map_err(|e| anyhow::anyhow!("H2D copy failed: {}", e))?;
    let mut out = HiddenStates::zeros(ctx, hidden_dim, seq_len)?;
    ops::embedding_batch(ctx, embed_tokens, &token_ids_gpu, &mut out)?;
    debug_dump_hidden(ctx, &out, "after embedding", hidden_dim);
    Ok(out)
}

/// Print first 8 and last 4 elements of a hidden-state buffer to stderr,
/// gated by `PEGAINFER_DEBUG_DUMP=1`. Used to bisect forward-pass divergence
/// between safetensors and GGUF load paths.
pub(crate) fn debug_dump_hidden(
    ctx: &DeviceContext,
    hidden: &HiddenStates,
    label: &str,
    hidden_dim: usize,
) {
    if std::env::var("PEGAINFER_DEBUG_DUMP").is_err() {
        return;
    }
    use half::bf16;
    let last_idx = hidden.seq_len.saturating_sub(1);
    let view = hidden
        .data
        .slice(last_idx * hidden_dim..last_idx * hidden_dim + hidden_dim);
    let buf: Vec<bf16> = match ctx.stream.memcpy_dtov(&view) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[debug-dump] {label}: copy failed: {e}");
            return;
        }
    };
    let _ = ctx.sync();
    let f32s: Vec<f32> = buf.iter().map(|v| v.to_f32()).collect();
    let head: Vec<String> = f32s.iter().take(8).map(|v| format!("{v:+.5}")).collect();
    let tail_start = hidden_dim.saturating_sub(4);
    let tail: Vec<String> = f32s
        .iter()
        .skip(tail_start)
        .map(|v| format!("{v:+.5}"))
        .collect();
    let max = f32s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = f32s.iter().cloned().fold(f32::INFINITY, f32::min);
    let nan = f32s.iter().any(|v| v.is_nan());
    let inf = f32s.iter().any(|v| v.is_infinite());
    let rms = (f32s.iter().map(|v| v * v).sum::<f32>() / hidden_dim as f32).sqrt();
    eprintln!(
        "[debug-dump] {label}: head={:?} tail={:?} min={min:.4} max={max:.4} rms={rms:.4} nan={nan} inf={inf}",
        head, tail
    );
}

// ─── Logits ──────────────────────────────────────────────────────────────────

/// Compute logits from a prefill hidden-state batch.
///
/// Extracts the last token's hidden state, applies final RMSNorm,
/// then projects to vocab size via the output projection (LM head).
///
/// `use_offset_norm`: when true, uses the `(1 + weight)` RMSNorm variant
/// (Qwen3.5); when false, uses standard RMSNorm (Qwen3).
pub(crate) fn compute_logits_batch(
    ctx: &DeviceContext,
    hidden: &HiddenStates,
    norm_weight: &DeviceVec,
    output_proj: &DeviceMatrix,
    eps: f32,
    use_offset_norm: bool,
) -> Result<DeviceVec> {
    anyhow::ensure!(
        hidden.seq_len > 0,
        "compute_logits_batch: empty hidden states"
    );
    let last_hidden = ops::extract_vec(ctx, hidden, hidden.seq_len - 1)?;
    let normed = if use_offset_norm {
        let mut out = DeviceVec::zeros(ctx, last_hidden.len)?;
        ops::rms_norm_offset_into(ctx, &last_hidden, norm_weight, eps, &mut out)?;
        out
    } else {
        ops::rms_norm(ctx, &last_hidden, norm_weight, eps)?
    };
    ops::linear(ctx, &normed, output_proj)
}

/// Return the output projection matrix: separate LM head if present, otherwise
/// tied embeddings.
pub(crate) fn output_projection<'a>(
    lm_head: &'a Option<DeviceMatrix>,
    embed_tokens: &'a DeviceMatrix,
) -> &'a DeviceMatrix {
    lm_head.as_ref().unwrap_or(embed_tokens)
}

// ─── Weight loading ──────────────────────────────────────────────────────────

/// Load and memory-map safetensors shards from a model directory.
///
/// When `fix_shard_names` is true, applies filename fixup for models whose
/// index.json has mismatched shard names (e.g. Qwen3.5).
///
/// Returns `(mmaps, weight_map)`. Caller should pass `&mmaps` to
/// [`deserialize_shards`] to get the `SafeTensors` views.
pub(crate) fn load_safetensors(
    model_path: &str,
    fix_shard_names: bool,
) -> Result<(Vec<Mmap>, HashMap<String, usize>)> {
    let (shard_paths, weight_map) = if fix_shard_names {
        load_shard_info_fixed(model_path)?
    } else {
        load_shard_info(model_path)?
    };
    log::debug!("Loading {} safetensor shard(s)", shard_paths.len());
    let mmaps = mmap_shards(&shard_paths)?;
    Ok((mmaps, weight_map))
}

/// Deserialize `SafeTensors` views from memory-mapped shards.
///
/// The returned `SafeTensors` borrow from `mmaps`, so `mmaps` must outlive them.
pub(crate) fn deserialize_shards(mmaps: &[Mmap]) -> Result<Vec<SafeTensors<'_>>> {
    mmaps
        .iter()
        .map(|m| {
            SafeTensors::deserialize(m).map_err(|e| anyhow::anyhow!("Deserialize error: {}", e))
        })
        .collect()
}
