//! Whether to gradient-checkpoint, the checkpointed layer stack, and the measured-peak report.

use super::*;

/// The one config→params mapping — every linear-attention call site and the
/// checkpoint gate's ctx-bytes model must agree field-for-field.
pub(super) fn la_params(cfg: &Qwen35Config, batch: usize, seq_len: usize) -> LinearAttentionParams {
    LinearAttentionParams {
        batch,
        seq_len,
        num_key_heads: cfg.linear_num_key_heads,
        num_value_heads: cfg.linear_num_value_heads,
        key_dim: cfg.linear_key_head_dim,
        value_dim: cfg.linear_value_head_dim,
        conv_kernel: cfg.linear_conv_kernel_dim,
        eps: cfg.rms_norm_eps,
    }
}

/// Peak device bytes one LINEAR-attention layer holds — the whole O(seq) story of a
/// checkpointed forward, since `linear_attention_core` runs the full sequence in one
/// call while MLP and full attention chunk. UNDER-models the measured 27B slope
/// (211712 vs 231424 B/token, −9%); `log_ckpt_peak` exists to name that drift.
pub(super) fn la_layer_peak_bytes(cfg: &Qwen35Config, batch: usize, seq_len: usize) -> usize {
    let p = la_params(cfg, batch, seq_len);
    // x and h: the residual stream is the layer's, not the kernel's.
    let residual = 2 * cfg.hidden_size;
    linear_attention_row_transient_bytes(p)
        .saturating_add(linear_attention_ctx_bytes(p))
        .saturating_add(batch.saturating_mul(seq_len).saturating_mul(4 * residual))
}

/// Peak device bytes a single FULL-attention layer holds. CP and single-GPU both
/// run `forward_full_attention_chunked`, so only k/v, the chunk accumulator and
/// the residual stream are full-sequence; the whole q side costs one chunk.
///
/// Full-sequence terms: the k/v chain (k_proj, v_proj, two `split_heads`, k_norm,
/// k_rope) at kv width, plus a full-head pair — off-CP that is `repeat_kv`, under
/// CP it stands in for the ring's rotated blocks (an over-model until cp_size
/// exceeds `1 + heads/kv_heads`) — plus the `checkpoint_seq_chunked` output and
/// x/h. Chunk terms: q_proj at gate width, then the two slices, two transposes,
/// q_norm, rope, attention out, sigmoid, mul and merge at q width, then o_proj.
pub(super) fn full_attn_layer_peak_bytes(
    cfg: &Qwen35Config,
    tp: TpContext,
    batch: usize,
    seq_len: usize,
) -> Result<usize> {
    let q_dim = tp.local_attention_heads(cfg)? * cfg.head_dim;
    let kv_dim = tp.local_key_value_heads(cfg)? * cfg.head_dim;
    let full_seq = 6 * kv_dim + 2 * q_dim + 3 * cfg.hidden_size;
    let per_chunk = if cfg.full_attn_gated {
        12 * q_dim + cfg.hidden_size
    } else {
        6 * q_dim + cfg.hidden_size
    };
    let chunk = crate::runtime_flags::OPD_SEQ_CHUNK.min(seq_len);
    Ok(batch.saturating_mul(4).saturating_mul(
        seq_len
            .saturating_mul(full_seq)
            .saturating_add(chunk.saturating_mul(per_chunk)),
    ))
}

impl Qwen35Model {
    pub fn gradient_checkpointing(&self) -> bool {
        self.gradient_checkpointing
    }

    pub fn set_gradient_checkpointing(&mut self, enabled: bool) {
        self.gradient_checkpointing = enabled;
    }

    /// The one checkpointing decision. Checkpointing trades a full forward
    /// re-run in backward (plus boundary host offload) for tape memory — a
    /// good trade only when the full tape would NOT fit: modeled footprint
    /// (per layer the hidden-stream + dominant MLP term, plus the exact LA
    /// ctx bytes for hybrid layers via `linear_attention_ctx_bytes`) ×4
    /// against free VRAM. ×4 is calibrated, not a guess: the measured
    /// full-tape footprint is 2.9–3.3× the modeled floor (0.8B B=4 seq=1040
    /// [vram-ramp]: 39.1 GB forward vs 13.6 modeled, 45 GB with CE+backward,
    /// wins/2026-07-24; 27B hybrid measured 2.54×), ×~1.2 headroom for pool
    /// retention and shape variance. The 2026-07-24 ×4 revert was the broken
    /// batched LA backward crashing under checkpoint — fixed by ecc058b20 —
    /// not the ratio. Unknown memory info keeps the safe default
    /// (checkpoint). `ARLE_FORCE_CHECKPOINT` engages unconditionally — the
    /// CUDA finite-diff gate needs the checkpoint path at tiny shapes the
    /// estimate would decline.
    pub(super) fn should_checkpoint(
        &self,
        batch: usize,
        seq_len: usize,
        store: &TensorStore,
        tape: &Tape,
    ) -> bool {
        self.gradient_checkpointing
            && tape.enabled
            && (std::env::var_os("ARLE_FORCE_CHECKPOINT").is_some()
                || store.backend().device_mem_info().is_none_or(|(free, _)| {
                    let cfg = &self.config;
                    let per_layer =
                        batch * seq_len * (8 * cfg.hidden_size + 3 * cfg.intermediate_size) * 4;
                    let la_layers = self
                        .layers
                        .iter()
                        .filter(|layer| matches!(layer.self_attn, Qwen35Attention::Linear(_)))
                        .count();
                    let la_ctx = linear_attention_ctx_bytes(la_params(cfg, batch, seq_len));
                    let total = per_layer
                        .saturating_mul(self.layers.len())
                        .saturating_add(la_ctx.saturating_mul(la_layers));
                    let engage = total.saturating_mul(4) > free;
                    // Log first decision and every flip: the gate's failure class is
                    // a silent false-non-engage (97 GB OOM, errors/2026-07-24), so
                    // the breadcrumb must exist even when it never engages.
                    static LAST: AtomicU8 = AtomicU8::new(u8::MAX);
                    if LAST.swap(engage as u8, Ordering::Relaxed) != engage as u8 {
                        eprintln!(
                            "[ckpt-gate] engage={engage} batch={batch} seq={seq_len} \
                         modeled={total}B (la_layers={la_layers}) free={free}B"
                        );
                    }
                    engage
                }))
    }

    /// Run the layer stack checkpointed one layer per group. Groups of K do one
    /// host offload per K layers, but a group's forward holds all K layers' live
    /// activations until it saves — K× the per-layer peak, which OOMs at long
    /// seq. The frozen/LoRA boundary still forces a split.
    pub(super) fn checkpoint_layers<PF, FF>(
        &self,
        hidden: TensorId,
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
        layer_params: PF,
        layer_fn: FF,
    ) -> Result<TensorId>
    where
        FF: Fn(usize, TensorId, &mut TensorStore, &mut Tape) -> autograd::Result<TensorId>
            + Clone
            + Send
            + Sync
            + 'static,
        PF: Fn(usize) -> Vec<TensorId>,
    {
        // The rebase sets the watermark to current used, so it reads back as the floor.
        let floor = store
            .backend()
            .reset_mem_pool_used_high()
            .ok()
            .and_then(|()| store.backend().mem_pool_used_high());
        let out = checkpoint_sequential(
            hidden,
            self.layers.len(),
            self.lora_layer_start,
            store,
            tape,
            layer_params,
            layer_fn,
        )?;
        self.log_ckpt_peak(batch, seq_len, floor, store, tape)?;
        Ok(out)
    }

    /// Print the modeled post-checkpoint peak against the measured one, once per
    /// new high-water. `should_checkpoint` only answers "would the FULL tape
    /// fit" — nothing modeled what checkpointing actually leaves resident, so
    /// every memory wall cost a ~3000 s run to find. The model: one layer is
    /// live at a time, so peak = the floor measured just before the stack + the
    /// widest single layer + the saved checkpoints that stayed on device.
    ///
    /// Treat the number as a FIRST MEASUREMENT, not a validated model. The two
    /// H20 points on hand (seq 65536 → 66987 MiB, 131072 → 81451 MiB) are two
    /// unknowns in two equations and cannot validate anything, and no constant
    /// here was fitted to close a gap. `drift` is the output that matters.
    pub(super) fn log_ckpt_peak(
        &self,
        batch: usize,
        seq_len: usize,
        floor: Option<u64>,
        store: &TensorStore,
        tape: &Tape,
    ) -> Result<()> {
        // Without both the floor and the watermark there is no drift to report.
        let (Some(floor), Some(actual)) = (floor, store.backend().mem_pool_used_high()) else {
            return Ok(());
        };
        static HIGH: AtomicU64 = AtomicU64::new(0);
        if HIGH.fetch_max(actual, Ordering::Relaxed) >= actual {
            return Ok(());
        }
        let cfg = &self.config;
        // Group size is 1, so the peak is the largest single layer.
        let mut layer = 0;
        for l in &self.layers {
            layer = layer.max(match &l.self_attn {
                Qwen35Attention::Linear(_) => la_layer_peak_bytes(cfg, batch, seq_len),
                Qwen35Attention::Full(_) => {
                    full_attn_layer_peak_bytes(cfg, self.tp, batch, seq_len)?
                }
            });
        }
        // Offloaded groups leave nothing behind; resident ones keep one hidden each.
        let saved = if tape.offload_checkpoints() {
            0
        } else {
            batch
                .saturating_mul(seq_len)
                .saturating_mul(cfg.hidden_size * 4)
                .saturating_mul(self.layers.len())
        };
        let modeled = floor.saturating_add(layer.saturating_add(saved) as u64);
        let mib = |bytes: u64| bytes >> 20;
        eprintln!(
            "[ckpt-peak] unvalidated batch={batch} seq={seq_len} floor={}MiB layer={}MiB \
             saved={}MiB modeled={}MiB actual={}MiB drift={:+}MiB",
            mib(floor),
            mib(layer as u64),
            mib(saved as u64),
            mib(modeled),
            mib(actual),
            mib(actual) as i64 - mib(modeled) as i64,
        );
        Ok(())
    }
}
