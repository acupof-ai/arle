use anyhow::Result;
use cudarc::driver::CudaSlice;

use super::weights::{Qwen3Model, TransformerBlock};
use crate::model::kv_cache::KVCache;
use crate::ops;
use crate::paged_kv::TokenKVPool;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

/// Pre-allocated scratch buffers for one prefill forward pass.
/// Created once per prefill in `process_all_layers_batch`, eliminating
/// per-layer `cuMemAllocAsync` overhead (~11k calls / 88ms at seq=2048).
///
/// Buffer reuse across steps (all kernels serialized on a single stream):
///   `normed`  reused for `normed2`  (steps 1-4 done before step 8)
///   `o_buf`   reused for `mlp_out`  (step 7 done before step 12)
struct PrefillBuffers {
    /// Output ping-pong: layer writes result here; caller swaps with the incoming hidden.
    hidden_out: HiddenStates, // hidden_dim × seq_len
    normed: HiddenStates,      // hidden_dim × seq_len (reused for normed2)
    q_batch: HiddenStates,     // q_dim × seq_len
    k_batch: HiddenStates,     // kv_dim × seq_len
    v_batch: HiddenStates,     // kv_dim × seq_len
    o_buf: HiddenStates,       // hidden_dim × seq_len (reused for mlp_out)
    gate_out: HiddenStates,    // inter_dim × seq_len
    up_out: HiddenStates,      // inter_dim × seq_len
    act_out: HiddenStates,     // inter_dim × seq_len
    attn_output: HiddenStates, // q_dim × seq_len
}

impl PrefillBuffers {
    fn new(
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        seq_len: usize,
    ) -> Result<Self> {
        Ok(Self {
            hidden_out: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            normed: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, seq_len)?,
            k_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            v_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            o_buf: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            gate_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, seq_len)?,
        })
    }
}

impl Qwen3Model {
    #[fastrace::trace(name = "get_embeddings_batch")]
    pub(super) fn get_embeddings_batch(&self, token_ids: &[u32]) -> Result<HiddenStates> {
        crate::model::common::get_embeddings_batch(
            &self.ctx,
            &self.embed_tokens,
            token_ids,
            self.config.hidden_size,
        )
    }

    #[fastrace::trace(name = "process_all_layers_batch")]
    pub(super) fn process_all_layers_batch(
        &self,
        mut hidden: HiddenStates,
        start_pos: usize,
        kv_cache: &mut KVCache,
    ) -> Result<HiddenStates> {
        let seq_len = hidden.seq_len;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let inter_dim = self.config.intermediate_size;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Allocate all intermediates once — eliminates ~11k cuMemAllocAsync calls.
        let mut bufs = PrefillBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            q_dim,
            kv_dim,
            inter_dim,
            seq_len,
        )?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_layer_batch(
                layer_idx,
                layer,
                &mut hidden,
                start_pos,
                kv_cache,
                &mut bufs,
            )?;
        }

        // Increment sequence length AFTER all layers processed
        for _ in 0..seq_len {
            kv_cache.increment_seq_len();
        }

        Ok(hidden)
    }

    /// Like `process_all_layers_batch`, but also scatter-writes K/V to the
    /// token pool after each layer's QKV GEMM. The Triton prefill attention
    /// kernel still reads from the contiguous cache (dual-write approach).
    #[fastrace::trace(name = "process_all_layers_batch_with_pool")]
    pub(super) fn process_all_layers_batch_with_pool(
        &self,
        mut hidden: HiddenStates,
        start_pos: usize,
        kv_cache: &mut KVCache,
        pool: &TokenKVPool,
        new_token_indices: &CudaSlice<i32>,
    ) -> Result<HiddenStates> {
        let seq_len = hidden.seq_len;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let inter_dim = self.config.intermediate_size;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let mut bufs = PrefillBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            q_dim,
            kv_dim,
            inter_dim,
            seq_len,
        )?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_layer_batch_with_pool(
                layer_idx,
                layer,
                &mut hidden,
                start_pos,
                kv_cache,
                &mut bufs,
                pool,
                new_token_indices,
            )?;
        }

        // Increment sequence length AFTER all layers processed
        for _ in 0..seq_len {
            kv_cache.increment_seq_len();
        }

        Ok(hidden)
    }

    /// Like `forward_layer_batch`, but scatter-writes K/V to the token pool
    /// after QKV GEMM (before Triton attention). The contiguous cache write
    /// still happens inside the Triton attention kernel.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_batch_with_pool(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        start_pos: usize,
        kv_cache: &mut KVCache,
        bufs: &mut PrefillBuffers,
        pool: &TokenKVPool,
        new_token_indices: &CudaSlice<i32>,
    ) -> Result<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        kv_cache.init_if_needed(&self.ctx, self.config.head_dim)?;

        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} pre-norm hidden"),
            self.config.hidden_size,
        );
        // 1. RMSNorm
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        );
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.normed,
            &format!("L{layer_idx} after-input-norm"),
            self.config.hidden_size,
        );

        // 2. QKV projections
        ops::gemm_into(
            &self.ctx,
            &layer.attention.q_proj,
            &bufs.normed,
            &mut bufs.q_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.attention.k_proj,
            &bufs.normed,
            &mut bufs.k_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.attention.v_proj,
            &bufs.normed,
            &mut bufs.v_batch,
        );

        // 2b. Scatter-write K/V to token pool (dual-write path).
        // The Triton attention kernel below will also write to contiguous cache.
        if pool.is_active() {
            ops::scatter_write_kv(
                &self.ctx,
                &bufs.k_batch,
                &bufs.v_batch,
                pool.k_ptr(layer_idx, &self.ctx.stream),
                pool.v_ptr(layer_idx, &self.ctx.stream),
                new_token_indices,
                num_kv_heads,
                head_dim,
            )?;
        }

        // 3. FlashAttention-2 (Triton) — also writes to contiguous cache
        let (k_cache_layer, v_cache_layer) = kv_cache.prepare_layer(&self.ctx, layer_idx)?;
        let nrp = ops::NormRopeParams {
            q_norm: &layer.attention.q_norm,
            k_norm: &layer.attention.k_norm,
            cos_cache: &self.cos_cache,
            sin_cache: &self.sin_cache,
            rms_eps: self.config.rms_norm_eps,
        };
        let heads = ops::HeadConfig {
            num_q_heads: num_heads,
            num_kv_heads,
            head_dim,
        };
        ops::prefill_attention_batch(
            &self.ctx,
            &mut bufs.q_batch,
            &mut bufs.k_batch,
            &bufs.v_batch,
            &nrp,
            k_cache_layer,
            v_cache_layer,
            &mut bufs.attn_output,
            &heads,
            start_pos,
        )?;
        kv_cache.commit_layer(&self.ctx, layer_idx, start_pos, hidden.seq_len)?;

        // 4-8: Same as forward_layer_batch (O proj, residual, MLP)
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        );

        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} after-attn+residual"),
            self.config.hidden_size,
        );

        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        );

        ops::gemm_into(
            &self.ctx,
            &layer.mlp.gate_proj,
            &bufs.normed,
            &mut bufs.gate_out,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.up_proj,
            &bufs.normed,
            &mut bufs.up_out,
        );
        ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        );

        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} layer-end"),
            self.config.hidden_size,
        );

        Ok(())
    }

    pub(super) fn compute_logits_batch(&self, hidden: &HiddenStates) -> Result<DeviceVec> {
        crate::model::common::compute_logits_batch(
            &self.ctx,
            hidden,
            &self.norm,
            self.output_projection(),
            self.config.rms_norm_eps,
            false, // standard RMSNorm (not offset)
        )
    }

    fn forward_layer_batch(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        start_pos: usize,
        kv_cache: &mut KVCache,
        bufs: &mut PrefillBuffers,
    ) -> Result<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        kv_cache.init_if_needed(&self.ctx, self.config.head_dim)?;

        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} pre-norm hidden"),
            self.config.hidden_size,
        );
        // 1. RMSNorm → bufs.normed
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        );
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.normed,
            &format!("L{layer_idx} after-input-norm"),
            self.config.hidden_size,
        );

        // 2. QKV projections → bufs.q_batch, bufs.k_batch, bufs.v_batch
        ops::gemm_into(
            &self.ctx,
            &layer.attention.q_proj,
            &bufs.normed,
            &mut bufs.q_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.attention.k_proj,
            &bufs.normed,
            &mut bufs.k_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.attention.v_proj,
            &bufs.normed,
            &mut bufs.v_batch,
        );
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.q_batch,
            &format!("L{layer_idx} q_proj_out (pre-norm-rope)"),
            bufs.q_batch.hidden_dim,
        );
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.k_batch,
            &format!("L{layer_idx} k_proj_out (pre-norm-rope)"),
            bufs.k_batch.hidden_dim,
        );
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.v_batch,
            &format!("L{layer_idx} v_proj_out"),
            bufs.v_batch.hidden_dim,
        );

        // 3. FlashAttention-2 (Triton) → bufs.attn_output
        let (k_cache_layer, v_cache_layer) = kv_cache.prepare_layer(&self.ctx, layer_idx)?;
        let nrp = ops::NormRopeParams {
            q_norm: &layer.attention.q_norm,
            k_norm: &layer.attention.k_norm,
            cos_cache: &self.cos_cache,
            sin_cache: &self.sin_cache,
            rms_eps: self.config.rms_norm_eps,
        };
        let heads = ops::HeadConfig {
            num_q_heads: num_heads,
            num_kv_heads,
            head_dim,
        };
        ops::prefill_attention_batch(
            &self.ctx,
            &mut bufs.q_batch,
            &mut bufs.k_batch,
            &bufs.v_batch,
            &nrp,
            k_cache_layer,
            v_cache_layer,
            &mut bufs.attn_output,
            &heads,
            start_pos,
        )?;
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.q_batch,
            &format!("L{layer_idx} q (post-norm-rope)"),
            bufs.q_batch.hidden_dim,
        );
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.attn_output,
            &format!("L{layer_idx} attn_output (pre-o-proj)"),
            bufs.attn_output.hidden_dim,
        );
        // Quantize newly written KV tokens → INT8 storage (no-op for BF16)
        kv_cache.commit_layer(&self.ctx, layer_idx, start_pos, hidden.seq_len)?;

        // 4. O projection → bufs.o_buf (as o_batch)
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        );
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.o_buf,
            &format!("L{layer_idx} o_proj_out"),
            bufs.o_buf.hidden_dim,
        );

        // 5. Residual add: hidden_in + o_batch → bufs.hidden_out
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        // Swap: hidden = attn_residual, bufs.hidden_out = old hidden_in (now free)
        std::mem::swap(hidden, &mut bufs.hidden_out);
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} after-attn+residual"),
            self.config.hidden_size,
        );

        // 6. MLP RMSNorm → bufs.normed (reused for normed2; steps 1-4 are done)
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        );

        // 7. MLP: gate + up → act → down → bufs.o_buf (reused for mlp_out; step 5 is done)
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.gate_proj,
            &bufs.normed,
            &mut bufs.gate_out,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.up_proj,
            &bufs.normed,
            &mut bufs.up_out,
        );
        ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        );

        // 8. Residual add: attn_residual + mlp_out → bufs.hidden_out (old hidden_in, free to overwrite)
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        // Swap: hidden = layer output, bufs.hidden_out = attn_residual (free next layer)
        std::mem::swap(hidden, &mut bufs.hidden_out);
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} layer-end"),
            self.config.hidden_size,
        );

        Ok(())
    }
}
