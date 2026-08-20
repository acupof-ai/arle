//! Full (GQA) attention in every phase: seq-chunked, CP ring, prefix-KV capture, and gen segment.

use super::*;

impl Qwen35Layer {
    /// Full-attention forward with pre-computed K/V (used by the chunked
    /// prefix pass: K/V spans prefix + chunk, Q spans the chunk only).
    /// `q_start` = absolute row of the chunk's first token, so the causal
    /// mask lets Q[i] attend to K/V[0 .. q_start+i+1].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_full_attention_with_kv(
        &self,
        h: TensorId,
        attn: &Qwen35FullAttention,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos: TensorId,
        sin: TensorId,
        kv: &PrefixKv,
        batch: usize,
        seq_len: usize,
        q_start: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let local_attention_heads = tp.local_attention_heads(cfg)?;
        let q_full = attn.q_proj.forward(h, store, tape)?;
        let (q, gate) = if cfg.full_attn_gated {
            let q_full = reshape(
                q_full,
                &[batch, seq_len, local_attention_heads, cfg.head_dim * 2],
                store,
                tape,
            )?;
            let q = slice(
                q_full,
                &[0, 0, 0, 0],
                &[batch, seq_len, local_attention_heads, cfg.head_dim],
                store,
                tape,
            )?;
            let gate = slice(
                q_full,
                &[0, 0, 0, cfg.head_dim],
                &[batch, seq_len, local_attention_heads, cfg.head_dim * 2],
                store,
                tape,
            )?;
            (
                transpose(q, 1, 2, store, tape)?,
                Some(transpose(gate, 1, 2, store, tape)?),
            )
        } else {
            let q = reshape(
                q_full,
                &[batch, seq_len, local_attention_heads, cfg.head_dim],
                store,
                tape,
            )?;
            (transpose(q, 1, 2, store, tape)?, None)
        };
        let q = qwen35_rmsnorm(q, attn.q_norm, cfg.rms_norm_eps, store, tape)?;
        let q = rope(q, cos, sin, store, tape)?;
        // kv.k / kv.v are already RMSNorm'd, RoPE'd, and repeat_kv'd (full heads).
        let attn_hidden = causal_sdpa_recompute_with_q_start(q, kv.k, kv.v, q_start, store, tape)?;
        let attn_hidden = if let Some(gate) = gate {
            let gate = sigmoid(gate, store, tape)?;
            mul(attn_hidden, gate, store, tape)?
        } else {
            attn_hidden
        };
        let attn_hidden = merge_heads(
            attn_hidden,
            batch,
            seq_len,
            local_attention_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let out = attn.o_proj.forward(attn_hidden, store, tape)?;
        Ok(maybe_all_reduce(out, tp, store, tape)?)
    }

    pub(super) fn forward_full_attention(
        &self,
        h: TensorId,
        attn: &Qwen35FullAttention,
        cfg: &Qwen35Config,
        tp: TpContext,
        cp: crate::context_parallel::CpContext,
        cos: TensorId,
        sin: TensorId,
        batch: usize,
        seq_len: usize,
        cp_positions: Option<&[usize]>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let local_attention_heads = tp.local_attention_heads(cfg)?;
        let local_key_value_heads = tp.local_key_value_heads(cfg)?;
        let kv_repeat = local_attention_heads / local_key_value_heads;

        // Non-CP: k/v full-seq (small), chunk q_proj+SDPA+o_proj to bound the
        // full-seq f32 GEMM output. CP uses its own q-shard + all-gather-kv.
        if !cp.is_enabled() {
            return self.forward_full_attention_chunked(
                h,
                attn,
                cfg,
                tp,
                cos,
                sin,
                batch,
                seq_len,
                local_attention_heads,
                local_key_value_heads,
                kv_repeat,
                crate::runtime_flags::OPD_SEQ_CHUNK,
                store,
                tape,
            );
        }

        // CP: k/v stay full-shard at kv_heads width (cheap); q_proj -> RoPE ->
        // ring SDPA (q tile) -> gate -> o_proj run per sequence chunk, so no
        // q-sized intermediate is resident at full local seq (was +25.9 GB per
        // full-attn layer replay at global 131,072). dK/dV accumulate across q
        // tiles in the chunked backward; the ring supports tiled q natively
        // (positions = tile rows, k_positions = full local shard).
        //
        // Qwen3.5 / Qwen3.6 ship a gated Q projection: q_proj rows =
        // `num_heads * head_dim * 2`, with the second half acting as a
        // per-head sigmoid gate applied to the attention output. The arch flag
        // `cfg.full_attn_gated` selects between the two paths.
        let k = attn.k_proj.forward(h, store, tape)?;
        let v = attn.v_proj.forward(h, store, tape)?;
        let k = split_heads(
            k,
            batch,
            seq_len,
            local_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let v = split_heads(
            v,
            batch,
            seq_len,
            local_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let k = qwen35_rmsnorm(k, attn.k_norm, cfg.rms_norm_eps, store, tape)?;
        let k = rope(k, cos, sin, store, tape)?;

        let positions = cp_positions.ok_or(AutogradError::TapeInvariant(
            "CP full-attention requires cp_positions (the shard's absolute rows)",
        ))?;
        debug_assert_eq!(
            positions.len(),
            seq_len,
            "CP: cp_positions must give one absolute position per local row"
        );
        let positions = positions.to_vec();
        let (cp_size, cp_rank) = (cp.size, cp.rank);
        let q_proj = attn.q_proj.clone();
        let o_proj = attn.o_proj.clone();
        let q_norm = attn.q_norm;
        let (head_dim, eps, gated) = (cfg.head_dim, cfg.rms_norm_eps, cfg.full_attn_gated);
        let heads = local_attention_heads;
        let mut extra = vec![k, v, cos, sin];
        collect_linear_ids(&attn.q_proj, &mut extra);
        collect_linear_ids(&attn.o_proj, &mut extra);
        extra.push(q_norm);
        let out = autograd::ops::checkpoint_seq_chunked(
            h,
            extra,
            crate::runtime_flags::OPD_SEQ_CHUNK,
            store,
            tape,
            move |st, tp, start, inp| {
                let (h_c, k, v, cos, sin) = (inp[0], inp[1], inp[2], inp[3], inp[4]);
                let rows = st
                    .get(h_c)
                    .ok_or(AutogradError::InvalidTensorId(h_c))?
                    .shape[1];
                let width = if gated { head_dim * 2 } else { head_dim };
                let q_full = q_proj.forward(h_c, st, tp)?;
                let q_full = reshape(q_full, &[batch, rows, heads, width], st, tp)?;
                let (q, gate) = if gated {
                    let q = slice(
                        q_full,
                        &[0, 0, 0, 0],
                        &[batch, rows, heads, head_dim],
                        st,
                        tp,
                    )?;
                    let gate = slice(
                        q_full,
                        &[0, 0, 0, head_dim],
                        &[batch, rows, heads, head_dim * 2],
                        st,
                        tp,
                    )?;
                    (q, Some(gate))
                } else {
                    (q_full, None)
                };
                let q = transpose(q, 1, 2, st, tp)?;
                let q = qwen35_rmsnorm(q, q_norm, eps, st, tp).map_err(qwen35_to_autograd)?;
                let half = st
                    .get(cos)
                    .ok_or(AutogradError::InvalidTensorId(cos))?
                    .shape[1];
                let cos_c = slice(cos, &[start, 0], &[start + rows, half], st, tp)?;
                let sin_c = slice(sin, &[start, 0], &[start + rows, half], st, tp)?;
                let q = rope(q, cos_c, sin_c, st, tp)?;
                let q_pos = &positions[start..start + rows];
                let hidden = autograd::ops::ring_attention::cp_causal_sdpa(
                    q,
                    k,
                    v,
                    cp_size,
                    cp_rank,
                    Some(q_pos),
                    Some(&positions),
                    st,
                    tp,
                )?;
                let hidden = match gate {
                    Some(gate) => {
                        let gate = transpose(gate, 1, 2, st, tp)?;
                        let gate = sigmoid(gate, st, tp)?;
                        mul(hidden, gate, st, tp)?
                    }
                    None => hidden,
                };
                let hidden = transpose(hidden, 1, 2, st, tp)?;
                let hidden = reshape(hidden, &[batch, rows, heads * head_dim], st, tp)?;
                o_proj.forward(hidden, st, tp)
            },
        )?;
        Ok(maybe_all_reduce(out, tp, store, tape)?)
    }

    /// Long-sequence single-GPU full attention: k/v are computed full-sequence
    /// (kv_dim is small), then q_proj + causal SDPA + o_proj are run per
    /// sequence chunk via `checkpoint_seq_chunked`. This bounds the q/o
    /// projection f32 outputs (and their LoRA deltas) to the chunk row count
    /// instead of the full sequence — the 131K forward-OOM site.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_full_attention_chunked(
        &self,
        h: TensorId,
        attn: &Qwen35FullAttention,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos: TensorId,
        sin: TensorId,
        batch: usize,
        seq_len: usize,
        local_attention_heads: usize,
        local_key_value_heads: usize,
        kv_repeat: usize,
        chunk: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let head_dim = cfg.head_dim;
        let rms_norm_eps = cfg.rms_norm_eps;
        let full_attn_gated = cfg.full_attn_gated;

        // k/v: small enough to keep full-sequence.
        let k = attn.k_proj.forward(h, store, tape)?;
        let v = attn.v_proj.forward(h, store, tape)?;
        let k = split_heads(
            k,
            batch,
            seq_len,
            local_key_value_heads,
            head_dim,
            store,
            tape,
        )?;
        let v = split_heads(
            v,
            batch,
            seq_len,
            local_key_value_heads,
            head_dim,
            store,
            tape,
        )?;
        let k = qwen35_rmsnorm(k, attn.k_norm, rms_norm_eps, store, tape)?;
        let k = rope(k, cos, sin, store, tape)?;
        let k = repeat_kv(k, kv_repeat, store, tape)?;
        let v = repeat_kv(v, kv_repeat, store, tape)?;

        // Capture the linears and norm for the replay closure.
        let q_proj = attn.q_proj.clone();
        let o_proj = attn.o_proj.clone();
        let q_norm = attn.q_norm;

        // Saved inputs: k, v (full-seq intermediates), cos/sin, and the
        // trainable q_proj/o_proj/q_norm weights. `checkpoint_seq_chunked`
        // accumulates their gradients across chunks during backward.
        let mut param_ids = vec![k, v, cos, sin];
        collect_linear_ids(&q_proj, &mut param_ids);
        collect_linear_ids(&o_proj, &mut param_ids);
        param_ids.push(q_norm);

        let out = autograd::ops::checkpoint_seq_chunked(
            h,
            param_ids,
            chunk,
            store,
            tape,
            move |st, tp_tape, start, inp| {
                let h_chunk = inp[0];
                let k = inp[1];
                let v = inp[2];
                let cos_full = inp[3];
                let sin_full = inp[4];

                let chunk_shape = st
                    .get(h_chunk)
                    .ok_or(AutogradError::InvalidTensorId(h_chunk))?
                    .shape
                    .clone();
                let chunk_len = chunk_shape[1];

                // q projection (with LoRA) for this chunk.
                let q_full = q_proj.forward(h_chunk, st, tp_tape)?;

                let (q, gate) = if full_attn_gated {
                    let q_full = reshape(
                        q_full,
                        &[batch, chunk_len, local_attention_heads, head_dim * 2],
                        st,
                        tp_tape,
                    )?;
                    let q = slice(
                        q_full,
                        &[0, 0, 0, 0],
                        &[batch, chunk_len, local_attention_heads, head_dim],
                        st,
                        tp_tape,
                    )?;
                    let gate = slice(
                        q_full,
                        &[0, 0, 0, head_dim],
                        &[batch, chunk_len, local_attention_heads, head_dim * 2],
                        st,
                        tp_tape,
                    )?;
                    (
                        transpose(q, 1, 2, st, tp_tape)?,
                        Some(transpose(gate, 1, 2, st, tp_tape)?),
                    )
                } else {
                    let q = reshape(
                        q_full,
                        &[batch, chunk_len, local_attention_heads, head_dim],
                        st,
                        tp_tape,
                    )?;
                    (transpose(q, 1, 2, st, tp_tape)?, None)
                };

                // q norm + rope (cos/sin sliced to this chunk's positions).
                let q = qwen35_rmsnorm(q, q_norm, rms_norm_eps, st, tp_tape)
                    .map_err(qwen35_to_autograd)?;
                let cos_shape = st
                    .get(cos_full)
                    .ok_or(AutogradError::InvalidTensorId(cos_full))?
                    .shape
                    .clone();
                let rotary_dim = cos_shape[1];
                let cos_chunk = slice(
                    cos_full,
                    &[start, 0],
                    &[start + chunk_len, rotary_dim],
                    st,
                    tp_tape,
                )?;
                let sin_chunk = slice(
                    sin_full,
                    &[start, 0],
                    &[start + chunk_len, rotary_dim],
                    st,
                    tp_tape,
                )?;
                let q = rope(q, cos_chunk, sin_chunk, st, tp_tape)?;

                // Causal SDPA: q at [start, start+chunk) attends k/v at [0, start+chunk).
                let attn_hidden = causal_sdpa_recompute_with_q_start(q, k, v, start, st, tp_tape)?;

                let attn_hidden = if let Some(gate) = gate {
                    let gate = sigmoid(gate, st, tp_tape)?;
                    mul(attn_hidden, gate, st, tp_tape)?
                } else {
                    attn_hidden
                };

                let attn_hidden = merge_heads(
                    attn_hidden,
                    batch,
                    chunk_len,
                    local_attention_heads,
                    head_dim,
                    st,
                    tp_tape,
                )
                .map_err(qwen35_to_autograd)?;
                o_proj.forward(attn_hidden, st, tp_tape)
            },
        )?;

        Ok(maybe_all_reduce(out, tp, store, tape)?)
    }

    /// OPD frozen-prompt-KV phase 1: compute the prompt prefix's repeat_kv'd K/V
    /// at absolute positions `0..gen_start` (k_proj/v_proj → split_heads →
    /// k_norm → rope(cos/sin sliced to the prefix) → repeat_kv), returning them
    /// as `requires_grad=false` constants. Off-tape (caller passes a disabled
    /// tape). Only K/V — the prompt's Q is never queried by the gen segment.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_full_attention_capture_prefix_kv(
        &self,
        h_prefix: TensorId,
        attn: &Qwen35FullAttention,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos_prefix: TensorId,
        sin_prefix: TensorId,
        batch: usize,
        gen_start: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<PrefixKv> {
        let local_attention_heads = tp.local_attention_heads(cfg)?;
        let local_key_value_heads = tp.local_key_value_heads(cfg)?;
        let k = attn.k_proj.forward(h_prefix, store, tape)?;
        let v = attn.v_proj.forward(h_prefix, store, tape)?;
        let k = split_heads(
            k,
            batch,
            gen_start,
            local_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let v = split_heads(
            v,
            batch,
            gen_start,
            local_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let k = qwen35_rmsnorm(k, attn.k_norm, cfg.rms_norm_eps, store, tape)?;
        let k = rope(k, cos_prefix, sin_prefix, store, tape)?;
        let kv_repeat = local_attention_heads / local_key_value_heads;
        let k = repeat_kv(k, kv_repeat, store, tape)?;
        let v = repeat_kv(v, kv_repeat, store, tape)?;
        // Pin as constants so the gen-segment concat treats them as frozen.
        store.set_requires_grad(k, false)?;
        store.set_requires_grad(v, false)?;
        // Largest retained buffer, no grad chain — the bf16 tape's first target.
        store.quantize_frozen_bf16(k)?;
        store.quantize_frozen_bf16(v)?;
        Ok(PrefixKv { k, v })
    }

    /// OPD frozen-prompt-KV phase 2: TAPED full attention over the gen rows only
    /// (`h_gen` is `[batch, gen_len, hidden]`). Projects Q/K/V for the gen rows,
    /// q_norm/k_norm, rope at gen positions (cos_gen/sin_gen already sliced to
    /// `gen_start..seq_len`), repeat_kv, then concatenates the frozen prefix K/V
    /// along the seq axis (grad flows into k_gen/v_gen only) and runs cached SDPA
    /// with `q_start = gen_start`. The gate/merge/o_proj/all_reduce tail is
    /// identical to `forward_full_attention`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_full_attention_gen_segment(
        &self,
        h_gen: TensorId,
        attn: &Qwen35FullAttention,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos_gen: TensorId,
        sin_gen: TensorId,
        prefix_kv: &PrefixKv,
        batch: usize,
        gen_start: usize,
        gen_len: usize,
        cp: crate::context_parallel::CpContext,
        cp_positions: Option<&[usize]>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let q_full = attn.q_proj.forward(h_gen, store, tape)?;
        let local_attention_heads = tp.local_attention_heads(cfg)?;
        let local_key_value_heads = tp.local_key_value_heads(cfg)?;
        let (q, gate) = if cfg.full_attn_gated {
            let q_full = reshape(
                q_full,
                &[batch, gen_len, local_attention_heads, cfg.head_dim * 2],
                store,
                tape,
            )?;
            let q = slice(
                q_full,
                &[0, 0, 0, 0],
                &[batch, gen_len, local_attention_heads, cfg.head_dim],
                store,
                tape,
            )?;
            let gate = slice(
                q_full,
                &[0, 0, 0, cfg.head_dim],
                &[batch, gen_len, local_attention_heads, cfg.head_dim * 2],
                store,
                tape,
            )?;
            (
                transpose(q, 1, 2, store, tape)?,
                Some(transpose(gate, 1, 2, store, tape)?),
            )
        } else {
            let q = reshape(
                q_full,
                &[batch, gen_len, local_attention_heads, cfg.head_dim],
                store,
                tape,
            )?;
            (transpose(q, 1, 2, store, tape)?, None)
        };

        let k = attn.k_proj.forward(h_gen, store, tape)?;
        let v = attn.v_proj.forward(h_gen, store, tape)?;
        let k = split_heads(
            k,
            batch,
            gen_len,
            local_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let v = split_heads(
            v,
            batch,
            gen_len,
            local_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;

        let q = qwen35_rmsnorm(q, attn.q_norm, cfg.rms_norm_eps, store, tape)?;
        let k = qwen35_rmsnorm(k, attn.k_norm, cfg.rms_norm_eps, store, tape)?;
        let q = rope(q, cos_gen, sin_gen, store, tape)?;
        let k = rope(k, cos_gen, sin_gen, store, tape)?;

        // CP + off-CP share one path: cp_causal_sdpa_with_prefix handles cp_size==1
        // by falling through to its prefix CPU block (repeat_kv + cat + q_start
        // causal SDPA), identical to the old inline else branch. Gen k/v stay at
        // kv_heads width under CP (GQA resolved in the ring kernel); off-CP the
        // prefix block repeat_kv's them to match the full-heads prefix.
        let attn_hidden = autograd::ops::ring_attention::cp_causal_sdpa_with_prefix(
            q,
            k,
            v,
            cp.size,
            cp.rank,
            cp_positions,
            None,
            Some((prefix_kv.k, prefix_kv.v)),
            gen_start,
            store,
            tape,
        )?;

        let attn_hidden = if let Some(gate) = gate {
            let gate = sigmoid(gate, store, tape)?;
            mul(attn_hidden, gate, store, tape)?
        } else {
            attn_hidden
        };
        let attn_hidden = merge_heads(
            attn_hidden,
            batch,
            gen_len,
            local_attention_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let out = attn.o_proj.forward(attn_hidden, store, tape)?;
        Ok(maybe_all_reduce(out, tp, store, tape)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_full_attention_profiled(
        &self,
        layer_index: usize,
        h: TensorId,
        attn: &Qwen35FullAttention,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos: TensorId,
        sin: TensorId,
        batch: usize,
        seq_len: usize,
        trace: bool,
        store: &mut TensorStore,
        tape: &mut Tape,
        profile: &mut Qwen35AttentionForwardProfile,
    ) -> Result<TensorId> {
        let started = Instant::now();
        let q_full = attn.q_proj.forward(h, store, tape)?;
        profile.q_proj += started.elapsed();
        trace_attention_component(trace, layer_index, "q_proj", profile.q_proj);

        let local_attention_heads = tp.local_attention_heads(cfg)?;
        let local_key_value_heads = tp.local_key_value_heads(cfg)?;
        let started = Instant::now();
        let (q, gate) = if cfg.full_attn_gated {
            let q_full = reshape(
                q_full,
                &[batch, seq_len, local_attention_heads, cfg.head_dim * 2],
                store,
                tape,
            )?;
            let q = slice(
                q_full,
                &[0, 0, 0, 0],
                &[batch, seq_len, local_attention_heads, cfg.head_dim],
                store,
                tape,
            )?;
            let gate = slice(
                q_full,
                &[0, 0, 0, cfg.head_dim],
                &[batch, seq_len, local_attention_heads, cfg.head_dim * 2],
                store,
                tape,
            )?;
            (
                transpose(q, 1, 2, store, tape)?,
                Some(transpose(gate, 1, 2, store, tape)?),
            )
        } else {
            let q = reshape(
                q_full,
                &[batch, seq_len, local_attention_heads, cfg.head_dim],
                store,
                tape,
            )?;
            (transpose(q, 1, 2, store, tape)?, None)
        };
        profile.q_layout += started.elapsed();
        trace_attention_component(trace, layer_index, "q_layout", profile.q_layout);

        let started = Instant::now();
        let k = attn.k_proj.forward(h, store, tape)?;
        profile.k_proj += started.elapsed();
        trace_attention_component(trace, layer_index, "k_proj", profile.k_proj);

        let started = Instant::now();
        let v = attn.v_proj.forward(h, store, tape)?;
        profile.v_proj += started.elapsed();
        trace_attention_component(trace, layer_index, "v_proj", profile.v_proj);

        let started = Instant::now();
        let k = split_heads(
            k,
            batch,
            seq_len,
            local_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let v = split_heads(
            v,
            batch,
            seq_len,
            local_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        profile.kv_split += started.elapsed();
        trace_attention_component(trace, layer_index, "kv_split", profile.kv_split);

        let started = Instant::now();
        let q = qwen35_rmsnorm(q, attn.q_norm, cfg.rms_norm_eps, store, tape)?;
        let k = qwen35_rmsnorm(k, attn.k_norm, cfg.rms_norm_eps, store, tape)?;
        profile.qk_norm += started.elapsed();
        trace_attention_component(trace, layer_index, "qk_norm", profile.qk_norm);

        let started = Instant::now();
        let q = rope(q, cos, sin, store, tape)?;
        let k = rope(k, cos, sin, store, tape)?;
        profile.rope += started.elapsed();
        trace_attention_component(trace, layer_index, "rope", profile.rope);

        let kv_repeat = local_attention_heads / local_key_value_heads;
        let started = Instant::now();
        let k = repeat_kv(k, kv_repeat, store, tape)?;
        let v = repeat_kv(v, kv_repeat, store, tape)?;
        profile.repeat_kv += started.elapsed();
        trace_attention_component(trace, layer_index, "repeat_kv", profile.repeat_kv);

        let started = Instant::now();
        let attn_hidden = causal_sdpa_recompute(q, k, v, store, tape)?;
        profile.sdpa += started.elapsed();
        trace_attention_component(trace, layer_index, "sdpa", profile.sdpa);

        let started = Instant::now();
        let attn_hidden = if let Some(gate) = gate {
            let gate = sigmoid(gate, store, tape)?;
            mul(attn_hidden, gate, store, tape)?
        } else {
            attn_hidden
        };
        profile.gate += started.elapsed();
        trace_attention_component(trace, layer_index, "gate", profile.gate);

        let started = Instant::now();
        let attn_hidden = merge_heads(
            attn_hidden,
            batch,
            seq_len,
            local_attention_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        profile.merge += started.elapsed();
        trace_attention_component(trace, layer_index, "merge", profile.merge);

        let started = Instant::now();
        let out = attn.o_proj.forward(attn_hidden, store, tape)?;
        let out = maybe_all_reduce(out, tp, store, tape)?;
        profile.o_proj += started.elapsed();
        trace_attention_component(trace, layer_index, "o_proj", profile.o_proj);
        Ok(out)
    }
}
