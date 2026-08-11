use super::*;

impl Qwen35Model {
    pub(crate) fn forward_decode_step_captured(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        start_pos: usize,
    ) -> Result<()> {
        let mut rows = [LinearRow {
            slot,
            len: 1,
            capture: None,
        }];
        self.forward_hidden_staged(&mut rows, ws, start_pos, None, None)?;
        self.lm_head_logits(ws, 1)
    }

    pub(crate) fn forward_decode_step_paged_captured(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        start_pos: usize,
        recall: &mut Qwen35RecallForward,
    ) -> Result<()> {
        let mut rows = [LinearRow {
            slot,
            len: 1,
            capture: None,
        }];
        self.forward_hidden_staged(&mut rows, ws, start_pos, Some(recall), None)?;
        self.lm_head_logits(ws, 1)
    }

    pub(crate) fn paged_decode_fa3_active(&self) -> bool {
        self.config.head_dim == 256 && qwen35_fa3_enabled(&self.ctx)
    }

    pub(crate) fn decode_graph_unsupported_reason(&self) -> Option<&'static str> {
        let has_moe = self.layers.iter().any(|l| l.moe.is_some());
        if !has_moe {
            return None;
        }
        let Some(cfg) = self.moe_config.as_ref() else {
            return Some("MoE layers present but no moe_config");
        };
        if !crate::moe::qwen35_decode_moe_graph_capturable(cfg) {
            return Some(
                "MoE decode is not device-routable (host router fallback active — \
                 --qwen35-gpu-router false or non-greedy/grouped routing)",
            );
        }
        None
    }

    pub(crate) fn batched_copy(
        &self,
        s: &mut Qwen35CopyScratch,
        dst: &[u64],
        src: &[u64],
        bytes: &[usize],
    ) -> Result<()> {
        ensure!(
            dst.len() == src.len(),
            "batched copy dst/src length mismatch"
        );
        ensure!(
            bytes.len() == 1 || bytes.len() == dst.len(),
            "batched copy {} sizes for {} buffers",
            bytes.len(),
            dst.len()
        );
        if dst.is_empty() || bytes.iter().all(|b| *b == 0) {
            return Ok(());
        }
        ensure!(
            bytes.iter().all(|b| b % 16 == 0),
            "batched copy sizes must be 16B multiples"
        );
        let ctx = &self.ctx;
        let n = dst.len();
        s.host.clear();
        s.host.extend_from_slice(dst);
        s.host.extend_from_slice(src);
        let tbl = s.ptrs.get(ctx, 2 * n)?;
        ctx.stream
            .memcpy_htod(&s.host, tbl)
            .map_err(|e| anyhow!("H2D batched copy tables: {e}"))?;
        let (base, _g) = tbl.device_ptr(&ctx.stream);
        let (len_ptr, max_words) = if bytes.len() == 1 {
            (0u64, 0usize)
        } else {
            s.hlen.clear();
            s.hlen.extend(bytes.iter().map(|b| (b / 16) as i32));
            let max = s.hlen.iter().copied().max().unwrap_or(0) as usize;
            let d = s.lens.get(ctx, n)?;
            ctx.stream
                .memcpy_htod(&s.hlen, d)
                .map_err(|e| anyhow!("H2D batched copy sizes: {e}"))?;
            (d.device_ptr(&ctx.stream).0, max)
        };
        // SAFETY: the table holds `n` dst then `n` src live addresses, each
        // buffer at least its `bytes` entry and cudaMalloc-aligned.
        unsafe {
            ffi::batched_copy_uniform_cuda(
                base as *const *mut std::ffi::c_void,
                (base + (n as u64) * 8) as *const *const std::ffi::c_void,
                len_ptr as *const i32,
                bytes[0],
                max_words,
                n as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
        Ok(())
    }

    pub(crate) fn forward_decode_batch(
        &self,
        slots: &mut [Qwen35SlotState],
        bd: &mut Qwen35BatchDecodeState,
        slot_indices: &[usize],
        tokens: &[u32],
        kv_seq_lens: &[usize],
        params: &[SamplingParams],
        sample_positions: &[u64],
    ) -> Result<Vec<(u32, Option<f32>)>> {
        let b = tokens.len();
        ensure!(b >= 1, "Qwen3.5 batched decode requires at least one row");
        ensure!(
            slot_indices.len() == b
                && kv_seq_lens.len() == b
                && params.len() == b
                && sample_positions.len() == b,
            "Qwen3.5 batched decode surface length mismatch: slots={} tokens={} kv_lens={} params={} positions={}",
            slot_indices.len(),
            b,
            kv_seq_lens.len(),
            params.len(),
            sample_positions.len()
        );
        // Validate before any device state is touched.
        for (r, &si) in slot_indices.iter().enumerate() {
            ensure!(
                si < slots.len(),
                "Qwen3.5 batched decode slot {si} outside executor slots {}",
                slots.len()
            );
            ensure!(
                slots[si].seq_len() == kv_seq_lens[r],
                "Qwen3.5 batched decode materialized seq_len {} != scheduler kv_seq_len {} for slot {si}",
                slots[si].seq_len(),
                kv_seq_lens[r]
            );
            // A decode-batch row's slot was activated at its start_pos==0 prefill;
            // its recurrent block MUST still be resident (the pointer tables below
            // dereference `gdr_states`).
            ensure!(
                slots[si].has_recurrent(),
                "Qwen3.6 batched decode: slot {si} recurrent state not acquired"
            );
            ensure!(
                kv_seq_lens[r] < self.max_seq_len,
                "Qwen3.5 batched decode sequence {} exceeds KV cache budget {}",
                kv_seq_lens[r] + 1,
                self.max_seq_len
            );
        }

        let c = &self.config;
        let eps = c.rms_norm_eps;
        let hidden_size = c.hidden_size;
        let vocab = self.output_projection().rows;

        // no-op when the row→slot mapping is unchanged.
        bd.stage_pointer_tables(&self.ctx, slots, slot_indices)?;

        let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let positions_host: Vec<i32> = kv_seq_lens.iter().map(|&len| len as i32).collect();
        let seq_lens_host: Vec<i32> = positions_host.iter().map(|&p| p + 1).collect();

        let Qwen35BatchDecodeState {
            ws,
            positions,
            seq_lens,
            full_k_cache_ptrs,
            full_v_cache_ptrs,
            conv_state_ptrs,
            gdr_state_ptrs,
            logits_batch,
            argmax,
            ..
        } = bd;
        let Qwen35Workspace {
            token_ids,
            hidden,
            normed,
            hidden_mid,
            attn_out,
            mlp_out,
            full,
            linear,
            dense,
            moe,
            logits: row_logits,
            ..
        } = ws;
        let token_ids = token_ids.upload(&self.ctx, &token_ids_host)?;
        let positions_dev = positions.upload(&self.ctx, &positions_host)?;
        let seq_lens_dev = seq_lens.upload(&self.ctx, &seq_lens_host)?;

        let hidden = hidden.get(&self.ctx, hidden_size, b)?;
        crate::profile::profile_op(&self.ctx, "embedding", None, b, || {
            embedding_batch(&self.ctx, &self.embed_tokens, token_ids, hidden)
        })?;
        let normed = normed.get(&self.ctx, hidden_size, b)?;
        let hidden_mid = hidden_mid.get(&self.ctx, hidden_size, b)?;
        let attn_out = attn_out.get(&self.ctx, hidden_size, b)?;
        let mlp_out = mlp_out.get(&self.ctx, hidden_size, b)?;

        let mut full_idx = 0usize;
        let mut linear_idx = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            crate::profile::profile_op(&self.ctx, "input_norm", Some(layer_idx), b, || {
                rms_norm_offset(&self.ctx, hidden, &layer.input_layernorm, eps, normed)
            })?;

            match &layer.attn {
                Qwen35Attn::Full(full_attn) => {
                    crate::profile::profile_op(
                        &self.ctx,
                        "full_attention",
                        Some(layer_idx),
                        b,
                        || {
                            self.full_attention_batch_rows(
                                full_attn,
                                normed,
                                slots,
                                slot_indices,
                                full_idx,
                                positions_dev,
                                seq_lens_dev,
                                &full_k_cache_ptrs[full_idx],
                                &full_v_cache_ptrs[full_idx],
                                full,
                                attn_out,
                            )
                        },
                    )?;
                    full_idx += 1;
                }
                Qwen35Attn::Linear(lin) => {
                    ensure!(
                        linear_idx < conv_state_ptrs.len(),
                        "Qwen3.5 batched decode linear layer {linear_idx} outside pointer tables {}",
                        conv_state_ptrs.len()
                    );
                    crate::profile::profile_op(
                        &self.ctx,
                        "linear_attention",
                        Some(layer_idx),
                        b,
                        || {
                            self.linear_attention(
                                lin,
                                normed,
                                LinearCore::Tables {
                                    conv: &conv_state_ptrs[linear_idx],
                                    gdr: &gdr_state_ptrs[linear_idx],
                                },
                                linear_idx,
                                linear,
                                attn_out,
                            )
                        },
                    )?;
                    linear_idx += 1;
                }
            }

            crate::profile::profile_op(&self.ctx, "post_attn_norm", Some(layer_idx), b, || {
                add_batch(&self.ctx, hidden, attn_out, hidden_mid)?;
                rms_norm_offset(
                    &self.ctx,
                    hidden_mid,
                    &layer.post_attention_layernorm,
                    eps,
                    normed,
                )
            })?;
            let mlp_in: &HiddenStates = normed;
            if let Some(moe_weights) = &layer.moe {
                let cfg = self
                    .moe_config
                    .as_ref()
                    .ok_or_else(|| anyhow!("MoE layer present but model has no moe_config"))?;
                crate::profile::profile_op(&self.ctx, "moe_ffn", Some(layer_idx), b, || {
                    moe_forward_into(
                        &self.ctx,
                        moe_weights,
                        mlp_in,
                        cfg,
                        &self.expert_split,
                        moe,
                        mlp_out,
                    )
                })?;
            } else {
                let mlp = layer
                    .mlp
                    .as_ref()
                    .ok_or_else(|| anyhow!("dense layer missing both mlp and moe weights"))?;
                crate::profile::profile_op(&self.ctx, "dense_ffn", Some(layer_idx), b, || {
                    self.dense_mlp(mlp, mlp_in, dense, mlp_out, None)
                })?;
            }
            // ONE all-reduce covers the whole FFN partial (see the per-layer
            // enumeration in the method docs); exact `[hidden, B]` message.
            crate::profile::profile_op(&self.ctx, "ffn_allreduce", Some(layer_idx), b, || {
                self.tp.all_reduce_sum(&self.ctx, mlp_out)
            })?;

            crate::profile::profile_op(&self.ctx, "ffn_residual", Some(layer_idx), b, || {
                add_batch(&self.ctx, hidden_mid, mlp_out, hidden)
            })?;
        }

        crate::profile::profile_op(&self.ctx, "final_norm", None, b, || {
            rms_norm_offset(&self.ctx, hidden, &self.norm, eps, normed)
        })?;
        let logits_buf = logits_batch.get(&self.ctx, vocab, b)?;
        crate::profile::profile_op(&self.ctx, "lm_head_gemm", None, b, || {
            gemm_batch(&self.ctx, self.output_projection(), normed, logits_buf)
        })?;
        crate::numeric_check::check_numeric(&self.ctx, &logits_buf.data, "qwen35_decode_logits");

        // Host seq_len advance: the device state (KV rows, conv rings, GDR
        // states) advanced in-stream above, so the host counters advance here
        // regardless of how sampling below fares — host and device stay
        // consistent (mirrors `forward_hidden`).
        for &si in slot_indices {
            slots[si].advance_seq_len(1);
        }

        let argmax_buf = argmax.get(&self.ctx, b)?;
        crate::profile::profile_op(&self.ctx, "sample", None, b, || {
            let (l_ptr, _gl) = logits_buf.data.device_ptr(&self.ctx.stream);
            let (a_ptr, _ga) = argmax_buf.device_ptr_mut(&self.ctx.stream);
            // SAFETY: logits is a live `[B, vocab]` bf16 buffer and argmax a
            // live `[B]` i32 buffer on ctx.stream.
            unsafe {
                ffi::argmax_batch_cuda(
                    l_ptr as *const ffi::Half,
                    a_ptr as *mut i32,
                    b as i32,
                    vocab as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
            }
            Ok(())
        })?;
        self.ctx.sync()?;
        let greedy_ids = self
            .ctx
            .stream
            .clone_dtoh(argmax_buf)
            .map_err(|e| anyhow!("D2H qwen35 batched argmax failed: {e}"))?;
        let out = params
            .iter()
            .enumerate()
            .map(|(r, p)| -> anyhow::Result<(u32, Option<f32>)> {
                if p.is_greedy() {
                    return Ok((greedy_ids[r] as u32, None));
                }
                let row_vec = row_logits.get(&self.ctx, vocab)?;
                copy_row_to_vec(&self.ctx, logits_buf, r, row_vec)?;
                let host = row_vec.to_host(&self.ctx)?;
                Ok(infer_plan::sample_token_logprob(
                    &host,
                    p,
                    sample_positions[r],
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(out)
    }

    pub(crate) fn forward_decode_batch_paged(
        &self,
        slots: &mut [Qwen35SlotState],
        bd: &mut Qwen35BatchDecodeState,
        pool: &mut PagedKVPool,
        meta: &crate::loader::PageMeta,
        slot_indices: &[usize],
        tokens: &[u32],
        kv_seq_lens: &[usize],
        params: &[SamplingParams],
        sample_positions: &[u64],
    ) -> Result<Vec<(u32, Option<f32>)>> {
        let b = tokens.len();
        ensure!(
            b >= 1,
            "Qwen3.6 paged batched decode requires at least one row"
        );
        ensure!(
            slot_indices.len() == b
                && kv_seq_lens.len() == b
                && params.len() == b
                && sample_positions.len() == b,
            "Qwen3.6 paged batched decode surface length mismatch: slots={} tokens={} kv_lens={} params={} positions={}",
            slot_indices.len(),
            b,
            kv_seq_lens.len(),
            params.len(),
            sample_positions.len()
        );
        ensure!(
            meta.batch == b && meta.total_q == b,
            "Qwen3.6 paged batched decode meta (batch {}, total_q {}) != {} one-token rows",
            meta.batch,
            meta.total_q,
            b
        );
        // Pool already holds POST-append length (engine appended one token per row before building `meta`).
        for (r, &si) in slot_indices.iter().enumerate() {
            ensure!(
                si < slots.len(),
                "Qwen3.6 paged batched decode slot {si} outside executor slots {}",
                slots.len()
            );
            ensure!(
                slots[si].seq_len() == kv_seq_lens[r],
                "Qwen3.6 paged batched decode materialized seq_len {} != scheduler kv_seq_len {} for slot {si}",
                slots[si].seq_len(),
                kv_seq_lens[r]
            );
            ensure!(
                slots[si].has_recurrent(),
                "Qwen3.6 paged batched decode: slot {si} recurrent state not acquired"
            );
            ensure!(
                pool.seq_len(si) == kv_seq_lens[r] + 1,
                "Qwen3.6 paged batched decode: pool seq_len {} != kv_seq_len+1 {} for slot {si}",
                pool.seq_len(si),
                kv_seq_lens[r] + 1
            );
        }

        let c = &self.config;
        let eps = c.rms_norm_eps;
        let hidden_size = c.hidden_size;
        let vocab = self.output_projection().rows;

        // Paged full-attn needs no contiguous K/V tables; no-op when row→slot mapping unchanged.
        bd.stage_recurrent_pointer_tables(&self.ctx, slots, slot_indices)?;

        let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();

        let Qwen35BatchDecodeState {
            ws,
            conv_state_ptrs,
            gdr_state_ptrs,
            logits_batch,
            argmax,
            ..
        } = bd;
        let Qwen35Workspace {
            token_ids,
            hidden,
            normed,
            hidden_mid,
            attn_out,
            mlp_out,
            full,
            linear,
            dense,
            moe,
            logits: row_logits,
            ..
        } = ws;
        let token_ids = token_ids.upload(&self.ctx, &token_ids_host)?;

        let hidden = hidden.get(&self.ctx, hidden_size, b)?;
        crate::profile::profile_op(&self.ctx, "embedding", None, b, || {
            embedding_batch(&self.ctx, &self.embed_tokens, token_ids, hidden)
        })?;
        let normed = normed.get(&self.ctx, hidden_size, b)?;
        let hidden_mid = hidden_mid.get(&self.ctx, hidden_size, b)?;
        let attn_out = attn_out.get(&self.ctx, hidden_size, b)?;
        let mlp_out = mlp_out.get(&self.ctx, hidden_size, b)?;

        let mut full_idx = 0usize;
        let mut linear_idx = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            crate::profile::profile_op(&self.ctx, "input_norm", Some(layer_idx), b, || {
                rms_norm_offset(&self.ctx, hidden, &layer.input_layernorm, eps, normed)
            })?;

            match &layer.attn {
                Qwen35Attn::Full(full_attn) => {
                    crate::profile::profile_op(
                        &self.ctx,
                        "full_attention",
                        Some(layer_idx),
                        b,
                        || {
                            self.full_attention_paged(
                                full_attn, normed, full_idx, pool, meta, full, attn_out, None,
                            )
                        },
                    )?;
                    full_idx += 1;
                }
                Qwen35Attn::Linear(lin) => {
                    ensure!(
                        linear_idx < conv_state_ptrs.len(),
                        "Qwen3.6 paged batched decode linear layer {linear_idx} outside pointer tables {}",
                        conv_state_ptrs.len()
                    );
                    crate::profile::profile_op(
                        &self.ctx,
                        "linear_attention",
                        Some(layer_idx),
                        b,
                        || {
                            self.linear_attention(
                                lin,
                                normed,
                                LinearCore::Tables {
                                    conv: &conv_state_ptrs[linear_idx],
                                    gdr: &gdr_state_ptrs[linear_idx],
                                },
                                linear_idx,
                                linear,
                                attn_out,
                            )
                        },
                    )?;
                    linear_idx += 1;
                }
            }

            crate::profile::profile_op(&self.ctx, "post_attn_norm", Some(layer_idx), b, || {
                add_batch(&self.ctx, hidden, attn_out, hidden_mid)?;
                rms_norm_offset(
                    &self.ctx,
                    hidden_mid,
                    &layer.post_attention_layernorm,
                    eps,
                    normed,
                )
            })?;
            let mlp_in: &HiddenStates = normed;
            if let Some(moe_weights) = &layer.moe {
                let cfg = self
                    .moe_config
                    .as_ref()
                    .ok_or_else(|| anyhow!("MoE layer present but model has no moe_config"))?;
                crate::profile::profile_op(&self.ctx, "moe_ffn", Some(layer_idx), b, || {
                    moe_forward_into(
                        &self.ctx,
                        moe_weights,
                        mlp_in,
                        cfg,
                        &self.expert_split,
                        moe,
                        mlp_out,
                    )
                })?;
            } else {
                let mlp = layer
                    .mlp
                    .as_ref()
                    .ok_or_else(|| anyhow!("dense layer missing both mlp and moe weights"))?;
                crate::profile::profile_op(&self.ctx, "dense_ffn", Some(layer_idx), b, || {
                    self.dense_mlp(mlp, mlp_in, dense, mlp_out, None)
                })?;
            }
            crate::profile::profile_op(&self.ctx, "ffn_allreduce", Some(layer_idx), b, || {
                self.tp.all_reduce_sum(&self.ctx, mlp_out)
            })?;
            crate::profile::profile_op(&self.ctx, "ffn_residual", Some(layer_idx), b, || {
                add_batch(&self.ctx, hidden_mid, mlp_out, hidden)
            })?;
        }

        crate::profile::profile_op(&self.ctx, "final_norm", None, b, || {
            rms_norm_offset(&self.ctx, hidden, &self.norm, eps, normed)
        })?;
        let logits_buf = logits_batch.get(&self.ctx, vocab, b)?;
        crate::profile::profile_op(&self.ctx, "lm_head_gemm", None, b, || {
            gemm_batch(&self.ctx, self.output_projection(), normed, logits_buf)
        })?;
        crate::numeric_check::check_numeric(&self.ctx, &logits_buf.data, "qwen35_paged_decode_logits");

        // Host seq_len advance (device KV/conv/GDR advanced in-stream above).
        for &si in slot_indices {
            slots[si].advance_seq_len(1);
        }

        let argmax_buf = argmax.get(&self.ctx, b)?;
        crate::profile::profile_op(&self.ctx, "sample", None, b, || {
            let (l_ptr, _gl) = logits_buf.data.device_ptr(&self.ctx.stream);
            let (a_ptr, _ga) = argmax_buf.device_ptr_mut(&self.ctx.stream);
            // SAFETY: logits `[B, vocab]` bf16, argmax `[B]` i32, both on ctx.stream.
            unsafe {
                ffi::argmax_batch_cuda(
                    l_ptr as *const ffi::Half,
                    a_ptr as *mut i32,
                    b as i32,
                    vocab as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
            }
            Ok(())
        })?;
        self.ctx.sync()?;
        let greedy_ids = self
            .ctx
            .stream
            .clone_dtoh(argmax_buf)
            .map_err(|e| anyhow!("D2H qwen36 paged batched argmax failed: {e}"))?;
        let out = params
            .iter()
            .enumerate()
            .map(|(r, p)| -> anyhow::Result<(u32, Option<f32>)> {
                if p.is_greedy() {
                    return Ok((greedy_ids[r] as u32, None));
                }
                let row_vec = row_logits.get(&self.ctx, vocab)?;
                copy_row_to_vec(&self.ctx, logits_buf, r, row_vec)?;
                let host = row_vec.to_host(&self.ctx)?;
                Ok(infer_plan::sample_token_logprob(
                    &host,
                    p,
                    sample_positions[r],
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(out)
    }
}
