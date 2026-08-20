//! The head tail: stream-row HC fold, final RMSNorm, lm_head projection,
//! sampling.

use super::*;

impl Dsv4Model {
    /// Head recipe for stream rows `rows`: per-row HC fold (GLM `hc_mult == 1`
    /// identity) + final RMSNorm, written to `out` rows `0..rows.len()`.
    pub(in crate::dsv4) fn head_normed_rows(
        &self,
        stream: &HiddenStates,
        rows: std::ops::Range<usize>,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let hidden_size = self.config.hidden_size;
        let eps = self.config.rms_norm_eps;
        ensure!(
            stream.seq_len >= rows.end
                && out.hidden_dim == hidden_size
                && out.seq_len >= rows.len(),
            "DSv4 head rows {rows:?} out of stream rows {} / out {}x{}",
            stream.seq_len,
            out.hidden_dim,
            out.seq_len
        );
        let mut last_hidden = DeviceVec::zeros(&self.ctx, hidden_size)?;
        let mut last_normed = DeviceVec::zeros(&self.ctx, hidden_size)?;
        for (i, row) in rows.enumerate() {
            if self.config.is_glm() {
                // GLM: head hidden = stream row (stream_dim==hidden, no head HC mixer).
                // ponytail: pod-verify GLM hc_mult==1 head hidden = stream row
                // (identity)
                crate::ops::copy_row_to_vec(&self.ctx, stream, row, &mut last_hidden)?;
            } else {
                crate::hc::head_hidden_from_stream(
                    &self.ctx,
                    &self.config,
                    &self.head_hc,
                    stream,
                    row,
                    &mut last_hidden,
                )?;
            }
            crate::ops::rms_norm_vec(&self.ctx, &last_hidden, &self.norm, eps, &mut last_normed)?;
            let mut dst = out.data.slice_mut(i * hidden_size..(i + 1) * hidden_size);
            self.ctx
                .stream
                .memcpy_dtod(&last_normed.data, &mut dst)
                .map_err(|e| anyhow!("DSv4 verify head row copy failed: {e}"))?;
        }
        Ok(())
    }

    /// Fold every row's stream into a full target logits matrix.
    pub(super) fn verify_logits_from_stream(
        &self,
        stream: &HiddenStates,
        rows: usize,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<HiddenStates> {
        ensure!(rows > 0, "DSv4 verify logits requires at least one row");
        ensure!(
            stream.seq_len >= rows,
            "DSv4 verify stream rows {} < requested logits rows {rows}",
            stream.seq_len
        );
        let hidden_size = self.config.hidden_size;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut head_normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, rows)? };
        self.head_normed_rows(stream, 0..rows, &mut head_normed)?;
        keepalive.keep_hidden(&head_normed);
        // SAFETY: uninit device scratch; fully written before first read.
        let mut logits = unsafe { HiddenStates::uninit(&self.ctx, self.lm_head.rows, rows)? };
        self.lm_head_project_batch(&head_normed, &mut logits)?;
        Ok(logits)
    }

    pub(super) fn capture_mtp_stream_hidden(
        &self,
        stream: &HiddenStates,
        offset: usize,
        len: usize,
        out: &mut DeviceVec,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<()> {
        let stream_dim = self.config.hidden_size * self.config.hc_mult;
        ensure!(
            stream.hidden_dim == stream_dim,
            "DSv4 MTP hidden source stream dim {} != hidden_size {} * hc_mult {}",
            stream.hidden_dim,
            self.config.hidden_size,
            self.config.hc_mult
        );
        let elems = len * stream_dim;
        ensure!(
            out.len >= elems,
            "DSv4 MTP hidden capture len {} < elems {elems}",
            out.len
        );
        let src = stream
            .data
            .slice(offset * stream_dim..offset * stream_dim + elems);
        let mut dst = out.data.slice_mut(0..elems);
        self.ctx
            .stream
            .memcpy_dtod(&src, &mut dst)
            .map_err(|e| anyhow!("DSv4 MTP hidden capture D2D failed: {e}"))?;
        keepalive.keep_vec(out);
        Ok(())
    }

    pub(super) fn forward_stream_last_token(
        &self,
        stream: &HiddenStates,
        seq_len: usize,
        params: &SamplingParams,
        position: u64,
        penalty: infer_plan::PenaltyHistory<'_>,
        last_hidden_out: Option<&mut DeviceVec>,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<u32> {
        ensure!(seq_len > 0, "DSv4 head stage requires seq_len > 0");
        let hidden_size = self.config.hidden_size;
        let eps = self.config.rms_norm_eps;
        let ctx = &self.ctx;

        let mut last_hidden = DeviceVec::zeros(ctx, hidden_size)?;
        if self.config.is_glm() {
            // GLM: head hidden = stream row (stream_dim==hidden, no head HC mixer).
            // ponytail: pod-verify GLM hc_mult==1 head hidden = stream row (identity)
            crate::profile::profile_op(ctx, "head_hc", None, seq_len, || {
                crate::ops::copy_row_to_vec(ctx, stream, seq_len - 1, &mut last_hidden)
            })?;
        } else {
            crate::profile::profile_op(ctx, "head_hc", None, seq_len, || {
                crate::hc::head_hidden_from_stream(
                    ctx,
                    &self.config,
                    &self.head_hc,
                    stream,
                    seq_len - 1,
                    &mut last_hidden,
                )
            })?;
        }
        keepalive.keep_hidden(stream);
        keepalive.keep_vec(&last_hidden);
        if let Some(out) = last_hidden_out {
            ensure!(
                out.len == hidden_size,
                "DSv4 hidden capture len {} != hidden_size {hidden_size}",
                out.len
            );
            ctx.stream
                .memcpy_dtod(&last_hidden.data, &mut out.data)
                .map_err(|e| anyhow!("DSv4 hidden capture D2D failed: {e}"))?;
            keepalive.keep_vec(out);
        }

        let mut last_normed = DeviceVec::zeros(ctx, hidden_size)?;
        crate::profile::profile_op(ctx, "head_norm", None, seq_len, || {
            crate::ops::rms_norm_vec(ctx, &last_hidden, &self.norm, eps, &mut last_normed)
        })?;
        keepalive.keep_vec(&last_normed);
        let mut logits = DeviceVec::zeros(ctx, self.lm_head.rows)?;
        crate::profile::profile_op(ctx, "lm_head_project", None, seq_len, || {
            self.lm_head_project(&last_normed, &mut logits)
        })?;
        keepalive.keep_vec(&logits);
        let token = crate::profile::profile_op(ctx, "sample", None, seq_len, || {
            self.sample_logits(&logits, params, position, penalty)
        })?;
        Ok(token)
    }

    /// Sample the next token from the decode-tail logits, or the cross-rank merge
    /// when the vocab-sharded lm_head is active (`logits` = this rank's padded slice).
    pub(super) fn sample_logits(
        &self,
        logits: &DeviceVec,
        params: &SamplingParams,
        position: u64,
        penalty: infer_plan::PenaltyHistory<'_>,
    ) -> Result<u32> {
        crate::executor::sample_cuda_token(&self.ctx, logits, params, position, penalty)
    }

    /// Batched lm_head: `[m, hidden] → [m, vocab]`, one GEMM for every weight
    /// format (`dsv4_linear`).
    pub(in crate::dsv4) fn lm_head_project_batch(
        &self,
        x: &HiddenStates,
        out: &mut HiddenStates,
    ) -> Result<()> {
        ensure!(
            self.lm_head.cols == x.hidden_dim
                && self.lm_head.rows == out.hidden_dim
                && x.seq_len == out.seq_len,
            "DSv4 lm_head batch shape mismatch: [{}x{}] x {}x{} out {}x{}",
            self.lm_head.rows,
            self.lm_head.cols,
            x.hidden_dim,
            x.seq_len,
            out.hidden_dim,
            out.seq_len
        );
        // The lm_head weight is read ONCE for all `m` rows. The old per-row GEMV
        // re-read the whole ~1 GB bf16 weight PER ROW — at c=16×depth verify ~48×
        // the HBM traffic and the #1 decode kernel (nsys 19%, 558µs).
        crate::attention::dsv4_linear(&self.ctx, &self.lm_head, x, out)
    }

    /// Project the final hidden vector through the LM head. The head can be bf16
    /// or DSv4 FP8/FP4 block-scaled, so dispatch the matching kernel.
    pub(super) fn lm_head_project(&self, x: &DeviceVec, logits: &mut DeviceVec) -> Result<()> {
        use cuda_kernels::tensor::WeightFormat;
        ensure!(
            self.lm_head.cols == x.len && self.lm_head.rows == logits.len,
            "DSv4 lm_head shape mismatch: [{}x{}] x.len {} logits.len {}",
            self.lm_head.rows,
            self.lm_head.cols,
            x.len,
            logits.len
        );
        match self.lm_head.weight_format {
            WeightFormat::DenseBf16 => crate::ops::gemv(&self.ctx, &self.lm_head, x, logits),
            // FP8/FP4 block-scaled: run the batched GEMV path at batch=1, then
            // copy the one-token output row into the caller's logits vec.
            WeightFormat::Dsv4Fp8BlockScaled | WeightFormat::Dsv4Fp4BlockScaled => {
                let x_batch = HiddenStates {
                    data: x.data.clone(),
                    hidden_dim: x.len,
                    seq_len: 1,
                };
                // SAFETY: mla_linear writes the full one-token logits batch.
                let mut out_batch = unsafe { HiddenStates::uninit(&self.ctx, logits.len, 1)? };
                crate::attention::mla_linear(&self.ctx, &self.lm_head, &x_batch, &mut out_batch)?;
                self.ctx
                    .stream
                    .memcpy_dtod(&out_batch.data, &mut logits.data)
                    .map_err(|e| anyhow!("DSv4 lm_head logits copy-back failed: {e}"))?;
                Ok(())
            }
            other => anyhow::bail!("DSv4 lm_head unsupported weight format {other:?}"),
        }
    }
}
