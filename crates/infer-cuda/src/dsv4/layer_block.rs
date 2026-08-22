use super::*;

/// Selects the weights and the profile labels.
#[derive(Copy, Clone)]
pub(super) enum HcHalf {
    Attn,
    Ffn,
}

impl HcHalf {
    fn norm(self, layer: &Dsv4Layer) -> &DeviceVec {
        match self {
            HcHalf::Attn => &layer.attn_norm,
            HcHalf::Ffn => &layer.ffn_norm,
        }
    }

    fn hc(self, layer: &Dsv4Layer) -> &Dsv4HyperConnection {
        match self {
            HcHalf::Attn => &layer.hc_attn,
            HcHalf::Ffn => &layer.hc_ffn,
        }
    }

    fn params_label(self) -> &'static str {
        match self {
            HcHalf::Attn => "attn_hc_params",
            HcHalf::Ffn => "ffn_hc_params",
        }
    }

    fn pre_norm_label(self) -> &'static str {
        match self {
            HcHalf::Attn => "attn_hc_pre_norm",
            HcHalf::Ffn => "ffn_hc_pre_norm",
        }
    }

    fn post_label(self) -> &'static str {
        match self {
            HcHalf::Attn => "attn_hc_post",
            HcHalf::Ffn => "ffn_hc_post",
        }
    }
}

impl Dsv4Model {
    pub(super) fn attn_ctx<'a>(
        &'a self,
        layer: &'a Dsv4Layer,
        layer_idx: usize,
        pos: crate::attention::Dsv4Position<'a>,
        chain_verify: Option<&'a crate::attention::Dsv4ChainVerifyAttnMeta>,
    ) -> crate::attention::Dsv4AttnCtx<'a> {
        crate::attention::Dsv4AttnCtx {
            ctx: &self.ctx,
            config: &self.config,
            attention: &layer.attention,
            mode: layer.mode,
            compress_ratio: layer.compress_ratio,
            layer_idx,
            tp: &self.tp,
            pos,
            chain_verify,
        }
    }

    /// `stream` must be `[stream_dim, seq_len]`.
    pub(super) fn embed_stream(
        &self,
        token_ids: &CudaSlice<i32>,
        seq_len: usize,
        stream: &mut HiddenStates,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<()> {
        let hidden_size = self.config.hidden_size;
        // SAFETY: embedding_batch writes the full [seq_len, hidden_size] buffer.
        let mut embeddings = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
        crate::profile::profile_op(&self.ctx, "embedding", None, seq_len, || {
            crate::ops::embedding_batch(&self.ctx, &self.embed_tokens, token_ids, &mut embeddings)
                .map_err(|e| anyhow!("DSv4 embed lookup failed: {e}"))?;
            crate::hc::initial_stream_from_embeddings(
                &self.ctx,
                &embeddings,
                hidden_size,
                self.config.hc_mult,
                stream,
            )
        })?;
        keepalive.keep_hidden(&embeddings);
        keepalive.keep_hidden(stream);
        Ok(())
    }

    /// `None` ⇒ GLM (`hc_mult == 1`), where the C identity placeholders are 1x1
    /// ZERO mixers: `gen_mhc`/`mhc_pre` would zero the stream, so a plain
    /// RMSNorm runs instead.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_pre_norm(
        &self,
        layer: &Dsv4Layer,
        half: HcHalf,
        layer_idx: usize,
        seq_len: usize,
        stream: &HiddenStates,
        normed: &mut HiddenStates,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<Option<crate::hc::MhcParams>> {
        let norm = half.norm(layer);
        let eps = self.config.rms_norm_eps;
        if self.config.is_glm() {
            crate::profile::profile_op(
                &self.ctx,
                half.pre_norm_label(),
                Some(layer_idx),
                seq_len,
                || crate::ops::rms_norm_batch(&self.ctx, stream, norm, eps, normed),
            )?;
            return Ok(None);
        }
        let mhc = crate::profile::profile_op(
            &self.ctx,
            half.params_label(),
            Some(layer_idx),
            seq_len,
            || crate::hc::gen_mhc_params(&self.ctx, &self.config, half.hc(layer), stream),
        )?;
        keepalive.keep_f32(&mhc.pre);
        keepalive.keep_f32(&mhc.post);
        keepalive.keep_f32(&mhc.comb);
        crate::profile::profile_op(
            &self.ctx,
            half.pre_norm_label(),
            Some(layer_idx),
            seq_len,
            || {
                crate::hc::mhc_pre_rms_norm(
                    &self.ctx,
                    stream,
                    &mhc.pre,
                    norm,
                    eps,
                    self.config.hidden_size,
                    self.config.hc_mult,
                    normed,
                )
            },
        )?;
        Ok(Some(mhc))
    }

    /// `mhc == None` ⇒ GLM plain residual (`stream_dim == hidden_size`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_post_fold(
        &self,
        mhc: Option<&crate::hc::MhcParams>,
        half: HcHalf,
        layer_idx: usize,
        seq_len: usize,
        block_out: &HiddenStates,
        stream: &HiddenStates,
        out: &mut HiddenStates,
    ) -> Result<()> {
        crate::profile::profile_op(
            &self.ctx,
            half.post_label(),
            Some(layer_idx),
            seq_len,
            || match mhc {
                Some(mhc) => crate::hc::hc_post(
                    &self.ctx,
                    block_out,
                    stream,
                    &mhc.post,
                    &mhc.comb,
                    self.config.hidden_size,
                    self.config.hc_mult,
                    out,
                ),
                None => crate::ops::add_batch(&self.ctx, block_out, stream, out),
            },
        )
    }
}
