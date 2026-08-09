use super::*;

pub(crate) struct DenseMlp {
    /// Row-fused `[gate; up]` (`[2*inter, hidden]`): one GEMM per step feeds
    /// the fused SwiGLU. Gate = rows `0..inter`, up = rows `inter..2*inter`.
    pub(crate) gate_up_proj: DeviceMatrix,
    pub(crate) down_proj: DeviceMatrix,
}

impl Qwen35Model {
    pub(crate) fn dense_mlp(
        &self,
        mlp: &DenseMlp,
        normed: &HiddenStates,
        dw: &mut DenseMlpScratch,
        out: &mut HiddenStates,
        probe: Option<(usize, usize)>,
    ) -> Result<()> {
        let inter = mlp.inter_dim();
        let seq_len = normed.seq_len;
        let gate_up = dw.gate_up.get(&self.ctx, 2 * inter, seq_len)?;
        gemm_batch(&self.ctx, &mlp.gate_up_proj, normed, gate_up)?;
        #[cfg(feature = "cuda")]
        self.stage_mlp_step(probe, "gate_up", gate_up);
        let act = dw.act.get(&self.ctx, inter, seq_len)?;
        silu_mul_fused(&self.ctx, gate_up, act)?;
        #[cfg(feature = "cuda")]
        self.stage_mlp_step(probe, "silu_act", act);
        gemm_batch(&self.ctx, &mlp.down_proj, act, out)?;
        Ok(())
    }

    pub(crate) fn stage_mlp_step(
        &self,
        probe: Option<(usize, usize)>,
        stage: &str,
        buf: &HiddenStates,
    ) {
        let Some((layer_idx, start_pos)) = probe else {
            return;
        };
        let width = buf.hidden_dim;
        for r in 0..buf.seq_len {
            let pos = (start_pos + r) as u64;
            if crate::probe::stage_want(pos) {
                crate::probe::stage_bf16(
                    &self.ctx,
                    stage,
                    layer_idx,
                    r,
                    pos,
                    buf.data.slice(r * width..(r + 1) * width),
                );
            }
        }
    }
}
