//! The doc(hidden) probes the finite-difference gate example drives.

use super::*;

impl Qwen35Model {
    #[doc(hidden)]
    pub fn forward_profiled_for_diagnostics(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
        trace: bool,
    ) -> Result<(TensorId, Qwen35RolloutForwardProfile)> {
        let token_indices = input_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let positions = position_ids
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        self.forward_batch_indices_profiled(store, tape, &token_indices, &positions, 1, trace)
    }

    #[doc(hidden)]
    pub fn forward_with_moe_routes_for_diagnostics(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
    ) -> Result<(TensorId, Vec<Qwen35MoeRouteSignature>)> {
        let mut route_signatures = Vec::new();
        let logits = self.forward_moe_routes_for_diagnostics(
            store,
            tape,
            input_ids,
            position_ids,
            &mut MoeRouteMode::Collect(&mut route_signatures),
        )?;
        Ok((logits, route_signatures))
    }

    #[doc(hidden)]
    pub fn forward_with_frozen_moe_routes_for_diagnostics(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
        frozen_routes: &[Qwen35MoeRouteSignature],
    ) -> Result<TensorId> {
        let mut next = 0usize;
        self.forward_moe_routes_for_diagnostics(
            store,
            tape,
            input_ids,
            position_ids,
            &mut MoeRouteMode::Frozen {
                signatures: frozen_routes,
                next: &mut next,
            },
        )
    }

    pub(super) fn forward_moe_routes_for_diagnostics(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
        mode: &mut MoeRouteMode<'_>,
    ) -> Result<TensorId> {
        let seq_len = position_ids.len();
        if input_ids.len() != seq_len {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: input_ids.len(),
                expected_len: seq_len,
            });
        }
        let max_seq_len = self
            .config
            .rope_cache_len_hint
            .ok_or(Qwen35Error::InvalidConfig(
                "train-side qwen3.5 requires rope_cache_len_hint",
            ))?;
        if seq_len > max_seq_len {
            return Err(Qwen35Error::InvalidConfig(
                "sequence length exceeds configured rope cache length",
            ));
        }

        let token_indices = input_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let positions = position_ids
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        let cos = select_cache_rows(self.cos_cache, &positions, store)?;
        let sin = select_cache_rows(self.sin_cache, &positions, store)?;

        let mut hidden = embedding(self.embed_tokens, &token_indices, store, tape)?;
        hidden = reshape(hidden, &[1, seq_len, self.config.hidden_size], store, tape)?;
        for layer in &self.layers {
            hidden = layer.forward_moe_routes(
                hidden,
                &self.config,
                self.tp,
                cos,
                sin,
                mode,
                store,
                tape,
            )?;
        }
        if let MoeRouteMode::Frozen { signatures, next } = mode
            && **next != signatures.len()
        {
            return Err(Qwen35Error::InvalidConfig(
                "frozen MoE routes contain unused signatures",
            ));
        }
        let hidden = qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )?;
        linear_forward(hidden, self.lm_head, store, tape)
    }

    #[doc(hidden)]
    pub fn forward_mlp_for_diagnostics(
        &self,
        layer_idx: usize,
        hidden: TensorId,
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or(Qwen35Error::InvalidConfig(
                "diagnostic MLP layer index out of range",
            ))?;
        layer.forward_mlp(
            hidden,
            &self.config,
            self.tp,
            batch,
            seq_len,
            &mut MoeRouteMode::Free,
            store,
            tape,
        )
    }

    #[doc(hidden)]
    pub fn forward_lm_head_tail_for_diagnostics(
        &self,
        hidden: TensorId,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let hidden = qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )?;
        linear_forward(hidden, self.lm_head, store, tape)
    }
}
