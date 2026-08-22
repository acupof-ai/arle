use super::*;

impl Qwen35Layer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_mlp(
        &self,
        h: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        batch: usize,
        seq_len: usize,
        mode: &mut MoeRouteMode<'_>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        match &self.mlp {
            Qwen35Mlp::Dense(mlp) => {
                let gate_raw = mlp.gate_proj.forward(h, store, tape)?;
                let up = mlp.up_proj.forward(h, store, tape)?;
                let gate = silu(gate_raw, store, tape)?;
                let act = mul(gate, up, store, tape)?;
                let mlp_out = mlp.down_proj.forward(act, store, tape)?;
                // tape-disabled checkpoint forward: free dead transients now
                // instead of at closure exit, cutting the single-layer peak.
                if !tape.enabled {
                    for id in [gate_raw, gate, up, act] {
                        store.free(id)?;
                    }
                }
                Ok(maybe_all_reduce(mlp_out, tp, store, tape)?)
            }
            Qwen35Mlp::Sparse(mlp) => {
                self.forward_sparse_mlp(mlp, h, cfg, tp, batch, seq_len, mode, store, tape)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_sparse_mlp(
        &self,
        mlp: &Qwen35SparseMlp,
        h: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        batch: usize,
        seq_len: usize,
        mode: &mut MoeRouteMode<'_>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let tokens = batch * seq_len;
        let flat_h = reshape(h, &[tokens, cfg.hidden_size], store, tape)?;
        let router_logits = mlp.router_gate.forward(flat_h, store, tape)?;
        let routes = match mode {
            MoeRouteMode::Free => moe_topk_softmax(router_logits, mlp.top_k, store, tape)?,
            MoeRouteMode::Collect(signatures) => {
                let routes = moe_topk_softmax(router_logits, mlp.top_k, store, tape)?;
                signatures.push(Qwen35MoeRouteSignature {
                    layer: self.index,
                    tokens: routes.tokens,
                    experts: routes.experts,
                    top_k: routes.top_k,
                    indices: routes.indices.clone(),
                });
                routes
            }
            MoeRouteMode::Frozen { signatures, next } => {
                let signature = signatures.get(**next).ok_or(Qwen35Error::InvalidConfig(
                    "frozen MoE routes missing sparse layer signature",
                ))?;
                **next += 1;
                validate_qwen35_moe_route_signature(
                    signature,
                    self.index,
                    tokens,
                    mlp.experts.len(),
                    mlp.top_k,
                )?;
                moe_topk_softmax_with_indices(
                    router_logits,
                    mlp.top_k,
                    &signature.indices,
                    store,
                    tape,
                )?
            }
        };
        let grouped_routes = build_qwen35_grouped_routes(&routes)?;

        let gate_experts = qwen35_grouped_linear_experts(&mlp.experts, |expert| &expert.gate_proj);
        let up_experts = qwen35_grouped_linear_experts(&mlp.experts, |expert| &expert.up_proj);
        let down_experts = qwen35_grouped_linear_experts(&mlp.experts, |expert| &expert.down_proj);

        let routed_gate = moe_grouped_linear(
            flat_h,
            &gate_experts,
            &grouped_routes,
            MoeGroupedLinearInput::TokenRows,
            store,
            tape,
        )?;
        let routed_up = moe_grouped_linear(
            flat_h,
            &up_experts,
            &grouped_routes,
            MoeGroupedLinearInput::TokenRows,
            store,
            tape,
        )?;
        let routed_gate = silu(routed_gate, store, tape)?;
        let routed_hidden = mul(routed_gate, routed_up, store, tape)?;
        let routed_down = moe_grouped_linear(
            routed_hidden,
            &down_experts,
            &grouped_routes,
            MoeGroupedLinearInput::Packed,
            store,
            tape,
        )?;
        let routed = moe_grouped_weighted_scatter(
            routed_down,
            routes.weights,
            &grouped_routes,
            tokens,
            store,
            tape,
        )?;

        let shared_gate = mlp.shared_gate_proj.forward(flat_h, store, tape)?;
        let shared_up = mlp.shared_up_proj.forward(flat_h, store, tape)?;
        let shared_gate = silu(shared_gate, store, tape)?;
        let shared_hidden = mul(shared_gate, shared_up, store, tape)?;
        let shared = mlp.shared_down_proj.forward(shared_hidden, store, tape)?;
        let shared_expert_gate = mlp.shared_expert_gate.forward(flat_h, store, tape)?;
        let shared_expert_gate = sigmoid(shared_expert_gate, store, tape)?;
        let shared_expert_gate =
            broadcast_to_shape(shared_expert_gate, &[tokens, cfg.hidden_size], store, tape)?;
        let shared = mul(shared, shared_expert_gate, store, tape)?;

        // Row-parallel down_proj (routed + shared) yields per-rank partial sums;
        // one all-reduce on their sum completes both across the TP group.
        let out = add(routed, shared, store, tape)?;
        let out = maybe_all_reduce(out, tp, store, tape)?;
        Ok(reshape(
            out,
            &[batch, seq_len, cfg.hidden_size],
            store,
            tape,
        )?)
    }
}

pub(super) fn build_qwen35_grouped_routes(routes: &MoeTopK) -> Result<Vec<MoeGroupedRoute>> {
    if routes.indices.len() != routes.tokens * routes.top_k {
        return Err(AutogradError::InvalidIndicesLen {
            expected: routes.tokens * routes.top_k,
            got: routes.indices.len(),
        }
        .into());
    }
    let mut counts = vec![0usize; routes.experts];
    let mut grouped = Vec::with_capacity(routes.indices.len());
    for token in 0..routes.tokens {
        for slot in 0..routes.top_k {
            let expert = routes.indices[token * routes.top_k + slot];
            if expert >= routes.experts {
                return Err(AutogradError::IndexOutOfBounds {
                    index: expert,
                    upper: routes.experts,
                }
                .into());
            }
            let row = counts[expert];
            counts[expert] += 1;
            grouped.push(MoeGroupedRoute {
                expert,
                row,
                token,
                slot,
            });
        }
    }
    grouped.sort_by_key(|route| (route.expert, route.row));
    Ok(grouped)
}

pub(super) fn validate_qwen35_moe_route_signature(
    route: &Qwen35MoeRouteSignature,
    layer: usize,
    tokens: usize,
    experts: usize,
    top_k: usize,
) -> Result<()> {
    if route.layer != layer {
        return Err(Qwen35Error::InvalidConfig(
            "frozen MoE route layer index mismatch",
        ));
    }
    if route.tokens != tokens || route.experts != experts || route.top_k != top_k {
        return Err(Qwen35Error::InvalidConfig(
            "frozen MoE route shape mismatch",
        ));
    }
    if route.indices.len() != tokens * top_k {
        return Err(AutogradError::InvalidIndicesLen {
            expected: tokens * top_k,
            got: route.indices.len(),
        }
        .into());
    }
    for &expert in &route.indices {
        if expert >= experts {
            return Err(AutogradError::IndexOutOfBounds {
                index: expert,
                upper: experts,
            }
            .into());
        }
    }
    Ok(())
}

pub(super) fn qwen35_grouped_linear_experts(
    experts: &[Qwen35SparseExpert],
    select: impl Fn(&Qwen35SparseExpert) -> &LinearWithLora,
) -> Vec<MoeGroupedLinearExpert> {
    experts
        .iter()
        .map(|expert| {
            let parts = select(expert).parts();
            MoeGroupedLinearExpert {
                weight: parts.weight,
                lora_a: parts.lora_a,
                lora_b: parts.lora_b,
                lora_scale: parts.lora_scale,
            }
        })
        .collect()
}
