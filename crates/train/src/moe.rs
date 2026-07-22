use std::collections::HashMap;

use autograd::{
    AutogradError, Result, Tape, TensorId, TensorStore,
    ops::{
        MoeGroupedLinearExpert, MoeGroupedLinearInput, MoeGroupedRoute, MoeTopK,
        moe_grouped_linear, moe_grouped_weighted_scatter, moe_topk_softmax, mul, silu,
    },
};

use crate::lora::{LinearWithLora, LoraConfig, leak_name};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub lora: LoraConfig,
}

#[derive(Debug, Clone)]
pub struct MoeWithLora {
    router: LinearWithLora,
    experts: Vec<SwiGluExpert>,
    cfg: MoeConfig,
}

#[derive(Debug, Clone)]
struct SwiGluExpert {
    gate: LinearWithLora,
    up: LinearWithLora,
    down: LinearWithLora,
}

impl MoeWithLora {
    pub fn new(prefix: &'static str, cfg: MoeConfig, store: &mut TensorStore) -> Result<Self> {
        validate_config(cfg)?;
        let router = LinearWithLora::new(
            leak_name(format!("{prefix}.router.weight")),
            cfg.hidden_size,
            cfg.num_experts,
            true,
            None,
            store,
        )?;

        let experts = (0..cfg.num_experts)
            .map(|expert| {
                let base = format!("{prefix}.experts.{expert}");
                Ok(SwiGluExpert {
                    gate: LinearWithLora::new(
                        leak_name(format!("{base}.gate_proj.weight")),
                        cfg.hidden_size,
                        cfg.intermediate_size,
                        false,
                        Some(cfg.lora),
                        store,
                    )?,
                    up: LinearWithLora::new(
                        leak_name(format!("{base}.up_proj.weight")),
                        cfg.hidden_size,
                        cfg.intermediate_size,
                        false,
                        Some(cfg.lora),
                        store,
                    )?,
                    down: LinearWithLora::new(
                        leak_name(format!("{base}.down_proj.weight")),
                        cfg.intermediate_size,
                        cfg.hidden_size,
                        false,
                        Some(cfg.lora),
                        store,
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            router,
            experts,
            cfg,
        })
    }

    pub fn forward(
        &self,
        x: TensorId,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let x_shape = store
            .get(x)
            .ok_or(AutogradError::InvalidTensorId(x))?
            .shape
            .clone();
        if x_shape.len() != 2 {
            return Err(AutogradError::InvalidRank {
                expected: "2",
                got: x_shape.len(),
            });
        }
        let tokens = x_shape[0];
        if x_shape[1] != self.cfg.hidden_size {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![tokens, self.cfg.hidden_size],
                got: x_shape,
            });
        }

        let router_logits = self.router.forward(x, store, tape)?;
        let routes = moe_topk_softmax(router_logits, self.cfg.top_k, store, tape)?;
        let grouped_routes = build_grouped_routes(&routes)?;
        let gate_experts = grouped_linear_experts(&self.experts, |expert| &expert.gate);
        let up_experts = grouped_linear_experts(&self.experts, |expert| &expert.up);
        let down_experts = grouped_linear_experts(&self.experts, |expert| &expert.down);

        let gate = moe_grouped_linear(
            x,
            &gate_experts,
            &grouped_routes,
            MoeGroupedLinearInput::TokenRows,
            store,
            tape,
        )?;
        let up = moe_grouped_linear(
            x,
            &up_experts,
            &grouped_routes,
            MoeGroupedLinearInput::TokenRows,
            store,
            tape,
        )?;
        let activated = silu(gate, store, tape)?;
        let hidden = mul(activated, up, store, tape)?;
        let down = moe_grouped_linear(
            hidden,
            &down_experts,
            &grouped_routes,
            MoeGroupedLinearInput::Packed,
            store,
            tape,
        )?;
        moe_grouped_weighted_scatter(down, routes.weights, &grouped_routes, tokens, store, tape)
    }

    pub fn parameter_name_map(&self) -> HashMap<&'static str, TensorId> {
        let mut out = self.router.parameter_name_map();
        for expert in &self.experts {
            out.extend(expert.gate.parameter_name_map());
            out.extend(expert.up.parameter_name_map());
            out.extend(expert.down.parameter_name_map());
        }
        out
    }

    pub fn adapter_name_map(&self) -> HashMap<&'static str, TensorId> {
        let mut out = HashMap::new();
        for expert in &self.experts {
            out.extend(expert.gate.adapter_name_map());
            out.extend(expert.up.adapter_name_map());
            out.extend(expert.down.adapter_name_map());
        }
        out
    }
}

fn build_grouped_routes(routes: &MoeTopK) -> Result<Vec<MoeGroupedRoute>> {
    if routes.indices.len() != routes.tokens * routes.top_k {
        return Err(AutogradError::InvalidIndicesLen {
            expected: routes.tokens * routes.top_k,
            got: routes.indices.len(),
        });
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
                });
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

fn grouped_linear_experts(
    experts: &[SwiGluExpert],
    select: impl Fn(&SwiGluExpert) -> &LinearWithLora,
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

fn validate_config(cfg: MoeConfig) -> Result<()> {
    if cfg.hidden_size == 0 || cfg.intermediate_size == 0 || cfg.num_experts == 0 {
        return Err(AutogradError::TapeInvariant(
            "moe dimensions must all be non-zero",
        ));
    }
    if cfg.top_k == 0 || cfg.top_k > cfg.num_experts {
        return Err(AutogradError::InvalidIndicesLen {
            expected: cfg.num_experts,
            got: cfg.top_k,
        });
    }
    if cfg.lora.rank == 0 {
        return Err(AutogradError::TapeInvariant("moe lora rank must be > 0"));
    }
    Ok(())
}
