//! DeepSeek V4 MoE FFN scaffold.
//!
//! The local V4 1B checkpoint uses routed experts plus a shared expert on each
//! layer. Phase 0.5 records the correct tensor shape; Phase 1 supplies the
//! shared CUDA MoE primitive, and Phase 2A wires this block into forward.

#[cfg(feature = "cuda")]
use anyhow::{Result, ensure};
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
#[cfg(feature = "cuda")]
use deepseek_spec::{DeepSeekV4Config, DeepSeekV4MoeRoutingKind};

#[cfg(feature = "cuda")]
use crate::distributed::expert_state::{
    ExpertGroup, ExpertRoute, ExpertRoutingWeights, LocalExpertRouting,
};
#[cfg(feature = "cuda")]
use crate::ops;

/// One SwiGLU expert: `w2(silu(w1(x)) * w3(x))`.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // populated once the Phase 2A loader allocates tensors
pub(super) struct DeepseekV4Expert {
    pub(super) w1: DeviceMatrix,
    pub(super) w2: DeviceMatrix,
    pub(super) w3: DeviceMatrix,
}

#[cfg(feature = "cuda")]
impl DeepseekV4Expert {
    /// Run one DeepSeek V4 SwiGLU expert on a packed `[tokens, hidden]` row block.
    pub(super) fn forward(
        &self,
        ctx: &DeviceContext,
        hidden: &HiddenStates,
        swiglu_limit: f32,
    ) -> Result<HiddenStates> {
        ensure!(
            self.w1.cols == hidden.hidden_dim && self.w3.cols == hidden.hidden_dim,
            "DeepSeek V4 expert input width mismatch: hidden_dim={} w1.cols={} w3.cols={}",
            hidden.hidden_dim,
            self.w1.cols,
            self.w3.cols
        );
        ensure!(
            self.w1.rows == self.w3.rows && self.w2.cols == self.w1.rows,
            "DeepSeek V4 expert intermediate mismatch: w1.rows={} w3.rows={} w2.cols={}",
            self.w1.rows,
            self.w3.rows,
            self.w2.cols
        );

        let gate = ops::gemm(ctx, &self.w1, hidden)?;
        let up = ops::gemm(ctx, &self.w3, hidden)?;
        let mut act = HiddenStates::zeros(ctx, self.w1.rows, hidden.seq_len)?;
        ops::dsv4_swiglu_clamped_batch_into(ctx, &gate, &up, &mut act, swiglu_limit)?;
        ops::gemm(ctx, &self.w2, &act)
    }
}

/// V4 routed MoE block plus optional shared expert.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // populated once the Phase 2A loader allocates tensors
pub(super) struct DeepseekV4MoeBlock {
    pub(super) gate_weight: DeviceMatrix,
    pub(super) gate_bias: Option<DeviceVec>,
    /// Hash-router table for early layers. The exact integer storage type is
    /// finalized with the Phase 2A loader; Phase 0.5 only validates the tensor
    /// name and keeps the field explicit.
    pub(super) gate_tid2eid: Option<CudaSlice<i64>>,
    pub(super) experts: Vec<DeepseekV4Expert>,
    pub(super) shared_experts: Option<DeepseekV4Expert>,
}

#[cfg(feature = "cuda")]
#[allow(dead_code)] // method called from forward.rs once MoE kernels land
impl DeepseekV4MoeBlock {
    /// Run routed V4 MoE for a packed `[tokens, hidden]` row block.
    pub(super) fn forward(
        &self,
        ctx: &DeviceContext,
        hidden: &HiddenStates,
        swiglu_limit: f32,
    ) -> Result<HiddenStates> {
        ensure!(
            self.experts.is_empty(),
            "DeepSeek V4 routed MoE combine is not wired yet; local experts loaded={}",
            self.experts.len()
        );
        let shared = self
            .shared_experts
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 MoE block has no shared expert"))?;
        shared.forward(ctx, hidden, swiglu_limit)
    }

    /// Run the EP-local portion of routed V4 MoE and add the shared expert.
    ///
    /// `routing` must already be localized for this EP rank. The returned
    /// tensor is this rank's partial MoE output; callers that run multiple EP
    /// ranks still need the cross-rank reduction step.
    pub(super) fn forward_local_routes(
        &self,
        ctx: &DeviceContext,
        hidden: &HiddenStates,
        routing: &LocalExpertRouting,
        swiglu_limit: f32,
    ) -> Result<HiddenStates> {
        ensure!(
            routing.experts_per_rank == self.experts.len(),
            "DeepSeek V4 routing expects {} local experts but block loaded {}",
            routing.experts_per_rank,
            self.experts.len()
        );

        let mut out = if let Some(shared) = &self.shared_experts {
            shared.forward(ctx, hidden, swiglu_limit)?
        } else {
            HiddenStates::zeros(ctx, hidden.hidden_dim, hidden.seq_len)?
        };

        for route in &routing.routes {
            ensure!(
                route.token_idx < hidden.seq_len,
                "DeepSeek V4 route token {} out of range for seq_len {}",
                route.token_idx,
                hidden.seq_len
            );
            let expert = self.experts.get(route.local_expert_idx).ok_or_else(|| {
                anyhow::anyhow!(
                    "DeepSeek V4 local expert {} out of range for {} local experts",
                    route.local_expert_idx,
                    self.experts.len()
                )
            })?;
            let token_hidden = hidden_token(ctx, hidden, route.token_idx)?;
            let expert_out = expert.forward(ctx, &token_hidden, swiglu_limit)?;
            ops::add_scaled_row_into(ctx, &expert_out, &mut out, route.token_idx, route.weight)?;
        }

        Ok(out)
    }

    /// Route tokens with the loaded gate tensors, localize routes to this EP
    /// rank, and run the local MoE contribution.
    pub(super) fn forward_routed(
        &self,
        ctx: &DeviceContext,
        layer_idx: usize,
        config: &DeepSeekV4Config,
        ep: &ExpertGroup,
        hidden: &HiddenStates,
        token_ids: &[u32],
    ) -> Result<HiddenStates> {
        let routing = self.route_local(ctx, layer_idx, config, ep, hidden, token_ids)?;
        self.forward_local_routes(ctx, hidden, &routing, config.swiglu_limit)
    }

    fn route_local(
        &self,
        ctx: &DeviceContext,
        layer_idx: usize,
        config: &DeepSeekV4Config,
        ep: &ExpertGroup,
        hidden: &HiddenStates,
        token_ids: &[u32],
    ) -> Result<LocalExpertRouting> {
        ensure!(
            token_ids.len() == hidden.seq_len,
            "DeepSeek V4 route token count {} does not match hidden seq_len {}",
            token_ids.len(),
            hidden.seq_len
        );
        ensure!(
            self.gate_weight.rows == config.n_routed_experts
                && self.gate_weight.cols == hidden.hidden_dim,
            "DeepSeek V4 gate shape mismatch: gate={}x{} hidden_dim={} n_routed_experts={}",
            self.gate_weight.rows,
            self.gate_weight.cols,
            hidden.hidden_dim,
            config.n_routed_experts
        );
        if let Some(bias) = &self.gate_bias {
            ensure!(
                bias.len == config.n_routed_experts,
                "DeepSeek V4 gate bias len {} does not match n_routed_experts {}",
                bias.len,
                config.n_routed_experts
            );
        }

        let logits = ops::gemm(ctx, &self.gate_weight, hidden)?;
        let logits_host = ctx.stream.clone_dtoh(&logits.data)?;
        let bias_host = self
            .gate_bias
            .as_ref()
            .map(|bias| ctx.stream.clone_dtoh(&bias.data))
            .transpose()?
            .map(|bias| {
                bias.into_iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>()
            });
        let mut routes = Vec::with_capacity(hidden.seq_len * config.num_experts_per_tok);

        for token_idx in 0..hidden.seq_len {
            let start = token_idx * logits.hidden_dim;
            let token_logits = logits_host[start..start + logits.hidden_dim]
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>();
            let scores = config.router_scores_from_logits(&token_logits)?;
            let hash_experts = match config.moe_routing_kind(layer_idx) {
                DeepSeekV4MoeRoutingKind::Hash => {
                    Some(self.hash_experts_for_token(ctx, config, token_ids[token_idx])?)
                }
                DeepSeekV4MoeRoutingKind::LearnedBias => None,
            };
            let token_routes = config.moe_routes_from_scores(
                layer_idx,
                token_idx,
                &scores,
                bias_host.as_deref(),
                hash_experts.as_deref(),
            )?;
            routes.extend(token_routes.into_iter().map(|route| ExpertRoute {
                token_idx: route.token_idx,
                expert_idx: route.expert_idx,
                weight: route.weight,
            }));
        }

        ep.localize_routing(&ExpertRoutingWeights::new(config.n_routed_experts, routes))
    }

    fn hash_experts_for_token(
        &self,
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        token_id: u32,
    ) -> Result<Vec<usize>> {
        let table = self
            .gate_tid2eid
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("hash-routed DeepSeek V4 MoE layer missing tid2eid"))?;
        ensure!(
            (token_id as usize) < config.vocab_size,
            "DeepSeek V4 token id {token_id} exceeds vocab_size {}",
            config.vocab_size
        );
        let start = token_id as usize * config.num_experts_per_tok;
        let end = start + config.num_experts_per_tok;
        ensure!(
            end <= table.len(),
            "DeepSeek V4 tid2eid table too short: need {} entries for token {}, have {}",
            end,
            token_id,
            table.len()
        );
        let experts_i64 = ctx.stream.clone_dtoh(&table.slice(start..end))?;
        experts_i64
            .into_iter()
            .map(|expert_idx| {
                ensure!(
                    expert_idx >= 0,
                    "DeepSeek V4 tid2eid contains negative expert id"
                );
                usize::try_from(expert_idx)
                    .map_err(|_| anyhow::anyhow!("DeepSeek V4 tid2eid expert id overflow"))
            })
            .collect()
    }
}

#[cfg(feature = "cuda")]
fn hidden_token(
    ctx: &DeviceContext,
    hidden: &HiddenStates,
    token_idx: usize,
) -> Result<HiddenStates> {
    let token = ops::extract_vec(ctx, hidden, token_idx)?;
    Ok(HiddenStates {
        data: token.data,
        hidden_dim: hidden.hidden_dim,
        seq_len: 1,
    })
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::distributed::expert_state::{ExpertGroup, LocalExpertRoute};
    use half::bf16;

    fn bf16_vec(values: &[f32]) -> Vec<bf16> {
        values.iter().map(|&value| bf16::from_f32(value)).collect()
    }

    fn silu(value: f32) -> f32 {
        value / (1.0 + (-value).exp())
    }

    fn tiny_config() -> DeepSeekV4Config {
        DeepSeekV4Config::from_json_str(
            r#"{
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "torch_dtype": "bfloat16",
            "vocab_size": 16,
            "hidden_size": 2,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 1,
            "hidden_act": "silu",
            "swiglu_limit": 10.0,
            "q_lora_rank": 1,
            "o_lora_rank": 1,
            "o_groups": 1,
            "qk_rope_head_dim": 1,
            "n_routed_experts": 4,
            "n_shared_experts": 0,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 1,
            "routed_scaling_factor": 1.0,
            "norm_topk_prob": false,
            "scoring_func": "softmax",
            "topk_method": "noaux_tc",
            "index_n_heads": 1,
            "index_head_dim": 1,
            "index_topk": 1,
            "num_hash_layers": 0,
            "sliding_window": 4,
            "compress_ratios": [0],
            "compress_rope_theta": 160000.0,
            "hc_mult": 1,
            "hc_sinkhorn_iters": 1,
            "hc_eps": 1.0e-6,
            "num_nextn_predict_layers": 0,
            "max_position_embeddings": 16,
            "rope_theta": 10000.0,
            "rope_scaling": {
                "type": "yarn",
                "factor": 1.0,
                "original_max_position_embeddings": 16,
                "beta_fast": 32.0,
                "beta_slow": 1.0
            },
            "rms_norm_eps": 1.0e-6,
            "initializer_range": 0.02,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "eos_token_id": 1
        }"#,
        )
        .unwrap()
    }

    #[test]
    fn expert_forward_runs_clamped_swiglu_on_gpu() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let hidden = HiddenStates {
            data: ctx.stream.clone_htod(&bf16_vec(&[1.0, -2.0, 0.5, 3.0]))?,
            hidden_dim: 2,
            seq_len: 2,
        };
        let expert = DeepseekV4Expert {
            w1: DeviceMatrix::from_host(
                &ctx,
                &bf16_vec(&[
                    1.0, 0.0, //
                    0.0, 1.0, //
                    1.0, 1.0,
                ]),
                3,
                2,
            )?,
            w2: DeviceMatrix::from_host(
                &ctx,
                &bf16_vec(&[
                    1.0, 0.0, 0.5, //
                    0.0, 1.0, -1.0,
                ]),
                2,
                3,
            )?,
            w3: DeviceMatrix::from_host(
                &ctx,
                &bf16_vec(&[
                    0.5, 0.0, //
                    0.0, -1.0, //
                    1.0, -1.0,
                ]),
                3,
                2,
            )?,
        };

        let out = expert.forward(&ctx, &hidden, 2.0)?;
        let out_host = ctx.stream.clone_dtoh(&out.data)?;
        ctx.sync()?;

        let inputs = [[1.0_f32, -2.0_f32], [0.5_f32, 3.0_f32]];
        let mut expected = Vec::new();
        for x in inputs {
            let gate = [x[0], x[1], x[0] + x[1]];
            let up = [0.5 * x[0], -x[1], x[0] - x[1]];
            let act = [
                silu(gate[0].min(2.0)) * up[0].clamp(-2.0, 2.0),
                silu(gate[1].min(2.0)) * up[1].clamp(-2.0, 2.0),
                silu(gate[2].min(2.0)) * up[2].clamp(-2.0, 2.0),
            ];
            expected.push(act[0] + 0.5 * act[2]);
            expected.push(act[1] - act[2]);
        }

        for (idx, got) in out_host.iter().enumerate() {
            assert!(
                (got.to_f32() - expected[idx]).abs() < 0.05,
                "idx={idx} expected={} got={}",
                expected[idx],
                got.to_f32()
            );
        }
        Ok(())
    }

    #[test]
    fn moe_forward_local_routes_accumulates_ep_local_experts() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let hidden = HiddenStates {
            data: ctx.stream.clone_htod(&bf16_vec(&[1.0, 2.0, 3.0, -1.0]))?,
            hidden_dim: 2,
            seq_len: 2,
        };
        let expert0 = DeepseekV4Expert {
            w1: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 0.0]), 1, 2)?,
            w2: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 2.0]), 2, 1)?,
            w3: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 0.0]), 1, 2)?,
        };
        let expert1 = DeepseekV4Expert {
            w1: DeviceMatrix::from_host(&ctx, &bf16_vec(&[0.0, 1.0]), 1, 2)?,
            w2: DeviceMatrix::from_host(&ctx, &bf16_vec(&[-1.0, 0.5]), 2, 1)?,
            w3: DeviceMatrix::from_host(&ctx, &bf16_vec(&[0.0, 1.0]), 1, 2)?,
        };
        let block = DeepseekV4MoeBlock {
            gate_weight: DeviceMatrix::from_host(&ctx, &bf16_vec(&[0.0, 0.0]), 1, 2)?,
            gate_bias: None,
            gate_tid2eid: None,
            experts: vec![expert0, expert1],
            shared_experts: None,
        };
        let routing = LocalExpertRouting {
            num_global_experts: 4,
            experts_per_rank: 2,
            routes: vec![
                LocalExpertRoute {
                    token_idx: 0,
                    global_expert_idx: 0,
                    local_expert_idx: 0,
                    weight: 0.25,
                },
                LocalExpertRoute {
                    token_idx: 0,
                    global_expert_idx: 1,
                    local_expert_idx: 1,
                    weight: 0.5,
                },
                LocalExpertRoute {
                    token_idx: 1,
                    global_expert_idx: 1,
                    local_expert_idx: 1,
                    weight: 1.0,
                },
            ],
        };

        let out = block.forward_local_routes(&ctx, &hidden, &routing, 10.0)?;
        let out_host = ctx.stream.clone_dtoh(&out.data)?;
        ctx.sync()?;

        let e0_t0 = silu(1.0) * 1.0;
        let e1_t0 = silu(2.0) * 2.0;
        let e1_t1 = silu(-1.0) * -1.0;
        let expected = [
            0.25 * e0_t0 - 0.5 * e1_t0,
            0.25 * (2.0 * e0_t0) + 0.5 * (0.5 * e1_t0),
            -e1_t1,
            0.5 * e1_t1,
        ];

        for (idx, got) in out_host.iter().enumerate() {
            assert!(
                (got.to_f32() - expected[idx]).abs() < 0.05,
                "idx={idx} expected={} got={}",
                expected[idx],
                got.to_f32()
            );
        }
        Ok(())
    }

    #[test]
    fn moe_forward_routed_computes_gate_routes_and_localizes_ep() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let config = tiny_config();
        let ep = ExpertGroup::new(0, 2, config.n_routed_experts)?;
        let hidden = HiddenStates {
            data: ctx.stream.clone_htod(&bf16_vec(&[1.0, 0.0, 0.0, 2.0]))?,
            hidden_dim: 2,
            seq_len: 2,
        };
        let expert0 = DeepseekV4Expert {
            w1: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 0.0]), 1, 2)?,
            w2: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 2.0]), 2, 1)?,
            w3: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 0.0]), 1, 2)?,
        };
        let expert1 = DeepseekV4Expert {
            w1: DeviceMatrix::from_host(&ctx, &bf16_vec(&[0.0, 1.0]), 1, 2)?,
            w2: DeviceMatrix::from_host(&ctx, &bf16_vec(&[-1.0, 0.5]), 2, 1)?,
            w3: DeviceMatrix::from_host(&ctx, &bf16_vec(&[0.0, 1.0]), 1, 2)?,
        };
        let block = DeepseekV4MoeBlock {
            gate_weight: DeviceMatrix::from_host(
                &ctx,
                &bf16_vec(&[
                    1.0, 0.0, //
                    0.0, 1.0, //
                    -1.0, 0.0, //
                    0.0, -1.0,
                ]),
                4,
                2,
            )?,
            gate_bias: Some(DeviceVec::from_host(
                &ctx,
                &bf16_vec(&[0.0, 0.0, 0.0, 0.0]),
            )?),
            gate_tid2eid: None,
            experts: vec![expert0, expert1],
            shared_experts: None,
        };

        let out = block.forward_routed(&ctx, 0, &config, &ep, &hidden, &[3, 4])?;
        let out_host = ctx.stream.clone_dtoh(&out.data)?;
        ctx.sync()?;

        let token0_scores = config.router_scores_from_logits(&[1.0, 0.0, -1.0, 0.0])?;
        let token1_scores = config.router_scores_from_logits(&[0.0, 2.0, 0.0, -2.0])?;
        let e0_t0 = silu(1.0) * 1.0;
        let e1_t1 = silu(2.0) * 2.0;
        let expected = [
            token0_scores[0] * e0_t0,
            token0_scores[0] * (2.0 * e0_t0),
            -token1_scores[1] * e1_t1,
            token1_scores[1] * (0.5 * e1_t1),
        ];

        for (idx, got) in out_host.iter().enumerate() {
            assert!(
                (got.to_f32() - expected[idx]).abs() < 0.05,
                "idx={idx} expected={} got={}",
                expected[idx],
                got.to_f32()
            );
        }
        Ok(())
    }
}
