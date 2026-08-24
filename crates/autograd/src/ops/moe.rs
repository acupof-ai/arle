use std::{
    cmp::Ordering,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering as AtomicOrdering},
    },
    time::Instant,
};

use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    backend::Device,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Dirty, Tensor, TensorId, TensorStore},
};

type OptionalMatmulGrads = (Option<Vec<f32>>, Option<Vec<f32>>);

static MOE_GROUPED_PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);
static MOE_GROUPED_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();

#[derive(Clone, Copy)]
struct MoeGroupedLinearProfile {
    enabled: bool,
    call: u64,
}

impl MoeGroupedLinearProfile {
    fn new() -> Self {
        let enabled = *MOE_GROUPED_PROFILE_ENABLED.get_or_init(|| {
            std::env::var("ARLE_MOE_GROUPED_PROFILE")
                .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
                .unwrap_or(false)
        });
        let call = if enabled {
            MOE_GROUPED_PROFILE_CALLS.fetch_add(1, AtomicOrdering::Relaxed) + 1
        } else {
            0
        };
        Self { enabled, call }
    }

    fn emit_summary(
        &self,
        experts: usize,
        active: usize,
        max_rows: usize,
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        input_kind: MoeGroupedLinearInput,
        need_input_grad: bool,
        need_weight_grad: bool,
        need_lora_a_grad: bool,
        need_lora_b_grad: bool,
    ) {
        if !self.enabled {
            return;
        }
        eprintln!(
            "moe_grouped_linear_profile call={} stage=summary experts={} active={} \
             max_rows={} in_dim={} out_dim={} rank={} input_kind={:?} \
             need_input_grad={} need_weight_grad={} need_lora_a_grad={} \
             need_lora_b_grad={}",
            self.call,
            experts,
            active,
            max_rows,
            in_dim,
            out_dim,
            rank,
            input_kind,
            need_input_grad,
            need_weight_grad,
            need_lora_a_grad,
            need_lora_b_grad
        );
    }

    fn time_result<T>(
        &self,
        label: &'static str,
        stage: &'static str,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if !self.enabled {
            return f();
        }
        let started = Instant::now();
        let result = f();
        eprintln!(
            "moe_grouped_linear_profile call={} label={} stage={} seconds={:.6} ok={}",
            self.call,
            label,
            stage,
            started.elapsed().as_secs_f64(),
            result.is_ok()
        );
        result
    }

    fn time_value<T>(&self, label: &'static str, stage: &'static str, f: impl FnOnce() -> T) -> T {
        if !self.enabled {
            return f();
        }
        let started = Instant::now();
        let result = f();
        eprintln!(
            "moe_grouped_linear_profile call={} label={} stage={} seconds={:.6}",
            self.call,
            label,
            stage,
            started.elapsed().as_secs_f64()
        );
        result
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeRoute {
    pub token: usize,
    pub slot: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeGroupedRoute {
    pub expert: usize,
    pub row: usize,
    pub token: usize,
    pub slot: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeGroupedLinearInput {
    TokenRows,
    Packed,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeGroupedLinearExpert {
    pub weight: TensorId,
    pub lora_a: Option<TensorId>,
    pub lora_b: Option<TensorId>,
    pub lora_scale: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeTopK {
    pub weights: TensorId,
    pub indices: Vec<usize>,
    pub tokens: usize,
    pub experts: usize,
    pub top_k: usize,
}

pub fn moe_topk_softmax(
    logits: TensorId,
    top_k: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<MoeTopK> {
    let input = store.tensor_host(logits)?;
    if input.shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: input.shape.len(),
        });
    }
    let tokens = input.shape[0];
    let experts = input.shape[1];
    validate_top_k(top_k, experts)?;

    let indices: Vec<usize> = (0..tokens)
        .flat_map(|token| {
            let row = &input.data[token * experts..(token + 1) * experts];
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&lhs, &rhs| {
                row[rhs]
                    .partial_cmp(&row[lhs])
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| lhs.cmp(&rhs))
            });
            order.into_iter().take(top_k)
        })
        .collect();

    moe_topk_softmax_with_indices(logits, top_k, &indices, store, tape)
}

pub fn moe_topk_softmax_with_indices(
    logits: TensorId,
    top_k: usize,
    indices: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<MoeTopK> {
    let input = store.tensor_host(logits)?;
    if input.shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: input.shape.len(),
        });
    }
    let tokens = input.shape[0];
    let experts = input.shape[1];
    validate_top_k(top_k, experts)?;
    if indices.len() != tokens * top_k {
        return Err(AutogradError::InvalidIndicesLen {
            expected: tokens * top_k,
            got: indices.len(),
        });
    }

    let mut weights = vec![0.0_f32; tokens * top_k];
    for token in 0..tokens {
        let row = &input.data[token * experts..(token + 1) * experts];
        let selected = &indices[token * top_k..(token + 1) * top_k];
        for &expert in selected {
            if expert >= experts {
                return Err(AutogradError::IndexOutOfBounds {
                    index: expert,
                    upper: experts,
                });
            }
        }
        let max_logit = selected
            .iter()
            .map(|&expert| row[expert])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0_f32;
        for &expert in selected {
            denom += (row[expert] - max_logit).exp();
        }
        for (slot, &expert) in selected.iter().enumerate() {
            weights[token * top_k + slot] = (row[expert] - max_logit).exp() / denom;
        }
    }
    let indices = indices.to_vec();

    let weights_id = store.alloc(Tensor::new(weights, vec![tokens, top_k], false)?);
    TapeEntry {
        op: BackwardOp::MoeTopKSoftmax,
        output_id: weights_id,
        input_ids: smallvec![logits],
        saved: SavedContext::MoeTopKSoftmaxCtx {
            y: weights_id,
            indices: indices.clone(),
            logits_shape: input.shape,
            top_k,
        },
    }
    .record(store, tape)?;

    Ok(MoeTopK {
        weights: weights_id,
        indices,
        tokens,
        experts,
        top_k,
    })
}

pub(crate) fn moe_topk_softmax_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let logits = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("moe top-k missing logits"))?;
    if !store.tensor(logits)?.requires_grad {
        return Ok(GradPairs::new());
    }
    let SavedContext::MoeTopKSoftmaxCtx {
        y,
        indices,
        logits_shape,
        top_k,
    } = entry.saved.clone()
    else {
        return Err(AutogradError::TapeInvariant(
            "moe top-k backward missing saved context",
        ));
    };
    if logits_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: logits_shape.len(),
        });
    }
    let tokens = logits_shape[0];
    let experts = logits_shape[1];
    validate_top_k(top_k, experts)?;
    if indices.len() != tokens * top_k {
        return Err(AutogradError::InvalidIndicesLen {
            expected: tokens * top_k,
            got: indices.len(),
        });
    }

    let weights = store.tensor_host(y)?;
    let upstream = store.tensor_host(output_grad_id)?;
    let expected_shape = vec![tokens, top_k];
    if weights.shape != expected_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_shape,
            got: weights.shape,
        });
    }
    if upstream.shape != vec![tokens, top_k] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![tokens, top_k],
            got: upstream.shape,
        });
    }

    let mut grad = vec![0.0_f32; tokens * experts];
    for token in 0..tokens {
        let slot_base = token * top_k;
        let mut dot = 0.0_f32;
        for slot in 0..top_k {
            dot += upstream.data[slot_base + slot] * weights.data[slot_base + slot];
        }
        for slot in 0..top_k {
            let expert = indices[slot_base + slot];
            if expert >= experts {
                return Err(AutogradError::IndexOutOfBounds {
                    index: expert,
                    upper: experts,
                });
            }
            grad[token * experts + expert] +=
                weights.data[slot_base + slot] * (upstream.data[slot_base + slot] - dot);
        }
    }

    let grad_id = store.alloc(Tensor::new(grad, logits_shape, false)?);
    Ok(smallvec![(logits, grad_id)])
}

pub fn moe_grouped_linear(
    input: TensorId,
    experts: &[MoeGroupedLinearExpert],
    routes: &[MoeGroupedRoute],
    input_kind: MoeGroupedLinearInput,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let input_t = store.tensor_host(input)?;
    let (experts_len, max_rows, in_dim) =
        validate_grouped_linear_input(&input_t.shape, experts, routes, input_kind, store)?;
    let (out_dim, expert_shapes) = validate_grouped_linear_experts(experts, in_dim, store)?;

    let (active_expert_indices, active_expert_map) = active_expert_map(experts_len, routes)?;
    let active_routes = remap_routes_to_active(routes, &active_expert_map)?;
    let packed_input_shape = vec![active_expert_indices.len(), max_rows, in_dim];
    let resident_base_output = if store.backend().device() == Device::Cuda {
        let packed_input = pack_active_grouped_input(
            &input_t.data,
            &input_t.shape,
            input_kind,
            &active_routes,
            &active_expert_indices,
            max_rows,
            in_dim,
        )?;
        grouped_base_forward_resident(
            store,
            experts,
            &active_expert_indices,
            &packed_input,
            &packed_input_shape,
            max_rows,
            in_dim,
            out_dim,
        )?
    } else {
        None
    };
    let resident_base_forward = resident_base_output.is_some();
    let mut route_lookup = vec![None; experts_len * max_rows];
    for route in routes {
        route_lookup[route.expert * max_rows + route.row] = Some(*route);
    }

    let mut out =
        resident_base_output.unwrap_or_else(|| vec![0.0_f32; experts_len * max_rows * out_dim]);
    for &expert_index in &active_expert_indices {
        let expert = experts[expert_index];
        let weight = if resident_base_forward {
            None
        } else {
            Some(store.tensor_host(expert.weight)?)
        };
        let lora_a = match expert.lora_a {
            Some(id) => Some(store.tensor_host(id)?),
            None => None,
        };
        let lora_b = match expert.lora_b {
            Some(id) => Some(store.tensor_host(id)?),
            None => None,
        };
        for row in 0..max_rows {
            let Some(input_offset) = grouped_input_offset(
                &input_t.shape,
                input_kind,
                &route_lookup,
                expert_index,
                row,
                max_rows,
                in_dim,
            )?
            else {
                continue;
            };
            let x = &input_t.data[input_offset..input_offset + in_dim];
            let out_offset = (expert_index * max_rows + row) * out_dim;
            if let Some(weight) = weight.as_ref() {
                grouped_linear_row_forward(
                    x,
                    &weight.data,
                    out_dim,
                    in_dim,
                    lora_a.as_ref().map(|t| t.data.as_slice()),
                    lora_b.as_ref().map(|t| t.data.as_slice()),
                    expert_shapes[expert_index].rank,
                    expert.lora_scale,
                    &mut out[out_offset..out_offset + out_dim],
                );
            } else {
                grouped_lora_delta_row_forward(
                    x,
                    out_dim,
                    in_dim,
                    lora_a.as_ref().map(|t| t.data.as_slice()),
                    lora_b.as_ref().map(|t| t.data.as_slice()),
                    expert_shapes[expert_index].rank,
                    expert.lora_scale,
                    &mut out[out_offset..out_offset + out_dim],
                );
            }
        }
    }

    let requires_grad = store.tensor(input)?.requires_grad
        || experts.iter().any(|expert| {
            store
                .tensor(expert.weight)
                .map(|t| t.requires_grad)
                .unwrap_or(false)
                || expert
                    .lora_a
                    .and_then(|id| store.tensor(id).ok().map(|t| t.requires_grad))
                    .unwrap_or(false)
                || expert
                    .lora_b
                    .and_then(|id| store.tensor(id).ok().map(|t| t.requires_grad))
                    .unwrap_or(false)
        });
    let output_shape = vec![experts_len, max_rows, out_dim];
    let output_id = store.alloc(Tensor::new(out, output_shape.clone(), requires_grad)?);
    if requires_grad {
        let mut input_ids = smallvec![input];
        for expert in experts {
            input_ids.push(expert.weight);
            if let Some(id) = expert.lora_a {
                input_ids.push(id);
            }
            if let Some(id) = expert.lora_b {
                input_ids.push(id);
            }
        }
        TapeEntry {
            op: BackwardOp::MoeGroupedLinear,
            output_id,
            input_ids,
            saved: SavedContext::MoeGroupedLinearCtx {
                input,
                experts: experts.to_vec(),
                routes: routes.to_vec(),
                input_kind,
                input_shape: input_t.shape,
                output_shape,
            },
        }
        .record(store, tape)?;
    }
    Ok(output_id)
}

pub(crate) fn moe_grouped_linear_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let profile = MoeGroupedLinearProfile::new();
    let total_started = profile.enabled.then(Instant::now);
    let SavedContext::MoeGroupedLinearCtx {
        input,
        experts,
        routes,
        input_kind,
        input_shape,
        output_shape,
    } = entry.saved.clone()
    else {
        return Err(AutogradError::TapeInvariant(
            "moe grouped linear backward missing saved context",
        ));
    };
    let upstream = store.tensor_host(output_grad_id)?;
    if upstream.shape != output_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: output_shape.clone(),
            got: upstream.shape,
        });
    }
    let need_input_grad = {
        let input_t = store.tensor(input)?;
        if input_t.shape != input_shape {
            return Err(AutogradError::ShapeMismatch {
                expected: input_shape.clone(),
                got: input_t.shape.clone(),
            });
        }
        input_t.requires_grad
    };

    let (experts_len, max_rows, in_dim) =
        validate_grouped_linear_input(&input_shape, &experts, &routes, input_kind, store)?;
    let (out_dim, expert_shapes) = validate_grouped_linear_experts(&experts, in_dim, store)?;
    if output_shape != vec![experts_len, max_rows, out_dim] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![experts_len, max_rows, out_dim],
            got: output_shape,
        });
    }
    let (active_expert_indices, active_expert_map) = active_expert_map(experts_len, &routes)?;
    let active_experts = active_expert_indices
        .iter()
        .map(|&expert_index| experts[expert_index])
        .collect::<Vec<_>>();
    let active_routes = remap_routes_to_active(&routes, &active_expert_map)?;
    let active_len = active_experts.len();

    let need_weight_grad = experts
        .iter()
        .any(|expert| store.tensor(expert.weight).is_ok_and(|t| t.requires_grad));
    let need_lora_a_grad = experts.iter().any(|expert| {
        expert
            .lora_a
            .is_some_and(|id| store.tensor(id).is_ok_and(|t| t.requires_grad))
    });
    let need_lora_b_grad = experts.iter().any(|expert| {
        expert
            .lora_b
            .is_some_and(|id| store.tensor(id).is_ok_and(|t| t.requires_grad))
    });
    let common_rank = common_lora_rank(&expert_shapes)?;
    profile.emit_summary(
        experts_len,
        active_len,
        max_rows,
        in_dim,
        out_dim,
        common_rank,
        input_kind,
        need_input_grad,
        need_weight_grad,
        need_lora_a_grad,
        need_lora_b_grad,
    );

    let packed_input_shape = vec![active_len, max_rows, in_dim];
    let packed_grad_out = profile.time_result("pack", "grad_out", || {
        pack_active_expert_blocks(
            &upstream.data,
            experts_len,
            max_rows,
            out_dim,
            &active_expert_indices,
        )
    })?;
    let grad_out_shape = vec![active_len, max_rows, out_dim];

    let resident_base_input_grad =
        if need_input_grad && !need_weight_grad && store.backend().device() == Device::Cuda {
            profile.time_result("base_resident_input_grad", "total", || {
                grouped_base_input_grad_resident(
                    store,
                    &active_experts,
                    &packed_grad_out,
                    &grad_out_shape,
                    max_rows,
                    in_dim,
                    out_dim,
                )
            })?
        } else {
            None
        };

    let needs_packed_input = common_rank > 0
        || (resident_base_input_grad.is_none() && (need_input_grad || need_weight_grad));
    let packed_input = if needs_packed_input {
        let input_t = store.tensor_host(input)?;
        Some(profile.time_result("pack", "active_input", || {
            pack_active_grouped_input(
                &input_t.data,
                &input_shape,
                input_kind,
                &active_routes,
                &active_expert_indices,
                max_rows,
                in_dim,
            )
        })?)
    } else {
        None
    };

    let (grad_packed_input_base, grad_packed_weight_t) =
        if let Some(grad) = resident_base_input_grad {
            (Some(grad), None)
        } else if need_input_grad || need_weight_grad {
            let packed_input = packed_input.as_ref().ok_or(AutogradError::TapeInvariant(
                "moe grouped linear fallback missing packed input",
            ))?;
            let packed_weight_t = profile.time_result("pack", "base_weight_t", || {
                pack_grouped_weight_t(&active_experts, out_dim, in_dim, store)
            })?;
            let packed_weight_t_shape = vec![active_len, in_dim, out_dim];
            grouped_matmul_backward(
                store,
                &profile,
                "base_backward",
                packed_input,
                &packed_input_shape,
                &packed_weight_t,
                &packed_weight_t_shape,
                &packed_grad_out,
                &grad_out_shape,
                need_input_grad,
                need_weight_grad,
            )?
        } else {
            (None, None)
        };

    let grad_packed_input_len = active_len * max_rows * in_dim;
    let mut grad_packed_input = if need_input_grad && common_rank == 0 {
        grad_packed_input_base.or_else(|| Some(vec![0.0_f32; grad_packed_input_len]))
    } else if need_input_grad {
        let mut grad = vec![0.0_f32; grad_packed_input_len];
        if let Some(src) = grad_packed_input_base.as_ref() {
            profile.time_value("merge", "base_input_grad", || {
                add_slice_in_place(&mut grad, src)
            });
        }
        Some(grad)
    } else {
        None
    };

    let mut grad_weights: Vec<Option<Vec<f32>>> =
        profile.time_value("alloc", "grad_weights", || {
            experts
                .iter()
                .map(|expert| {
                    store
                        .tensor(expert.weight)
                        .ok()
                        .filter(|t| t.requires_grad)
                        .map(|t| vec![0.0_f32; t.size])
                })
                .collect()
        });
    if let Some(grad_weight_t) = grad_packed_weight_t {
        profile.time_value("unpack", "base_weight_t_grad", || {
            unpack_grouped_weight_t_grad_active(
                &grad_weight_t,
                &active_expert_indices,
                out_dim,
                in_dim,
                &mut grad_weights,
            );
        });
    }

    let mut grad_lora_a: Vec<Option<Vec<f32>>> = profile.time_value("alloc", "grad_lora_a", || {
        experts
            .iter()
            .map(|expert| {
                expert.lora_a.and_then(|id| {
                    store
                        .tensor(id)
                        .ok()
                        .filter(|t| t.requires_grad)
                        .map(|t| vec![0.0_f32; t.size])
                })
            })
            .collect()
    });
    let mut grad_lora_b: Vec<Option<Vec<f32>>> = profile.time_value("alloc", "grad_lora_b", || {
        experts
            .iter()
            .map(|expert| {
                expert.lora_b.and_then(|id| {
                    store
                        .tensor(id)
                        .ok()
                        .filter(|t| t.requires_grad)
                        .map(|t| vec![0.0_f32; t.size])
                })
            })
            .collect()
    });

    if common_rank > 0 {
        let packed_input = packed_input.as_ref().ok_or(AutogradError::TapeInvariant(
            "moe grouped linear LoRA path missing packed input",
        ))?;
        let expert_tile = crate::runtime_flags::moe_lora_bwd_expert_tile();
        if active_len > expert_tile {
            grouped_lora_backward_tiled(
                store,
                &profile,
                &active_experts,
                &active_expert_indices,
                packed_input,
                &packed_input_shape,
                &packed_grad_out,
                max_rows,
                in_dim,
                out_dim,
                common_rank,
                need_input_grad,
                need_lora_a_grad,
                need_lora_b_grad,
                &mut grad_packed_input,
                &mut grad_lora_a,
                &mut grad_lora_b,
                expert_tile,
            )?;
        } else {
            let packed_a_t = profile.time_result("pack", "lora_a_t", || {
                pack_grouped_lora_a_t(&active_experts, common_rank, in_dim, store)
            })?;
            let packed_a_t_shape = vec![active_len, in_dim, common_rank];
            let packed_b_t = profile.time_result("pack", "lora_b_t", || {
                pack_grouped_lora_b_t(&active_experts, out_dim, common_rank, store)
            })?;
            let packed_b_t_shape = vec![active_len, common_rank, out_dim];
            let low = grouped_matmul_forward(
                store,
                &profile,
                "lora_low_forward",
                packed_input,
                &packed_input_shape,
                &packed_a_t,
                &packed_a_t_shape,
            )?;
            let low_shape = vec![active_len, max_rows, common_rank];
            let mut scaled_grad_out =
                profile.time_value("clone", "scaled_grad_out", || packed_grad_out.clone());
            profile.time_value("scale", "grad_out", || {
                for (expert_index, expert) in active_experts.iter().enumerate().take(active_len) {
                    let scale = expert.lora_scale;
                    let expert_base = expert_index * max_rows * out_dim;
                    for value in &mut scaled_grad_out[expert_base..expert_base + max_rows * out_dim]
                    {
                        *value *= scale;
                    }
                }
            });
            let (grad_low, grad_b_t) = grouped_matmul_backward(
                store,
                &profile,
                "lora_b_backward",
                &low,
                &low_shape,
                &packed_b_t,
                &packed_b_t_shape,
                &scaled_grad_out,
                &grad_out_shape,
                need_input_grad || need_lora_a_grad,
                need_lora_b_grad,
            )?;
            if let Some(grad_b_t) = grad_b_t {
                profile.time_value("unpack", "lora_b_t_grad", || {
                    unpack_grouped_lora_b_t_grad_active(
                        &grad_b_t,
                        &active_expert_indices,
                        out_dim,
                        common_rank,
                        &mut grad_lora_b,
                    );
                });
            }
            if let Some(grad_low) = grad_low {
                let (grad_x_lora, grad_a_t) = grouped_matmul_backward(
                    store,
                    &profile,
                    "lora_a_backward",
                    packed_input,
                    &packed_input_shape,
                    &packed_a_t,
                    &packed_a_t_shape,
                    &grad_low,
                    &low_shape,
                    need_input_grad,
                    need_lora_a_grad,
                )?;
                if let (Some(dst), Some(src)) = (grad_packed_input.as_mut(), grad_x_lora.as_ref()) {
                    profile.time_value("merge", "lora_input_grad", || add_slice_in_place(dst, src));
                }
                if let Some(grad_a_t) = grad_a_t {
                    profile.time_value("unpack", "lora_a_t_grad", || {
                        unpack_grouped_lora_a_t_grad_active(
                            &grad_a_t,
                            &active_expert_indices,
                            common_rank,
                            in_dim,
                            &mut grad_lora_a,
                        );
                    });
                }
            }
        }
    }

    let grad_input = match grad_packed_input {
        Some(grad) => Some(profile.time_result("unpack", "input_grad", || {
            unpack_active_grouped_input_grad(
                &grad,
                &input_shape,
                input_kind,
                &active_routes,
                &active_expert_indices,
                experts_len,
                active_len,
                max_rows,
                in_dim,
            )
        })?),
        None => None,
    };

    let mut grads = GradPairs::new();
    if let Some(grad) = grad_input {
        let grad_id = store.alloc(Tensor::new(grad, input_shape, false)?);
        grads.push((input, grad_id));
    }
    for (expert_index, expert) in experts.iter().enumerate() {
        if let Some(grad) = grad_weights[expert_index].take() {
            let shape = store.tensor(expert.weight)?.shape.clone();
            let grad_id = store.alloc(Tensor::new(grad, shape, false)?);
            grads.push((expert.weight, grad_id));
        }
        if let (Some(id), Some(grad)) = (expert.lora_a, grad_lora_a[expert_index].take()) {
            let shape = store.tensor(id)?.shape.clone();
            let grad_id = store.alloc(Tensor::new(grad, shape, false)?);
            grads.push((id, grad_id));
        }
        if let (Some(id), Some(grad)) = (expert.lora_b, grad_lora_b[expert_index].take()) {
            let shape = store.tensor(id)?.shape.clone();
            let grad_id = store.alloc(Tensor::new(grad, shape, false)?);
            grads.push((id, grad_id));
        }
    }

    if let Some(started) = total_started {
        eprintln!(
            "moe_grouped_linear_profile call={} stage=total seconds={:.6}",
            profile.call,
            started.elapsed().as_secs_f64()
        );
    }
    Ok(grads)
}

pub fn moe_grouped_weighted_scatter(
    values: TensorId,
    weights: TensorId,
    routes: &[MoeGroupedRoute],
    out_rows: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let values_t = store.tensor_host(values)?;
    let weights_t = store.tensor_host(weights)?;
    validate_grouped_scatter_shapes(&values_t.shape, &weights_t.shape, routes, out_rows)?;
    let experts = values_t.shape[0];
    let max_rows = values_t.shape[1];
    let hidden = values_t.shape[2];
    let top_k = weights_t.shape[1];

    let mut out = vec![0.0_f32; out_rows * hidden];
    for route in routes {
        let value_row = (route.expert * max_rows + route.row) * hidden;
        let weight = weights_t.data[route.token * top_k + route.slot];
        for col in 0..hidden {
            out[route.token * hidden + col] += values_t.data[value_row + col] * weight;
        }
    }

    let requires_grad = values_t.requires_grad || weights_t.requires_grad;
    let output_id = store.alloc(Tensor::new(out, vec![out_rows, hidden], requires_grad)?);
    TapeEntry {
        op: BackwardOp::MoeGroupedWeightedScatter,
        output_id,
        input_ids: smallvec![values, weights],
        saved: SavedContext::MoeGroupedWeightedScatterCtx {
            routes: routes.to_vec(),
            values_shape: vec![experts, max_rows, hidden],
            weights_shape: weights_t.shape,
            out_rows,
        },
    }
    .record(store, tape)?;
    Ok(output_id)
}

pub(crate) fn moe_grouped_weighted_scatter_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let values = *entry.input_ids.first().ok_or(AutogradError::TapeInvariant(
        "moe grouped weighted scatter missing values",
    ))?;
    let weights = *entry.input_ids.get(1).ok_or(AutogradError::TapeInvariant(
        "moe grouped weighted scatter missing weights",
    ))?;
    let SavedContext::MoeGroupedWeightedScatterCtx {
        routes,
        values_shape,
        weights_shape,
        out_rows,
    } = entry.saved.clone()
    else {
        return Err(AutogradError::TapeInvariant(
            "moe grouped weighted scatter backward missing saved context",
        ));
    };
    validate_grouped_scatter_shapes(&values_shape, &weights_shape, &routes, out_rows)?;
    let experts = values_shape[0];
    let max_rows = values_shape[1];
    let hidden = values_shape[2];
    let weight_rows = weights_shape[0];
    let top_k = weights_shape[1];

    let upstream = store.tensor_host(output_grad_id)?;
    if upstream.shape != vec![out_rows, hidden] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![out_rows, hidden],
            got: upstream.shape,
        });
    }
    let values_t = store.tensor_host(values)?;
    let weights_t = store.tensor_host(weights)?;

    let mut grads = GradPairs::new();
    if store.tensor(values)?.requires_grad {
        let mut grad_values = vec![0.0_f32; experts * max_rows * hidden];
        for route in &routes {
            let value_row = (route.expert * max_rows + route.row) * hidden;
            let weight = weights_t.data[route.token * top_k + route.slot];
            for col in 0..hidden {
                grad_values[value_row + col] += upstream.data[route.token * hidden + col] * weight;
            }
        }
        let grad_id = store.alloc(Tensor::new(grad_values, values_shape.clone(), false)?);
        grads.push((values, grad_id));
    }
    if store.tensor(weights)?.requires_grad {
        let mut grad_weights = vec![0.0_f32; weight_rows * top_k];
        for route in &routes {
            let value_row = (route.expert * max_rows + route.row) * hidden;
            let mut dot_product = 0.0_f32;
            for col in 0..hidden {
                dot_product +=
                    upstream.data[route.token * hidden + col] * values_t.data[value_row + col];
            }
            grad_weights[route.token * top_k + route.slot] += dot_product;
        }
        let grad_id = store.alloc(Tensor::new(grad_weights, weights_shape, false)?);
        grads.push((weights, grad_id));
    }
    Ok(grads)
}

fn validate_top_k(top_k: usize, experts: usize) -> Result<()> {
    if top_k == 0 {
        return Err(AutogradError::TapeInvariant("moe top_k must be > 0"));
    }
    if top_k > experts {
        return Err(AutogradError::InvalidIndicesLen {
            expected: experts,
            got: top_k,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct GroupedExpertShape {
    rank: usize,
}

fn validate_grouped_linear_input(
    input_shape: &[usize],
    experts: &[MoeGroupedLinearExpert],
    routes: &[MoeGroupedRoute],
    input_kind: MoeGroupedLinearInput,
    store: &mut TensorStore,
) -> Result<(usize, usize, usize)> {
    if experts.is_empty() {
        return Err(AutogradError::TapeInvariant(
            "moe grouped linear requires at least one expert",
        ));
    }
    let experts_len = experts.len();
    let (max_rows, in_dim) = match input_kind {
        MoeGroupedLinearInput::TokenRows => {
            if input_shape.len() != 2 {
                return Err(AutogradError::InvalidRank {
                    expected: "2",
                    got: input_shape.len(),
                });
            }
            let token_rows = input_shape[0];
            let max_rows = routes.iter().map(|route| route.row + 1).max().unwrap_or(0);
            if max_rows == 0 {
                return Err(AutogradError::TapeInvariant(
                    "moe grouped linear requires at least one route",
                ));
            }
            for route in routes {
                if route.expert >= experts_len {
                    return Err(AutogradError::IndexOutOfBounds {
                        index: route.expert,
                        upper: experts_len,
                    });
                }
                if route.token >= token_rows {
                    return Err(AutogradError::IndexOutOfBounds {
                        index: route.token,
                        upper: token_rows,
                    });
                }
            }
            (max_rows, input_shape[1])
        }
        MoeGroupedLinearInput::Packed => {
            if input_shape.len() != 3 {
                return Err(AutogradError::InvalidRank {
                    expected: "3",
                    got: input_shape.len(),
                });
            }
            if input_shape[0] != experts_len {
                return Err(AutogradError::ShapeMismatch {
                    expected: vec![experts_len],
                    got: vec![input_shape[0]],
                });
            }
            (input_shape[1], input_shape[2])
        }
    };

    for expert in experts {
        store.tensor(expert.weight)?;
        if let Some(id) = expert.lora_a {
            store.tensor(id)?;
        }
        if let Some(id) = expert.lora_b {
            store.tensor(id)?;
        }
    }
    Ok((experts_len, max_rows, in_dim))
}

fn validate_grouped_linear_experts(
    experts: &[MoeGroupedLinearExpert],
    in_dim: usize,
    store: &mut TensorStore,
) -> Result<(usize, Vec<GroupedExpertShape>)> {
    let mut out_dim = None;
    let mut shapes = Vec::with_capacity(experts.len());
    for expert in experts {
        let weight_shape = store.tensor(expert.weight)?.shape.clone();
        if weight_shape.len() != 2 {
            return Err(AutogradError::InvalidRank {
                expected: "2",
                got: weight_shape.len(),
            });
        }
        if weight_shape[1] != in_dim {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![in_dim],
                got: vec![weight_shape[1]],
            });
        }
        match out_dim {
            Some(expected) if expected != weight_shape[0] => {
                return Err(AutogradError::ShapeMismatch {
                    expected: vec![expected],
                    got: vec![weight_shape[0]],
                });
            }
            None => out_dim = Some(weight_shape[0]),
            _ => {}
        }

        let rank = match (expert.lora_a, expert.lora_b) {
            (Some(a_id), Some(b_id)) => {
                let a_shape = store.tensor(a_id)?.shape.clone();
                let b_shape = store.tensor(b_id)?.shape.clone();
                if a_shape.len() != 2 {
                    return Err(AutogradError::InvalidRank {
                        expected: "2",
                        got: a_shape.len(),
                    });
                }
                if b_shape.len() != 2 {
                    return Err(AutogradError::InvalidRank {
                        expected: "2",
                        got: b_shape.len(),
                    });
                }
                if a_shape[1] != in_dim {
                    return Err(AutogradError::ShapeMismatch {
                        expected: vec![in_dim],
                        got: vec![a_shape[1]],
                    });
                }
                if b_shape[0] != weight_shape[0] || b_shape[1] != a_shape[0] {
                    return Err(AutogradError::ShapeMismatch {
                        expected: vec![weight_shape[0], a_shape[0]],
                        got: b_shape,
                    });
                }
                a_shape[0]
            }
            (None, None) => 0,
            _ => {
                return Err(AutogradError::TapeInvariant(
                    "moe grouped linear requires both LoRA A and B or neither",
                ));
            }
        };
        shapes.push(GroupedExpertShape { rank });
    }
    Ok((out_dim.unwrap_or(0), shapes))
}

#[allow(clippy::too_many_arguments)]
fn grouped_input_offset(
    input_shape: &[usize],
    input_kind: MoeGroupedLinearInput,
    route_lookup: &[Option<MoeGroupedRoute>],
    expert: usize,
    row: usize,
    max_rows: usize,
    in_dim: usize,
) -> Result<Option<usize>> {
    match input_kind {
        MoeGroupedLinearInput::TokenRows => {
            let Some(route) = route_lookup[expert * max_rows + row] else {
                return Ok(None);
            };
            Ok(Some(route.token * in_dim))
        }
        MoeGroupedLinearInput::Packed => {
            let expected = input_shape[0] * input_shape[1] * input_shape[2];
            let offset = (expert * max_rows + row) * in_dim;
            if offset + in_dim > expected {
                return Err(AutogradError::DataLengthMismatch {
                    len: offset + in_dim,
                    shape: input_shape.to_vec(),
                    size: expected,
                });
            }
            Ok(Some(offset))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn grouped_linear_row_forward(
    x: &[f32],
    weight: &[f32],
    out_dim: usize,
    in_dim: usize,
    lora_a: Option<&[f32]>,
    lora_b: Option<&[f32]>,
    rank: usize,
    lora_scale: f32,
    out: &mut [f32],
) {
    for n in 0..out_dim {
        out[n] = dot(x, &weight[n * in_dim..(n + 1) * in_dim]);
    }
    if let (Some(a), Some(b)) = (lora_a, lora_b) {
        let mut low = vec![0.0_f32; rank];
        for r in 0..rank {
            low[r] = dot(x, &a[r * in_dim..(r + 1) * in_dim]);
        }
        for n in 0..out_dim {
            let mut delta = 0.0_f64;
            for r in 0..rank {
                delta += f64::from(low[r]) * f64::from(b[n * rank + r]);
            }
            out[n] += (delta * f64::from(lora_scale)) as f32;
        }
    }
}

fn grouped_lora_delta_row_forward(
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    lora_a: Option<&[f32]>,
    lora_b: Option<&[f32]>,
    rank: usize,
    lora_scale: f32,
    out: &mut [f32],
) {
    let (Some(a), Some(b)) = (lora_a, lora_b) else {
        return;
    };
    if rank == 0 {
        return;
    }
    let mut low = vec![0.0_f32; rank];
    for r in 0..rank {
        low[r] = dot(x, &a[r * in_dim..(r + 1) * in_dim]);
    }
    for n in 0..out_dim {
        let mut delta = 0.0_f64;
        for r in 0..rank {
            delta += f64::from(low[r]) * f64::from(b[n * rank + r]);
        }
        out[n] += (delta * f64::from(lora_scale)) as f32;
    }
}

fn active_expert_map(
    experts: usize,
    routes: &[MoeGroupedRoute],
) -> Result<(Vec<usize>, Vec<Option<usize>>)> {
    let mut seen = vec![false; experts];
    for route in routes {
        if route.expert >= experts {
            return Err(AutogradError::IndexOutOfBounds {
                index: route.expert,
                upper: experts,
            });
        }
        seen[route.expert] = true;
    }
    let active = seen
        .iter()
        .enumerate()
        .filter_map(|(expert, is_active)| is_active.then_some(expert))
        .collect::<Vec<_>>();
    if active.is_empty() {
        return Err(AutogradError::TapeInvariant(
            "moe grouped linear backward requires at least one active expert",
        ));
    }
    let mut map = vec![None; experts];
    for (compact, &expert) in active.iter().enumerate() {
        map[expert] = Some(compact);
    }
    Ok((active, map))
}

fn remap_routes_to_active(
    routes: &[MoeGroupedRoute],
    active_map: &[Option<usize>],
) -> Result<Vec<MoeGroupedRoute>> {
    routes
        .iter()
        .map(|route| {
            let expert = active_map
                .get(route.expert)
                .and_then(|entry| *entry)
                .ok_or(AutogradError::IndexOutOfBounds {
                    index: route.expert,
                    upper: active_map.len(),
                })?;
            Ok(MoeGroupedRoute { expert, ..*route })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn pack_grouped_input(
    input: &[f32],
    input_shape: &[usize],
    input_kind: MoeGroupedLinearInput,
    routes: &[MoeGroupedRoute],
    experts: usize,
    max_rows: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    match input_kind {
        MoeGroupedLinearInput::Packed => Ok(input.to_vec()),
        MoeGroupedLinearInput::TokenRows => {
            let mut packed = vec![0.0_f32; experts * max_rows * in_dim];
            for route in routes {
                let src = route.token * in_dim;
                let dst = (route.expert * max_rows + route.row) * in_dim;
                if src + in_dim > input.len() {
                    return Err(AutogradError::DataLengthMismatch {
                        len: src + in_dim,
                        shape: input_shape.to_vec(),
                        size: input.len(),
                    });
                }
                packed[dst..dst + in_dim].copy_from_slice(&input[src..src + in_dim]);
            }
            Ok(packed)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pack_active_grouped_input(
    input: &[f32],
    input_shape: &[usize],
    input_kind: MoeGroupedLinearInput,
    active_routes: &[MoeGroupedRoute],
    active_expert_indices: &[usize],
    max_rows: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    match input_kind {
        MoeGroupedLinearInput::TokenRows => pack_grouped_input(
            input,
            input_shape,
            input_kind,
            active_routes,
            active_expert_indices.len(),
            max_rows,
            in_dim,
        ),
        MoeGroupedLinearInput::Packed => {
            let mut packed = vec![0.0_f32; active_expert_indices.len() * max_rows * in_dim];
            for (compact, &expert) in active_expert_indices.iter().enumerate() {
                let src = (expert * max_rows) * in_dim;
                let dst = (compact * max_rows) * in_dim;
                let len = max_rows * in_dim;
                if src + len > input.len() {
                    return Err(AutogradError::DataLengthMismatch {
                        len: src + len,
                        shape: input_shape.to_vec(),
                        size: input.len(),
                    });
                }
                packed[dst..dst + len].copy_from_slice(&input[src..src + len]);
            }
            Ok(packed)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn unpack_grouped_input_grad(
    grad: &[f32],
    input_shape: &[usize],
    input_kind: MoeGroupedLinearInput,
    routes: &[MoeGroupedRoute],
    experts: usize,
    max_rows: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    match input_kind {
        MoeGroupedLinearInput::Packed => Ok(grad.to_vec()),
        MoeGroupedLinearInput::TokenRows => {
            let mut out = vec![0.0_f32; input_shape.iter().product()];
            for route in routes {
                let src = (route.expert * max_rows + route.row) * in_dim;
                let dst = route.token * in_dim;
                if route.expert >= experts || src + in_dim > grad.len() || dst + in_dim > out.len()
                {
                    return Err(AutogradError::DataLengthMismatch {
                        len: src + in_dim,
                        shape: vec![experts, max_rows, in_dim],
                        size: grad.len(),
                    });
                }
                for k in 0..in_dim {
                    out[dst + k] += grad[src + k];
                }
            }
            Ok(out)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn unpack_active_grouped_input_grad(
    grad: &[f32],
    input_shape: &[usize],
    input_kind: MoeGroupedLinearInput,
    active_routes: &[MoeGroupedRoute],
    active_expert_indices: &[usize],
    experts: usize,
    active_experts: usize,
    max_rows: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    match input_kind {
        MoeGroupedLinearInput::TokenRows => unpack_grouped_input_grad(
            grad,
            input_shape,
            input_kind,
            active_routes,
            active_experts,
            max_rows,
            in_dim,
        ),
        MoeGroupedLinearInput::Packed => {
            let mut out = vec![0.0_f32; input_shape.iter().product()];
            for (compact, &expert) in active_expert_indices.iter().enumerate() {
                if expert >= experts {
                    return Err(AutogradError::IndexOutOfBounds {
                        index: expert,
                        upper: experts,
                    });
                }
                let src = (compact * max_rows) * in_dim;
                let dst = (expert * max_rows) * in_dim;
                let len = max_rows * in_dim;
                if src + len > grad.len() || dst + len > out.len() {
                    return Err(AutogradError::DataLengthMismatch {
                        len: src + len,
                        shape: vec![active_experts, max_rows, in_dim],
                        size: grad.len(),
                    });
                }
                out[dst..dst + len].copy_from_slice(&grad[src..src + len]);
            }
            Ok(out)
        }
    }
}

fn pack_active_expert_blocks(
    input: &[f32],
    experts: usize,
    max_rows: usize,
    dim: usize,
    active_expert_indices: &[usize],
) -> Result<Vec<f32>> {
    let mut packed = vec![0.0_f32; active_expert_indices.len() * max_rows * dim];
    for (compact, &expert) in active_expert_indices.iter().enumerate() {
        if expert >= experts {
            return Err(AutogradError::IndexOutOfBounds {
                index: expert,
                upper: experts,
            });
        }
        let src = (expert * max_rows) * dim;
        let dst = (compact * max_rows) * dim;
        let len = max_rows * dim;
        if src + len > input.len() {
            return Err(AutogradError::DataLengthMismatch {
                len: src + len,
                shape: vec![experts, max_rows, dim],
                size: input.len(),
            });
        }
        packed[dst..dst + len].copy_from_slice(&input[src..src + len]);
    }
    Ok(packed)
}

fn pack_grouped_weight_t(
    experts: &[MoeGroupedLinearExpert],
    out_dim: usize,
    in_dim: usize,
    store: &mut TensorStore,
) -> Result<Vec<f32>> {
    let mut packed = vec![0.0_f32; experts.len() * in_dim * out_dim];
    for (expert_index, expert) in experts.iter().enumerate() {
        let weight = store.tensor_host(expert.weight)?;
        for n in 0..out_dim {
            for k in 0..in_dim {
                packed[(expert_index * in_dim + k) * out_dim + n] = weight.data[n * in_dim + k];
            }
        }
    }
    Ok(packed)
}

#[allow(clippy::too_many_arguments)]
fn grouped_base_forward_resident(
    store: &TensorStore,
    experts: &[MoeGroupedLinearExpert],
    active_expert_indices: &[usize],
    packed_input: &[f32],
    packed_input_shape: &[usize],
    max_rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Result<Option<Vec<f32>>> {
    if store.backend().device() != Device::Cuda {
        return Ok(None);
    }
    if packed_input_shape != [active_expert_indices.len(), max_rows, in_dim] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![active_expert_indices.len(), max_rows, in_dim],
            got: packed_input_shape.to_vec(),
        });
    }

    let mut weight_handles = Vec::with_capacity(active_expert_indices.len());
    for &expert_index in active_expert_indices {
        let expert = experts[expert_index];
        let tensor = store.tensor(expert.weight)?;
        if tensor.shape != [out_dim, in_dim] {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![out_dim, in_dim],
                got: tensor.shape.clone(),
            });
        }
        if tensor.requires_grad || tensor.dirty == Dirty::Host {
            return Ok(None);
        }
        let Some(handle) = tensor.device_handle.as_ref().cloned() else {
            return Ok(None);
        };
        weight_handles.push(handle);
    }

    let mut out = vec![0.0_f32; experts.len() * max_rows * out_dim];
    let a_shape = [max_rows, in_dim];
    let b_shape = [out_dim, in_dim];
    let input_stride = max_rows * in_dim;
    let output_stride = max_rows * out_dim;
    for (compact, (&expert_index, weight_handle)) in active_expert_indices
        .iter()
        .zip(weight_handles.iter())
        .enumerate()
    {
        let input_start = compact * input_stride;
        let a_handle = store.backend().upload(
            &packed_input[input_start..input_start + input_stride],
            &a_shape,
        )?;
        let (out_handle, out_shape) =
            store
                .backend()
                .matmul_bt(&a_handle, &a_shape, weight_handle, &b_shape)?;
        if out_shape != [max_rows, out_dim] {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![max_rows, out_dim],
                got: out_shape,
            });
        }
        store.backend().eval(&[&out_handle])?;
        let host = store.backend().readback(&out_handle)?;
        if host.len() != output_stride {
            return Err(AutogradError::DataLengthMismatch {
                len: host.len(),
                shape: vec![max_rows, out_dim],
                size: output_stride,
            });
        }
        let out_start = (expert_index * max_rows) * out_dim;
        out[out_start..out_start + output_stride].copy_from_slice(&host);
    }
    Ok(Some(out))
}

#[allow(clippy::too_many_arguments)]
fn grouped_base_input_grad_resident(
    store: &TensorStore,
    experts: &[MoeGroupedLinearExpert],
    packed_grad_out: &[f32],
    grad_out_shape: &[usize],
    max_rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Result<Option<Vec<f32>>> {
    if store.backend().device() != Device::Cuda {
        return Ok(None);
    }
    if grad_out_shape != [experts.len(), max_rows, out_dim] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![experts.len(), max_rows, out_dim],
            got: grad_out_shape.to_vec(),
        });
    }

    let mut weight_handles = Vec::with_capacity(experts.len());
    for expert in experts {
        let tensor = store.tensor(expert.weight)?;
        if tensor.shape != [out_dim, in_dim] {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![out_dim, in_dim],
                got: tensor.shape.clone(),
            });
        }
        if tensor.dirty == Dirty::Host {
            return Ok(None);
        }
        let Some(handle) = tensor.device_handle.as_ref().cloned() else {
            return Ok(None);
        };
        weight_handles.push(handle);
    }

    let mut out = vec![0.0_f32; experts.len() * max_rows * in_dim];
    let a_shape = [max_rows, in_dim];
    let b_shape = [out_dim, in_dim];
    let g_shape = [max_rows, out_dim];
    let input_stride = max_rows * in_dim;
    let grad_stride = max_rows * out_dim;
    for (compact, weight_handle) in weight_handles.iter().enumerate() {
        let input_start = compact * input_stride;
        let grad_start = compact * grad_stride;
        let g_handle = store.backend().upload(
            &packed_grad_out[grad_start..grad_start + grad_stride],
            &g_shape,
        )?;
        let grad_a = store.backend().matmul_bt_input_grad_device(
            weight_handle,
            &b_shape,
            &g_handle,
            &g_shape,
            &a_shape,
        )?;
        store.backend().eval(&[&grad_a])?;
        let host = store.backend().readback(&grad_a)?;
        if host.len() != input_stride {
            return Err(AutogradError::DataLengthMismatch {
                len: host.len(),
                shape: a_shape.to_vec(),
                size: input_stride,
            });
        }
        out[input_start..input_start + input_stride].copy_from_slice(&host);
    }
    Ok(Some(out))
}

fn unpack_grouped_weight_t_grad_active(
    grad_t: &[f32],
    active_expert_indices: &[usize],
    out_dim: usize,
    in_dim: usize,
    grad_weights: &mut [Option<Vec<f32>>],
) {
    for (compact, &expert_index) in active_expert_indices.iter().enumerate() {
        let Some(grad_w) = grad_weights[expert_index].as_mut() else {
            continue;
        };
        for n in 0..out_dim {
            for k in 0..in_dim {
                grad_w[n * in_dim + k] += grad_t[(compact * in_dim + k) * out_dim + n];
            }
        }
    }
}

fn common_lora_rank(shapes: &[GroupedExpertShape]) -> Result<usize> {
    let rank = shapes.first().map(|shape| shape.rank).unwrap_or(0);
    for shape in shapes {
        if shape.rank != rank {
            return Err(AutogradError::TapeInvariant(
                "moe grouped linear requires equal LoRA rank across experts",
            ));
        }
    }
    Ok(rank)
}

fn pack_grouped_lora_a_t(
    experts: &[MoeGroupedLinearExpert],
    rank: usize,
    in_dim: usize,
    store: &mut TensorStore,
) -> Result<Vec<f32>> {
    let mut packed = vec![0.0_f32; experts.len() * in_dim * rank];
    for (expert_index, expert) in experts.iter().enumerate() {
        let Some(a_id) = expert.lora_a else {
            continue;
        };
        let a = store.tensor_host(a_id)?;
        for r in 0..rank {
            for k in 0..in_dim {
                packed[(expert_index * in_dim + k) * rank + r] = a.data[r * in_dim + k];
            }
        }
    }
    Ok(packed)
}

fn unpack_grouped_lora_a_t_grad_active(
    grad_t: &[f32],
    active_expert_indices: &[usize],
    rank: usize,
    in_dim: usize,
    grad_lora_a: &mut [Option<Vec<f32>>],
) {
    for (compact, &expert_index) in active_expert_indices.iter().enumerate() {
        let Some(grad_a) = grad_lora_a[expert_index].as_mut() else {
            continue;
        };
        for r in 0..rank {
            for k in 0..in_dim {
                grad_a[r * in_dim + k] += grad_t[(compact * in_dim + k) * rank + r];
            }
        }
    }
}

fn pack_grouped_lora_b_t(
    experts: &[MoeGroupedLinearExpert],
    out_dim: usize,
    rank: usize,
    store: &mut TensorStore,
) -> Result<Vec<f32>> {
    let mut packed = vec![0.0_f32; experts.len() * rank * out_dim];
    for (expert_index, expert) in experts.iter().enumerate() {
        let Some(b_id) = expert.lora_b else {
            continue;
        };
        let b = store.tensor_host(b_id)?;
        for n in 0..out_dim {
            for r in 0..rank {
                packed[(expert_index * rank + r) * out_dim + n] = b.data[n * rank + r];
            }
        }
    }
    Ok(packed)
}

fn unpack_grouped_lora_b_t_grad_active(
    grad_t: &[f32],
    active_expert_indices: &[usize],
    out_dim: usize,
    rank: usize,
    grad_lora_b: &mut [Option<Vec<f32>>],
) {
    for (compact, &expert_index) in active_expert_indices.iter().enumerate() {
        let Some(grad_b) = grad_lora_b[expert_index].as_mut() else {
            continue;
        };
        for n in 0..out_dim {
            for r in 0..rank {
                grad_b[n * rank + r] += grad_t[(compact * rank + r) * out_dim + n];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn grouped_lora_backward_tiled(
    store: &mut TensorStore,
    profile: &MoeGroupedLinearProfile,
    active_experts: &[MoeGroupedLinearExpert],
    active_expert_indices: &[usize],
    packed_input: &[f32],
    packed_input_shape: &[usize],
    packed_grad_out: &[f32],
    max_rows: usize,
    in_dim: usize,
    out_dim: usize,
    rank: usize,
    need_input_grad: bool,
    need_lora_a_grad: bool,
    need_lora_b_grad: bool,
    grad_packed_input: &mut Option<Vec<f32>>,
    grad_lora_a: &mut [Option<Vec<f32>>],
    grad_lora_b: &mut [Option<Vec<f32>>],
    expert_tile: usize,
) -> Result<()> {
    let active_len = active_experts.len();
    if active_expert_indices.len() != active_len {
        return Err(AutogradError::InvalidIndicesLen {
            expected: active_len,
            got: active_expert_indices.len(),
        });
    }
    if packed_input_shape != [active_len, max_rows, in_dim] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![active_len, max_rows, in_dim],
            got: packed_input_shape.to_vec(),
        });
    }
    let input_stride = max_rows * in_dim;
    let grad_stride = max_rows * out_dim;
    if packed_input.len() != active_len * input_stride {
        return Err(AutogradError::DataLengthMismatch {
            len: packed_input.len(),
            shape: vec![active_len, max_rows, in_dim],
            size: active_len * input_stride,
        });
    }
    if packed_grad_out.len() != active_len * grad_stride {
        return Err(AutogradError::DataLengthMismatch {
            len: packed_grad_out.len(),
            shape: vec![active_len, max_rows, out_dim],
            size: active_len * grad_stride,
        });
    }

    let expert_tile = expert_tile.max(1);
    for chunk_start in (0..active_len).step_by(expert_tile) {
        let chunk_end = (chunk_start + expert_tile).min(active_len);
        let chunk_len = chunk_end - chunk_start;
        let chunk_experts = &active_experts[chunk_start..chunk_end];
        let chunk_indices = &active_expert_indices[chunk_start..chunk_end];
        let input_start = chunk_start * input_stride;
        let input_end = input_start + chunk_len * input_stride;
        let grad_start = chunk_start * grad_stride;
        let grad_end = grad_start + chunk_len * grad_stride;
        let chunk_input = &packed_input[input_start..input_end];
        let chunk_input_shape = vec![chunk_len, max_rows, in_dim];
        let chunk_grad_shape = vec![chunk_len, max_rows, out_dim];

        let packed_a_t = profile.time_result("pack", "lora_a_t_tile", || {
            pack_grouped_lora_a_t(chunk_experts, rank, in_dim, store)
        })?;
        let packed_a_t_shape = vec![chunk_len, in_dim, rank];
        let packed_b_t = profile.time_result("pack", "lora_b_t_tile", || {
            pack_grouped_lora_b_t(chunk_experts, out_dim, rank, store)
        })?;
        let packed_b_t_shape = vec![chunk_len, rank, out_dim];
        let low = grouped_matmul_forward(
            store,
            profile,
            "lora_low_forward_tile",
            chunk_input,
            &chunk_input_shape,
            &packed_a_t,
            &packed_a_t_shape,
        )?;
        let low_shape = vec![chunk_len, max_rows, rank];
        let mut scaled_grad_out = profile.time_value("clone", "scaled_grad_out_tile", || {
            packed_grad_out[grad_start..grad_end].to_vec()
        });
        profile.time_value("scale", "grad_out_tile", || {
            for (compact, expert) in chunk_experts.iter().enumerate() {
                let scale = expert.lora_scale;
                let expert_base = compact * grad_stride;
                for value in &mut scaled_grad_out[expert_base..expert_base + grad_stride] {
                    *value *= scale;
                }
            }
        });

        let (grad_low, grad_b_t) = grouped_matmul_backward(
            store,
            profile,
            "lora_b_backward_tile",
            &low,
            &low_shape,
            &packed_b_t,
            &packed_b_t_shape,
            &scaled_grad_out,
            &chunk_grad_shape,
            need_input_grad || need_lora_a_grad,
            need_lora_b_grad,
        )?;
        if let Some(grad_b_t) = grad_b_t {
            profile.time_value("unpack", "lora_b_t_grad_tile", || {
                unpack_grouped_lora_b_t_grad_active(
                    &grad_b_t,
                    chunk_indices,
                    out_dim,
                    rank,
                    grad_lora_b,
                );
            });
        }
        if let Some(grad_low) = grad_low {
            let (grad_x_lora, grad_a_t) = grouped_matmul_backward(
                store,
                profile,
                "lora_a_backward_tile",
                chunk_input,
                &chunk_input_shape,
                &packed_a_t,
                &packed_a_t_shape,
                &grad_low,
                &low_shape,
                need_input_grad,
                need_lora_a_grad,
            )?;
            if let (Some(dst), Some(src)) = (grad_packed_input.as_mut(), grad_x_lora.as_ref()) {
                profile.time_value("merge", "lora_input_grad_tile", || {
                    add_active_chunk_input_grad(dst, src, chunk_start, chunk_len, max_rows, in_dim)
                });
            }
            if let Some(grad_a_t) = grad_a_t {
                profile.time_value("unpack", "lora_a_t_grad_tile", || {
                    unpack_grouped_lora_a_t_grad_active(
                        &grad_a_t,
                        chunk_indices,
                        rank,
                        in_dim,
                        grad_lora_a,
                    );
                });
            }
        }
    }
    Ok(())
}

fn add_active_chunk_input_grad(
    dst: &mut [f32],
    src: &[f32],
    chunk_start: usize,
    chunk_len: usize,
    max_rows: usize,
    in_dim: usize,
) {
    let stride = max_rows * in_dim;
    let offset = chunk_start * stride;
    let len = chunk_len * stride;
    add_slice_in_place(&mut dst[offset..offset + len], &src[..len]);
}

fn grouped_matmul_forward(
    store: &TensorStore,
    profile: &MoeGroupedLinearProfile,
    label: &'static str,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Result<Vec<f32>> {
    let a_handle = profile.time_result(label, "upload_a", || store.backend().upload(a, a_shape))?;
    let b_handle = profile.time_result(label, "upload_b", || store.backend().upload(b, b_shape))?;
    let (out_handle, _out_shape) = profile.time_result(label, "matmul", || {
        store
            .backend()
            .matmul(&a_handle, a_shape, &b_handle, b_shape)
    })?;
    profile.time_result(label, "eval", || store.backend().eval(&[&out_handle]))?;
    profile.time_result(label, "readback", || store.backend().readback(&out_handle))
}

#[allow(clippy::too_many_arguments)]
fn grouped_matmul_backward(
    store: &TensorStore,
    profile: &MoeGroupedLinearProfile,
    label: &'static str,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    grad_out: &[f32],
    grad_out_shape: &[usize],
    need_grad_a: bool,
    need_grad_b: bool,
) -> Result<OptionalMatmulGrads> {
    if !need_grad_a && !need_grad_b {
        return Ok((None, None));
    }
    let a_handle = profile.time_result(label, "upload_a", || store.backend().upload(a, a_shape))?;
    let b_handle = profile.time_result(label, "upload_b", || store.backend().upload(b, b_shape))?;
    let grad_handle = profile.time_result(label, "upload_grad", || {
        store.backend().upload(grad_out, grad_out_shape)
    })?;
    let (grad_a, grad_b) = profile.time_result(label, "matmul_backward_device", || {
        store.backend().matmul_backward_device(
            &a_handle,
            a_shape,
            &b_handle,
            b_shape,
            &grad_handle,
            grad_out_shape,
            need_grad_a,
            need_grad_b,
        )
    })?;
    let grad_a = match grad_a {
        Some(handle) => {
            profile.time_result(label, "eval_grad_a", || store.backend().eval(&[&handle]))?;
            Some(profile.time_result(label, "readback_grad_a", || {
                store.backend().readback(&handle)
            })?)
        }
        None => None,
    };
    let grad_b = match grad_b {
        Some(handle) => {
            profile.time_result(label, "eval_grad_b", || store.backend().eval(&[&handle]))?;
            Some(profile.time_result(label, "readback_grad_b", || {
                store.backend().readback(&handle)
            })?)
        }
        None => None,
    };
    Ok((grad_a, grad_b))
}

fn add_slice_in_place(dst: &mut [f32], src: &[f32]) {
    for (lhs, rhs) in dst.iter_mut().zip(src) {
        *lhs += rhs;
    }
}

fn validate_grouped_scatter_shapes(
    values_shape: &[usize],
    weights_shape: &[usize],
    routes: &[MoeGroupedRoute],
    out_rows: usize,
) -> Result<()> {
    if values_shape.len() != 3 {
        return Err(AutogradError::InvalidRank {
            expected: "3",
            got: values_shape.len(),
        });
    }
    if weights_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: weights_shape.len(),
        });
    }
    let experts = values_shape[0];
    let max_rows = values_shape[1];
    let weight_rows = weights_shape[0];
    let top_k = weights_shape[1];
    if out_rows < weight_rows {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![weight_rows],
            got: vec![out_rows],
        });
    }
    for route in routes {
        if route.expert >= experts {
            return Err(AutogradError::IndexOutOfBounds {
                index: route.expert,
                upper: experts,
            });
        }
        if route.row >= max_rows {
            return Err(AutogradError::IndexOutOfBounds {
                index: route.row,
                upper: max_rows,
            });
        }
        if route.token >= weight_rows {
            return Err(AutogradError::IndexOutOfBounds {
                index: route.token,
                upper: weight_rows,
            });
        }
        if route.slot >= top_k {
            return Err(AutogradError::IndexOutOfBounds {
                index: route.slot,
                upper: top_k,
            });
        }
    }
    Ok(())
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum::<f64>() as f32
}

fn validate_routes(
    routes: &[MoeRoute],
    value_rows: usize,
    weight_rows: usize,
    top_k: usize,
) -> Result<()> {
    if routes.len() != value_rows {
        return Err(AutogradError::InvalidIndicesLen {
            expected: value_rows,
            got: routes.len(),
        });
    }
    for route in routes {
        if route.token >= weight_rows {
            return Err(AutogradError::IndexOutOfBounds {
                index: route.token,
                upper: weight_rows,
            });
        }
        if route.slot >= top_k {
            return Err(AutogradError::IndexOutOfBounds {
                index: route.slot,
                upper: top_k,
            });
        }
    }
    Ok(())
}
