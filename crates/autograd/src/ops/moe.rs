use std::cmp::Ordering;

use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

type OptionalMatmulGrads = (Option<Vec<f32>>, Option<Vec<f32>>);

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

    let mut indices = Vec::with_capacity(tokens * top_k);
    let mut weights = vec![0.0_f32; tokens * top_k];
    for token in 0..tokens {
        let row = &input.data[token * experts..(token + 1) * experts];
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|&lhs, &rhs| {
            row[rhs]
                .partial_cmp(&row[lhs])
                .unwrap_or(Ordering::Equal)
                .then_with(|| lhs.cmp(&rhs))
        });
        let selected = &order[..top_k];
        let max_logit = selected
            .iter()
            .map(|&expert| row[expert])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0_f32;
        for &expert in selected {
            denom += (row[expert] - max_logit).exp();
        }
        for (slot, &expert) in selected.iter().enumerate() {
            indices.push(expert);
            weights[token * top_k + slot] = (row[expert] - max_logit).exp() / denom;
        }
    }

    let weights_id = store.alloc(Tensor::new(
        weights,
        vec![tokens, top_k],
        input.requires_grad,
    )?);
    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::MoeTopKSoftmax,
            output_id: weights_id,
            input_ids: smallvec![logits],
            saved: SavedContext::MoeTopKSoftmaxCtx {
                y: weights_id,
                indices: indices.clone(),
                logits_shape: input.shape,
                top_k,
            },
        });
    }

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

pub fn moe_gather_rows(
    src: TensorId,
    rows: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let input = store.tensor_host(src)?;
    if input.shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: input.shape.len(),
        });
    }
    let src_rows = input.shape[0];
    let cols = input.shape[1];
    for &row in rows {
        if row >= src_rows {
            return Err(AutogradError::IndexOutOfBounds {
                index: row,
                upper: src_rows,
            });
        }
    }

    let mut data = Vec::with_capacity(rows.len() * cols);
    for &row in rows {
        data.extend_from_slice(&input.data[row * cols..(row + 1) * cols]);
    }
    let output_id = store.alloc(Tensor::new(
        data,
        vec![rows.len(), cols],
        input.requires_grad,
    )?);
    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::MoeGatherRows,
            output_id,
            input_ids: smallvec![src],
            saved: SavedContext::MoeGatherRowsCtx {
                rows: rows.to_vec(),
                input_shape: input.shape,
            },
        });
    }
    Ok(output_id)
}

pub(crate) fn moe_gather_rows_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let src = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("moe gather missing src"))?;
    if !store.tensor(src)?.requires_grad {
        return Ok(GradPairs::new());
    }
    let SavedContext::MoeGatherRowsCtx { rows, input_shape } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "moe gather backward missing saved context",
        ));
    };
    if input_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: input_shape.len(),
        });
    }
    let src_rows = input_shape[0];
    let cols = input_shape[1];
    let upstream = store.tensor_host(output_grad_id)?;
    let expected_shape = vec![rows.len(), cols];
    if upstream.shape != expected_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_shape,
            got: upstream.shape,
        });
    }

    let mut grad = vec![0.0_f32; src_rows * cols];
    for (out_row, &src_row) in rows.iter().enumerate() {
        if src_row >= src_rows {
            return Err(AutogradError::IndexOutOfBounds {
                index: src_row,
                upper: src_rows,
            });
        }
        for col in 0..cols {
            grad[src_row * cols + col] += upstream.data[out_row * cols + col];
        }
    }
    let grad_id = store.alloc(Tensor::new(grad, input_shape, false)?);
    Ok(smallvec![(src, grad_id)])
}

pub fn moe_weighted_scatter(
    values: TensorId,
    weights: TensorId,
    routes: &[MoeRoute],
    out_rows: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let values_t = store.tensor_host(values)?;
    let weights_t = store.tensor_host(weights)?;
    if values_t.shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: values_t.shape.len(),
        });
    }
    if weights_t.shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: weights_t.shape.len(),
        });
    }
    let value_rows = values_t.shape[0];
    let hidden = values_t.shape[1];
    let weight_rows = weights_t.shape[0];
    let top_k = weights_t.shape[1];
    validate_routes(routes, value_rows, weight_rows, top_k)?;
    if out_rows < weight_rows {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![weight_rows],
            got: vec![out_rows],
        });
    }

    let mut out = vec![0.0_f32; out_rows * hidden];
    for (value_row, route) in routes.iter().enumerate() {
        let weight = weights_t.data[route.token * top_k + route.slot];
        for col in 0..hidden {
            out[route.token * hidden + col] += values_t.data[value_row * hidden + col] * weight;
        }
    }

    let requires_grad = values_t.requires_grad || weights_t.requires_grad;
    let output_id = store.alloc(Tensor::new(out, vec![out_rows, hidden], requires_grad)?);
    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::MoeWeightedScatter,
            output_id,
            input_ids: smallvec![values, weights],
            saved: SavedContext::MoeWeightedScatterCtx {
                routes: routes.to_vec(),
                values_shape: values_t.shape,
                weights_shape: weights_t.shape,
                out_rows,
            },
        });
    }
    Ok(output_id)
}

pub(crate) fn moe_weighted_scatter_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let values = *entry.input_ids.first().ok_or(AutogradError::TapeInvariant(
        "moe weighted scatter missing values",
    ))?;
    let weights = *entry.input_ids.get(1).ok_or(AutogradError::TapeInvariant(
        "moe weighted scatter missing weights",
    ))?;
    let SavedContext::MoeWeightedScatterCtx {
        routes,
        values_shape,
        weights_shape,
        out_rows,
    } = entry.saved.clone()
    else {
        return Err(AutogradError::TapeInvariant(
            "moe weighted scatter backward missing saved context",
        ));
    };
    if values_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: values_shape.len(),
        });
    }
    if weights_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: weights_shape.len(),
        });
    }
    let value_rows = values_shape[0];
    let hidden = values_shape[1];
    let weight_rows = weights_shape[0];
    let top_k = weights_shape[1];
    validate_routes(&routes, value_rows, weight_rows, top_k)?;

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
        let mut grad_values = vec![0.0_f32; value_rows * hidden];
        for (value_row, route) in routes.iter().enumerate() {
            let weight = weights_t.data[route.token * top_k + route.slot];
            for col in 0..hidden {
                grad_values[value_row * hidden + col] =
                    upstream.data[route.token * hidden + col] * weight;
            }
        }
        let grad_id = store.alloc(Tensor::new(grad_values, values_shape, false)?);
        grads.push((values, grad_id));
    }
    if store.tensor(weights)?.requires_grad {
        let mut grad_weights = vec![0.0_f32; weight_rows * top_k];
        for (value_row, route) in routes.iter().enumerate() {
            let mut dot = 0.0_f32;
            for col in 0..hidden {
                dot += upstream.data[route.token * hidden + col]
                    * values_t.data[value_row * hidden + col];
            }
            grad_weights[route.token * top_k + route.slot] += dot;
        }
        let grad_id = store.alloc(Tensor::new(grad_weights, weights_shape, false)?);
        grads.push((weights, grad_id));
    }
    Ok(grads)
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

    let mut route_lookup = vec![None; experts_len * max_rows];
    for route in routes {
        route_lookup[route.expert * max_rows + route.row] = Some(*route);
    }

    let mut out = vec![0.0_f32; experts_len * max_rows * out_dim];
    for (expert_index, expert) in experts.iter().enumerate() {
        let weight = store.tensor_host(expert.weight)?;
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
        tape.record(TapeEntry {
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
        });
    }
    Ok(output_id)
}

pub(crate) fn moe_grouped_linear_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
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
    let input_t = store.tensor_host(input)?;
    if input_t.shape != input_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: input_shape.clone(),
            got: input_t.shape,
        });
    }

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

    let need_input_grad = store.tensor(input)?.requires_grad;
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

    let packed_input = pack_active_grouped_input(
        &input_t.data,
        &input_shape,
        input_kind,
        &active_routes,
        &active_expert_indices,
        max_rows,
        in_dim,
    )?;
    let packed_input_shape = vec![active_len, max_rows, in_dim];
    let packed_grad_out = pack_active_expert_blocks(
        &upstream.data,
        experts_len,
        max_rows,
        out_dim,
        &active_expert_indices,
    )?;
    let grad_out_shape = vec![active_len, max_rows, out_dim];

    let (grad_packed_input_base, grad_packed_weight_t) = if need_input_grad || need_weight_grad {
        let packed_weight_t = pack_grouped_weight_t(&active_experts, out_dim, in_dim, store)?;
        let packed_weight_t_shape = vec![active_len, in_dim, out_dim];
        grouped_matmul_backward(
            store,
            &packed_input,
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

    let mut grad_packed_input = need_input_grad.then(|| vec![0.0_f32; packed_input.len()]);
    if let (Some(dst), Some(src)) = (grad_packed_input.as_mut(), grad_packed_input_base.as_ref()) {
        add_slice_in_place(dst, src);
    }

    let mut grad_weights: Vec<Option<Vec<f32>>> = experts
        .iter()
        .map(|expert| {
            store
                .tensor(expert.weight)
                .ok()
                .filter(|t| t.requires_grad)
                .map(|t| vec![0.0_f32; t.size])
        })
        .collect();
    if let Some(grad_weight_t) = grad_packed_weight_t {
        unpack_grouped_weight_t_grad_active(
            &grad_weight_t,
            &active_expert_indices,
            out_dim,
            in_dim,
            &mut grad_weights,
        );
    }

    let mut grad_lora_a: Vec<Option<Vec<f32>>> = experts
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
        .collect();
    let mut grad_lora_b: Vec<Option<Vec<f32>>> = experts
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
        .collect();

    let common_rank = common_lora_rank(&expert_shapes)?;
    if common_rank > 0 {
        let packed_a_t = pack_grouped_lora_a_t(&active_experts, common_rank, in_dim, store)?;
        let packed_a_t_shape = vec![active_len, in_dim, common_rank];
        let packed_b_t = pack_grouped_lora_b_t(&active_experts, out_dim, common_rank, store)?;
        let packed_b_t_shape = vec![active_len, common_rank, out_dim];
        let low = grouped_matmul_forward(
            store,
            &packed_input,
            &packed_input_shape,
            &packed_a_t,
            &packed_a_t_shape,
        )?;
        let low_shape = vec![active_len, max_rows, common_rank];
        let mut scaled_grad_out = packed_grad_out.clone();
        for (expert_index, expert) in active_experts.iter().enumerate().take(active_len) {
            let scale = expert.lora_scale;
            let expert_base = expert_index * max_rows * out_dim;
            for value in &mut scaled_grad_out[expert_base..expert_base + max_rows * out_dim] {
                *value *= scale;
            }
        }
        let (grad_low, grad_b_t) = grouped_matmul_backward(
            store,
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
            unpack_grouped_lora_b_t_grad_active(
                &grad_b_t,
                &active_expert_indices,
                out_dim,
                common_rank,
                &mut grad_lora_b,
            );
        }
        if let Some(grad_low) = grad_low {
            let (grad_x_lora, grad_a_t) = grouped_matmul_backward(
                store,
                &packed_input,
                &packed_input_shape,
                &packed_a_t,
                &packed_a_t_shape,
                &grad_low,
                &low_shape,
                need_input_grad,
                need_lora_a_grad,
            )?;
            if let (Some(dst), Some(src)) = (grad_packed_input.as_mut(), grad_x_lora.as_ref()) {
                add_slice_in_place(dst, src);
            }
            if let Some(grad_a_t) = grad_a_t {
                unpack_grouped_lora_a_t_grad_active(
                    &grad_a_t,
                    &active_expert_indices,
                    common_rank,
                    in_dim,
                    &mut grad_lora_a,
                );
            }
        }
    }

    let grad_input = match grad_packed_input {
        Some(grad) => Some(unpack_active_grouped_input_grad(
            &grad,
            &input_shape,
            input_kind,
            &active_routes,
            &active_expert_indices,
            experts_len,
            active_len,
            max_rows,
            in_dim,
        )?),
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
    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::MoeGroupedWeightedScatter,
            output_id,
            input_ids: smallvec![values, weights],
            saved: SavedContext::MoeGroupedWeightedScatterCtx {
                routes: routes.to_vec(),
                values_shape: vec![experts, max_rows, hidden],
                weights_shape: weights_t.shape,
                out_rows,
            },
        });
    }
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

fn grouped_matmul_forward(
    store: &TensorStore,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Result<Vec<f32>> {
    let a_handle = store.backend().upload(a, a_shape)?;
    let b_handle = store.backend().upload(b, b_shape)?;
    let (out_handle, _out_shape) = store
        .backend()
        .matmul(&a_handle, a_shape, &b_handle, b_shape)?;
    store.backend().eval(&[&out_handle])?;
    store.backend().readback(&out_handle)
}

#[allow(clippy::too_many_arguments)]
fn grouped_matmul_backward(
    store: &TensorStore,
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
    let a_handle = store.backend().upload(a, a_shape)?;
    let b_handle = store.backend().upload(b, b_shape)?;
    let grad_handle = store.backend().upload(grad_out, grad_out_shape)?;
    let (grad_a, grad_b) = store.backend().matmul_backward_device(
        &a_handle,
        a_shape,
        &b_handle,
        b_shape,
        &grad_handle,
        grad_out_shape,
        need_grad_a,
        need_grad_b,
    )?;
    let grad_a = match grad_a {
        Some(handle) => {
            store.backend().eval(&[&handle])?;
            Some(store.backend().readback(&handle)?)
        }
        None => None,
    };
    let grad_b = match grad_b {
        Some(handle) => {
            store.backend().eval(&[&handle])?;
            Some(store.backend().readback(&handle)?)
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
