use std::cmp::Ordering;

use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeRoute {
    pub token: usize,
    pub slot: usize,
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
