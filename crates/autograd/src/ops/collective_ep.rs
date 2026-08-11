//! Expert-parallel all-to-all (EP) — the differentiable dispatch/combine core.
//!
//! MoE-EP places each expert on one rank; tokens routed to a remote expert are
//! dispatched there, computed, then combined back. Dispatch and combine are the
//! two directions of ONE selection permutation S (0/1, capacity-masked): each
//! dispatched slot selects exactly one source token row; a source token feeds its
//! top_k destinations; a capacity-dropped slot selects nothing.
//!
//!   dispatch:  recv = S · send            (gather token rows to expert slots)
//!   combine:   out  = Sᵀ · expert_out     (scatter-add expert outputs to tokens)
//!
//! They are transposes of one linear map, so the adjoints swap — exactly the
//! all_gather_seq↔reduce_scatter_sum symmetry:
//!   bwd(dispatch) = combine ,   bwd(combine) = dispatch
//! S is a PURE permutation (no gate weights — ARLE keeps the gate multiply in
//! `moe_grouped_weighted_scatter`); folding weights in would break the adjoint.
//!
//! This module owns the verified host kernel (apply S / apply Sᵀ) AND the live
//! differentiable tape ops (`ep_dispatch_op`/`ep_combine_op` +
//! `BackwardOp::EpDispatch`/`EpCombine`): at world==1 the op IS the host
//! permutation, gated through the real autograd tape (a dropped token gets zero
//! grad). The multi-rank NCCL all-to-all that moves rows across ranks is the
//! pending-remote transport feeding the same permutation.

use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    backend::CommAxis,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

/// A capacity-masked selection: `src[slot]` is the source token row gathered into
/// dispatch `slot`, or `None` if the slot is a capacity drop (contributes zero).
/// `num_tokens` is the source row count (the combine output width).
pub struct EpPlan {
    pub src: Vec<Option<usize>>,
    pub num_tokens: usize,
}

impl EpPlan {
    pub fn num_slots(&self) -> usize {
        self.src.len()
    }
}

/// Dispatch: `recv[slot] = send[src[slot]]` (S · send). `send` is
/// `[num_tokens, dim]`, output `[num_slots, dim]`. Dropped slots are zero.
pub fn ep_dispatch(send: &[f32], plan: &EpPlan, dim: usize) -> Vec<f32> {
    let mut recv = vec![0.0f32; plan.num_slots() * dim];
    for (slot, src) in plan.src.iter().enumerate() {
        if let Some(t) = src {
            recv[slot * dim..(slot + 1) * dim].copy_from_slice(&send[t * dim..(t + 1) * dim]);
        }
    }
    recv
}

/// Combine: `out[t] = sum over slots selecting t of expert_out[slot]` (Sᵀ · y).
/// `expert_out` is `[num_slots, dim]`, output `[num_tokens, dim]`. A token feeds
/// multiple slots (top_k), so this SUMS — the adjoint of dispatch's copy.
pub fn ep_combine(expert_out: &[f32], plan: &EpPlan, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; plan.num_tokens * dim];
    for (slot, src) in plan.src.iter().enumerate() {
        if let Some(t) = src {
            for d in 0..dim {
                out[t * dim + d] += expert_out[slot * dim + d];
            }
        }
    }
    out
}

// --- Differentiable tape ops (world==1 = the host permutation above) ---
//
// dispatch and combine are S and Sᵀ, so their adjoints swap. Both record the
// plan in the tape; backward applies the transpose permutation. At world==1 this
// is the whole op (a local gather/scatter); the multi-rank NCCL all-to-all is the
// pending-remote transport that feeds remote rows into the same permutation.

/// `usize::MAX` marks a capacity-dropped slot inside the saved plan.
const DROP: usize = usize::MAX;

fn plan_src_flat(plan: &EpPlan) -> Vec<usize> {
    plan.src.iter().map(|s| s.unwrap_or(DROP)).collect()
}

fn plan_from_flat(src: &[usize], num_tokens: usize) -> EpPlan {
    EpPlan {
        src: src.iter().map(|&s| (s != DROP).then_some(s)).collect(),
        num_tokens,
    }
}

/// Differentiable dispatch: `[num_tokens, dim]` → `[num_slots, dim]` via S.
/// Backward is combine (Sᵀ). Host-only (world==1); records the plan for backward.
pub fn ep_dispatch_op(
    input: TensorId,
    plan: &EpPlan,
    dim: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let data = store.tensor_host(input)?.data;
    let out = ep_dispatch(&data, plan, dim);
    let output_id = store.alloc(Tensor::new(out, vec![plan.num_slots(), dim], false)?);
    TapeEntry {
        op: BackwardOp::EpDispatch,
        output_id,
        input_ids: smallvec![input],
        saved: SavedContext::EpPlanCtx {
            input,
            src: plan_src_flat(plan),
            num_tokens: plan.num_tokens,
            dim,
        },
    }
    .record(store, tape)?;
    Ok(output_id)
}

/// Differentiable combine: `[num_slots, dim]` → `[num_tokens, dim]` via Sᵀ.
/// Backward is dispatch (S).
pub fn ep_combine_op(
    input: TensorId,
    plan: &EpPlan,
    dim: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let data = store.tensor_host(input)?.data;
    let out = ep_combine(&data, plan, dim);
    let output_id = store.alloc(Tensor::new(out, vec![plan.num_tokens, dim], false)?);
    TapeEntry {
        op: BackwardOp::EpCombine,
        output_id,
        input_ids: smallvec![input],
        saved: SavedContext::EpPlanCtx {
            input,
            src: plan_src_flat(plan),
            num_tokens: plan.num_tokens,
            dim,
        },
    }
    .record(store, tape)?;
    Ok(output_id)
}

/// This rank's `recv_counts` from every rank's `send_counts` row: all-gather the
/// `[world]` rows over the expert group, take column `rank`.
pub fn ep_exchange_counts(send_counts: &[usize], store: &mut TensorStore) -> Result<Vec<usize>> {
    let (world, rank) = store.backend().comm_world_rank(CommAxis::Expert);
    if world <= 1 {
        return Ok(send_counts.to_vec());
    }
    if send_counts.len() != world {
        return Err(AutogradError::TapeInvariant(
            "ep_exchange_counts: send_counts length must equal the group size",
        ));
    }
    let row: Vec<f32> = send_counts.iter().map(|&c| c as f32).collect();
    let handle = store.backend().upload(&row, &[world])?;
    let gathered = store
        .backend()
        .all_gather_seq_device(&handle, &[world], CommAxis::Expert)?;
    let matrix = store.backend().readback(&gathered)?;
    Ok((0..world)
        .map(|j| matrix[j * world + rank].round() as usize)
        .collect())
}

/// Differentiable variable-split row exchange over the expert group — the wire
/// between dispatch (rows pre-grouped by destination rank) and the local expert
/// compute. Backward is the reverse exchange (adjoint of a permutation).
pub fn ep_exchange_rows_op(
    input: TensorId,
    send_counts: &[usize],
    recv_counts: &[usize],
    dim: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_device(input)?;
    let handle = store
        .tensor(input)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "ep_exchange_rows: ensure_device left tensor without a device handle",
        ))?
        .clone();
    let out = store.backend().ep_exchange_rows_device(
        &handle,
        dim,
        send_counts,
        recv_counts,
        CommAxis::Expert,
    )?;
    let recv_rows: usize = recv_counts.iter().sum();
    let output_id = store.alloc_device_tensor(vec![recv_rows, dim], out)?;
    TapeEntry {
        op: BackwardOp::EpExchange,
        output_id,
        input_ids: smallvec![input],
        saved: SavedContext::EpExchangeCtx {
            input,
            send_counts: send_counts.to_vec(),
            recv_counts: recv_counts.to_vec(),
            dim,
        },
    }
    .record(store, tape)?;
    Ok(output_id)
}

/// bwd(exchange) = exchange with the counts swapped.
pub(crate) fn ep_exchange_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::EpExchangeCtx {
        input,
        send_counts,
        recv_counts,
        dim,
    } = &entry.saved
    else {
        return Err(AutogradError::TapeInvariant(
            "EpExchange missing EpExchangeCtx",
        ));
    };
    let (input, send_counts, recv_counts, dim) =
        (*input, send_counts.clone(), recv_counts.clone(), *dim);
    if !store.tensor(input)?.requires_grad {
        return Ok(GradPairs::new());
    }
    store.ensure_device(output_grad_id)?;
    let upstream = store
        .tensor(output_grad_id)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "EpExchange backward: upstream missing device handle",
        ))?
        .clone();
    let grad = store.backend().ep_exchange_rows_device(
        &upstream,
        dim,
        &recv_counts,
        &send_counts,
        CommAxis::Expert,
    )?;
    let send_rows: usize = send_counts.iter().sum();
    let grad_id = store.alloc_device_tensor(vec![send_rows, dim], grad)?;
    Ok(smallvec![(input, grad_id)])
}

fn ep_plan_ctx(entry: &TapeEntry) -> Result<(TensorId, EpPlan, usize)> {
    let SavedContext::EpPlanCtx {
        input,
        src,
        num_tokens,
        dim,
    } = &entry.saved
    else {
        return Err(AutogradError::TapeInvariant("EP op missing EpPlanCtx"));
    };
    Ok((*input, plan_from_flat(src, *num_tokens), *dim))
}

/// bwd(dispatch) = combine: grad flows `[num_slots,dim]` → `[num_tokens,dim]`.
pub(crate) fn ep_dispatch_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let (input, plan, dim) = ep_plan_ctx(entry)?;
    if !store.tensor(input)?.requires_grad {
        return Ok(GradPairs::new());
    }
    let upstream = store.tensor_host(output_grad_id)?.data;
    let grad = ep_combine(&upstream, &plan, dim);
    let grad_id = store.alloc(Tensor::new(grad, vec![plan.num_tokens, dim], false)?);
    Ok(smallvec![(input, grad_id)])
}

/// bwd(combine) = dispatch: grad flows `[num_tokens,dim]` → `[num_slots,dim]`.
pub(crate) fn ep_combine_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let (input, plan, dim) = ep_plan_ctx(entry)?;
    if !store.tensor(input)?.requires_grad {
        return Ok(GradPairs::new());
    }
    let upstream = store.tensor_host(output_grad_id)?.data;
    let grad = ep_dispatch(&upstream, &plan, dim);
    let grad_id = store.alloc(Tensor::new(grad, vec![plan.num_slots(), dim], false)?);
    Ok(smallvec![(input, grad_id)])
}
