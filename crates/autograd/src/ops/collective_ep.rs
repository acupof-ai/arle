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
    let requires_grad = store.tensor(input)?.requires_grad;
    let data = store.tensor_host(input)?.data;
    let out = ep_dispatch(&data, plan, dim);
    let output_id = store.alloc(Tensor::new(out, vec![plan.num_slots(), dim], false)?);
    store.set_requires_grad(output_id, requires_grad)?;
    if tape.enabled && requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::EpDispatch,
            output_id,
            input_ids: smallvec![input],
            saved: SavedContext::EpPlanCtx {
                input,
                src: plan_src_flat(plan),
                num_tokens: plan.num_tokens,
                dim,
            },
        });
    }
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
    let requires_grad = store.tensor(input)?.requires_grad;
    let data = store.tensor_host(input)?.data;
    let out = ep_combine(&data, plan, dim);
    let output_id = store.alloc(Tensor::new(out, vec![plan.num_tokens, dim], false)?);
    store.set_requires_grad(output_id, requires_grad)?;
    if tape.enabled && requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::EpCombine,
            output_id,
            input_ids: smallvec![input],
            saved: SavedContext::EpPlanCtx {
                input,
                src: plan_src_flat(plan),
                num_tokens: plan.num_tokens,
                dim,
            },
        });
    }
    Ok(output_id)
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

#[cfg(test)]
mod tests {
    use super::*;

    // dispatch and combine are S and Sᵀ, so each is the other's adjoint. The
    // inner-product identity <S·x, y> == <x, Sᵀ·y> is the exact adjoint gate — if
    // it holds, backward(dispatch)=combine and backward(combine)=dispatch are
    // correct. No NCCL: world==1 is these host permutations verbatim.
    #[test]
    fn dispatch_combine_are_adjoint() {
        let dim = 3;
        let num_tokens = 4;
        // top_k=2 routing with a capacity drop (slot 5 = None).
        let plan = EpPlan {
            src: vec![Some(0), Some(1), Some(0), Some(3), Some(2), None],
            num_tokens,
        };
        let x: Vec<f32> = (0..num_tokens * dim)
            .map(|i| i as f32 * 0.5 - 1.0)
            .collect();
        let y: Vec<f32> = (0..plan.num_slots() * dim)
            .map(|i| (i as f32).sin())
            .collect();

        let sx = ep_dispatch(&x, &plan, dim); // S·x
        let sty = ep_combine(&y, &plan, dim); // Sᵀ·y

        let lhs: f32 = sx.iter().zip(&y).map(|(a, b)| a * b).sum(); // <S·x, y>
        let rhs: f32 = x.iter().zip(&sty).map(|(a, b)| a * b).sum(); // <x, Sᵀ·y>
        assert!(
            (lhs - rhs).abs() < 1e-4,
            "adjoint identity broken: {lhs} vs {rhs}"
        );
    }

    // Round-trip with no drops and top_k=1 (S a true permutation) must be identity
    // up to the token multiplicity: combine∘dispatch scales each token by how many
    // slots selected it.
    #[test]
    fn combine_of_dispatch_scales_by_selection_count() {
        let dim = 2;
        let num_tokens = 3;
        let plan = EpPlan {
            src: vec![Some(0), Some(1), Some(2), Some(0)], // token 0 selected twice
            num_tokens,
        };
        let x: Vec<f32> = (0..num_tokens * dim).map(|i| i as f32 + 1.0).collect();
        let round = ep_combine(&ep_dispatch(&x, &plan, dim), &plan, dim);
        // token 0 appears in 2 slots → 2×; tokens 1,2 once → 1×.
        for d in 0..dim {
            assert!((round[d] - 2.0 * x[d]).abs() < 1e-5);
            assert!((round[dim + d] - x[dim + d]).abs() < 1e-5);
            assert!((round[2 * dim + d] - x[2 * dim + d]).abs() < 1e-5);
        }
    }

    // ep_combine must equal Sᵀ built explicitly from plan.src, element-wise — a
    // single inner-product equality can be fooled by a compensating index bug, so
    // pin the operator itself. Covers top_k=3 (token 0 in 3 slots) and a fully
    // dropped token (token 2 in no slot).
    #[test]
    fn combine_is_explicit_transpose_topk3_and_dropped_token() {
        let dim = 2;
        let num_tokens = 3;
        // token 0 → slots 0,2,4 (top_k 3); token 1 → slot 1; token 2 → none; drop @5.
        let plan = EpPlan {
            src: vec![Some(0), Some(1), Some(0), None, Some(0), None],
            num_tokens,
        };
        let y: Vec<f32> = (0..plan.num_slots() * dim)
            .map(|i| i as f32 - 3.0)
            .collect();
        let got = ep_combine(&y, &plan, dim);
        // Explicit Sᵀ: out[t] = sum of y[slot] over slots selecting t.
        let mut want = vec![0.0f32; num_tokens * dim];
        for (slot, s) in plan.src.iter().enumerate() {
            if let Some(t) = s {
                for d in 0..dim {
                    want[t * dim + d] += y[slot * dim + d];
                }
            }
        }
        assert_eq!(got, want, "combine must be the explicit transpose");
        // Fully dropped token 2 gets exactly zero.
        assert_eq!(&got[2 * dim..3 * dim], &[0.0, 0.0]);
    }

    // The tape op governs real gradients: at world==1 dispatch records EpDispatch,
    // backward routes to ep_combine. A dropped token must receive ZERO grad. This
    // is the gate the host-only kernel lacked — it proves the adjoint drives
    // autograd, not just an isolated dot product.
    #[test]
    fn dispatch_tape_op_backward_delivers_transpose_and_zero_for_dropped() {
        use crate::{Tensor, ops, tape::Tape, tensor::TensorStore};
        let dim = 2;
        let num_tokens = 3;
        let plan = EpPlan {
            src: vec![Some(0), Some(1), Some(0), None], // token 2 dropped; token 0 twice
            num_tokens,
        };
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let x =
            store.alloc(Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], true).unwrap());
        let disp = ep_dispatch_op(x, &plan, dim, &mut store, &mut tape).unwrap();
        let loss = ops::sum(disp, &mut store, &mut tape).unwrap();
        let grads = tape.backward(loss, &mut store).unwrap();
        let gx = store.to_host(grads[&x]).unwrap();
        // d(sum S·x)/dx[t] = number of slots selecting t: token0→2, token1→1, token2→0.
        assert_eq!(gx, vec![2.0, 2.0, 1.0, 1.0, 0.0, 0.0]);
    }
}
