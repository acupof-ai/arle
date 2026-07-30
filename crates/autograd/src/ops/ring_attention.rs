//! Context-parallel ring attention (option A) — the flash-2 online-softmax core.
//!
//! CP shards the query over N ranks. Each rank rings the KV block-by-block and
//! attends its local q against one block at a time, merging partial outputs with
//! running (max, denom) so the full-attn transient is O(seq/N · seq/N) per rank
//! instead of option B's O(full_seq) gathered KV. The merge recurrence is NOT
//! differentiated; the saved per-row LSE lets the backward reconstruct each
//! block's probabilities directly (FlashAttention-2).
//!
//! This module owns the pure host math (merge forward + backward adjoint) — the
//! U2 risk, a wrong merge silently corrupts every gradient — AND the live tape op
//! `cp_causal_sdpa` (`BackwardOp::RingAttention`): at world==1 it degenerates to
//! plain causal attention and its taped backward is gated bit-close against
//! `causal_sdpa_recompute`, plus the multi-block merge/adjoint is gated against
//! the full-softmax reference. The NCCL KV ring that feeds remote blocks into the
//! same tile kernels is the pending-remote transport.
//!
//! Block layout is per-(batch·head) `[rows, dim]` row-major, matching the CPU
//! reference `cpu_causal_sdpa_recompute_backward`. Absolute causal masking: q row
//! `r` (absolute `q_start+r`) attends k col `c` of block `j` (absolute
//! `j*block_len+c`) iff `j*block_len+c <= q_start+r`.

/// Per-block partial attention statistics for one (batch·head) tile.
/// `out` is the UNNORMALIZED `P_j @ V_j`; `m`/`l` are the per-row running max and
/// denominator of the online softmax.
struct BlockStats {
    out: Vec<f32>, // [rows, dim] unnormalized
    m: Vec<f32>,   // [rows] row max
    l: Vec<f32>,   // [rows] row denom (sum of exp(S - m))
}

/// One block's unnormalized softmax numerator + row stats. `q`:[rows,dim],
/// `k`/`v`:[blk,dim]. `q_abs` = absolute row of q's row 0; `k_abs` = absolute row
/// of the block's col 0. Future-masked cols contribute nothing.
pub(crate) fn block_stats(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    blk: usize,
    dim: usize,
    scale: f32,
    q_abs: usize,
    k_abs: usize,
) -> BlockStats {
    let mut out = vec![0.0; rows * dim];
    let mut m = vec![f32::NEG_INFINITY; rows];
    let mut l = vec![0.0; rows];
    let mut scores = vec![0.0f32; blk];
    for r in 0..rows {
        let mut max_s = f32::NEG_INFINITY;
        let mut visible = 0usize;
        for c in 0..blk {
            if k_abs + c > q_abs + r {
                break; // causal: cols are ordered, rest are future
            }
            let mut dot = 0.0;
            for d in 0..dim {
                dot += q[r * dim + d] * k[c * dim + d];
            }
            let s = dot * scale;
            scores[c] = s;
            max_s = max_s.max(s);
            visible = c + 1;
        }
        if visible == 0 {
            continue; // whole block is future for this row → zero contribution
        }
        let mut denom = 0.0;
        for c in 0..visible {
            let p = (scores[c] - max_s).exp();
            scores[c] = p;
            denom += p;
            for d in 0..dim {
                out[r * dim + d] += p * v[c * dim + d];
            }
        }
        m[r] = max_s;
        l[r] = denom;
    }
    BlockStats { out, m, l }
}

/// Merge one block's stats into the running (m, l, out) accumulators — the exact
/// flash-2 rescale. All buffers are `[rows*dim]` / `[rows]`.
fn merge_block(
    acc_m: &mut [f32],
    acc_l: &mut [f32],
    acc_out: &mut [f32],
    blk: &BlockStats,
    rows: usize,
    dim: usize,
) {
    for r in 0..rows {
        if blk.l[r] == 0.0 {
            continue; // block was all-future for this row
        }
        let m_new = acc_m[r].max(blk.m[r]);
        let a = if acc_m[r].is_finite() {
            (acc_m[r] - m_new).exp()
        } else {
            0.0
        };
        let b = (blk.m[r] - m_new).exp();
        acc_l[r] = a * acc_l[r] + b * blk.l[r];
        for d in 0..dim {
            acc_out[r * dim + d] = a * acc_out[r * dim + d] + b * blk.out[r * dim + d];
        }
        acc_m[r] = m_new;
    }
}

/// Ring attention forward for one (batch·head) tile over `n` KV blocks in ring
/// order. Returns `(out [rows,dim] normalized, lse [rows])`. `blocks[j] =
/// (k_j, v_j, k_abs_j)`. Local q covers absolute rows `[q_abs, q_abs+rows)`.
///
/// Verified host kernel (see the U2 gate below). The device tape op + NCCL
/// `ring_send_recv_kv` transport that feed remote blocks into this on GPU are the
/// pending-remote integration boundary — a device ring needs per-block SDPA +
/// merge kernels, pod-only. This function IS the correctness-load-bearing math.
#[allow(clippy::type_complexity)]
pub fn ring_forward_tile(
    q: &[f32],
    blocks: &[(&[f32], &[f32], usize)],
    rows: usize,
    dim: usize,
    scale: f32,
    q_abs: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut acc_m = vec![f32::NEG_INFINITY; rows];
    let mut acc_l = vec![0.0f32; rows];
    let mut acc_out = vec![0.0f32; rows * dim];
    for &(k, v, k_abs) in blocks {
        let blk = k.len() / dim;
        let st = block_stats(q, k, v, rows, blk, dim, scale, q_abs, k_abs);
        merge_block(&mut acc_m, &mut acc_l, &mut acc_out, &st, rows, dim);
    }
    let mut lse = vec![f32::NEG_INFINITY; rows];
    for r in 0..rows {
        if acc_l[r] > 0.0 {
            for d in 0..dim {
                acc_out[r * dim + d] /= acc_l[r];
            }
            lse[r] = acc_m[r] + acc_l[r].ln();
        }
    }
    (acc_out, lse)
}

/// Ring attention backward for one (batch·head) tile — the flash-2 adjoint.
/// Replays each block, rebuilding `P_j = exp(S_j - lse)` from the SAVED lse (not
/// a fresh per-block softmax), and accumulates grads. `out`/`lse` are the saved
/// forward outputs. Returns `(grad_q [rows,dim], per-block (grad_k, grad_v))`.
///
/// Verified host kernel (U2 gate). Pairs with [`ring_forward_tile`]; the device
/// integration is pending-remote (see that fn's note).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ring_backward_tile(
    q: &[f32],
    blocks: &[(&[f32], &[f32], usize)],
    out: &[f32],
    lse: &[f32],
    d_out: &[f32],
    rows: usize,
    dim: usize,
    scale: f32,
    q_abs: usize,
) -> (Vec<f32>, Vec<(Vec<f32>, Vec<f32>)>) {
    // Row delta D_r = sum_d d_out[r,d] * out[r,d] (flash backward's row correction).
    let mut delta = vec![0.0f32; rows];
    for r in 0..rows {
        for d in 0..dim {
            delta[r] += d_out[r * dim + d] * out[r * dim + d];
        }
    }
    let mut grad_q = vec![0.0f32; rows * dim];
    let mut per_block = Vec::with_capacity(blocks.len());
    for &(k, v, k_abs) in blocks {
        let blk = k.len() / dim;
        let mut grad_k = vec![0.0f32; blk * dim];
        let mut grad_v = vec![0.0f32; blk * dim];
        for r in 0..rows {
            if lse[r] == f32::NEG_INFINITY {
                continue;
            }
            for c in 0..blk {
                if k_abs + c > q_abs + r {
                    break;
                }
                let mut s = 0.0;
                for d in 0..dim {
                    s += q[r * dim + d] * k[c * dim + d];
                }
                let p = (s * scale - lse[r]).exp(); // reconstructed prob
                let mut dp = 0.0; // d_out · v
                for d in 0..dim {
                    dp += d_out[r * dim + d] * v[c * dim + d];
                    grad_v[c * dim + d] += p * d_out[r * dim + d];
                }
                let d_score = p * (dp - delta[r]);
                for d in 0..dim {
                    grad_q[r * dim + d] += scale * d_score * k[c * dim + d];
                    grad_k[c * dim + d] += scale * d_score * q[r * dim + d];
                }
            }
        }
        per_block.push((grad_k, grad_v));
    }
    (grad_q, per_block)
}

// --- Differentiable tape op ---
//
// `cp_causal_sdpa` folds q/k/v `[B,H,S,D]` into per-(batch·head) tiles and runs
// the verified ring kernels. At world==1 there is ONE local KV block (q_abs=0),
// so it degenerates to plain causal attention and is gate-able on CPU against
// `causal_sdpa_recompute`. The multi-rank ring feeds additional remote KV blocks
// into the SAME tile kernels via the pending-remote NCCL transport; the merge and
// its adjoint (the U2 risk) are identical and already verified.

use crate::{
    AutogradError, Result,
    backend::{Device, DeviceHandle, RingBlockDims},
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};
use smallvec::smallvec;

/// Context-parallel causal attention over the LOCAL shard. `q`/`k`/`v` are
/// `[B,H,S,D]` — this rank's contiguous sequence shard at absolute rows
/// `[cp_rank*S, (cp_rank+1)*S)`. On CUDA with `cp_size > 1` it runs the device
/// ring: `cp_size` KV rotations feeding the one-block flash-2 kernel + on-device
/// merge, never materializing the full sequence (peak O(S·D), not O(full_seq·D) —
/// the fix for the option-B slice_bwd OOM). CPU / world==1 keep the verified host
/// `ring_forward_tile` path (one local block, q_abs=0). Records out + per-row LSE
/// so backward replays the flash-2 adjoint.
pub fn cp_causal_sdpa(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    cp_size: usize,
    cp_rank: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let shape = store.tensor(q)?.shape.clone();
    if shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: shape.len(),
        });
    }
    let (b, h, s, d) = (shape[0], shape[1], shape[2], shape[3]);

    if store.backend().device() == Device::Cuda && cp_size > 1 {
        return cp_causal_sdpa_device_ring(q, k, v, cp_size, cp_rank, &shape, store, tape);
    }

    // CPU / world==1: the whole local sequence is one KV block at absolute row 0.
    let tiles = b * h;
    let scale = 1.0 / (d as f32).sqrt();
    let qd = store.tensor_host(q)?.data;
    let kd = store.tensor_host(k)?.data;
    let vd = store.tensor_host(v)?.data;
    let tile = s * d;
    let mut out = vec![0.0f32; tiles * tile];
    let mut lse = vec![0.0f32; tiles * s];
    for t in 0..tiles {
        let (qt, kt, vt) = (&qd[t * tile..], &kd[t * tile..], &vd[t * tile..]);
        let blocks = [(&kt[..tile], &vt[..tile], 0usize)];
        let (o, l) = ring_forward_tile(&qt[..tile], &blocks, s, d, scale, 0);
        out[t * tile..(t + 1) * tile].copy_from_slice(&o);
        lse[t * s..(t + 1) * s].copy_from_slice(&l);
    }

    let out_id = store.alloc(Tensor::new(out, shape.clone(), false)?);
    let lse_id = store.alloc(Tensor::new(lse, vec![b, h, s], false)?);
    TapeEntry {
        op: BackwardOp::RingAttention,
        output_id: out_id,
        input_ids: smallvec![q, k, v],
        saved: SavedContext::RingAttentionCtx {
            q,
            blocks: smallvec![(k, v, 0usize)],
            lse: lse_id,
            out: out_id,
            rows: s,
            dim: d,
            q_abs: 0,
            cp_size: 1,
            cp_rank: 0,
        },
    }
    .record(store, tape)?;
    Ok(out_id)
}

/// The device ring forward. `q`/`k`/`v` are `[b, H, s, d]` local shards; kernel
/// tiles are `[b*H, s, d]` (q) / `[b*Hkv, s, d]` (k/v), GQA resolved per block in
/// the kernel. Rings k and v `cp_size` times; step j attends the block owned by
/// rank `(cp_rank - j + cp_size) % cp_size` (its rows `owner*s .. owner*s+s`),
/// merged on-device. Every rank issues exactly `cp_size` symmetric send/recvs.
#[allow(clippy::too_many_arguments)]
fn cp_causal_sdpa_device_ring(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    cp_size: usize,
    cp_rank: usize,
    shape: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let (b, h, s, d) = (shape[0], shape[1], shape[2], shape[3]);
    let (kv_shape, kv_heads) = {
        let ks = store.tensor(k)?.shape.clone();
        (ks.clone(), ks[1])
    };
    let scale = 1.0 / (d as f32).sqrt();
    let num_q_tiles = b * h;
    let dims_for = |blk_len: usize, k_abs: usize| RingBlockDims {
        num_q_tiles,
        num_q_heads: h,
        num_kv_heads: kv_heads,
        head_dim: d,
        q_rows: s,
        blk_len,
        q_abs: cp_rank * s,
        k_abs,
        sm_scale: scale,
    };

    store.ensure_device(q)?;
    store.ensure_device(k)?;
    store.ensure_device(v)?;
    let q_h = device_handle(store, q, "ring q")?;
    let rows = num_q_tiles * s;
    // Accumulator init: M=-inf, L=0, O=0 (device-resident, f32).
    let mut acc_m = store
        .backend()
        .upload(&vec![f32::NEG_INFINITY; rows], &[rows])?;
    let mut acc_l = store.backend().upload(&vec![0.0f32; rows], &[rows])?;
    let mut acc_o = store
        .backend()
        .upload(&vec![0.0f32; rows * d], &[rows * d])?;

    // Rotate fresh k/v handles so the tape inputs stay immutable; save each step's
    // block handles + k_abs for the backward replay.
    let mut k_cur = device_handle(store, k, "ring k")?;
    let mut v_cur = device_handle(store, v, "ring v")?;
    let block_elems = b * kv_heads * s * d;
    let mut step_blocks: smallvec::SmallVec<[(TensorId, TensorId, usize); 4]> = smallvec![];
    for j in 0..cp_size {
        let owner = (cp_rank + cp_size - j) % cp_size;
        let k_abs = owner * s;
        let (m2, l2, o2) = store.backend().ring_block_fwd_merge(
            &q_h,
            &k_cur,
            &v_cur,
            &acc_m,
            &acc_l,
            &acc_o,
            dims_for(s, k_abs),
        )?;
        acc_m = m2;
        acc_l = l2;
        acc_o = o2;
        // Persist this step's block for backward (as tape tensors), then rotate.
        let k_id = store.alloc_device_tensor(kv_shape.clone(), k_cur.clone())?;
        let v_id = store.alloc_device_tensor(kv_shape.clone(), v_cur.clone())?;
        step_blocks.push((k_id, v_id, k_abs));
        if j + 1 < cp_size {
            k_cur = store.backend().ring_send_recv_kv(&k_cur, &[block_elems])?;
            v_cur = store.backend().ring_send_recv_kv(&v_cur, &[block_elems])?;
        }
    }

    let (out_h, lse_h) = store
        .backend()
        .ring_block_finalize(&acc_m, &acc_l, &acc_o, rows, d)?;
    let out_id = store.alloc_device_tensor(shape.to_vec(), out_h)?;
    let lse_id = store.alloc_device_tensor(vec![b, h, s], lse_h)?;

    TapeEntry {
        op: BackwardOp::RingAttention,
        output_id: out_id,
        input_ids: smallvec![q, k, v],
        saved: SavedContext::RingAttentionCtx {
            q,
            blocks: step_blocks,
            lse: lse_id,
            out: out_id,
            rows: s,
            dim: d,
            q_abs: cp_rank * s,
            cp_size,
            cp_rank,
        },
    }
    .record(store, tape)?;
    Ok(out_id)
}

fn device_handle(store: &mut TensorStore, id: TensorId, op: &'static str) -> Result<DeviceHandle> {
    store.ensure_device(id)?;
    store
        .tensor(id)?
        .device_handle
        .clone()
        .ok_or(AutogradError::TapeInvariant(match op {
            "ring q" => "ring: q missing device handle",
            "ring k" => "ring: k missing device handle",
            "ring v" => "ring: v missing device handle",
            _ => "ring: input missing device handle",
        }))
}

pub(crate) fn cp_ring_attention_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::RingAttentionCtx {
        q,
        blocks,
        lse,
        out,
        rows,
        dim,
        q_abs,
        cp_size,
        cp_rank,
    } = &entry.saved
    else {
        return Err(AutogradError::TapeInvariant(
            "ring attention backward missing RingAttentionCtx",
        ));
    };
    let (q, lse, out, rows, dim, q_abs, cp_size, _cp_rank) =
        (*q, *lse, *out, *rows, *dim, *q_abs, *cp_size, *cp_rank);

    if store.backend().device() == Device::Cuda && cp_size > 1 {
        return cp_ring_attention_backward_device(
            &entry.input_ids,
            blocks,
            q,
            lse,
            out,
            output_grad_id,
            rows,
            dim,
            q_abs,
            cp_size,
            store,
        );
    }

    // world==1: one local block covers everything (host reference path).
    let (k, v, _k_abs) = blocks[0];
    let need = store.tensor(q)?.requires_grad
        || store.tensor(k)?.requires_grad
        || store.tensor(v)?.requires_grad;
    let mut grads = GradPairs::new();
    if !need {
        return Ok(grads);
    }

    let shape = store.tensor(q)?.shape.clone();
    let (b, h) = (shape[0], shape[1]);
    let tiles = b * h;
    let scale = 1.0 / (dim as f32).sqrt();
    let tile = rows * dim;
    let qd = store.tensor_host(q)?.data;
    let kd = store.tensor_host(k)?.data;
    let vd = store.tensor_host(v)?.data;
    let od = store.tensor_host(out)?.data;
    let ld = store.tensor_host(lse)?.data;
    let dod = store.tensor_host(output_grad_id)?.data;

    let mut gq = vec![0.0f32; tiles * tile];
    let mut gk = vec![0.0f32; tiles * tile];
    let mut gv = vec![0.0f32; tiles * tile];
    for t in 0..tiles {
        let blocks = [(
            &kd[t * tile..t * tile + tile],
            &vd[t * tile..t * tile + tile],
            0usize,
        )];
        let (gq_t, per_block) = ring_backward_tile(
            &qd[t * tile..t * tile + tile],
            &blocks,
            &od[t * tile..t * tile + tile],
            &ld[t * rows..t * rows + rows],
            &dod[t * tile..t * tile + tile],
            rows,
            dim,
            scale,
            q_abs,
        );
        gq[t * tile..(t + 1) * tile].copy_from_slice(&gq_t);
        let (gk_t, gv_t) = &per_block[0];
        gk[t * tile..(t + 1) * tile].copy_from_slice(gk_t);
        gv[t * tile..(t + 1) * tile].copy_from_slice(gv_t);
    }
    grads.push((q, store.alloc(Tensor::new(gq, shape.clone(), false)?)));
    grads.push((k, store.alloc(Tensor::new(gk, shape.clone(), false)?)));
    grads.push((v, store.alloc(Tensor::new(gv, shape, false)?)));
    Ok(grads)
}

/// Device ring backward. `input_ids = [q, k, v]` are THIS rank's local shards;
/// `blocks[j]` are the rotated (k, v, k_abs) handles the forward saved per step.
/// Recompute each block's (grad_q, grad_k, grad_v) from the saved out/lse,
/// accumulate grad_q locally, and ring grad_k/grad_v BACK to their owners: step j's
/// block came from j hops forward, so its grad returns via `cp_size - j` more
/// forward hops (a full loop), landing each on the rank whose LOCAL k/v produced
/// it, where it sums. Rank-symmetric — the ring-home is a function of the hop
/// count `j`, not of `cp_rank`.
#[allow(clippy::too_many_arguments)]
fn cp_ring_attention_backward_device(
    input_ids: &[TensorId],
    blocks: &smallvec::SmallVec<[(TensorId, TensorId, usize); 4]>,
    q: TensorId,
    lse: TensorId,
    out: TensorId,
    output_grad_id: TensorId,
    rows: usize,
    dim: usize,
    q_abs: usize,
    cp_size: usize,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let (k_local, v_local) = (input_ids[1], input_ids[2]);
    let need = store.tensor(q)?.requires_grad
        || store.tensor(k_local)?.requires_grad
        || store.tensor(v_local)?.requires_grad;
    let mut grads = GradPairs::new();
    if !need {
        return Ok(grads);
    }

    let shape = store.tensor(q)?.shape.clone();
    let (b, h, d) = (shape[0], shape[1], shape[3]);
    let kv_shape = store.tensor(k_local)?.shape.clone();
    let kv_heads = kv_shape[1];
    let scale = 1.0 / (dim as f32).sqrt();
    let num_q_tiles = b * h;
    let block_elems = b * kv_heads * rows * d;

    let q_h = device_handle(store, q, "ring q")?;
    let out_h = device_handle(store, out, "ring out")?;
    let lse_h = device_handle(store, lse, "ring lse")?;
    store.ensure_device(output_grad_id)?;
    let dout_h = device_handle(store, output_grad_id, "ring d_out")?;

    // grad_q accumulates across blocks (device-resident); start at zeros.
    let mut grad_q = store.backend().upload(
        &vec![0.0f32; num_q_tiles * rows * d],
        &[num_q_tiles * rows * d],
    )?;
    // Per-step grad_k/grad_v handles, produced in forward-ring order.
    let mut step_gk: smallvec::SmallVec<[DeviceHandle; 4]> = smallvec![];
    let mut step_gv: smallvec::SmallVec<[DeviceHandle; 4]> = smallvec![];
    for &(k_id, v_id, k_abs) in blocks.iter() {
        let k_h = device_handle(store, k_id, "ring k")?;
        let v_h = device_handle(store, v_id, "ring v")?;
        let dims = RingBlockDims {
            num_q_tiles,
            num_q_heads: h,
            num_kv_heads: kv_heads,
            head_dim: d,
            q_rows: rows,
            blk_len: rows,
            q_abs,
            k_abs,
            sm_scale: scale,
        };
        let (gq2, gk_b, gv_b) = store
            .backend()
            .ring_block_bwd(&q_h, &k_h, &v_h, &out_h, &lse_h, &dout_h, &grad_q, dims)?;
        grad_q = gq2;
        step_gk.push(gk_b);
        step_gv.push(gv_b);
    }

    // Ring each step's grad block back to its owner and sum. Step j's block was
    // received after j forward hops; `cp_size - j` more forward hops complete the
    // loop, returning it to the rank that owns that K/V — where it accumulates
    // into that rank's local grad. Every rank issues the same symmetric rotations.
    let mut gk_acc = store
        .backend()
        .upload(&vec![0.0f32; block_elems], &[block_elems])?;
    let mut gv_acc = store
        .backend()
        .upload(&vec![0.0f32; block_elems], &[block_elems])?;
    for (j, (mut gk_h, mut gv_h)) in step_gk.into_iter().zip(step_gv).enumerate() {
        for _ in 0..(cp_size - j) {
            gk_h = store.backend().ring_send_recv_kv(&gk_h, &[block_elems])?;
            gv_h = store.backend().ring_send_recv_kv(&gv_h, &[block_elems])?;
        }
        gk_acc = store
            .backend()
            .add_into_device(&gk_acc, &gk_h, &[block_elems])?;
        gv_acc = store
            .backend()
            .add_into_device(&gv_acc, &gv_h, &[block_elems])?;
    }

    grads.push((q, store.alloc_device_tensor(shape, grad_q)?));
    grads.push((
        k_local,
        store.alloc_device_tensor(kv_shape.clone(), gk_acc)?,
    ));
    grads.push((v_local, store.alloc_device_tensor(kv_shape, gv_acc)?));
    Ok(grads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu_causal_sdpa_recompute_backward;

    fn full_softmax_forward(q: &[f32], k: &[f32], v: &[f32], seq: usize, dim: usize) -> Vec<f32> {
        let scale = 1.0 / (dim as f32).sqrt();
        let mut out = vec![0.0f32; seq * dim];
        let mut sc = vec![0.0f32; seq];
        for r in 0..seq {
            let mut mx = f32::NEG_INFINITY;
            for c in 0..=r {
                let mut dot = 0.0;
                for d in 0..dim {
                    dot += q[r * dim + d] * k[c * dim + d];
                }
                sc[c] = dot * scale;
                mx = mx.max(sc[c]);
            }
            let mut den = 0.0;
            for c in 0..=r {
                sc[c] = (sc[c] - mx).exp();
                den += sc[c];
            }
            for c in 0..=r {
                let p = sc[c] / den;
                for d in 0..dim {
                    out[r * dim + d] += p * v[c * dim + d];
                }
            }
        }
        out
    }

    fn synth(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    // Split a single-head sequence into N ring blocks; the ring forward must match
    // the full causal softmax, and the ring backward must match the full-softmax
    // reference backward. This is the U2 gate — a wrong merge/adjoint fails here,
    // no NCCL required (world==1 semantics; blocks simulate what ranks would send).
    #[test]
    fn ring_matches_full_softmax_forward_and_backward() {
        let (seq, dim, n) = (12usize, 4usize, 3usize);
        let q = synth(seq * dim, 1);
        let k = synth(seq * dim, 2);
        let v = synth(seq * dim, 3);
        let d_out = synth(seq * dim, 4);
        let scale = 1.0 / (dim as f32).sqrt();
        let block_len = seq / n;

        // Reference forward + backward over the whole sequence.
        let ref_out = full_softmax_forward(&q, &k, &v, seq, dim);
        let (ref_gq, ref_gk, ref_gv) = cpu_causal_sdpa_recompute_backward(
            &q,
            &k,
            &v,
            &d_out,
            &[1, 1, seq, dim],
            true,
            true,
            true,
        )
        .unwrap();
        let (ref_gq, ref_gk, ref_gv) = (ref_gq.unwrap(), ref_gk.unwrap(), ref_gv.unwrap());

        // Ring: each of the N q-shards attends the KV blocks 0..=its own (causal
        // prefix), block j owning absolute rows [j*block_len, ..).
        let mut got_out = vec![0.0f32; seq * dim];
        let mut got_gq = vec![0.0f32; seq * dim];
        let mut got_gk = vec![0.0f32; seq * dim];
        let mut got_gv = vec![0.0f32; seq * dim];
        for shard in 0..n {
            let q_abs = shard * block_len;
            let q_shard = &q[q_abs * dim..(q_abs + block_len) * dim];
            let do_shard = &d_out[q_abs * dim..(q_abs + block_len) * dim];
            let blocks: Vec<(&[f32], &[f32], usize)> = (0..=shard)
                .map(|j| {
                    let s = j * block_len;
                    (
                        &k[s * dim..(s + block_len) * dim],
                        &v[s * dim..(s + block_len) * dim],
                        s,
                    )
                })
                .collect();
            let (out, lse) = ring_forward_tile(q_shard, &blocks, block_len, dim, scale, q_abs);
            got_out[q_abs * dim..(q_abs + block_len) * dim].copy_from_slice(&out);
            let (gq, per_block) = ring_backward_tile(
                q_shard, &blocks, &out, &lse, do_shard, block_len, dim, scale, q_abs,
            );
            for (r, val) in gq.iter().enumerate() {
                got_gq[q_abs * dim + r] += val;
            }
            for (j, (gk, gv)) in per_block.iter().enumerate() {
                let s = j * block_len;
                for (i, val) in gk.iter().enumerate() {
                    got_gk[s * dim + i] += val;
                }
                for (i, val) in gv.iter().enumerate() {
                    got_gv[s * dim + i] += val;
                }
            }
        }

        assert!(
            max_diff(&got_out, &ref_out) < 1e-5,
            "ring forward != full softmax"
        );
        assert!(
            max_diff(&got_gq, &ref_gq) < 1e-5,
            "ring grad_q != reference"
        );
        assert!(
            max_diff(&got_gk, &ref_gk) < 1e-5,
            "ring grad_k != reference"
        );
        assert!(
            max_diff(&got_gv, &ref_gv) < 1e-5,
            "ring grad_v != reference"
        );
    }

    // Adversarial coverage: ragged (unequal) blocks + a q_abs NOT aligned to a
    // block boundary + a strictly-future block (all-masked). Reconstruct the same
    // full-softmax reference over the absolute positions and compare — this
    // exercises the k_abs-vs-slice mapping and the visible==0 / l==0 / lse==-inf
    // guard paths the aligned even-block test holds constant.
    #[test]
    fn ring_ragged_blocks_nonaligned_qabs_and_future_block() {
        let dim = 3;
        // Full causal sequence of length 9; local q is the middle rows [2, 6) —
        // q_abs=2 is NOT a multiple of any block length. KV blocks are ragged:
        // [0,3), [3,4) (size 1), [4,7), and a strictly-FUTURE block [7,9).
        let seq = 9;
        let (q_abs, rows) = (2usize, 4usize);
        let kfull = synth(seq * dim, 11);
        let vfull = synth(seq * dim, 12);
        let qfull = synth(seq * dim, 13);
        let scale = 1.0 / (dim as f32).sqrt();
        let q_local = &qfull[q_abs * dim..(q_abs + rows) * dim];
        let d_out = synth(rows * dim, 14);

        let spans = [(0usize, 3usize), (3, 4), (4, 7), (7, 9)];
        let blocks: Vec<(&[f32], &[f32], usize)> = spans
            .iter()
            .map(|&(a, e)| (&kfull[a * dim..e * dim], &vfull[a * dim..e * dim], a))
            .collect();
        let (out, lse) = ring_forward_tile(q_local, &blocks, rows, dim, scale, q_abs);
        let (gq, per_block) = ring_backward_tile(
            q_local, &blocks, &out, &lse, &d_out, rows, dim, scale, q_abs,
        );

        // Reference: full causal softmax for the 4 local rows against ALL keys,
        // masked by absolute causal position (col <= q_abs+r).
        let mut ref_out = vec![0.0f32; rows * dim];
        for r in 0..rows {
            let qrow = q_abs + r;
            let mut sc = vec![f32::NEG_INFINITY; seq];
            let mut mx = f32::NEG_INFINITY;
            for c in 0..=qrow {
                let mut dot = 0.0;
                for e in 0..dim {
                    dot += q_local[r * dim + e] * kfull[c * dim + e];
                }
                sc[c] = dot * scale;
                mx = mx.max(sc[c]);
            }
            let mut den = 0.0;
            for c in 0..=qrow {
                sc[c] = (sc[c] - mx).exp();
                den += sc[c];
            }
            for c in 0..=qrow {
                let p = sc[c] / den;
                for e in 0..dim {
                    ref_out[r * dim + e] += p * vfull[c * dim + e];
                }
            }
        }
        assert!(
            max_diff(&out, &ref_out) < 1e-5,
            "ragged ring forward != reference"
        );

        // The strictly-future block [7,9) is past every local row (max abs row 5),
        // so it must contribute zero grad to its k/v.
        let future = &per_block[3];
        assert!(
            future.0.iter().chain(&future.1).all(|&g| g == 0.0),
            "future block must get zero grad"
        );
        // grad_q is finite and shaped (sanity; full grad parity is the even-block gate).
        assert_eq!(gq.len(), rows * dim);
        assert!(gq.iter().all(|g| g.is_finite()));
    }

    // The tape op governs real gradients: at world==1 cp_causal_sdpa is plain
    // causal attention, so its taped backward must match causal_sdpa_recompute's
    // grads bit-close. This is the gate the host-only kernel lacked.
    #[test]
    fn cp_causal_sdpa_world1_matches_causal_sdpa_recompute() {
        use crate::ops::attention::causal_sdpa_recompute;
        use crate::{Tensor, ops, tape::Tape, tensor::TensorStore};

        let (b, h, s, d) = (1usize, 2usize, 5usize, 4usize);
        let n = b * h * s * d;
        let mk = |store: &mut TensorStore, seed: u64| {
            store.alloc(Tensor::new(synth(n, seed), vec![b, h, s, d], true).unwrap())
        };

        // Reference path.
        let mut s0 = TensorStore::default();
        let mut t0 = Tape::new();
        let (q0, k0, v0) = (mk(&mut s0, 1), mk(&mut s0, 2), mk(&mut s0, 3));
        let o0 = causal_sdpa_recompute(q0, k0, v0, &mut s0, &mut t0).unwrap();
        let loss0 = ops::sum(o0, &mut s0, &mut t0).unwrap();
        let g0 = t0.backward(loss0, &mut s0).unwrap();
        let (rq, rk, rv) = (
            s0.to_host(g0[&q0]).unwrap(),
            s0.to_host(g0[&k0]).unwrap(),
            s0.to_host(g0[&v0]).unwrap(),
        );
        let ro = s0.to_host(o0).unwrap();

        // Ring op path (same synth seeds → identical inputs).
        let mut s1 = TensorStore::default();
        let mut t1 = Tape::new();
        let (q1, k1, v1) = (mk(&mut s1, 1), mk(&mut s1, 2), mk(&mut s1, 3));
        let o1 = cp_causal_sdpa(q1, k1, v1, 1, 0, &mut s1, &mut t1).unwrap();
        let loss1 = ops::sum(o1, &mut s1, &mut t1).unwrap();
        let g1 = t1.backward(loss1, &mut s1).unwrap();

        assert!(
            max_diff(&s1.to_host(o1).unwrap(), &ro) < 1e-5,
            "forward mismatch"
        );
        assert!(
            max_diff(&s1.to_host(g1[&q1]).unwrap(), &rq) < 1e-5,
            "grad_q mismatch"
        );
        assert!(
            max_diff(&s1.to_host(g1[&k1]).unwrap(), &rk) < 1e-5,
            "grad_k mismatch"
        );
        assert!(
            max_diff(&s1.to_host(g1[&v1]).unwrap(), &rv) < 1e-5,
            "grad_v mismatch"
        );
    }
}
