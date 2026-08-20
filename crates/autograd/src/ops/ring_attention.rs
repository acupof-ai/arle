//! Context-parallel ring attention (option A) — the flash-2 online-softmax core.
//!
//! CP shards the query over N ranks. Each rank rings the KV block-by-block and
//! attends its local q against one block at a time, merging partial outputs with
//! running (max, denom) so the full-attn transient is O(seq/N · seq/N) per rank
//! instead of option B's O(full_seq) gathered KV. The merge recurrence is NOT
//! differentiated; the saved per-row LSE lets the backward reconstruct each
//! block's probabilities directly (FlashAttention-2).
//!
//! The pure host math (merge forward + backward adjoint — the U2 risk, a wrong
//! merge silently corrupts every gradient) and the FA3 pair decomposition live
//! tape-free in `cuda_kernels::ring_attention` (engine- and train-callable) and
//! are re-exported here. This module owns the live tape op `cp_causal_sdpa`
//! (`BackwardOp::RingAttention`): at world==1 it degenerates to plain causal
//! attention and its taped backward is gated bit-close against
//! `causal_sdpa_recompute`, plus the multi-block merge/adjoint is gated against
//! the full-softmax reference. The NCCL KV ring that feeds remote blocks into the
//! same tile kernels is the pending-remote transport.
//!
//! Block layout is per-(batch·head) `[rows, dim]` row-major, matching the CPU
//! reference `cpu_causal_sdpa_recompute_backward`. Absolute causal masking: q row
//! `r` (absolute `q_start+r`) attends k col `c` of block `j` (absolute
//! `j*block_len+c`) iff `j*block_len+c <= q_start+r`.

pub use cuda_kernels::ring_attention::{
    PairClass, PosRun, classify_pair, contiguous_pos_runs, ring_backward_tile, ring_forward_tile,
};

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
/// `[B,H,S,D]` — this rank's sequence shard. `positions` gives the ABSOLUTE
/// position of each local row: `None` = a contiguous shard at `[cp_rank*S, ..)`
/// (byte-identical legacy path); `Some` = an explicit map, so a zigzag
/// load-balanced shard (two non-contiguous chunks) masks causally by true
/// position, not local row. On CUDA with `cp_size > 1` it runs the device ring;
/// CPU / world==1 keep the verified host `ring_forward_tile` path. Records out +
/// per-row LSE so backward replays the flash-2 adjoint.
pub fn cp_causal_sdpa(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    cp_size: usize,
    cp_rank: usize,
    positions: Option<&[usize]>,
    k_positions: Option<&[usize]>,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    cp_causal_sdpa_with_prefix(
        q,
        k,
        v,
        cp_size,
        cp_rank,
        positions,
        k_positions,
        None,
        0,
        store,
        tape,
    )
}

/// CP causal SDPA with an optional frozen-prompt-KV prefix. When `prefix_k`/`prefix_v`
/// are given, every q row attends to the full prefix (positions `0..gen_start`) plus
/// the ringed gen K/V. The prefix is a constant leaf — its grad is dropped in backward.
#[allow(clippy::too_many_arguments)]
pub fn cp_causal_sdpa_with_prefix(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    cp_size: usize,
    cp_rank: usize,
    positions: Option<&[usize]>,
    k_positions: Option<&[usize]>,
    prefix: Option<(TensorId, TensorId)>,
    gen_start: usize,
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
        return cp_causal_sdpa_device_ring(
            q,
            k,
            v,
            cp_size,
            cp_rank,
            positions,
            k_positions,
            prefix,
            gen_start,
            &shape,
            store,
            tape,
        );
    }

    // CPU / world==1: the whole local sequence is one KV block. With a prefix,
    // concatenate prefix+gen k/v and run plain causal SDPA with q_start=gen_start
    // (identical to the non-CP frozen path). Gen k/v arrive at kv_heads width;
    // repeat to full heads to match the prefix (which is repeat_kv'd at capture).
    if let Some((prefix_k, prefix_v)) = prefix {
        let kv_heads = store.tensor(k)?.shape[1];
        let kv_repeat = h / kv_heads;
        let k_rep = crate::ops::repeat_kv(k, kv_repeat, store, tape)?;
        let v_rep = crate::ops::repeat_kv(v, kv_repeat, store, tape)?;
        let k_full = crate::ops::cat_seq(prefix_k, k_rep, store, tape)?;
        let v_full = crate::ops::cat_seq(prefix_v, v_rep, store, tape)?;
        return crate::ops::attention::causal_sdpa_recompute_with_q_start(
            q, k_full, v_full, gen_start, store, tape,
        );
    }

    let tiles = b * h;
    let scale = 1.0 / (d as f32).sqrt();
    let q_pos: Vec<usize> = match positions {
        Some(p) => p.to_vec(),
        None => (0..s).map(|r| cp_rank * s + r).collect(),
    };
    // k may be the full local shard while q is a tile of it, so its row count and
    // positions are its own. Silent corruption if this is taken from q.
    let blk_rows = store.tensor(k)?.shape[2];
    let k_pos: Vec<usize> = match k_positions {
        Some(p) => p.to_vec(),
        None if blk_rows == s => q_pos.clone(),
        None => (0..blk_rows).map(|r| cp_rank * blk_rows + r).collect(),
    };
    if k_pos.len() != blk_rows {
        return Err(AutogradError::TapeInvariant(
            "cp_causal_sdpa: k positions do not cover the local k block",
        ));
    }
    let qd = store.tensor_host(q)?.data;
    let kd = store.tensor_host(k)?.data;
    let vd = store.tensor_host(v)?.data;
    let q_tile = s * d;
    let k_tile = blk_rows * d;
    let mut out = vec![0.0f32; tiles * q_tile];
    let mut lse = vec![0.0f32; tiles * s];
    for t in 0..tiles {
        let (qt, kt, vt) = (&qd[t * q_tile..], &kd[t * k_tile..], &vd[t * k_tile..]);
        let blocks = [(&kt[..k_tile], &vt[..k_tile], k_pos.as_slice())];
        let (o, l) = ring_forward_tile(&qt[..q_tile], &blocks, s, d, scale, &q_pos);
        out[t * q_tile..(t + 1) * q_tile].copy_from_slice(&o);
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
            blocks: smallvec![(k, v, q_pos.clone())],
            prefix: None,
            lse: lse_id,
            out: out_id,
            rows: s,
            dim: d,
            q_pos,
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
///
/// Contiguous shards only: the scalar-`q_abs`/`k_abs` kernel assumes each rank's
/// rows are the contiguous block `[owner*s, owner*s+s)`. A zigzag shard
/// (`positions.is_some()`) needs a per-row-position kernel — pending-remote, so
/// it's a loud error here, never a silent contiguous mis-attend.
#[allow(clippy::too_many_arguments)]
fn cp_causal_sdpa_device_ring(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    cp_size: usize,
    cp_rank: usize,
    positions: Option<&[usize]>,
    k_positions: Option<&[usize]>,
    prefix: Option<(TensorId, TensorId)>,
    gen_start: usize,
    shape: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let (b, h, s, d) = (shape[0], shape[1], shape[2], shape[3]);
    // q may be a tile of the local shard while k/v stay full-shard, so every
    // k-side extent comes from k's own shape, never from q's row count.
    let (kv_shape, kv_heads, blk_rows) = {
        let ks = store.tensor(k)?.shape.clone();
        (ks.clone(), ks[1], ks[2])
    };
    let scale = 1.0 / (d as f32).sqrt();
    let num_q_tiles = b * h;
    // Absolute position of each local q row: `positions` if given (zigzag shard),
    // else the contiguous default `cp_rank*s + row`. The kernel masks by these
    // per-row positions (not a scalar base), so a non-contiguous zigzag shard
    // attends the right causal prefix. f32 (exact for seq < 2^24) so positions ride
    // the same f32 ring as k/v — each rank declares only its own, no equal-shard math.
    let q_pos: Vec<usize> = match positions {
        Some(p) => p.to_vec(),
        None => (0..s).map(|r| cp_rank * s + r).collect(),
    };
    let q_pos_f32: Vec<f32> = q_pos.iter().map(|&p| p as f32).collect();
    let dims = RingBlockDims {
        num_q_tiles,
        num_q_heads: h,
        num_kv_heads: kv_heads,
        head_dim: d,
        q_rows: s,
        blk_len: blk_rows,
        sm_scale: scale,
    };

    store.ensure_device(q)?;
    store.ensure_device(k)?;
    store.ensure_device(v)?;
    let q_h = device_handle(store, q, "ring q")?;
    let q_pos_h = store.backend().upload(&q_pos_f32, &[s])?;
    let rows = num_q_tiles * s;

    // Accumulator init: M=-inf, L=0, O=0 (device-resident, f32). With a frozen
    // prompt prefix, run one extra merge step against the full prefix K/V first —
    // every q row (abs position >= gen_start) can attend to every prefix row
    // (0..gen_start), so the causal mask passes all prefix keys. The prefix is
    // full-heads width (repeat_kv'd at capture), so num_kv_heads = num_q_heads.
    let mut acc_m = store
        .backend()
        .upload(&vec![f32::NEG_INFINITY; rows], &[rows])?;
    let mut acc_l = store.backend().upload(&vec![0.0f32; rows], &[rows])?;
    let mut acc_o = store
        .backend()
        .upload(&vec![0.0f32; rows * d], &[rows * d])?;

    let prefix_ctx = if let Some((prefix_k, prefix_v)) = prefix {
        store.ensure_device(prefix_k)?;
        store.ensure_device(prefix_v)?;
        let pk_h = device_handle(store, prefix_k, "ring prefix k")?;
        let pv_h = device_handle(store, prefix_v, "ring prefix v")?;
        // Prefix positions are always 0..gen_start (contiguous from 0).
        let prefix_pos: Vec<usize> = (0..gen_start).collect();
        let prefix_pos_f32: Vec<f32> = (0..gen_start).map(|p| p as f32).collect();
        let prefix_pos_h = store.backend().upload(&prefix_pos_f32, &[gen_start])?;
        let prefix_dims = RingBlockDims {
            num_q_tiles,
            num_q_heads: h,
            num_kv_heads: h, // prefix k/v is repeat_kv'd to full heads
            head_dim: d,
            q_rows: s,
            blk_len: gen_start,
            sm_scale: scale,
        };
        let (m2, l2, o2) = store.backend().ring_block_fwd_merge(
            &q_h,
            &pk_h,
            &pv_h,
            &acc_m,
            &acc_l,
            &acc_o,
            &q_pos_h,
            &prefix_pos_h,
            &q_pos,
            &prefix_pos,
            prefix_dims,
        )?;
        acc_m = m2;
        acc_l = l2;
        acc_o = o2;
        Some((prefix_k, prefix_v, gen_start))
    } else {
        None
    };

    // Rotate fresh k/v handles + their positions so the tape inputs stay immutable;
    // save each step's block handles + k_pos Vec for the backward replay. k_pos
    // starts as THIS rank's own positions and rings with k/v — the block arriving
    // at step j carries the true positions of the rank that owns it (contiguous or
    // zigzag), so no rank computes another's layout.
    let mut k_cur = device_handle(store, k, "ring k")?;
    let mut v_cur = device_handle(store, v, "ring v")?;
    // This rank's own k rows. Defaults to the q positions, which is exact when q is
    // the whole local shard — the only shape that existed before q was tiled.
    let mut kpos_vec: Vec<usize> = match k_positions {
        Some(p) => p.to_vec(),
        None => q_pos.clone(),
    };
    if kpos_vec.len() != blk_rows {
        return Err(AutogradError::TapeInvariant(
            "cp_causal_sdpa: k positions do not cover the local k block",
        ));
    }
    let kpos_f32: Vec<f32> = kpos_vec.iter().map(|&p| p as f32).collect();
    let mut kpos_cur = store.backend().upload(&kpos_f32, &[blk_rows])?;
    let block_elems = b * kv_heads * blk_rows * d;
    let mut step_blocks: smallvec::SmallVec<[(TensorId, TensorId, Vec<usize>); 4]> = smallvec![];
    for j in 0..cp_size {
        let (m2, l2, o2) = store.backend().ring_block_fwd_merge(
            &q_h, &k_cur, &v_cur, &acc_m, &acc_l, &acc_o, &q_pos_h, &kpos_cur, &q_pos, &kpos_vec,
            dims,
        )?;
        acc_m = m2;
        acc_l = l2;
        acc_o = o2;
        // Persist this step's block + its positions for backward, then rotate all
        // three (k, v, k_pos) one hop so the next block's positions match its k/v.
        let k_id = store.alloc_device_tensor(kv_shape.clone(), k_cur.clone())?;
        let v_id = store.alloc_device_tensor(kv_shape.clone(), v_cur.clone())?;
        step_blocks.push((k_id, v_id, kpos_vec.clone()));
        if j + 1 < cp_size {
            k_cur = store.backend().ring_send_recv_kv(&k_cur, &[block_elems])?;
            v_cur = store.backend().ring_send_recv_kv(&v_cur, &[block_elems])?;
            kpos_cur = store.backend().ring_send_recv_kv(&kpos_cur, &[blk_rows])?;
            kpos_vec = ring_rotate_positions(store, &kpos_cur, blk_rows)?;
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
            prefix: prefix_ctx,
            lse: lse_id,
            out: out_id,
            rows: s,
            dim: d,
            q_pos,
            cp_size,
            cp_rank,
        },
    }
    .record(store, tape)?;
    Ok(out_id)
}

/// Read back a ringed f32 position buffer to the `Vec<usize>` the tape ctx stores
/// (backward re-uploads it). Positions are small integers exact in f32; this is one
/// readback per ring step (cp_size total), not per row of the hot loop.
fn ring_rotate_positions(
    store: &mut TensorStore,
    kpos_h: &DeviceHandle,
    s: usize,
) -> Result<Vec<usize>> {
    let f = store.backend().readback(kpos_h)?;
    debug_assert_eq!(f.len(), s);
    Ok(f.iter().map(|&x| x.round() as usize).collect())
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
        prefix,
        lse,
        out,
        rows,
        dim,
        q_pos,
        cp_size,
        cp_rank,
    } = &entry.saved
    else {
        return Err(AutogradError::TapeInvariant(
            "ring attention backward missing RingAttentionCtx",
        ));
    };
    let (q, lse, out, rows, dim, cp_size, _cp_rank) =
        (*q, *lse, *out, *rows, *dim, *cp_size, *cp_rank);
    // Borrowed from `entry` — independent of the `&mut store` calls below, so no clone.
    let q_pos: &[usize] = q_pos;

    if store.backend().device() == Device::Cuda && cp_size > 1 {
        return cp_ring_attention_backward_device(
            &entry.input_ids,
            blocks,
            prefix.as_ref(),
            q,
            lse,
            out,
            output_grad_id,
            rows,
            dim,
            q_pos,
            cp_size,
            store,
        );
    }

    // world==1: one local block covers everything (host reference path).
    let (k, v, k_pos) = &blocks[0];
    let (k, v, k_pos) = (*k, *v, k_pos.as_slice());
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
            k_pos,
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
            q_pos,
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
/// `blocks[j]` are the rotated (k, v, k_pos) handles the forward saved per step.
/// Recompute each block's (grad_q, grad_k, grad_v) from the saved out/lse,
/// accumulate grad_q locally, and ring grad_k/grad_v BACK to their owners: step j's
/// block came from j hops forward, so its grad returns via `cp_size - j` more
/// forward hops (a full loop), landing each on the rank whose LOCAL k/v produced
/// it, where it sums. Rank-symmetric — the ring-home is a function of the hop
/// count `j`, not of `cp_rank`. Contiguous only (the scalar kernel), so positions
/// are `pos[0]` bases; zigzag per-row positions have a CPU forward but no device
/// kernel.
#[allow(clippy::too_many_arguments)]
fn cp_ring_attention_backward_device(
    input_ids: &[TensorId],
    blocks: &smallvec::SmallVec<[(TensorId, TensorId, Vec<usize>); 4]>,
    prefix: Option<&(TensorId, TensorId, usize)>,
    q: TensorId,
    lse: TensorId,
    out: TensorId,
    output_grad_id: TensorId,
    rows: usize,
    dim: usize,
    q_pos: &[usize],
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
    // q may be a tile of the local shard; every k-side extent comes from k's
    // own shape, never from q's row count (mirrors the forward).
    let blk_rows = kv_shape[2];
    let scale = 1.0 / (dim as f32).sqrt();
    let num_q_tiles = b * h;
    let block_elems = b * kv_heads * blk_rows * d;

    let q_h = device_handle(store, q, "ring q")?;
    let out_h = device_handle(store, out, "ring out")?;
    let lse_h = device_handle(store, lse, "ring lse")?;
    store.ensure_device(output_grad_id)?;
    let dout_h = device_handle(store, output_grad_id, "ring d_out")?;
    // q positions (this rank's rows) uploaded once; each block re-uploads its saved
    // k positions so the kernel masks by true position (matches the forward).
    let q_pos_f32: Vec<f32> = q_pos.iter().map(|&p| p as f32).collect();
    let q_pos_h = store.backend().upload(&q_pos_f32, &[rows])?;

    // grad_q accumulates across blocks (device-resident); start at zeros.
    let mut grad_q = store.backend().upload(
        &vec![0.0f32; num_q_tiles * rows * d],
        &[num_q_tiles * rows * d],
    )?;

    // Frozen-prompt-KV prefix replay: run the backward block against the full
    // prefix K/V. The prefix is a constant leaf (requires_grad=false), so its
    // grad_k/grad_v are dropped — only grad_q is kept. Prefix is full-heads width
    // (repeat_kv'd at capture), so num_kv_heads = num_q_heads. Prefix is NOT
    // ring-backed: every rank holds the full prefix, so there's no owner to return
    // its grad to.
    if let Some((prefix_k, prefix_v, gen_start)) = prefix {
        let prefix_len = *gen_start;
        let pk_h = device_handle(store, *prefix_k, "ring prefix k")?;
        let pv_h = device_handle(store, *prefix_v, "ring prefix v")?;
        let prefix_pos: Vec<usize> = (0..prefix_len).collect();
        let prefix_pos_f32: Vec<f32> = (0..prefix_len).map(|p| p as f32).collect();
        let prefix_pos_h = store.backend().upload(&prefix_pos_f32, &[prefix_len])?;
        let prefix_dims = RingBlockDims {
            num_q_tiles,
            num_q_heads: h,
            num_kv_heads: h,
            head_dim: d,
            q_rows: rows,
            blk_len: prefix_len,
            sm_scale: scale,
        };
        let (gq2, _gk_prefix, _gv_prefix) = store.backend().ring_block_bwd(
            &q_h,
            &pk_h,
            &pv_h,
            &out_h,
            &lse_h,
            &dout_h,
            &grad_q,
            &q_pos_h,
            &prefix_pos_h,
            q_pos,
            &prefix_pos,
            prefix_dims,
        )?;
        grad_q = gq2;
    }

    // Per-step grad_k/grad_v handles, produced in forward-ring order.
    let mut step_gk: smallvec::SmallVec<[DeviceHandle; 4]> = smallvec![];
    let mut step_gv: smallvec::SmallVec<[DeviceHandle; 4]> = smallvec![];
    for (k_id, v_id, k_pos) in blocks.iter() {
        let (k_id, v_id) = (*k_id, *v_id);
        let k_pos_f32: Vec<f32> = k_pos.iter().map(|&p| p as f32).collect();
        let k_pos_h = store.backend().upload(&k_pos_f32, &[k_pos.len()])?;
        let k_h = device_handle(store, k_id, "ring k")?;
        let v_h = device_handle(store, v_id, "ring v")?;
        let dims = RingBlockDims {
            num_q_tiles,
            num_q_heads: h,
            num_kv_heads: kv_heads,
            head_dim: d,
            q_rows: rows,
            blk_len: blk_rows,
            sm_scale: scale,
        };
        let (gq2, gk_b, gv_b) = store.backend().ring_block_bwd(
            &q_h, &k_h, &v_h, &out_h, &lse_h, &dout_h, &grad_q, &q_pos_h, &k_pos_h, q_pos, k_pos,
            dims,
        )?;
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
