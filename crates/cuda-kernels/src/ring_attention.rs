//! Context-parallel ring attention core — tape-free, shared by training
//! (autograd's `cp_causal_sdpa` tape op) and the engine's CP path.
//!
//! Host math: the flash-2 online-softmax merge forward + backward adjoint (the
//! U2 risk — a wrong merge silently corrupts every gradient) plus the FA3 pair
//! decomposition of a (q_pos, k_pos) block. Device glue: the FA3 per-block
//! forward merge / backward launches over strided head-major views. f32↔bf16
//! staging and tape bookkeeping stay with the caller (autograd's adapters).
//!
//! Block layout is per-(batch·head) `[rows, dim]` row-major. Absolute causal
//! masking: q row `r` (absolute `q_pos[r]`) attends k col `c` iff
//! `k_pos[c] <= q_pos[r]` — position arrays, not scalar bases, so a zigzag
//! shard (non-contiguous rows) masks correctly.

use anyhow::Result;

#[cfg(feature = "cuda")]
use crate::ffi;
#[cfg(feature = "cuda")]
use anyhow::Context;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Geometry for one context-parallel ring-attention block op. `num_q_tiles =
/// batch * num_q_heads`; `q_rows` = this rank's local q length; `blk_len` = the
/// current ring block's KV length. GQA is resolved device-side via
/// `num_q_heads / num_kv_heads`.
#[derive(Debug, Clone, Copy)]
pub struct RingBlockDims {
    pub num_q_tiles: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub q_rows: usize,
    pub blk_len: usize,
    pub sm_scale: f32,
}

/// Per-block partial attention statistics for one (batch·head) tile.
/// `out` is the UNNORMALIZED `P_j @ V_j`; `m`/`l` are the per-row running max and
/// denominator of the online softmax.
pub struct BlockStats {
    out: Vec<f32>, // [rows, dim] unnormalized
    m: Vec<f32>,   // [rows] row max
    l: Vec<f32>,   // [rows] row denom (sum of exp(S - m))
}

/// One block's unnormalized softmax numerator + row stats. `q`:[rows,dim],
/// `k`/`v`:[blk,dim]. `q_pos[r]` / `k_pos[c]` are the ABSOLUTE positions of q row
/// `r` / k col `c` — arrays, not scalar bases, so a zigzag shard (non-contiguous
/// rows) masks correctly. q row `r` attends k col `c` iff `k_pos[c] <= q_pos[r]`;
/// cols are not assumed ordered (skip future, don't break).
#[allow(clippy::too_many_arguments)]
pub fn block_stats(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    blk: usize,
    dim: usize,
    scale: f32,
    q_pos: &[usize],
    k_pos: &[usize],
) -> BlockStats {
    let mut out = vec![0.0; rows * dim];
    let mut m = vec![f32::NEG_INFINITY; rows];
    let mut l = vec![0.0; rows];
    let mut scores = vec![0.0f32; blk];
    for r in 0..rows {
        let mut max_s = f32::NEG_INFINITY;
        let mut visible = false;
        for c in 0..blk {
            if k_pos[c] > q_pos[r] {
                continue; // future key — masked (cols not ordered under zigzag)
            }
            let mut dot = 0.0;
            for d in 0..dim {
                dot += q[r * dim + d] * k[c * dim + d];
            }
            let s = dot * scale;
            scores[c] = s;
            max_s = max_s.max(s);
            visible = true;
        }
        if !visible {
            continue; // whole block is future for this row → zero contribution
        }
        let mut denom = 0.0;
        for c in 0..blk {
            if k_pos[c] > q_pos[r] {
                continue;
            }
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
pub fn merge_block(
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
/// (k_j, v_j, k_pos_j)` where `k_pos_j[c]` is the absolute position of block j's
/// col c. `q_pos[r]` is the absolute position of local q row r — an array, so a
/// zigzag (non-contiguous) shard masks correctly.
///
/// Verified host kernel (the U2 gate) — this function IS the
/// correctness-load-bearing math the device merge kernels are gated against.
#[allow(clippy::type_complexity)]
pub fn ring_forward_tile(
    q: &[f32],
    blocks: &[(&[f32], &[f32], &[usize])],
    rows: usize,
    dim: usize,
    scale: f32,
    q_pos: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let mut acc_m = vec![f32::NEG_INFINITY; rows];
    let mut acc_l = vec![0.0f32; rows];
    let mut acc_out = vec![0.0f32; rows * dim];
    for &(k, v, k_pos) in blocks {
        let blk = k.len() / dim;
        let st = block_stats(q, k, v, rows, blk, dim, scale, q_pos, k_pos);
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
/// Verified host kernel (U2 gate). Pairs with [`ring_forward_tile`].
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ring_backward_tile(
    q: &[f32],
    blocks: &[(&[f32], &[f32], &[usize])],
    out: &[f32],
    lse: &[f32],
    d_out: &[f32],
    rows: usize,
    dim: usize,
    scale: f32,
    q_pos: &[usize],
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
    for &(k, v, k_pos) in blocks {
        let blk = k.len() / dim;
        let mut grad_k = vec![0.0f32; blk * dim];
        let mut grad_v = vec![0.0f32; blk * dim];
        for r in 0..rows {
            if lse[r] == f32::NEG_INFINITY {
                continue;
            }
            for c in 0..blk {
                if k_pos[c] > q_pos[r] {
                    continue; // future key (cols not ordered under zigzag)
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

// --- FA3 pair decomposition ---
//
// The FA3 route splits a (q_pos, k_pos) block into rectangular FA3 calls: each
// position map is 1 (contiguous) or 2 (zigzag) maximal contiguous runs, and
// with equal-length chunk-aligned shards every (q_run, k_run) pair is either
// strictly past (FULL), the identical diagonal (CAUSAL), or strictly future
// (skip). Anything else is a loud error — never a silent mis-attend.

/// One maximal contiguous ascending run in a position map: local rows
/// `[row, row+len)` hold absolute positions `[abs, abs+len)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PosRun {
    pub row: usize,
    pub len: usize,
    pub abs: usize,
}

/// Split a position map into its maximal contiguous (+1-step) runs.
pub fn contiguous_pos_runs(pos: &[usize]) -> Vec<PosRun> {
    let mut runs: Vec<PosRun> = Vec::new();
    for (row, &abs) in pos.iter().enumerate() {
        match runs.last_mut() {
            Some(run) if abs == run.abs + run.len => run.len += 1,
            _ => runs.push(PosRun { row, len: 1, abs }),
        }
    }
    runs
}

/// How one (q_run, k_run) pair attends under the absolute causal mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairClass {
    /// Every key precedes every query row: full attention.
    Full,
    /// Identical range: the causal diagonal.
    Causal,
    /// Every key is strictly future: no contribution.
    Skip,
}

/// Classify a (q_run, k_run) pair, erroring on any partial overlap — with
/// equal-length chunk-aligned zigzag shards it cannot happen.
pub fn classify_pair(q: PosRun, k: PosRun) -> Result<PairClass> {
    if k.abs + k.len <= q.abs {
        Ok(PairClass::Full)
    } else if k.abs == q.abs && k.len == q.len {
        Ok(PairClass::Causal)
    } else if k.abs > q.abs + q.len - 1 {
        Ok(PairClass::Skip)
    } else {
        anyhow::bail!("ring FA3: q/k runs partially overlap — shard is not chunk-aligned")
    }
}

// --- CP ring FA3 pair route (hd256 / sm_90) ---
// Decomposes the (q_pos, k_pos) block into rectangular FA3 calls per the run
// classification above: each visible (q_run, k_run) pair is one non-varlen
// batch=1 FA3 launch over strided head-major views of the tile-major
// [tiles, seq, d] bf16 buffers (head_stride = seq*d, row_stride = d, run
// pointer offset = run_row*d). Forward merges each pair's normalized (o, lse)
// into the running accumulators as block stats (m = lse, l = 1); backward
// mirrors ring-flash-attention's global-LSE trick (final lse + final o +
// upstream d_out, per-pair causal flag). Scalar kernels remain the non-sm90 /
// non-hd256 fallback (autograd's adapters).

#[cfg(feature = "cuda")]
fn ring_i32(v: usize, label: &'static str) -> Result<i32> {
    i32::try_from(v).map_err(|_| anyhow::anyhow!("{label}"))
}

#[cfg(feature = "cuda")]
fn check_cuda_ffi(status: cudarc::driver::sys::CUresult, label: &'static str) -> Result<()> {
    status.result().context(label)
}

#[cfg(feature = "cuda")]
pub fn ring_fa3_route(
    stream: &CudaStream,
    dims: RingBlockDims,
    q_pos: &[usize],
    k_pos: &[usize],
) -> bool {
    let shape_ok = dims.head_dim == 256
        && q_pos.len() == dims.q_rows
        && k_pos.len() == dims.blk_len
        && ring_fa3_is_sm90(stream);
    // SAFETY: pure host query exported by both the real shim and the stub.
    let real = unsafe { ffi::arle_fa3_real_kernel_marker_cuda() } == 1;
    if shape_ok && !real {
        // A stub build falls back silently at ~1/3 the speed, and such a run has
        // already been mistaken for an FA3 measurement.
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            eprintln!(
                "[cuda-kernels] FA3 ring shape qualifies but the real kernel is absent — running \
                 scalar. Rebuild for an sm_90 target with vendor/flash-attention/hopper present."
            );
        });
    }
    shape_ok && real
}

/// The vendored FA3 units are sm_90a-only — dispatch strictly on 9.0.
#[cfg(feature = "cuda")]
fn ring_fa3_is_sm90(stream: &CudaStream) -> bool {
    use cudarc::driver::sys::CUdevice_attribute as Attr;
    let ctx = stream.context();
    ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .is_ok_and(|v| v == 9)
        && ctx
            .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .is_ok_and(|v| v == 0)
}

#[cfg(feature = "cuda")]
struct RingFa3Pair {
    q: PosRun,
    k: PosRun,
    causal: bool,
}

/// All visible (q_run, k_run) pairs, classified up front so a mis-aligned
/// shard errors loudly before any state is touched.
#[cfg(feature = "cuda")]
fn ring_fa3_pairs(q_pos: &[usize], k_pos: &[usize]) -> Result<Vec<RingFa3Pair>> {
    let k_runs = contiguous_pos_runs(k_pos);
    let mut pairs = Vec::new();
    for q_run in contiguous_pos_runs(q_pos) {
        for &k_run in &k_runs {
            match classify_pair(q_run, k_run)? {
                PairClass::Full => pairs.push(RingFa3Pair {
                    q: q_run,
                    k: k_run,
                    causal: false,
                }),
                PairClass::Causal => pairs.push(RingFa3Pair {
                    q: q_run,
                    k: k_run,
                    causal: true,
                }),
                PairClass::Skip => {}
            }
        }
    }
    Ok(pairs)
}

/// Non-varlen causal metadata ceiling: `round_up(batch=1, 4) * 4 + 1` (see the
/// fwd shim's carve) — covers every pair class.
#[cfg(feature = "cuda")]
const RING_FA3_FWD_METADATA_I32: usize = 17;

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn ring_block_fwd_merge_fa3(
    stream: &Arc<CudaStream>,
    q_bf16: &CudaSlice<u16>,
    k_bf16: &CudaSlice<u16>,
    v_bf16: &CudaSlice<u16>,
    m_in: &CudaSlice<f32>,
    l_in: &CudaSlice<f32>,
    o_in: &CudaSlice<f32>,
    q_pos: &[usize],
    k_pos: &[usize],
    dims: RingBlockDims,
) -> Result<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>)> {
    let pairs = ring_fa3_pairs(q_pos, k_pos)?;
    let (h, hk, d, s, blk) = (
        dims.num_q_heads,
        dims.num_kv_heads,
        dims.head_dim,
        dims.q_rows,
        dims.blk_len,
    );
    let b = dims.num_q_tiles / h.max(1);
    let rows = dims.num_q_tiles * s;
    // Fresh accumulator copies (functional like the scalar merge); each pair
    // then rescales only its run rows in place.
    let mut m_out = stream
        .alloc_zeros::<f32>(rows)
        .context("ring fa3 m_out alloc")?;
    let mut l_out = stream
        .alloc_zeros::<f32>(rows)
        .context("ring fa3 l_out alloc")?;
    let mut o_out = stream
        .alloc_zeros::<f32>(rows * d)
        .context("ring fa3 o_out alloc")?;
    stream
        .memcpy_dtod(m_in, &mut m_out)
        .context("ring fa3 acc carry D2D")?;
    stream
        .memcpy_dtod(l_in, &mut l_out)
        .context("ring fa3 acc carry D2D")?;
    stream
        .memcpy_dtod(o_in, &mut o_out)
        .context("ring fa3 acc carry D2D")?;

    for pair in &pairs {
        let (lq, lk) = (pair.q.len, pair.k.len);
        // Per-pair scratch, reused across the batch loop (stream-ordered).
        let mut o_pair = stream
            .alloc_zeros::<u16>(h * lq * d)
            .context("ring fa3 o_pair alloc")?;
        let mut lse_pair = stream
            .alloc_zeros::<f32>(h * lq)
            .context("ring fa3 lse_pair alloc")?;
        let mut meta = stream
            .alloc_zeros::<i32>(RING_FA3_FWD_METADATA_I32)
            .context("ring fa3 meta alloc")?;
        for bi in 0..b {
            {
                let (q_ptr, _qg) = q_bf16.device_ptr(stream);
                let (k_ptr, _kg) = k_bf16.device_ptr(stream);
                let (v_ptr, _vg) = v_bf16.device_ptr(stream);
                let (op_ptr, _og) = o_pair.device_ptr_mut(stream);
                let (lse_ptr, _lg) = lse_pair.device_ptr_mut(stream);
                let (meta_ptr, _mg) = meta.device_ptr_mut(stream);
                let args = ffi::ArleFa3FwdHd256Args {
                    q: (q_ptr + (((bi * h * s) + pair.q.row) * d) as u64) as *const ffi::Half,
                    k: (k_ptr + (((bi * hk * blk) + pair.k.row) * d) as u64) as *const ffi::Half,
                    v: (v_ptr + (((bi * hk * blk) + pair.k.row) * d) as u64) as *const ffi::Half,
                    o: op_ptr as *mut ffi::Half,
                    softmax_lse: lse_ptr as *mut f32,
                    out_accum: std::ptr::null_mut(),
                    softmax_lse_accum: std::ptr::null_mut(),
                    tile_count_semaphore: meta_ptr as *mut i32,
                    metadata_capacity: RING_FA3_FWD_METADATA_I32 as i32,
                    cu_seqlens_q: std::ptr::null(),
                    seqused_k: std::ptr::null(),
                    batch: 1,
                    total_q: ring_i32(lq, "ring fa3 total_q i32")?,
                    seqlen_q: ring_i32(lq, "ring fa3 seqlen_q i32")?,
                    seqlen_k: ring_i32(lk, "ring fa3 seqlen_k i32")?,
                    num_heads: ring_i32(h, "ring fa3 num_heads i32")?,
                    num_heads_k: ring_i32(hk, "ring fa3 num_heads_k i32")?,
                    head_dim: 256,
                    q_row_stride: d as i64,
                    k_row_stride: d as i64,
                    v_row_stride: d as i64,
                    o_row_stride: d as i64,
                    q_head_stride: (s * d) as i64,
                    k_head_stride: (blk * d) as i64,
                    v_head_stride: (blk * d) as i64,
                    o_head_stride: (lq * d) as i64,
                    softmax_scale: dims.sm_scale,
                    is_causal: pair.causal as i32,
                    num_splits: 1,
                    page_table: std::ptr::null(),
                    page_table_batch_stride: 0,
                    page_size: 0,
                    num_pages: 0,
                    k_page_stride: 0,
                    v_page_stride: 0,
                };
                check_cuda_ffi(
                    // SAFETY: strided head-major views of live bf16 buffers; o/lse
                    // are compact [h, lq(, d)] scratch; meta covers the shim's carve.
                    unsafe { ffi::arle_fa3_fwd_hd256_bf16_cuda(&args, stream.cu_stream()) },
                    "arle_fa3_fwd_hd256_bf16_cuda",
                )?;
            }
            let (mo_ptr, _mog) = m_out.device_ptr_mut(stream);
            let (lo_ptr, _log) = l_out.device_ptr_mut(stream);
            let (oo_ptr, _oog) = o_out.device_ptr_mut(stream);
            let (op_ptr, _og2) = o_pair.device_ptr(stream);
            let (lse_ptr, _lg2) = lse_pair.device_ptr(stream);
            check_cuda_ffi(
                // SAFETY: acc buffers are full [tiles, s(, d)]; pair buffers compact.
                unsafe {
                    ffi::ring_fa3_merge_pair_cuda(
                        mo_ptr as *mut f32,
                        lo_ptr as *mut f32,
                        oo_ptr as *mut f32,
                        lse_ptr as *const f32,
                        op_ptr as *const ffi::Half,
                        ring_i32(h, "ring fa3 merge heads i32")?,
                        ring_i32(bi * h, "ring fa3 merge tile_base i32")?,
                        ring_i32(s, "ring fa3 merge seq i32")?,
                        ring_i32(pair.q.row, "ring fa3 merge run_start i32")?,
                        ring_i32(lq, "ring fa3 merge run_len i32")?,
                        ring_i32(d, "ring fa3 merge head_dim i32")?,
                        stream.cu_stream(),
                    )
                },
                "ring_fa3_merge_pair_cuda",
            )?;
        }
    }
    Ok((m_out, l_out, o_out))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn ring_block_bwd_fa3(
    stream: &Arc<CudaStream>,
    q_bf16: &CudaSlice<u16>,
    k_bf16: &CudaSlice<u16>,
    v_bf16: &CudaSlice<u16>,
    o_bf16: &CudaSlice<u16>,
    lse_slice: &CudaSlice<f32>,
    do_bf16: &CudaSlice<u16>,
    mut gq_out: CudaSlice<f32>,
    mut gk: CudaSlice<f32>,
    mut gv: CudaSlice<f32>,
    q_pos: &[usize],
    k_pos: &[usize],
    dims: RingBlockDims,
) -> Result<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>)> {
    let pairs = ring_fa3_pairs(q_pos, k_pos)?;
    let (h, hk, d, s, blk) = (
        dims.num_q_heads,
        dims.num_kv_heads,
        dims.head_dim,
        dims.q_rows,
        dims.blk_len,
    );
    let b = dims.num_q_tiles / h.max(1);
    let gqa = h != hk;

    for pair in &pairs {
        let (lq, lk) = (pair.q.len, pair.k.len);
        // hd256 sm90 bwd tiles: kBlockM=64 / kBlockN=80 (see the bwd shim).
        let sq_r = lq.div_ceil(64) * 64;
        let sk_r = lk.div_ceil(80) * 80;
        let mut lse_c = stream
            .alloc_zeros::<f32>(h * lq)
            .context("ring fa3 bwd lse alloc")?;
        let mut dq = stream
            .alloc_zeros::<u16>(h * lq * d)
            .context("ring fa3 dq alloc")?;
        let mut dk = stream
            .alloc_zeros::<u16>(hk * lk * d)
            .context("ring fa3 dk alloc")?;
        let mut dv = stream
            .alloc_zeros::<u16>(hk * lk * d)
            .context("ring fa3 dv alloc")?;
        let mut softmax_d = stream
            .alloc_zeros::<f32>(h * sq_r)
            .context("ring fa3 softmax_d alloc")?;
        let mut lse_log2 = stream
            .alloc_zeros::<f32>(h * sq_r)
            .context("ring fa3 lse_log2 alloc")?;
        let mut dq_accum = stream
            .alloc_zeros::<f32>(h * sq_r * 256)
            .context("ring fa3 dq_accum alloc")?;
        let mut dkv_accum = if gqa {
            let dk_a = stream
                .alloc_zeros::<f32>(hk * sk_r * 256)
                .context("ring fa3 dk_accum alloc")?;
            let dv_a = stream
                .alloc_zeros::<f32>(hk * sk_r * 256)
                .context("ring fa3 dv_accum alloc")?;
            Some((dk_a, dv_a))
        } else {
            None
        };
        let dq_sem_len = lq.div_ceil(64) * h;
        let mut dq_sem = stream
            .alloc_zeros::<i32>(dq_sem_len)
            .context("ring fa3 dq_semaphore alloc")?;
        for bi in 0..b {
            {
                let (lse_ptr, _lg) = lse_slice.device_ptr(stream);
                let (lsec_ptr, _lcg) = lse_c.device_ptr_mut(stream);
                check_cuda_ffi(
                    // SAFETY: lse is full [tiles, s]; dst is compact [h, lq].
                    unsafe {
                        ffi::ring_fa3_gather_lse_cuda(
                            lsec_ptr as *mut f32,
                            lse_ptr as *const f32,
                            ring_i32(h, "ring fa3 gather heads i32")?,
                            ring_i32(bi * h, "ring fa3 gather tile_base i32")?,
                            ring_i32(s, "ring fa3 gather seq i32")?,
                            ring_i32(pair.q.row, "ring fa3 gather run_start i32")?,
                            ring_i32(lq, "ring fa3 gather run_len i32")?,
                            stream.cu_stream(),
                        )
                    },
                    "ring_fa3_gather_lse_cuda",
                )?;
            }
            {
                let (q_ptr, _qg) = q_bf16.device_ptr(stream);
                let (k_ptr, _kg) = k_bf16.device_ptr(stream);
                let (v_ptr, _vg) = v_bf16.device_ptr(stream);
                let (o_ptr, _og) = o_bf16.device_ptr(stream);
                let (do_ptr, _dg) = do_bf16.device_ptr(stream);
                let (lsec_ptr, _lcg) = lse_c.device_ptr(stream);
                let (dq_ptr, _dqg) = dq.device_ptr_mut(stream);
                let (dk_ptr, _dkg) = dk.device_ptr_mut(stream);
                let (dv_ptr, _dvg) = dv.device_ptr_mut(stream);
                let (sd_ptr, _sdg) = softmax_d.device_ptr_mut(stream);
                let (ll2_ptr, _llg) = lse_log2.device_ptr_mut(stream);
                let (dqa_ptr, _dqag) = dq_accum.device_ptr_mut(stream);
                let (dka_ptr, dva_ptr, _dkag, _dvag) = match dkv_accum.as_mut() {
                    Some((dk_a, dv_a)) => {
                        let (dka, dkag) = dk_a.device_ptr_mut(stream);
                        let (dva, dvag) = dv_a.device_ptr_mut(stream);
                        (dka as *mut f32, dva as *mut f32, Some(dkag), Some(dvag))
                    }
                    None => (std::ptr::null_mut(), std::ptr::null_mut(), None, None),
                };
                let (sem_ptr, _semg) = dq_sem.device_ptr_mut(stream);
                let q_off = (((bi * h * s) + pair.q.row) * d) as u64;
                let kv_off = (((bi * hk * blk) + pair.k.row) * d) as u64;
                let args = ffi::ArleFa3BwdHd256Args {
                    q: (q_ptr + q_off) as *const ffi::Half,
                    k: (k_ptr + kv_off) as *const ffi::Half,
                    v: (v_ptr + kv_off) as *const ffi::Half,
                    o: (o_ptr + q_off) as *const ffi::Half,
                    dout: (do_ptr + q_off) as *const ffi::Half,
                    softmax_lse: lsec_ptr as *const f32,
                    dq: dq_ptr as *mut ffi::Half,
                    dk: dk_ptr as *mut ffi::Half,
                    dv: dv_ptr as *mut ffi::Half,
                    softmax_d: sd_ptr as *mut f32,
                    softmax_lse_log2: ll2_ptr as *mut f32,
                    dq_accum: dqa_ptr as *mut f32,
                    dk_accum: dka_ptr,
                    dv_accum: dva_ptr,
                    dq_semaphore: sem_ptr as *mut i32,
                    softmax_d_capacity: (h * sq_r) as i64,
                    softmax_lse_log2_capacity: (h * sq_r) as i64,
                    dq_accum_capacity: (h * sq_r * 256) as i64,
                    dk_accum_capacity: if gqa { (hk * sk_r * 256) as i64 } else { 0 },
                    dv_accum_capacity: if gqa { (hk * sk_r * 256) as i64 } else { 0 },
                    dq_semaphore_capacity: dq_sem_len as i64,
                    seqlen_q: ring_i32(lq, "ring fa3 bwd seqlen_q i32")?,
                    seqlen_k: ring_i32(lk, "ring fa3 bwd seqlen_k i32")?,
                    num_heads: ring_i32(h, "ring fa3 bwd num_heads i32")?,
                    num_heads_k: ring_i32(hk, "ring fa3 bwd num_heads_k i32")?,
                    head_dim: 256,
                    q_row_stride: d as i64,
                    k_row_stride: d as i64,
                    v_row_stride: d as i64,
                    o_row_stride: d as i64,
                    do_row_stride: d as i64,
                    dq_row_stride: d as i64,
                    dk_row_stride: d as i64,
                    dv_row_stride: d as i64,
                    q_head_stride: (s * d) as i64,
                    k_head_stride: (blk * d) as i64,
                    v_head_stride: (blk * d) as i64,
                    o_head_stride: (s * d) as i64,
                    do_head_stride: (s * d) as i64,
                    dq_head_stride: (lq * d) as i64,
                    dk_head_stride: (lk * d) as i64,
                    dv_head_stride: (lk * d) as i64,
                    softmax_scale: dims.sm_scale,
                    is_causal: pair.causal as i32,
                };
                check_cuda_ffi(
                    // SAFETY: strided head-major views of live bf16 buffers; compact
                    // dq/dk/dv outputs; scratch sized per the bwd shim's contract
                    // (round_up 64/80 tiles, GQA fp32 accum, dq semaphore).
                    unsafe { ffi::arle_fa3_bwd_hd256_bf16_cuda(&args, stream.cu_stream()) },
                    "arle_fa3_bwd_hd256_bf16_cuda",
                )?;
            }
            // Accumulate the pair's compact bf16 grads into the running f32
            // buffers (dq += into gq carry; dk/dv += into this block's grads).
            let accum = |dst: &mut CudaSlice<f32>,
                         src: &CudaSlice<u16>,
                         heads: usize,
                         tile_base: usize,
                         seq: usize,
                         run: &PosRun|
             -> Result<()> {
                let (dst_ptr, _dg) = dst.device_ptr_mut(stream);
                let (src_ptr, _sg) = src.device_ptr(stream);
                check_cuda_ffi(
                    // SAFETY: dst full [tiles, seq, d]; src compact [heads, run.len, d].
                    unsafe {
                        ffi::ring_fa3_accum_grad_bf16_cuda(
                            dst_ptr as *mut f32,
                            src_ptr as *const ffi::Half,
                            ring_i32(heads, "ring fa3 accum heads i32")?,
                            ring_i32(tile_base, "ring fa3 accum tile_base i32")?,
                            ring_i32(seq, "ring fa3 accum seq i32")?,
                            ring_i32(run.row, "ring fa3 accum run_start i32")?,
                            ring_i32(run.len, "ring fa3 accum run_len i32")?,
                            ring_i32(d, "ring fa3 accum head_dim i32")?,
                            stream.cu_stream(),
                        )
                    },
                    "ring_fa3_accum_grad_bf16_cuda",
                )
            };
            accum(&mut gq_out, &dq, h, bi * h, s, &pair.q)?;
            accum(&mut gk, &dk, hk, bi * hk, blk, &pair.k)?;
            accum(&mut gv, &dv, hk, bi * hk, blk, &pair.k)?;
        }
    }
    Ok((gq_out, gk, gv))
}
