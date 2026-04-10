//! Linear projection ops: GEMV (decode) and GEMM (prefill/batch).
//!
//! Dispatch priority for `gemv()` (single token, decode path):
//!   1. Quantized INT2/4/8 → `w{2,4,8}a16_gemv_cuda` (fused dequant)
//!   2. BF16 → `gemv_cuda` (handwritten BF16×4 vectorized kernel)
//!
//! Dispatch priority for `gemm_into()` (batched, prefill path):
//!   1. Marlin W4 → `marlin_gemm_cuda` (tensor core, 5-25× TTFT speedup)
//!   2. TurboQuant → bulk dequant + cuBLAS GEMM
//!   3. Quantized INT → `w{2,4,8}a16_gemv_batch_cuda`
//!   4. BF16, N=1 → `gemm_graphsafe_cuda` (cuBLAS, CUDA Graph safe)
//!   5. BF16, N>1 → `gemm_cuda` (cuBLAS with workspace)

use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};

/// Matrix-vector multiplication: y = A @ x
/// A: (M, K) row-major, x: (K,), y: (M,)
/// Supports BF16, W8A16, W4A16, and W2A16 weights.
pub fn gemv(ctx: &DeviceContext, a: &DeviceMatrix, x: &DeviceVec, y: &mut DeviceVec) -> Result<()> {
    assert_eq!(a.cols, x.len, "A cols {} != x len {}", a.cols, x.len);
    assert_eq!(a.rows, y.len, "A rows {} != y len {}", a.rows, y.len);

    // ── Q4_K native GEMV (packed superblocks, no BF16 intermediate) ──
    if a.quant_bits == 44 {
        let qw = a
            .qweight
            .as_ref()
            .expect("Q4_K DeviceMatrix missing qweight");
        let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (y_ptr, _gy) = y.data.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::q4k_gemv_cuda(
                qw_ptr as *const u8,
                x_ptr as *const ffi::Half,
                y_ptr as *mut ffi::Half,
                a.rows as i32,
                a.cols as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
        return Ok(());
    }

    // ── Quantized weight dispatch (W2A16 / W4A16 / W8A16) ──
    if let (Some(qw), Some(qs)) = (&a.qweight, &a.qscales) {
        let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
        let (qs_ptr, _gqs) = qs.device_ptr(&ctx.stream);
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (y_ptr, _gy) = y.data.device_ptr_mut(&ctx.stream);

        let rows = a.rows as i32;
        let cols = a.cols as i32;
        let grp = a.group_size as i32;
        let stream = ctx.stream.cu_stream();

        unsafe {
            match a.quant_bits {
                2 => ffi::w2a16_gemv_cuda(
                    qw_ptr as *const u8,
                    qs_ptr as *const ffi::Half,
                    x_ptr as *const ffi::Half,
                    y_ptr as *mut ffi::Half,
                    rows,
                    cols,
                    grp,
                    stream,
                ),
                4 => ffi::w4a16_gemv_cuda(
                    qw_ptr as *const u8,
                    qs_ptr as *const ffi::Half,
                    x_ptr as *const ffi::Half,
                    y_ptr as *mut ffi::Half,
                    rows,
                    cols,
                    grp,
                    stream,
                ),
                _ => ffi::w8a16_gemv_cuda(
                    qw_ptr as *const i8,
                    qs_ptr as *const ffi::Half,
                    x_ptr as *const ffi::Half,
                    y_ptr as *mut ffi::Half,
                    rows,
                    cols,
                    grp,
                    stream,
                ),
            }
            .result()?;
        }
        return Ok(());
    }

    let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = y.data.device_ptr_mut(&ctx.stream);

    // Handwritten GEMV with BF16×4 vectorized loads.
    // cuBLAS GEMM(M,1,K) was tested but has higher dispatch overhead
    // on Ada (L4) for single-vector operations. The handwritten kernel
    // wins at B=1; cuBLAS wins at B≥2 (handled by gemm_into path).
    // On A100/H100 with tensor cores, cuBLAS may be faster — profile first.
    unsafe {
        ffi::gemv_cuda(
            a_ptr as *const ffi::Half,
            x_ptr as *const ffi::Half,
            y_ptr as *mut ffi::Half,
            a.rows as i32,
            a.cols as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}
/// Linear layer: y = weight @ x
pub(crate) fn linear(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceMatrix,
) -> Result<DeviceVec> {
    let mut y = DeviceVec::zeros(ctx, weight.rows)?;
    gemv(ctx, weight, x, &mut y)?;
    Ok(y)
}

/// Fully fused MLP into pre-allocated output buffer.
/// For quantized weights, falls back to separate gate/up GEMVs + silu_mul + down GEMV.
pub fn fused_mlp_into(
    ctx: &DeviceContext,
    x: &DeviceVec,
    gate_proj: &DeviceMatrix,
    up_proj: &DeviceMatrix,
    down_proj: &DeviceMatrix,
    act: &mut DeviceVec,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(gate_proj.cols, x.len, "gate_proj cols != x len");
    assert_eq!(up_proj.cols, x.len, "up_proj cols != x len");
    assert_eq!(
        gate_proj.rows, up_proj.rows,
        "gate and up must have same output dim"
    );
    assert_eq!(
        down_proj.cols, gate_proj.rows,
        "down_proj cols != intermediate_size"
    );
    assert_eq!(down_proj.rows, out.len, "down_proj rows != out len");
    assert_eq!(act.len, gate_proj.rows, "act len != intermediate_size");

    // ── Quantized weights: separate gate/up GEMVs + silu_mul + down GEMV ──
    if gate_proj.is_quantized() {
        let intermediate_size = gate_proj.rows;
        let mut up_out = DeviceVec::zeros(ctx, intermediate_size)?;
        gemv(ctx, gate_proj, x, act)?;
        gemv(ctx, up_proj, x, &mut up_out)?;
        // silu(gate) * up → act (in-place)
        {
            let (act_ptr, _ga) = act.data.device_ptr_mut(&ctx.stream);
            let (up_ptr, _gu) = up_out.data.device_ptr(&ctx.stream);
            unsafe {
                ffi::silu_mul_triton_aot_cuda(
                    act_ptr as *const ffi::Half,
                    up_ptr as *const ffi::Half,
                    act_ptr as *mut ffi::Half,
                    intermediate_size as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
        // down_proj @ act → out
        let act_ref: &DeviceVec = act;
        gemv(ctx, down_proj, act_ref, out)?;
        return Ok(());
    }

    let hidden_size = x.len;
    let intermediate_size = gate_proj.rows;

    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (gate_ptr, _gg) = gate_proj.data.device_ptr(&ctx.stream);
    let (up_ptr, _gu) = up_proj.data.device_ptr(&ctx.stream);
    let (down_ptr, _gd) = down_proj.data.device_ptr(&ctx.stream);
    let (act_ptr, _ga) = act.data.device_ptr_mut(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::fused_mlp_cuda(
            x_ptr as *const ffi::Half,
            gate_ptr as *const ffi::Half,
            up_ptr as *const ffi::Half,
            down_ptr as *const ffi::Half,
            act_ptr as *mut ffi::Half,
            out_ptr as *mut ffi::Half,
            hidden_size as i32,
            intermediate_size as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// GEMM: Y = weight @ X (batched linear projection)
/// weight: [out_dim, in_dim] row-major, X: HiddenStates [in_dim, seq_len], Y: HiddenStates [out_dim, seq_len]
pub fn gemm(ctx: &DeviceContext, weight: &DeviceMatrix, x: &HiddenStates) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, weight.rows, x.seq_len)?;
    gemm_into(ctx, weight, x, &mut out);
    Ok(out)
}

/// GEMM into pre-allocated output buffer (zero allocation).
/// For seq_len=1, uses the graph-safe cuBLAS handle (no workspace) for lower
/// latency while preserving numerical parity with the prefill path.
pub(crate) fn gemm_into(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) {
    assert_eq!(
        weight.cols, x.hidden_dim,
        "weight cols {} != hidden_dim {}",
        weight.cols, x.hidden_dim
    );
    assert_eq!(
        out.hidden_dim, weight.rows,
        "out hidden_dim {} != weight rows {}",
        out.hidden_dim, weight.rows
    );
    assert_eq!(
        out.seq_len, x.seq_len,
        "out seq_len {} != x seq_len {}",
        out.seq_len, x.seq_len
    );

    // ── Marlin W4 GEMM for prefill (seq_len > 1) ──
    // Massive speedup (5-25x TTFT) but causes decode quality degradation
    // due to FP16↔BF16 precision differences at prefill/decode boundary.
    // Enable with models repacked via scripts/marlin_repack.py.
    if x.seq_len > 1 && weight.has_marlin() {
        let mp = weight.marlin_packed.as_ref().unwrap();
        let ms = weight.marlin_scales.as_ref().unwrap();
        let m = x.seq_len;
        let n = weight.rows;
        let k = weight.cols;

        // BF16→FP16 input conversion
        let mut x_fp16: CudaSlice<u16> = ctx.stream.alloc_zeros(m * k).expect("alloc x_fp16");
        {
            let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
            let (xf_ptr, _gf) = x_fp16.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::bf16_to_fp16_cuda(
                    x_ptr as *const ffi::Half,
                    xf_ptr as *mut ffi::Half,
                    (m * k) as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .expect("bf16_to_fp16 failed");
            }
        }

        // FP16 output buffer
        let mut y_fp16: CudaSlice<u16> = ctx.stream.alloc_zeros(m * n).expect("alloc y_fp16");

        // Marlin workspace (lock buffer)
        let sms = ctx.sm_count() as i32;
        let ws_size = unsafe { ffi::marlin_workspace_size(n as i32, sms) };
        let ws_elems = (ws_size + 3) / 4; // round up to i32 count
        let mut workspace: CudaSlice<i32> = ctx
            .stream
            .alloc_zeros(ws_elems)
            .expect("alloc marlin workspace");

        // Run Marlin GEMM
        {
            let (xf_ptr, _g1) = x_fp16.device_ptr(&ctx.stream);
            let (mp_ptr, _g2) = mp.device_ptr(&ctx.stream);
            let (yf_ptr, _g3) = y_fp16.device_ptr_mut(&ctx.stream);
            let (ms_ptr, _g4) = ms.device_ptr(&ctx.stream);
            let (ws_ptr, _g5) = workspace.device_ptr_mut(&ctx.stream);
            let ret = unsafe {
                ffi::marlin_gemm_cuda(
                    xf_ptr as *const ffi::Half,
                    mp_ptr as *const u8,
                    yf_ptr as *mut ffi::Half,
                    ms_ptr as *mut ffi::Half, // scales (fp16)
                    m as i32,
                    n as i32,
                    k as i32,
                    ws_ptr as *mut i32,
                    weight.group_size as i32,
                    0, // dev
                    ctx.stream.cu_stream(),
                    -1, // thread_k (auto)
                    -1, // thread_n (auto)
                    sms,
                    16, // max_par
                )
            };
            assert_eq!(ret, 0, "marlin_gemm_cuda failed with code {ret}");
        }

        // FP16→BF16 output conversion
        {
            let (yf_ptr, _g1) = y_fp16.device_ptr(&ctx.stream);
            let (y_ptr, _g2) = out.data.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::fp16_to_bf16_cuda(
                    yf_ptr as *const ffi::Half,
                    y_ptr as *mut ffi::Half,
                    (m * n) as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .expect("fp16_to_bf16 failed");
            }
        }

        return;
    }

    // ── TurboQuant weight dispatch (fused dequant GEMV / dequant + cuBLAS) ──
    if weight.has_tq() {
        let tq_p = weight.tq_packed.as_ref().unwrap();
        let tq_s = weight.tq_scales.as_ref().unwrap();
        let tq_sg = weight.tq_signs.as_ref().unwrap();
        let tq_c = weight.tq_centroids.as_ref().unwrap();

        let n = weight.rows as i32;
        let k = weight.cols as i32;
        let gs = weight.group_size as i32;
        let num_groups = (weight.cols / weight.group_size) as i32;
        let effective_bits = if weight.tq_bits == 3 {
            4
        } else {
            weight.tq_bits as usize
        };
        let packed_cols = ((weight.cols * effective_bits + 7) / 8) as i32;
        let bits = weight.tq_bits as i32;
        let stream = ctx.stream.cu_stream();

        let (tp_ptr, _g1) = tq_p.device_ptr(&ctx.stream);
        let (ts_ptr, _g2) = tq_s.device_ptr(&ctx.stream);
        let (tsg_ptr, _g3) = tq_sg.device_ptr(&ctx.stream);
        let (tc_ptr, _g4) = tq_c.device_ptr(&ctx.stream);
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

        if x.seq_len == 1 {
            // Decode: fused dequant-GEMV (reads packed weights, ~4x less bandwidth)
            unsafe {
                ffi::turboquant_weight_gemv_cuda(
                    tp_ptr as *const u8,
                    ts_ptr as *const ffi::Half,
                    tsg_ptr as *const i8,
                    tc_ptr as *const f32,
                    x_ptr as *const ffi::Half,
                    y_ptr as *mut ffi::Half,
                    n,
                    k,
                    gs,
                    packed_cols,
                    num_groups,
                    bits,
                    stream,
                );
            }
        } else {
            // Prefill: bulk dequant to BF16 workspace → cuBLAS GEMM
            let ws_size = weight.rows * weight.cols;
            let mut workspace: CudaSlice<bf16> = ctx
                .stream
                .alloc_zeros(ws_size)
                .expect("alloc TQ dequant workspace");
            let (ws_ptr, _gws) = workspace.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::turboquant_weight_dequant_cuda(
                    tp_ptr as *const u8,
                    ts_ptr as *const ffi::Half,
                    tsg_ptr as *const i8,
                    tc_ptr as *const f32,
                    ws_ptr as *mut ffi::Half,
                    n,
                    k,
                    gs,
                    packed_cols,
                    num_groups,
                    bits,
                    stream,
                );
                ffi::gemm_cuda(
                    ws_ptr as *const ffi::Half,
                    x_ptr as *const ffi::Half,
                    y_ptr as *mut ffi::Half,
                    n,
                    x.seq_len as i32,
                    k,
                    stream,
                )
                .result()
                .expect("TQ dequant+cuBLAS GEMM failed");
            }
        }
        return;
    }

    // ── Q4_K native packed dispatch ──
    // Decode (seq_len == 1): q4k_gemv_batch_cuda with B=1 (equivalent to q4k_gemv_cuda).
    // Prefill (seq_len > 1): chunked dequant-to-BF16 tile + cuBLAS GEMM to reuse tensor cores.
    if weight.quant_bits == 44 {
        let qw = weight
            .qweight
            .as_ref()
            .expect("Q4_K DeviceMatrix missing qweight");
        let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);
        let n = weight.rows as i32;
        let k = weight.cols as i32;
        let stream = ctx.stream.cu_stream();

        if x.seq_len == 1 {
            unsafe {
                ffi::q4k_gemv_cuda(
                    qw_ptr as *const u8,
                    x_ptr as *const ffi::Half,
                    y_ptr as *mut ffi::Half,
                    n,
                    k,
                    stream,
                )
                .result()
                .expect("q4k_gemv_cuda failed");
            }
        } else if x.seq_len <= 8 {
            // Small batch — batched GEMV is cheaper than allocating a bf16 tile.
            unsafe {
                ffi::q4k_gemv_batch_cuda(
                    qw_ptr as *const u8,
                    x_ptr as *const ffi::Half,
                    y_ptr as *mut ffi::Half,
                    x.seq_len as i32,
                    n,
                    k,
                    stream,
                )
                .result()
                .expect("q4k_gemv_batch_cuda failed");
            }
        } else {
            // Prefill path: dequant the whole weight into a BF16 workspace then cuBLAS GEMM.
            // Workspace is [n, k] bf16 — sized per-call, freed on scope exit.
            // Future optimisation: chunk over K to cap workspace to [n, chunk] and loop.
            let ws_elems = (weight.rows * weight.cols) as usize;
            let mut workspace: CudaSlice<bf16> = ctx
                .stream
                .alloc_zeros(ws_elems)
                .expect("alloc Q4_K dequant workspace");
            let (ws_ptr, _gws) = workspace.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::q4k_dequant_chunk_cuda(
                    qw_ptr as *const u8,
                    ws_ptr as *mut ffi::Half,
                    n,
                    k,
                    0,
                    k,
                    stream,
                )
                .result()
                .expect("q4k_dequant_chunk_cuda failed");
                ffi::gemm_cuda(
                    ws_ptr as *const ffi::Half,
                    x_ptr as *const ffi::Half,
                    y_ptr as *mut ffi::Half,
                    n,
                    x.seq_len as i32,
                    k,
                    stream,
                )
                .result()
                .expect("Q4_K prefill cuBLAS GEMM failed");
            }
        }
        return;
    }

    // ── Quantized weight dispatch (W2A16 / W4A16 / W8A16) ──
    if let (Some(qw), Some(qs)) = (&weight.qweight, &weight.qscales) {
        let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
        let (qs_ptr, _gqs) = qs.device_ptr(&ctx.stream);
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

        let is_decode = x.seq_len == 1;
        let n = weight.rows as i32;
        let k = weight.cols as i32;
        let gs = weight.group_size as i32;
        let stream = ctx.stream.cu_stream();

        unsafe {
            let res = match (is_decode, weight.quant_bits) {
                (true, 2) => ffi::w2a16_gemv_cuda(
                    qw_ptr as *const u8,
                    qs_ptr as *const ffi::Half,
                    x_ptr as *const ffi::Half,
                    y_ptr as *mut ffi::Half,
                    n,
                    k,
                    gs,
                    stream,
                ),
                (true, 4) => ffi::w4a16_gemv_cuda(
                    qw_ptr as *const u8,
                    qs_ptr as *const ffi::Half,
                    x_ptr as *const ffi::Half,
                    y_ptr as *mut ffi::Half,
                    n,
                    k,
                    gs,
                    stream,
                ),
                (true, _) => ffi::w8a16_gemv_cuda(
                    qw_ptr as *const i8,
                    qs_ptr as *const ffi::Half,
                    x_ptr as *const ffi::Half,
                    y_ptr as *mut ffi::Half,
                    n,
                    k,
                    gs,
                    stream,
                ),
                (false, 2) => ffi::w2a16_gemv_batch_cuda(
                    qw_ptr as *const u8,
                    qs_ptr as *const ffi::Half,
                    x_ptr as *const ffi::Half,
                    y_ptr as *mut ffi::Half,
                    x.seq_len as i32,
                    n,
                    k,
                    gs,
                    stream,
                ),
                (false, 4) => ffi::w4a16_gemv_batch_cuda(
                    qw_ptr as *const u8,
                    qs_ptr as *const ffi::Half,
                    x_ptr as *const ffi::Half,
                    y_ptr as *mut ffi::Half,
                    x.seq_len as i32,
                    n,
                    k,
                    gs,
                    stream,
                ),
                (false, _) => ffi::w8a16_gemv_batch_cuda(
                    qw_ptr as *const i8,
                    qs_ptr as *const ffi::Half,
                    x_ptr as *const ffi::Half,
                    y_ptr as *mut ffi::Half,
                    x.seq_len as i32,
                    n,
                    k,
                    gs,
                    stream,
                ),
            };
            res.result().expect("quantized GEMV/GEMM failed");
        }
        return;
    }

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        if x.seq_len == 1 {
            ffi::gemm_graphsafe_cuda(
                w_ptr as *const ffi::Half,
                x_ptr as *const ffi::Half,
                y_ptr as *mut ffi::Half,
                weight.rows as i32,
                1,
                weight.cols as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .expect("gemm_graphsafe_cuda failed");
        } else {
            ffi::gemm_cuda(
                w_ptr as *const ffi::Half,
                x_ptr as *const ffi::Half,
                y_ptr as *mut ffi::Half,
                weight.rows as i32,
                x.seq_len as i32,
                weight.cols as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .expect("gemm_cuda failed");
        }
    }
}
