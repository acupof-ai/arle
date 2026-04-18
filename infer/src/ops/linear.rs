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

use infer_cuda_kernels::ffi;
use infer_cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};

/// `DeviceMatrix::quant_bits` discriminators. Named so dispatch sites can
/// `match` on intent instead of hardcoding magic numbers.
///
/// Uniform group-scale formats use the "real" bit width (2/4/8). GGUF
/// superblock formats reserve two-digit sentinels distinct from those:
///   * `Q3K` (Q3_K_{S,M,L}) : 3-bit in 256-element superblocks, 110 bytes
///   * `Q4K` (Q4_K_{S,M})   : 4-bit in 256-element superblocks, 144 bytes
mod qbits {
    pub(super) const W2: usize = 2;
    pub(super) const W4: usize = 4;
    pub(super) const Q3K: usize = 33;
    pub(super) const Q4K: usize = 44;
    pub(super) const Q6K: usize = 66;
}

/// Additive LoRA GEMV: `y += B @ (A @ x)`.
///
/// The B matrix is expected to be pre-scaled at load time (see
/// `model::qwen3::lora::upload_as_bf16` — it folds `scale = alpha / r`
/// into B), so no runtime scalar multiply is needed here.
///
/// Shapes:
///   * `a` — `[rank, in_features]`  (LoRA A)
///   * `b` — `[out_features, rank]` (LoRA B, pre-scaled)
///   * `x` — `[in_features]`
///   * `y` — `[out_features]`  (accumulated, not overwritten)
///
/// Allocates two small temporaries (`tmp_a` size `rank`, `tmp_b` size
/// `out_features`); `rank` is typically 8–64 so the alloc cost is
/// negligible relative to the two GEMVs. Phase 2 will revisit if the
/// decode path shows churn overhead in practice.
pub fn apply_lora_gemv_add(
    ctx: &DeviceContext,
    a: &DeviceMatrix,
    b: &DeviceMatrix,
    x: &DeviceVec,
    y: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(a.cols, x.len, "lora A cols {} != x len {}", a.cols, x.len);
    assert_eq!(b.rows, y.len, "lora B rows {} != y len {}", b.rows, y.len);
    assert_eq!(
        a.rows, b.cols,
        "lora rank mismatch: A rows {} != B cols {}",
        a.rows, b.cols
    );

    let rank = a.rows;
    let mut tmp_a = DeviceVec::zeros(ctx, rank)?;
    gemv(ctx, a, x, &mut tmp_a)?;

    let mut tmp_b = DeviceVec::zeros(ctx, b.rows)?;
    gemv(ctx, b, &tmp_a, &mut tmp_b)?;

    let (tmp_ptr, _gt) = tmp_b.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = y.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::add_cuda(
            y_ptr as *const ffi::Half,
            tmp_ptr as *const ffi::Half,
            y_ptr as *mut ffi::Half,
            y.len as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Additive LoRA GEMM: `Y += B @ (A @ X)`, batched across `seq_len`.
///
/// Mirrors `apply_lora_gemv_add` for the prefill path. Shapes:
///   * `a` — `[rank, in_features]`
///   * `b` — `[out_features, rank]` (pre-scaled)
///   * `x` — `HiddenStates [in_features, seq_len]`
///   * `y` — `HiddenStates [out_features, seq_len]` (accumulated)
///
/// Allocates `tmp_a` of shape `[rank, seq_len]` and `tmp_b` of shape
/// `[out_features, seq_len]`.
pub fn apply_lora_gemm_add(
    ctx: &DeviceContext,
    a: &DeviceMatrix,
    b: &DeviceMatrix,
    x: &HiddenStates,
    y: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(
        a.cols, x.hidden_dim,
        "lora A cols {} != x hidden_dim {}",
        a.cols, x.hidden_dim
    );
    assert_eq!(
        b.rows, y.hidden_dim,
        "lora B rows {} != y hidden_dim {}",
        b.rows, y.hidden_dim
    );
    assert_eq!(
        a.rows, b.cols,
        "lora rank mismatch: A rows {} != B cols {}",
        a.rows, b.cols
    );
    assert_eq!(
        x.seq_len, y.seq_len,
        "lora gemm seq_len mismatch: x {} != y {}",
        x.seq_len, y.seq_len
    );

    let rank = a.rows;
    let mut tmp_a = HiddenStates::zeros(ctx, rank, x.seq_len)?;
    gemm_into(ctx, a, x, &mut tmp_a);

    let mut tmp_b = HiddenStates::zeros(ctx, b.rows, x.seq_len)?;
    gemm_into(ctx, b, &tmp_a, &mut tmp_b);

    let n = y.hidden_dim * y.seq_len;
    let (tmp_ptr, _gt) = tmp_b.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = y.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::add_cuda(
            y_ptr as *const ffi::Half,
            tmp_ptr as *const ffi::Half,
            y_ptr as *mut ffi::Half,
            n as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Matrix-vector multiplication: y = A @ x
/// A: (M, K) row-major, x: (K,), y: (M,)
/// Supports BF16, W8A16, W4A16, and W2A16 weights.
pub fn gemv(ctx: &DeviceContext, a: &DeviceMatrix, x: &DeviceVec, y: &mut DeviceVec) -> Result<()> {
    assert_eq!(a.cols, x.len, "A cols {} != x len {}", a.cols, x.len);
    assert_eq!(a.rows, y.len, "A rows {} != y len {}", a.rows, y.len);

    if a.is_quantized() {
        let qw = a
            .qweight
            .as_ref()
            .expect("quantized matrix missing qweight");
        let qs = a
            .qscales
            .as_ref()
            .expect("quantized matrix missing qscales");
        let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
        let (qs_ptr, _gqs) = qs.device_ptr(&ctx.stream);
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (y_ptr, _gy) = y.data.device_ptr_mut(&ctx.stream);
        let n = a.rows as i32;
        let k = a.cols as i32;
        let grp = a.group_size as i32;
        let stream = ctx.stream.cu_stream();
        let wptr = qw_ptr as *const u8;
        let xptr = x_ptr as *const ffi::Half;
        let yptr = y_ptr as *mut ffi::Half;
        let sptr = qs_ptr as *const ffi::Half;

        unsafe {
            use qbits::*;
            let res = match a.quant_bits {
                Q3K => ffi::q3k_gemv_cuda(wptr, xptr, yptr, n, k, stream),
                Q4K => ffi::q4k_gemv_cuda(wptr, xptr, yptr, n, k, stream),
                Q6K => ffi::q6k_gemv_cuda(wptr, xptr, yptr, n, k, stream),
                W2 => ffi::w2a16_gemv_cuda(wptr, sptr, xptr, yptr, n, k, grp, stream),
                W4 => ffi::w4a16_gemv_cuda(wptr, sptr, xptr, yptr, n, k, grp, stream),
                // Default: W8A16 (signed int8).
                _ => ffi::w8a16_gemv_cuda(qw_ptr as *const i8, sptr, xptr, yptr, n, k, grp, stream),
            };
            res.result()?;
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

/// Unfused decode-path MLP with optional LoRA adapters on any of gate/up/down.
///
/// Used when LoRA is active on one or more of gate_proj / up_proj / down_proj,
/// since the fused kernel has no LoRA hook. Numerically matches the quantized
/// fallback branch of `fused_mlp_into`:
///   * `act = silu(gate_proj(x)) * up_proj(x)`
///   * `out = down_proj(act)`
/// with LoRA adds applied right after their respective base GEMVs (before the
/// SiLU for gate/up, after the base GEMV for down).
///
/// `up_scratch` must be a caller-owned `DeviceVec` of length `intermediate_size`
/// (see `DecodeBuffers::mlp_up_scratch`).
pub fn mlp_decode_with_lora_into(
    ctx: &DeviceContext,
    x: &DeviceVec,
    gate_proj: &DeviceMatrix,
    up_proj: &DeviceMatrix,
    down_proj: &DeviceMatrix,
    lora_gate: Option<(&DeviceMatrix, &DeviceMatrix)>,
    lora_up: Option<(&DeviceMatrix, &DeviceMatrix)>,
    lora_down: Option<(&DeviceMatrix, &DeviceMatrix)>,
    act: &mut DeviceVec,
    up_scratch: &mut DeviceVec,
    out: &mut DeviceVec,
) -> Result<()> {
    let intermediate_size = gate_proj.rows;
    assert_eq!(
        up_scratch.len, intermediate_size,
        "up_scratch len {} != intermediate_size {}",
        up_scratch.len, intermediate_size
    );

    gemv(ctx, gate_proj, x, act)?;
    if let Some((a, b)) = lora_gate {
        apply_lora_gemv_add(ctx, a, b, x, act)?;
    }
    gemv(ctx, up_proj, x, up_scratch)?;
    if let Some((a, b)) = lora_up {
        apply_lora_gemv_add(ctx, a, b, x, up_scratch)?;
    }

    // silu(gate) * up → act (in-place)
    {
        let (act_ptr, _ga) = act.data.device_ptr_mut(&ctx.stream);
        let (up_ptr, _gu) = up_scratch.data.device_ptr(&ctx.stream);
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

    gemv(ctx, down_proj, act, out)?;
    if let Some((a, b)) = lora_down {
        apply_lora_gemv_add(ctx, a, b, act, out)?;
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

    // ── Quantized weight dispatch (Q3_K / Q4_K / W2/4/8 A16) ──
    //
    // Decode (B=1)          → single-row GEMV kernel for the matching bit width.
    // Small batch (2..=8)   → batched GEMV (cheap; avoids the dequant-to-tile cost).
    // Prefill (B>8)         → QxK: dequant the whole weight to a BF16 tile then
    //                         cuBLAS GEMM. W2/W4/W8: batched GEMV (cuBLAS-free
    //                         path keeps CUDA Graph compatibility).
    if weight.is_quantized() {
        let qw = weight
            .qweight
            .as_ref()
            .expect("quantized matrix missing qweight");
        let qs = weight
            .qscales
            .as_ref()
            .expect("quantized matrix missing qscales");
        let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
        let (qs_ptr, _gqs) = qs.device_ptr(&ctx.stream);
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);
        let n = weight.rows as i32;
        let k = weight.cols as i32;
        let gs = weight.group_size as i32;
        let b = x.seq_len as i32;
        let stream = ctx.stream.cu_stream();

        let wptr = qw_ptr as *const u8;
        let wptr_i8 = qw_ptr as *const i8;
        let xptr = x_ptr as *const ffi::Half;
        let yptr = y_ptr as *mut ffi::Half;
        let sptr = qs_ptr as *const ffi::Half;

        // QxK prefill path: dequant → cuBLAS GEMM. Branches out on big B.
        let is_qxk = matches!(weight.quant_bits, qbits::Q3K | qbits::Q4K | qbits::Q6K);
        if b > 8 && is_qxk {
            let ws_elems = weight.rows * weight.cols;
            let mut workspace: CudaSlice<bf16> = ctx
                .stream
                .alloc_zeros(ws_elems)
                .expect("alloc QxK dequant workspace");
            let (ws_ptr, _gws) = workspace.device_ptr_mut(&ctx.stream);
            let tile = ws_ptr as *mut ffi::Half;
            unsafe {
                let dq = match weight.quant_bits {
                    qbits::Q3K => ffi::q3k_dequant_chunk_cuda(wptr, tile, n, k, 0, k, stream),
                    qbits::Q4K => ffi::q4k_dequant_chunk_cuda(wptr, tile, n, k, 0, k, stream),
                    qbits::Q6K => ffi::q6k_dequant_chunk_cuda(wptr, tile, n, k, 0, k, stream),
                    _ => unreachable!(),
                };
                dq.result().expect("qxk_dequant_chunk_cuda failed");
                ffi::gemm_cuda(tile as *const ffi::Half, xptr, yptr, n, b, k, stream)
                    .result()
                    .expect("QxK prefill cuBLAS GEMM failed");
            }
            return;
        }

        // GEMV path for everything else (decode or batched).
        unsafe {
            use qbits::*;
            let res = match (b == 1, weight.quant_bits) {
                (true, Q3K) => ffi::q3k_gemv_cuda(wptr, xptr, yptr, n, k, stream),
                (true, Q4K) => ffi::q4k_gemv_cuda(wptr, xptr, yptr, n, k, stream),
                (true, Q6K) => ffi::q6k_gemv_cuda(wptr, xptr, yptr, n, k, stream),
                (true, W2) => ffi::w2a16_gemv_cuda(wptr, sptr, xptr, yptr, n, k, gs, stream),
                (true, W4) => ffi::w4a16_gemv_cuda(wptr, sptr, xptr, yptr, n, k, gs, stream),
                (true, _) => ffi::w8a16_gemv_cuda(wptr_i8, sptr, xptr, yptr, n, k, gs, stream),
                (false, Q3K) => ffi::q3k_gemv_batch_cuda(wptr, xptr, yptr, b, n, k, stream),
                (false, Q4K) => ffi::q4k_gemv_batch_cuda(wptr, xptr, yptr, b, n, k, stream),
                (false, Q6K) => ffi::q6k_gemv_batch_cuda(wptr, xptr, yptr, b, n, k, stream),
                (false, W2) => {
                    ffi::w2a16_gemv_batch_cuda(wptr, sptr, xptr, yptr, b, n, k, gs, stream)
                }
                (false, W4) => {
                    ffi::w4a16_gemv_batch_cuda(wptr, sptr, xptr, yptr, b, n, k, gs, stream)
                }
                (false, _) => {
                    ffi::w8a16_gemv_batch_cuda(wptr_i8, sptr, xptr, yptr, b, n, k, gs, stream)
                }
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
