//! Linear projection ops: GEMV (decode) and GEMM (prefill/batch).
//!
//! Dispatch priority for `gemv()` (single token, decode path):
//!   1. Packed quantized weights → matching fused-dequant GEMV kernel
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

use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cuda_kernels::tensor::WeightFormat;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LinearKernelPlan {
    Bf16Gemv,
    Bf16GraphsafeGemm,
    Bf16CublasGemm,
    W2A16Gemv,
    W4A16Gemv,
    W8A16Gemv,
    W2A16BatchGemv,
    W4A16BatchGemv,
    W8A16BatchGemv,
    Q3KGemv,
    Q4KGemv,
    Q6KGemv,
    Q3KBatchGemv,
    Q4KBatchGemv,
    Q6KBatchGemv,
    Q3KDequantCublasGemm,
    Q4KDequantCublasGemm,
    Q6KDequantCublasGemm,
    MarlinW4Gemm,
    TurboQuantGemv,
    TurboQuantDequantCublasGemm,
}

impl LinearKernelPlan {
    fn decode(weight: &DeviceMatrix) -> Self {
        match weight.weight_format() {
            WeightFormat::DenseBf16 => Self::Bf16Gemv,
            WeightFormat::W2A16 => Self::W2A16Gemv,
            WeightFormat::W4A16 => Self::W4A16Gemv,
            WeightFormat::W8A16 => Self::W8A16Gemv,
            WeightFormat::GgufQ3K => Self::Q3KGemv,
            WeightFormat::GgufQ4K => Self::Q4KGemv,
            WeightFormat::GgufQ6K => Self::Q6KGemv,
            WeightFormat::TurboQuant => Self::TurboQuantGemv,
        }
    }

    fn batched(weight: &DeviceMatrix, batch: usize) -> Self {
        if batch > 1 && marlin_prefill_aligned(weight).is_ok() {
            return Self::MarlinW4Gemm;
        }
        if batch > 1
            && weight.has_marlin()
            && let Err(reason) = marlin_prefill_aligned(weight)
        {
            log::trace!("Marlin W4 fallback: {reason}");
        }

        match (batch, weight.weight_format()) {
            (1, WeightFormat::DenseBf16) => Self::Bf16GraphsafeGemm,
            (_, WeightFormat::DenseBf16) => Self::Bf16CublasGemm,
            (1, _) => Self::decode(weight),
            (_, WeightFormat::W2A16) => Self::W2A16BatchGemv,
            (_, WeightFormat::W4A16) => Self::W4A16BatchGemv,
            (_, WeightFormat::W8A16) => Self::W8A16BatchGemv,
            (2..=8, WeightFormat::GgufQ3K) => Self::Q3KBatchGemv,
            (2..=8, WeightFormat::GgufQ4K) => Self::Q4KBatchGemv,
            (2..=8, WeightFormat::GgufQ6K) => Self::Q6KBatchGemv,
            (_, WeightFormat::GgufQ3K) => Self::Q3KDequantCublasGemm,
            (_, WeightFormat::GgufQ4K) => Self::Q4KDequantCublasGemm,
            (_, WeightFormat::GgufQ6K) => Self::Q6KDequantCublasGemm,
            (_, WeightFormat::TurboQuant) => Self::TurboQuantDequantCublasGemm,
        }
    }
}

fn marlin_prefill_aligned(weight: &DeviceMatrix) -> std::result::Result<(), &'static str> {
    if !weight.has_marlin() {
        return Err("missing marlin-packed side buffer");
    }
    if weight.weight_format() != WeightFormat::W4A16 {
        return Err("source format is not W4A16");
    }
    if !weight.cols.is_multiple_of(16) {
        return Err("K is not multiple of 16");
    }
    if !weight.rows.is_multiple_of(64) {
        return Err("N is not multiple of 64");
    }
    Ok(())
}

fn turboquant_params(weight: &DeviceMatrix) -> (i32, i32, i32, i32, i32, i32) {
    let n = weight.rows as i32;
    let k = weight.cols as i32;
    let group_size = weight.group_size as i32;
    let num_groups = (weight.cols / weight.group_size) as i32;
    let effective_bits = if weight.tq_bits == 3 {
        4
    } else {
        weight.tq_bits as usize
    };
    let packed_cols = (weight.cols * effective_bits).div_ceil(8) as i32;
    let bits = weight.tq_bits as i32;
    (n, k, group_size, packed_cols, num_groups, bits)
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
    lora_a: &DeviceMatrix,
    lora_b: &DeviceMatrix,
    input: &DeviceVec,
    output: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(
        lora_a.cols, input.len,
        "lora A cols {} != x len {}",
        lora_a.cols, input.len
    );
    assert_eq!(
        lora_b.rows, output.len,
        "lora B rows {} != y len {}",
        lora_b.rows, output.len
    );
    assert_eq!(
        lora_a.rows, lora_b.cols,
        "lora rank mismatch: A rows {} != B cols {}",
        lora_a.rows, lora_b.cols
    );

    let rank = lora_a.rows;
    let mut tmp_a = DeviceVec::zeros(ctx, rank)?;
    gemv(ctx, lora_a, input, &mut tmp_a)?;

    let mut tmp_b = DeviceVec::zeros(ctx, lora_b.rows)?;
    gemv(ctx, lora_b, &tmp_a, &mut tmp_b)?;

    let (tmp_ptr, _gt) = tmp_b.data.device_ptr(&ctx.stream);
    let (output_ptr, _gy) = output.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::add_cuda(
            output_ptr as *const ffi::Half,
            tmp_ptr as *const ffi::Half,
            output_ptr as *mut ffi::Half,
            output.len as i32,
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
    lora_a: &DeviceMatrix,
    lora_b: &DeviceMatrix,
    input: &HiddenStates,
    output: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(
        lora_a.cols, input.hidden_dim,
        "lora A cols {} != x hidden_dim {}",
        lora_a.cols, input.hidden_dim
    );
    assert_eq!(
        lora_b.rows, output.hidden_dim,
        "lora B rows {} != y hidden_dim {}",
        lora_b.rows, output.hidden_dim
    );
    assert_eq!(
        lora_a.rows, lora_b.cols,
        "lora rank mismatch: A rows {} != B cols {}",
        lora_a.rows, lora_b.cols
    );
    assert_eq!(
        input.seq_len, output.seq_len,
        "lora gemm seq_len mismatch: x {} != y {}",
        input.seq_len, output.seq_len
    );

    let rank = lora_a.rows;
    let mut tmp_a = HiddenStates::zeros(ctx, rank, input.seq_len)?;
    gemm_into(ctx, lora_a, input, &mut tmp_a);

    let mut tmp_b = HiddenStates::zeros(ctx, lora_b.rows, input.seq_len)?;
    gemm_into(ctx, lora_b, &tmp_a, &mut tmp_b);

    let total_elems = output.hidden_dim * output.seq_len;
    let (tmp_ptr, _gt) = tmp_b.data.device_ptr(&ctx.stream);
    let (output_ptr, _gy) = output.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::add_cuda(
            output_ptr as *const ffi::Half,
            tmp_ptr as *const ffi::Half,
            output_ptr as *mut ffi::Half,
            total_elems as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Matrix-vector multiplication: y = A @ x
/// A: (M, K) row-major, x: (K,), y: (M,)
/// Supports BF16, W8A16, W4A16, and W2A16 weights.
pub fn gemv(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    input: &DeviceVec,
    output: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(
        weight.cols, input.len,
        "A cols {} != x len {}",
        weight.cols, input.len
    );
    assert_eq!(
        weight.rows, output.len,
        "A rows {} != y len {}",
        weight.rows, output.len
    );

    let plan = LinearKernelPlan::decode(weight);
    if plan == LinearKernelPlan::Bf16Gemv {
        let (weight_ptr, _ga) = weight.data.device_ptr(&ctx.stream);
        let (input_ptr, _gx) = input.data.device_ptr(&ctx.stream);
        let (output_ptr, _gy) = output.data.device_ptr_mut(&ctx.stream);

        // Handwritten GEMV with BF16×4 vectorized loads.
        // cuBLAS GEMM(M,1,K) was tested but has higher dispatch overhead
        // on Ada (L4) for single-vector operations. The handwritten kernel
        // wins at B=1; cuBLAS wins at B≥2 (handled by gemm_into path).
        // On A100/H100 with tensor cores, cuBLAS may be faster — profile first.
        unsafe {
            ffi::gemv_cuda(
                weight_ptr as *const ffi::Half,
                input_ptr as *const ffi::Half,
                output_ptr as *mut ffi::Half,
                weight.rows as i32,
                weight.cols as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
        return Ok(());
    }

    if plan == LinearKernelPlan::TurboQuantGemv {
        let tq_p = weight
            .tq_packed
            .as_ref()
            .expect("TQ matrix missing packed weights");
        let tq_s = weight.tq_scales.as_ref().expect("TQ matrix missing scales");
        let tq_sg = weight.tq_signs.as_ref().expect("TQ matrix missing signs");
        let tq_c = weight
            .tq_centroids
            .as_ref()
            .expect("TQ matrix missing centroids");
        let (out_dim, in_dim, group_size, packed_cols, num_groups, bits) =
            turboquant_params(weight);
        let (tp_ptr, _g1) = tq_p.device_ptr(&ctx.stream);
        let (ts_ptr, _g2) = tq_s.device_ptr(&ctx.stream);
        let (tsg_ptr, _g3) = tq_sg.device_ptr(&ctx.stream);
        let (tc_ptr, _g4) = tq_c.device_ptr(&ctx.stream);
        let (input_ptr, _gx) = input.data.device_ptr(&ctx.stream);
        let (output_ptr, _gy) = output.data.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::turboquant_weight_gemv_cuda(
                tp_ptr as *const u8,
                ts_ptr as *const ffi::Half,
                tsg_ptr as *const i8,
                tc_ptr as *const f32,
                input_ptr as *const ffi::Half,
                output_ptr as *mut ffi::Half,
                out_dim,
                in_dim,
                group_size,
                packed_cols,
                num_groups,
                bits,
                ctx.stream.cu_stream(),
            );
        }
        return Ok(());
    }

    if let Some(qw) = weight.qweight.as_ref() {
        let qs = weight
            .qscales
            .as_ref()
            .expect("quantized matrix missing qscales");
        let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
        let (qs_ptr, _gqs) = qs.device_ptr(&ctx.stream);
        let (input_ptr, _gx) = input.data.device_ptr(&ctx.stream);
        let (output_ptr, _gy) = output.data.device_ptr_mut(&ctx.stream);
        let out_dim = weight.rows as i32;
        let in_dim = weight.cols as i32;
        let group_size = weight.group_size as i32;
        let stream = ctx.stream.cu_stream();
        let wptr = qw_ptr as *const u8;
        let xptr = input_ptr as *const ffi::Half;
        let yptr = output_ptr as *mut ffi::Half;
        let sptr = qs_ptr as *const ffi::Half;

        unsafe {
            let res = match plan {
                LinearKernelPlan::Q3KGemv => {
                    ffi::q3k_gemv_cuda(wptr, xptr, yptr, out_dim, in_dim, stream)
                }
                LinearKernelPlan::Q4KGemv => {
                    ffi::q4k_gemv_cuda(wptr, xptr, yptr, out_dim, in_dim, stream)
                }
                LinearKernelPlan::Q6KGemv => {
                    ffi::q6k_gemv_cuda(wptr, xptr, yptr, out_dim, in_dim, stream)
                }
                LinearKernelPlan::W2A16Gemv => ffi::w2a16_gemv_cuda(
                    wptr, sptr, xptr, yptr, out_dim, in_dim, group_size, stream,
                ),
                LinearKernelPlan::W4A16Gemv => ffi::w4a16_gemv_cuda(
                    wptr, sptr, xptr, yptr, out_dim, in_dim, group_size, stream,
                ),
                LinearKernelPlan::W8A16Gemv => ffi::w8a16_gemv_cuda(
                    qw_ptr as *const i8,
                    sptr,
                    xptr,
                    yptr,
                    out_dim,
                    in_dim,
                    group_size,
                    stream,
                ),
                _ => unreachable!("unexpected decode linear plan {plan:?}"),
            };
            res.result()?;
        }
        return Ok(());
    }

    unreachable!("linear decode plan {plan:?} has no matching weight storage")
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
///
/// LoRA adds applied right after their respective base GEMVs (before the
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

/// Graph-safe BF16 GEMM for captured batched prefill kernels.
///
/// This path is intentionally narrow: it only accepts plain BF16 weights and
/// always uses the workspace-free cuBLAS handle so CUDA Graph capture can span
/// multi-token prefill projections. Non-BF16 formats stay on `gemm_into`.
pub(crate) fn gemm_graphsafe_batched_into(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
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
    anyhow::ensure!(
        weight.is_dense_bf16(),
        "graph-safe batched GEMM requires plain BF16 weights"
    );

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::gemm_graphsafe_cuda(
            w_ptr as *const ffi::Half,
            x_ptr as *const ffi::Half,
            y_ptr as *mut ffi::Half,
            weight.rows as i32,
            x.seq_len as i32,
            weight.cols as i32,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow::anyhow!("gemm_graphsafe_cuda failed: {e}"))?;
    }

    Ok(())
}

fn run_marlin_w4_gemm(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) {
    let mp = weight.marlin_packed.as_ref().unwrap();
    let ms = weight.marlin_scales.as_ref().unwrap();
    let m = x.seq_len;
    let n = weight.rows;
    let k = weight.cols;

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

    let mut y_fp16: CudaSlice<u16> = ctx.stream.alloc_zeros(m * n).expect("alloc y_fp16");
    let sms = ctx.sm_count() as i32;
    let ws_size = unsafe { ffi::marlin_workspace_size(n as i32, sms) };
    let ws_elems = ws_size.div_ceil(4);
    let mut workspace: CudaSlice<i32> = ctx
        .stream
        .alloc_zeros(ws_elems)
        .expect("alloc marlin workspace");

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
                ms_ptr as *mut ffi::Half,
                m as i32,
                n as i32,
                k as i32,
                ws_ptr as *mut i32,
                weight.group_size as i32,
                0,
                ctx.stream.cu_stream(),
                -1,
                -1,
                sms,
                16,
            )
        };
        assert_eq!(ret, 0, "marlin_gemm_cuda failed with code {ret}");
    }

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
}

fn run_turboquant_linear(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
    plan: LinearKernelPlan,
) {
    let tq_p = weight.tq_packed.as_ref().unwrap();
    let tq_s = weight.tq_scales.as_ref().unwrap();
    let tq_sg = weight.tq_signs.as_ref().unwrap();
    let tq_c = weight.tq_centroids.as_ref().unwrap();
    let (n, k, group_size, packed_cols, num_groups, bits) = turboquant_params(weight);
    let stream = ctx.stream.cu_stream();

    let (tp_ptr, _g1) = tq_p.device_ptr(&ctx.stream);
    let (ts_ptr, _g2) = tq_s.device_ptr(&ctx.stream);
    let (tsg_ptr, _g3) = tq_sg.device_ptr(&ctx.stream);
    let (tc_ptr, _g4) = tq_c.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    match plan {
        LinearKernelPlan::TurboQuantGemv => unsafe {
            ffi::turboquant_weight_gemv_cuda(
                tp_ptr as *const u8,
                ts_ptr as *const ffi::Half,
                tsg_ptr as *const i8,
                tc_ptr as *const f32,
                x_ptr as *const ffi::Half,
                y_ptr as *mut ffi::Half,
                n,
                k,
                group_size,
                packed_cols,
                num_groups,
                bits,
                stream,
            );
        },
        LinearKernelPlan::TurboQuantDequantCublasGemm => {
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
                    group_size,
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
        _ => unreachable!("unexpected TurboQuant linear plan {plan:?}"),
    }
}

fn run_qweight_linear(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
    plan: LinearKernelPlan,
) {
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
    let group_size = weight.group_size as i32;
    let batch = x.seq_len as i32;
    let stream = ctx.stream.cu_stream();

    let wptr = qw_ptr as *const u8;
    let wptr_i8 = qw_ptr as *const i8;
    let xptr = x_ptr as *const ffi::Half;
    let yptr = y_ptr as *mut ffi::Half;
    let sptr = qs_ptr as *const ffi::Half;

    unsafe {
        let res = match plan {
            LinearKernelPlan::Q3KDequantCublasGemm
            | LinearKernelPlan::Q4KDequantCublasGemm
            | LinearKernelPlan::Q6KDequantCublasGemm => {
                let ws_elems = weight.rows * weight.cols;
                let mut workspace: CudaSlice<bf16> = ctx
                    .stream
                    .alloc_zeros(ws_elems)
                    .expect("alloc QxK dequant workspace");
                let (ws_ptr, _gws) = workspace.device_ptr_mut(&ctx.stream);
                let tile = ws_ptr as *mut ffi::Half;
                let dq = match plan {
                    LinearKernelPlan::Q3KDequantCublasGemm => {
                        ffi::q3k_dequant_chunk_cuda(wptr, tile, n, k, 0, k, stream)
                    }
                    LinearKernelPlan::Q4KDequantCublasGemm => {
                        ffi::q4k_dequant_chunk_cuda(wptr, tile, n, k, 0, k, stream)
                    }
                    LinearKernelPlan::Q6KDequantCublasGemm => {
                        ffi::q6k_dequant_chunk_cuda(wptr, tile, n, k, 0, k, stream)
                    }
                    _ => unreachable!(),
                };
                dq.result().expect("qxk_dequant_chunk_cuda failed");
                ffi::gemm_cuda(tile.cast_const(), xptr, yptr, n, batch, k, stream)
            }
            LinearKernelPlan::Q3KGemv => ffi::q3k_gemv_cuda(wptr, xptr, yptr, n, k, stream),
            LinearKernelPlan::Q4KGemv => ffi::q4k_gemv_cuda(wptr, xptr, yptr, n, k, stream),
            LinearKernelPlan::Q6KGemv => ffi::q6k_gemv_cuda(wptr, xptr, yptr, n, k, stream),
            LinearKernelPlan::W2A16Gemv => {
                ffi::w2a16_gemv_cuda(wptr, sptr, xptr, yptr, n, k, group_size, stream)
            }
            LinearKernelPlan::W4A16Gemv => {
                ffi::w4a16_gemv_cuda(wptr, sptr, xptr, yptr, n, k, group_size, stream)
            }
            LinearKernelPlan::W8A16Gemv => {
                ffi::w8a16_gemv_cuda(wptr_i8, sptr, xptr, yptr, n, k, group_size, stream)
            }
            LinearKernelPlan::Q3KBatchGemv => {
                ffi::q3k_gemv_batch_cuda(wptr, xptr, yptr, batch, n, k, stream)
            }
            LinearKernelPlan::Q4KBatchGemv => {
                ffi::q4k_gemv_batch_cuda(wptr, xptr, yptr, batch, n, k, stream)
            }
            LinearKernelPlan::Q6KBatchGemv => {
                ffi::q6k_gemv_batch_cuda(wptr, xptr, yptr, batch, n, k, stream)
            }
            LinearKernelPlan::W2A16BatchGemv => {
                ffi::w2a16_gemv_batch_cuda(wptr, sptr, xptr, yptr, batch, n, k, group_size, stream)
            }
            LinearKernelPlan::W4A16BatchGemv => {
                ffi::w4a16_gemv_batch_cuda(wptr, sptr, xptr, yptr, batch, n, k, group_size, stream)
            }
            LinearKernelPlan::W8A16BatchGemv => ffi::w8a16_gemv_batch_cuda(
                wptr_i8, sptr, xptr, yptr, batch, n, k, group_size, stream,
            ),
            _ => unreachable!("unexpected qweight linear plan {plan:?}"),
        };
        res.result().expect("quantized linear kernel failed");
    }
}

fn run_bf16_linear(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
    plan: LinearKernelPlan,
) {
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        match plan {
            LinearKernelPlan::Bf16GraphsafeGemm => ffi::gemm_graphsafe_cuda(
                w_ptr as *const ffi::Half,
                x_ptr as *const ffi::Half,
                y_ptr as *mut ffi::Half,
                weight.rows as i32,
                1,
                weight.cols as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .expect("gemm_graphsafe_cuda failed"),
            LinearKernelPlan::Bf16CublasGemm => ffi::gemm_cuda(
                w_ptr as *const ffi::Half,
                x_ptr as *const ffi::Half,
                y_ptr as *mut ffi::Half,
                weight.rows as i32,
                x.seq_len as i32,
                weight.cols as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .expect("gemm_cuda failed"),
            _ => unreachable!("unexpected BF16 linear plan {plan:?}"),
        }
    }
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

    let plan = LinearKernelPlan::batched(weight, x.seq_len);
    match plan {
        LinearKernelPlan::MarlinW4Gemm => run_marlin_w4_gemm(ctx, weight, x, out),
        LinearKernelPlan::TurboQuantGemv | LinearKernelPlan::TurboQuantDequantCublasGemm => {
            run_turboquant_linear(ctx, weight, x, out, plan);
        }
        LinearKernelPlan::Bf16GraphsafeGemm | LinearKernelPlan::Bf16CublasGemm => {
            run_bf16_linear(ctx, weight, x, out, plan);
        }
        LinearKernelPlan::Bf16Gemv => unreachable!("batched linear never selects BF16 GEMV"),
        _ => run_qweight_linear(ctx, weight, x, out, plan),
    }
}
