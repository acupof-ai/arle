// All unsafe blocks in these tests invoke CUDA FFI with device pointers
// obtained from the CudaSlice/CudaVec allocator, whose validity is guarded
// by the SyncOnDrop handles held in the same scope.
#![allow(clippy::undocumented_unsafe_blocks)]

use super::*;
use crate::tensor::{DeviceContext, DeviceVec};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

#[test]
fn int8_row_quantization_scales_match_absmax() {
    let ctx = DeviceContext::new().expect("failed to create CUDA context");
    let rows = 2usize;
    let cols = 513usize;
    let mut input_host = vec![bf16::ZERO; rows * cols];
    for col in 0..cols {
        let value = if col == 257 {
            -2.0
        } else {
            ((col % 17) as f32 - 8.0) * 0.03125
        };
        input_host[cols + col] = bf16::from_f32(value);
    }

    let input = ctx.stream.clone_htod(&input_host).expect("input H2D");
    let mut output = ctx
        .stream
        .alloc_zeros::<i8>(rows * cols)
        .expect("output alloc");
    let mut scales = ctx.stream.alloc_zeros::<f32>(rows).expect("scales alloc");
    {
        let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
        let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
        let (scales_ptr, _scales_guard) = scales.device_ptr_mut(&ctx.stream);

        unsafe {
            quantize_bf16_rows_to_int8_cuda(
                input_ptr as *const Half,
                output_ptr as *mut i8,
                scales_ptr as *mut f32,
                rows as i32,
                cols as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .expect("int8 row quantize");
        }
    }
    ctx.sync().expect("sync int8 row quantize");

    let got_scales = ctx.stream.clone_dtoh(&scales).expect("scales D2H");
    assert_eq!(got_scales[0], 1.0);
    assert!(
        (got_scales[1] - (2.0 / 127.0)).abs() < 1.0e-7,
        "nonzero row scale mismatch: got {}, expected {}",
        got_scales[1],
        2.0 / 127.0
    );

    let got_output = ctx.stream.clone_dtoh(&output).expect("output D2H");
    assert!(
        got_output[..cols].iter().all(|&byte| byte == 0),
        "zero row should quantize to all-zero int8 values"
    );
    assert_eq!(got_output[cols + 257], -127);
}

#[test]
fn fp8_row_quantization_scales_match_absmax() {
    let ctx = DeviceContext::new().expect("failed to create CUDA context");
    let rows = 2usize;
    let cols = 513usize;
    let mut input_host = vec![bf16::ZERO; rows * cols];
    for col in 0..cols {
        let value = if col == 257 {
            -2.0
        } else {
            ((col % 17) as f32 - 8.0) * 0.03125
        };
        input_host[cols + col] = bf16::from_f32(value);
    }

    let input = ctx.stream.clone_htod(&input_host).expect("input H2D");
    let mut output = ctx
        .stream
        .alloc_zeros::<u8>(rows * cols)
        .expect("output alloc");
    let mut scales = ctx.stream.alloc_zeros::<f32>(rows).expect("scales alloc");
    {
        let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
        let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
        let (scales_ptr, _scales_guard) = scales.device_ptr_mut(&ctx.stream);

        unsafe {
            quantize_bf16_rows_to_fp8_e4m3_cuda(
                input_ptr as *const Half,
                output_ptr as *mut u8,
                scales_ptr as *mut f32,
                rows as i32,
                cols as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .expect("fp8 row quantize");
        }
    }
    ctx.sync().expect("sync fp8 row quantize");

    let got_scales = ctx.stream.clone_dtoh(&scales).expect("scales D2H");
    assert_eq!(got_scales[0], 1.0);
    assert!(
        (got_scales[1] - (2.0 / 448.0)).abs() < 1.0e-7,
        "nonzero row scale mismatch: got {}, expected {}",
        got_scales[1],
        2.0 / 448.0
    );

    let got_output = ctx.stream.clone_dtoh(&output).expect("output D2H");
    assert!(
        got_output[..cols].iter().all(|&byte| byte == 0),
        "zero row should quantize to all-zero fp8 bytes"
    );
    assert_eq!(
        got_output[cols + 257],
        0xfe,
        "largest-magnitude negative value should quantize to E4M3 negative max"
    );
}

fn assert_bf16_close(got: &[bf16], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len());
    for (idx, (got, expected)) in got.iter().zip(expected).enumerate() {
        let delta = (got.to_f32() - expected).abs();
        assert!(
            delta <= tol,
            "idx {idx}: got {} expected {} delta {} > {}",
            got.to_f32(),
            expected,
            delta,
            tol
        );
    }
}

fn decode_e4m3(byte: u8) -> f32 {
    match byte {
        0x00 => 0.0,
        0x30 => 0.5,
        0x38 => 1.0,
        0xb8 => -1.0,
        0x40 => 2.0,
        0xc0 => -2.0,
        other => panic!("test byte 0x{other:02x} missing e4m3 reference"),
    }
}

fn decode_e2m1(nibble: u8) -> f32 {
    const LUT: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    LUT[(nibble & 0x0f) as usize]
}

fn single_ptr_table<T>(ctx: &DeviceContext, slice: &CudaSlice<T>) -> CudaSlice<u64> {
    let (base, _guard) = slice.device_ptr(&ctx.stream);
    ctx.stream.clone_htod(&[base]).expect("pointer table H2D")
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[test]
fn fp8_block_scaled_gemv_matches_reference() {
    let ctx = DeviceContext::new().expect("failed to create CUDA context");
    let b = 2usize;
    let n = 3usize;
    let k = 64usize;
    let block_m = 2usize;
    let block_k = 16usize;
    let scale_rows = 2usize;
    let scale_cols = 4usize;
    let weight_values = [0x38, 0x40, 0xb8, 0x30, 0xc0, 0x00];
    let mut weight = vec![0u8; n * k];
    for row in 0..n {
        for col in 0..k {
            weight[row * k + col] = weight_values[(row * 7 + col * 3) % weight_values.len()];
        }
    }
    let scales = [0.5f32, 2.0, 1.25, 0.25, 1.0, 0.75, 1.5, 0.125];
    let mut input_host = vec![bf16::ZERO; b * k];
    for batch in 0..b {
        for col in 0..k {
            let raw = ((batch * 11 + col * 5) % 17) as f32 - 8.0;
            input_host[batch * k + col] = bf16::from_f32(raw * 0.125);
        }
    }
    let mut expected = vec![0.0f32; b * n];
    for batch in 0..b {
        for row in 0..n {
            let mut sum = 0.0f32;
            for col in 0..k {
                let scale_row = (row / block_m).min(scale_rows - 1);
                let scale_col = (col / block_k).min(scale_cols - 1);
                let scale = scales[scale_row * scale_cols + scale_col];
                sum += decode_e4m3(weight[row * k + col])
                    * scale
                    * input_host[batch * k + col].to_f32();
            }
            expected[batch * n + row] = sum;
        }
    }

    let weight_dev = ctx.stream.clone_htod(&weight).expect("weight H2D");
    let scales_dev = ctx.stream.clone_htod(&scales).expect("scales H2D");
    let input_dev = ctx.stream.clone_htod(&input_host).expect("input H2D");
    let mut output_dev = ctx.stream.alloc_zeros::<bf16>(b * n).expect("output alloc");
    {
        let (weight_ptr, _weight_guard) = weight_dev.device_ptr(&ctx.stream);
        let (scales_ptr, _scales_guard) = scales_dev.device_ptr(&ctx.stream);
        let (input_ptr, _input_guard) = input_dev.device_ptr(&ctx.stream);
        let (output_ptr, _output_guard) = output_dev.device_ptr_mut(&ctx.stream);
        unsafe {
            gemv_fp8_block_scaled_batch_cuda(
                weight_ptr as *const u8,
                scales_ptr as *const f32,
                input_ptr as *const Half,
                output_ptr as *mut Half,
                b as i32,
                n as i32,
                k as i32,
                scale_rows as i32,
                scale_cols as i32,
                block_m as i32,
                block_k as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .expect("fp8 block GEMV");
        }
    }
    ctx.sync().expect("sync fp8 block GEMV");

    let got = ctx.stream.clone_dtoh(&output_dev).expect("output D2H");
    assert_bf16_close(&got, &expected, 0.04);
}

#[test]
fn dsv4_fp8_grouped_decode_matches_reference() {
    let ctx = DeviceContext::new().expect("failed to create CUDA context");
    let routes = 2usize;
    let hidden = 16usize;
    let intermediate = 16usize;
    let scale_cols = 1usize;
    let fp8_values = [0x38, 0x40, 0xb8, 0x30, 0xc0, 0x00];

    let mut input_host = vec![bf16::ZERO; routes * hidden];
    for route in 0..routes {
        for col in 0..hidden {
            let raw = ((route * 7 + col * 3) % 11) as f32 - 5.0;
            input_host[route * hidden + col] = bf16::from_f32(raw * 0.125);
        }
    }

    let mut gate = vec![0u8; intermediate * hidden];
    let mut up = vec![0u8; intermediate * hidden];
    let mut down = vec![0u8; hidden * intermediate];
    for row in 0..intermediate {
        for col in 0..hidden {
            gate[row * hidden + col] = fp8_values[(row * 5 + col * 3) % fp8_values.len()];
            up[row * hidden + col] = fp8_values[(row * 7 + col * 2 + 1) % fp8_values.len()];
            down[col * intermediate + row] =
                fp8_values[(col * 11 + row * 3 + 2) % fp8_values.len()];
        }
    }

    let gate_scale = [0.5f32];
    let up_scale = [0.25f32];
    let down_scale = [1.5f32];
    let mut expected_act = vec![0.0f32; routes * intermediate];
    for route in 0..routes {
        for row in 0..intermediate {
            let mut gate_acc = 0.0f32;
            let mut up_acc = 0.0f32;
            for col in 0..hidden {
                let x = input_host[route * hidden + col].to_f32();
                gate_acc += decode_e4m3(gate[row * hidden + col]) * gate_scale[0] * x;
                up_acc += decode_e4m3(up[row * hidden + col]) * up_scale[0] * x;
            }
            expected_act[route * intermediate + row] =
                bf16::from_f32(silu(gate_acc) * up_acc).to_f32();
        }
    }
    let mut expected_down = vec![0.0f32; routes * hidden];
    for route in 0..routes {
        for row in 0..hidden {
            let mut acc = 0.0f32;
            for col in 0..intermediate {
                acc += decode_e4m3(down[row * intermediate + col])
                    * down_scale[0]
                    * expected_act[route * intermediate + col];
            }
            expected_down[route * hidden + row] = acc;
        }
    }

    let input_dev = ctx.stream.clone_htod(&input_host).expect("input H2D");
    let gate_dev = ctx.stream.clone_htod(&gate).expect("gate H2D");
    let up_dev = ctx.stream.clone_htod(&up).expect("up H2D");
    let down_dev = ctx.stream.clone_htod(&down).expect("down H2D");
    let gate_scale_dev = ctx.stream.clone_htod(&gate_scale).expect("gate scale H2D");
    let up_scale_dev = ctx.stream.clone_htod(&up_scale).expect("up scale H2D");
    let down_scale_dev = ctx.stream.clone_htod(&down_scale).expect("down scale H2D");
    let gate_ptrs = single_ptr_table(&ctx, &gate_dev);
    let up_ptrs = single_ptr_table(&ctx, &up_dev);
    let down_ptrs = single_ptr_table(&ctx, &down_dev);
    let gate_scale_ptrs = single_ptr_table(&ctx, &gate_scale_dev);
    let up_scale_ptrs = single_ptr_table(&ctx, &up_scale_dev);
    let down_scale_ptrs = single_ptr_table(&ctx, &down_scale_dev);
    let offsets = ctx.stream.clone_htod(&[0i32]).expect("offsets H2D");
    let counts = ctx.stream.clone_htod(&[routes as i32]).expect("counts H2D");
    let expert_indices = ctx.stream.clone_htod(&[0i32]).expect("expert indices H2D");
    let mut act_dev = ctx
        .stream
        .alloc_zeros::<bf16>(routes * intermediate)
        .expect("act alloc");
    let mut out_dev = ctx
        .stream
        .alloc_zeros::<bf16>(routes * hidden)
        .expect("out alloc");

    {
        let (gate_ptrs, _gpg) = gate_ptrs.device_ptr(&ctx.stream);
        let (gate_scale_ptrs, _gsg) = gate_scale_ptrs.device_ptr(&ctx.stream);
        let (up_ptrs, _upg) = up_ptrs.device_ptr(&ctx.stream);
        let (up_scale_ptrs, _usg) = up_scale_ptrs.device_ptr(&ctx.stream);
        let (input_ptr, _ig) = input_dev.device_ptr(&ctx.stream);
        let (act_ptr, _ag) = act_dev.device_ptr_mut(&ctx.stream);
        let (offsets_ptr, _og) = offsets.device_ptr(&ctx.stream);
        let (counts_ptr, _cg) = counts.device_ptr(&ctx.stream);
        let (expert_indices_ptr, _eg) = expert_indices.device_ptr(&ctx.stream);
        unsafe {
            dsv4_fp8_grouped_swiglu_decode_cuda(
                gate_ptrs as *const u64,
                gate_scale_ptrs as *const u64,
                up_ptrs as *const u64,
                up_scale_ptrs as *const u64,
                input_ptr as *const Half,
                act_ptr as *mut Half,
                offsets_ptr as *const i32,
                counts_ptr as *const i32,
                expert_indices_ptr as *const i32,
                1,
                routes as i32,
                intermediate as i32,
                hidden as i32,
                scale_cols as i32,
                f32::INFINITY,
                ctx.stream.cu_stream(),
            )
            .result()
            .expect("fp8 decode swiglu");
        }
    }
    ctx.sync().expect("sync fp8 decode swiglu");
    let got_act = ctx.stream.clone_dtoh(&act_dev).expect("act D2H");
    assert_bf16_close(&got_act, &expected_act, 0.05);

    {
        let (down_ptrs, _dpg) = down_ptrs.device_ptr(&ctx.stream);
        let (down_scale_ptrs, _dsg) = down_scale_ptrs.device_ptr(&ctx.stream);
        let (act_ptr, _ag) = act_dev.device_ptr(&ctx.stream);
        let (out_ptr, _outg) = out_dev.device_ptr_mut(&ctx.stream);
        let (offsets_ptr, _og) = offsets.device_ptr(&ctx.stream);
        let (counts_ptr, _cg) = counts.device_ptr(&ctx.stream);
        let (expert_indices_ptr, _eg) = expert_indices.device_ptr(&ctx.stream);
        unsafe {
            dsv4_fp8_grouped_down_decode_cuda(
                down_ptrs as *const u64,
                down_scale_ptrs as *const u64,
                act_ptr as *const Half,
                out_ptr as *mut Half,
                offsets_ptr as *const i32,
                counts_ptr as *const i32,
                expert_indices_ptr as *const i32,
                1,
                routes as i32,
                hidden as i32,
                intermediate as i32,
                scale_cols as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .expect("fp8 decode down");
        }
    }
    ctx.sync().expect("sync fp8 decode down");
    let got_down = ctx.stream.clone_dtoh(&out_dev).expect("down D2H");
    assert_bf16_close(&got_down, &expected_down, 0.08);
}

#[test]
fn fp4_group_gemv_matches_reference() {
    let ctx = DeviceContext::new().expect("failed to create CUDA context");
    let b = 2usize;
    let n = 2usize;
    let k = 64usize;
    let group_size = 16usize;
    let scale_cols = 4usize;
    let nibbles = [0x2, 0x4, 0x0a, 0x1, 0x6, 0x9, 0x3, 0x0d];
    let mut weight = vec![0u8; n * k / 2];
    for row in 0..n {
        for pair in 0..(k / 2) {
            let lo = nibbles[(row * 5 + pair * 2) % nibbles.len()];
            let hi = nibbles[(row * 5 + pair * 2 + 1) % nibbles.len()];
            weight[row * (k / 2) + pair] = lo | (hi << 4);
        }
    }
    let group_scales = [
        0x38, 0x30, 0x40, 0x38, // [1, 0.5, 2, 1]
        0x40, 0x38, 0x30, 0x40, // [2, 1, 0.5, 2]
    ];
    let global = [0.25f32];
    let mut input_host = vec![bf16::ZERO; b * k];
    for batch in 0..b {
        for col in 0..k {
            let raw = ((batch * 13 + col * 7) % 19) as f32 - 9.0;
            input_host[batch * k + col] = bf16::from_f32(raw * 0.125);
        }
    }
    let mut expected = vec![0.0f32; b * n];
    for batch in 0..b {
        for row in 0..n {
            let mut sum = 0.0f32;
            for col in 0..k {
                let packed = weight[row * (k / 2) + (col / 2)];
                let nibble = if col & 1 == 0 {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                let scale =
                    decode_e4m3(group_scales[row * scale_cols + col / group_size]) * global[0];
                sum += decode_e2m1(nibble) * scale * input_host[batch * k + col].to_f32();
            }
            expected[batch * n + row] = sum;
        }
    }

    let weight_dev = ctx.stream.clone_htod(&weight).expect("weight H2D");
    let scales_dev = ctx.stream.clone_htod(&group_scales).expect("scales H2D");
    let global_dev = ctx.stream.clone_htod(&global).expect("global H2D");
    let input_dev = ctx.stream.clone_htod(&input_host).expect("input H2D");
    let mut output_dev = ctx.stream.alloc_zeros::<bf16>(b * n).expect("output alloc");
    {
        let (weight_ptr, _weight_guard) = weight_dev.device_ptr(&ctx.stream);
        let (scales_ptr, _scales_guard) = scales_dev.device_ptr(&ctx.stream);
        let (global_ptr, _global_guard) = global_dev.device_ptr(&ctx.stream);
        let (input_ptr, _input_guard) = input_dev.device_ptr(&ctx.stream);
        let (output_ptr, _output_guard) = output_dev.device_ptr_mut(&ctx.stream);
        unsafe {
            gemv_fp4_e2m1_group_batch_cuda(
                weight_ptr as *const u8,
                scales_ptr as *const u8,
                global_ptr as *const f32,
                input_ptr as *const Half,
                output_ptr as *mut Half,
                b as i32,
                n as i32,
                k as i32,
                group_size as i32,
                scale_cols as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .expect("fp4 group GEMV");
        }
    }
    ctx.sync().expect("sync fp4 group GEMV");

    let got = ctx.stream.clone_dtoh(&output_dev).expect("output D2H");
    assert_bf16_close(&got, &expected, 0.04);
}

// Pre-Hopper dense FP8 fallback parity: the dequant→BF16 cuBLAS GEMM path
// (`dequantize_fp8_block_scaled_to_bf16_cuda` + `gemm_cuda`, used by infer-cuda's
// `try_fp8_dequant_bf16_gemm_batch` on sm < 9) vs the scalar block-scaled GEMV
// reference (`gemv_fp8_block_scaled_batch_cuda`), on the SAME fp8 block-scaled
// weights + input. Verifies TWO things on a pre-Hopper Ampere/Ada box (sm_80-89):
//   1. `gemm_cuda` (bf16 cuBLASLt) actually runs — settles the open hypothesis
//      that CUBLAS_GEMM_DEFAULT_TENSOR_OP bf16 might be NOT_SUPPORTED below sm_90
//      (the `.expect` on the gemm_cuda result panics if it errors).
//   2. dequant→BF16→GEMM is numerically equivalent to the scalar GEMV.
//
// Layout: gemm_cuda computes Y[M,N] col-major = W[M,K] row-major @ X[K,N]
// col-major. With M=n, N=b, K=k the col-major X[k,b] / Y[n,b] are byte-identical
// in memory to the scalar path's row-major input[b,k] / output[b,n]
// (index k_idx + batch*k ≡ batch*k + k_idx), so the same buffers drive both
// paths — an exact A/B. Mirrors the production call (M=weight.rows, N=seq,
// K=weight.cols). b=1024 matches the QWEN_FP8_DEEPGEMM_DENSE_MIN_M prefill gate.
//
// Run on the CUDA box (L4 / A100, sm_80-89):
//   CUDA_HOME=/usr/local/cuda cargo test -p cuda-kernels --release \
//     --features cuda fp8_dequant_bf16_gemm_matches_scalar -- --nocapture
#[test]
fn fp8_dequant_bf16_gemm_matches_scalar() {
    let ctx = DeviceContext::new().expect("failed to create CUDA context");
    let b = 1024usize; // prefill shape (>= QWEN_FP8_DEEPGEMM_DENSE_MIN_M)
    let n = 256usize; // weight rows / out features (N % 8 == 0, % 128 == 0)
    let k = 256usize; // weight cols / contraction (= 2 × block_k)
    let block_m = 128usize; // canonical Qwen FP8 quant block
    let block_k = 128usize;
    let scale_rows = n.div_ceil(block_m); // = 2
    let scale_cols = k.div_ceil(block_k); // = 2

    // Finite, in-range e4m3 weight bytes (avoid NaN 0x7f/0xff).
    let fp8_pool: [u8; 12] = [
        0x38, 0x40, 0x30, 0x48, 0x34, 0x3c, // +1, +2, +0.5, +3, +0.75, +1.5
        0xb8, 0xc0, 0xb0, 0xc8, 0xb4, 0xbc, // negatives of the above
    ];
    let mut weight = vec![0u8; n * k];
    for row in 0..n {
        for col in 0..k {
            weight[row * k + col] = fp8_pool[(row * 31 + col * 17 + 5) % fp8_pool.len()];
        }
    }
    let mut scales = vec![0.0f32; scale_rows * scale_cols];
    for (i, s) in scales.iter_mut().enumerate() {
        *s = 0.125 * ((i % 5) as f32 + 1.0); // 0.125 .. 0.625
    }
    let mut input_host = vec![bf16::ZERO; b * k];
    for batch in 0..b {
        for col in 0..k {
            let raw = ((batch * 13 + col * 7) % 23) as f32 - 11.0;
            input_host[batch * k + col] = bf16::from_f32(raw * 0.0625);
        }
    }

    let weight_dev = ctx.stream.clone_htod(&weight).expect("weight H2D");
    let scales_dev = ctx.stream.clone_htod(&scales).expect("scales H2D");
    let input_dev = ctx.stream.clone_htod(&input_host).expect("input H2D");
    let mut scalar_out = ctx
        .stream
        .alloc_zeros::<bf16>(b * n)
        .expect("scalar out alloc");
    let mut fallback_out = ctx
        .stream
        .alloc_zeros::<bf16>(b * n)
        .expect("fallback out alloc");
    let mut weight_bf16 = ctx
        .stream
        .alloc_zeros::<bf16>(n * k)
        .expect("weight bf16 alloc");

    // ---- Scalar reference (gemv_fp8_block_scaled_batch_cuda) ----
    {
        let (weight_ptr, _wg) = weight_dev.device_ptr(&ctx.stream);
        let (scales_ptr, _sg) = scales_dev.device_ptr(&ctx.stream);
        let (input_ptr, _ig) = input_dev.device_ptr(&ctx.stream);
        let (out_ptr, _og) = scalar_out.device_ptr_mut(&ctx.stream);
        unsafe {
            gemv_fp8_block_scaled_batch_cuda(
                weight_ptr as *const u8,
                scales_ptr as *const f32,
                input_ptr as *const Half,
                out_ptr as *mut Half,
                b as i32,
                n as i32,
                k as i32,
                scale_rows as i32,
                scale_cols as i32,
                block_m as i32,
                block_k as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .expect("scalar fp8 block GEMV");
        }
    }
    ctx.sync().expect("sync scalar GEMV");

    // ---- Fallback: dequant fp8 → bf16 weight, then bf16 cuBLAS GEMM ----
    {
        let (weight_ptr, _wg) = weight_dev.device_ptr(&ctx.stream);
        let (scales_ptr, _sg) = scales_dev.device_ptr(&ctx.stream);
        let (wbf16_ptr, _bg) = weight_bf16.device_ptr_mut(&ctx.stream);
        unsafe {
            dequantize_fp8_block_scaled_to_bf16_cuda(
                weight_ptr as *const u8,
                scales_ptr as *const f32,
                wbf16_ptr as *mut Half,
                n as i32,
                k as i32,
                scale_rows as i32,
                scale_cols as i32,
                block_m as i32,
                block_k as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .expect("dequant fp8 → bf16");
        }
    }
    ctx.sync().expect("sync dequant");
    {
        let (wbf16_ptr, _bg) = weight_bf16.device_ptr(&ctx.stream);
        let (input_ptr, _ig) = input_dev.device_ptr(&ctx.stream);
        let (out_ptr, _og) = fallback_out.device_ptr_mut(&ctx.stream);
        unsafe {
            // gemm_cuda(W[M,K], X[K,N], Y[M,N], M=n, N=b, K=k). This bf16
            // cuBLASLt call is the one under test for the sm<90 NOT_SUPPORTED
            // hypothesis — `.expect` panics if it errors.
            gemm_cuda(
                wbf16_ptr as *const Half,
                input_ptr as *const Half,
                out_ptr as *mut Half,
                n as i32,
                b as i32,
                k as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .expect("bf16 cuBLAS GEMM (settles sm<90 NOT_SUPPORTED hypothesis)");
        }
    }
    ctx.sync().expect("sync fallback GEMM");

    let scalar = ctx.stream.clone_dtoh(&scalar_out).expect("scalar D2H");
    let fallback = ctx.stream.clone_dtoh(&fallback_out).expect("fallback D2H");

    let mut dot = 0.0f64;
    let mut norm_s = 0.0f64;
    let mut norm_f = 0.0f64;
    let mut max_abs = 0.0f64;
    for (s, f) in scalar.iter().zip(fallback.iter()) {
        let sf = s.to_f32() as f64;
        let ff = f.to_f32() as f64;
        dot += sf * ff;
        norm_s += sf * sf;
        norm_f += ff * ff;
        max_abs = max_abs.max((sf - ff).abs());
    }
    let cosine = if norm_s > 0.0 && norm_f > 0.0 {
        dot / (norm_s.sqrt() * norm_f.sqrt())
    } else {
        1.0
    };
    eprintln!(
        "[fp8-dequant-bf16-parity] B={b} N={n} K={k} block={block_m}x{block_k} \
         cosine={cosine:.6} max_abs_err={max_abs:.4}"
    );

    // Both paths f32-accumulate; the fallback additionally rounds the
    // dequantized weight to bf16, so allow a slightly looser max_abs than the
    // MMA test while keeping cosine as the strong parity gate.
    assert!(
        cosine >= 0.999,
        "dequant→bf16 GEMM vs scalar cosine {cosine:.6} below 0.999 — fallback parity broken"
    );
    assert!(
        max_abs <= 1.0,
        "dequant→bf16 GEMM vs scalar max_abs_err {max_abs:.4} above 1.0 — fallback parity broken"
    );
}

// ---- W8A16 / W4A16 group-quantized GEMV + dequant correctness ----
//
// Weight layout: row-major [N, K] (W8A16 int8) or [N, K/2] (W4A16 packed
// uint8, low nibble = even col, high nibble = odd col). Per-group BF16 scales
// [N, K/group_size]: scale[row, col/group_size] multiplies every weight in
// that group. W4A16 stores unsigned nibbles 0..15; the dequant value is
// nibble - 8 (signed range -8..7).

fn quantize_w8a16_group(
    weight_f32: &[f32],
    n: usize,
    k: usize,
    group_size: usize,
) -> (Vec<i8>, Vec<bf16>) {
    let num_groups = k / group_size;
    let mut weight_i8 = vec![0i8; n * k];
    let mut scales = vec![bf16::ZERO; n * num_groups];
    for row in 0..n {
        for g in 0..num_groups {
            let start = g * group_size;
            let end = start + group_size;
            let mut max_abs = 0.0f32;
            for col in start..end {
                max_abs = max_abs.max(weight_f32[row * k + col].abs());
            }
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales[row * num_groups + g] = bf16::from_f32(scale);
            for col in start..end {
                let q = (weight_f32[row * k + col] / scale)
                    .round()
                    .clamp(-127.0, 127.0);
                weight_i8[row * k + col] = q as i8;
            }
        }
    }
    (weight_i8, scales)
}

fn quantize_w4a16_group(
    weight_f32: &[f32],
    n: usize,
    k: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<bf16>) {
    let num_groups = k / group_size;
    let mut weight_packed = vec![0u8; n * (k / 2)];
    let mut scales = vec![bf16::ZERO; n * num_groups];
    for row in 0..n {
        for g in 0..num_groups {
            let start = g * group_size;
            let end = start + group_size;
            let mut max_abs = 0.0f32;
            for col in start..end {
                max_abs = max_abs.max(weight_f32[row * k + col].abs());
            }
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            scales[row * num_groups + g] = bf16::from_f32(scale);
            for col in start..end {
                let q = (weight_f32[row * k + col] / scale).round().clamp(-8.0, 7.0);
                let nibble = (q as i8 + 8) as u8; // 0..15
                let byte_idx = row * (k / 2) + col / 2;
                if col & 1 == 0 {
                    weight_packed[byte_idx] = nibble;
                } else {
                    weight_packed[byte_idx] |= nibble << 4;
                }
            }
        }
    }
    (weight_packed, scales)
}

#[test]
fn w8a16_gemv_matches_reference() {
    let ctx = DeviceContext::new().expect("failed to create CUDA context");
    let n = 4usize;
    let k = 256usize;
    let group_size = 128usize;
    let num_groups = k / group_size;

    let mut weight_f32 = vec![0.0f32; n * k];
    for row in 0..n {
        for col in 0..k {
            let raw = ((row * 31 + col * 17 + 5) % 23) as f32 - 11.0;
            weight_f32[row * k + col] = raw * 0.03125;
        }
    }
    let (weight_i8, scales) = quantize_w8a16_group(&weight_f32, n, k, group_size);

    let mut input_host = vec![bf16::ZERO; k];
    for (col, val) in input_host.iter_mut().enumerate() {
        let raw = ((col * 7 + 3) % 19) as f32 - 9.0;
        *val = bf16::from_f32(raw * 0.0625);
    }

    let mut expected = vec![0.0f32; n];
    for row in 0..n {
        let mut sum = 0.0f32;
        for col in 0..k {
            let scale = scales[row * num_groups + col / group_size].to_f32();
            sum += weight_i8[row * k + col] as f32 * scale * input_host[col].to_f32();
        }
        expected[row] = sum;
    }

    let weight_dev = ctx.stream.clone_htod(&weight_i8).expect("weight H2D");
    let scales_dev = ctx.stream.clone_htod(&scales).expect("scales H2D");
    let input_dev = ctx.stream.clone_htod(&input_host).expect("input H2D");
    let mut output_dev = ctx.stream.alloc_zeros::<bf16>(n).expect("output alloc");
    {
        let (weight_ptr, _wg) = weight_dev.device_ptr(&ctx.stream);
        let (scales_ptr, _sg) = scales_dev.device_ptr(&ctx.stream);
        let (input_ptr, _ig) = input_dev.device_ptr(&ctx.stream);
        let (output_ptr, _og) = output_dev.device_ptr_mut(&ctx.stream);
        unsafe {
            w8a16_gemv_cuda(
                weight_ptr as *const i8,
                scales_ptr as *const Half,
                input_ptr as *const Half,
                output_ptr as *mut Half,
                n as i32,
                k as i32,
                group_size as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .expect("w8a16 GEMV");
        }
    }
    ctx.sync().expect("sync w8a16 GEMV");

    let got = ctx.stream.clone_dtoh(&output_dev).expect("output D2H");
    assert_bf16_close(&got, &expected, 0.05);
}

#[test]
fn w4a16_gemv_matches_reference() {
    let ctx = DeviceContext::new().expect("failed to create CUDA context");
    let n = 4usize;
    let k = 256usize;
    let group_size = 128usize;
    let num_groups = k / group_size;

    let mut weight_f32 = vec![0.0f32; n * k];
    for row in 0..n {
        for col in 0..k {
            let raw = ((row * 31 + col * 17 + 5) % 23) as f32 - 11.0;
            weight_f32[row * k + col] = raw * 0.03125;
        }
    }
    let (weight_packed, scales) = quantize_w4a16_group(&weight_f32, n, k, group_size);

    let mut input_host = vec![bf16::ZERO; k];
    for (col, val) in input_host.iter_mut().enumerate() {
        let raw = ((col * 7 + 3) % 19) as f32 - 9.0;
        *val = bf16::from_f32(raw * 0.0625);
    }

    let mut expected = vec![0.0f32; n];
    for row in 0..n {
        let mut sum = 0.0f32;
        for col in 0..k {
            let byte = weight_packed[row * (k / 2) + col / 2];
            let nibble = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 };
            let q = nibble as i8 - 8;
            let scale = scales[row * num_groups + col / group_size].to_f32();
            sum += q as f32 * scale * input_host[col].to_f32();
        }
        expected[row] = sum;
    }

    let weight_dev = ctx.stream.clone_htod(&weight_packed).expect("weight H2D");
    let scales_dev = ctx.stream.clone_htod(&scales).expect("scales H2D");
    let input_dev = ctx.stream.clone_htod(&input_host).expect("input H2D");
    let mut output_dev = ctx.stream.alloc_zeros::<bf16>(n).expect("output alloc");
    {
        let (weight_ptr, _wg) = weight_dev.device_ptr(&ctx.stream);
        let (scales_ptr, _sg) = scales_dev.device_ptr(&ctx.stream);
        let (input_ptr, _ig) = input_dev.device_ptr(&ctx.stream);
        let (output_ptr, _og) = output_dev.device_ptr_mut(&ctx.stream);
        unsafe {
            w4a16_gemv_cuda(
                weight_ptr as *const u8,
                scales_ptr as *const Half,
                input_ptr as *const Half,
                output_ptr as *mut Half,
                n as i32,
                k as i32,
                group_size as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .expect("w4a16 GEMV");
        }
    }
    ctx.sync().expect("sync w4a16 GEMV");

    let got = ctx.stream.clone_dtoh(&output_dev).expect("output D2H");
    assert_bf16_close(&got, &expected, 0.2);
}

#[test]
fn dequantize_w8a16_to_bf16_matches_reference() {
    let ctx = DeviceContext::new().expect("failed to create CUDA context");
    let n = 4usize;
    let k = 256usize;
    let group_size = 128usize;
    let num_groups = k / group_size;

    let mut weight_f32 = vec![0.0f32; n * k];
    for row in 0..n {
        for col in 0..k {
            let raw = ((row * 31 + col * 17 + 5) % 23) as f32 - 11.0;
            weight_f32[row * k + col] = raw * 0.03125;
        }
    }
    let (weight_i8, scales) = quantize_w8a16_group(&weight_f32, n, k, group_size);

    let mut expected = vec![0.0f32; n * k];
    for row in 0..n {
        for col in 0..k {
            let scale = scales[row * num_groups + col / group_size].to_f32();
            expected[row * k + col] = weight_i8[row * k + col] as f32 * scale;
        }
    }

    let weight_dev = ctx.stream.clone_htod(&weight_i8).expect("weight H2D");
    let scales_dev = ctx.stream.clone_htod(&scales).expect("scales H2D");
    let mut output_dev = ctx.stream.alloc_zeros::<bf16>(n * k).expect("output alloc");
    {
        let (weight_ptr, _wg) = weight_dev.device_ptr(&ctx.stream);
        let (scales_ptr, _sg) = scales_dev.device_ptr(&ctx.stream);
        let (output_ptr, _og) = output_dev.device_ptr_mut(&ctx.stream);
        unsafe {
            dequantize_w8a16_to_bf16_cuda(
                weight_ptr as *const i8,
                scales_ptr as *const Half,
                output_ptr as *mut Half,
                n as i32,
                k as i32,
                group_size as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .expect("dequant w8a16");
        }
    }
    ctx.sync().expect("sync dequant w8a16");

    let got = ctx.stream.clone_dtoh(&output_dev).expect("output D2H");
    assert_bf16_close(&got, &expected, 0.02);
}

#[test]
fn dequantize_w4a16_to_bf16_matches_reference() {
    let ctx = DeviceContext::new().expect("failed to create CUDA context");
    let n = 4usize;
    let k = 256usize;
    let group_size = 128usize;
    let num_groups = k / group_size;

    let mut weight_f32 = vec![0.0f32; n * k];
    for row in 0..n {
        for col in 0..k {
            let raw = ((row * 31 + col * 17 + 5) % 23) as f32 - 11.0;
            weight_f32[row * k + col] = raw * 0.03125;
        }
    }
    let (weight_packed, scales) = quantize_w4a16_group(&weight_f32, n, k, group_size);

    let mut expected = vec![0.0f32; n * k];
    for row in 0..n {
        for col in 0..k {
            let byte = weight_packed[row * (k / 2) + col / 2];
            let nibble = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 };
            let q = nibble as i8 - 8;
            let scale = scales[row * num_groups + col / group_size].to_f32();
            expected[row * k + col] = q as f32 * scale;
        }
    }

    let weight_dev = ctx.stream.clone_htod(&weight_packed).expect("weight H2D");
    let scales_dev = ctx.stream.clone_htod(&scales).expect("scales H2D");
    let mut output_dev = ctx.stream.alloc_zeros::<bf16>(n * k).expect("output alloc");
    {
        let (weight_ptr, _wg) = weight_dev.device_ptr(&ctx.stream);
        let (scales_ptr, _sg) = scales_dev.device_ptr(&ctx.stream);
        let (output_ptr, _og) = output_dev.device_ptr_mut(&ctx.stream);
        unsafe {
            dequantize_w4a16_to_bf16_cuda(
                weight_ptr as *const u8,
                scales_ptr as *const Half,
                output_ptr as *mut Half,
                n as i32,
                k as i32,
                group_size as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .expect("dequant w4a16");
        }
    }
    ctx.sync().expect("sync dequant w4a16");

    let got = ctx.stream.clone_dtoh(&output_dev).expect("output D2H");
    assert_bf16_close(&got, &expected, 0.05);
}

// ─── KV cache quantization round-trip tests ───

/// INT8 KV round-trip: bf16 → INT8 (per-token per-head absmax/127) → bf16.
///
/// Layout: HND `[num_kv_heads, max_seq_len, head_dim]`.
/// Symmetric quantization: `scale = absmax / 127`, `int8 = round(x / scale)`.
/// Max abs error ≈ scale / 2 = absmax / 254, so relative to the value range
/// the error is bounded by ~1/127 of the per-row absmax.
#[test]
fn int8_kv_quantize_dequantize_roundtrip() {
    let ctx = DeviceContext::new().expect("failed to create CUDA context");
    let num_kv_heads = 2usize;
    let head_dim = 64usize;
    let max_seq_len = 32usize;
    let token_count = max_seq_len;

    let total = num_kv_heads * max_seq_len * head_dim;
    let mut input_host = vec![bf16::ZERO; total];
    for (i, val) in input_host.iter_mut().enumerate() {
        // Values in [-1, 1), deterministic pseudo-random pattern.
        let v = ((i * 37 + 13) % 256) as f32 / 256.0 * 2.0 - 1.0;
        *val = bf16::from_f32(v);
    }

    let kv_bf16 = DeviceVec::from_host(&ctx, &input_host).expect("input H2D");
    let mut kv_int8 = ctx.stream.alloc_zeros::<i8>(total).expect("int8 alloc");
    let mut scales = ctx
        .stream
        .alloc_zeros::<f32>(num_kv_heads * max_seq_len)
        .expect("scales alloc");

    crate::kv_quant::quantize_kv(
        &ctx,
        &kv_bf16,
        &mut kv_int8,
        &mut scales,
        num_kv_heads,
        head_dim,
        max_seq_len,
        0, // start_pos
        token_count,
    )
    .expect("quantize kv int8");
    ctx.sync().expect("sync quantize kv int8");

    let mut out_bf16 = DeviceVec::zeros(&ctx, total).expect("out bf16 alloc");
    crate::kv_quant::dequantize_kv(
        &ctx,
        &kv_int8,
        &scales,
        &mut out_bf16,
        num_kv_heads,
        head_dim,
        max_seq_len,
        token_count,
    )
    .expect("dequantize kv int8");
    ctx.sync().expect("sync dequantize kv int8");

    let got = ctx.stream.clone_dtoh(&out_bf16.data).expect("out D2H");
    let expected: Vec<f32> = input_host.iter().map(|x| x.to_f32()).collect();
    // INT8 symmetric: step = scale, max error ≈ scale/2. With absmax ≤ 1 the
    // scale ≤ 1/127 ≈ 7.9e-3, so error ≤ ~4e-3. 2e-2 leaves headroom for
    // bf16 rounding of the input and the absmax edge case.
    assert_bf16_close(&got, &expected, 2e-2);
}

/// FP8 E4M3 KV round-trip: bf16 → FP8 (per-token per-head absmax/448) → bf16.
///
/// Input bf16 lives in the paged HND work buffer
/// `[num_pages, num_kv_heads, 16, head_dim]`; the FP8 durable pool is NHD
/// `[max_total_tokens, kv_dim]` where `kv_dim = num_kv_heads * head_dim`.
/// `new_token_indices` maps each batch token to its pool row.
///
/// Symmetric quantization: `scale = absmax / 448`, `fp8 = x / scale`.
/// E4M3 mantissa has 3 bits (8 levels), so the relative step is ~1/8 of the
/// bin; max abs error ≈ scale * (1/16) for values near absmax, i.e.
/// absmax / 7168. With absmax ≤ 1 the error is well below 5e-2.
#[test]
fn fp8_kv_quantize_dequantize_roundtrip() {
    let ctx = DeviceContext::new().expect("failed to create CUDA context");
    let num_kv_heads = 2usize;
    let head_dim = 64usize;
    let kv_dim = num_kv_heads * head_dim;
    let max_total_tokens = 32usize;
    const PAGE_SIZE: usize = 16;
    let num_pages = max_total_tokens.div_ceil(PAGE_SIZE);
    let batch_size = max_total_tokens;

    // Paged HND layout: [num_pages, num_kv_heads, PAGE_SIZE, head_dim].
    let total = num_pages * num_kv_heads * PAGE_SIZE * head_dim;
    let mut input_host = vec![bf16::ZERO; total];
    for (i, val) in input_host.iter_mut().enumerate() {
        let v = ((i * 37 + 13) % 256) as f32 / 256.0 * 2.0 - 1.0;
        *val = bf16::from_f32(v);
    }

    let kv_bf16 = ctx.stream.clone_htod(&input_host).expect("input H2D");
    let mut kv_fp8 = ctx
        .stream
        .alloc_zeros::<u8>(max_total_tokens * kv_dim)
        .expect("fp8 alloc");
    let mut scales = ctx
        .stream
        .alloc_zeros::<f32>(max_total_tokens * num_kv_heads)
        .expect("scales alloc");
    let token_indices: Vec<i32> = (0..batch_size as i32).collect();
    let token_indices_dev = ctx
        .stream
        .clone_htod(&token_indices)
        .expect("token indices H2D");

    {
        let (bf16_ptr, _g1) = kv_bf16.device_ptr(&ctx.stream);
        let (fp8_ptr, _g2) = kv_fp8.device_ptr_mut(&ctx.stream);
        let (scales_ptr, _g3) = scales.device_ptr_mut(&ctx.stream);

        crate::kv_quant::quantize_paged_kv_fp8(
            &ctx,
            bf16_ptr,
            fp8_ptr,
            scales_ptr,
            &token_indices_dev,
            num_kv_heads,
            head_dim,
            kv_dim,
            batch_size,
        )
        .expect("quantize kv fp8");
    }
    ctx.sync().expect("sync quantize kv fp8");

    let mut out_bf16 = ctx
        .stream
        .alloc_zeros::<bf16>(total)
        .expect("out bf16 alloc");
    {
        let (out_ptr, _g4) = out_bf16.device_ptr_mut(&ctx.stream);
        let (fp8_ptr, _g5) = kv_fp8.device_ptr(&ctx.stream);
        let (scales_ptr, _g6) = scales.device_ptr(&ctx.stream);

        crate::kv_quant::dequantize_paged_kv_fp8_to_hnd(
            &ctx,
            fp8_ptr,
            scales_ptr,
            out_ptr,
            &token_indices_dev,
            num_kv_heads,
            head_dim,
            kv_dim,
            batch_size,
        )
        .expect("dequantize kv fp8");
    }
    ctx.sync().expect("sync dequantize kv fp8");

    let got = ctx.stream.clone_dtoh(&out_bf16).expect("out D2H");
    let expected: Vec<f32> = input_host.iter().map(|x| x.to_f32()).collect();
    // FP8 E4M3 has 3 mantissa bits → ~12.5% relative resolution per bin.
    // Per-row scale = absmax/448, so the worst-case absolute error for a
    // value near absmax is ~absmax * (1/448) * (1/2) ≈ absmax/896. With
    // absmax ≤ 1 this is ~1e-3; 5e-2 covers bf16 input rounding and the
    // E4M3 subnormal/edge bins.
    assert_bf16_close(&got, &expected, 5e-2);
}
