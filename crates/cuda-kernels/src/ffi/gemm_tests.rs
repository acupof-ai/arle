use super::*;
use crate::tensor::DeviceContext;
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

// Parity harness: the Qwen f32-block-scale FP8 tensor-core MMA GEMV
// (`gemv_fp8_block_scaled_batch_mma_launch`) vs the scalar fallback
// (`gemv_fp8_block_scaled_batch_cuda`) on the SAME fp8 block-scaled weights +
// input. Prints cosine + max_abs_err and asserts they are within the bf16/quant
// noise floor. Mirrors the DSv4 MMA port's correctness gate (cosine 0.999999).
//
// Canonical Qwen3.6-27B-FP8 quant block = 128×128, so block_k = 128 (==
// MMA_BLOCK_K) → the MMA launch activates (it rejects block_k % 128 != 0).
// N % 8 == 0 and K % 16 == 0 are also required by the MMA tile.
//
// Run on the pod (CUDA box):
//   CUDA_HOME=/usr/local/cuda cargo test -p cuda-kernels --release \
//     --features cuda fp8_block_scaled_gemv_mma_matches_scalar -- --nocapture
#[test]
fn fp8_block_scaled_gemv_mma_matches_scalar() {
    let ctx = DeviceContext::new().expect("failed to create CUDA context");
    let b = 4usize; // decode shape (B <= 16)
    let n = 128usize; // N % 8 == 0
    let k = 256usize; // K % 16 == 0, = 2 × block_k
    let block_m = 128usize; // canonical Qwen FP8 quant block
    let block_k = 128usize; // == MMA_BLOCK_K → MMA path activates
    let scale_rows = n.div_ceil(block_m); // = 1
    let scale_cols = k.div_ceil(block_k); // = 2

    // Finite, in-range e4m3 weight bytes (avoid NaN 0x7f/0xff). Deterministic
    // pseudo-random spread over a few representable magnitudes per sign.
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
    // f32 block scales (one per [block_m, block_k] tile).
    let mut scales = vec![0.0f32; scale_rows * scale_cols];
    for (i, s) in scales.iter_mut().enumerate() {
        *s = 0.125 * ((i % 5) as f32 + 1.0); // 0.125 .. 0.625
    }
    // bf16 activation.
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
    let mut mma_out = ctx
        .stream
        .alloc_zeros::<bf16>(b * n)
        .expect("mma out alloc");

    // ---- Scalar reference ----
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

    // ---- MMA path ----
    {
        let (weight_ptr, _wg) = weight_dev.device_ptr(&ctx.stream);
        let (scales_ptr, _sg) = scales_dev.device_ptr(&ctx.stream);
        let (input_ptr, _ig) = input_dev.device_ptr(&ctx.stream);
        let (out_ptr, _og) = mma_out.device_ptr_mut(&ctx.stream);
        unsafe {
            gemv_fp8_block_scaled_batch_mma_launch(
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
            .expect("mma fp8 block GEMV (shape should be MMA-tileable)");
        }
    }
    ctx.sync().expect("sync MMA GEMV");

    let scalar = ctx.stream.clone_dtoh(&scalar_out).expect("scalar D2H");
    let mma = ctx.stream.clone_dtoh(&mma_out).expect("mma D2H");

    let mut dot = 0.0f64;
    let mut norm_s = 0.0f64;
    let mut norm_m = 0.0f64;
    let mut max_abs = 0.0f64;
    for (s, m) in scalar.iter().zip(mma.iter()) {
        let sf = s.to_f32() as f64;
        let mf = m.to_f32() as f64;
        dot += sf * mf;
        norm_s += sf * sf;
        norm_m += mf * mf;
        max_abs = max_abs.max((sf - mf).abs());
    }
    let cosine = if norm_s > 0.0 && norm_m > 0.0 {
        dot / (norm_s.sqrt() * norm_m.sqrt())
    } else {
        // both all-zero → degenerate; treat as identical
        1.0
    };
    eprintln!(
        "[fp8-mma-parity] B={b} N={n} K={k} block={block_m}x{block_k} \
         cosine={cosine:.6} max_abs_err={max_abs:.4}"
    );

    // bf16 + per-block-scale quant noise floor. The DSv4 MMA port hit cosine
    // 0.999999 / max_abs 0.07-0.13; allow a slightly looser bound for the
    // smaller test shape.
    assert!(
        cosine >= 0.999,
        "MMA vs scalar cosine {cosine:.6} below 0.999 — kernel parity broken"
    );
    assert!(
        max_abs <= 0.5,
        "MMA vs scalar max_abs_err {max_abs:.4} above 0.5 — kernel parity broken"
    );
}
