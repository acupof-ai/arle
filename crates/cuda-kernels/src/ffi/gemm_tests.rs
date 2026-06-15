use super::*;
use crate::tensor::DeviceContext;
use cudarc::driver::{DevicePtr, DevicePtrMut};
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
