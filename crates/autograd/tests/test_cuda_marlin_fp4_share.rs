#![cfg(all(feature = "cuda", not(feature = "no-cuda")))]
//! A shared NVFP4 base read through the serving engine's Marlin layout must
//! dequantize to what the group layout gives, or the student trains against
//! different weights than the rollout engine serves.

use autograd::Backend;
use autograd::backend_cuda::CudaBackend;
use cuda_kernels::tensor::{DeviceContext, DeviceMatrix};
use cudarc::driver::DevicePtr;

/// NVFP4 E2M1 code -> value, matching the kernel's LUT.
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn e4m3_bits(value: f32) -> u8 {
    // Small exact powers of two only — enough for a test weight, and it keeps
    // the expected values free of encoder round-off.
    let exp = value.log2().round() as i32;
    (((exp + 7) as u8) & 0x0f) << 3
}

#[test]
fn cuda_marlin_fp4_dequant_matches_group_layout() {
    let (n, k) = (128usize, 128usize);
    let group_size = 16usize;
    let scale_cols = k / group_size;

    // Deterministic nibbles and per-group scales.
    let mut packed = vec![0u8; n * k / 2];
    for (i, byte) in packed.iter_mut().enumerate() {
        let lo = (i * 7 + 3) % 16;
        let hi = (i * 5 + 11) % 16;
        *byte = (lo as u8) | ((hi as u8) << 4);
    }
    let mut scales = vec![0u8; n * scale_cols];
    for (i, s) in scales.iter_mut().enumerate() {
        // 2^-2 .. 2^1, all inside E4M3 and inside the S0E5M3 window after the
        // repack's lift, so nothing is flushed and the two paths must agree.
        *s = e4m3_bits(2f32.powi((i % 4) as i32 - 2));
    }
    let global_scale = 0.5f32;

    let backend = CudaBackend::new(0).expect("cuda backend");
    let group = backend
        .upload_fp4_e2m1_group(
            &packed,
            &scales,
            global_scale,
            &[n, k],
            group_size,
            scale_cols,
        )
        .expect("upload group layout");

    // The engine's side: same bytes, repacked, group source released.
    let ctx = DeviceContext::new().expect("device context");
    let mut matrix = DeviceMatrix::from_fp4_e2m1_group(
        &ctx,
        &packed,
        &scales,
        &[global_scale],
        None,
        n,
        k,
        group_size,
    )
    .expect("nvfp4 device matrix");
    matrix.repack_for_marlin_fp4(&ctx).expect("marlin repack");
    let marlin_buf = matrix.marlin_packed.as_ref().expect("marlin weights");
    let global_buf = matrix.marlin_scales.as_ref().expect("marlin global scale");
    let (weight_ptr, _wg) = marlin_buf.device_ptr(&ctx.stream);
    let scale_tail_ptr = weight_ptr + (n * k / 2) as u64;
    let folded_bits = ctx
        .stream
        .clone_dtoh(global_buf)
        .expect("D2H marlin global scale");
    ctx.sync().expect("sync");
    let folded_global = f32::from_bits(u32::from(folded_bits[0]) << 16);
    // Dequant is internal; `I @ Wᵀ` reaches it through the public GEMM seam.
    let mut eye = vec![0f32; k * k];
    for i in 0..k {
        eye[i * k + i] = 1.0;
    }
    let eye_handle = backend.upload(&eye, &[k, k]).expect("upload identity");
    let dequant = |handle: &_, rows: usize| {
        let (out, _) = backend
            .matmul_bt(&eye_handle, &[k, k], handle, &[rows, k])
            .expect("matmul");
        backend.readback(&out).expect("readback")
    };

    let full = dequant(&group, n);
    assert!(
        full.iter().any(|v| *v != 0.0),
        "group dequant produced zeros"
    );

    // Whole matrix, then a row window — q/k/v share one fused Marlin buffer, so
    // an offset view has to walk the same tiles and keep only its own rows.
    for (row_offset, rows) in [(0usize, n), (64usize, 64usize)] {
        let marlin = backend
            .import_fp4_marlin_device_ptr(
                weight_ptr,
                scale_tail_ptr,
                folded_global,
                &[rows, k],
                (n, row_offset),
            )
            .expect("import marlin view");
        let got = dequant(&marlin, rows);
        // `matmul_bt` gives [k, rows] — column `r` is output row `row_offset + r`.
        assert_eq!(got.len(), k * rows);
        for i in 0..k {
            for r in 0..rows {
                let want = full[i * n + row_offset + r];
                let have = got[i * rows + r];
                assert!(
                    want == have,
                    "marlin dequant diverged at offset={row_offset} [{i},{r}]: \
                     group={want} marlin={have}"
                );
            }
        }
    }
    let _ = E2M1;
}
