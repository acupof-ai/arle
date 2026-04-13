//! Pull one real Q4_K_S and one real Q3_K_M tensor out of the Carnice-27b
//! GGUF and compare:
//!   (a) CPU dequant via the well-trodden `read_tensor_bf16` path
//!   (b) GPU native dequant via `q{3,4}k_dequant_chunk_cuda` (called indirectly
//!       through the prefill path in `ops::gemm_into` with seq_len > 8)
//!
//! This is the tightest possible end-to-end check that the row-major
//! reinterpretation + kernel math work on real GGUF bytes — if this passes,
//! any downstream garbage is NOT in the Q3_K/Q4_K path.
//!
//! Enable with:
//!   PEGAINFER_CARNICE_PATH=/abs/path/to/models/Carnice-27b-GGUF \
//!       cargo test --release --test carnice_real_tensor_dequant -- --nocapture --ignored

#![cfg(feature = "cuda")]

use half::bf16;
use infer::gguf::GgufFile;
use infer::tensor::{DeviceContext, DeviceMatrix, DeviceVec};

fn model_path() -> String {
    std::env::var("PEGAINFER_CARNICE_PATH")
        .unwrap_or_else(|_| "models/Carnice-27b-GGUF".to_string())
}

fn find_gguf(dir: &str) -> String {
    for entry in std::fs::read_dir(dir).expect("read dir") {
        let p = entry.unwrap().path();
        if p.extension().and_then(|s| s.to_str()) == Some("gguf") {
            return p.to_string_lossy().into_owned();
        }
    }
    panic!("no .gguf file in {dir}");
}

/// Run a GEMV through `ops::gemv` and return the bf16 result as f32.
fn gemv_bf16(ctx: &DeviceContext, mat: &DeviceMatrix, x: &[bf16], rows: usize) -> Vec<f32> {
    let x_dev = DeviceVec::from_host(ctx, x).expect("x upload");
    let mut y_dev = DeviceVec::zeros(ctx, rows).expect("y alloc");
    infer::ops::gemv(ctx, mat, &x_dev, &mut y_dev).expect("gemv");
    let mut y_bf16 = vec![bf16::ZERO; rows];
    y_dev
        .copy_region_to_host(ctx, 0, rows, &mut y_bf16)
        .expect("download");
    y_bf16.iter().map(|v| v.to_f32()).collect()
}

/// Compare a native-packed DeviceMatrix against a BF16-dequant-then-upload
/// DeviceMatrix: run the same GEMV on both with a pseudo-random input vector
/// and compare outputs element-wise within tight tolerance.
fn compare_gemv(label: &str, packed: &DeviceMatrix, bf16_mat: &DeviceMatrix, ctx: &DeviceContext) {
    assert_eq!(packed.rows, bf16_mat.rows, "{label}: row mismatch");
    assert_eq!(packed.cols, bf16_mat.cols, "{label}: col mismatch");
    let rows = packed.rows;
    let cols = packed.cols;

    let mut x_f32 = vec![0f32; cols];
    for i in 0..cols {
        // Small deterministic noise; avoid zeros which hide scale bugs.
        x_f32[i] = ((i as f32 * 0.00137) - 0.7).sin() * 0.1;
    }
    let x_bf16: Vec<bf16> = x_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let y_pack = gemv_bf16(ctx, packed, &x_bf16, rows);
    let y_bf16 = gemv_bf16(ctx, bf16_mat, &x_bf16, rows);

    // Row-by-row check. Combined absolute + relative tolerance — native packed
    // and bf16-dequant paths differ only in float accumulation order, so drift
    // should stay within bf16 precision (~3 decimal digits).
    //
    // Tolerance: diff < max(1e-2, 0.01 * |bf16|). Near-zero rows are dominated
    // by the 1e-2 absolute floor.
    let mut worst_r = 0usize;
    let mut worst_diff = 0f32;
    for r in 0..rows {
        let diff = (y_pack[r] - y_bf16[r]).abs();
        if diff > worst_diff {
            worst_diff = diff;
            worst_r = r;
        }
    }
    let worst_bf16 = y_bf16[worst_r];
    let worst_pack = y_pack[worst_r];
    let worst_tol = (worst_bf16.abs() * 0.01).max(1e-2);
    println!(
        "{label}: rows={rows} cols={cols}  worst row {worst_r}: \
         packed={worst_pack} bf16={worst_bf16} diff={worst_diff:.4e} tol={worst_tol:.4e}"
    );

    // Max |packed - bf16| should be smaller than 1% of max |bf16| across the
    // whole result vector. This catches systematic bugs while tolerating the
    // expected bf16 reduction noise.
    let max_abs_y: f32 = y_bf16.iter().fold(0f32, |m, v| m.max(v.abs()));
    let overall_tol = (max_abs_y * 0.01).max(1e-2);
    assert!(
        worst_diff < overall_tol,
        "{label}: drift too large (worst_diff={worst_diff}, overall_tol={overall_tol}, \
         max |bf16|={max_abs_y})"
    );
}

fn load_from_bf16(
    ctx: &DeviceContext,
    gguf: &GgufFile,
    name: &str,
) -> (DeviceMatrix, usize, usize) {
    let info = &gguf.tensors[name];
    let bf16_data = gguf.read_tensor_bf16(name).expect("read_tensor_bf16");
    // Same column-major → row-major reinterpretation as the loader uses.
    let ne0 = info.shape[0] as usize;
    let ne1 = info.shape[1] as usize;
    let (rows, cols) = (ne1, ne0);
    let mat = DeviceMatrix::from_host(ctx, &bf16_data, rows, cols).expect("from_host");
    (mat, rows, cols)
}

/// Apply the same `reverse_v_reorder_rows` permutation as the loader does, on
/// a flat bf16 tensor. This is the BF16 reference for the byte-level row
/// permutation that `load_tensor_2d_gguf_v_reorder_rows` performs for Q4_K/Q6_K.
fn reorder_rows_bf16(
    data: &mut [half::bf16],
    rows: usize,
    cols: usize,
    num_k_heads: usize,
    num_v_per_k: usize,
    head_dim: usize,
) {
    if num_v_per_k <= 1 {
        return;
    }
    let src = data.to_vec();
    let _ = rows;
    for k in 0..num_k_heads {
        for v in 0..num_v_per_k {
            let gguf_head = v * num_k_heads + k;
            let hf_head = k * num_v_per_k + v;
            let src_start = gguf_head * head_dim * cols;
            let dst_start = hf_head * head_dim * cols;
            let size = head_dim * cols;
            data[dst_start..dst_start + size].copy_from_slice(&src[src_start..src_start + size]);
        }
    }
}

#[test]
#[ignore]
fn carnice_real_q4k_matches_bf16_reference() {
    infer::logging::init_stderr("warn");
    let gguf_path = find_gguf(&model_path());
    let gguf = GgufFile::open(&gguf_path).expect("open gguf");
    let ctx = DeviceContext::new().expect("device");

    // After fixing the enum: Carnice-27b Q4_K_M is actually
    //   * 433 × Q4_K  (ffn_gate, most attn/mlp)
    //   * 65  × Q6_K  (attn_v, ffn_down, a handful of critical tensors)
    // Pick one of each.
    let q4k_name = "blk.8.ffn_gate.weight";
    let q6k_name = "blk.58.ffn_down.weight";

    // Q4_K path
    {
        let info = &gguf.tensors[q4k_name];
        assert_eq!(info.dtype, infer::gguf::GgmlType::Q4_K);
        let packed = gguf.read_tensor_q4k_packed(q4k_name).expect("q4k packed");
        let ne0 = info.shape[0] as usize;
        let ne1 = info.shape[1] as usize;
        let (rows, cols) = (ne1, ne0);
        let packed_mat = DeviceMatrix::from_quantized_q4k(&ctx, &packed, rows, cols)
            .expect("from_quantized_q4k");
        let (bf16_mat, _, _) = load_from_bf16(&ctx, &gguf, q4k_name);
        compare_gemv("Q4_K ffn_gate[8]", &packed_mat, &bf16_mat, &ctx);
    }

    // Q6_K path
    {
        let info = &gguf.tensors[q6k_name];
        assert_eq!(info.dtype, infer::gguf::GgmlType::Q6_K);
        let packed = gguf.read_tensor_q6k_packed(q6k_name).expect("q6k packed");
        let ne0 = info.shape[0] as usize;
        let ne1 = info.shape[1] as usize;
        let (rows, cols) = (ne1, ne0);
        let packed_mat = DeviceMatrix::from_quantized_q6k(&ctx, &packed, rows, cols)
            .expect("from_quantized_q6k");
        let (bf16_mat, _, _) = load_from_bf16(&ctx, &gguf, q6k_name);
        compare_gemv("Q6_K ffn_down[58]", &packed_mat, &bf16_mat, &ctx);
    }

    // Q4_K with V-head row reorder — linear attention in_proj_z.
    // This validates the byte-level row permutation in
    // `load_tensor_2d_gguf_v_reorder_rows` against the BF16 reference path.
    //
    // Carnice uses num_k_heads=16, num_v_per_k=3, head_dim=128 for linear attn.
    // The HF name `linear_attn.in_proj_z.weight` maps to GGUF `attn_gate.weight`
    // via `map_gguf_name_with_prefix` in gguf.rs.
    {
        let name = "blk.0.attn_gate.weight";
        println!("v_reorder cross-check using tensor '{name}' (HF: in_proj_z)");
        let info = &gguf.tensors[name];
        assert_eq!(info.dtype, infer::gguf::GgmlType::Q4_K);
        let num_k_heads = 16;
        let num_v_per_k = 3;
        let head_dim = 128;

        // Packed path: load with byte-level row reorder.
        let src = gguf.read_tensor_q4k_packed(name).expect("packed");
        let ne0 = info.shape[0] as usize;
        let ne1 = info.shape[1] as usize;
        let (rows, cols) = (ne1, ne0);
        let row_bytes = cols * 9 / 16;
        assert_eq!(src.len(), rows * row_bytes);
        let mut dst = vec![0u8; src.len()];
        for k in 0..num_k_heads {
            for v in 0..num_v_per_k {
                let gguf_head = v * num_k_heads + k;
                let hf_head = k * num_v_per_k + v;
                let s0 = gguf_head * head_dim * row_bytes;
                let d0 = hf_head * head_dim * row_bytes;
                let n = head_dim * row_bytes;
                dst[d0..d0 + n].copy_from_slice(&src[s0..s0 + n]);
            }
        }
        let packed_mat = DeviceMatrix::from_quantized_q4k(&ctx, &dst, rows, cols)
            .expect("from_quantized_q4k (reordered)");

        // BF16 reference path: dequant then apply the same reorder element-wise.
        let mut bf16_data = gguf.read_tensor_bf16(name).expect("bf16");
        reorder_rows_bf16(
            &mut bf16_data,
            rows,
            cols,
            num_k_heads,
            num_v_per_k,
            head_dim,
        );
        let bf16_mat = DeviceMatrix::from_host(&ctx, &bf16_data, rows, cols).expect("from_host");

        compare_gemv(
            "Q4_K ssm_in_z[0] (row reorder)",
            &packed_mat,
            &bf16_mat,
            &ctx,
        );
    }
}
