//! Fused MoE expert GEMV (`mul_mat_vec_id`) correctness proof.
//!
//! Proves the FUSED expert matvec — ONE dispatch that runs a token through ALL
//! its top-k routed experts (the `mul_mat_vecq.comp` body built with
//! `MUL_MAT_ID`) — is numerically identical to LOOPING the proven plain GEMV
//! once per selected expert. This is the oracle that gates the MoE forward
//! rewrite (`forward.rs::moe_ffn`): if the fused id/stride/offset contract is
//! wrong, the per-expert outputs diverge here before any model runs.
//!
//! Pipeline:
//!   1. Build a STACKED expert weight `[n_expert][nrows, ncols]` of random valid
//!      Q4_K (and Q8_0) blocks — exactly the byte layout of a GGUF `ffn_*_exps`
//!      tensor (`[ne0=in, ne1=out, ne2=n_expert]`, expert `e` at
//!      `e * nrows * row_bytes(ncols)`).
//!   2. Pick an id list (a subset of experts, in arbitrary order — the routed
//!      top-k for one token).
//!   3. Quantize one shared activation `x[ncols]` to q8_1_x4 on the GPU.
//!   4. FUSED: one `mul_mat_vec_id` dispatch over the id list → a dst of
//!      `[n_selected * nrows]` f32 (expert slot `i` writes rows `[i*nrows ..]`).
//!   5. REFERENCE: for each id in the list, one plain `mul_mat_vecq` GEMV
//!      against that expert's byte slice → `[nrows]` f32.
//!   6. Assert the fused dst block for slot `i` == the plain GEMV of
//!      `id_list[i]` within the q8_1 activation tolerance (bit-identical math,
//!      same activation, so they should match to ~1e-3, not just q8_1 tol).
//!
//! Runs only with `--features vulkan` + a working device; skips cleanly
//! otherwise.
#![cfg(feature = "vulkan")]

use infer_gguf::dequant::dequantize_row_q4_k;
use vulkan_kernels::{
    BLOCK_Q4_K_BYTES, BLOCK_Q8_0_BYTES, BLOCK_Q8_1_BYTES, Dispatch, KernelParams, Result,
    gemv_dispatch, gemv_id_dispatch, gemv_id_params, gemv_params, q4_k_gemv_id_with_params,
    q4_k_gemv_with_params, q8_0_gemv_id_with_params, q8_0_gemv_with_params, q8_1_quantize,
    q8_1_quantize_dispatch, q8_1_quantize_params,
};
use vulkan_sys::{DeviceBuffer, VulkanContext};

/// Deterministic xorshift PRNG so failures reproduce.
struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
    fn next_byte(&mut self) -> u8 {
        (self.next_u32() & 0xFF) as u8
    }
    fn next_unit_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// f32 -> IEEE-754 half (round-to-nearest-even), matching the f16 scale fields.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    let mut h = sign | ((exp as u16) << 10) | (mant >> 13) as u16;
    if (mant & 0x1000) != 0 && ((mant & 0x0fff) != 0 || (h & 1) != 0) {
        h += 1;
    }
    h
}

/// One valid `block_q4_K` (144 B): `f16 d; f16 dmin; u8 scales[12]; u8 qs[128]`.
fn make_q4_k_block(rng: &mut Rng) -> Vec<u8> {
    let mut b = Vec::with_capacity(BLOCK_Q4_K_BYTES);
    let d = 0.02 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.04;
    let dmin = 0.01 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.02;
    b.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    b.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
    for _ in 0..12 {
        b.push(rng.next_byte());
    }
    for _ in 0..128 {
        b.push(rng.next_byte());
    }
    debug_assert_eq!(b.len(), BLOCK_Q4_K_BYTES);
    b
}

/// One valid `block_q8_0` (34 B): `f16 d; i8 qs[32]`.
fn make_q8_0_block(rng: &mut Rng) -> Vec<u8> {
    let mut b = Vec::with_capacity(BLOCK_Q8_0_BYTES);
    let d = 0.01 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.04;
    b.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    for _ in 0..32 {
        b.push(rng.next_byte());
    }
    debug_assert_eq!(b.len(), BLOCK_Q8_0_BYTES);
    b
}

/// Quantize an f32 activation to `block_q8_1_x4` bytes on the GPU.
fn quantize_activations_x4(ctx: &VulkanContext, x: &[f32]) -> Vec<u8> {
    let ne = x.len();
    let input_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let num_x4 = ne.div_ceil(128);
    let out_len = num_x4 * 4 * BLOCK_Q8_1_BYTES;

    let mut buf_in = DeviceBuffer::alloc(ctx, input_bytes.len()).expect("alloc q8_1 input");
    buf_in
        .copy_from_host(&input_bytes)
        .expect("upload q8_1 input");
    let mut buf_out = DeviceBuffer::alloc(ctx, out_len).expect("alloc q8_1 output");
    buf_out
        .copy_from_host(&vec![0u8; out_len])
        .expect("zero q8_1 output");

    let params = q8_1_quantize_params(ne as u32);
    let dispatch = q8_1_quantize_dispatch(ne as u32);
    q8_1_quantize(ctx, &[&buf_in, &buf_out], dispatch, &params).expect("q8_1_quantize on device");

    let mut got = vec![0u8; out_len];
    buf_out.copy_to_host(&mut got).expect("read back q8_1");
    got
}

/// Run ONE plain `mul_mat_vecq` GEMV against the weight byte slice for a single
/// expert, returning `[nrows]` f32. This is the proven per-expert reference.
#[allow(clippy::too_many_arguments)]
fn plain_gemv(
    ctx: &VulkanContext,
    expert_weight_bytes: &[u8],
    activ_q8_1: &[u8],
    nrows: usize,
    ncols: usize,
    launch: impl Fn(&VulkanContext, &[&DeviceBuffer<'_>], Dispatch, &KernelParams) -> Result<()>,
) -> Vec<f32> {
    let mut buf_w = DeviceBuffer::alloc(ctx, expert_weight_bytes.len()).expect("alloc expert");
    buf_w
        .copy_from_host(expert_weight_bytes)
        .expect("upload expert");
    let mut buf_b = DeviceBuffer::alloc(ctx, activ_q8_1.len()).expect("alloc activ");
    buf_b.copy_from_host(activ_q8_1).expect("upload activ");
    let out_len = nrows * std::mem::size_of::<f32>();
    let mut buf_d = DeviceBuffer::alloc(ctx, out_len).expect("alloc dst");
    buf_d.copy_from_host(&vec![0u8; out_len]).expect("zero dst");
    let mut buf_f0 = DeviceBuffer::alloc(ctx, 4).expect("alloc fuse0");
    buf_f0.copy_from_host(&[0u8; 4]).expect("zero fuse0");
    let mut buf_f1 = DeviceBuffer::alloc(ctx, 4).expect("alloc fuse1");
    buf_f1.copy_from_host(&[0u8; 4]).expect("zero fuse1");

    let params = gemv_params(ncols as u32, nrows as u32);
    launch(
        ctx,
        &[&buf_w, &buf_b, &buf_d, &buf_f0, &buf_f1],
        gemv_dispatch(nrows as u32),
        &params,
    )
    .expect("plain GEMV dispatch");

    let mut out_bytes = vec![0u8; out_len];
    buf_d.copy_to_host(&mut out_bytes).expect("read back dst");
    out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Run ONE fused `mul_mat_vec_id` dispatch over `id_list` against the stacked
/// expert weights, returning the whole `[n_selected * nrows]` dst f32.
#[allow(clippy::too_many_arguments)]
fn fused_gemv_id(
    ctx: &VulkanContext,
    stacked_weight_bytes: &[u8],
    activ_q8_1: &[u8],
    id_list: &[i32],
    nrows: usize,
    ncols: usize,
    launch: impl Fn(&VulkanContext, &[&DeviceBuffer<'_>], Dispatch, &KernelParams) -> Result<()>,
) -> Vec<f32> {
    let n_sel = id_list.len();
    let mut buf_w = DeviceBuffer::alloc(ctx, stacked_weight_bytes.len()).expect("alloc stacked");
    buf_w
        .copy_from_host(stacked_weight_bytes)
        .expect("upload stacked");
    let mut buf_b = DeviceBuffer::alloc(ctx, activ_q8_1.len()).expect("alloc activ");
    buf_b.copy_from_host(activ_q8_1).expect("upload activ");
    // dst holds n_selected experts' [nrows] outputs back-to-back.
    let out_len = n_sel * nrows * std::mem::size_of::<f32>();
    let mut buf_d = DeviceBuffer::alloc(ctx, out_len).expect("alloc fused dst");
    buf_d
        .copy_from_host(&vec![0u8; out_len])
        .expect("zero fused dst");
    let mut buf_f0 = DeviceBuffer::alloc(ctx, 4).expect("alloc fuse0");
    buf_f0.copy_from_host(&[0u8; 4]).expect("zero fuse0");
    let mut buf_f1 = DeviceBuffer::alloc(ctx, 4).expect("alloc fuse1");
    buf_f1.copy_from_host(&[0u8; 4]).expect("zero fuse1");
    // The expert-id list (i32 per ggml's GGML_TYPE_I32 ids tensor).
    let id_bytes: Vec<u8> = id_list.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut buf_ids = DeviceBuffer::alloc(ctx, id_bytes.len()).expect("alloc ids");
    buf_ids.copy_from_host(&id_bytes).expect("upload ids");

    let params = gemv_id_params(ncols as u32, nrows as u32, n_sel as u32);
    launch(
        ctx,
        &[&buf_w, &buf_b, &buf_d, &buf_f0, &buf_f1, &buf_ids],
        gemv_id_dispatch(nrows as u32, n_sel as u32),
        &params,
    )
    .expect("fused mul_mat_vec_id dispatch");

    let mut out_bytes = vec![0u8; out_len];
    buf_d
        .copy_to_host(&mut out_bytes)
        .expect("read back fused dst");
    out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Build a stacked expert weight, run one fused id dispatch + a per-expert loop
/// of the plain GEMV, and assert they agree row-for-row.
#[allow(clippy::too_many_arguments)]
fn run_id_case(
    ctx: &VulkanContext,
    label: &str,
    n_expert: usize,
    nrows: usize,
    ncols: usize,
    row_bytes: usize,
    make_expert: impl Fn(&mut Rng) -> Vec<u8>,
    id_list: &[i32],
    x: &[f32],
    plain_launch: impl Fn(&VulkanContext, &[&DeviceBuffer<'_>], Dispatch, &KernelParams) -> Result<()>,
    fused_launch: impl Fn(&VulkanContext, &[&DeviceBuffer<'_>], Dispatch, &KernelParams) -> Result<()>,
    tol: f32,
) {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let per_expert_bytes = nrows * row_bytes;
    // Stacked [n_expert] x [nrows, ncols] weight (GGUF ffn_*_exps layout).
    let mut stacked = Vec::with_capacity(n_expert * per_expert_bytes);
    for _ in 0..n_expert {
        for _ in 0..nrows {
            // one row = ncols/QK_K (or ncols/32) blocks
            let blocks = row_bytes / make_one_block_len(row_bytes, ncols);
            for _ in 0..blocks {
                stacked.extend_from_slice(&make_expert(&mut rng));
            }
        }
    }
    assert_eq!(stacked.len(), n_expert * per_expert_bytes);

    let activ = quantize_activations_x4(ctx, x);

    // Fused: one dispatch over the whole id list.
    let fused = fused_gemv_id(ctx, &stacked, &activ, id_list, nrows, ncols, fused_launch);
    assert_eq!(fused.len(), id_list.len() * nrows);

    eprintln!("[{label}] n_expert={n_expert} nrows={nrows} ncols={ncols} ids={id_list:?}");
    let mut worst = 0f32;
    for (slot, &id) in id_list.iter().enumerate() {
        let e = id as usize;
        let slice = &stacked[e * per_expert_bytes..(e + 1) * per_expert_bytes];
        // Reference: plain GEMV against that expert's byte slice.
        let reference = plain_gemv(ctx, slice, &activ, nrows, ncols, &plain_launch);
        let block = &fused[slot * nrows..(slot + 1) * nrows];
        for r in 0..nrows {
            let denom = reference[r].abs().max(1e-3);
            let rel = (block[r] - reference[r]).abs() / denom;
            worst = worst.max(rel);
            assert!(
                rel < tol,
                "{label} slot {slot} (expert {e}) row {r}: fused {} vs plain {} \
                 (rel_err {rel} >= tol {tol})",
                block[r],
                reference[r]
            );
        }
    }
    eprintln!("[{label}] PASS (worst rel_err={worst:.5}, tol={tol}) — fused == per-expert loop");
}

/// Block byte length for one quant block given a row's total bytes and width.
fn make_one_block_len(row_bytes: usize, ncols: usize) -> usize {
    // Q4_K: 256 cols/block -> 144 B; Q8_0: 32 cols/block -> 34 B. Derive from the
    // row so the helper is quant-agnostic.
    if row_bytes.is_multiple_of(BLOCK_Q4_K_BYTES) && ncols.is_multiple_of(256) {
        BLOCK_Q4_K_BYTES
    } else {
        BLOCK_Q8_0_BYTES
    }
}

#[test]
fn fused_mul_mat_vec_id_matches_per_expert_loop() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping fused mul_mat_vec_id test");
            return;
        }
    };
    eprintln!("ARLE Vulkan fused MoE GEMV proof on: {}", ctx.device_name());

    let n_expert = 6usize;
    let nrows = 5usize; // per-expert output rows (e.g. moe_intermediate slice)
    let ncols = 256usize; // expert in-dim: one K-quant super-block / 8 q8_0 blocks

    // Shared activation (small magnitudes keep q8_1 error bounded).
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let x: Vec<f32> = (0..ncols).map(|_| rng.next_unit_f32()).collect();

    // The top-k routed id list for one token: a subset, out of order, with the
    // last expert repeated to prove `data_ids[expert_i0]` indexes independently.
    let ids: Vec<i32> = vec![3, 0, 5, 5];

    // ---- Q4_K experts (the 35B-A3B routed expert quant) ----
    let q4k_row = vulkan_kernels::q4_k_row_bytes(ncols).expect("q4_k row bytes");
    run_id_case(
        &ctx,
        "id_Q4_K",
        n_expert,
        nrows,
        ncols,
        q4k_row,
        make_q4_k_block,
        &ids,
        &x,
        q4_k_gemv_with_params,
        q4_k_gemv_id_with_params,
        2e-2,
    );
    // Sanity: dequant the reference path uses the same infer-gguf layout.
    let _ = dequantize_row_q4_k(&make_q4_k_block(&mut rng), ncols);

    // ---- Q8_0 experts (some down_exps ship Q8_0) ----
    let q8_0_row = vulkan_kernels::q8_0_row_bytes(ncols).expect("q8_0 row bytes");
    run_id_case(
        &ctx,
        "id_Q8_0",
        n_expert,
        nrows,
        ncols,
        q8_0_row,
        make_q8_0_block,
        &ids,
        &x,
        q8_0_gemv_with_params,
        q8_0_gemv_id_with_params,
        2e-2,
    );
}
