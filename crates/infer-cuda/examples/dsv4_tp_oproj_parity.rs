//! DSv4 TP attention-output sharding numeric-parity gate: the head o-slice,
//! the grouped low-rank O-LoRA projection (both H20-default DeepGEMM and the
//! scalar fallback) and the TP partial-sum.
//!
//! Standalone kernel harness — NO engine, NO serve, NO NCCL. DSv4-Flash
//! geometry from /tmp/arle-dsv4-flash-config.json: 64 attention heads ×
//! head_dim 512 (W = 32768), residual hidden H = 4096, o_groups G = 8,
//! o_lora_rank C = 1024. Checkpoint dtype (read from the safetensors header):
//! wo_a [8192,4096] E4M3 with [64,32] E8M0 block scales, wo_b [4096,8192]
//! with [32,64]. Attention output v [S, W] reaches residual hidden through:
//!
//! 1. `dsv4_tp_out_slice_cuda` — rank r copies its [S, W/tp] head block at
//!    head_offset r·W/tp (attention.rs:1702 prefill, attention/flashmla.rs:1100
//!    decode). Checked BIT-EXACT.
//! 2. The sharded O-LoRA `wo_a` (Column{0}: rank owns G/tp global groups),
//!    two lanes on the same weights:
//!    - **DeepGEMM (the H20 default)**: g>1 per-group
//!      `dsv4_oproj_group_gather_cuda` → per-128-block activation pack-quantize
//!      (`dsv4_deepgemm_pack_quantize_bf16_to_fp8`, expert 0/offset 0) →
//!      `dsv4_deepgemm_fp8_gemm_nt` with the per-group resident
//!      `Dsv4Fp8DeepGemmWeightCache` (built as load.rs:1716-1731) →
//!      `dsv4_oproj_group_scatter_cuda` (production
//!      `dsv4_wo_a_grouped_deepgemm_decode` attention.rs:5040 / `_prefill`
//!      :5186); g==1 (TP=8) skips gather/scatter and takes the contiguous
//!      `decode_proj_deepgemm`/`prefill_proj_deepgemm` lane (:5300/:5335).
//!    - **scalar fallback (non-sm90 / failed native DeepGEMM preflight)**:
//!      g>1 `dsv4_fp8_route_gemv_batch_cuda` (route t·g+lg, attention.rs:4942),
//!      g==1 `dsv4_fp8_gemv_batch_cuda`.
//! 3. `wo_b` [H, G·C], Row{dim1}: DeepGEMM (`wo_b_deepgemm`,
//!    attention.rs:5401/5413) or the scalar FP8 GEMV; rank bf16 partial. The
//!    NCCL all-reduce is a host sum of bf16 rank partials here.
//!
//! DeepGEMM's GEMM is vendor code; the gate covers our pieces around it — the
//! gather/scatter staging, the per-block activation quantization, the
//! E8M0→FP32 resident cache layout and the rank→group mapping. Both lanes
//! share one f64 oracle (the DeepGEMM oracle prequantizes each activation
//! operand to e4m3 per 128 block like the pack kernel), every output column,
//! TP in {1,2,4,8}, decode S=1/8 and one S=32 prefill chunk.
//!
//! Default lane (not inferred): `dsv4_fused_wqkv_decode_enabled()` is
//! `has_deepgemm_native()`, default ON (attention.rs:892-897); the
//! decode-alloc gate load.rs:1715 builds the grouped caches; mla_oproj
//! :5290-5299/:5393 selects DeepGEMM whenever the native preflight passes on
//! sm90. This example runs the same probe: the DeepGEMM families HARD-RUN
//! when it returns true (the H20 batch), and print SKIP on a non-native host
//! while the scalar families still gate.
//!
//! `--negative-control` runs one sabotage per family and asserts ONLY that
//! family trips: shifted head o-slice (TP>1), scalar route-table off-by-one,
//! DeepGEMM gather-offset (g>1), DeepGEMM cache→group mis-map, dropped TP
//! rank (TP>1), and one corrupted wo_b weight scale per lane. Prints
//! NEGATIVE CONTROL OK, exit 0; teeth are internal.
//!
//! Run on a pod:
//! `INFER_CUDA_DEVICE=<free-gpu> target/release/examples/dsv4_tp_oproj_parity`

fn main() -> anyhow::Result<()> {
    if std::env::args().nth(1).as_deref() == Some("--kernel-build-id") {
        println!("{}", cuda_kernels::KERNEL_BUILD_ID);
        return Ok(());
    }
    let negative = std::env::args().any(|a| a == "--negative-control");
    real::run(negative)
}

#[cfg(not(feature = "cuda"))]
mod real {
    pub(super) fn run(_negative: bool) -> anyhow::Result<()> {
        eprintln!("dsv4_tp_oproj_parity is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
#[path = "support/attn_common.rs"]
#[allow(dead_code)] // shared gate helper; this family uses Rng/bf16/e4m3 only
mod attn_common;

#[cfg(feature = "cuda")]
mod real {
    use super::attn_common::{Rng, bf, e4m3_decode, e4m3_encode};
    use anyhow::{Result, ensure};
    use cuda_kernels::attention::{
        dsv4_oproj_group_gather_raw, dsv4_oproj_group_scatter_raw, dsv4_tp_out_slice_raw,
    };
    use cuda_kernels::moe::{dsv4_deepgemm_fp8_gemm_nt, dsv4_deepgemm_pack_quantize_bf16_to_fp8};
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::quant_linear::{
        Dsv4RouteGemvArgs, dsv4_fp8_gemv_batch, dsv4_fp8_route_gemv_batch,
    };
    use cuda_kernels::tensor::{DeviceMatrix, Dsv4Fp8DeepGemmWeightCache, RawDevicePtr};
    use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
    use half::bf16;

    const HEADS: usize = 64;
    const HEAD_D: usize = 512;
    const W: usize = HEADS * HEAD_D; // 32768 global attention-output width
    const H: usize = 4096; // residual hidden
    const G: usize = 8; // o_groups
    const C: usize = 1024; // o_lora_rank
    const WG: usize = H; // per-group wo_a input width
    const BLK: usize = 128; // FP8 scale block (m, n, k)
    const E4M3_MAX: f64 = 448.0;
    const DG_STRIDE: usize = 128; // activation scale_m stride (production scratch)
    const TPS: &[usize] = &[1, 2, 4, 8];
    const DECODE_B: &[usize] = &[1, 8];
    const PREFILL_S: usize = 32;
    const SEED: u64 = 0xD5D4_0754_C0DE_0036;

    // Scalar lane: one fp8 weight operand, bf16 activation (dsv4_decode_moe
    // bounds). DeepGEMM lane: activation is also e4m3 per 128 block, so its
    // bound is looser; the <=8-way bf16 TP sum uses the loose bound too.
    const LAT_REL_L2: f64 = 7e-2;
    const LAT_SLOPE: f64 = 1e-1;
    const LAT_FLOOR: f64 = 2.5e-2;
    const DG_REL_L2: f64 = 9e-2;
    const DG_SLOPE: f64 = 1.4e-1;
    const DG_FLOOR: f64 = 3.5e-2;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Sabotage {
        None,
        Slice,
        GemvGroupMap,
        DgGather,
        DgCacheMap,
        GemvWbScale,
        DgWbScale,
    }

    fn e8m0_decode(code: u8) -> f64 {
        2f64.powi((code as i32) - 127)
    }

    /// FP8 payload bytes upload as `i8` (DSv4 wrappers take `DevicePtr<i8>`).
    fn as_i8(bytes: &[u8]) -> &[i8] {
        // SAFETY: u8/i8 are the same one-byte layout.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, bytes.len()) }
    }

    fn bound(got: &[f64], want: &[f64], rel_cap: f64, floor: f64, slope: f64) -> (bool, f64) {
        let mut viol = 0usize;
        let (mut d2, mut r2) = (0f64, 0f64);
        for (g, w) in got.iter().zip(want) {
            let d = g - w;
            d2 += d * d;
            r2 += w * w;
            if d.abs() > floor + slope * w.abs() {
                viol += 1;
            }
        }
        let rel = (d2 / r2.max(1e-12)).sqrt();
        (rel < rel_cap && viol == 0, rel)
    }

    /// Per-128-block activation pack-quantization, f64 mirror of
    /// dsv4_deepgemm_pack_quantize_bf16_to_fp8_kernel (amax/448 per block).
    fn quantize_blocks(x: &[f64]) -> Vec<f64> {
        let mut out = vec![0f64; x.len()];
        for block in 0..x.len().div_ceil(BLK) {
            let lo = block * BLK;
            let hi = (lo + BLK).min(x.len());
            let amax = x[lo..hi].iter().fold(0f64, |a, v| a.max(v.abs()));
            let scale = if amax > 0.0 { amax / E4M3_MAX } else { 1.0 };
            for (i, v) in x[lo..hi].iter().enumerate() {
                out[lo + i] = f64::from(e4m3_encode((v / scale) as f32));
            }
        }
        out
    }

    /// Global FP8 weights shared by every TP run of the process.
    struct Global {
        qa: Vec<u8>,   // wo_a e4m3 [G*C, WG]
        qa_s: Vec<u8>, // wo_a e8m0 [G*C/BLK, WG/BLK]
        qb: Vec<u8>,   // wo_b e4m3 [H, G*C]
        qb_s: Vec<u8>, // wo_b e8m0 [H/BLK, G*C/BLK]
    }

    impl Global {
        fn build() -> Self {
            let mut rng = Rng::new(SEED);
            // N(0,0.25) e4m3 weights; per-block e8m0 powers of two in 2^-3..2^1.
            let qa = (0..G * C * WG)
                .map(|_| e4m3_encode(rng.normal() * 0.25))
                .collect();
            let qa_s = (0..(G * C / BLK) * (WG / BLK))
                .map(|_| 127 - 3 + (rng.next_u64() % 5) as u8)
                .collect();
            let qb = (0..H * G * C)
                .map(|_| e4m3_encode(rng.normal() * 0.25))
                .collect();
            let qb_s = (0..(H / BLK) * (G * C / BLK))
                .map(|_| 127 - 3 + (rng.next_u64() % 5) as u8)
                .collect();
            Self { qa, qa_s, qb, qb_s }
        }

        fn group_bytes(&self, gr: usize) -> (&[u8], &[u8]) {
            let n = (C / BLK) * (WG / BLK);
            (
                &self.qa[gr * C * WG..(gr + 1) * C * WG],
                &self.qa_s[gr * n..(gr + 1) * n],
            )
        }

        /// `groups` contiguous global group blocks starting at g0.
        fn qa_shard(&self, g0: usize, groups: usize) -> (Vec<u8>, Vec<u8>) {
            let mut w = Vec::with_capacity(groups * C * WG);
            let mut s = Vec::with_capacity(groups * (C / BLK) * (WG / BLK));
            for gr in g0..g0 + groups {
                let (gw, gs) = self.group_bytes(gr);
                w.extend_from_slice(gw);
                s.extend_from_slice(gs);
            }
            (w, s)
        }

        /// Rank wo_b column shard [H, lc] + scale shard [H/BLK, lc/BLK];
        /// `scale_bump` flips the first e8m0 byte (×8, negative tooth).
        fn qb_shard(&self, c0: usize, lc: usize, scale_bump: bool) -> (Vec<u8>, Vec<u8>) {
            let mut w = vec![0u8; H * lc];
            for row in 0..H {
                w[row * lc..(row + 1) * lc]
                    .copy_from_slice(&self.qb[row * G * C + c0..row * G * C + c0 + lc]);
            }
            let s_cols = G * C / BLK;
            let l_cols = lc / BLK;
            let mut s = vec![0u8; (H / BLK) * l_cols];
            for sr in 0..H / BLK {
                s[sr * l_cols..(sr + 1) * l_cols].copy_from_slice(
                    &self.qb_s[sr * s_cols + c0 / BLK..sr * s_cols + c0 / BLK + l_cols],
                );
            }
            if scale_bump {
                s[0] = s[0].wrapping_add(3);
            }
            (w, s)
        }
    }

    struct Oracle {
        /// Per-rank bf16-rounded wo_a latents; dg variant uses e4m3 inputs.
        latents: Vec<Vec<f64>>,
        latents_dg: Vec<Vec<f64>>,
        /// TP sums of bf16 wo_b partials.
        out: Vec<f64>,
        out_dg: Vec<f64>,
    }

    #[allow(clippy::needless_range_loop)] // c/k index the same weight in multiple arrays
    fn wo_a_group(g: &Global, xq: &[f64], t: usize, gr: usize) -> Vec<f64> {
        let mut out = vec![0f64; C];
        for c in 0..C {
            let mut acc = 0f64;
            for k in 0..WG {
                let qb = g.qa[(gr * C + c) * WG + k];
                let scale = e8m0_decode(g.qa_s[(gr * C / BLK + c / BLK) * (WG / BLK) + k / BLK]);
                acc += f64::from(e4m3_decode(qb)) * scale * xq[t * W + gr * WG + k];
            }
            out[c] = f64::from(bf(acc as f32));
        }
        out
    }

    fn oracle(g: &Global, v: &[bf16], s: usize, tp: usize) -> Oracle {
        let groups = G / tp;
        let vf: Vec<f64> = v.iter().map(|x| f64::from(*x)).collect();
        let vq = quantize_blocks(&vf);
        let mut lat = vec![0f64; s * G * C];
        let mut lat_dg = vec![0f64; s * G * C];
        for t in 0..s {
            for gr in 0..G {
                let base = t * G * C + gr * C;
                lat[base..base + C].copy_from_slice(&wo_a_group(g, &vf, t, gr));
                lat_dg[base..base + C].copy_from_slice(&wo_a_group(g, &vq, t, gr));
            }
        }
        let mut rank_lat = Vec::with_capacity(tp);
        let mut rank_lat_dg = Vec::with_capacity(tp);
        for r in 0..tp {
            let g0 = r * groups;
            for (global, dst) in [(&lat, &mut rank_lat), (&lat_dg, &mut rank_lat_dg)] {
                let mut local = vec![0f64; s * groups * C];
                for t in 0..s {
                    for lg in 0..groups {
                        let gr = g0 + lg;
                        local[t * groups * C + lg * C..t * groups * C + (lg + 1) * C]
                            .copy_from_slice(&global[t * G * C + gr * C..t * G * C + (gr + 1) * C]);
                    }
                }
                dst.push(local);
            }
        }
        let mut out = vec![0f64; s * H];
        let mut out_dg = vec![0f64; s * H];
        for r in 0..tp {
            let g0 = r * groups;
            let lc = groups * C;
            let lq = quantize_blocks(&rank_lat_dg[r]);
            for t in 0..s {
                for row in 0..H {
                    let (mut acc, mut acc_dg) = (0f64, 0f64);
                    for j in 0..lc {
                        let qb = g.qb[row * G * C + g0 * C + j];
                        let scale =
                            e8m0_decode(g.qb_s[(row / BLK) * (G * C / BLK) + (g0 * C + j) / BLK]);
                        let wv = f64::from(e4m3_decode(qb)) * scale;
                        acc += wv * rank_lat[r][t * lc + j];
                        acc_dg += wv * lq[t * lc + j];
                    }
                    out[t * H + row] += f64::from(bf(acc as f32));
                    out_dg[t * H + row] += f64::from(bf(acc_dg as f32));
                }
            }
        }
        Oracle {
            latents: rank_lat,
            latents_dg: rank_lat_dg,
            out,
            out_dg,
        }
    }

    /// Per-token activation pack-quantize + dense DeepGEMM scratch. Layout
    /// follows the production Dsv4FusedWqkvDecodeScratch
    /// (attention/flashmla.rs:1336-1365): stride_m 128, one active expert 0.
    struct DgScratch {
        input_fp8: CudaSlice<u8>,
        scales: CudaSlice<f32>,
        active_experts: CudaSlice<i32>,
        active_offsets: CudaSlice<i32>,
        active_counts: CudaSlice<i32>,
    }

    impl DgScratch {
        fn new(ctx: &DeviceContext, m_max: usize, k_max: usize) -> Result<Self> {
            Ok(Self {
                input_fp8: ctx.stream.alloc_zeros::<u8>(m_max * k_max)?,
                scales: ctx
                    .stream
                    .alloc_zeros::<f32>(DG_STRIDE * k_max.div_ceil(BLK))?,
                active_experts: ctx.stream.clone_htod(&[0_i32])?,
                active_offsets: ctx.stream.clone_htod(&[0_i32])?,
                active_counts: ctx.stream.clone_htod(&[m_max as i32])?,
            })
        }

        /// Quantize [m,k] input then dense DeepGEMM -> [m,n] bf16.
        fn gemm(
            &mut self,
            ctx: &DeviceContext,
            cache: &Dsv4Fp8DeepGemmWeightCache,
            input: &CudaSlice<bf16>,
            out: &mut CudaSlice<bf16>,
            m: usize,
        ) -> Result<()> {
            ctx.stream
                .memcpy_htod(&[m as i32], &mut self.active_counts)?;
            let stream = ctx.stream.cu_stream();
            let (ip, _) = input.device_ptr(&ctx.stream);
            let (fp, _) = self.input_fp8.device_ptr(&ctx.stream);
            let (sp, _) = self.scales.device_ptr(&ctx.stream);
            let (ep, _) = self.active_experts.device_ptr(&ctx.stream);
            let (op, _) = self.active_offsets.device_ptr(&ctx.stream);
            let (cp, _) = self.active_counts.device_ptr(&ctx.stream);
            let k = cache.cols;
            // SAFETY: scratch covers m*k; cache is [n,k]; expert 0 offset 0.
            unsafe {
                dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                    RawDevicePtr::from_raw(ip),
                    RawDevicePtr::from_raw(fp),
                    RawDevicePtr::from_raw(sp),
                    RawDevicePtr::from_raw(ep),
                    RawDevicePtr::from_raw(op),
                    RawDevicePtr::from_raw(cp),
                    1,
                    m,
                    k,
                    DG_STRIDE,
                    stream,
                )?;
                let (wp, _) = cache.weight.device_ptr(&ctx.stream);
                let (ws, _) = cache.scales.device_ptr(&ctx.stream);
                let (dst, _) = out.device_ptr_mut(&ctx.stream);
                dsv4_deepgemm_fp8_gemm_nt(
                    RawDevicePtr::from_raw(fp),
                    RawDevicePtr::from_raw(sp),
                    RawDevicePtr::from_raw(wp),
                    RawDevicePtr::from_raw(ws),
                    RawDevicePtr::from_raw(dst),
                    m,
                    cache.rows,
                    k,
                    DG_STRIDE,
                    stream,
                )?;
            }
            Ok(())
        }
    }

    /// Resident DeepGEMM cache for global group `gr` (built the same way
    /// load.rs builds wo_a_group_deepgemm: DeviceMatrix → row-range cache).
    fn group_cache(
        ctx: &DeviceContext,
        g: &Global,
        gr: usize,
    ) -> Result<Dsv4Fp8DeepGemmWeightCache> {
        let (w, s) = g.qa_shard(gr, 1);
        let mat = DeviceMatrix::from_dsv4_fp8_block_scaled(ctx, &w, &s, C, WG, C / BLK, WG / BLK)?;
        Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight(ctx, &mat)
    }

    /// One rank's run on one lane. Returns host f64 (head slice, latent,
    /// wo_b partial).
    #[allow(clippy::too_many_arguments)]
    fn run_rank(
        ctx: &DeviceContext,
        glob: &Global,
        v_full: &CudaSlice<bf16>,
        s: usize,
        tp: usize,
        rank: usize,
        dg: bool,
        sabotage: Sabotage,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let groups = G / tp;
        let g0 = rank * groups;
        let local_w = W / tp;
        let lc = groups * C;

        // Family 1: head o-slice (same for both lanes).
        let mut slice_d = ctx.stream.alloc_zeros::<bf16>(s * local_w)?;
        let head_off = if sabotage == Sabotage::Slice && rank == 0 {
            HEAD_D
        } else {
            rank * local_w
        };
        {
            let (fp, _) = v_full.device_ptr(&ctx.stream);
            let (lp, _) = slice_d.device_ptr_mut(&ctx.stream);
            dsv4_tp_out_slice_raw(
                &ctx.stream,
                fp,
                lp,
                s as i32,
                W as i32,
                local_w as i32,
                head_off as i32,
            )?;
        }

        let mut lat_d = ctx.stream.alloc_zeros::<bf16>(s * lc)?;

        if dg {
            // ---- DeepGEMM lane (H20 default) ----
            let mut scr = DgScratch::new(ctx, s, WG)?;
            if groups == 1 {
                // TP=8: contiguous GEMM; the cache-map tooth points at the
                // next global group's cache.
                let gr = if sabotage == Sabotage::DgCacheMap && rank == 0 {
                    (g0 + 1) % G
                } else {
                    g0
                };
                let cache = group_cache(ctx, glob, gr)?;
                scr.gemm(ctx, &cache, &slice_d, &mut lat_d, s)?;
            } else {
                let mut in_g = ctx.stream.alloc_zeros::<bf16>(s * WG)?;
                let mut out_g = ctx.stream.alloc_zeros::<bf16>(s * C)?;
                for lg in 0..groups {
                    // DgGather tooth gathers local group lg+1's columns;
                    // DgCacheMap tooth uses local group lg+1's cache.
                    let gather_g = if sabotage == Sabotage::DgGather && rank == 0 {
                        (lg + 1) % groups
                    } else {
                        lg
                    };
                    let cache_g = if sabotage == Sabotage::DgCacheMap && rank == 0 {
                        g0 + (lg + 1) % groups
                    } else {
                        g0 + lg
                    };
                    {
                        let (srcp, _) = slice_d.device_ptr(&ctx.stream);
                        let (dstp, _) = in_g.device_ptr_mut(&ctx.stream);
                        dsv4_oproj_group_gather_raw(
                            &ctx.stream,
                            srcp,
                            dstp,
                            s as i32,
                            groups as i32,
                            WG as i32,
                            gather_g as i32,
                        )?;
                    }
                    let cache = group_cache(ctx, glob, cache_g)?;
                    scr.gemm(ctx, &cache, &in_g, &mut out_g, s)?;
                    {
                        let (ogp, _) = out_g.device_ptr(&ctx.stream);
                        let (dstp, _) = lat_d.device_ptr_mut(&ctx.stream);
                        dsv4_oproj_group_scatter_raw(
                            &ctx.stream,
                            ogp,
                            dstp,
                            s as i32,
                            groups as i32,
                            C as i32,
                            lg as i32,
                        )?;
                    }
                }
            }
            // wo_b DeepGEMM.
            let (wb, wb_s) =
                glob.qb_shard(g0 * C, lc, sabotage == Sabotage::DgWbScale && rank == 0);
            let wb_mat = DeviceMatrix::from_dsv4_fp8_block_scaled(
                ctx,
                &wb,
                &wb_s,
                H,
                lc,
                H / BLK,
                lc / BLK,
            )?;
            let wb_cache = Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight(ctx, &wb_mat)?;
            let mut scrb = DgScratch::new(ctx, s, lc)?;
            let mut y_d = ctx.stream.alloc_zeros::<bf16>(s * H)?;
            scrb.gemm(ctx, &wb_cache, &lat_d, &mut y_d, s)?;
            ctx.sync()?;
            let to_f64 = |x: &CudaSlice<bf16>| {
                ctx.stream
                    .clone_dtoh(x)
                    .map(|v| v.iter().map(|x| f64::from(*x)).collect::<Vec<_>>())
            };
            return Ok((to_f64(&slice_d)?, to_f64(&lat_d)?, to_f64(&y_d)?));
        }

        // ---- Scalar fallback lane ----
        let mut w_ptrs = Vec::with_capacity(groups);
        let mut s_ptrs = Vec::with_capacity(groups);
        let mut qbufs = Vec::with_capacity(groups);
        let mut sbufs = Vec::with_capacity(groups);
        for lg in 0..groups {
            let gr = if sabotage == Sabotage::GemvGroupMap && rank == 0 {
                if groups == 1 {
                    (g0 + 1) % G
                } else {
                    g0 + (lg + 1) % groups
                }
            } else {
                g0 + lg
            };
            let (qw, qs) = glob.group_bytes(gr);
            let qd = ctx.stream.clone_htod(as_i8(qw))?;
            let sd = ctx.stream.clone_htod(qs)?;
            w_ptrs.push(qd.device_ptr(&ctx.stream).0);
            s_ptrs.push(sd.device_ptr(&ctx.stream).0);
            qbufs.push(qd);
            sbufs.push(sd);
        }
        let w_tbl = ctx.stream.clone_htod(&w_ptrs)?;
        let s_tbl = ctx.stream.clone_htod(&s_ptrs)?;
        if groups == 1 {
            dsv4_fp8_gemv_batch(
                ctx,
                &qbufs[0],
                &sbufs[0],
                &slice_d,
                &mut lat_d,
                s,
                C,
                WG,
                C / BLK,
                WG / BLK,
            )?;
        } else {
            dsv4_fp8_route_gemv_batch(
                ctx,
                &w_tbl,
                &s_tbl,
                &slice_d,
                &mut lat_d,
                Dsv4RouteGemvArgs {
                    route_meta: None,
                    local_expert_start: 0,
                    experts_per_rank: groups,
                    num_routes: s * groups,
                    n: C,
                    k: WG,
                    scale_rows: C / BLK,
                    scale_cols: WG / BLK,
                    apply_route_weight: false,
                },
            )?;
        }
        let (wb, wb_s) = glob.qb_shard(g0 * C, lc, sabotage == Sabotage::GemvWbScale && rank == 0);
        let wb_d = ctx.stream.clone_htod(as_i8(&wb))?;
        let wb_s_d = ctx.stream.clone_htod(&wb_s)?;
        let mut y_d = ctx.stream.alloc_zeros::<bf16>(s * H)?;
        dsv4_fp8_gemv_batch(
            ctx,
            &wb_d,
            &wb_s_d,
            &lat_d,
            &mut y_d,
            s,
            H,
            lc,
            H / BLK,
            lc / BLK,
        )?;
        ctx.sync()?;
        let to_f64 = |x: &CudaSlice<bf16>| {
            ctx.stream
                .clone_dtoh(x)
                .map(|v| v.iter().map(|x| f64::from(*x)).collect::<Vec<_>>())
        };
        Ok((to_f64(&slice_d)?, to_f64(&lat_d)?, to_f64(&y_d)?))
    }

    fn want_slice(v: &[bf16], s: usize, tp: usize, rank: usize) -> Vec<f64> {
        let local_w = W / tp;
        let off = rank * local_w;
        let mut out = vec![0f64; s * local_w];
        for t in 0..s {
            for j in 0..local_w {
                out[t * local_w + j] = f64::from(v[t * W + off + j]);
            }
        }
        out
    }

    /// Sum one lane's rank partials over `take`.
    fn reduce(
        ctx: &DeviceContext,
        glob: &Global,
        v_d: &CudaSlice<bf16>,
        s: usize,
        tp: usize,
        dg: bool,
        take: &[usize],
        sabotage: Sabotage,
    ) -> Result<Vec<f64>> {
        let mut sum = vec![0f64; s * H];
        for &r in take {
            let sb = if r == 0 { sabotage } else { Sabotage::None };
            let (_, _, ya) = run_rank(ctx, glob, v_d, s, tp, r, dg, sb)?;
            for j in 0..s * H {
                sum[j] += ya[j];
            }
        }
        Ok(sum)
    }

    #[allow(clippy::too_many_arguments)]
    fn check_lane(
        ctx: &DeviceContext,
        glob: &Global,
        v_d: &CudaSlice<bf16>,
        s: usize,
        tp: usize,
        dg: bool,
        orc: &Oracle,
        negative: bool,
    ) -> Result<()> {
        let tag = if dg { "deepgemm" } else { "gemv" };
        let label = format!("tp={tp} s={s} {tag}");
        let (lat_b, out_b) = if dg {
            (
                (DG_REL_L2, DG_FLOOR, DG_SLOPE),
                (DG_REL_L2, DG_FLOOR, DG_SLOPE),
            )
        } else {
            (
                (LAT_REL_L2, LAT_FLOOR, LAT_SLOPE),
                (LAT_REL_L2, LAT_FLOOR, LAT_SLOPE),
            )
        };
        let latents_want = if dg { &orc.latents_dg } else { &orc.latents };
        let out_want = if dg { &orc.out_dg } else { &orc.out };

        let mut sum = vec![0f64; s * H];
        let mut lat_ok = true;
        for (r, want) in latents_want.iter().enumerate() {
            let (_, la, ya) = run_rank(ctx, glob, v_d, s, tp, r, dg, Sabotage::None)?;
            let (ok, rel) = bound(&la, want, lat_b.0, lat_b.1, lat_b.2);
            if !ok {
                eprintln!("  [{label}] rank={r} latent l2={rel:.3e}");
            }
            lat_ok &= ok;
            for j in 0..s * H {
                sum[j] += ya[j];
            }
        }
        let (out_ok, orel) = bound(&sum, out_want, out_b.0, out_b.1, out_b.2);
        if !negative {
            ensure!(lat_ok, "{label}: wo_a latent FAILED");
            ensure!(out_ok, "{label}: TP-summed output FAILED (l2={orel:.3e})");
            eprintln!("[{label}] PASS latent+partial-sum in bound (l2={orel:.3e})");
            return Ok(());
        }
        ensure!(lat_ok && out_ok, "{label}: baseline not green before teeth");

        let (map_sabotage, map_name) = if dg {
            (Sabotage::DgCacheMap, "dg cache-map")
        } else {
            (Sabotage::GemvGroupMap, "gemv group-map")
        };
        let (wb_sabotage, wb_name) = if dg {
            (Sabotage::DgWbScale, "dg wb-scale")
        } else {
            (Sabotage::GemvWbScale, "gemv wb-scale")
        };

        // Gather-offset tooth exists only with g>1 (TP<8) on the DeepGEMM lane.
        if dg && tp < 8 {
            let (_, la_bad, _) = run_rank(ctx, glob, v_d, s, tp, 0, true, Sabotage::DgGather)?;
            let (ok, rel) = bound(&la_bad, &latents_want[0], lat_b.0, lat_b.1, lat_b.2);
            ensure!(!ok, "{label}: dg gather-offset tooth dead (l2={rel:.3e})");
        }
        let (_, la_bad, _) = run_rank(ctx, glob, v_d, s, tp, 0, dg, map_sabotage)?;
        let (map_ok, map_rel) = bound(&la_bad, &latents_want[0], lat_b.0, lat_b.1, lat_b.2);
        ensure!(!map_ok, "{label}: {map_name} tooth dead (l2={map_rel:.3e})");

        if tp > 1 {
            let take: Vec<usize> = (1..tp).collect();
            let drop_sum = reduce(ctx, glob, v_d, s, tp, dg, &take, Sabotage::None)?;
            let (drop_ok, drop_rel) = bound(&drop_sum, out_want, out_b.0, out_b.1, out_b.2);
            ensure!(
                !drop_ok,
                "{label}: dropped-rank tooth dead (l2={drop_rel:.3e})"
            );
        }

        // wo_b scale tooth: latent unchanged, final sum wrong.
        let (_, la_good, y_bad) = run_rank(ctx, glob, v_d, s, tp, 0, dg, wb_sabotage)?;
        let (lat_still, _) = bound(&la_good, &latents_want[0], lat_b.0, lat_b.1, lat_b.2);
        ensure!(
            lat_still,
            "{label}: {wb_name} tooth leaked into wo_a latent"
        );
        let mut scale_sum = y_bad;
        for r in 1..tp {
            let (_, _, ya) = run_rank(ctx, glob, v_d, s, tp, r, dg, Sabotage::None)?;
            for j in 0..s * H {
                scale_sum[j] += ya[j];
            }
        }
        let (scale_ok, scale_rel) = bound(&scale_sum, out_want, out_b.0, out_b.1, out_b.2);
        ensure!(
            !scale_ok,
            "{label}: {wb_name} tooth dead (l2={scale_rel:.3e})"
        );
        eprintln!(
            "[{label}] teeth OK: {}{map_name}, dropped-rank, {wb_name}",
            if dg && tp < 8 {
                "dg gather-offset, "
            } else {
                ""
            }
        );
        Ok(())
    }

    /// Returns true when the DeepGEMM families ran for this case.
    #[allow(clippy::too_many_arguments)]
    fn run_case(
        ctx: &DeviceContext,
        glob: &Global,
        v: &[bf16],
        s: usize,
        tp: usize,
        native: bool,
        negative: bool,
    ) -> Result<bool> {
        let v_d = ctx.stream.clone_htod(v)?;
        let orc = oracle(glob, v, s, tp);

        // Family 1: head o-slice bit-exact.
        let mut slice_ok = true;
        for r in 0..tp {
            let (sl, _, _) = run_rank(ctx, glob, &v_d, s, tp, r, false, Sabotage::None)?;
            slice_ok &= sl == want_slice(v, s, tp, r);
        }
        if negative {
            ensure!(
                slice_ok,
                "tp={tp} s={s}: baseline slice not exact before teeth"
            );
            if tp > 1 {
                let (sl, _, _) = run_rank(ctx, glob, &v_d, s, tp, 0, false, Sabotage::Slice)?;
                ensure!(
                    sl != want_slice(v, s, tp, 0),
                    "tp={tp} s={s}: slice tooth dead"
                );
            }
        } else {
            ensure!(slice_ok, "tp={tp} s={s}: head o-slice FAILED");
            eprintln!("[tp={tp} s={s}] PASS head o-slice exact");
        }

        check_lane(ctx, glob, &v_d, s, tp, false, &orc, negative)?;
        if native {
            check_lane(ctx, glob, &v_d, s, tp, true, &orc, negative)?;
        } else {
            eprintln!("[tp={tp} s={s}] SKIP deepgemm lane (has_deepgemm_native()=false)");
        }
        Ok(native)
    }

    pub(super) fn run(negative: bool) -> Result<()> {
        let ctx = DeviceContext::new()?;
        let native = cuda_kernels::has_deepgemm_native();
        eprintln!(
            "[dsv4-tp-oproj-parity] device={} build={} deepgemm_native={native}{}",
            ctx.ordinal(),
            cuda_kernels::KERNEL_BUILD_ID,
            if negative { " NEGATIVE-CONTROL" } else { "" }
        );
        let glob = Global::build();
        let cases: Vec<usize> = DECODE_B
            .iter()
            .copied()
            .chain(std::iter::once(PREFILL_S))
            .collect();
        let total_cases = TPS.len() * cases.len();
        let mut dg_ran = 0usize;
        for &tp in TPS {
            for &s in &cases {
                let mut rng = Rng::new(SEED ^ ((s as u64) << 20) ^ ((tp as u64) << 32));
                let v: Vec<bf16> = (0..s * W).map(|_| bf(rng.normal() * 0.3)).collect();
                let ran = run_case(&ctx, &glob, &v, s, tp, native, negative)?;
                if ran {
                    dg_ran += 1;
                }
            }
        }
        // One explicit coverage line so the batch log proves whether the H20
        // default lane ran. Per-case SKIP messages use the "[tp=..] SKIP …"
        // form (not the gate-wide `^SKIP: ` protocol): this is a per-family
        // skip, never a whole-gate skip.
        if native {
            eprintln!("[dsv4-tp-oproj-parity] deepgemm families: ran {dg_ran}/{total_cases}");
        } else {
            eprintln!(
                "[dsv4-tp-oproj-parity] deepgemm families: ran 0/{total_cases} (native preflight false; scalar lane gated)"
            );
        }
        if negative {
            eprintln!("[dsv4-tp-oproj-parity] NEGATIVE CONTROL OK");
        } else {
            eprintln!("[dsv4-tp-oproj-parity] ALL PASS");
        }
        Ok(())
    }
}
