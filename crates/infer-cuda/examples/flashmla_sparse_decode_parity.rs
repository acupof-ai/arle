//! FlashMLA SM90 sparse-decode numeric-parity gate for the DSv4-Flash MODEL1
//! CSA (compress_ratio=4) single-row decode path.
//!
//! Standalone harness — NO engine, NO model. Drives the same public
//! `cuda_kernels::attention` wrappers production drives, at the
//! DeepSeek-V4-Flash-0731 decode geometry: h_q=64, d_qk=d_v=512, packed
//! latent KV 584 B/token (448 FP8-E4M3 NoPE in 64-elem tiles, 64 BF16 RoPE
//! dims, 8 E8M0 per-tile scales), page block 64, sliding_window 128,
//! index_topk 512, topk_unified 640.
//!
//! Compared families, each with its own verdict and negative control:
//!   - indices — production `..._build_indices_start_pos_ptr` vs a host
//!     replica of the device `index_at` math (SW ring + selected compressed
//!     blocks + causality + -1 masks);
//!   - attention output — sparse decode fwd vs an f32 host reference that
//!     decodes the packed pool bytes itself (E4M3×E8M0 tile scales + BF16
//!     RoPE), gathers only the production index set, and runs masked softmax
//!     + value sum over d_v=512;
//!   - LSE — fwd log-sum-exp vs the reference natural-log LSE;
//!   - scheduler metadata — determinism of `..._sched_meta`/num_splits and
//!     sensitivity to the topk length (the vendor partitioner has no host
//!     reimplementation; see gate_gap in registry.toml).
//!
//! Normal run: every family must pass. Under `--negative-control`, one
//! corruption per family (selected[0] masked for indices/output/LSE; an
//! 8x topk for the scheduler), each family's check must independently
//! FAIL; the run prints NEGATIVE CONTROL OK and exits 0 only if all four
//! fired.
//!
//! sm90 only (H20). Build/run on a pod (pod.sh build rejects --example):
//!   cargo build --release -p infer-cuda --features cuda \
//!     --example flashmla_sparse_decode_parity
//!   target/release/examples/flashmla_sparse_decode_parity
//!   target/release/examples/flashmla_sparse_decode_parity --negative-control

#![allow(clippy::print_stdout, clippy::print_stderr)]

fn main() -> anyhow::Result<()> {
    if std::env::args().nth(1).as_deref() == Some("--kernel-build-id") {
        println!("{}", cuda_kernels::KERNEL_BUILD_ID);
        return Ok(());
    }
    real::run(std::env::args().nth(1).as_deref() == Some("--negative-control"))
}

#[cfg(not(feature = "cuda"))]
mod real {
    pub(super) fn run(_negative: bool) -> anyhow::Result<()> {
        eprintln!(
            "flashmla_sparse_decode_parity is a CUDA/sm90 harness; rebuild with \
             --features cuda."
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::{Result, ensure};
    use cuda_kernels::attention;
    use cuda_kernels::prelude::DeviceContext;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    // DeepSeek-V4-Flash-0731 MODEL1 decode geometry (config.json), consistent
    // with Dsv4FlashMlaDecodeShape::new.
    const H_Q: usize = 64;
    const H_KV: i32 = 1;
    const D_QK: usize = 512; // 448 NoPE + 64 RoPE
    const D_V: usize = 512;
    const HEAD_NOPE: usize = 448;
    const HEAD_ROPE: usize = 64;
    const PAGE: usize = 64;
    const SLIDING_WINDOW: usize = 128; // 2 SW blocks
    const SW_BLOCKS: usize = 2;
    const INDEX_TOPK: usize = 512;
    const COMPRESS_RATIO: usize = 4; // a CSA layer
    const MODE_CSA: i32 = 1;
    const MODEL1: i32 = 1;
    const TOPK_UNIFIED: usize = SLIDING_WINDOW + INDEX_TOPK; // 640
    const START_POS: i32 = 127; // sw_count 128
    // Builder's CSA causality gate: block_end = 4c + 3 <= 127 → c <= 31.
    const COMP_SELECTED: usize = 32; // selected = 0..32
    const COMP_PACKED: usize = 32; // fill compressed block rows 0..31
    const ROW_BYTES: usize = 584;
    const BLOCK_BYTES: usize = PAGE * ROW_BYTES; // 37376
    // Pool blocks 0,1 = SW ring; block 2 = the compressed block.
    const NUM_BLOCKS: i32 = (SW_BLOCKS + 1) as i32;
    const DOMINANT_COMP: usize = 0; // selected entry carrying the dominant key

    const PASS_MAX_REL_OUT: f64 = 0.05;
    const PASS_MAX_ABS_LSE: f64 = 0.20;

    struct Fwd {
        indices: Vec<i32>,
        out: Vec<bf16>,
        lse: Vec<f32>,
    }

    pub(super) fn run(negative: bool) -> Result<()> {
        let ctx = DeviceContext::new()?;
        let cc = ctx.compute_capability();
        eprintln!(
            "[flashmla-sparse-parity] device={} cc={}.{} sms={} build={}",
            ctx.ordinal(),
            cc.0,
            cc.1,
            ctx.sm_count(),
            cuda_kernels::KERNEL_BUILD_ID
        );
        ensure!(cuda_kernels::HAS_FLASHMLA, "binary built without FlashMLA");
        ensure!(
            cc.0 >= 9,
            "FlashMLA sparse decode needs sm90+; got sm_{}{}",
            cc.0,
            cc.1
        );
        ensure!(
            TOPK_UNIFIED.is_multiple_of(128),
            "topk_unified must be a multiple of 128"
        );

        // ── packed pool: production pack kernel from BF16 NoPE/RoPE sources.
        let n_pack = SLIDING_WINDOW + COMP_PACKED;
        let mut nope_src = vec![0f32; n_pack * HEAD_NOPE];
        let mut rope_src = vec![0f32; n_pack * HEAD_ROPE];
        let mut block_ids = vec![0i32; n_pack];
        let mut row_ids = vec![0i32; n_pack];
        for ring in 0..SLIDING_WINDOW {
            block_ids[ring] = (ring / PAGE) as i32;
            row_ids[ring] = (ring % PAGE) as i32;
            fill_random(
                &mut nope_src,
                &mut rope_src,
                ring,
                block_ids[ring],
                row_ids[ring],
            );
        }
        for c in 0..COMP_PACKED {
            let t = SLIDING_WINDOW + c;
            block_ids[t] = (SW_BLOCKS + c / PAGE) as i32;
            row_ids[t] = (c % PAGE) as i32;
            if c == DOMINANT_COMP {
                // Constant 0.5 latent aligned with the constant Q below:
                // score ≈ 512·0.25/√512 ≈ 5.7 vs ≈0 for the random rows, so
                // this key carries ~all of the attention weight.
                nope_src[t * HEAD_NOPE..(t + 1) * HEAD_NOPE].fill(0.5);
                rope_src[t * HEAD_ROPE..(t + 1) * HEAD_ROPE].fill(0.5);
            } else {
                fill_random(&mut nope_src, &mut rope_src, t, block_ids[t], row_ids[t]);
            }
        }
        let nope_dev = ctx.stream.clone_htod(&to_bf16(&nope_src))?;
        let rope_dev = ctx.stream.clone_htod(&to_bf16(&rope_src))?;
        let block_dev = ctx.stream.clone_htod(&block_ids)?;
        let row_dev = ctx.stream.clone_htod(&row_ids)?;
        let mut pool = ctx
            .stream
            .alloc_zeros::<u8>(NUM_BLOCKS as usize * BLOCK_BYTES)?;
        {
            let (nope_ptr, _g1) = nope_dev.device_ptr(&ctx.stream);
            let (rope_ptr, _g2) = rope_dev.device_ptr(&ctx.stream);
            let (pool_ptr, _g3) = pool.device_ptr_mut(&ctx.stream);
            attention::dsv4_fp8_kv_pack_strided_raw(
                &ctx, nope_ptr, rope_ptr, pool_ptr, &block_dev, &row_dev, n_pack, PAGE, HEAD_NOPE,
                HEAD_ROPE, None,
            )?;
        }
        ctx.sync()?;
        let pool_bytes = ctx.stream.clone_dtoh(&pool)?;
        let records = decode_pool(&pool_bytes);

        // Constant Q shared by all 64 heads (h_kv=1).
        let q_dev = ctx
            .stream
            .clone_htod(&vec![bf16::from_f32(0.5); H_Q * D_QK])?;

        // Scheduler tuning triple (host-only vendor call).
        let (num_sm_parts, fixed_overhead, block_size_topk) =
            attention::flashmla_sm90_sparse_decode_get_meta(H_Q as i32, 1, MODEL1)?;
        let parts_max = num_sm_parts.max(256) as usize;

        // ── positive forward.
        let selected = selected_identity();
        let mut sched_good = vec![0i32];
        let mut splits_good = vec![0i32];
        let got = run_fwd(
            &ctx,
            &q_dev,
            &pool,
            &selected,
            num_sm_parts,
            parts_max,
            fixed_overhead,
            block_size_topk,
            &mut sched_good,
            &mut splits_good,
        )?;

        let ref_indices = host_reference_indices();
        // Single split: the split kernel writes LSE without the sink and
        // output with the sink denominator. Multi split: combine folds the
        // sink into both. num_splits[0] is the measured split count.
        let multi_split = splits_good[0] > 1;
        let (ref_out, ref_lse) =
            reference_attention(&records, &ref_indices, multi_split);

        if !negative {
            let idx_ok = got.indices == ref_indices;
            let (max_rel_out, max_abs_lse) = compare(&got, &ref_out, &ref_lse);
            let out_ok = max_rel_out <= PASS_MAX_REL_OUT;
            let lse_ok = max_abs_lse <= PASS_MAX_ABS_LSE;
            // Scheduler: deterministic, nonempty, bounded; rerun to prove it.
            let mut meta2 = vec![-1i32; sched_good.len()];
            let mut splits2 = vec![-1i32; splits_good.len()];
            let _ = run_fwd(
                &ctx,
                &q_dev,
                &pool,
                &selected,
                num_sm_parts,
                parts_max,
                fixed_overhead,
                block_size_topk,
                &mut meta2,
                &mut splits2,
            )?;
            let sched_ok = meta2 == sched_good
                && splits2 == splits_good
                && splits_good[0] >= 1
                && (splits_good[0] as usize) <= num_sm_parts as usize;

            println!("indices_exact={idx_ok}");
            println!("PASS_MAX_REL_out={max_rel_out:.6} (cap {PASS_MAX_REL_OUT}) pass={out_ok}");
            println!("PASS_MAX_ABS_lse={max_abs_lse:.6} (cap {PASS_MAX_ABS_LSE}) pass={lse_ok}");
            println!(
                "scheduler deterministic+bounded pass={sched_ok} (num_splits={}, num_sm_parts={num_sm_parts})",
                splits_good[0]
            );
            ensure!(idx_ok, "indices differ from host index_at replica");
            ensure!(out_ok, "attention output exceeds relative cap");
            ensure!(lse_ok, "LSE exceeds absolute cap");
            ensure!(sched_ok, "scheduler metadata not deterministic/bounded");
            println!("[flashmla-sparse-parity] ALL PASS");
            return Ok(());
        }

        // ── negative control, one corruption per family.
        // Family 1-3: mask the dominant selected entry (-1). The attention
        // SET changes (dominant key removed), so indices, output and LSE all
        // move independently.
        let mut bad_selected = selected.clone();
        bad_selected[DOMINANT_COMP] = -1;
        let mut meta_bad = Vec::new();
        let mut splits_bad = Vec::new();
        let got_neg = run_fwd(
            &ctx,
            &q_dev,
            &pool,
            &bad_selected,
            num_sm_parts,
            parts_max,
            fixed_overhead,
            block_size_topk,
            &mut meta_bad,
            &mut splits_bad,
        )?;
        let neg_indices = got_neg.indices != ref_indices;
        let (ref_out_neg, ref_lse_neg) =
            reference_attention(&records, &ref_indices, splits_bad[0] > 1);
        let (rel_out, abs_lse) = compare(&got_neg, &ref_out_neg, &ref_lse_neg);
        let neg_out = rel_out > PASS_MAX_REL_OUT;
        let neg_lse = abs_lse > PASS_MAX_ABS_LSE;

        // Family 4: build scheduler metadata for half the topk length; the
        // partition (sched_meta / num_splits) must change. No fwd launch —
        // that metadata with the full-length indices is not a legal config.
        // 8x topk (5120 = 80 topk-blocks) must force the partitioner to emit a
        // different split count / tile layout than 640 (10 blocks).
        let (meta_8x, splits_8x) = run_sched_meta(
            &ctx,
            num_sm_parts,
            parts_max,
            fixed_overhead,
            block_size_topk,
            TOPK_UNIFIED * 8,
        )?;
        let neg_sched = meta_8x != sched_good || splits_8x != splits_good;

        println!("negative indices mismatch fired={neg_indices}");
        println!("negative out rel={rel_out:.6} (must exceed {PASS_MAX_REL_OUT}) fired={neg_out}");
        println!("negative lse abs={abs_lse:.6} (must exceed {PASS_MAX_ABS_LSE}) fired={neg_lse}");
        println!(
            "negative scheduler changed fired={neg_sched} (splits {} vs {})",
            splits_8x[0], splits_good[0]
        );
        ensure!(
            neg_indices,
            "negative control did not fail the indices family"
        );
        ensure!(
            neg_out,
            "negative control did not fail the attention-output family"
        );
        ensure!(neg_lse, "negative control did not fail the LSE family");
        ensure!(
            neg_sched,
            "negative control did not fail the scheduler family"
        );
        println!("NEGATIVE CONTROL OK");
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_fwd(
        ctx: &DeviceContext,
        q_dev: &cudarc::driver::CudaSlice<bf16>,
        pool: &cudarc::driver::CudaSlice<u8>,
        selected: &[i32],
        num_sm_parts: i32,
        parts_max: usize,
        fixed_overhead: i32,
        block_size_topk: i32,
        sched_meta_out: &mut Vec<i32>,
        num_splits_out: &mut Vec<i32>,
    ) -> Result<Fwd> {
        let selected_dev = ctx.stream.clone_htod(selected)?;
        let mut indices_dev = ctx.stream.alloc_zeros::<i32>(TOPK_UNIFIED)?;
        let start_dev = ctx.stream.clone_htod(&[START_POS])?;
        let topk_length_dev = ctx.stream.clone_htod(&[TOPK_UNIFIED as i32])?;
        let mut sched_meta = ctx.stream.alloc_zeros::<i32>(parts_max * 8)?;
        let mut num_splits = ctx.stream.alloc_zeros::<i32>(2)?;
        let sink = ctx.stream.alloc_zeros::<f32>(H_Q)?;
        let mut out_dev = ctx.stream.alloc_zeros::<bf16>(H_Q * D_V)?;
        let mut lse_dev = ctx.stream.alloc_zeros::<f32>(H_Q)?;
        let mut lse_accum = ctx.stream.alloc_zeros::<f32>((parts_max + 1) * H_Q)?;
        let mut o_accum = ctx.stream.alloc_zeros::<f32>((parts_max + 1) * H_Q * D_V)?;

        // Production index builder (graph-safe device start_pos variant).
        {
            let (idx_ptr, _gi) = indices_dev.device_ptr_mut(&ctx.stream);
            let (sel_ptr, _gs) = selected_dev.device_ptr(&ctx.stream);
            let (start_ptr, _gp) = start_dev.device_ptr(&ctx.stream);
            attention::dsv4_flashmla_decode_build_indices_start_pos_ptr_raw(
                ctx,
                idx_ptr,
                sel_ptr,
                SW_BLOCKS,
                SLIDING_WINDOW,
                start_ptr,
                INDEX_TOPK,
                COMPRESS_RATIO,
                MODE_CSA,
                PAGE,
                None,
                SW_BLOCKS + 1,
            )?;
        }
        ctx.sync()?;
        let indices = ctx.stream.clone_dtoh(&indices_dev)?;

        // Production scheduler metadata (topk length = TOPK_UNIFIED).
        build_sched_meta(
            ctx,
            &mut sched_meta,
            &mut num_splits,
            num_sm_parts,
            block_size_topk,
            fixed_overhead,
            TOPK_UNIFIED,
        )?;
        ctx.sync()?;
        *sched_meta_out = ctx.stream.clone_dtoh(&sched_meta)?;
        *num_splits_out = ctx.stream.clone_dtoh(&num_splits)?;

        // Production sparse decode.
        {
            let (q_ptr, _gq) = q_dev.device_ptr(&ctx.stream);
            let (pool_ptr, _gk) = pool.device_ptr(&ctx.stream);
            let (idx_ptr, _gidx) = indices_dev.device_ptr(&ctx.stream);
            let (topk_ptr, _glk) = topk_length_dev.device_ptr(&ctx.stream);
            let (sink_ptr, _gsk) = sink.device_ptr(&ctx.stream);
            let (out_ptr, _go) = out_dev.device_ptr_mut(&ctx.stream);
            let (lse_ptr, _gl) = lse_dev.device_ptr_mut(&ctx.stream);
            let (lacc_ptr, _gla) = lse_accum.device_ptr_mut(&ctx.stream);
            let (oacc_ptr, _goa) = o_accum.device_ptr_mut(&ctx.stream);
            let (meta_ptr, _gm) = sched_meta.device_ptr(&ctx.stream);
            let (split_ptr, _gn) = num_splits.device_ptr(&ctx.stream);
            attention::flashmla_sm90_sparse_decode_fwd_raw(
                &ctx.stream,
                q_ptr,
                pool_ptr,
                idx_ptr,
                topk_ptr,
                sink_ptr,
                out_ptr,
                lse_ptr,
                lacc_ptr,
                oacc_ptr,
                meta_ptr,
                split_ptr,
                1,
                1,
                H_Q as i32,
                H_KV,
                D_QK as i32,
                D_V as i32,
                NUM_BLOCKS,
                PAGE as i32,
                TOPK_UNIFIED as i32,
                num_sm_parts,
                MODEL1,
                1.0 / (D_QK as f32).sqrt(),
                (H_Q * D_QK) as i32,
                (H_Q * D_QK) as i32,
                D_QK as i32,
                BLOCK_BYTES as i32,
                ROW_BYTES as i32,
                TOPK_UNIFIED as i32,
                TOPK_UNIFIED as i32,
                H_Q as i32,
                1,
                (H_Q * D_V) as i32,
                (H_Q * D_V) as i32,
                D_V as i32,
                H_Q as i32,
                H_Q as i32,
                (H_Q * D_V) as i32,
                (H_Q * D_V) as i32,
                D_V as i32,
            )?;
        }
        ctx.sync()?;
        Ok(Fwd {
            indices,
            out: ctx.stream.clone_dtoh(&out_dev)?,
            lse: ctx.stream.clone_dtoh(&lse_dev)?,
        })
    }

    /// Allocate scheduler buffers and build metadata at a chosen topk length;
    /// returns host copies. Used by the positive determinism check and the
    /// scheduler negative control (no fwd launch at the altered length —
    /// 320-topk metadata with 640-long indices is not a legal fwd config).
    #[allow(clippy::too_many_arguments)]
    fn run_sched_meta(
        ctx: &DeviceContext,
        num_sm_parts: i32,
        parts_max: usize,
        fixed_overhead: i32,
        block_size_topk: i32,
        topk: usize,
    ) -> Result<(Vec<i32>, Vec<i32>)> {
        let mut sched_meta = ctx.stream.alloc_zeros::<i32>(parts_max * 8)?;
        let mut num_splits = ctx.stream.alloc_zeros::<i32>(2)?;
        build_sched_meta(
            ctx,
            &mut sched_meta,
            &mut num_splits,
            num_sm_parts,
            block_size_topk,
            fixed_overhead,
            topk,
        )?;
        ctx.sync()?;
        Ok((
            ctx.stream.clone_dtoh(&sched_meta)?,
            ctx.stream.clone_dtoh(&num_splits)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_sched_meta(
        ctx: &DeviceContext,
        sched_meta: &mut cudarc::driver::CudaSlice<i32>,
        num_splits: &mut cudarc::driver::CudaSlice<i32>,
        num_sm_parts: i32,
        block_size_topk: i32,
        fixed_overhead: i32,
        topk: usize,
    ) -> Result<()> {
        let topk_input = ctx.stream.clone_htod(&[topk as i32])?;
        let (topk_ptr, _gt) = topk_input.device_ptr(&ctx.stream);
        let (meta_ptr, _gm) = sched_meta.device_ptr_mut(&ctx.stream);
        let (split_ptr, _gn) = num_splits.device_ptr_mut(&ctx.stream);
        attention::flashmla_sm90_sparse_decode_sched_meta_raw(
            &ctx.stream,
            1,
            1,
            block_size_topk,
            fixed_overhead,
            topk as i32,
            0,
            topk_ptr,
            0,
            meta_ptr,
            split_ptr,
            num_sm_parts,
        )
    }

    fn selected_identity() -> Vec<i32> {
        let mut v = vec![-1i32; INDEX_TOPK];
        for (i, slot) in v.iter_mut().enumerate().take(COMP_SELECTED) {
            *slot = i as i32;
        }
        v
    }

    fn to_bf16(v: &[f32]) -> Vec<bf16> {
        v.iter().map(|&x| bf16::from_f32(x)).collect()
    }

    fn fill_random(nope: &mut [f32], rope: &mut [f32], t: usize, block: i32, row: i32) {
        let seed = (block as u64) * 1000 + (row as u64) * 31 + t as u64;
        for (d, x) in nope.iter_mut().enumerate() {
            *x = hash01(seed, d as u64) * 0.1 - 0.05;
        }
        for (d, x) in rope.iter_mut().enumerate() {
            *x = hash01(seed ^ 0x9e37_79b9, d as u64) * 0.1 - 0.05;
        }
    }

    fn hash01(a: u64, b: u64) -> f32 {
        let mut x = a.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ b.wrapping_mul(0x517c_c1cc_9e37_79b9);
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        (x >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Decoded MODEL1 latent record: [448 f32 NoPE][64 f32 RoPE].
    fn decode_pool(bytes: &[u8]) -> Vec<Vec<f32>> {
        let mut records = Vec::with_capacity(NUM_BLOCKS as usize * PAGE);
        for block in 0..NUM_BLOCKS as usize {
            for row in 0..PAGE {
                let mut v = vec![0f32; D_QK];
                let block_base = block * BLOCK_BYTES;
                let data_base = block_base + row * (HEAD_NOPE + HEAD_ROPE * 2);
                for tile in 0..7usize {
                    let e8m0 =
                        bytes[block_base + PAGE * (HEAD_NOPE + HEAD_ROPE * 2) + row * 8 + tile];
                    let scale = if e8m0 == 0 {
                        0.0
                    } else {
                        2f32.powi(i32::from(e8m0) - 127)
                    };
                    for lane in 0..64usize {
                        let d = tile * 64 + lane;
                        v[d] = decode_e4m3(bytes[data_base + d]) * scale;
                    }
                }
                for d in 0..HEAD_ROPE {
                    let lo = u16::from(bytes[data_base + HEAD_NOPE + d * 2]);
                    let hi = u16::from(bytes[data_base + HEAD_NOPE + d * 2 + 1]);
                    v[HEAD_NOPE + d] = bf16::from_bits(lo | (hi << 8)).to_f32();
                }
                records.push(v);
            }
        }
        records
    }

    fn decode_e4m3(byte: u8) -> f32 {
        let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
        let e = i32::from((byte >> 3) & 0x0F);
        let m = f32::from(byte & 0x07);
        let mag = if e == 0 {
            m / 512.0
        } else if e == 15 {
            (1.0 + m / 8.0) * 256.0
        } else {
            (1.0 + m / 8.0) * 2f32.powi(e - 7)
        };
        sign * mag
    }

    /// Host replica of arle_dsv4_flashmla_decode_index_at, band path
    /// (page_table == null), CSA mode.
    fn host_reference_indices() -> Vec<i32> {
        let mut out = vec![-1i32; TOPK_UNIFIED];
        let sw_start = (START_POS - SLIDING_WINDOW as i32 + 1).max(0);
        let sw_count = (START_POS - sw_start + 1) as usize;
        for (tid, slot) in out.iter_mut().enumerate() {
            if tid < sw_count {
                let ring_idx = (sw_start + tid as i32) % SLIDING_WINDOW as i32;
                let block_id = ring_idx / PAGE as i32;
                let row = ring_idx % PAGE as i32;
                if block_id < SW_BLOCKS as i32 {
                    *slot = block_id * PAGE as i32 + row;
                }
            } else if tid < sw_count + INDEX_TOPK {
                let k = (tid - sw_count) as i32;
                if k >= 0 {
                    let block_end = k * COMPRESS_RATIO as i32 + (COMPRESS_RATIO as i32 - 1);
                    if block_end <= START_POS {
                        let abs_block = SW_BLOCKS as i32 + k / PAGE as i32;
                        *slot = abs_block * PAGE as i32 + k % PAGE as i32;
                    }
                }
            }
        }
        out
    }

    /// f32 masked sparse attention over the production index set. Q is the
    /// bf16 constant 0.5; V is the same 512-dim latent K record. The output
    /// always normalizes with the zero sink (denominator sum_exp+1): the
    /// single-split store_o divides by rL+exp2(sink-m), and the multi-split
    /// combine scales accumulators by exp2(local_lse - global_lse) with
    /// global_lse including the same sink. The LSE differs by path: the
    /// single-split kernel writes ln(sum_exp)+m WITHOUT the sink
    /// (`splitkv_mla.cuh` gSoftmaxLse); only the multi-split combine folds
    /// it (`combine.cu` global_lse += log2(1+exp2(...))).
    fn reference_attention(
        records: &[Vec<f32>],
        indices: &[i32],
        multi_split: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let sm_scale = 1.0 / (D_QK as f32).sqrt();
        let valid: Vec<usize> = indices
            .iter()
            .filter_map(|&i| (i >= 0).then_some(i as usize))
            .collect();
        let mut out = vec![0f32; H_Q * D_V];
        let mut lse = vec![0f32; H_Q];
        for h in 0..H_Q {
            let mut scores = Vec::with_capacity(valid.len());
            let mut maxs = f32::NEG_INFINITY;
            for &idx in &valid {
                let dot: f32 = records[idx][..D_QK].iter().map(|&k| 0.5 * k).sum();
                let s = dot * sm_scale;
                scores.push(s);
                maxs = maxs.max(s);
            }
            let sum_exp: f32 = scores.iter().map(|&s| (s - maxs).exp()).sum();
            let out_denom = sum_exp + 1.0; // zero attn_sink, both split paths
            let lse_denom = if multi_split { out_denom } else { sum_exp };
            lse[h] = maxs + lse_denom.ln();
            for (&idx, &s) in valid.iter().zip(&scores) {
                let p = (s - maxs).exp() / out_denom;
                for d in 0..D_V {
                    out[h * D_V + d] += p * records[idx][d];
                }
            }
        }
        (out, lse)
    }

    fn compare(got: &Fwd, ref_out: &[f32], ref_lse: &[f32]) -> (f64, f64) {
        let mut sum_sq = 0f64;
        let mut max_abs = 0f64;
        for (&g, &r) in got.out.iter().zip(ref_out) {
            let err = (g.to_f32() - r) as f64;
            sum_sq += (r as f64) * (r as f64);
            max_abs = max_abs.max(err.abs());
        }
        let rel = max_abs / sum_sq.sqrt().max(f64::MIN_POSITIVE);
        let lse = got
            .lse
            .iter()
            .zip(ref_lse)
            .map(|(&g, &r)| (g - r).abs() as f64)
            .fold(0f64, f64::max);
        (rel, lse)
    }
}
