//! FlashMLA SM90 sparse-decode numeric-parity gate for the DSv4-Flash MODEL1
//! CSA (compress_ratio=4) BATCHED decode path.
//!
//! Standalone harness — NO engine, NO model. Drives the same public
//! `cuda_kernels::attention` wrappers production drives, at the
//! DeepSeek-V4-Flash-0731 decode geometry: h_q=64, d_qk=d_v=512, packed
//! latent KV 584 B/token (448 FP8-E4M3 NoPE in 64-elem tiles, 64 BF16 RoPE
//! dims, 8 E8M0 per-tile scales), page block 64, sliding_window 128,
//! index_topk 512, topk_unified 640.
//!
//! Batched shape B=8 with per-row distinct start_pos (a not-yet-full window,
//! 64-token ring boundaries ±1, and compressed-block boundaries),
//! row-specific selected sets, and discontinuous per-row Stage-B page tables.
//!
//! Compared families, each with its own verdict and negative flag:
//!   - pack — production `dsv4_fp8_kv_pack_strided` output decoded from pool
//!     bytes vs the BF16 sources, within the E4M3×E8M0 half-ULP bound;
//!   - indices — production batched `..._build_indices_batched` vs an oracle
//!     written from address semantics (last-128 positions routed through the
//!     page table, then selected indexer keys, causally gated);
//!   - attention output — batched sparse decode fwd vs an f32 host reference
//!     that decodes the pool bytes and masks softmax over each row's index set;
//!   - LSE — fwd log-sum-exp vs the reference, split-path sink aware;
//!   - scheduler metadata — determinism and bounds of `..._sched_meta`.
//!
//! Normal run: every family must pass. Under `--negative-control` one
//! corruption per family must independently FAIL it; the run prints
//! NEGATIVE CONTROL OK and exits 0 only if all five flags fired.
//!
//! sm90 only (H20). Build/run on a pod (pod.sh build rejects --example):
//!   cargo build --release -p infer-cuda --features cuda \
//!     --example flashmla_sparse_decode_parity
//!   target/release/examples/flashmla_sparse_decode_parity --kernel-build-id
//!   target/release/examples/flashmla_sparse_decode_parity
//!   target/release/examples/flashmla_sparse_decode_parity --negative-control

#![allow(clippy::print_stdout, clippy::print_stderr)]

fn main() -> anyhow::Result<()> {
    use parity_common::Parsed;
    match parity_common::cli() {
        Parsed::BuildIdPrinted => Ok(()),
        Parsed::Run(cli) => real::run(cli.negative),
    }
}

#[allow(dead_code)] // shared harness; each gate uses only the subset it needs
#[path = "support/parity_common.rs"]
mod parity_common;

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
    use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
    use half::bf16;

    // DeepSeek-V4-Flash-0731 MODEL1 decode geometry (config.json).
    const H_Q: usize = 64;
    const H_KV: i32 = 1;
    const D_QK: usize = 512; // 448 NoPE + 64 RoPE
    const D_V: usize = 512;
    const HEAD_NOPE: usize = 448;
    const HEAD_ROPE: usize = 64;
    const PAGE: usize = 64;
    const SLIDING_WINDOW: usize = 128;
    const SW_BLOCKS: usize = 2;
    const INDEX_TOPK: usize = 512;
    const COMPRESS_RATIO: usize = 4;
    const MODE_CSA: i32 = 1;
    const MODEL1: i32 = 1;
    const TOPK_UNIFIED: usize = SLIDING_WINDOW + INDEX_TOPK; // 640
    const ROW_BYTES: usize = 584;
    const BLOCK_BYTES: usize = PAGE * ROW_BYTES; // 37376

    const B: usize = 8;
    // 2 SW + up to 2 compressed logical blocks per row (start_pos ≤ 256).
    const BLOCKS_PER_ROW: usize = 4;
    const TOTAL_BLOCKS: usize = B * BLOCKS_PER_ROW;

    // Not-yet-full window (63), ring-block boundaries ±1 (64,65,127,128,129),
    // compressed-block boundary (255,256 → floor/4 = 63,64 keys).
    const START_POS: [i32; B] = [63, 64, 65, 127, 128, 129, 255, 256];
    const DOM_SLOT: usize = 0;

    // Caps for THIS construction: the dominant 0.5 latent packs exactly
    // (0.5/2^-9 = 256, an E4M3 level) and carries ~all the weight, so the
    // dominant-key channel is quantization-exact and only small random keys
    // contribute bounded error (E4M3 worst half-ULP 1/16 in the top binade;
    // they land in much tighter low binades). Pre-GPU analytic caps, to be
    // confirmed/tightened from the measured run.
    const PASS_MAX_REL_OUT: f64 = 0.05;
    const PASS_MAX_ABS_LSE: f64 = 0.10;
    const PACK_NOPE_REL: f64 = 0.07;
    const PACK_ROPE_ABS: f32 = 0.01;

    struct Fwd {
        indices: Vec<i32>,
        out: Vec<bf16>,
        lse: Vec<f32>,
    }

    /// Physical block for (row, logical block): a row-specific rotation so no
    /// row's logical block order equals its physical order.
    fn phys_of(r: usize, l: usize) -> usize {
        r * BLOCKS_PER_ROW + (l + 1 + r) % BLOCKS_PER_ROW
    }

    pub(super) fn run(negative: bool) -> Result<()> {
        let ctx = DeviceContext::new()?;
        let cc = ctx.compute_capability();
        eprintln!(
            "[flashmla-sparse-parity] device={} cc={}.{} sms={} b={B} build={}",
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

        let mut tables = vec![0i32; B * BLOCKS_PER_ROW];
        for r in 0..B {
            for l in 0..BLOCKS_PER_ROW {
                tables[r * BLOCKS_PER_ROW + l] = phys_of(r, l) as i32;
            }
        }

        let total_tokens = TOTAL_BLOCKS * PAGE;
        let mut nope_src = vec![0f32; total_tokens * HEAD_NOPE];
        let mut rope_src = vec![0f32; total_tokens * HEAD_ROPE];
        let mut used = vec![false; total_tokens];
        let mut selected = vec![-1i32; B * INDEX_TOPK];
        let mut dom_phys = [0usize; B];

        let mut pool = ctx.stream.alloc_zeros::<u8>(TOTAL_BLOCKS * BLOCK_BYTES)?;

        for r in 0..B {
            let start = START_POS[r];
            let comp_count = (start / COMPRESS_RATIO as i32) as usize;
            let dom_c = comp_count - 1;
            dom_phys[r] = {
                let logical = SW_BLOCKS + dom_c / PAGE;
                phys_of(r, logical) * PAGE + dom_c % PAGE
            };

            let mut pack_nope = Vec::new();
            let mut pack_rope = Vec::new();
            let mut pack_block = Vec::new();
            let mut pack_row = Vec::new();
            let emit = |nope: &mut Vec<f32>,
                        rope: &mut Vec<f32>,
                        blocks: &mut Vec<i32>,
                        rows: &mut Vec<i32>,
                        c_block: usize,
                        row_in_block: usize,
                        is_dom: bool,
                        nope_src: &mut [f32],
                        rope_src: &mut [f32],
                        used: &mut [bool]| {
                let phys = phys_of(r, c_block) * PAGE + row_in_block;
                assert!(!used[phys], "physical token {phys} written twice");
                used[phys] = true;
                blocks.push(c_block as i32);
                rows.push(row_in_block as i32);
                for d in 0..HEAD_NOPE {
                    let v = if is_dom {
                        0.5
                    } else {
                        rand01(r as u64, c_block as u64, row_in_block as u64, d as u64, 0) * 0.1
                            - 0.05
                    };
                    nope_src[phys * HEAD_NOPE + d] = v;
                    nope.push(v);
                }
                for d in 0..HEAD_ROPE {
                    let v = if is_dom {
                        0.5
                    } else {
                        rand01(r as u64, c_block as u64, row_in_block as u64, d as u64, 1) * 0.1
                            - 0.05
                    };
                    rope_src[phys * HEAD_ROPE + d] = v;
                    rope.push(v);
                }
            };
            for ring in 0..SLIDING_WINDOW {
                emit(
                    &mut pack_nope,
                    &mut pack_rope,
                    &mut pack_block,
                    &mut pack_row,
                    ring / PAGE,
                    ring % PAGE,
                    false,
                    &mut nope_src,
                    &mut rope_src,
                    &mut used,
                );
            }
            for c in 0..comp_count {
                let is_dom = c == dom_c;
                emit(
                    &mut pack_nope,
                    &mut pack_rope,
                    &mut pack_block,
                    &mut pack_row,
                    SW_BLOCKS + c / PAGE,
                    c % PAGE,
                    is_dom,
                    &mut nope_src,
                    &mut rope_src,
                    &mut used,
                );
            }
            pack_row_band(
                &ctx,
                &mut pool,
                &tables[r * BLOCKS_PER_ROW..(r + 1) * BLOCKS_PER_ROW],
                &pack_nope,
                &pack_rope,
                &pack_block,
                &pack_row,
            )?;

            // Fisher-Yates of the causal keys; dominant pinned at DOM_SLOT.
            let row_sel = &mut selected[r * INDEX_TOPK..(r + 1) * INDEX_TOPK];
            let mut order: Vec<i32> = (0..comp_count as i32).collect();
            let mut state =
                0x9e37_79b9_7f4a_7c15u64 ^ (r as u64).wrapping_mul(0x517c_c1cc_9e37_79b9);
            let mut next = || {
                state ^= state >> 33;
                state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
                state ^= state >> 33;
                state
            };
            for i in (1..order.len()).rev() {
                let j = (next() as usize) % (i + 1);
                order.swap(i, j);
            }
            // Place dominant at slot 0; shift whoever was there to dom's spot.
            let dom_pos = order.iter().position(|&k| k as usize == dom_c).unwrap();
            order.swap(0, dom_pos);
            for (slot, &k) in order.iter().enumerate() {
                row_sel[slot] = k;
            }
        }
        ctx.sync()?;

        // Nonzero random per-head sink, sign + scale covered.
        let sink: Vec<f32> = (0..H_Q)
            .map(|h| rand01(0xA11C, h as u64, 0, 7, 3) * 1.0 - 0.2)
            .collect();
        let q_host = vec![bf16::from_f32(0.5); B * H_Q * D_QK];

        let pool_bytes = ctx.stream.clone_dtoh(&pool)?;
        let records = decode_pool(&pool_bytes);
        let (pack_ok, pack_max_nope, pack_max_rope) =
            check_pack(&pool_bytes, &nope_src, &rope_src, &used, &tables);

        let (num_sm_parts, fixed_overhead, block_size_topk) =
            attention::flashmla_sm90_sparse_decode_get_meta(H_Q as i32, 1, MODEL1)?;
        let parts_max = num_sm_parts.max(256) as usize;

        let q_dev = ctx.stream.clone_htod(&q_host)?;
        let table_dev = ctx.stream.clone_htod(&tables)?;
        let start_dev = ctx.stream.clone_htod(&START_POS)?;
        let offsets_dev = ctx.stream.clone_htod(&vec![0i32; B])?;
        let sink_dev = ctx.stream.clone_htod(&sink)?;

        let selected_dev = ctx.stream.clone_htod(&selected)?;
        ensure!(
            pack_ok,
            "pack family: decoded pool differs from BF16 sources (nope {pack_max_nope}, rope {pack_max_rope})"
        );
        println!(
            "pack max_rel_nope={pack_max_nope:.6} max_abs_rope={pack_max_rope:.6} pass={pack_ok}"
        );

        if !negative {
            // Explicit B=1 (default serving shape) and B=8; the scheduler
            // partition depends on batch size, so both are verified.
            for &b in &[1usize, B] {
                let mut sched_good = Vec::new();
                let mut splits_good = Vec::new();
                build_sched(
                    &ctx,
                    num_sm_parts,
                    parts_max,
                    fixed_overhead,
                    block_size_topk,
                    TOPK_UNIFIED,
                    &mut sched_good,
                    &mut splits_good,
                    b,
                )?;
                let got = run_fwd(
                    &ctx,
                    &q_dev,
                    &pool,
                    &table_dev,
                    &selected_dev,
                    &start_dev,
                    &offsets_dev,
                    &sink_dev,
                    num_sm_parts,
                    &sched_good,
                    &splits_good,
                    b,
                )?;
                let ref_indices = oracle_indices(&tables, &selected, b);
                let (ref_out, ref_lse) =
                    reference_attention(&records, &ref_indices, &sink, &splits_good, b);
                let idx_ok = got.indices == ref_indices;
                let (max_rel_out, max_abs_lse) = compare(&got, &ref_out, &ref_lse);
                let mut sched2 = Vec::new();
                let mut splits2 = Vec::new();
                build_sched(
                    &ctx,
                    num_sm_parts,
                    parts_max,
                    fixed_overhead,
                    block_size_topk,
                    TOPK_UNIFIED,
                    &mut sched2,
                    &mut splits2,
                    b,
                )?;
                let sched_ok = sched2 == sched_good
                    && splits2 == splits_good
                    && splits_good.len() == b + 1
                    && splits_good.windows(2).all(|w| w[1] >= w[0])
                    && *splits_good.last().unwrap() >= 1
                    && (*splits_good.last().unwrap() as usize) <= parts_max;
                println!("B={b} indices_exact={idx_ok}");
                println!(
                    "B={b} PASS_MAX_REL_out={max_rel_out:.6} (cap {PASS_MAX_REL_OUT}) pass={}",
                    max_rel_out <= PASS_MAX_REL_OUT
                );
                println!(
                    "B={b} PASS_MAX_ABS_lse={max_abs_lse:.6} (cap {PASS_MAX_ABS_LSE}) pass={}",
                    max_abs_lse <= PASS_MAX_ABS_LSE
                );
                println!("B={b} scheduler deterministic+bounded pass={sched_ok}");
                ensure!(idx_ok, "B={b} indices differ from semantic oracle");
                ensure!(max_rel_out <= PASS_MAX_REL_OUT, "B={b} output exceeds cap");
                ensure!(max_abs_lse <= PASS_MAX_ABS_LSE, "B={b} LSE exceeds cap");
                ensure!(
                    sched_ok,
                    "B={b} scheduler metadata not deterministic/bounded"
                );
            }
            println!("[flashmla-sparse-parity] ALL PASS");
            return Ok(());
        }

        // ── negative controls at B.
        let mut sched_good = Vec::new();
        let mut splits_good = Vec::new();
        build_sched(
            &ctx,
            num_sm_parts,
            parts_max,
            fixed_overhead,
            block_size_topk,
            TOPK_UNIFIED,
            &mut sched_good,
            &mut splits_good,
            B,
        )?;
        let ref_indices = oracle_indices(&tables, &selected, B);
        let (ref_out, ref_lse) =
            reference_attention(&records, &ref_indices, &sink, &splits_good, B);

        // Mask row 0's dominant selected entry.
        let mut bad_sel = selected.clone();
        bad_sel[DOM_SLOT] = -1;
        let bad_sel_dev = ctx.stream.clone_htod(&bad_sel)?;
        let got_neg = run_fwd(
            &ctx,
            &q_dev,
            &pool,
            &table_dev,
            &bad_sel_dev,
            &start_dev,
            &offsets_dev,
            &sink_dev,
            num_sm_parts,
            &sched_good,
            &splits_good,
            B,
        )?;
        let bad_oracle = oracle_indices(&tables, &bad_sel, B);
        let (rel_bad, lse_bad) = {
            let (o, l) = reference_attention(&records, &bad_oracle, &sink, &splits_good, B);
            compare(&got_neg, &o, &l)
        };
        let neg_indices = got_neg.indices != ref_indices;
        let (rel_good, lse_good) = compare(&got_neg, &ref_out, &ref_lse);
        let neg_out = rel_good > PASS_MAX_REL_OUT;
        let neg_lse = lse_good > PASS_MAX_ABS_LSE;
        let bad_self_consistent = rel_bad <= PASS_MAX_REL_OUT && lse_bad <= PASS_MAX_ABS_LSE;

        // Pack family: flip a NoPE exponent byte on a written token.
        let mut corrupt = pool_bytes.clone();
        let byte_idx = dom_phys[0] * ROW_BYTES;
        corrupt[byte_idx] ^= 0x20;
        let (neg_pack, _, _) = check_pack(&corrupt, &nope_src, &rope_src, &used, &tables);

        // Scheduler: 8x topk forces a different partition.
        let mut meta8 = Vec::new();
        let mut splits8 = Vec::new();
        build_sched(
            &ctx,
            num_sm_parts,
            parts_max,
            fixed_overhead,
            block_size_topk,
            TOPK_UNIFIED * 8,
            &mut meta8,
            &mut splits8,
            B,
        )?;
        let neg_sched = meta8 != sched_good || splits8 != splits_good;

        println!("negative pack fired={neg_pack} (nope rel {pack_max_nope:.6})");
        println!("negative indices fired={neg_indices}");
        println!(
            "negative out rel_vs_good={rel_good:.6} (self-consistent {bad_self_consistent}) fired={neg_out}"
        );
        println!("negative lse abs_vs_good={lse_good:.6} fired={neg_lse}");
        println!("negative scheduler fired={neg_sched}");
        ensure!(!neg_pack, "pack negative control unexpectedly passed");
        ensure!(neg_indices, "negative control did not fail indices");
        ensure!(
            neg_out && bad_self_consistent,
            "output negative control invalid (move {rel_good}, self {rel_bad})"
        );
        ensure!(neg_lse, "negative control did not fail LSE");
        ensure!(neg_sched, "negative control did not fail scheduler");
        println!("NEGATIVE CONTROL OK");
        Ok(())
    }

    fn rand01(a: u64, b: u64, c: u64, d: u64, plane: u64) -> f32 {
        let mut x = a.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ b.wrapping_mul(0x517c_c1cc_9e37_79b9)
            ^ c.wrapping_mul(0xff51_afd7_ed55_8ccd)
            ^ d.wrapping_mul(0x2545_f491_4f6c_dd1d)
            ^ plane.wrapping_mul(0x1000_0000_0000_0001);
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        (x >> 40) as f32 / (1u64 << 24) as f32
    }

    #[allow(clippy::too_many_arguments)]
    fn pack_row_band(
        ctx: &DeviceContext,
        pool: &mut CudaSlice<u8>,
        row_table: &[i32],
        nope: &[f32],
        rope: &[f32],
        block_ids: &[i32],
        row_ids: &[i32],
    ) -> Result<()> {
        let nope_bf: Vec<bf16> = nope.iter().map(|&v| bf16::from_f32(v)).collect();
        let rope_bf: Vec<bf16> = rope.iter().map(|&v| bf16::from_f32(v)).collect();
        let nope_dev = ctx.stream.clone_htod(&nope_bf)?;
        let rope_dev = ctx.stream.clone_htod(&rope_bf)?;
        let block_dev = ctx.stream.clone_htod(block_ids)?;
        let row_dev = ctx.stream.clone_htod(row_ids)?;
        let table_dev = ctx.stream.clone_htod(row_table)?;
        let n = nope.len() / HEAD_NOPE;
        let (nope_ptr, _g1) = nope_dev.device_ptr(&ctx.stream);
        let (rope_ptr, _g2) = rope_dev.device_ptr(&ctx.stream);
        let (pool_ptr, _g3) = pool.device_ptr_mut(&ctx.stream);
        attention::dsv4_fp8_kv_pack_strided_raw(
            ctx,
            nope_ptr,
            rope_ptr,
            pool_ptr,
            &block_dev,
            &row_dev,
            n,
            PAGE,
            HEAD_NOPE,
            HEAD_ROPE,
            Some(&table_dev),
        )
    }

    /// Pack family: every written pool token decodes back to the BF16 source
    /// within the tile half-ULP (NoPE) and bf16 half-ULP (RoPE copy).
    #[allow(clippy::too_many_arguments)]
    fn check_pack(
        pool: &[u8],
        nope_src: &[f32],
        rope_src: &[f32],
        used: &[bool],
        tables: &[i32],
    ) -> (bool, f64, f32) {
        let _ = tables;
        let mut max_rel_nope = 0f64;
        let mut max_abs_rope = 0f32;
        for phys in 0..used.len() {
            if !used[phys] {
                continue;
            }
            let block = phys / PAGE;
            let row = phys % PAGE;
            let block_base = block * BLOCK_BYTES;
            let data_base = block_base + row * (HEAD_NOPE + HEAD_ROPE * 2);
            for tile in 0..7usize {
                let e8m0 = pool[block_base + PAGE * (HEAD_NOPE + HEAD_ROPE * 2) + row * 8 + tile];
                let scale = if e8m0 == 0 {
                    0.0
                } else {
                    2f32.powi(i32::from(e8m0) - 127)
                };
                for lane in 0..64usize {
                    let d = tile * 64 + lane;
                    let got = decode_e4m3(pool[data_base + d]) * scale;
                    let want = nope_src[phys * HEAD_NOPE + d];
                    let denom = want.abs().max(f32::MIN_POSITIVE);
                    max_rel_nope = max_rel_nope.max(((got - want) / denom).abs() as f64);
                }
            }
            for d in 0..HEAD_ROPE {
                let lo = u16::from(pool[data_base + HEAD_NOPE + d * 2]);
                let hi = u16::from(pool[data_base + HEAD_NOPE + d * 2 + 1]);
                let got = bf16::from_bits(lo | (hi << 8)).to_f32();
                let want = rope_src[phys * HEAD_ROPE + d];
                max_abs_rope = max_abs_rope.max((got - want).abs());
            }
        }
        let ok = max_rel_nope <= PACK_NOPE_REL && max_abs_rope <= PACK_ROPE_ABS;
        (ok, max_rel_nope, max_abs_rope)
    }

    fn decode_pool(bytes: &[u8]) -> Vec<Vec<f32>> {
        let mut records = Vec::with_capacity(TOTAL_BLOCKS * PAGE);
        for block in 0..TOTAL_BLOCKS {
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

    /// Semantic index oracle, independent of the device index_at code.
    /// Per row: the last min(128, start+1) token POSITIONS, each mapped
    /// (ring slot → logical SW block → physical via the table), then the
    /// selected indexer keys k in [0, floor(start/4)] mapped to compressed
    /// physical rows via the table. Order matches the kernel slot order
    /// (SW first, then selected slots); -1 elsewhere.
    fn oracle_indices(tables: &[i32], selected: &[i32], b: usize) -> Vec<i32> {
        let mut out = vec![-1i32; b * TOPK_UNIFIED];
        for r in 0..b {
            let start = START_POS[r];
            let table = &tables[r * BLOCKS_PER_ROW..(r + 1) * BLOCKS_PER_ROW];
            let row_out = &mut out[r * TOPK_UNIFIED..(r + 1) * TOPK_UNIFIED];
            let sel = &selected[r * INDEX_TOPK..(r + 1) * INDEX_TOPK];

            let sw_start = (start - SLIDING_WINDOW as i32 + 1).max(0);
            let sw_count = (start - sw_start + 1) as usize;
            for (p, ()) in std::iter::repeat_n((), sw_count).enumerate() {
                let pos = sw_start + p as i32;
                let ring = pos % SLIDING_WINDOW as i32;
                let logical_block = ring / PAGE as i32;
                let row_in_block = ring % PAGE as i32;
                if (logical_block as usize) < SW_BLOCKS {
                    let phys_block = table[logical_block as usize];
                    row_out[p] = phys_block * PAGE as i32 + row_in_block;
                }
            }
            for (slot, &k) in sel.iter().enumerate() {
                if k < 0 {
                    continue;
                }
                let block_end = k * COMPRESS_RATIO as i32 + (COMPRESS_RATIO as i32 - 1);
                if block_end > start {
                    continue;
                }
                let logical_block = SW_BLOCKS as i32 + k / PAGE as i32;
                let row_in_block = k % PAGE as i32;
                let phys_block = table[logical_block as usize];
                row_out[sw_count + slot] = phys_block * PAGE as i32 + row_in_block;
            }
        }
        out
    }

    /// f32 masked sparse attention. Q is the bf16 constant 0.5; V is the
    /// same 512-dim latent record. The zero-of-many-heads sink here is
    /// nonzero per head (given). Output denominator always includes
    /// exp(sink - m); LSE includes it only on the multi-split combine path.
    fn reference_attention(
        records: &[Vec<f32>],
        indices: &[i32],
        sink: &[f32],
        num_splits: &[i32],
        b: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let sm_scale = 1.0 / (D_QK as f32).sqrt();
        let mut out = vec![0f32; b * H_Q * D_V];
        let mut lse = vec![0f32; b * H_Q];
        for r in 0..b {
            let valid: Vec<usize> = indices[r * TOPK_UNIFIED..(r + 1) * TOPK_UNIFIED]
                .iter()
                .filter_map(|&i| (i >= 0).then_some(i as usize))
                .collect();
            let multi = num_splits[r + 1] - num_splits[r] > 1;
            for h in 0..H_Q {
                let mut scores = Vec::with_capacity(valid.len());
                let mut maxs = f32::NEG_INFINITY;
                for &idx in &valid {
                    let dot: f32 = records[idx][..D_QK].iter().map(|&k| 0.5 * k).sum();
                    let s = dot * sm_scale;
                    scores.push(s);
                    maxs = maxs.max(s);
                }
                let sink_eff = sink[h];
                let m_eff = maxs.max(sink_eff);
                let tok_exp: f32 = scores.iter().map(|&s| (s - m_eff).exp()).sum();
                let sink_exp = (sink_eff - m_eff).exp();
                // Single-split kernel writes token-only LSE; multi-split
                // combine folds the sink in. Output always divides by the
                // token+sink sum.
                lse[r * H_Q + h] = if multi {
                    m_eff + (tok_exp + sink_exp).ln()
                } else {
                    m_eff + tok_exp.ln()
                };
                let out_denom = tok_exp + sink_exp;
                for (&idx, &s) in valid.iter().zip(&scores) {
                    let p = (s - m_eff).exp() / out_denom;
                    for d in 0..D_V {
                        out[r * H_Q * D_V + h * D_V + d] += p * records[idx][d];
                    }
                }
            }
        }
        (out, lse)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_fwd(
        ctx: &DeviceContext,
        q: &CudaSlice<bf16>,
        pool: &CudaSlice<u8>,
        table: &CudaSlice<i32>,
        selected: &CudaSlice<i32>,
        start: &CudaSlice<i32>,
        offsets: &CudaSlice<i32>,
        sink: &CudaSlice<f32>,
        num_sm_parts: i32,
        sched_meta: &[i32],
        num_splits: &[i32],
        b: usize,
    ) -> Result<Fwd> {
        let parts_max = num_sm_parts.max(256) as usize;
        let mut indices = ctx.stream.alloc_zeros::<i32>(b * TOPK_UNIFIED)?;
        let mut topk_length = ctx.stream.alloc_zeros::<i32>(b)?;
        let sched_meta_dev = ctx.stream.clone_htod(sched_meta)?;
        let num_splits_dev = ctx.stream.clone_htod(num_splits)?;
        {
            let (idx_ptr, _gi) = indices.device_ptr_mut(&ctx.stream);
            let (start_ptr, _gp) = start.device_ptr(&ctx.stream);
            let (off_ptr, _go) = offsets.device_ptr(&ctx.stream);
            let (sel_ptr, _gs) = selected.device_ptr(&ctx.stream);
            let (topk_ptr, _gt) = topk_length.device_ptr_mut(&ctx.stream);
            attention::dsv4_flashmla_decode_build_indices_batched_raw(
                ctx,
                idx_ptr,
                start_ptr,
                off_ptr,
                sel_ptr,
                topk_ptr,
                b,
                SW_BLOCKS,
                SLIDING_WINDOW,
                INDEX_TOPK,
                COMPRESS_RATIO,
                MODE_CSA,
                PAGE,
                TOTAL_BLOCKS,
                Some(table),
                BLOCKS_PER_ROW,
            )?;
        }
        ctx.sync()?;
        let indices_host = ctx.stream.clone_dtoh(&indices)?;

        let mut out = ctx.stream.alloc_zeros::<bf16>(b * H_Q * D_V)?;
        let mut lse = ctx.stream.alloc_zeros::<f32>(b * H_Q)?;
        let mut lse_accum = ctx.stream.alloc_zeros::<f32>((parts_max + B) * H_Q)?;
        let mut o_accum = ctx.stream.alloc_zeros::<f32>((parts_max + B) * H_Q * D_V)?;
        {
            let (q_ptr, _gq) = q.device_ptr(&ctx.stream);
            let (pool_ptr, _gk) = pool.device_ptr(&ctx.stream);
            let (idx_ptr, _gi) = indices.device_ptr(&ctx.stream);
            let (topk_ptr, _glk) = topk_length.device_ptr(&ctx.stream);
            let (sink_ptr, _gsk) = sink.device_ptr(&ctx.stream);
            let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
            let (lse_ptr, _gl) = lse.device_ptr_mut(&ctx.stream);
            let (lacc_ptr, _gla) = lse_accum.device_ptr_mut(&ctx.stream);
            let (oacc_ptr, _goa) = o_accum.device_ptr_mut(&ctx.stream);
            let (meta_ptr, _gm) = sched_meta_dev.device_ptr(&ctx.stream);
            let (split_ptr, _gn) = num_splits_dev.device_ptr(&ctx.stream);
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
                b as i32,
                1,
                H_Q as i32,
                H_KV,
                D_QK as i32,
                D_V as i32,
                TOTAL_BLOCKS as i32,
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
            indices: indices_host,
            out: ctx.stream.clone_dtoh(&out)?,
            lse: ctx.stream.clone_dtoh(&lse)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_sched(
        ctx: &DeviceContext,
        num_sm_parts: i32,
        parts_max: usize,
        fixed_overhead: i32,
        block_size_topk: i32,
        topk: usize,
        meta_out: &mut Vec<i32>,
        splits_out: &mut Vec<i32>,
        b: usize,
    ) -> Result<()> {
        let mut meta = ctx.stream.alloc_zeros::<i32>(parts_max * 8)?;
        let mut splits = ctx.stream.alloc_zeros::<i32>(b + 1)?;
        let topk_input = ctx.stream.clone_htod(&vec![topk as i32; b])?;
        {
            let (topk_ptr, _gt) = topk_input.device_ptr(&ctx.stream);
            let (meta_ptr, _gm) = meta.device_ptr_mut(&ctx.stream);
            let (split_ptr, _gn) = splits.device_ptr_mut(&ctx.stream);
            attention::flashmla_sm90_sparse_decode_sched_meta_raw(
                &ctx.stream,
                B as i32,
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
            )?;
        }
        ctx.sync()?;
        *meta_out = ctx.stream.clone_dtoh(&meta)?;
        *splits_out = ctx.stream.clone_dtoh(&splits)?;
        Ok(())
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
