//! DSv4 MoE routing-family numeric-parity gate.
//!
//! Standalone kernel harness — NO engine, NO serve, NO expert GEMMs. Drives the
//! production DSv4 token-to-slot routing pipeline in `cuda-kernels/src/moe.rs`
//! over BF16 router logits and compares against an independent f64 host oracle
//! written from the model spec (`deepseek-spec::v4`), not from the kernels:
//!
//! - `dsv4_route`: both learned-bias (noaux_tc) and hash ABIs, sqrtsoftplus
//!   scoring + selected-sum renorm
//! - `dsv4_count_local_experts`
//! - `dsv4_exclusive_scan_i32` / `moe_exclusive_scan_aligned_i32`
//! - `dsv4_pack_local_experts_with_slots`
//! - `dsv4_fill_m_indices_from_counts`
//! - `dsv4_scatter_all_route_slots` / `dsv4_combine_route_slot_outputs`
//!
//! Config is production DSv4-Flash-0731 (config.json): 256 routed experts,
//! top-k 6, hidden 4096, scoring_func sqrtsoftplus, topk_method noaux_tc,
//! norm_topk_prob, routed_scaling_factor 1.5, no n_group/topk_group. The
//! separate `qwen36_renorm_topk_weights` kernel is NOT driven: DSv4 renorms
//! inside `dsv4_route`; that kernel's only caller is qwen.rs:287 (qwen35/36).
//!
//! Integer outputs (expert indices, counts, offsets, totals, packed route-slot
//! assignment, m-indices) must match EXACTLY. The weighted combine is compared
//! against the kernel's own slot mapping with its exact BF16 rounding.
//!
//! Two learned-bias regimes are run: a coarse-bias case that pins the exact
//! lower-expert-wins tie contract deterministically, and a score-decides case
//! with bias on the scale of the sqrtsoftplus spread, where the score term
//! actually chooses experts (a kernel ignoring scores can't pass). Near-tie
//! rows (boundary gap below a fast-math-derived epsilon) are exempted and
//! counted; the run fails if >5% of rows need an exemption.
//!
//! The first `num_hash_layers=3` layers hash-route; learned-bias and hash ABIs
//! are both covered. EP is a single rank owning the contiguous
//! `[0, experts_per_rank)` window, swept 32→256.
//!
//! `--negative-control` runs one pipeline per regime and sabotages each
//! comparison family (route / counts / offsets / pack / weights / combine)
//! independently, asserting that family trips. It prints NEGATIVE CONTROL OK
//! and exits 0; the teeth assertions are internal.
//!
//! Run on a pod:
//! `INFER_CUDA_DEVICE=<free-gpu> target/release/examples/moe_routing_parity`

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
        eprintln!("moe_routing_parity is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::Result;
    use cuda_kernels::moe;
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::tensor::cache_ptr;
    use half::bf16;

    // Production DSv4-Flash-0731 routing geometry (config.json):
    // n_routed_experts 256, num_experts_per_tok 6, hidden_size 4096,
    // scoring_func "sqrtsoftplus", topk_method "noaux_tc", norm_topk_prob true,
    // routed_scaling_factor 1.5, no n_group/topk_group (ungrouped).
    const N_EXPERTS: usize = 256;
    const TOPK: usize = 6;
    const HIDDEN: usize = 4096;
    const ROUTED_SCALING: f64 = 1.5;
    // sqrt(softplus(logit)) — the kernel's scoring_kind=2 branch
    // (dsv4_route.cu `dsv4_route_score`).
    const SCORING_KIND: i32 = 2;
    const ALIGNMENT: usize = 128;
    // First `num_hash_layers=3` MoE layers hash-route; the rest use the
    // learned noaux_tc bias. Both ABIs are driven here.
    const MODES: &[Mode] = &[Mode::LearnedBias, Mode::ScoreDecides, Mode::Hash];

    const TOKEN_COUNTS: &[usize] = &[1, 8, 32];
    const EXPERTS_PER_RANK: &[usize] = &[32, 64, 128, 256];
    const SEED: u64 = 0xD5D4_0000_0000_0001;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn unit(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.unit().max(1e-12);
            let u2 = self.unit();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Mode {
        /// routing_kind=1 with coarse bias: exact, deterministic tie contract.
        LearnedBias,
        /// routing_kind=1 with bias on the scale of the score spread: the
        /// sqrtsoftplus score (not the bias) decides the expert order.
        ScoreDecides,
        /// routing_kind=0: experts read from the tid2eid table by token id.
        Hash,
    }

    /// Kernel scoring_kind=2: sqrt(softplus(logit)), softplus clamped above 20.
    fn sqrt_softplus(x: f64) -> f64 {
        let sp = if x > 20.0 { x } else { x.exp().ln_1p() };
        sp.sqrt()
    }

    /// Deterministic stand-in for one packed slot's expert output, in [-1, 1).
    fn expert_output(slot: usize, col: usize) -> f64 {
        let h = (slot as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((col as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
        ((h >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// f64 host oracle for the whole single-rank routing pipeline.
    /// `logits`/`bias` are `[num_tokens, N_EXPERTS]`; BF16-rounded inputs are
    /// passed in so the oracle and kernel see identical values.
    #[derive(Clone)]
    struct Oracle {
        /// Global expert per route, `[num_tokens * TOPK]`.
        indices: Vec<i32>,
        /// Post-renorm, scaled gate weight per route, `[num_tokens * TOPK]`.
        weights: Vec<f64>,
        /// Per-local-expert route count.
        counts: Vec<i32>,
        /// Exclusive prefix-sum of counts.
        offsets: Vec<i32>,
        total: i32,
        /// Aligned exclusive prefix-sum.
        aligned_offsets: Vec<i32>,
        aligned_total: i32,
        /// Packed-slot assignment: `slot -> global route`.
        packed_route_slot: Vec<i32>,
        /// Filled m-indices over the aligned span.
        m_indices: Vec<i32>,
        /// Learned routing only: combined-score gap between the k-th selected
        /// and the best rejected expert, per token. f64::INFINITY for hash.
        boundary_gap: Vec<f64>,
    }

    fn host_oracle(
        mode: Mode,
        logits: &[f64],
        bias: &[f64],
        // Hash ABI only: tid2eid[token*TOPK + k], token id == token index.
        hash_eid: &[i64],
        num_tokens: usize,
        experts_per_rank: usize,
    ) -> Oracle {
        let total_routes = num_tokens * TOPK;
        let mut indices = vec![-1i32; total_routes];
        let mut weights = vec![0.0f64; total_routes];
        let mut boundary_gap = vec![f64::INFINITY; num_tokens];

        for t in 0..num_tokens {
            let base = t * N_EXPERTS;
            // Gate scores feed the weights in BOTH modes; only selection differs.
            let scores: Vec<f64> = (0..N_EXPERTS)
                .map(|e| sqrt_softplus(logits[base + e]))
                .collect();
            let chosen: Vec<usize> = if mode == Mode::Hash {
                (0..TOPK).map(|k| hash_eid[t * TOPK + k] as usize).collect()
            } else {
                // Masked top-k over `score + bias` (lower index wins ties).
                // Track the k-th pick's margin over the best remaining expert.
                let combined: Vec<f64> =
                    (0..N_EXPERTS).map(|e| scores[e] + bias[base + e]).collect();
                let mut chosen = vec![0usize; TOPK];
                let mut order: Vec<usize> = (0..N_EXPERTS).collect();
                // Stable sort by (-combined, index): ties resolve to lower e.
                order.sort_by(|&a, &b| {
                    combined[b]
                        .partial_cmp(&combined[a])
                        .unwrap()
                        .then_with(|| a.cmp(&b))
                });
                for (k, &e) in order.iter().take(TOPK).enumerate() {
                    chosen[k] = e;
                }
                let kth = combined[chosen[TOPK - 1]];
                let best_rest = order
                    .iter()
                    .skip(TOPK)
                    .map(|&e| combined[e])
                    .fold(f64::NEG_INFINITY, f64::max);
                boundary_gap[t] = if best_rest == f64::NEG_INFINITY {
                    f64::INFINITY
                } else {
                    kth - best_rest
                };
                chosen
            };
            // norm_topk_prob: divide the selected scores by their sum (+1e-9),
            // then multiply routed_scaling_factor (dsv4_route phase 3).
            let selected_sum: f64 = chosen.iter().map(|&e| scores[e]).sum();
            let denom = selected_sum + 1e-9;
            for (k, &e) in chosen.iter().enumerate() {
                let r = t * TOPK + k;
                indices[r] = e as i32;
                weights[r] = scores[e] / denom * ROUTED_SCALING;
            }
        }

        // Count local routes (single rank owns global experts [0, ep)).
        let mut counts = vec![0i32; experts_per_rank];
        for &e in &indices {
            if (0..experts_per_rank as i32).contains(&e) {
                counts[e as usize] += 1;
            }
        }
        // Plain exclusive scan.
        let mut offsets = vec![0i32; experts_per_rank];
        let mut acc = 0i32;
        for e in 0..experts_per_rank {
            offsets[e] = acc;
            acc += counts[e];
        }
        let total = acc;
        // Aligned exclusive scan.
        let mut aligned_offsets = vec![0i32; experts_per_rank];
        let mut aacc = 0i32;
        for e in 0..experts_per_rank {
            aligned_offsets[e] = aacc;
            let span = (counts[e] as usize).div_ceil(ALIGNMENT) * ALIGNMENT;
            aacc += span as i32;
        }
        let aligned_total = aacc;

        // Pack: within each local expert, global routes ascending.
        let n_local = total as usize;
        let mut packed_route_slot = vec![-1i32; n_local.max(1)];
        let mut cursors = vec![0i32; experts_per_rank];
        for (r, &e) in indices.iter().enumerate() {
            if (0..experts_per_rank as i32).contains(&e) {
                let e = e as usize;
                let slot = offsets[e] + cursors[e];
                cursors[e] += 1;
                packed_route_slot[slot as usize] = r as i32;
            }
        }

        // m-indices over the aligned span.
        let mut m_indices = vec![-1i32; aligned_total as usize];
        for e in 0..experts_per_rank {
            for row in 0..counts[e] as usize {
                m_indices[aligned_offsets[e] as usize + row] = e as i32;
            }
        }

        Oracle {
            indices,
            weights,
            counts,
            offsets,
            total,
            aligned_offsets,
            aligned_total,
            packed_route_slot,
            m_indices,
            boundary_gap,
        }
    }

    pub(super) fn run(negative: bool) -> Result<()> {
        let ctx = DeviceContext::new()?;
        eprintln!(
            "[moe-routing-parity] device={} build={} experts={} topk={} hidden={}{}",
            ctx.ordinal(),
            cuda_kernels::KERNEL_BUILD_ID,
            N_EXPERTS,
            TOPK,
            HIDDEN,
            if negative { " NEGATIVE-CONTROL" } else { "" }
        );

        if negative {
            // Per-family teeth: one pipeline per mode (mid geometry), then six
            // one-sided sabotages, each of which only its own comparator must
            // catch. Only the six sabotaged families are asserted here; totals
            // and m_indices share the offsets/counts comparators and get their
            // teeth transitively, so this gate's negative control is not shaped
            // as all-eight-must-fail and stays on its bespoke tail.
            for &mode in MODES {
                let bundle = pipeline(&ctx, mode, 8, 64)?;
                verify_negative_teeth(&bundle)?;
            }
            eprintln!("[moe-routing-parity] NEGATIVE CONTROL OK");
            return Ok(());
        }

        const MAX_EXEMPT_FRACTION: f64 = 0.05;
        let mut fam_failed = [false; 8];
        let mut exempt_overflow = false;
        for &mode in MODES {
            for &num_tokens in TOKEN_COUNTS {
                for &ep in EXPERTS_PER_RANK {
                    let b = pipeline(&ctx, mode, num_tokens, ep)?;
                    let fam = b.compare(
                        &format!("mode={mode:?} tokens={num_tokens} ep={ep}"),
                        NEAR_TIE_EPS,
                    );
                    let mut row_fail = false;
                    for f in Family::all() {
                        if !f.get(&fam) {
                            row_fail = true;
                            fam_failed[f.idx()] = true;
                        }
                    }
                    if mode == Mode::ScoreDecides {
                        let frac = fam.near_tie_exemptions as f64 / num_tokens.max(1) as f64;
                        if frac > MAX_EXEMPT_FRACTION {
                            row_fail = true;
                            exempt_overflow = true;
                            eprintln!(
                                "[mode={mode:?} tokens={num_tokens} ep={ep}] FAIL near-tie {n_ex}/{n_tok} = {frac:.2}",
                                n_ex = fam.near_tie_exemptions,
                                n_tok = num_tokens,
                            );
                        }
                    }
                    if !row_fail {
                        eprintln!(
                            "[mode={mode:?} tokens={num_tokens} ep={ep}] PASS (local={} near_tie={} combine_worst={:.3e})",
                            b.n_local, fam.near_tie_exemptions, fam.combine_worst,
                        );
                    }
                }
            }
        }
        anyhow::ensure!(
            !exempt_overflow,
            "moe_routing_parity FAILED — see violations above"
        );
        let mut families = super::parity_common::Families::new();
        for f in Family::all() {
            families.record(f.name(), fam_failed[f.idx()]);
        }
        families.finish("moe-routing-parity", false)
    }

    #[allow(clippy::too_many_lines)]
    /// Run the full device pipeline and collect host-side outputs + oracle.
    fn pipeline(ctx: &DeviceContext, mode: Mode, num_tokens: usize, ep: usize) -> Result<Bundle> {
        let total_routes = num_tokens * TOPK;
        let mode_seed: u64 = if mode == Mode::Hash { 0x0A5C } else { 0xB1A5 };
        let mut rng =
            Rng::new(SEED ^ ((num_tokens as u64) << 20) ^ ((ep as u64) << 8) ^ (mode_seed << 40));

        // Hash ABI: deterministic per-token experts, all inside the smallest
        // (ep=32) local window so pack/count get exercised at every EP width.
        let hash_eid: Vec<i64> = (0..total_routes)
            .map(|r| {
                let t = r / TOPK;
                let k = r % TOPK;
                ((t * 7 + k * 5) % 30) as i64
            })
            .collect();

        // Build bf16 logits + bias, retaining f64 values for the oracle.
        let mut logits_f64 = vec![0.0f64; num_tokens * N_EXPERTS];
        let mut bias_f64 = vec![0.0f64; num_tokens * N_EXPERTS];
        let mut logits_bf16 = Vec::with_capacity(num_tokens * N_EXPERTS);
        let mut bias_bf16 = Vec::with_capacity(num_tokens * N_EXPERTS);
        for t in 0..num_tokens {
            for e in 0..N_EXPERTS {
                // Logits in [-1,1]: sqrtsoftplus there spans ~0.55..0.97.
                let l = rng.unit().mul_add(2.0, -1.0);
                logits_f64[t * N_EXPERTS + e] = f64::from(bf16::from_f32(l as f32));
                bias_f64[t * N_EXPERTS + e] = 0.0;
            }
            if mode != Mode::Hash {
                // Both learned-bias regimes feed a noaux correction; the regime
                // changes its scale relative to the score spread.
                let raw_bias: Vec<f32> = (0..N_EXPERTS).map(|_| rng.normal() as f32).collect();
                match mode {
                    Mode::LearnedBias => {
                        // Coarse bias dominates the <=0.42 score swing; experts
                        // 0 and 1 share BOTH bias and logit (exact combined tie):
                        // round 1 must take 0 (lower), round 2 takes 1. Integer
                        // picks are then stable regardless of fast math.
                        let top_bias = (N_EXPERTS - 1) as f32;
                        for e in 0..N_EXPERTS {
                            bias_f64[t * N_EXPERTS + e] = f64::from(bf16::from_f32(if e <= 1 {
                                top_bias
                            } else {
                                (N_EXPERTS - e) as f32
                            }));
                        }
                        logits_f64[t * N_EXPERTS] = 0.0;
                        logits_f64[t * N_EXPERTS + 1] = 0.0;
                    }
                    Mode::ScoreDecides => {
                        // Bias ~ N(0, 0.03): same order as the score spread, so
                        // the sqrtsoftplus term decides most picks. A kernel
                        // ignoring or mis-scaling the score can't pass this.
                        for e in 0..N_EXPERTS {
                            bias_f64[t * N_EXPERTS + e] =
                                f64::from(bf16::from_f32(raw_bias[e] * 0.03));
                        }
                    }
                    Mode::Hash => {}
                }
            }
            for v in &logits_f64[t * N_EXPERTS..(t + 1) * N_EXPERTS] {
                logits_bf16.push(bf16::from_f32(*v as f32));
            }
            for v in &bias_f64[t * N_EXPERTS..(t + 1) * N_EXPERTS] {
                bias_bf16.push(bf16::from_f32(*v as f32));
            }
        }

        let orc = host_oracle(mode, &logits_f64, &bias_f64, &hash_eid, num_tokens, ep);

        // ---- device route ----
        let logits_d = ctx.stream.clone_htod(&logits_bf16)?;
        let indices_d = ctx.stream.alloc_zeros::<i32>(total_routes)?;
        let weights_d = ctx.stream.alloc_zeros::<f32>(total_routes)?;
        // Bias is uploaded for both learned-bias regimes; hash gets neither.
        let learned = mode != Mode::Hash;
        let bias_d = if learned {
            Some(ctx.stream.clone_htod(&bias_bf16)?)
        } else {
            None
        };
        let (hash_tbl_d, token_ids_d) = if mode == Mode::Hash {
            let tbl = ctx.stream.clone_htod(&hash_eid)?;
            // Token id equals token index for this harness.
            let ids = ctx
                .stream
                .clone_htod(&(0..num_tokens as u32).collect::<Vec<u32>>())?;
            (Some(tbl), Some(ids))
        } else {
            (None, None)
        };
        let routing_kind = if mode == Mode::Hash { 0 } else { 1 };
        // SAFETY: logits [num_tokens,N_EXPERTS], indices/weights total_routes;
        // bias present iff learned-bias, tid2eid+token_ids present iff hash.
        unsafe {
            moe::dsv4_route(
                cache_ptr(&logits_d, ctx),
                bias_d.as_ref().map(|b| cache_ptr(b, ctx)),
                hash_tbl_d.as_ref().map(|t| cache_ptr(t, ctx)),
                token_ids_d.as_ref().map(|t| cache_ptr(t, ctx)),
                cache_ptr(&indices_d, ctx),
                cache_ptr(&weights_d, ctx),
                num_tokens,
                N_EXPERTS,
                TOPK,
                routing_kind,
                SCORING_KIND,
                ROUTED_SCALING as f32,
                ctx.stream.cu_stream(),
            )?;
        }
        ctx.sync()?;
        let got_indices = ctx.stream.clone_dtoh(&indices_d)?;
        let got_weights = ctx.stream.clone_dtoh(&weights_d)?;

        // ---- count + scans ----
        let counts_d = ctx.stream.alloc_zeros::<i32>(ep)?;
        let offsets_d = ctx.stream.alloc_zeros::<i32>(ep)?;
        let total_d = ctx.stream.alloc_zeros::<i32>(1)?;
        let aligned_offsets_d = ctx.stream.alloc_zeros::<i32>(ep)?;
        let aligned_total_d = ctx.stream.alloc_zeros::<i32>(1)?;
        // SAFETY: buffers are sized exactly [ep]/[1]; indices holds total_routes.
        unsafe {
            moe::dsv4_count_local_experts(
                cache_ptr(&indices_d, ctx),
                cache_ptr(&counts_d, ctx),
                num_tokens,
                TOPK,
                0,
                ep,
                ctx.stream.cu_stream(),
            )?;
            moe::dsv4_exclusive_scan_i32(
                cache_ptr(&counts_d, ctx),
                cache_ptr(&offsets_d, ctx),
                cache_ptr(&total_d, ctx),
                ep,
                ctx.stream.cu_stream(),
            )?;
            moe::moe_exclusive_scan_aligned_i32(
                cache_ptr(&counts_d, ctx),
                cache_ptr(&aligned_offsets_d, ctx),
                cache_ptr(&aligned_total_d, ctx),
                ep,
                ALIGNMENT,
                ctx.stream.cu_stream(),
            )?;
        }
        ctx.sync()?;
        let got_counts = ctx.stream.clone_dtoh(&counts_d)?;
        let got_offsets = ctx.stream.clone_dtoh(&offsets_d)?;
        let got_total = ctx.stream.clone_dtoh(&total_d)?;
        let got_aligned_offsets = ctx.stream.clone_dtoh(&aligned_offsets_d)?;
        let got_aligned_total = ctx.stream.clone_dtoh(&aligned_total_d)?;

        // ---- pack ----
        let n_local = orc.total as usize;
        let cap = n_local.max(1);
        // Input hidden is not used by the combine math (scatter reads
        // expert_out, not packed_hidden); a constant row is enough to exercise
        // the pack copy path.
        let hidden_host = vec![bf16::from_f32(0.25f32); num_tokens * HIDDEN];
        let hidden_d = ctx.stream.clone_htod(&hidden_host)?;
        let cursors_d = ctx.stream.alloc_zeros::<i32>(ep)?;
        let packed_hidden_d = ctx.stream.alloc_zeros::<bf16>(cap * HIDDEN)?;
        let packed_slot_d = ctx.stream.alloc_zeros::<i32>(cap)?;
        let packed_weight_d = ctx.stream.alloc_zeros::<f32>(cap)?;
        // SAFETY: shapes match the single-rank [0,ep) window; pack buffers are
        // sized `cap` rows × HIDDEN, cursors/offsets `ep`.
        unsafe {
            moe::dsv4_pack_local_experts_with_slots(
                cache_ptr(&hidden_d, ctx),
                cache_ptr(&indices_d, ctx),
                cache_ptr(&weights_d, ctx),
                cache_ptr(&offsets_d, ctx),
                cache_ptr(&cursors_d, ctx),
                cache_ptr(&packed_hidden_d, ctx),
                cache_ptr(&packed_slot_d, ctx),
                cache_ptr(&packed_weight_d, ctx),
                num_tokens,
                HIDDEN,
                TOPK,
                0,
                ep,
                ctx.stream.cu_stream(),
            )?;
        }
        ctx.sync()?;
        let got_packed_slot = ctx.stream.clone_dtoh(&packed_slot_d)?;
        let got_packed_weight = ctx.stream.clone_dtoh(&packed_weight_d)?;

        // ---- fill m-indices ----
        let row_capacity = orc.aligned_total as usize;
        let m_indices_d = ctx.stream.alloc_zeros::<i32>(row_capacity)?;
        // SAFETY: m_indices has aligned_total rows; counts/offsets are `ep`.
        unsafe {
            moe::dsv4_fill_m_indices_from_counts(
                cache_ptr(&counts_d, ctx),
                cache_ptr(&aligned_offsets_d, ctx),
                cache_ptr(&m_indices_d, ctx),
                ep,
                row_capacity,
                ctx.stream.cu_stream(),
            )?;
        }
        ctx.sync()?;
        let got_m_indices = ctx.stream.clone_dtoh(&m_indices_d)?;

        // ---- synthetic expert outputs (packed-slot order) → scatter/combine ----
        let mut expert_out_host = vec![bf16::ZERO; cap * HIDDEN];
        for slot in 0..n_local {
            for col in 0..HIDDEN {
                expert_out_host[slot * HIDDEN + col] =
                    bf16::from_f32(expert_output(slot, col) as f32);
            }
        }
        let expert_out_d = ctx.stream.clone_htod(&expert_out_host)?;
        let route_out_d = ctx.stream.alloc_zeros::<bf16>(total_routes * HIDDEN)?;
        let combined_d = ctx.stream.alloc_zeros::<bf16>(num_tokens * HIDDEN)?;
        // SAFETY: expert/route outputs are total_routes×HIDDEN, combined
        // num_tokens×HIDDEN; slot/weight arrays cover n_local.
        unsafe {
            moe::dsv4_scatter_all_route_slots(
                cache_ptr(&expert_out_d, ctx),
                cache_ptr(&route_out_d, ctx),
                cache_ptr(&packed_slot_d, ctx),
                cache_ptr(&packed_weight_d, ctx),
                n_local,
                HIDDEN,
                ctx.stream.cu_stream(),
            )?;
            moe::dsv4_combine_route_slot_outputs(
                cache_ptr(&route_out_d, ctx),
                cache_ptr(&combined_d, ctx),
                num_tokens,
                TOPK,
                HIDDEN,
                ctx.stream.cu_stream(),
            )?;
        }
        ctx.sync()?;
        let got_combined_bf16 = ctx.stream.clone_dtoh(&combined_d)?;

        // ---- expected combine from the kernel's own slot map ----
        // Reconstructed with the kernel's exact bf16 rounding and k add order,
        // so this isolates scatter-weighting + combine from router math.
        let mut exp_route = vec![0.0f64; total_routes * HIDDEN];
        for s in 0..n_local {
            let r = got_packed_slot[s] as usize;
            let w = got_packed_weight[s] as f64;
            for col in 0..HIDDEN {
                let term = (w * expert_output(s, col)) as f32;
                exp_route[r * HIDDEN + col] = f64::from(bf16::from_f32(term));
            }
        }
        let mut exp_combined = vec![0.0f64; num_tokens * HIDDEN];
        for t in 0..num_tokens {
            for col in 0..HIDDEN {
                let mut sum = 0.0f64;
                for k in 0..TOPK {
                    sum += exp_route[(t * TOPK + k) * HIDDEN + col];
                }
                exp_combined[t * HIDDEN + col] = f64::from(bf16::from_f32(sum as f32));
            }
        }

        Ok(Bundle {
            mode,
            num_tokens,
            n_local,
            orc,
            got_indices,
            got_weights,
            got_counts,
            got_offsets,
            got_total: got_total[0],
            got_aligned_offsets,
            got_aligned_total: got_aligned_total[0],
            got_packed_slot,
            got_m_indices,
            got_combined_bf16,
            exp_combined,
        })
    }

    /// One pass of every comparison family. `route` allows near-tie exemptions
    /// only in [`Mode::ScoreDecides`].
    #[allow(clippy::too_many_arguments)]
    fn compare_families(
        tag: &str,
        mode: Mode,
        got_idx: &[i32],
        got_w: &[f32],
        got_counts: &[i32],
        got_offsets: &[i32],
        got_total: i32,
        got_aligned_offsets: &[i32],
        got_aligned_total: i32,
        got_pack: &[i32],
        got_m: &[i32],
        got_combined: &[bf16],
        orc: &Oracle,
        exp_combined: &[f64],
        near_tie_eps: f64,
    ) -> Families {
        let mut fam = Families::default();

        // ---- integer route ----
        let mut exempt = 0usize;
        if mode == Mode::ScoreDecides {
            // Set equality per token; a boundary closer than eps may flip.
            for t in 0..orc.boundary_gap.len() {
                let mut want: Vec<i32> = orc.indices[t * TOPK..(t + 1) * TOPK].to_vec();
                let mut have: Vec<i32> = got_idx[t * TOPK..(t + 1) * TOPK].to_vec();
                want.sort_unstable();
                have.sort_unstable();
                if have != want && orc.boundary_gap[t] < near_tie_eps {
                    exempt += 1;
                } else if have != want {
                    fam.route = false;
                    eprintln!(
                        "[{tag}] FAIL route-set token={t} gap={:.3e} >= {near_tie_eps:.0e} have{have:?} want{want:?}",
                        orc.boundary_gap[t],
                    );
                }
            }
        } else {
            for (i, &w) in orc.indices.iter().enumerate() {
                if got_idx[i] != w {
                    fam.route = false;
                    eprintln!("[{tag}] FAIL indices[{i}] got={} want={w}", got_idx[i]);
                }
            }
        }
        fam.near_tie_exemptions = exempt;

        // ---- integer counts / offsets / totals / pack / m-indices ----
        fam.counts = got_counts == orc.counts;
        fam.offsets = got_offsets == orc.offsets && got_aligned_offsets == orc.aligned_offsets;
        fam.totals = got_total == orc.total && got_aligned_total == orc.aligned_total;
        fam.pack = got_pack == &orc.packed_route_slot[..got_pack.len()];
        fam.m_indices = got_m == orc.m_indices;
        if !fam.counts {
            eprintln!("[{tag}] FAIL counts got{got_counts:?} want{:?}", orc.counts);
        }
        if !fam.offsets {
            eprintln!("[{tag}] FAIL offsets");
        }
        if !fam.totals {
            eprintln!(
                "[{tag}] FAIL totals got({got_total},{got_aligned_total}) want({},{})",
                orc.total, orc.aligned_total,
            );
        }
        if !fam.pack {
            eprintln!("[{tag}] FAIL packed_route_slot");
        }
        if !fam.m_indices {
            eprintln!("[{tag}] FAIL m_indices");
        }

        // ---- route weights (f32 kernel vs f64 spec) ----
        const WEIGHT_TOL: f64 = 2e-5;
        fam.weights = got_w
            .iter()
            .zip(orc.weights.iter())
            .all(|(&g, &w)| (g as f64 - w).abs() <= WEIGHT_TOL * w.abs().max(1.0));
        if !fam.weights {
            for (i, (&g, &w)) in got_w.iter().zip(orc.weights.iter()).enumerate() {
                if (g as f64 - w).abs() > WEIGHT_TOL * w.abs().max(1.0) {
                    eprintln!("[{tag}] FAIL weights[{i}] got={g:.6e} want={w:.6e}");
                    break;
                }
            }
        }

        // ---- scatter + combine ----
        let combine_tol = 3.0 * 2f64.powi(-8) * ROUTED_SCALING;
        let mut worst = 0.0f64;
        fam.combine = true;
        for i in 0..exp_combined.len() {
            let g = f64::from(got_combined[i]);
            let diff = (g - exp_combined[i]).abs();
            worst = worst.max(diff);
            if diff > combine_tol {
                fam.combine = false;
                eprintln!(
                    "[{tag}] FAIL combined[{i}] got={g:.5e} want={:.5e}",
                    exp_combined[i],
                );
                break;
            }
        }
        fam.combine_worst = worst;
        fam
    }

    /// Per-family comparator liveness: corrupt exactly one side per family and
    /// require THAT family to fail. A dead comparator would stay green under
    /// its own corruption and be caught here.
    fn verify_negative_teeth(b: &Bundle) -> Result<()> {
        let tag = format!("NEG mode={:?} tokens={} ep-agnostic", b.mode, b.num_tokens);
        let clean = b.compare("clean", NEAR_TIE_EPS);
        // Baseline must be all-green before any sabotage means anything.
        for f in Family::all() {
            anyhow::ensure!(f.get(&clean), "negative baseline already failing `{f:?}`");
        }

        for name in ["route", "counts", "offsets", "pack", "weights", "combine"] {
            let mut expected = b.clone();
            match name {
                "route" => expected.got_indices[0] ^= 1,
                "counts" => expected.orc.counts[0] ^= 1,
                "offsets" => {
                    expected.orc.aligned_offsets[0] =
                        expected.orc.aligned_offsets[0].wrapping_add(7)
                }
                "pack" => {
                    if b.n_local > 1 {
                        expected.orc.packed_route_slot.swap(0, 1);
                    } else {
                        expected.orc.packed_route_slot[0] ^= 1;
                    }
                }
                "weights" => expected.got_weights[0] = expected.got_weights[0].mul_add(2.0, 1.0),
                "combine" => expected.exp_combined[0] += 1.0,
                _ => unreachable!(),
            }
            let fam = expected.compare(name, NEAR_TIE_EPS);
            anyhow::ensure!(
                !Family::by_name(name).get(&fam),
                "negative control did NOT trip `{name}` — comparator is dead"
            );
        }
        eprintln!("[{tag}] negative teeth OK (6 families each trip on their own corruption)");
        Ok(())
    }

    const NEAR_TIE_EPS: f64 = 1e-5;

    /// Everything one pipeline run produced, on the host.
    #[derive(Clone)]
    struct Bundle {
        mode: Mode,
        num_tokens: usize,
        n_local: usize,
        orc: Oracle,
        got_indices: Vec<i32>,
        got_weights: Vec<f32>,
        got_counts: Vec<i32>,
        got_offsets: Vec<i32>,
        got_total: i32,
        got_aligned_offsets: Vec<i32>,
        got_aligned_total: i32,
        got_packed_slot: Vec<i32>,
        got_m_indices: Vec<i32>,
        got_combined_bf16: Vec<bf16>,
        exp_combined: Vec<f64>,
    }

    #[cfg(feature = "cuda")]
    impl Bundle {
        fn compare(&self, tag: &str, near_tie_eps: f64) -> Families {
            compare_families(
                tag,
                self.mode,
                &self.got_indices,
                &self.got_weights,
                &self.got_counts,
                &self.got_offsets,
                self.got_total,
                &self.got_aligned_offsets,
                self.got_aligned_total,
                &self.got_packed_slot[..self.n_local],
                &self.got_m_indices,
                &self.got_combined_bf16,
                &self.orc,
                &self.exp_combined,
                near_tie_eps,
            )
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Family {
        Route,
        Counts,
        Offsets,
        Totals,
        Pack,
        MIndices,
        Weights,
        Combine,
    }

    #[derive(Clone, Copy)]
    struct Families {
        route: bool,
        counts: bool,
        offsets: bool,
        totals: bool,
        pack: bool,
        m_indices: bool,
        weights: bool,
        combine: bool,
        combine_worst: f64,
        near_tie_exemptions: usize,
    }

    impl Default for Families {
        fn default() -> Self {
            Self {
                route: true,
                counts: true,
                offsets: true,
                totals: true,
                pack: true,
                m_indices: true,
                weights: true,
                combine: true,
                combine_worst: 0.0,
                near_tie_exemptions: 0,
            }
        }
    }

    impl Family {
        fn all() -> &'static [Family] {
            &[
                Family::Route,
                Family::Counts,
                Family::Offsets,
                Family::Totals,
                Family::Pack,
                Family::MIndices,
                Family::Weights,
                Family::Combine,
            ]
        }
        fn by_name(s: &str) -> Family {
            match s {
                "route" => Family::Route,
                "counts" => Family::Counts,
                "offsets" => Family::Offsets,
                "pack" => Family::Pack,
                "weights" => Family::Weights,
                "combine" => Family::Combine,
                other => panic!("unknown family {other}"),
            }
        }
        /// Stable index into the per-run fail array, matching `all()`.
        fn idx(self) -> usize {
            match self {
                Family::Route => 0,
                Family::Counts => 1,
                Family::Offsets => 2,
                Family::Totals => 3,
                Family::Pack => 4,
                Family::MIndices => 5,
                Family::Weights => 6,
                Family::Combine => 7,
            }
        }
        /// Human family name used in the shared aggregator.
        fn name(self) -> &'static str {
            match self {
                Family::Route => "route",
                Family::Counts => "counts",
                Family::Offsets => "offsets",
                Family::Totals => "totals",
                Family::Pack => "pack",
                Family::MIndices => "m_indices",
                Family::Weights => "weights",
                Family::Combine => "combine",
            }
        }
        fn get(&self, f: &Families) -> bool {
            match self {
                Family::Route => f.route,
                Family::Counts => f.counts,
                Family::Offsets => f.offsets,
                Family::Totals => f.totals,
                Family::Pack => f.pack,
                Family::MIndices => f.m_indices,
                Family::Weights => f.weights,
                Family::Combine => f.combine,
            }
        }
    }
}
