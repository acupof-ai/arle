//! DSpark drafter ring-varlen attention numeric-parity gate.
//!
//! Standalone kernel harness — NO engine, NO serve. The DSpark draft head's
//! full attention runs the ragged-window ring kernels
//! `nonpaged_prefill_attention_ring_varlen_cuda` (single slot,
//! `qwen35/dspark.rs:948`) and `..._batched_cuda` (c>=2, `:1331`). Neither
//! had a gate, a registry row, or a test; c=8 is where DSpark acceptance is 0%.
//!
//! Both ABIs are the SAME kernel — the batched form only adds a gridZ slot
//! axis with per-slot k/v ring pointers. Each query row t attends non-causally
//! to kv_len[t] keys starting at absolute ring_base[t], physical row
//! `(ring_base[t] + j) % cap`; standard scaled softmax (sm_scale = hd^-0.5).
//! The f64 oracle is written from `infer_model::dspark::draft_attention_windows`
//! semantics (`[lo; block] ++ [kv_len; block]`, HF sliding lower bound clamped
//! at ctx_base), not ported from the CUDA source.
//!
//! Geometry: Qwen3.8-27B-DSpark host config
//! `/data00/Qwen3.8-27B-DSpark/config.json` — 40 q / 8 kv heads, head_dim 128
//! (GQA 5), `block_size = 7`, `sliding_window = null`. With no window
//! `ctx_cap = rope_cap = min(max_seq_len, max_total_tokens) + block_size`
//! (`infer-model/src/dspark.rs:81-89`, `qwen35/dspark.rs:603-604`); the serve
//! default `max_total_tokens = 32768` (`infer-core/src/lib.rs:87`) makes the
//! ctx ring request-length (32775 rows). Production is FULL attention over
//! thousands of keys with NO ring wraparound.
//!
//! Three worlds, each at B=1 and B=8:
//!
//! - **production long-context** (window None, modulus 32775): per-slot
//!   kv_len spread over long contexts (hundreds up to 4096, unequal), qlen =
//!   block 7, ring_base 0 — the regime c=8 runs. The oracle sums ALL key
//!   columns (softmax is over the full set); rows/heads are sampled across
//!   slots for host time, but every sampled row covers ALL 40 heads and ALL
//!   128 dims. Rings are hash-filled deterministically (cheap, no giant RNG).
//! - **block_size_cap clamp** (`qwen35/dspark.rs:462-471`): the config block
//!   clamped to 4 by `--dspark-block-size`, exercising the override path.
//! - **small-wrap contract** (synthetic cap 12 / window 6 / block 7): NOT
//!   production — a tiny ring forcing physical wraparound through row 0 and a
//!   window at kv_len==cap, checking the modulus arithmetic the production
//!   regime never reaches. Window 6 keeps the ABI invariant kv_len <= cap
//!   (6+7-1 = 12).
//!
//! Every world also compares the single and batched ABIs on identical inputs;
//! they share one kernel, so they must match element-for-element.
//!
//! `--negative-control` applies one corruption per family — single output,
//! batched output, single-vs-batched agreement — aggregated across all worlds,
//! asserting each trips ONLY its comparator. Prints NEGATIVE CONTROL OK, exits 0.
//!
//! Run on a pod:
//! `INFER_CUDA_DEVICE=<free-gpu> target/release/examples/dspark_draft_attn_parity`

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
        eprintln!("dspark_draft_attn_parity is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::{Result, ensure};
    use cuda_kernels::attention::{
        nonpaged_prefill_attention_ring_varlen_batched_raw,
        nonpaged_prefill_attention_ring_varlen_raw,
    };
    use cuda_kernels::prelude::DeviceContext;
    use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
    use half::bf16;

    const Q_HEADS: usize = 40;
    const KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 128;
    const BLOCK: usize = 7; // /data00/Qwen3.8-27B-DSpark/config.json block_size
    const CLAMP_BLOCK: usize = 4; // --dspark-block-size clamp path
    const SLOTS: &[usize] = &[1, 8];
    const SEED: u64 = 0xD57A_2710_D8A1_FEED;
    const Q_DIM: usize = Q_HEADS * HEAD_DIM;

    // Production: serve default max_total_tokens (infer-core/src/lib.rs:87);
    // ctx_cap = min(max_seq_len, 32768) + block = 32775.
    const SERVE_MAX_TOTAL_TOKENS: usize = 32_768;
    const PROD_CAP: usize = SERVE_MAX_TOTAL_TOKENS + BLOCK;
    // Per-slot long-context kv_len (window None → base 0, no wrap).
    const PROD_KVLENS: [usize; 8] = [256, 1024, 4096, 300, 2048, 768, 4000, 512];
    // Small synthetic wrap world (kernel-contract check only). block 7 keeps
    // the ABI invariant kv_len <= cap: window 6 → max kv_len = 6+7-1 = 12.
    const WRAP_CAP: usize = 12;
    const WRAP_WINDOW: usize = 6;

    // bf16 dot products (hd=128) vs f64 softmax.
    const REL_L2_MAX: f64 = 4e-2;
    const ABS_FLOOR: f64 = 2e-2;
    const ABS_SLOPE: f64 = 6e-2;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Sabotage {
        None,
        Single,
        Batched,
        Cross,
    }

    /// Deterministic hash -> bf16 in [-1,1); used to fill rings cheaply without
    /// a giant RNG draw. Stable per (slot-salt, linear index).
    fn hash_bf16(salt: u64, idx: usize) -> f64 {
        let mut z = (salt ^ 0x9E37_79B9_7F4A_7C15)
            .wrapping_add((idx as u64).wrapping_mul(0x0010_0000_00B1))
            .wrapping_add(1);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let u = (z >> 40) as f64 / (1u64 << 24) as f64; // [0,1)
        let v = (u * 2.0 - 1.0) as f32;
        f64::from(bf16::from_f32(v))
    }

    fn rng_normal(seed: u64, idx: &mut u64) -> f64 {
        let mut z = seed.wrapping_add(*idx).wrapping_add(1);
        *idx = idx.wrapping_add(1);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let u1 = ((z >> 40) as f64 / (1u64 << 24) as f64).max(1e-7);
        // second draw
        let mut z2 = seed.wrapping_add(*idx).wrapping_add(0x9E37);
        *idx = idx.wrapping_add(1);
        z2 = (z2 ^ (z2 >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z2 = (z2 ^ (z2 >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z2 ^= z2 >> 31;
        let u2 = (z2 >> 40) as f64 / (1u64 << 24) as f64;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// HF sliding-window lower bound.
    fn window_lo(window: Option<usize>, pos: usize) -> usize {
        window.map_or(0, |w| pos.saturating_sub(w.saturating_sub(1)))
    }

    /// `[lo; block] ++ [kv_len; block]` per `draft_attention_windows`.
    fn draft_windows(
        window: Option<usize>,
        start: usize,
        block: usize,
        ctx_base: usize,
    ) -> Vec<i32> {
        let mut win = vec![0i32; 2 * block];
        for row in 0..block {
            let lo = window_lo(window, start + row).max(ctx_base);
            let kv_len = start + block - lo;
            win[row] = lo as i32;
            win[block + row] = kv_len as i32;
        }
        win
    }

    /// Explicit window from per-row (ring_base, kv_len) pairs.
    fn explicit_windows(rows: &[(usize, usize)], block: usize) -> Vec<i32> {
        debug_assert_eq!(rows.len(), block);
        let mut win = vec![0i32; 2 * block];
        for (r, (base, len)) in rows.iter().enumerate() {
            win[r] = *base as i32;
            win[block + r] = *len as i32;
        }
        win
    }

    /// f64 attention for one (row, q_head): sums ALL key columns, so it matches
    /// the device output exactly (softmax over the whole key set). Physical row
    /// `(base+j)%cap`, non-causal scaled softmax, GQA kv-head mapping.
    fn oracle_row_head(
        q: &[bf16],
        k_ring: &[bf16],
        v_ring: &[bf16],
        win: &[i32],
        block: usize,
        cap: usize,
        t: usize,
        qh: usize,
        dst: &mut [f64],
    ) {
        debug_assert_eq!(dst.len(), HEAD_DIM);
        let sm = (HEAD_DIM as f64).sqrt().recip();
        let qdim = Q_HEADS * HEAD_DIM;
        let base = win[t] as usize;
        let kv_len = win[block + t] as usize;
        let kvh = qh / (Q_HEADS / KV_HEADS);
        let qv: Vec<f64> = (0..HEAD_DIM)
            .map(|d| f64::from(q[t * qdim + qh * HEAD_DIM + d]))
            .collect();
        // Two-pass softmax, all kv_len keys.
        let mut logits = vec![0f64; kv_len];
        let mut m = f64::NEG_INFINITY;
        for (j, lg) in logits.iter_mut().enumerate() {
            let row = (base + j) % cap;
            let mut dot = 0f64;
            for d in 0..HEAD_DIM {
                dot += qv[d] * f64::from(k_ring[kvh * cap * HEAD_DIM + row * HEAD_DIM + d]);
            }
            let s = dot * sm;
            *lg = s;
            if s > m {
                m = s;
            }
        }
        let mut weights = vec![0f64; kv_len];
        let mut denom = 0f64;
        for (w, lg) in weights.iter_mut().zip(logits.iter()) {
            let e = (lg - m).exp();
            *w = e;
            denom += e;
        }
        if denom <= 0.0 {
            for x in dst.iter_mut() {
                *x = 0.0;
            }
            return;
        }
        for (d, x) in dst.iter_mut().enumerate() {
            let mut acc = 0f64;
            for (j, w) in weights.iter().enumerate() {
                let row = (base + j) % cap;
                acc += *w * f64::from(v_ring[kvh * cap * HEAD_DIM + row * HEAD_DIM + d]);
            }
            *x = acc / denom;
        }
    }

    type HeadOut = ((usize, usize), [f64; HEAD_DIM]);

    /// Parallel f64 oracle for the selected (row, head) set over ALL key
    /// columns. `pick` lists (t, qh); unpicked outputs stay NaN and are not
    /// compared. Every picked row/head covers all 128 dims.
    fn oracle_pick(
        q: &[bf16],
        k_ring: &[bf16],
        v_ring: &[bf16],
        win: &[i32],
        block: usize,
        cap: usize,
        pick: &[(usize, usize)],
    ) -> Vec<f64> {
        let mut out = vec![f64::NAN; block * Q_DIM];
        let jobs: Vec<(usize, usize)> = pick.to_vec();
        let n = jobs.len();
        let chunk = n.div_ceil(4);
        let mut partial: Vec<Vec<HeadOut>> = Vec::new();
        std::thread::scope(|sc| {
            let mut handles = Vec::new();
            for c in 0..4 {
                let lo = c * chunk;
                let hi = (lo + chunk).min(n);
                let my = &jobs[lo..hi];
                handles.push(sc.spawn(move || {
                    let mut res = Vec::new();
                    for &(t, qh) in my {
                        let mut dst = [0f64; HEAD_DIM];
                        oracle_row_head(q, k_ring, v_ring, win, block, cap, t, qh, &mut dst);
                        res.push(((t, qh), dst));
                    }
                    res
                }));
            }
            for h in handles {
                partial.push(h.join().unwrap());
            }
        });
        for res in partial {
            for ((t, qh), dst) in res {
                out[t * Q_DIM + qh * HEAD_DIM..t * Q_DIM + (qh + 1) * HEAD_DIM]
                    .copy_from_slice(&dst);
            }
        }
        out
    }

    struct Slot {
        q: Vec<bf16>,
        k_ring: Vec<bf16>,
        v_ring: Vec<bf16>,
        win: Vec<i32>,
        /// Which (row,head) outputs are compared; None = all.
        pick: Option<Vec<(usize, usize)>>,
    }

    fn ptr_table_rw<T>(ctx: &DeviceContext, slices: &mut [CudaSlice<T>]) -> Result<CudaSlice<u64>> {
        let ptrs: Vec<u64> = slices
            .iter_mut()
            .map(|s| s.device_ptr_mut(&ctx.stream).0)
            .collect();
        Ok(ctx.stream.clone_htod(&ptrs)?)
    }

    /// Hash-fill the reachable per-head prefix `[0, rows)` of a modulus-sized
    /// ring; the tail stays zero. Production windows start at ring_base 0 and
    /// never wrap, so only the prefix is read; the buffer keeps modulus stride.
    fn fill_ring_prefix(salt: u64, ring: &mut [bf16], cap: usize, rows: usize) {
        for kvh in 0..KV_HEADS {
            for row in 0..rows {
                for d in 0..HEAD_DIM {
                    let i = kvh * cap * HEAD_DIM + row * HEAD_DIM + d;
                    ring[i] = bf16::from_f32(hash_bf16(salt, i) as f32);
                }
            }
        }
    }

    fn zero_ring(n: usize) -> Vec<bf16> {
        vec![bf16::from_f32(0.0); n]
    }

    /// Production long-context world: window None, modulus PROD_CAP, varied
    /// kv_len up to 4096, qlen=block. Only the kv_len prefix of each ring is
    /// reached, but the buffer is allocated at modulus so the stride matches
    /// production. Rows/heads are sampled across slots (all 128 dims per pick).
    fn build_slots_production(
        seed: u64,
        b: usize,
        block: usize,
        clamp: bool,
    ) -> (usize, Vec<Slot>) {
        let cap = if clamp {
            SERVE_MAX_TOTAL_TOKENS + CLAMP_BLOCK
        } else {
            PROD_CAP
        };
        let mut slots = Vec::with_capacity(b);
        for s in 0..b {
            let salt = seed ^ ((s as u64) << 12) ^ if clamp { 0xC1_A9 } else { 0 };
            let kv_len = PROD_KVLENS[s % PROD_KVLENS.len()];
            let rows = vec![(0usize, kv_len); block];
            let win = explicit_windows(&rows, block);
            let q: Vec<bf16> = {
                let mut idx = 0u64;
                (0..block * Q_DIM)
                    .map(|_| bf16::from_f32(rng_normal(salt, &mut idx) as f32 * 0.4))
                    .collect()
            };
            let ring_n = KV_HEADS * cap * HEAD_DIM;
            let mut k_ring = zero_ring(ring_n);
            let mut v_ring = zero_ring(ring_n);
            fill_ring_prefix(salt ^ 0x4B, &mut k_ring, cap, kv_len);
            fill_ring_prefix(salt ^ 0x56, &mut v_ring, cap, kv_len);
            // Sample rows/heads: 2 of `block` rows × 8 q heads per slot, but
            // ensure across the batch every row and every q-head is covered at
            // least once (s=0 covers full head set on row 0).
            let pick = if s == 0 {
                (0..Q_HEADS).map(|qh| (0, qh)).collect()
            } else {
                let rows = [s % block, (s + 1) % block];
                let heads = [
                    (s * 5) % Q_HEADS,
                    (s * 11 + 3) % Q_HEADS,
                    (s * 17 + 7) % Q_HEADS,
                    (s * 23 + 1) % Q_HEADS,
                    (s * 29 + 9) % Q_HEADS,
                    (s * 31 + 13) % Q_HEADS,
                    (s * 37 + 19) % Q_HEADS,
                    (s * 7 + 2) % Q_HEADS,
                ];
                rows.iter()
                    .flat_map(|&t| heads.iter().map(move |&qh| (t, qh)))
                    .collect()
            };
            slots.push(Slot {
                q,
                k_ring,
                v_ring,
                win,
                pick: Some(pick),
            });
        }
        (cap, slots)
    }

    /// Small wrap world: cap 12 / window 8 / block 7, dense wrap + boundary.
    fn build_slots_wrap(seed: u64, b: usize) -> (usize, Vec<Slot>) {
        let cap = WRAP_CAP;
        let window = Some(WRAP_WINDOW);
        let mut starts = Vec::new();
        // block 7 ≤ cap: choose starts so windows wrap, plus an explicit
        // kv_len==cap full-ring slot.
        let candidates = [11usize, 3, 9, 6, 1, 10, 4, 8];
        for s in 0..b {
            starts.push(candidates[s % candidates.len()]);
        }
        let mut slots = Vec::with_capacity(b);
        for (s, &start) in starts.iter().enumerate() {
            let salt = seed ^ 0x99 ^ ((s as u64) << 12);
            let ctx_base = if s % 3 == 0 { 2 } else { 0 };
            let win = if s % 4 == 1 {
                explicit_windows(&[(0usize, cap); BLOCK], BLOCK)
            } else {
                draft_windows(window, start, BLOCK, ctx_base)
            };
            let q: Vec<bf16> = {
                let mut idx = 0u64;
                (0..BLOCK * Q_DIM)
                    .map(|_| bf16::from_f32(rng_normal(salt, &mut idx) as f32 * 0.4))
                    .collect()
            };
            let ring_n = KV_HEADS * cap * HEAD_DIM;
            let mut k_ring = zero_ring(ring_n);
            let mut v_ring = zero_ring(ring_n);
            fill_ring_prefix(salt ^ 0x4B, &mut k_ring, cap, cap);
            fill_ring_prefix(salt ^ 0x56, &mut v_ring, cap, cap);
            slots.push(Slot {
                q,
                k_ring,
                v_ring,
                win,
                pick: None,
            });
        }
        (cap, slots)
    }

    fn run_single(
        ctx: &DeviceContext,
        slots: &[Slot],
        cap: usize,
        block: usize,
    ) -> Result<Vec<Vec<bf16>>> {
        let mut outs = Vec::with_capacity(slots.len());
        for s in slots {
            let q_d = ctx.stream.clone_htod(&s.q)?;
            let k_d = ctx.stream.clone_htod(&s.k_ring)?;
            let v_d = ctx.stream.clone_htod(&s.v_ring)?;
            let win_d = ctx.stream.clone_htod(&s.win)?;
            let mut o_d = ctx.stream.alloc_zeros::<bf16>(block * Q_DIM)?;
            {
                let (qp, _) = q_d.device_ptr(&ctx.stream);
                let (kp, _) = k_d.device_ptr(&ctx.stream);
                let (vp, _) = v_d.device_ptr(&ctx.stream);
                let (bp, _) = win_d.device_ptr(&ctx.stream);
                let (op, _) = o_d.device_ptr_mut(&ctx.stream);
                nonpaged_prefill_attention_ring_varlen_raw(
                    &ctx.stream,
                    qp,
                    kp,
                    vp,
                    op,
                    Q_HEADS,
                    KV_HEADS,
                    HEAD_DIM,
                    block,
                    bp,
                    bp + (block * std::mem::size_of::<i32>()) as u64,
                    cap,
                    (HEAD_DIM as f64).sqrt().recip() as f32,
                )?;
                ctx.sync()?;
            }
            outs.push(ctx.stream.clone_dtoh(&o_d)?);
        }
        Ok(outs)
    }

    fn run_batched(
        ctx: &DeviceContext,
        slots: &[Slot],
        cap: usize,
        block: usize,
    ) -> Result<Vec<Vec<bf16>>> {
        let b = slots.len();
        let q_flat: Vec<bf16> = slots.iter().flat_map(|s| s.q.iter().copied()).collect();
        let q_d = ctx.stream.clone_htod(&q_flat)?;
        let mut k_d: Vec<_> = slots
            .iter()
            .map(|s| ctx.stream.clone_htod(&s.k_ring))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut v_d: Vec<_> = slots
            .iter()
            .map(|s| ctx.stream.clone_htod(&s.v_ring))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut win_flat = vec![0i32; 2 * b * block];
        for (s, sl) in slots.iter().enumerate() {
            win_flat[s * block..(s + 1) * block].copy_from_slice(&sl.win[..block]);
            win_flat[b * block + s * block..b * block + (s + 1) * block]
                .copy_from_slice(&sl.win[block..]);
        }
        let win_d = ctx.stream.clone_htod(&win_flat)?;
        let k_tbl = ptr_table_rw(ctx, &mut k_d)?;
        let v_tbl = ptr_table_rw(ctx, &mut v_d)?;
        let mut o_d = ctx.stream.alloc_zeros::<bf16>(b * block * Q_DIM)?;
        {
            let (qp, _) = q_d.device_ptr(&ctx.stream);
            let (kt, _) = k_tbl.device_ptr(&ctx.stream);
            let (vt, _) = v_tbl.device_ptr(&ctx.stream);
            let (wp, _) = win_d.device_ptr(&ctx.stream);
            let (op, _) = o_d.device_ptr_mut(&ctx.stream);
            nonpaged_prefill_attention_ring_varlen_batched_raw(
                &ctx.stream,
                qp,
                kt,
                vt,
                op,
                Q_HEADS,
                KV_HEADS,
                HEAD_DIM,
                block,
                b,
                wp,
                wp + (b * block * std::mem::size_of::<i32>()) as u64,
                cap,
                (HEAD_DIM as f64).sqrt().recip() as f32,
            )?;
            ctx.sync()?;
        }
        let flat = ctx.stream.clone_dtoh(&o_d)?;
        Ok((0..b)
            .map(|s| flat[s * block * Q_DIM..(s + 1) * block * Q_DIM].to_vec())
            .collect())
    }

    fn bound(got: &[f64], want: &[f64], bias: f64) -> (bool, f64, usize) {
        let mut viol = 0usize;
        let (mut ds, mut rs) = (0f64, 0f64);
        for (g, w0) in got.iter().zip(want) {
            let w = w0 + bias;
            let d = g - w;
            ds += d * d;
            rs += w * w;
            if d.abs() > ABS_FLOOR + ABS_SLOPE * w.abs() {
                viol += 1;
            }
        }
        let rel = (ds / rs.max(1e-12)).sqrt();
        (rel < REL_L2_MAX && viol == 0, rel, viol)
    }

    /// One world's neutral comparison data, computed ONCE: picked single and
    /// batched outputs vs the shared f64 oracle, plus full outputs for the
    /// exact single-vs-batched check. Sabotage conditions are evaluated
    /// host-side against these (a comparator-input bias), so the device runs
    /// once per world.
    struct World {
        s_got: Vec<f64>,
        b_got: Vec<f64>,
        want: Vec<f64>,
        cross_maxdiff: f64,
    }

    fn gather_world(
        ctx: &DeviceContext,
        label: &str,
        cap: usize,
        slots: &[Slot],
        block: usize,
    ) -> Result<World> {
        let b = slots.len();
        let single = run_single(ctx, slots, cap, block)?;
        let batched = run_batched(ctx, slots, cap, block)?;

        let mut s_got = Vec::new();
        let mut b_got = Vec::new();
        let mut want = Vec::new();
        let mut cross_maxdiff = 0f64;
        for (s, sl) in slots.iter().enumerate() {
            let pick: Vec<(usize, usize)> = match &sl.pick {
                Some(p) => p.clone(),
                None => (0..block)
                    .flat_map(|t| (0..Q_HEADS).map(move |qh| (t, qh)))
                    .collect(),
            };
            let oracle = oracle_pick(&sl.q, &sl.k_ring, &sl.v_ring, &sl.win, block, cap, &pick);
            let sg: Vec<f64> = single[s].iter().map(|v| f64::from(*v)).collect();
            let bg: Vec<f64> = batched[s].iter().map(|v| f64::from(*v)).collect();
            for &(t, qh) in &pick {
                for d in 0..HEAD_DIM {
                    let idx = t * Q_DIM + qh * HEAD_DIM + d;
                    s_got.push(sg[idx]);
                    b_got.push(bg[idx]);
                    want.push(oracle[idx]);
                }
            }
            for (a, c) in single[s].iter().zip(&batched[s]) {
                cross_maxdiff = cross_maxdiff.max((f64::from(*a) - f64::from(*c)).abs());
            }
        }

        let (s_ok, s_rel, s_v) = bound(&s_got, &want, 0.0);
        let (b_ok, b_rel, b_v) = bound(&b_got, &want, 0.0);
        let cross_ok = cross_maxdiff == 0.0;
        eprintln!(
            "[{label} B={b}] single relL2={s_rel:.3e} viol={s_v} {} | batched relL2={b_rel:.3e} viol={b_v} {} | cross maxdiff={cross_maxdiff:.3e} {cross_ok}",
            s_ok, b_ok
        );
        Ok(World {
            s_got,
            b_got,
            want,
            cross_maxdiff,
        })
    }

    fn evaluate(worlds: &[World], sabotage: Sabotage) -> (bool, bool, bool) {
        let (mut s_all, mut b_all, mut cross_all) = (true, true, true);
        for w in worlds {
            let s_bias = if sabotage == Sabotage::Single {
                1.0
            } else {
                0.0
            };
            let b_bias = if sabotage == Sabotage::Batched {
                1.0
            } else {
                0.0
            };
            let (s_ok, _, _) = bound(&w.s_got, &w.want, s_bias);
            let (b_ok, _, _) = bound(&w.b_got, &w.want, b_bias);
            let cross_diff = w.cross_maxdiff
                + if sabotage == Sabotage::Cross {
                    1.0
                } else {
                    0.0
                };
            s_all &= s_ok;
            b_all &= b_ok;
            cross_all &= cross_diff == 0.0;
        }
        (s_all, b_all, cross_all)
    }

    fn gather(ctx: &DeviceContext, b: usize) -> Result<Vec<World>> {
        let (wc, ws) = build_slots_wrap(SEED, b);
        let (pc, ps) = build_slots_production(SEED, b, BLOCK, false);
        let (cc, cs) = build_slots_production(SEED ^ 0x3355, b, CLAMP_BLOCK, true);
        Ok(vec![
            gather_world(ctx, "wrap-contract", wc, &ws, BLOCK)?,
            gather_world(ctx, "longctx", pc, &ps, BLOCK)?,
            gather_world(ctx, "blockcap4", cc, &cs, CLAMP_BLOCK)?,
        ])
    }

    pub(super) fn run(negative: bool) -> Result<()> {
        let ctx = DeviceContext::new()?;
        eprintln!(
            "[dspark-draft-attn] device={} build={}{}",
            ctx.ordinal(),
            cuda_kernels::KERNEL_BUILD_ID,
            if negative { " NEGATIVE-CONTROL" } else { "" }
        );

        if negative {
            for &b in SLOTS {
                let label = format!("B={b}");
                let worlds = gather(&ctx, b)?;
                let (s0, b0, c0) = evaluate(&worlds, Sabotage::None);
                ensure!(s0 && b0 && c0, "{label}: baseline not green before teeth");
                let (s1, b1, c1) = evaluate(&worlds, Sabotage::Single);
                ensure!(!s1 && b1 && c1, "{label}: single tooth dead or leaked");
                let (s2, b2, c2) = evaluate(&worlds, Sabotage::Batched);
                ensure!(s2 && !b2 && c2, "{label}: batched tooth dead or leaked");
                let (s3, b3, c3) = evaluate(&worlds, Sabotage::Cross);
                ensure!(s3 && b3 && !c3, "{label}: cross tooth dead or leaked");
                eprintln!("[{label}] teeth OK");
            }
            eprintln!("[dspark-draft-attn] NEGATIVE CONTROL OK");
            return Ok(());
        }

        let mut all = true;
        for &b in SLOTS {
            let worlds = gather(&ctx, b)?;
            let (s, bk, c) = evaluate(&worlds, Sabotage::None);
            all &= s && bk && c;
        }
        ensure!(all, "dspark_draft_attn_parity FAILED");
        eprintln!("[dspark-draft-attn] ALL PASS");
        Ok(())
    }
}
