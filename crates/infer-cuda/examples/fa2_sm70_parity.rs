//! FA2 sm_70 (V100) attention numeric-parity gate.
//!
//! Standalone kernel harness — NO engine, NO serve. Covers the hand-written
//! `arle_fa2_sm70_attention_cuda`
//! (`crates/cuda-kernels/csrc/attention/arle_fa2_sm70.cu`, BF16 I/O, FP16
//! half2 math, FP32 accumulation, Br=16/Bc=64 online softmax, causal
//! chunked-prefill). It replaces the dense seq4/q2-kv1/hd256-only
//! `fa2_sm70_matches_host_reference` unit test with production geometry.
//!
//! Production call map (full-tree, 2026-09-11):
//! - `infer-cuda/src/qwen35_attention.rs:201` single-row decode
//!   (`seq_len==1`, non-graph) and `:281` multi-token prefill
//!   (`seq_len>1`, no FA3), both inside `full_attention_into`.
//! - `full_attention_into`'s only callers are the MTP spec head
//!   (`qwen35_spec.rs:77` single, `:316` batched per-row loop). Regular
//!   serving prefill/decode goes through `full_attention_paged`
//!   (`qwen35_forward.rs:477`), whose sm<80 fallback is the separate paged
//!   `paged_attention_v1_raw` (qwen35_attention.rs paged branch), NOT this
//!   kernel: arle_fa2_sm70 reads a contiguous head-major cache and has no
//!   page-table ABI. No non-identity page table is applicable here.
//! - The kernel grid is `(num_q_heads, ceil(seq_len/16))` with NO batch
//!   axis; serving B>1 issues one launch per slot. This harness does the
//!   same, with an independent per-row cache/kv_len.
//!
//! Geometry: all CUDA Qwen3.5/3.6 full-attn targets are head_dim 256 —
//! 0.8B 8/2, 4B 16/4, 27B 24/4, 35B-A3B 16/2 (GQA 4..6). Cases cover the
//! smallest (8/2) and largest (24/4) ratios at 256 plus a head_dim=128
//! contract row, at B=1 and B=8 rows. Families: decode (q=1, long history),
//! causal chunked prefill (q_start>0, lengths off the 16/64 tiles), a long
//! 4096 case, and hdim128.
//!
//! f64 oracle is SDPA over the bf16-rounded inputs: q token i at absolute
//! position q_start+i attends keys 0..=q_start+i, GQA head h -> h/ratio.
//! Every output column is compared with rel-L2 plus a slope+floor
//! elementwise band. `--negative-control=<family>` corrupts that family's
//! device K and asserts it trips only its own comparator; the bare flag
//! corrupts every family. `--kernel-build-id` prints the build identity.
//!
//! Pending-remote: V100 (sm_70). On sm80+ the harness exits skipped.
//!
//! Run on the V100 box:
//!   cargo build --release -p infer-cuda --features cuda --example fa2_sm70_parity
//!   target/release/examples/fa2_sm70_parity --kernel-build-id
//!   INFER_CUDA_DEVICE=<free-v100> target/release/examples/fa2_sm70_parity

fn main() -> anyhow::Result<()> {
    if std::env::args().nth(1).as_deref() == Some("--kernel-build-id") {
        println!("{}", cuda_kernels::KERNEL_BUILD_ID);
        return Ok(());
    }
    let family =
        std::env::args().find_map(|a| a.strip_prefix("--negative-control=").map(str::to_string));
    let corrupt_all = std::env::args().any(|a| a == "--negative-control");
    let corrupt = family.or_else(|| corrupt_all.then_some("all".to_string()));
    real::run(corrupt)
}

#[cfg(feature = "cuda")]
#[path = "support/attn_common.rs"]
#[allow(dead_code, clippy::needless_range_loop)]
// shared gate helper module; only a subset used here
mod attn_common;

#[cfg(not(feature = "cuda"))]
mod real {
    pub(super) fn run(_corrupt: Option<String>) -> anyhow::Result<()> {
        eprintln!("fa2_sm70_parity is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::{Result, bail};
    use cuda_kernels::attention::fa2_sm70_attention_raw;
    use cuda_kernels::prelude::DeviceContext;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    use crate::attn_common::{Rng, Tol, bf, compare_rows, metrics_pass};

    // FP16 half2 dots over 256 bf16 dims leave scattered outlier columns;
    // floor covers single-column roundoff, slope tracks magnitude. Both
    // operands are bf16 and accumulation is f32, so the only quant ingredient
    // is one bf16 operand step (2^-8 ≈ 3.9e-3 worst / 2.3e-3 rms) plus the
    // single bf16 output store; the 4e-2/6e-2/2e-2 tuple is ~10x that floor and
    // absorbs fp16 half2 dot + online-softmax order differences that are not
    // bounded in closed form. Clean-run-set, needs a supremum.
    const TOL: Tol = Tol {
        rel_l2: 4e-2,
        slope: 6e-2,
        floor: 2e-2,
        max_viol_frac: 1e-3,
    };

    #[derive(Clone, Copy)]
    struct Geo {
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    }

    struct Row {
        seq_len: usize,
        kv_len: usize,
        max_seq_len: usize,
        // When Some, compare only these query token rows (all heads, every
        // column); long prefills make a full f64 oracle infeasible, so the
        // long case samples rows spanning every Br/Bc tile. None = all rows.
        sample_q: Option<Vec<usize>>,
    }

    struct Family {
        name: &'static str,
        geo: Geo,
        rows: Vec<Row>,
    }

    fn families() -> Vec<Family> {
        let g08 = Geo {
            q_heads: 8,
            kv_heads: 2,
            head_dim: 256,
        }; // 0.8B, GQA 4
        let g27 = Geo {
            q_heads: 24,
            kv_heads: 4,
            head_dim: 256,
        }; // 27B, GQA 6
        let g128 = Geo {
            q_heads: 8,
            kv_heads: 2,
            head_dim: 128,
        };
        vec![
            // B=1 single row (first) then B=8 independent decode rows with
            // unequal, tile-unaligned histories.
            Family {
                name: "decode",
                geo: g08,
                rows: vec![
                    Row {
                        seq_len: 1,
                        kv_len: 317,
                        max_seq_len: 512,
                        sample_q: None,
                    },
                    Row {
                        seq_len: 1,
                        kv_len: 1024,
                        max_seq_len: 2048,
                        sample_q: None,
                    },
                    Row {
                        seq_len: 1,
                        kv_len: 63,
                        max_seq_len: 128,
                        sample_q: None,
                    },
                    Row {
                        seq_len: 1,
                        kv_len: 509,
                        max_seq_len: 1024,
                        sample_q: None,
                    },
                    Row {
                        seq_len: 1,
                        kv_len: 2000,
                        max_seq_len: 4096,
                        sample_q: None,
                    },
                    Row {
                        seq_len: 1,
                        kv_len: 129,
                        max_seq_len: 256,
                        sample_q: None,
                    },
                    Row {
                        seq_len: 1,
                        kv_len: 777,
                        max_seq_len: 1024,
                        sample_q: None,
                    },
                    Row {
                        seq_len: 1,
                        kv_len: 256,
                        max_seq_len: 512,
                        sample_q: None,
                    },
                ],
            },
            // Causal chunked prefill: q_start = kv_len-seq_len > 0; lengths
            // deliberately off the 16-row Br tile and 64-key Bc tile.
            Family {
                name: "prefill",
                geo: g08,
                rows: vec![
                    Row {
                        seq_len: 17,
                        kv_len: 129,
                        max_seq_len: 256,
                        sample_q: None,
                    },
                    Row {
                        seq_len: 3,
                        kv_len: 67,
                        max_seq_len: 128,
                        sample_q: None,
                    },
                    Row {
                        seq_len: 33,
                        kv_len: 201,
                        max_seq_len: 512,
                        sample_q: None,
                    },
                    Row {
                        seq_len: 65,
                        kv_len: 500,
                        max_seq_len: 1024,
                        sample_q: None,
                    },
                    Row {
                        seq_len: 129,
                        kv_len: 777,
                        max_seq_len: 1024,
                        sample_q: None,
                    },
                    Row {
                        seq_len: 31,
                        kv_len: 300,
                        max_seq_len: 512,
                        sample_q: None,
                    },
                    Row {
                        seq_len: 97,
                        kv_len: 401,
                        max_seq_len: 1024,
                        sample_q: None,
                    },
                    Row {
                        seq_len: 19,
                        kv_len: 64,
                        max_seq_len: 128,
                        sample_q: None,
                    },
                ],
            },
            // Long decode + causal prefill across many Bc tiles, GQA 6.
            Family {
                name: "long4096",
                geo: g27,
                rows: vec![
                    Row {
                        seq_len: 1,
                        kv_len: 4096,
                        max_seq_len: 4096,
                        sample_q: None,
                    },
                    Row {
                        seq_len: 4093,
                        kv_len: 4096,
                        max_seq_len: 4096,
                        // First/last q rows plus one per ~512-key span,
                        // covering every 64-key Bc tile boundary region.
                        sample_q: Some(
                            [
                                0usize, 15, 16, 63, 64, 127, 128, 255, 511, 1023, 2047, 3071, 4079,
                                4092,
                            ]
                            .to_vec(),
                        ),
                    },
                ],
            },
            // head_dim=128 kernel-contract path (no shipping CUDA config).
            Family {
                name: "hdim128",
                geo: g128,
                rows: vec![
                    Row {
                        seq_len: 1,
                        kv_len: 255,
                        max_seq_len: 512,
                        sample_q: None,
                    },
                    Row {
                        seq_len: 19,
                        kv_len: 133,
                        max_seq_len: 256,
                        sample_q: None,
                    },
                ],
            },
        ]
    }

    struct Inputs {
        q: Vec<bf16>,
        k: Vec<bf16>,
        v: Vec<bf16>,
    }

    // Deterministic bf16 inputs for one launch. Live cache region only; the
    // tail (kv_len..max_seq_len) is left as a junk fill to prove the causal
    // bound and kernel never read past kv_len.
    #[allow(clippy::needless_range_loop)]
    fn make_inputs(geo: Geo, row: &Row, seed: u64, corrupt: bool) -> Inputs {
        let Geo {
            q_heads,
            kv_heads,
            head_dim: d,
        } = geo;
        let mut rng = Rng::new(seed);
        let q: Vec<bf16> = (0..row.seq_len * q_heads * d)
            .map(|_| bf(rng.unit() * 0.3))
            .collect();
        let mut k = vec![bf16::from_f32(0.0); kv_heads * row.max_seq_len * d];
        let mut v = vec![bf16::from_f32(0.0); kv_heads * row.max_seq_len * d];
        for h in 0..kv_heads {
            for tok in 0..row.max_seq_len {
                for dd in 0..d {
                    let idx = (h * row.max_seq_len + tok) * d + dd;
                    if tok < row.kv_len {
                        k[idx] = bf(rng.unit() * 0.3);
                        v[idx] = bf(rng.unit() * 0.3);
                    } else {
                        // Poison tail: anything reading past kv_len breaks.
                        k[idx] = bf16::from_f32(9.0);
                        v[idx] = bf16::from_f32(9.0);
                        let _ = rng.unit();
                        let _ = rng.unit();
                    }
                }
            }
        }
        if corrupt {
            // Spike the first live key vector of kv head 0 (every q row
            // attends key 0). Device K only; oracle stays on clean inputs.
            for dd in 0..d {
                k[dd] = bf16::from_f32(if dd == 0 { 4.0 } else { 0.0 });
            }
        }
        Inputs { q, k, v }
    }

    // f64 SDPA over clean bf16 inputs. Returns token-major rows for the
    // requested query tokens across ALL heads, each [d].
    #[allow(clippy::needless_range_loop)]
    fn oracle(inp: &Inputs, geo: Geo, row: &Row, q_tokens: &[usize]) -> Vec<Vec<f64>> {
        let Geo {
            q_heads,
            kv_heads,
            head_dim: d,
        } = geo;
        let ratio = q_heads / kv_heads;
        let q_start = row.kv_len - row.seq_len;
        let qdim = q_heads * d;
        let mut out = Vec::with_capacity(q_tokens.len() * q_heads);
        for &qi in q_tokens {
            let qpos = q_start + qi;
            for h in 0..q_heads {
                let hk = h / ratio;
                let mut scores = vec![0f64; qpos + 1];
                for key in 0..=qpos {
                    let mut dot = 0f64;
                    for dd in 0..d {
                        dot += inp.q[qi * qdim + h * d + dd].to_f32() as f64
                            * inp.k[(hk * row.max_seq_len + key) * d + dd].to_f32() as f64;
                    }
                    scores[key] = dot / (d as f64).sqrt();
                }
                let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let ex: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let z: f64 = ex.iter().sum();
                let mut o = vec![0f64; d];
                for dd in 0..d {
                    let mut acc = 0f64;
                    for key in 0..=qpos {
                        acc +=
                            ex[key] * inp.v[(hk * row.max_seq_len + key) * d + dd].to_f32() as f64;
                    }
                    // Single final bf16 RN, matching the kernel store.
                    o[dd] = bf((acc / z) as f32).to_f32() as f64;
                }
                out.push(o);
            }
        }
        out
    }

    fn run_one(ctx: &DeviceContext, fam: &Family, ri: usize, corrupt: bool) -> Result<bool> {
        let geo = fam.geo;
        let row = &fam.rows[ri];
        let seed: u64 = 0x5117_0c70u64
            .wrapping_mul(31)
            .wrapping_add(fam.name.len() as u64)
            .wrapping_mul(1_000_003)
            .wrapping_add(ri as u64 + 1);
        let clean = make_inputs(geo, row, seed, false);
        let device_inputs = if corrupt {
            make_inputs(geo, row, seed, true)
        } else {
            Inputs {
                q: clean.q.clone(),
                k: clean.k.clone(),
                v: clean.v.clone(),
            }
        };
        let qn = row.seq_len * geo.q_heads * geo.head_dim;

        let q_d = ctx.stream.clone_htod(&device_inputs.q)?;
        let k_d = ctx.stream.clone_htod(&device_inputs.k)?;
        let v_d = ctx.stream.clone_htod(&device_inputs.v)?;
        let mut o_d = ctx.stream.alloc_zeros::<bf16>(qn)?;
        {
            let (q_p, _g0) = q_d.device_ptr(&ctx.stream);
            let (k_p, _g1) = k_d.device_ptr(&ctx.stream);
            let (v_p, _g2) = v_d.device_ptr(&ctx.stream);
            let (o_p, _g3) = o_d.device_ptr_mut(&ctx.stream);
            fa2_sm70_attention_raw(
                &ctx.stream,
                q_p,
                k_p,
                v_p,
                o_p,
                geo.q_heads,
                geo.kv_heads,
                geo.head_dim,
                row.seq_len,
                row.kv_len,
                row.max_seq_len,
                1.0 / (geo.head_dim as f32).sqrt(),
            )?;
            ctx.sync()?;
        }
        let got = ctx.stream.clone_dtoh(&o_d)?;

        // The negative oracle is the CLEAN expectation: a corrupted kernel
        // must fail to reproduce it.
        let q_tokens: Vec<usize> = row
            .sample_q
            .clone()
            .unwrap_or_else(|| (0..row.seq_len).collect());
        let wants = oracle(&clean, geo, row, &q_tokens);
        let row_ids: Vec<(usize, usize)> = q_tokens
            .iter()
            .flat_map(|&t| (0..geo.q_heads).map(move |h| (t, h)))
            .collect();
        let m = compare_rows(
            &got,
            &wants,
            row_ids.len(),
            geo.head_dim,
            geo.q_heads * geo.head_dim,
            &row_ids,
            &TOL,
            false,
        );
        let pass = metrics_pass(&m, &TOL);
        println!(
            "{mark} {name:<9} row{ri:<2} seq={seq:<5} kv={kv:<5} hd={hd} gqa={qh}/{kh} \
             rel_l2={l2:.3e} viol={vf:.2e}{note}",
            mark = if pass { '✓' } else { '✗' },
            name = fam.name,
            ri = ri + 1,
            seq = row.seq_len,
            kv = row.kv_len,
            hd = geo.head_dim,
            qh = geo.q_heads,
            kh = geo.kv_heads,
            l2 = m.rel_l2,
            vf = m.viol_frac,
            note = if corrupt { " [corrupt]" } else { "" },
        );
        Ok(pass)
    }

    pub(super) fn run(corrupt: Option<String>) -> Result<()> {
        let ctx = DeviceContext::new()?;
        let (major, minor) = ctx.compute_capability();
        if major >= 8 {
            // Machine-readable skip for parity_gpu_batch.sh: one SKIP: line,
            // rc 0, and deliberately NO pass/negative marker (a real run
            // prints ALL PASS / NEGATIVE CONTROL OK). Both positive and
            // --negative-control invocations skip identically.
            let mode = if corrupt.is_some() {
                " (negative-control run)"
            } else {
                ""
            };
            println!(
                "SKIP: requires sm_70, device is sm_{major}{minor}{mode}; \
                 arle_fa2_sm70 is the sm<80 kernel — run on the V100 box"
            );
            return Ok(());
        }

        let fams = families();
        let mut failures = Vec::new();
        for fam in &fams {
            let corrupt_this = match &corrupt {
                None => false,
                Some(c) => c == "all" || c == fam.name,
            };
            for ri in 0..fam.rows.len() {
                let pass = run_one(&ctx, fam, ri, corrupt_this)?;
                if corrupt_this == pass {
                    // Clean run must pass; corrupted run must fail.
                    failures.push((fam.name, ri + 1, corrupt_this));
                }
            }
        }

        match &corrupt {
            Some(c) => {
                if !failures.is_empty() {
                    bail!("negative control {c} failed to trip/tripped wrong: {failures:?}");
                }
                println!("NEGATIVE CONTROL OK ({c})");
            }
            None => {
                if !failures.is_empty() {
                    bail!("fa2_sm70 parity failures: {failures:?}");
                }
                println!("FA2 sm70 ALL PASS — {}", cuda_kernels::KERNEL_BUILD_ID);
            }
        }
        Ok(())
    }
}
