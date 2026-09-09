//! THE equivalence gate on `qwen4_exp` speculative decode: greedy speculative
//! output must EQUAL plain greedy decode token for token. Under greedy
//! acceptance speculation is lossless BY CONSTRUCTION — the verify chunk runs
//! decode's own kernels and only target argmaxes are ever emitted — so any
//! mismatch here is a rollback bug, not a quality tradeoff.
//!
//! ## What runs by default
//!
//! `speculative_equals_decode_on_a_truncated_model` loads the checkpoint's
//! first four layers (`SubsetF32`, the same residency as the prefill gate),
//! decodes a 24-token prompt plus 12 greedy tokens as ground truth, then
//! replays generation through `generate_speculative` with three SYNTHETIC
//! draft sources — always-right (every cycle fully accepts), always-wrong
//! (every cycle rolls back), and mixed — across draft depths. Tokens must
//! match exactly in every configuration. The draft source is synthetic on
//! purpose: the loop's correctness must not depend on WHAT is proposed, and
//! an adversarial source exercises the rollback lanes far harder than a good
//! drafter would.
//!
//! `rollback_restores_the_recurrent_state_bit_for_bit` then pins the
//! mechanism directly: after a fully-rejected cycle the gated-delta S, conv
//! rings, PLE ring and n-gram window must be BYTE-identical to their
//! pre-cycle values — and, via `ARLE_QWEN4_SPEC_FAULT`, each deliberately
//! skipped restore (`skip-gdn`, `skip-ple`, `skip-ngram`) must make exactly
//! that comparison fail. A gate that cannot fail proves nothing; this is the
//! proof it can.
//!
//! ## The env-gated full-scale run
//!
//! `full_scale_speculative_gate_and_sweep` (`ARLE_QWEN4_SPEC=1`, use
//! `--release`, `--test-threads=1`) loads the WHOLE model (hybrid residency,
//! Q4_K dense default, MTP head aboard) and, over three prompt classes
//! (factual QA / code / chat continuation):
//!
//! - measures the plain greedy decode baseline (ms/token, same sitting),
//! - for each draft depth `k` in `ARLE_QWEN4_SPEC_KS` (default `1,2,3,4`)
//!   runs MTP speculative generation, ASSERTS token equality with the plain
//!   run, and reports acceptance per position, mean accepted length,
//!   draft/verify/rollback wall split, and effective tok/s.
//!
//! `ARLE_QWEN4_SPEC_PARITY=1` additionally diffs the MTP head's host oracle
//! against its device route on the same inputs (drafts are proposals — this
//! parity bounds numeric drift, it is not a correctness gate).
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use infer_gguf::safetensors::SafeTensorsDir;
use infer_vulkan::model_qwen4_exp::{
    Qwen4DraftSource, Qwen4ExpDeviceMode, VulkanQwen4ExpModel, greedy_argmax,
};
use infer_vulkan::qwen4_config::{Qwen4ExpConfig, Qwen4LayerType};
use vulkan_sys::VulkanContext;

const CKPT_ENV: &str = "ARLE_QWEN4_CKPT";
const CKPT_DEFAULT: &str = r"C:\Users\Asus\models\qwen3.8-flash-next-nvfp4";

fn checkpoint_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var(CKPT_ENV).unwrap_or_else(|_| CKPT_DEFAULT.into()));
    if dir.is_dir() {
        Some(dir)
    } else {
        eprintln!(
            "SKIP: checkpoint {} not present (set {CKPT_ENV})",
            dir.display()
        );
        None
    }
}

fn device() -> Option<VulkanContext> {
    match VulkanContext::create() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIP: no Vulkan device ({e})");
            None
        }
    }
}

/// A fixed prompt of in-vocab, non-EOS ids (the gate is path-vs-path token
/// equality; no particular text makes it stronger).
fn prompt_ids(cfg: &Qwen4ExpConfig, n: usize) -> Vec<u32> {
    (0..n as u32)
        .map(|i| {
            let id = (1009 + i * 37) % (cfg.vocab_size.min(30_000) as u32);
            if cfg.stop_token_ids.contains(&id) {
                id + 1
            } else {
                id
            }
        })
        .collect()
}

/// Plain greedy decode: the ground truth both gates compare against.
fn plain_greedy(model: &mut VulkanQwen4ExpModel<'_, '_>, prompt: &[u32], n_new: usize) -> Vec<u32> {
    let mut logits = Vec::new();
    for (i, &tok) in prompt.iter().enumerate() {
        logits = model
            .forward_token(0, 0, tok, i)
            .unwrap_or_else(|e| panic!("decode prompt token {i}: {e:#}"));
    }
    let mut out = vec![greedy_argmax(&logits)];
    while out.len() < n_new {
        let pos = prompt.len() + out.len() - 1;
        logits = model
            .forward_token(0, 0, *out.last().expect("non-empty"), pos)
            .unwrap_or_else(|e| panic!("decode gen token at {pos}: {e:#}"));
        out.push(greedy_argmax(&logits));
    }
    out
}

/// What a synthetic draft source proposes.
#[derive(Clone, Copy, Debug)]
enum DraftMode {
    /// The true greedy continuation — every cycle fully accepts.
    Right,
    /// Deliberately wrong tokens — every cycle rolls back with `L = 0`.
    Wrong,
    /// Alternates per cycle — both lanes exercised in one run.
    Mixed,
}

/// Draft source scripted from the plain run's continuation. Position-indexed
/// (`h_last_pos` names the position being continued), so it stays correct
/// across rollbacks and absorbs.
struct ScriptedDrafter {
    /// `prompt ++ plain_continuation` — token at absolute position `p` is
    /// `all[p]`.
    all: Vec<u32>,
    mode: DraftMode,
    calls: usize,
    vocab: usize,
}

impl<'ctx, 'st> Qwen4DraftSource<'ctx, 'st> for ScriptedDrafter {
    fn draft(
        &mut self,
        _model: &mut VulkanQwen4ExpModel<'ctx, 'st>,
        _h_last: &[f32],
        h_last_pos: usize,
        last_token: u32,
        k: usize,
    ) -> anyhow::Result<Vec<u32>> {
        assert_eq!(
            self.all.get(h_last_pos + 1).copied(),
            Some(last_token),
            "the driver's continuation point disagrees with the script"
        );
        let right = |j: usize| {
            self.all
                .get(h_last_pos + 2 + j)
                .copied()
                // Past the scripted horizon any token works: the verify can
                // only reject it.
                .unwrap_or(7)
        };
        let wrong = |j: usize| (right(j) + 1) % self.vocab as u32;
        let mode = match self.mode {
            DraftMode::Right => DraftMode::Right,
            DraftMode::Wrong => DraftMode::Wrong,
            DraftMode::Mixed if self.calls.is_multiple_of(2) => DraftMode::Right,
            DraftMode::Mixed => DraftMode::Wrong,
        };
        self.calls += 1;
        Ok((0..k)
            .map(|j| match mode {
                DraftMode::Wrong => wrong(j),
                _ => right(j),
            })
            .collect())
    }
}

/// Load the 4-layer truncated model in the `SubsetF32` parity residency (the
/// same setup as `tests/qwen4_prefill.rs`, whose prefill=decode gate is what
/// makes the verify chunk trustworthy here).
fn truncated_model<'ctx, 'st>(
    ctx: &'ctx VulkanContext,
    st: &'st SafeTensorsDir,
    cfg: &Qwen4ExpConfig,
) -> Option<VulkanQwen4ExpModel<'ctx, 'st>> {
    let mode = Qwen4ExpDeviceMode::SubsetF32(vec![0, 1, 2, 3]);
    let t0 = std::time::Instant::now();
    match VulkanQwen4ExpModel::load(Some(ctx), st, cfg.clone(), &mode) {
        Ok(m) => {
            eprintln!("subset model loaded in {:.1}s", t0.elapsed().as_secs_f64());
            Some(m)
        }
        Err(e) => {
            eprintln!("SKIP: subset load failed ({e:#}) — device memory likely contended");
            None
        }
    }
}

fn truncated_cfg(dir: &std::path::Path) -> Qwen4ExpConfig {
    let mut cfg = Qwen4ExpConfig::from_model_dir(dir).expect("parse config");
    assert!(cfg.num_hidden_layers >= 4);
    assert_eq!(cfg.layer_types[3], Qwen4LayerType::FullAttention);
    assert_eq!(cfg.ple_layer_ids, vec![1], "PLE sits on layer 1");
    cfg.num_hidden_layers = 4;
    cfg.layer_types.truncate(4);
    cfg
}

/// THE gate: speculative == plain greedy, for any draft source, any depth.
#[test]
fn speculative_equals_decode_on_a_truncated_model() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(ctx) = device() else { return };
    let st = SafeTensorsDir::open_dir(&dir).expect("open checkpoint");
    let cfg = truncated_cfg(&dir);
    let Some(mut model) = truncated_model(&ctx, &st, &cfg) else {
        return;
    };

    let prompt = prompt_ids(&cfg, 24);
    const N_NEW: usize = 12;
    let want = plain_greedy(&mut model, &prompt, N_NEW);
    eprintln!("plain greedy continuation: {want:?}");
    let all: Vec<u32> = prompt.iter().chain(want.iter()).copied().collect();

    for (k, mode) in [
        (2, DraftMode::Right),
        (2, DraftMode::Wrong),
        (2, DraftMode::Mixed),
        (1, DraftMode::Mixed),
        (3, DraftMode::Mixed),
    ] {
        let mut drafter = ScriptedDrafter {
            all: all.clone(),
            mode,
            calls: 0,
            vocab: cfg.vocab_size,
        };
        let (got, stats) = model
            .generate_speculative(&mut drafter, &prompt, N_NEW, k)
            .unwrap_or_else(|e| panic!("speculative k={k} {mode:?}: {e:#}"));
        eprintln!(
            "k={k} {mode:?}: cycles {}, accepted {}/{} (hist {:?}), absorbs {}",
            stats.cycles, stats.accepted_total, stats.drafted, stats.accept_hist, stats.absorbs,
        );
        assert_eq!(
            got, want,
            "speculative (k={k}, {mode:?}) diverged from plain greedy decode"
        );
        // The scripted modes must have exercised the lane they exist for.
        match mode {
            DraftMode::Right => assert_eq!(stats.full_accepts, stats.cycles),
            DraftMode::Wrong => assert_eq!(stats.full_accepts, 0),
            DraftMode::Mixed => {
                assert!(stats.full_accepts > 0, "mixed never fully accepted");
                assert!(stats.full_accepts < stats.cycles, "mixed never rolled back");
            }
        }
    }
}

/// Rollback restores every recurrent piece BIT for bit — and each
/// fault-injection knob breaks exactly its piece, so the gate above is proven
/// able to fail.
#[test]
fn rollback_restores_the_recurrent_state_bit_for_bit() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(ctx) = device() else { return };
    let st = SafeTensorsDir::open_dir(&dir).expect("open checkpoint");
    let cfg = truncated_cfg(&dir);
    let Some(mut model) = truncated_model(&ctx, &st, &cfg) else {
        return;
    };
    let prompt = prompt_ids(&cfg, 24);

    // One fully-rejected cycle per fault setting, from an identical prefix.
    let capture = |model: &VulkanQwen4ExpModel<'_, '_>| {
        let rl = model.resident_linear().expect("resident linear");
        let mut linear = BTreeMap::new();
        for (l, kind) in cfg.layer_types.iter().enumerate() {
            if *kind == Qwen4LayerType::LinearAttention {
                linear.insert(l, rl.read_state(&cfg, l).expect("read state"));
            }
        }
        let ple: BTreeMap<usize, Vec<f32>> = model
            .state()
            .ple_conv
            .iter()
            .map(|(&l, r)| (l, r.rows().to_vec()))
            .collect();
        (
            linear,
            ple,
            model.state().ngram.clone(),
            model.state().seq_len,
        )
    };

    for fault in ["", "skip-gdn", "skip-ple", "skip-ngram"] {
        // SAFETY: device tests run --test-threads=1; the var is read once per
        // rollback inside `speculative_verify_cycle`.
        unsafe {
            if fault.is_empty() {
                std::env::remove_var("ARLE_QWEN4_SPEC_FAULT");
            } else {
                std::env::set_var("ARLE_QWEN4_SPEC_FAULT", fault);
            }
        }
        let logits = model.forward_prompt(0, &prompt, 0).expect("prompt prefill");
        let before = capture(&model);
        let t0 = greedy_argmax(&logits);
        // Wrong drafts: L = 0, guaranteed rollback.
        let wrong = vec![
            (t0 + 1) % cfg.vocab_size as u32,
            (t0 + 2) % cfg.vocab_size as u32,
        ];
        let cycle = model
            .speculative_verify_cycle(&[t0], &wrong)
            .expect("cycle");
        assert!(cycle.rolled_back, "wrong drafts must roll back");
        assert_eq!(cycle.accepted, 0);
        let after = capture(&model);

        let gdn_equal = before.0 == after.0;
        let ple_equal = before.1 == after.1;
        let ngram_equal = before.2 == after.2;
        assert_eq!(before.3, after.3, "seq_len must rewind (fault `{fault}`)");
        match fault {
            "" => {
                assert!(gdn_equal, "GDN S + conv rings must restore bit-for-bit");
                assert!(ple_equal, "the PLE ring must restore bit-for-bit");
                assert!(ngram_equal, "the n-gram window must restore");
            }
            "skip-gdn" => assert!(
                !gdn_equal,
                "skip-gdn left the GDN state equal — the fault knob (and so the gate) is dead"
            ),
            "skip-ple" => assert!(
                !ple_equal,
                "skip-ple left the PLE ring equal — the fault knob (and so the gate) is dead"
            ),
            "skip-ngram" => assert!(
                !ngram_equal,
                "skip-ngram left the n-gram window equal — the fault knob (and so the gate) is dead"
            ),
            _ => unreachable!(),
        }
        eprintln!(
            "fault `{fault}`: gdn_equal={gdn_equal} ple_equal={ple_equal} ngram_equal={ngram_equal}"
        );
    }
    // SAFETY: single-threaded test binary; leave the environment clean.
    unsafe { std::env::remove_var("ARLE_QWEN4_SPEC_FAULT") };
}

// ─────────────────────────────────────────────────────────────────────────────
// Full scale (env-gated: the hybrid load stages ~66 GiB for minutes).
// ─────────────────────────────────────────────────────────────────────────────

/// `tokenizer.json`'s `model.vocab` as a token → id map.
fn load_vocab(dir: &std::path::Path) -> BTreeMap<String, u32> {
    let src = std::fs::read_to_string(dir.join("tokenizer.json")).expect("read tokenizer.json");
    let doc = infer_vulkan::qwen4_config::json::parse(&src).expect("parse tokenizer.json");
    let vocab = doc
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.as_object())
        .expect("tokenizer.json model.vocab");
    vocab
        .iter()
        .filter_map(|(tok, id)| {
            id.as_i64()
                .and_then(|v| u32::try_from(v).ok())
                .map(|v| (tok.clone(), v))
        })
        .collect()
}

/// Real prompts as single-BPE pieces, one per class (factual QA / code /
/// chat), so the id sequences are exactly what the tokenizer emits.
fn class_prompts(by_tok: &BTreeMap<String, u32>) -> Vec<(&'static str, Vec<u32>)> {
    let specs: [(&str, &[&str]); 3] = [
        (
            "factual-qa",
            &[
                "The",
                "Ġcapital",
                "Ġof",
                "ĠFrance",
                "Ġis",
                "Ġthe",
                "Ġcity",
                "Ġof",
            ],
        ),
        (
            "code",
            &[
                "def", "Ġfib", "(", "n", "):", "Ċ", "Ġif", "Ġn", "Ġ<", "Ġ", "2", ":", "Ġreturn",
                "Ġn", "Ċ", "Ġreturn",
            ],
        ),
        (
            "chat",
            &[
                "Hello", "!", "ĠHow", "Ġare", "Ġyou", "Ġtoday", "?", "ĠI", "Ġam",
            ],
        ),
    ];
    let mut prompts = Vec::new();
    for (label, spec) in specs {
        let ids: Option<Vec<u32>> = spec.iter().map(|p| by_tok.get(*p).copied()).collect();
        match ids {
            Some(ids) => prompts.push((label, ids)),
            None => eprintln!("  ({label} dropped, vocab missing a piece: {spec:?})"),
        }
    }
    prompts
}

#[test]
fn full_scale_speculative_gate_and_sweep() {
    if std::env::var_os("ARLE_QWEN4_SPEC").is_none() {
        eprintln!("SKIP: set ARLE_QWEN4_SPEC=1 (full ~68 GiB hybrid load; use --release)");
        return;
    }
    let Some(dir) = checkpoint_dir() else { return };
    let Some(ctx) = device() else { return };
    let st = SafeTensorsDir::open_dir(&dir).expect("open checkpoint");
    let cfg = Qwen4ExpConfig::from_model_dir(&dir).expect("parse config");
    let by_tok = load_vocab(&dir);
    let prompts = class_prompts(&by_tok);
    assert!(prompts.len() >= 3, "all three prompt classes must survive");

    let n_new: usize = std::env::var("ARLE_QWEN4_SPEC_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    let ks: Vec<usize> = std::env::var("ARLE_QWEN4_SPEC_KS")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 3, 4]);

    let t0 = std::time::Instant::now();
    let mut model = VulkanQwen4ExpModel::load(
        Some(&ctx),
        &st,
        cfg.clone(),
        &Qwen4ExpDeviceMode::HybridExperts,
    )
    .expect("hybrid model load");
    eprintln!("model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // ── the plain greedy baseline, same sitting ──
    struct Baseline {
        tokens: Vec<u32>,
        ms_per_token: f64,
    }
    let mut baselines: Vec<Baseline> = Vec::new();
    for (label, prompt) in &prompts {
        // Prefill through the batched path (same as speculative runs), then
        // per-token decode; steady-state timing skips the first 4 tokens.
        let logits = model.forward_prompt(0, prompt, 0).expect("prefill");
        let mut tokens = vec![greedy_argmax(&logits)];
        let mut steady = 0.0f64;
        let mut steady_n = 0usize;
        while tokens.len() < n_new {
            let pos = prompt.len() + tokens.len() - 1;
            let t = std::time::Instant::now();
            let logits = model
                .forward_token(0, 0, *tokens.last().expect("non-empty"), pos)
                .expect("decode");
            if tokens.len() >= 4 {
                steady += t.elapsed().as_secs_f64();
                steady_n += 1;
            }
            tokens.push(greedy_argmax(&logits));
        }
        let ms = 1e3 * steady / steady_n.max(1) as f64;
        eprintln!("[{label}] plain greedy: {ms:.1} ms/token  tokens {tokens:?}");
        baselines.push(Baseline {
            tokens,
            ms_per_token: ms,
        });
    }

    // ── the sweep: gate + acceptance + effective throughput per k ──
    println!();
    println!(
        "| prompt | k | accept/step | mean L | cycles(full) | absorbs | draft ms/cyc | verify ms/cyc | rollback ms/cyc | eff ms/tok | baseline ms/tok | speedup |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    for (pi, (label, prompt)) in prompts.iter().enumerate() {
        for &k in &ks {
            let mut drafter = model.mtp_drafter().expect("load MTP head");
            let (got, stats) = model
                .generate_speculative(&mut drafter, prompt, n_new, k)
                .unwrap_or_else(|e| panic!("speculative [{label}] k={k}: {e:#}"));
            assert_eq!(
                got, baselines[pi].tokens,
                "[{label}] k={k}: speculative output diverged from plain greedy — a rollback bug"
            );
            let cyc = stats.cycles.max(1) as f64;
            let eff_ms = 1e3 * stats.wall_s / stats.emitted.max(1) as f64;
            println!(
                "| {label} | {k} | {:.1}% | {:.2} | {}({}) | {} | {:.1} | {:.1} | {:.2} | {:.1} | {:.1} | {:.2}x |",
                100.0 * stats.acceptance(),
                stats.mean_accepted(),
                stats.cycles,
                stats.full_accepts,
                stats.absorbs,
                1e3 * stats.draft_s / cyc,
                1e3 * stats.verify_s / cyc,
                1e3 * stats.rollback_s / cyc,
                eff_ms,
                baselines[pi].ms_per_token,
                baselines[pi].ms_per_token / eff_ms,
            );
        }
    }

    // ── MTP parity (optional, AFTER the gate — a numeric report must never
    // pre-empt the correctness verdict): host oracle vs device route on
    // REAL conditioning (the last prompt's pre-mixer h and a real embedding
    // row). Out-of-distribution random inputs were measured to flip the
    // quantized router's expert selection wholesale, which reads as O(1e2)
    // max-rel without any wiring being wrong.
    if std::env::var_os("ARLE_QWEN4_SPEC_PARITY").is_some() {
        let (_, prompt) = &prompts[0];
        let _ = model.forward_prompt(0, prompt, 0).expect("parity prefill");
        let h = model.prefill_h_row(prompt.len() - 1).expect("parity h row");
        let tok = baselines[0].tokens[0];
        mtp_parity(&st, &cfg, &mut model, &h, tok);
    }
}

/// Host-oracle vs device-route MTP forward on identical inputs. Bounds drift;
/// drafts are proposals, so this is a numeric report, not a correctness gate.
fn mtp_parity(
    st: &SafeTensorsDir,
    cfg: &Qwen4ExpConfig,
    model: &mut VulkanQwen4ExpModel<'_, '_>,
    h: &[f32],
    token: u32,
) {
    use infer_vulkan::model_qwen4_exp::{HostKv, hc_config};
    use infer_vulkan::qwen4_mtp::MtpHead;

    let head = MtpHead::load(st, cfg).expect("load MTP head");
    let hc = hc_config(cfg);
    // The real embedding row of `token`, straight off the checkpoint.
    let bytes = st
        .tensor_data("model.language_model.embed_tokens.weight")
        .expect("embed table");
    let row = &bytes[token as usize * cfg.hidden_size * 2..][..cfg.hidden_size * 2];
    let e: Vec<f32> = row
        .chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect();

    let mut kv_host = HostKv::default();
    let host = head
        .forward(cfg, &hc, h, &e, 1, &mut kv_host, false, None, None)
        .expect("host oracle forward");
    let mut kv_dev = HostKv::default();
    let route = model.dev_and_weights().expect("hybrid model has a device");
    let dev = head
        .forward(cfg, &hc, h, &e, 1, &mut kv_dev, false, Some(route), None)
        .expect("device route forward");

    let rel = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs() / y.abs().max(1e-3))
            .fold(0.0f32, f32::max)
    };
    let mean_abs = |a: &[f32], b: &[f32]| {
        a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).sum::<f32>() / a.len() as f32
    };
    let h_rel = rel(&dev.h_out, &host.h_out);
    let kv_rel = rel(&kv_dev.k, &kv_host.k).max(rel(&kv_dev.v, &kv_host.v));
    eprintln!(
        "MTP parity (real inputs): h_out max rel {h_rel:.3e} / mean abs {:.3e}, kv max rel {kv_rel:.3e}",
        mean_abs(&dev.h_out, &host.h_out),
    );
    // A REPORT, not a gate: the device twins are Q4_K/Q8_0, so a razor-thin
    // top-10 router margin can legally flip an expert and move a few h_out
    // elements O(1) — drafts are proposals, and the number that actually
    // judges the device MTP is the sweep's acceptance rate above. What must
    // still hold: the routes agree they computed SOMETHING finite.
    assert!(
        h_rel.is_finite() && kv_rel.is_finite(),
        "MTP parity non-finite"
    );
}

/// Wiring parity for the MTP device route with quantization OUT of the
/// picture (`ARLE_QWEN4_MTP_F32=1`; loads a 1-layer subset plus the MTP head
/// on the F32 tier, ~14 GiB): per-slice GEMVs — the stacked-expert
/// addressing no text-stream test exercises — must match the host slice
/// views tightly, and a full forward's attention K/V (expert-selection-free)
/// must too. At F32 any real drift is a wiring bug, not noise.
#[test]
fn mtp_device_route_matches_the_host_oracle_at_f32() {
    if std::env::var_os("ARLE_QWEN4_MTP_F32").is_none() {
        eprintln!("SKIP: set ARLE_QWEN4_MTP_F32=1 (1-layer subset + F32 MTP, ~14 GiB load)");
        return;
    }
    let Some(dir) = checkpoint_dir() else { return };
    let Some(ctx) = device() else { return };
    // SAFETY: device suites run --test-threads=1; read once at model load.
    unsafe { std::env::set_var("ARLE_QWEN4_SUBSET_MTP", "1") };
    let st = SafeTensorsDir::open_dir(&dir).expect("open checkpoint");
    let mut cfg = Qwen4ExpConfig::from_model_dir(&dir).expect("parse config");
    cfg.num_hidden_layers = 1;
    cfg.layer_types.truncate(1);
    let t0 = std::time::Instant::now();
    let mut model = VulkanQwen4ExpModel::load(
        Some(&ctx),
        &st,
        cfg.clone(),
        &Qwen4ExpDeviceMode::SubsetF32(vec![0]),
    )
    .expect("subset+MTP load");
    eprintln!("subset+MTP loaded in {:.1}s", t0.elapsed().as_secs_f64());
    // SAFETY: as above; leave the environment clean for later loads.
    unsafe { std::env::remove_var("ARLE_QWEN4_SUBSET_MTP") };

    use infer_vulkan::model_qwen4_exp::{DenseGemv, HostKv, hc_config};
    use infer_vulkan::qwen4_mtp::MtpHead;
    use infer_vulkan::qwen4_names::ExpertProj;
    let head = MtpHead::load(&st, &cfg).expect("load MTP head");
    let hc = hc_config(&cfg);

    let mut s = 0x243F6A8885A308D3u64;
    let mut unit = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 32) as f64 / f64::from(u32::MAX) - 0.5) as f32
    };
    let x_h: Vec<f32> = (0..cfg.hidden_size).map(|_| unit()).collect();
    let x_i: Vec<f32> = (0..cfg.moe_intermediate_size).map(|_| unit()).collect();
    let rel = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs() / y.abs().max(1e-3))
            .fold(0.0f32, f32::max)
    };

    // Per-slice GEMV parity: host view vs the uploaded per-expert twin.
    for e in [0usize, 7, 511] {
        for proj in [ExpertProj::Gate, ExpertProj::Up, ExpertProj::Down] {
            let hd = head.expert_dense(e, proj).expect("slice view");
            let xin = if proj == ExpertProj::Down { &x_i } else { &x_h };
            let host = hd.matvec(xin);
            let (dev, w) = model.dev_and_weights().expect("device");
            let got = DenseGemv::new(dev, w)
                .matvec(&hd, xin)
                .expect("device gemv");
            let r = rel(&got, &host);
            eprintln!("  expert {e} {proj:?}: max rel {r:.3e}");
            assert!(
                r < 1e-4,
                "expert {e} {proj:?} slice drifts {r:.3e} at F32 — slice addressing is wrong"
            );
        }
    }

    // Full-forward: the attention K/V path is expert-selection-free, so at
    // F32 it must be tight; h_out is reported (a razor-thin router tie can
    // legally flip an expert even at F32 on a synthetic input).
    let h: Vec<f32> = (0..hc.hc_hidden()).map(|_| unit()).collect();
    let e: Vec<f32> = (0..cfg.hidden_size).map(|_| unit()).collect();
    let mut kv_host = HostKv::default();
    let host = head
        .forward(&cfg, &hc, &h, &e, 1, &mut kv_host, false, None, None)
        .expect("host forward");
    let mut kv_dev = HostKv::default();
    let route = model.dev_and_weights().expect("device");
    let dev = head
        .forward(&cfg, &hc, &h, &e, 1, &mut kv_dev, false, Some(route), None)
        .expect("device forward");
    let kv_rel = rel(&kv_dev.k, &kv_host.k).max(rel(&kv_dev.v, &kv_host.v));
    let h_rel = rel(&dev.h_out, &host.h_out);
    eprintln!("MTP F32 parity: kv max rel {kv_rel:.3e}, h_out max rel {h_rel:.3e}");
    assert!(
        kv_rel < 1e-3,
        "MTP K/V drifts {kv_rel:.3e} at F32 — the fuse or attention-site wiring is wrong"
    );
}
