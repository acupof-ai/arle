//! Prefill-equals-decode proof for `qwen4_exp` — THE gate on
//! `forward_prompt`: a chunked prefill that diverges from the per-token loop
//! is a wrong prefill, full stop.
//!
//! ## What runs by default
//!
//! `prefill_equals_decode_on_a_truncated_model` loads the real checkpoint's
//! FIRST FOUR layers (linear / linear+PLE / linear / full attention — every
//! stage class the model has) in the `SubsetF32` residency the parity harness
//! uses, drives a 24-token prompt through `forward_token` one token at a time,
//! resets, and replays the same prompt through `forward_prompt` at chunk
//! widths 7 (uneven multi-chunk: 7+7+7+3) and 24 (one chunk). It compares
//! - the final logits (max rel < 1e-4),
//! - the gated-delta S and conv-ring state of every linear layer,
//! - the PLE conv ring,
//! - the full-attention layer's written K/V rows,
//! - and `seq_len`.
//!
//! At this residency the dense tier is F32, so the prefill records the SAME
//! GEMV dispatches decode does and the comparison isolates the chunked
//! machinery itself (seq-mode conv/GDN/PLE, batched flash + mask, chunk
//! permutation maps, the batched ids fence) — any indexing error is O(1)
//! divergence, not noise. A truncated-layer model is not the shipping model,
//! but equality of the two PATHS does not care which layers run.
//!
//! `gemm_route_drift_stays_in_the_f16_envelope` then repeats the comparison
//! with the dense tier staged BF16, which turns the coopmat GEMM route on —
//! the piece the bit-exact gate structurally cannot see (F32 has no GEMM
//! kernel). Its bound is a drift envelope, not equality; see the test.
//!
//! ## The env-gated full-scale measurement
//!
//! `full_scale_prefill_tok_s` (`ARLE_QWEN4_PREFILL=1`, use `--release`) loads
//! the WHOLE model in the hybrid residency and measures prefill tokens/s at
//! chunk {64, 256} over `ARLE_QWEN4_PREFILL_TOKENS` (default 512) tokens, with
//! the per-stage wall table. `ARLE_QWEN4_PREFILL_PARITY=1` additionally runs
//! a 24-token decode-vs-prefill logits comparison at full scale, AFTER the
//! measurement so a parity regression cannot eat the numbers. On the default
//! (GEMV) route the comparison is bit-exact — measured 0.000e0. Set
//! `ARLE_QWEN4_PREFILL_GEMM=1` to measure the opt-in coopmat lane instead;
//! its f16-staged activations drift ~2.6 absolute over 48 layers (expert
//! flips in the 512-expert routers, near-identical for bf16 and f16 staging)
//! and the parity's argmax assert then fails loudly — which is exactly why
//! that lane is not the default.
//!
//! `ARLE_GPU_TIMESTAMPS=1` adds a GPU-busy table per chunk width, keyed by
//! the stage that RECORDED each dispatch. The host table books nearly all
//! wall against `pf.moe.ids_fence` — the per-layer drain of everything
//! recorded since the previous fence — so this second table is what actually
//! decomposes the drain (measured 2026-08-28 at chunk 256: linattn 2.8 s,
//! grouped MoE experts 1.9 s, hc.pre 1.4 s, fullattn 0.5 s of a 7.2 s
//! drain). The timestamps themselves cost ~5% tok/s; keep the headline
//! number from a run without them.
//!
//! Skips cleanly when the checkpoint (`ARLE_QWEN4_CKPT`) or a Vulkan device is
//! absent, and says so loudly when the subset load fails on device memory (the
//! GPU may be contended).
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use infer_gguf::safetensors::SafeTensorsDir;
use infer_vulkan::model_qwen4_exp::{Qwen4ExpDeviceMode, VulkanQwen4ExpModel, prof};
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

/// Max per-element relative error with a 1e-3 magnitude floor (logits and
/// recurrent state are O(0.01..10); the floor keeps dead elements from
/// dividing by dust).
fn max_rel(got: &[f32], want: &[f32], what: &str) -> f32 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0f32;
    let mut at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}[{i}]: prefill produced {g}");
        assert!(w.is_finite(), "{what}[{i}]: decode produced {w}");
        let rel = (g - w).abs() / w.abs().max(1e-3);
        if rel > worst {
            worst = rel;
            at = i;
        }
    }
    eprintln!(
        "  {what:<28} max rel {worst:.3e}  (at {at}: prefill {:+.6e} vs decode {:+.6e})",
        got[at], want[at]
    );
    worst
}

/// A fixed 24-token prompt of in-vocab, non-EOS ids — synthetic on purpose:
/// the gate is path-vs-path equality, which no particular text makes stronger.
fn prompt_ids(cfg: &Qwen4ExpConfig) -> Vec<u32> {
    (0..24)
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

/// Everything the decode pass materializes that the prefill pass must
/// reproduce.
struct CapturedState {
    logits: Vec<f32>,
    /// Per linear layer: (gdr S, conv ring), HF order.
    linear: BTreeMap<usize, (Vec<f32>, Vec<f32>)>,
    /// Per PLE layer: the conv ring rows.
    ple: BTreeMap<usize, Vec<f32>>,
    /// Per full layer: flattened K then V rows for every (kv_head, pos).
    kv: BTreeMap<usize, Vec<f32>>,
}

fn capture(
    model: &VulkanQwen4ExpModel<'_, '_>,
    cfg: &Qwen4ExpConfig,
    logits: Vec<f32>,
    n_pos: usize,
) -> CapturedState {
    let rl = model.resident_linear().expect("resident linear attention");
    let dev = model.dev_ref().expect("device runner");
    let mut linear = BTreeMap::new();
    let mut kv = BTreeMap::new();
    for (l, kind) in cfg.layer_types.iter().enumerate() {
        match kind {
            Qwen4LayerType::LinearAttention => {
                linear.insert(l, rl.read_state(cfg, l).expect("read linear state"));
            }
            Qwen4LayerType::FullAttention => {
                let mut rows = Vec::new();
                for is_v in [false, true] {
                    for kvh in 0..cfg.num_key_value_heads {
                        for pos in 0..n_pos {
                            rows.extend(dev.read_kv_row(l, kvh, pos, is_v).expect("read KV row"));
                        }
                    }
                }
                kv.insert(l, rows);
            }
        }
    }
    let ple = model
        .state()
        .ple_conv
        .iter()
        .map(|(&l, ring)| (l, ring.rows().to_vec()))
        .collect();
    CapturedState {
        logits,
        linear,
        ple,
        kv,
    }
}

fn assert_close(prefill: &CapturedState, decode: &CapturedState, bound: f32, label: &str) {
    let worst = report_close(prefill, decode, label);
    assert!(
        worst < bound,
        "{label}: prefill diverges from decode (max rel {worst:.3e} >= {bound:.0e})"
    );
}

/// Print the per-stage prefill-vs-decode table and return the worst rel —
/// the localization tool `assert_close` and the GEMM-drift envelope share.
fn report_close(prefill: &CapturedState, decode: &CapturedState, label: &str) -> f32 {
    eprintln!("── prefill vs decode ({label}) ──");
    let mut worst = max_rel(&prefill.logits, &decode.logits, "logits");
    for (l, (gdr_p, ring_p)) in &prefill.linear {
        let (gdr_d, ring_d) = &decode.linear[l];
        worst = worst.max(max_rel(gdr_p, gdr_d, &format!("layer {l} gdr S")));
        worst = worst.max(max_rel(ring_p, ring_d, &format!("layer {l} conv ring")));
    }
    for (l, ring_p) in &prefill.ple {
        worst = worst.max(max_rel(
            ring_p,
            &decode.ple[l],
            &format!("layer {l} PLE ring"),
        ));
    }
    for (l, kv_p) in &prefill.kv {
        worst = worst.max(max_rel(kv_p, &decode.kv[l], &format!("layer {l} KV rows")));
    }
    worst
}

/// THE gate: prefill-then-read-state must equal the per-token loop.
#[test]
fn prefill_equals_decode_on_a_truncated_model() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(ctx) = device() else { return };
    let st = SafeTensorsDir::open_dir(&dir).expect("open checkpoint");
    let mut cfg = Qwen4ExpConfig::from_model_dir(&dir).expect("parse config");
    // First four layers cover every stage class; equality of the two PATHS
    // does not need the other 44.
    assert!(
        cfg.num_hidden_layers >= 4,
        "checkpoint has fewer than 4 layers"
    );
    assert_eq!(cfg.layer_types[3], Qwen4LayerType::FullAttention);
    assert_eq!(cfg.ple_layer_ids, vec![1], "PLE sits on layer 1");
    cfg.num_hidden_layers = 4;
    cfg.layer_types.truncate(4);

    let mode = Qwen4ExpDeviceMode::SubsetF32(vec![0, 1, 2, 3]);
    let t0 = std::time::Instant::now();
    let mut model = match VulkanQwen4ExpModel::load(Some(&ctx), &st, cfg.clone(), &mode) {
        Ok(m) => m,
        Err(e) => {
            // The GPU may be contended (this subset stages ~6 GiB of experts);
            // a memory failure is a loud skip, not a red bar.
            eprintln!("SKIP: subset load failed ({e:#}) — device memory likely contended");
            return;
        }
    };
    eprintln!("subset model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let toks = prompt_ids(&cfg);
    let n = toks.len();

    // ── decode pass: the ground truth. ──
    let mut logits = Vec::new();
    for (i, &tok) in toks.iter().enumerate() {
        logits = model
            .forward_token(0, 0, tok, i)
            .unwrap_or_else(|e| panic!("decode token {i}: {e:#}"));
    }
    assert_eq!(model.state().seq_len, n);
    let decode = capture(&model, &cfg, logits, n);

    // ── prefill passes: uneven multi-chunk, then one chunk. ──
    for chunk in [7usize, 24] {
        let logits = model
            .forward_prompt_chunked(0, &toks, 0, chunk)
            .unwrap_or_else(|e| panic!("forward_prompt chunk={chunk}: {e:#}"));
        assert_eq!(model.state().seq_len, n, "prefill must advance seq_len");
        let prefill = capture(&model, &cfg, logits, n);
        assert_close(&prefill, &decode, 1e-4, &format!("chunk={chunk}"));
    }
}

/// The opt-in coopmat GEMM lane (`ARLE_QWEN4_PREFILL_GEMM=1`) vs the same
/// decode loop, on the truncated model with the dense tier staged BF16
/// (`ARLE_QWEN4_SUBSET_DENSE=bf16`) — the 20-second repro of the full model's
/// dense residency. NOT the bit-exact gate above: the GEMM stages activations
/// to f16 (2^-11 rounding decode's f32-activation GEMVs do not have), so
/// exact equality is not on offer — which is why the lane is opt-in.
/// This pins the drift ENVELOPE instead. Calibration, measured on this box at
/// 4 layers: a structural break (wrong mask, missed barrier, B decoded as
/// bf16 bits after a re-vendor clobbers the `TO_FLOAT_TYPE_B` seam) reads
/// O(1e2..1e3); the rejected bf16-staged B (2^-8) read 8.1e0; f16 staging
/// reads 2.0e0. The bound sits between the last two, so a staging-precision
/// regression fails and honest f16 rounding passes.
#[test]
fn gemm_route_drift_stays_in_the_f16_envelope() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(ctx) = device() else { return };
    let st = SafeTensorsDir::open_dir(&dir).expect("open checkpoint");
    let mut cfg = Qwen4ExpConfig::from_model_dir(&dir).expect("parse config");
    assert!(cfg.num_hidden_layers >= 4, "fewer than 4 layers");
    cfg.num_hidden_layers = 4;
    cfg.layer_types.truncate(4);

    // SAFETY: device tests run --test-threads=1. `SUBSET_DENSE` is read once
    // inside `load`; `PREFILL_GEMM` is read at record time, so it stays set
    // until after the GEMM-route prefill below.
    unsafe { std::env::set_var("ARLE_QWEN4_SUBSET_DENSE", "bf16") };
    // SAFETY: as above.
    unsafe { std::env::set_var("ARLE_QWEN4_PREFILL_GEMM", "1") };
    let loaded = VulkanQwen4ExpModel::load(
        Some(&ctx),
        &st,
        cfg.clone(),
        &Qwen4ExpDeviceMode::SubsetF32(vec![0, 1, 2, 3]),
    );
    // SAFETY: same single-threaded contract as the `set_var` above.
    unsafe { std::env::remove_var("ARLE_QWEN4_SUBSET_DENSE") };
    let mut model = match loaded {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP: subset load failed ({e:#}) — device memory likely contended");
            return;
        }
    };

    let toks = prompt_ids(&cfg);
    let n = toks.len();
    let mut logits = Vec::new();
    for (i, &tok) in toks.iter().enumerate() {
        logits = model
            .forward_token(0, 0, tok, i)
            .unwrap_or_else(|e| panic!("decode token {i}: {e:#}"));
    }
    let decode = capture(&model, &cfg, logits, n);

    let logits = model
        .forward_prompt_chunked(0, &toks, 0, 24)
        .expect("GEMM-route prefill");
    // SAFETY: same single-threaded contract as the `set_var` above.
    unsafe { std::env::remove_var("ARLE_QWEN4_PREFILL_GEMM") };
    let prefill = capture(&model, &cfg, logits, n);

    let worst = report_close(&prefill, &decode, "bf16 dense, GEMM route");
    let am = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i)
    };
    let (am_p, am_d) = (am(&prefill.logits), am(&decode.logits));
    eprintln!("GEMM-route drift: max rel {worst:.3e}, argmax prefill {am_p} vs decode {am_d}");
    assert_eq!(am_p, am_d, "GEMM-route prefill flips the argmax");
    assert!(
        worst < 4.0,
        "GEMM-route drift {worst:.3e} above the f16-staging envelope"
    );
}

fn aggregate_ms(rows: &[prof::Row]) -> Vec<(String, f64, u64)> {
    let mut by_stage: BTreeMap<String, (f64, u64)> = BTreeMap::new();
    for r in rows {
        let e = by_stage.entry(r.stage.to_string()).or_default();
        e.0 += r.nanos as f64 / 1e6;
        e.1 += r.calls;
    }
    let mut v: Vec<(String, f64, u64)> = by_stage
        .into_iter()
        .map(|(s, (ms, calls))| (s, ms, calls))
        .collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v
}

/// Full-scale measured prefill: tokens/s at chunk {64, 256} plus the
/// per-stage wall split. Env-gated — the load is ~70 GiB and minutes long.
#[test]
fn full_scale_prefill_tok_s() {
    if std::env::var("ARLE_QWEN4_PREFILL").as_deref() != Ok("1") {
        eprintln!(
            "SKIP: set ARLE_QWEN4_PREFILL=1 (full ~70 GiB load, --release; \
             ARLE_QWEN4_PREFILL_TOKENS overrides the 512-token prompt, \
             ARLE_QWEN4_PREFILL_PARITY=1 adds a 24-token decode-vs-prefill diff)"
        );
        return;
    }
    let Some(dir) = checkpoint_dir() else { return };
    let Some(ctx) = device() else { return };
    let st = SafeTensorsDir::open_dir(&dir).expect("open checkpoint");
    let cfg = Qwen4ExpConfig::from_model_dir(&dir).expect("parse config");
    let t0 = std::time::Instant::now();
    let mut model = VulkanQwen4ExpModel::load(
        Some(&ctx),
        &st,
        cfg.clone(),
        &Qwen4ExpDeviceMode::HybridExperts,
    )
    .expect("hybrid load");
    eprintln!("hybrid model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let n_tokens: usize = std::env::var("ARLE_QWEN4_PREFILL_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n >= 2 && n < cfg.max_context)
        .unwrap_or(512);
    let toks: Vec<u32> = (0..n_tokens as u32)
        .map(|i| {
            let id = (1009 + i * 37) % (cfg.vocab_size.min(30_000) as u32);
            if cfg.stop_token_ids.contains(&id) {
                id + 1
            } else {
                id
            }
        })
        .collect();

    for chunk in [64usize, 256] {
        let _ = prof::take();
        if let Some(d) = model.dev_mut() {
            let _ = d.take_gpu_profile(); // drop load/previous-chunk samples
        }
        prof::set_enabled(true);
        let t0 = std::time::Instant::now();
        let logits = model
            .forward_prompt_chunked(0, &toks, 0, chunk)
            .expect("full-scale prefill");
        let secs = t0.elapsed().as_secs_f64();
        prof::set_enabled(false);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
        eprintln!(
            "\n== chunk {chunk}: {n_tokens} tok in {secs:.2}s = {:.1} tok/s ==",
            n_tokens as f64 / secs
        );
        for (stage, ms, calls) in aggregate_ms(&prof::take()) {
            eprintln!("  {stage:<22} {ms:>10.1} ms  ({calls} calls)");
        }
        // The host table above books the ids-fence DRAIN wall against the
        // stage that happened to flush; this attributes GPU-busy time to the
        // stage that RECORDED each dispatch — inside-the-drain decomposition.
        if let Some(d) = model.dev_mut() {
            let mut gpu = d.take_gpu_profile();
            if !gpu.is_empty() {
                gpu.sort_by(|a, b| b.2.total_cmp(&a.2));
                eprintln!("  ── GPU-busy by recording stage (ARLE_GPU_TIMESTAMPS=1) ──");
                for (label, dispatches, ms) in gpu {
                    eprintln!("  {label:<22} {ms:>10.1} ms  ({dispatches} dispatches)");
                }
            }
        }
    }

    // Matched A/B of the NUM_COLS dense batching in the SAME load
    // (`qwen4_gemv_cols_cap` is read per recorded dispatch, so a runtime env
    // flip is a real arm switch): chunk 256 again with the batching off.
    if std::env::var("ARLE_QWEN4_PREFILL_COLS_AB").as_deref() == Ok("1") {
        // SAFETY: device suites run --test-threads=1; removed below.
        unsafe { std::env::set_var("ARLE_QWEN4_GEMV_COLS", "1") };
        let t0 = std::time::Instant::now();
        let logits = model
            .forward_prompt_chunked(0, &toks, 0, 256)
            .expect("cols-off prefill");
        let secs = t0.elapsed().as_secs_f64();
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
        eprintln!(
            "\n== chunk 256, ARLE_QWEN4_GEMV_COLS=1 (cols batching OFF): {n_tokens} tok in \
             {secs:.2}s = {:.1} tok/s ==",
            n_tokens as f64 / secs
        );
        // SAFETY: as above.
        unsafe { std::env::remove_var("ARLE_QWEN4_GEMV_COLS") };
    }

    // Parity LAST, so a numeric regression cannot eat the measurement above.
    // The GEMM route stages activations to f16 (2^-11) where decode's GEMVs
    // read f32, so exact equality is not on offer at full scale — the drift
    // is reported against the decode top-2 logit gap, and the argmax (what a
    // greedy decode of the prompt's next token would emit) is the assert.
    if std::env::var("ARLE_QWEN4_PREFILL_PARITY").as_deref() == Ok("1") {
        let short = &toks[..24];
        let mut dec = Vec::new();
        for (i, &tok) in short.iter().enumerate() {
            dec = model.forward_token(0, 0, tok, i).expect("decode token");
        }
        let pre = model
            .forward_prompt_chunked(0, short, 0, 24)
            .expect("prefill 24");
        let mut worst = 0f32;
        let mut worst_abs = 0f32;
        let mut argmax = (0usize, 0usize);
        for (i, (&p, &d)) in pre.iter().zip(&dec).enumerate() {
            worst = worst.max((p - d).abs() / d.abs().max(1e-3));
            worst_abs = worst_abs.max((p - d).abs());
            if p > pre[argmax.0] {
                argmax.0 = i;
            }
            if d > dec[argmax.1] {
                argmax.1 = i;
            }
        }
        let mut top2 = f32::NEG_INFINITY;
        for (i, &d) in dec.iter().enumerate() {
            if i != argmax.1 && d > top2 {
                top2 = d;
            }
        }
        eprintln!(
            "full-scale 24-token parity: max rel {worst:.3e}, max abs {worst_abs:.3e} \
             (decode top-2 gap {:.3e}), argmax prefill {} vs decode {}",
            dec[argmax.1] - top2,
            argmax.0,
            argmax.1
        );
        assert_eq!(
            argmax.0, argmax.1,
            "prefill flips the argmax token at full scale"
        );
    }
}
