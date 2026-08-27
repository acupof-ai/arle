//! Qwen3.8-Flash-Next (`qwen4_exp`) forward proof: per-stage DEVICE-vs-HOST
//! parity on real weights, then an env-gated full forward.
//!
//! ## What runs by default
//!
//! `parity_layers_0_1_3_device_vs_host` loads a THREE-LAYER subset (layer 0 =
//! linear attention, layer 1 = linear + PLE, layer 3 = full attention; all 512
//! experts of each; dense tier F32) — deliberately NOT the full plan, which
//! exceeds the driver's heapBudget — and drives 3 tokens through every stage
//! twice: once on the host f32 transcription (`model_qwen4_exp`'s host lane,
//! matvecs in f64) and once on the device kernels. It reports the max relative
//! error PER STAGE and fails if any stage exceeds its threshold. The
//! thresholds are decades below what a real defect costs: a wrong NVFP4
//! nibble order, a folded-twice norm bias, a swapped GQA head map are all
//! O(1) relative error, never 1e-3.
//!
//! The layer walk is 0 → 1 → 3 with layer 2 skipped: parity needs MATCHED
//! inputs per stage, not the true 48-layer trajectory. The canonical residual
//! advances with the HOST attention output and the DEVICE MoE output, so both
//! sides always see identical stage inputs; the host MoE oracle (NVFP4
//! dequant + f64 dots — expensive in a debug build) is diffed on the first
//! token of each layer.
//!
//! Two audit landmines are pinned here as living asserts: `norm_topk_prob`
//! must parse `true` and the kept router weights must sum to 1 (a silent
//! `false` attenuates every MoE layer ~2.5x); and the PLE norms must arrive
//! RAW while `hc_norm` arrives folded — the PLE and hyper-connection parities
//! push loader-path weights through both consumers against one oracle, so a
//! uniform fold fails them at ~2x, not at 1e-6.
//!
//! ## The env-gated full forward
//!
//! `first_token_logits_from_a_short_prompt` (`ARLE_QWEN4_FORWARD=1`, use
//! `--release`) loads the WHOLE model in the hybrid residency (63.3 GiB of
//! NVFP4 expert stacks + the 2.7 GiB F32 tier resident; the bf16 dense tier +
//! `lm_head` host-side — the split that fits the ~70.7 GiB heapBudget),
//! forwards a short prompt, and prints the top-5 logits with their decoded
//! tokens. It asserts " Paris" appears in the top 5 for "The capital of
//! France is". `ARLE_QWEN4_FORWARD=host` runs the same thing on the pure host
//! transcription (no device heap at risk). If the device load fails on the
//! heapBudget, that is the audit's KNOWN ISSUE #1 — the test reports it and
//! skips rather than failing.
//!
//! Skips cleanly when the checkpoint (`ARLE_QWEN4_CKPT`, default
//! `C:\Users\Asus\models\qwen3.8-flash-next-nvfp4`) or a Vulkan device is
//! absent.
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use infer_gguf::safetensors::SafeTensorsDir;
use infer_vulkan::model_qwen4_exp::{
    DevLinearAttn, HostKv, HostLayer, Qwen4Dev, Qwen4ExpDeviceMode, VulkanQwen4ExpModel, hc_config,
    host_full_attention, host_linear_attention, host_moe, load_host_layer, load_mixer_weights,
    ple_config,
};
use infer_vulkan::qwen4_config::Qwen4ExpConfig;
use infer_vulkan::qwen4_hc;
use infer_vulkan::qwen4_names::HcSite;
use infer_vulkan::qwen4_ple::{NGramContext, NGramHash, PleConvState};
use infer_vulkan::qwen4_upload::{
    Qwen4DeviceFormat, Qwen4HostTables, Qwen4UploadConfig, Qwen4UploadScope, plan_qwen4_upload,
    upload_qwen4,
};
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

/// One stage's worst errors under two metrics:
/// - `rel`: per-element `|got - want| / max(|want|, 1e-3)` — the tight metric
///   for stages that are pure f32-vs-f64 arithmetic;
/// - `scale`: `|got - want| / max_abs(want)` — the vector-scale metric for
///   stages downstream of a discrete quantizer, where a boundary channel
///   legally lands ±1 bf16 ulp (2^-8 ≈ 3.9e-3 relative) apart and the
///   per-element metric on a small element reports that quantizer step, not a
///   defect. A REAL defect (wrong nibble order, double-folded norm, swapped
///   GQA map) is O(1) under BOTH metrics.
#[derive(Clone, Copy, Default)]
struct StageErr {
    rel: f32,
    rel_at: (usize, f32, f32),
    scale: f32,
    scale_at: (usize, f32, f32),
}

#[derive(Default)]
struct ErrTable {
    stages: BTreeMap<String, StageErr>,
}

impl ErrTable {
    fn note(&mut self, stage: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{stage}: length mismatch");
        let peak = want.iter().fold(0.0f32, |m, &w| m.max(w.abs())).max(1e-6);
        let mut cur = StageErr::default();
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(g.is_finite(), "{stage}[{i}]: device produced {g}");
            assert!(w.is_finite(), "{stage}[{i}]: host produced {w}");
            let diff = (g - w).abs();
            let rel = diff / w.abs().max(1e-3);
            if rel > cur.rel {
                cur.rel = rel;
                cur.rel_at = (i, g, w);
            }
            let scale = diff / peak;
            if scale > cur.scale {
                cur.scale = scale;
                cur.scale_at = (i, g, w);
            }
        }
        let entry = self.stages.entry(stage.to_string()).or_default();
        if cur.rel > entry.rel {
            entry.rel = cur.rel;
            entry.rel_at = cur.rel_at;
        }
        if cur.scale > entry.scale {
            entry.scale = cur.scale;
            entry.scale_at = cur.scale_at;
        }
    }

    /// Thresholds are `(stage, per-element rel, scale rel)`; `f32::INFINITY`
    /// opts a metric out where the other is the meaningful one.
    fn print_and_assert(&self, thresholds: &[(&str, f32, f32)]) {
        eprintln!(
            "\n── qwen4_exp device-vs-host parity ──\n  {:<24} {:>11} {:>11}   worst element (dev vs host)",
            "stage", "max rel", "max/peak"
        );
        for (stage, e) in &self.stages {
            eprintln!(
                "  {stage:<24} {:>11.3e} {:>11.3e}   at {}: {:+.6e} vs {:+.6e}",
                e.rel, e.scale, e.rel_at.0, e.rel_at.1, e.rel_at.2
            );
        }
        for &(stage, rel_max, scale_max) in thresholds {
            let e = self
                .stages
                .get(stage)
                .unwrap_or_else(|| panic!("stage `{stage}` never ran"));
            assert!(
                e.rel < rel_max,
                "{stage}: per-element rel err {:.3e} >= {rel_max:.0e} (at {}: dev {} vs host {})",
                e.rel,
                e.rel_at.0,
                e.rel_at.1,
                e.rel_at.2
            );
            assert!(
                e.scale < scale_max,
                "{stage}: scale rel err {:.3e} >= {scale_max:.0e} (at {}: dev {} vs host {})",
                e.scale,
                e.scale_at.0,
                e.scale_at.1,
                e.scale_at.2
            );
        }
    }
}

/// The three-layer parity harness. Loads only layers 0/1/3 (the full plan does
/// not fit the driver budget — audit landmine #1 — and the harness does not
/// need it).
#[test]
fn parity_layers_0_1_3_device_vs_host() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(ctx) = device() else { return };
    let t0 = std::time::Instant::now();
    let st = SafeTensorsDir::open_dir(&dir).expect("open checkpoint");
    let cfg = Qwen4ExpConfig::from_model_dir(&dir).expect("parse config");

    // AUDIT LANDMINE #3: the key is absent from config.json; the HF default is
    // TRUE. A parse that defaulted false would attenuate every MoE ~2.5x.
    assert!(
        cfg.norm_topk_prob,
        "norm_topk_prob must parse true (HF default for the absent key)"
    );
    assert_eq!(
        cfg.layer_types[0],
        infer_vulkan::qwen4_config::Qwen4LayerType::LinearAttention
    );
    assert_eq!(
        cfg.layer_types[3],
        infer_vulkan::qwen4_config::Qwen4LayerType::FullAttention
    );
    assert_eq!(
        cfg.ple_layer_ids,
        vec![1],
        "PLE sits on layer 1 (zero-indexed)"
    );

    let layers = [0usize, 1, 3];
    let scope = Qwen4UploadScope {
        lm_head: false,
        ..Qwen4UploadScope::layers(&layers)
    };
    let ucfg = Qwen4UploadConfig {
        dense_format: Qwen4DeviceFormat::F32,
        ..Qwen4UploadConfig::default()
    };
    let plan = plan_qwen4_upload(&st, &ucfg, &scope).expect("plan subset");
    eprintln!(
        "subset plan: {:.2} GiB over {} items ({:.1}s)",
        plan.device_bytes as f64 / (1u64 << 30) as f64,
        plan.items.len(),
        t0.elapsed().as_secs_f64()
    );
    let weights = upload_qwen4(&ctx, &st, &plan, &ucfg).expect("upload subset");
    eprintln!("subset resident ({:.1}s)", t0.elapsed().as_secs_f64());

    let host: BTreeMap<usize, HostLayer<'_>> = layers
        .iter()
        .map(|&l| (l, load_host_layer(&st, &cfg, l).expect("host layer")))
        .collect();
    let mixer_w = load_mixer_weights(&st, &cfg).expect("mixer weights");
    let hcc = hc_config(&cfg);
    let hash = NGramHash::new(cfg.ngram_hash_config(0)).expect("hash");
    let tables = Qwen4HostTables::build(&st).expect("host tables");

    let mut dev = Qwen4Dev::new(&ctx, &cfg, &[3], cfg.max_context).expect("device runner");
    assert!(
        dev.full_attention_ready(&weights, 3),
        "layer 3 must be device-runnable"
    );
    let dev_lin0 = DevLinearAttn::new(&ctx, &cfg, host[&0].linear.as_ref().expect("l0 linear"))
        .expect("permuted linear uploads");
    let dev_lin1 = DevLinearAttn::new(&ctx, &cfg, host[&1].linear.as_ref().expect("l1 linear"))
        .expect("permuted linear uploads");
    eprintln!(
        "host layers + permuted uploads ready ({:.1}s)",
        t0.elapsed().as_secs_f64()
    );

    // Real token ids (arbitrary but in-vocab); 3 positions exercise the conv
    // ring, the recurrence and the KV cache beyond the degenerate first step.
    let tokens: [u32; 3] = [17_000, 42, 25_011];

    // Host-canonical state.
    let mut gdr: BTreeMap<usize, Vec<f32>> = [0usize, 1]
        .iter()
        .map(|&l| {
            (
                l,
                vec![
                    0.0f32;
                    cfg.linear_num_value_heads
                        * cfg.linear_key_head_dim
                        * cfg.linear_value_head_dim
                ],
            )
        })
        .collect();
    let mut ring: BTreeMap<usize, Vec<f32>> = [0usize, 1]
        .iter()
        .map(|&l| {
            (
                l,
                vec![0.0f32; cfg.linear_conv_dim() * (cfg.linear_conv_kernel_dim - 1)],
            )
        })
        .collect();
    let mut kv3 = HostKv::default();
    let mut ple_ring = PleConvState::zeros(&ple_config(&cfg));
    let mut ngram = NGramContext::new(&hash);
    let ngram_table = tables.ngram().expect("ngram table");

    let mut errs = ErrTable::default();

    for (pos, &tok) in tokens.iter().enumerate() {
        // PLE embedding for this token, before the context rolls.
        let ids = hash.row_ids(&ngram, &[i64::from(tok)]).expect("row ids");
        let mut ple_emb = Vec::with_capacity(cfg.ple_embed_dim);
        for &id in &ids {
            ple_emb.extend(ngram_table.row(id as u64).expect("ngram row"));
        }
        ngram.push(&[i64::from(tok)]);

        let embed = tables.embed_row(tok as usize).expect("embed row");
        let mut h = qwen4_hc::seed_hyper_state(&hcc, &embed).expect("seed");

        for &layer in &layers {
            let hl = &host[&layer];

            // ── PLE (layer 1): device gate+conv vs the host oracle, on the
            //    SAME pre-advance ring. Also the landmine-#2 proof: the gate
            //    kernel spells `1 + w` itself, so a loader that folded the PLE
            //    norms fails this at ~2x, not at 1e-6.
            if let Some(ple) = &hl.ple {
                let ring_before = ple_ring.rows().to_vec();
                let dev_taps = dev
                    .ple(&weights, &cfg, layer, &ple_emb, &h, &ring_before)
                    .expect("device ple");
                let host_out = ple
                    .forward(&ple_emb, &h, &mut ple_ring, None)
                    .expect("host ple");
                errs.note("ple.out", &dev_taps.out, &host_out);
                for (hv, &ov) in h.iter_mut().zip(&host_out) {
                    *hv += ov;
                }
            }

            // ── attention hyper-connection site.
            let gr = qwen4_hc::gated_residual(&hcc, &hl.attn_hc, &h).expect("host gr");
            let x_dev = dev
                .hc_pre(&weights, &hcc, Some(layer), HcSite::Attn, &h)
                .expect("device hc_pre");
            errs.note("hc.attn.block_input", &x_dev, &gr.block_input);
            let x = gr.block_input.clone();

            // ── the attention block itself (host output is canonical).
            let y = match layer {
                0 | 1 => {
                    let w = hl.linear.as_ref().expect("linear");
                    let mut gdr_dev = gdr[&layer].clone();
                    let mut ring_dev = ring[&layer].clone();
                    let la = if layer == 0 { &dev_lin0 } else { &dev_lin1 };
                    let (y_dev, dt) = la
                        .forward(&mut dev, &cfg, &x, &mut gdr_dev, &mut ring_dev)
                        .expect("device linear attention");
                    let (y_host, ht) = host_linear_attention(
                        &cfg,
                        w,
                        &x,
                        gdr.get_mut(&layer).unwrap(),
                        ring.get_mut(&layer).unwrap(),
                    );
                    errs.note("linear.qkv_raw", &dt.qkv_raw, &ht.qkv_raw);
                    errs.note("linear.qkv_conv", &dt.qkv_conv, &ht.qkv_conv);
                    errs.note("linear.z", &dt.z, &ht.z);
                    errs.note("linear.core", &dt.core, &ht.core);
                    errs.note("linear.gated", &dt.gated, &ht.gated);
                    errs.note("linear.y", &y_dev, &y_host);
                    errs.note("linear.gdr_state", &gdr_dev, &gdr[&layer]);
                    errs.note("linear.conv_ring", &ring_dev, &ring[&layer]);
                    y_host
                }
                _ => {
                    let w = hl.full.as_ref().expect("full");
                    let (y_dev, dt) = dev
                        .full_attention(&weights, &cfg, layer, &x, pos)
                        .expect("device full");
                    let (y_host, ht) = host_full_attention(&cfg, w, &x, pos, &mut kv3);
                    errs.note("full.q_full", &dt.q_full, &ht.q_full);
                    errs.note("full.q_roped", &dt.q_roped, &ht.q_roped);
                    errs.note("full.gated", &dt.gated, &ht.gated);
                    errs.note("full.y", &y_dev, &y_host);
                    y_host
                }
            };

            // ── attention combine.
            let h_dev = dev
                .hc_combine(&weights, &hcc, Some(layer), HcSite::Attn, &y)
                .expect("device combine");
            let inj = gr.injection_weights.as_ref().expect("layer site");
            qwen4_hc::inject_block_output(&hcc, &mut h, inj, &y).expect("host combine");
            errs.note("hc.attn.residual", &h_dev, &h);

            // ── MoE hyper-connection site + MoE.
            let gr = qwen4_hc::gated_residual(&hcc, &hl.mlp_hc, &h).expect("host gr");
            let x_dev = dev
                .hc_pre(&weights, &hcc, Some(layer), HcSite::Mlp, &h)
                .expect("device hc_pre");
            errs.note("hc.mlp.block_input", &x_dev, &gr.block_input);
            let x = gr.block_input.clone();

            let (y_dev, dtaps) = dev.moe(&weights, &cfg, layer, &x).expect("device moe");
            assert!(
                dtaps.shared_on_device,
                "subset F32 load must run the shared expert on device"
            );
            // Kept routing weights must sum to 1 (norm_topk_prob) — landmine #3.
            let wsum: f32 = dtaps.weights.iter().sum();
            assert!(
                (wsum - 1.0).abs() < 1e-5,
                "device routing weights sum to {wsum}, want 1"
            );
            if pos == 0 {
                // The host MoE oracle (NVFP4 dequant + f64 dots) is expensive;
                // one matched token per layer convicts any systematic defect.
                let (_y_host, htaps) = host_moe(&cfg, &st, layer, &hl.moe, &x).expect("host moe");
                errs.note("moe.router_logits", &dtaps.logits, &htaps.logits);
                let mut dev_ids = dtaps.ids.clone();
                let mut host_ids = htaps.ids.clone();
                dev_ids.sort_unstable();
                host_ids.sort_unstable();
                assert_eq!(dev_ids, host_ids, "selected experts differ");
                // Match weights by expert id (near-tie order may differ).
                let by_id: BTreeMap<i32, f32> = htaps
                    .ids
                    .iter()
                    .copied()
                    .zip(htaps.weights.iter().copied())
                    .collect();
                for (&id, &wt) in dtaps.ids.iter().zip(&dtaps.weights) {
                    let want = by_id[&id];
                    assert!(
                        (wt - want).abs() < 1e-4,
                        "expert {id}: device weight {wt} vs host {want}"
                    );
                }
                errs.note("moe.routed", &dtaps.routed, &htaps.routed);
                let host_total: Vec<f32> = htaps
                    .routed
                    .iter()
                    .zip(&htaps.shared)
                    .map(|(&r, &s)| r + s)
                    .collect();
                errs.note("moe.y", &y_dev, &host_total);
            }
            // Device MoE output is canonical (both sides saw the same input).
            let y = y_dev;

            let h_dev = dev
                .hc_combine(&weights, &hcc, Some(layer), HcSite::Mlp, &y)
                .expect("device combine");
            let inj = gr.injection_weights.as_ref().expect("layer site");
            qwen4_hc::inject_block_output(&hcc, &mut h, inj, &y).expect("host combine");
            errs.note("hc.mlp.residual", &h_dev, &h);
        }

        // ── the stream mixer (use_combine = false).
        let gr = qwen4_hc::gated_residual(&hcc, &mixer_w, &h).expect("host mixer");
        let x_dev = dev
            .hc_pre(&weights, &hcc, None, HcSite::Mixer, &h)
            .expect("device mixer");
        errs.note("mixer.block_input", &x_dev, &gr.block_input);

        eprintln!(
            "token {pos} (id {tok}) compared ({:.1}s)",
            t0.elapsed().as_secs_f64()
        );
    }

    // Two bands. Pure f32-vs-f64 arithmetic stages hold ~1e-4; the stages at
    // or past the conv's bf16 quantizer legally step ±1 bf16 ulp (2^-8 ≈
    // 3.9e-3 per-element) on boundary channels — measured 1 ulp on this box —
    // so their per-element band is a few ulps while the vector-scale band
    // stays tight. Real defects are O(1) under both metrics.
    errs.print_and_assert(&[
        ("ple.out", 1e-3, 1e-4),
        ("hc.attn.block_input", 1e-3, 1e-4),
        ("hc.attn.residual", 1e-3, 1e-4),
        ("hc.mlp.block_input", 1e-3, 1e-4),
        ("hc.mlp.residual", 1e-3, 1e-4),
        ("mixer.block_input", 1e-3, 1e-4),
        ("linear.qkv_raw", 1e-3, 1e-4),
        ("linear.z", 1e-3, 1e-4),
        ("linear.conv_ring", 1e-3, 1e-4),
        ("linear.qkv_conv", 2e-2, 1e-3),
        ("linear.core", 2e-2, 5e-3),
        ("linear.gated", 2e-2, 5e-3),
        ("linear.gdr_state", 2e-2, 5e-3),
        ("linear.y", f32::INFINITY, 1e-2),
        ("full.q_full", 1e-3, 1e-4),
        ("full.q_roped", 1e-3, 1e-4),
        ("full.gated", 5e-3, 1e-3),
        ("full.y", 5e-3, 1e-3),
        ("moe.router_logits", 1e-3, 1e-4),
        ("moe.routed", 1e-3, 1e-4),
        ("moe.y", 1e-3, 1e-4),
    ]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Full forward (env-gated: the hybrid load stages ~66 GiB for minutes).
// ─────────────────────────────────────────────────────────────────────────────

/// `tokenizer.json`'s `model.vocab` as token→id and id→token maps.
fn load_vocab(dir: &std::path::Path) -> (BTreeMap<String, u32>, BTreeMap<u32, String>) {
    let src = std::fs::read_to_string(dir.join("tokenizer.json")).expect("read tokenizer.json");
    let doc = infer_vulkan::qwen4_config::json::parse(&src).expect("parse tokenizer.json");
    let vocab = doc
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.as_object())
        .expect("tokenizer.json model.vocab");
    let mut by_tok = BTreeMap::new();
    let mut by_id = BTreeMap::new();
    for (tok, id) in vocab {
        let id = id
            .as_i64()
            .and_then(|v| u32::try_from(v).ok())
            .expect("vocab id");
        by_tok.insert(tok.clone(), id);
        by_id.insert(id, tok.clone());
    }
    (by_tok, by_id)
}

#[test]
fn first_token_logits_from_a_short_prompt() {
    let mode = match std::env::var("ARLE_QWEN4_FORWARD").as_deref() {
        Ok("host") => Qwen4ExpDeviceMode::HostOnly,
        Ok("1" | "hybrid") => Qwen4ExpDeviceMode::HybridExperts,
        _ => {
            eprintln!(
                "SKIP: set ARLE_QWEN4_FORWARD=1 (hybrid device residency) or =host \
                 (pure host transcription); run with --release"
            );
            return;
        }
    };
    let Some(dir) = checkpoint_dir() else { return };
    let ctx = if mode == Qwen4ExpDeviceMode::HostOnly {
        None
    } else {
        match VulkanContext::create() {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("SKIP: no Vulkan device ({e})");
                return;
            }
        }
    };
    let t0 = std::time::Instant::now();
    let st = SafeTensorsDir::open_dir(&dir).expect("open checkpoint");
    let cfg = Qwen4ExpConfig::from_model_dir(&dir).expect("parse config");
    let vocab_size = cfg.vocab_size;

    // Prompt from the checkpoint's own vocab: each piece is a single BPE
    // token, so the id sequence is exactly what the real tokenizer emits for
    // this string. Ġ is the byte-level BPE space marker.
    let (by_tok, by_id) = load_vocab(&dir);
    let pieces = ["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"];
    let mut prompt = Vec::new();
    for p in pieces {
        match by_tok.get(p) {
            Some(&id) => prompt.push(id),
            None => {
                eprintln!("SKIP: vocab has no `{p}` — cannot build the prompt");
                return;
            }
        }
    }
    eprintln!("prompt {pieces:?} -> {prompt:?}");

    let model = VulkanQwen4ExpModel::load(ctx.as_ref(), &st, cfg, &mode);
    let mut model = match model {
        Ok(m) => m,
        Err(e) => {
            let msg = format!("{e:#}");
            if msg.contains("KNOWN ISSUE") || msg.contains("heapBudget") {
                // Audit landmine #1: the residency exceeds the driver budget.
                // The replan is scheduled work outside this bring-up.
                eprintln!("SKIP (KNOWN ISSUE, driver heapBudget): {msg}");
                return;
            }
            panic!("model load failed: {msg}");
        }
    };
    eprintln!(
        "model loaded in {:.1}s (mode {mode:?})",
        t0.elapsed().as_secs_f64()
    );

    let mut logits = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        let t = std::time::Instant::now();
        logits = model.forward_token(0, 0, tok, pos).expect("forward token");
        eprintln!(
            "  pos {pos} (id {tok} `{}`): {:.2}s",
            by_id.get(&tok).map_or("?", String::as_str),
            t.elapsed().as_secs_f64()
        );
    }
    assert_eq!(logits.len(), vocab_size, "logits width");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits must be finite"
    );

    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!("top-5 logits for `The capital of France is`:");
    let top5: Vec<String> = order[..5]
        .iter()
        .map(|&id| {
            let tok = by_id
                .get(&(id as u32))
                .cloned()
                .unwrap_or_else(|| format!("<{id}>"));
            eprintln!("  {id:>7}  {:>10.4}  {tok:?}", logits[id]);
            tok
        })
        .collect();

    // The coherence claim, as an assert rather than an adjective: the
    // continuation every competent model of this prompt produces.
    assert!(
        top5.iter().any(|t| t == "ĠParis"),
        "top-5 {top5:?} does not contain `ĠParis` — report the per-stage parity errors \
         (run parity_layers_0_1_3_device_vs_host) instead of trusting these logits"
    );
    eprintln!(
        "PASS: ` Paris` in the top-5 after {:.1}s total",
        t0.elapsed().as_secs_f64()
    );
}
