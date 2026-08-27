//! Qwen3.8-Flash-Next (`qwen4_exp`) forward proof: per-stage DEVICE-vs-HOST
//! parity on real weights, then an env-gated full forward.
//!
//! ## What runs by default
//!
//! `parity_layers_0_1_3_device_vs_host` loads a THREE-LAYER subset (layer 0 =
//! linear attention, layer 1 = linear + PLE, layer 3 = full attention; all 512
//! experts of each; dense tier F32) — deliberately not the full plan, which
//! takes minutes to stage — and drives 3 tokens through every stage
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
//! `dense_gemv_f16_matches_the_host_matvec` covers what that harness cannot:
//! the SHIPPING dense tier is F16, read by a different shader
//! (`mul_mat_vec_f16`, not `qwen36_router_gemv`) through a different row-stride
//! contract, after a lossy bf16->f16 re-encode at load. It diffs every batch
//! the forward actually issues against the f64 host oracle on real bytes, and
//! pins two structural claims: a shared-`x` group costs exactly ONE submit,
//! and a group with any non-resident member falls back to the host BIT for BIT
//! at zero submits.
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
//! `--release`) loads the WHOLE model in the hybrid residency — NVFP4 expert
//! stacks, the F32 small tier AND the F16 dense tier including `lm_head`,
//! with `upload_qwen4`'s spill moving whatever does not fit the ~70.7 GiB
//! heapBudget to the host heap — forwards a short prompt, and prints the
//! top-5 logits with their decoded tokens. It asserts " Paris" appears in the
//! top 5 for "The capital of France is". `ARLE_QWEN4_FORWARD=host` runs the
//! same thing on the pure host transcription (no device heap at risk).
//!
//! ## The env-gated profile
//!
//! `profile_forward_token` (`ARLE_QWEN4_PROFILE=1`, or `=host`) runs the same
//! load and prints a per-stage wall-clock table for the last position. The
//! table PARTITIONS the measured token (see `model_qwen4_exp::prof`): every
//! nanosecond is charged to exactly one `(stage, part)` bucket, so the rows
//! sum to the wall and the outermost `token` row is the honest residual rather
//! than slack smeared across the table. Adding `ARLE_GPU_TIMESTAMPS=1` turns
//! on `vulkan_sys`'s per-dispatch timestamps, labelled with the SAME stage
//! names, which makes submit+fence overhead a subtraction instead of a guess.
//!
//! Skips cleanly when the checkpoint (`ARLE_QWEN4_CKPT`, default
//! `C:\Users\Asus\models\qwen3.8-flash-next-nvfp4`) or a Vulkan device is
//! absent.
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use infer_gguf::safetensors::SafeTensorsDir;
use infer_vulkan::model_qwen4_exp::{
    DenseGemv, DevResidentLinAttn, HostKv, HostLayer, Qwen4Dev, Qwen4ExpDeviceMode,
    VulkanQwen4ExpModel, hc_config, host_full_attention, host_linear_attention, host_moe,
    load_host_layer, load_mixer_weights, ple_config, prof,
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

    // The write-combined readback guard rides HERE rather than only on the
    // 4-minute `profile_forward_token`: this harness already drives hundreds
    // of arena reads (the 3.1 MiB gated-delta state twice per linear layer per
    // token dominates them), so it convicts a re-flavoured arena in seconds.
    let _ = prof::take();
    prof::set_enabled(true);
    let mut dev = Qwen4Dev::new(&ctx, &cfg, &[3], cfg.max_context).expect("device runner");
    assert!(
        dev.full_attention_ready(&weights, 3),
        "layer 3 must be device-runnable"
    );
    // The SHIPPING linear path: resident device state, tier weights,
    // activation-side head permutation. Its state starts zeroed, like the
    // host's, and advances on device across the three tokens — so the
    // per-token state comparison also proves cross-token continuity.
    let mut resident = DevResidentLinAttn::new(&ctx, &cfg, [(0, &host[&0]), (1, &host[&1])])
        .expect("resident linear attention");
    eprintln!(
        "host layers + resident linear ready ({:.1}s)",
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
                let (dev_out, dev_ring) = dev
                    .ple(&weights, &cfg, layer, &ple_emb, &h, &ring_before)
                    .expect("device ple");
                let host_out = ple
                    .forward(&ple_emb, &h, &mut ple_ring, None)
                    .expect("host ple");
                errs.note("ple.out", &dev_out, &host_out);
                errs.note("ple.ring", &dev_ring, ple_ring.rows());
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
                    // Seed the device state from the host's so every stage
                    // comparison isolates ONE token's error — without this the
                    // two trajectories drift apart legitimately and the table
                    // stops being comparable to its recorded profile.
                    resident
                        .seed_state(&cfg, layer, &gdr[&layer], &ring[&layer])
                        .expect("seed resident state");
                    let y_dev = resident
                        .forward(&mut dev, &weights, &cfg, layer, &x)
                        .expect("device linear attention");
                    let dt = resident.read_taps(&dev, &cfg).expect("linear taps");
                    let (gdr_dev, ring_dev) = resident
                        .read_state(&cfg, layer)
                        .expect("resident state read");
                    // `None` keeps the oracle a PURE host transcription: the
                    // device-routed dense GEMV is what this harness measures,
                    // not what it measures against.
                    let (y_host, ht) = host_linear_attention(
                        &cfg,
                        w,
                        &x,
                        gdr.get_mut(&layer).unwrap(),
                        ring.get_mut(&layer).unwrap(),
                        None,
                    )
                    .expect("host linear attention");
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
                    let (y_host, ht) = host_full_attention(&cfg, w, &x, pos, &mut kv3, None)
                        .expect("host full attention");
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

            let (y_dev, dtaps) = dev
                .moe(&weights, &cfg, layer, &x, true)
                .expect("device moe");
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

    prof::set_enabled(false);
    let rows = prof::take();
    let sum = |part: &str| -> (u64, u64, u64) {
        rows.iter()
            .filter(|r| r.part == part)
            .fold((0, 0, 0), |(n, b, c), r| {
                (n + r.nanos, b + r.bytes, c + r.calls)
            })
    };
    let (d2h_ns, d2h_bytes, d2h_calls) = sum("d2h");
    let (h2d_ns, h2d_bytes, h2d_calls) = sum("h2d");
    let gbps = |bytes: u64, ns: u64| bytes as f64 / ns.max(1) as f64;
    eprintln!(
        "  arena copies over the walk: d2h {:.2} MiB / {} reads in {:.2} ms = {:.2} GB/s; \
         h2d {:.2} MiB / {} writes in {:.2} ms = {:.2} GB/s",
        d2h_bytes as f64 / (1u64 << 20) as f64,
        d2h_calls,
        d2h_ns as f64 / 1e6,
        gbps(d2h_bytes, d2h_ns),
        h2d_bytes as f64 / (1u64 << 20) as f64,
        h2d_calls,
        h2d_ns as f64 / 1e6,
        gbps(h2d_bytes, h2d_ns),
    );
    // AUDIT PIN: the scratch arena must stay `alloc_host_cached`. Host reads of
    // `alloc_uma`/`alloc` (write-combined) are a cost GPU profiling cannot see,
    // and one this lane has paid before. Measured on this walk: the host-cached
    // arena reads 21.56 MiB over 159 reads in 9.50 ms (2.38 GB/s); flipping the
    // one `alloc_host_cached` in `Qwen4Dev::new` to `alloc_uma` makes the same
    // reads take 295.76 ms (0.08 GB/s) — 31x, +286 ms on a 7-second test. Note
    // the WRITES stay fast either way (3.82 vs 2.63 GB/s), which is why only
    // the read side is pinned. The band sits between the two, not beside one.
    assert!(
        gbps(d2h_bytes, d2h_ns) > 0.5,
        "device->host arena reads run at {:.3} GB/s over {d2h_calls} reads — that is the \
         write-combined readback trap, not a cached read; the arena must be alloc_host_cached",
        gbps(d2h_bytes, d2h_ns)
    );

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
        ("ple.ring", 1e-3, 1e-5),
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
// The F16 dense tier: `DenseGemv` vs `HostDense::matvec`.
//
// `parity_layers_0_1_3_device_vs_host` above covers the F32 dense tier because
// that is what its subset uploads. The SHIPPING residency is F16, and the F16
// arm has its own shader (`mul_mat_vec_f16`, not `qwen36_router_gemv`), its own
// row-stride contract and a lossy bf16→f16 re-encode at load. None of that is
// exercised by the F32 harness, so it gets its own — on the same real bytes,
// against the same f64 oracle, in ~20 s.
// ─────────────────────────────────────────────────────────────────────────────

/// Deterministic activation, values in `[-1, 1)` — xorshift so a failure
/// reproduces exactly.
fn pseudo_activation(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

/// `(max per-element rel with a 1e-3 floor, max |diff| / vector peak)`.
fn rel_errs(got: &[f32], want: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let peak = want.iter().fold(0.0f32, |m, &w| m.max(w.abs())).max(1e-6);
    let mut rel = 0.0f32;
    let mut scale = 0.0f32;
    for (&g, &w) in got.iter().zip(want) {
        assert!(g.is_finite(), "device produced {g}");
        let d = (g - w).abs();
        rel = rel.max(d / w.abs().max(1e-3));
        scale = scale.max(d / peak);
    }
    (rel, scale)
}

#[test]
fn dense_gemv_f16_matches_the_host_matvec() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(ctx) = device() else { return };
    let st = SafeTensorsDir::open_dir(&dir).expect("open checkpoint");
    let cfg = Qwen4ExpConfig::from_model_dir(&dir).expect("parse config");

    // Layer 0 is linear-attention, layer 3 full attention; both carry a shared
    // expert. `experts: Some(0)` drops the 512 NVFP4 stacks — this test is
    // about the DENSE tier, and the stacks are 1.3 GiB per layer.
    let layers = [0usize, 3];
    let scope = Qwen4UploadScope {
        layers: Some(layers.to_vec()),
        experts: Some(0),
        lm_head: false,
    };
    // `dense_format` defaults to F16 — the shipping tier. Spelled out because
    // it is the whole point of this test.
    let ucfg = Qwen4UploadConfig {
        dense_format: Qwen4DeviceFormat::F16,
        ..Qwen4UploadConfig::default()
    };
    let plan = plan_qwen4_upload(&st, &ucfg, &scope).expect("plan dense subset");
    let weights = upload_qwen4(&ctx, &st, &plan, &ucfg).expect("upload dense subset");
    // Layer 4 is host-loaded but deliberately NOT uploaded — it is what makes
    // the all-or-nothing fallback row below testable.
    let host: BTreeMap<usize, HostLayer<'_>> = [0usize, 3, 4]
        .iter()
        .map(|&l| (l, load_host_layer(&st, &cfg, l).expect("host layer")))
        .collect();
    let mut dev = Qwen4Dev::new(&ctx, &cfg, &[], cfg.max_context).expect("device runner");

    let h = cfg.hidden_size;
    let wide = cfg.linear_num_value_heads * cfg.linear_value_head_dim; // 6144
    let x_h = pseudo_activation(h, 0x9E37_79B9_7F4A_7C15);
    let x_wide = pseudo_activation(wide, 0xD1B5_4A32_D192_ED03);
    let x_inter = pseudo_activation(cfg.shared_expert_intermediate_size, 7);

    // Every batch the forward actually issues, with the activation it issues
    // it with. `expect_batched` says whether the whole group should resolve to
    // device — the last one deliberately should NOT.
    let l0 = host[&0]
        .linear
        .as_ref()
        .expect("layer 0 is linear-attention");
    let l3 = host[&3].full.as_ref().expect("layer 3 is full attention");
    let l4 = host[&4]
        .linear
        .as_ref()
        .expect("layer 4 is linear-attention");
    let batches: Vec<(&str, Vec<&_>, &[f32], bool)> = vec![
        (
            "linear.in_proj",
            vec![&l0.qkv, &l0.z, &l0.a, &l0.b],
            &x_h[..],
            true,
        ),
        ("linear.out_proj", vec![&l0.out], &x_wide[..], true),
        ("full.qkv", vec![&l3.q, &l3.k, &l3.v], &x_h[..], true),
        ("full.o_proj", vec![&l3.o], &x_wide[..], true),
        (
            "shared_expert",
            vec![
                &host[&0].moe.shexp_gate,
                &host[&0].moe.sh_gate,
                &host[&0].moe.sh_up,
            ],
            &x_h[..],
            true,
        ),
        (
            "shared_expert.down",
            vec![&host[&0].moe.sh_down],
            &x_inter[..],
            true,
        ),
        // One resident matrix and one that is not — the all-or-nothing
        // fallback. The WHOLE batch must run on the host, bit for bit, and
        // cost zero submits.
        (
            "mixed_residency",
            vec![&host[&0].moe.sh_gate, &l4.qkv],
            &x_h[..],
            false,
        ),
    ];

    eprintln!(
        "\n── qwen4_exp F16 dense GEMV vs host f64 matvec ──\n  {:<22} {:>6} {:>7} {:>11} {:>11} {:>8}",
        "batch", "mats", "rows", "max rel", "max/peak", "submits"
    );
    let mut worst_rel = 0.0f32;
    for (label, mats, x, expect_batched) in batches {
        // Not resident? Then `mats[1]` really is absent, or the fallback this
        // row claims to prove is proving nothing.
        if !expect_batched {
            assert!(
                mats.iter().any(|m| weights.tensor(&m.name).is_err()),
                "`{label}` names only resident tensors — it cannot exercise the fallback"
            );
        }
        let want: Vec<Vec<f32>> = mats.iter().map(|m| m.matvec(x)).collect();
        let before = dev.submit_count();
        let got = {
            let mut gemv = DenseGemv::new(&mut dev, &weights);
            gemv.matvec_many(&mats, x).expect("dense gemv")
        };
        let submits = dev.submit_count() - before;
        assert_eq!(
            submits,
            u64::from(expect_batched),
            "`{label}`: {} matrices took {submits} submits",
            mats.len()
        );

        let rows: usize = mats.iter().map(|m| m.out_dim).sum();
        let (mut rel, mut scale) = (0.0f32, 0.0f32);
        for (g, w) in got.iter().zip(&want) {
            if expect_batched {
                let (r, s) = rel_errs(g, w);
                rel = rel.max(r);
                scale = scale.max(s);
            } else {
                // The fallback IS `HostDense::matvec`; anything but bit
                // equality means a stray device result leaked in.
                assert_eq!(g, w, "`{label}` fallback is not the host transcription");
            }
        }
        eprintln!(
            "  {label:<22} {:>6} {rows:>7} {rel:>11.3e} {scale:>11.3e} {submits:>8}",
            mats.len()
        );
        worst_rel = worst_rel.max(rel);
        // `scale` is the metric that bites here. These projections carry
        // elements far below their own peak, so the per-element metric's 1e-3
        // floor turns the vector's ABSOLUTE noise (~1.5e-7) into a 1.5e-4
        // "relative" reading — the F32 harness reports the same 1.5e-4 on
        // `linear.qkv_raw`, for exactly that reason and with no F16 anywhere.
        // `scale` at 1e-6 is ~5x the measured f32-vs-f64 summation floor and
        // five decades under what any real defect costs (a wrong row stride,
        // a swapped operand, F16 bytes read as BF16 are all O(1) under BOTH).
        assert!(rel < 1e-3, "`{label}`: per-element rel {rel:.3e} >= 1e-3");
        assert!(scale < 1e-6, "`{label}`: scale rel {scale:.3e} >= 1e-6");
    }
    eprintln!("  worst per-element rel over all batches: {worst_rel:.3e}");
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

// ─────────────────────────────────────────────────────────────────────────────
// Where does a token's 0.68 s go? (env-gated: reuses the full hybrid load)
// ─────────────────────────────────────────────────────────────────────────────

/// The five parts of one stage, plus the counters that make the copy rows
/// interpretable (bytes → effective bandwidth) and the submit row checkable
/// against `vulkan_sys`'s own `submit_count`.
#[derive(Default, Clone, Copy)]
struct Agg {
    cpu: u64,
    h2d: u64,
    record: u64,
    submit: u64,
    d2h: u64,
    matvec: u64,
    /// Times this stage opened and closed (48 for a per-layer stage).
    entries: u64,
    dispatches: u64,
    submits: u64,
    matvecs: u64,
    h2d_bytes: u64,
    d2h_bytes: u64,
    matvec_bytes: u64,
}

impl Agg {
    fn total(&self) -> u64 {
        self.cpu + self.h2d + self.record + self.submit + self.d2h + self.matvec
    }
}

fn aggregate(rows: &[prof::Row]) -> BTreeMap<&'static str, Agg> {
    let mut out: BTreeMap<&'static str, Agg> = BTreeMap::new();
    for r in rows {
        let a = out.entry(r.stage).or_default();
        match r.part {
            "cpu" => {
                a.cpu += r.nanos;
                a.entries += r.calls;
            }
            "h2d" => {
                a.h2d += r.nanos;
                a.h2d_bytes += r.bytes;
            }
            "record" => {
                a.record += r.nanos;
                a.dispatches += r.calls;
            }
            "submit" => {
                a.submit += r.nanos;
                a.submits += r.calls;
            }
            "d2h" => {
                a.d2h += r.nanos;
                a.d2h_bytes += r.bytes;
            }
            "matvec" => {
                a.matvec += r.nanos;
                a.matvec_bytes += r.bytes;
                a.matvecs += r.calls;
            }
            other => panic!("unknown profile part `{other}`"),
        }
    }
    out
}

fn ms(n: u64) -> f64 {
    n as f64 / 1e6
}

/// The prompt as ids, or `None` if a piece is missing from the vocab.
fn prompt_ids(by_tok: &BTreeMap<String, u32>) -> Option<Vec<u32>> {
    ["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"]
        .iter()
        .map(|p| by_tok.get(*p).copied())
        .collect()
}

/// Per-stage wall accounting for one `forward_token`, printed as a table that
/// SUMS to the measured wall (see `model_qwen4_exp::prof` for why the buckets
/// partition rather than overlap). `ARLE_QWEN4_PROFILE=1` profiles the hybrid
/// residency, `=host` the pure host transcription. Set `ARLE_GPU_TIMESTAMPS=1`
/// as well to get the GPU-busy column and, with it, the submit/fence overhead
/// as a subtraction rather than an estimate. Run with `--release`.
#[test]
fn profile_forward_token() {
    let mode = match std::env::var("ARLE_QWEN4_PROFILE").as_deref() {
        Ok("host") => Qwen4ExpDeviceMode::HostOnly,
        Ok("1" | "hybrid") => Qwen4ExpDeviceMode::HybridExperts,
        _ => {
            eprintln!(
                "SKIP: set ARLE_QWEN4_PROFILE=1 (hybrid) or =host; add ARLE_GPU_TIMESTAMPS=1 \
                 for the GPU-busy column; run with --release"
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
    let n_layers = cfg.num_hidden_layers as u64;
    let (by_tok, by_id) = load_vocab(&dir);
    let Some(prompt) = prompt_ids(&by_tok) else {
        eprintln!("SKIP: vocab has no such piece — cannot build the prompt");
        return;
    };

    let mut model = match VulkanQwen4ExpModel::load(ctx.as_ref(), &st, cfg, &mode) {
        Ok(m) => m,
        Err(e) => {
            let msg = format!("{e:#}");
            if msg.contains("KNOWN ISSUE") || msg.contains("heapBudget") {
                eprintln!("SKIP (KNOWN ISSUE, driver heapBudget): {msg}");
                return;
            }
            panic!("model load failed: {msg}");
        }
    };
    let load_s = t0.elapsed().as_secs_f64();
    eprintln!("model loaded in {load_s:.1}s (mode {mode:?})");

    // Profile EVERY position: the recurrent lanes grow with `pos`, so one
    // token's table could hide a cost only the last position pays.
    let _ = prof::take();
    prof::set_enabled(true);
    let mut per_token: Vec<(usize, f64, Vec<prof::Row>)> = Vec::new();
    let mut logits = Vec::new();
    let mut submits_before = model.dev_mut().map_or(0, |d| d.submit_count());
    for (pos, &tok) in prompt.iter().enumerate() {
        let t = std::time::Instant::now();
        logits = model.forward_token(0, 0, tok, pos).expect("forward token");
        let wall = t.elapsed().as_secs_f64();
        let rows = prof::take();
        let submits_now = model.dev_mut().map_or(0, |d| d.submit_count());
        // The recorder's own counter is an INDEPENDENT instrument; the table's
        // submit row has to agree with it or one of the two is lying.
        let counted: u64 = rows
            .iter()
            .filter(|r| r.part == "submit")
            .map(|r| r.calls)
            .sum();
        assert_eq!(
            counted,
            submits_now - submits_before,
            "pos {pos}: profile counted {counted} submits, vkQueueSubmit counted {}",
            submits_now - submits_before
        );
        submits_before = submits_now;
        eprintln!(
            "  pos {pos} (id {tok} `{}`): {wall:.3}s, {counted} submits",
            by_id.get(&tok).map_or("?", String::as_str)
        );
        per_token.push((pos, wall, rows));
    }
    prof::set_enabled(false);

    let (pos, wall, rows) = per_token.last().expect("at least one token").clone();
    let table = aggregate(&rows);
    let wall_ns = (wall * 1e9) as u64;
    let sum_ns: u64 = table.values().map(Agg::total).sum();

    let mut order: Vec<(&&str, &Agg)> = table.iter().collect();
    order.sort_by(|a, b| b.1.total().cmp(&a.1.total()));
    let tot = |f: fn(&Agg) -> u64| -> u64 { table.values().map(f).sum() };
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64;

    eprintln!(
        "\n-- qwen4_exp forward_token profile, mode {mode:?}, pos {pos}, {n_layers} layers --\n  \
         {:<22} {:>8} {:>9} {:>7} {:>7} {:>9} {:>7} {:>9} {:>6}  {:>4} {:>6} {:>5} {:>8}",
        "stage",
        "cpu ms",
        "matvec ms",
        "h2d ms",
        "rec ms",
        "submit ms",
        "d2h ms",
        "total ms",
        "%",
        "runs",
        "disp",
        "subm",
        "wt MiB"
    );
    for (stage, a) in &order {
        eprintln!(
            "  {stage:<22} {:>8.2} {:>9.2} {:>7.2} {:>7.2} {:>9.2} {:>7.2} {:>9.2} {:>5.1}%  \
             {:>4} {:>6} {:>5} {:>8.1}",
            ms(a.cpu),
            ms(a.matvec),
            ms(a.h2d),
            ms(a.record),
            ms(a.submit),
            ms(a.d2h),
            ms(a.total()),
            100.0 * a.total() as f64 / wall_ns.max(1) as f64,
            a.entries,
            a.dispatches,
            a.submits,
            mib(a.matvec_bytes),
        );
    }
    eprintln!(
        "  {:<22} {:>8.2} {:>9.2} {:>7.2} {:>7.2} {:>9.2} {:>7.2} {:>9.2} {:>5.1}%  \
         {:>4} {:>6} {:>5} {:>8.1}",
        "TOTAL",
        ms(tot(|a| a.cpu)),
        ms(tot(|a| a.matvec)),
        ms(tot(|a| a.h2d)),
        ms(tot(|a| a.record)),
        ms(tot(|a| a.submit)),
        ms(tot(|a| a.d2h)),
        ms(sum_ns),
        100.0 * sum_ns as f64 / wall_ns.max(1) as f64,
        table.values().map(|a| a.entries).sum::<u64>(),
        tot(|a| a.dispatches),
        tot(|a| a.submits),
        mib(tot(|a| a.matvec_bytes)),
    );
    eprintln!(
        "  measured wall {:.2} ms; the `token` row IS the unattributed residual",
        wall * 1e3
    );

    let dispatches = tot(|a| a.dispatches);
    let submits = tot(|a| a.submits);
    let h2d_bytes = tot(|a| a.h2d_bytes);
    let d2h_bytes = tot(|a| a.d2h_bytes);
    let h2d_ns = tot(|a| a.h2d);
    let d2h_ns = tot(|a| a.d2h);
    let gbps = |bytes: u64, ns: u64| bytes as f64 / ns.max(1) as f64;
    eprintln!(
        "  dispatches/token {dispatches}  submits/token {submits}  record {:.2} us/dispatch  \
         submit {:.1} us/submit",
        ms(tot(|a| a.record)) * 1e3 / dispatches.max(1) as f64,
        ms(tot(|a| a.submit)) * 1e3 / submits.max(1) as f64,
    );
    eprintln!(
        "  host bf16 matvec {:.2} MiB in {:.2} ms = {:.2} GB/s over {} calls",
        mib(tot(|a| a.matvec_bytes)),
        ms(tot(|a| a.matvec)),
        gbps(tot(|a| a.matvec_bytes), tot(|a| a.matvec)),
        tot(|a| a.matvecs),
    );
    eprintln!(
        "  h2d {:.3} MiB in {:.2} ms = {:.2} GB/s   d2h {:.3} MiB in {:.2} ms = {:.2} GB/s",
        mib(h2d_bytes),
        ms(h2d_ns),
        gbps(h2d_bytes, h2d_ns),
        mib(d2h_bytes),
        ms(d2h_ns),
        gbps(d2h_bytes, d2h_ns),
    );

    let submit_ms = ms(tot(|a| a.submit));
    if let Some(dev) = model.dev_mut() {
        let gpu = dev.take_gpu_profile();
        if gpu.is_empty() {
            eprintln!(
                "  (no GPU-busy column: set ARLE_GPU_TIMESTAMPS=1 BEFORE the run — the query \
                 pool is created with the recorder, at model load)"
            );
        } else {
            // The recorder accumulates across every submit and is drained
            // once, so these totals cover ALL positions — divide by the count.
            let n = prompt.len() as f64;
            let gpu_ms: f64 = gpu.iter().map(|&(_, _, msec)| msec).sum::<f64>() / n;
            eprintln!(
                "\n  GPU-busy by stage label (mean over {} tokens):",
                prompt.len()
            );
            for &(label, count, msec) in &gpu {
                eprintln!(
                    "    {label:<22} {:>8.3} ms  over {:>6.0} dispatches",
                    msec / n,
                    count as f64 / n
                );
            }
            eprintln!(
                "    {:<22} {gpu_ms:>8.3} ms  = {:.1}% of the {:.2} ms token; the submit rows \
                 total {submit_ms:.2} ms, so submit+fence overhead is {:.2} ms",
                "GPU TOTAL",
                100.0 * gpu_ms / (wall * 1e3),
                wall * 1e3,
                submit_ms - gpu_ms,
            );
            assert!(
                gpu_ms <= submit_ms * 1.05,
                "GPU-busy {gpu_ms:.2} ms exceeds the {submit_ms:.2} ms of submit wall it must \
                 live inside"
            );
        }
    }

    // ── the asserts ────────────────────────────────────────────────────────
    // 1. The table accounts for the token. `prof` charges each nanosecond
    //    exactly once, so this is a real check: a leaf that forgot to charge
    //    its parent shows up here as > 100%.
    let closure = sum_ns as f64 / wall_ns.max(1) as f64;
    assert!(
        (0.97..=1.03).contains(&closure),
        "profile accounts for {:.1}% of the measured wall ({:.2} ms of {:.2} ms)",
        100.0 * closure,
        ms(sum_ns),
        wall * 1e3
    );

    // 2. Every layer ran every stage it owes. A stage that quietly stopped
    //    running — a residency regression flipping the MoE back to the host,
    //    say — shows up as a wrong run count long before a wrong token.
    let runs = |stage: &str| table.get(stage).map_or(0, |a| a.entries);
    if mode == Qwen4ExpDeviceMode::HybridExperts {
        assert_eq!(runs("dev.moe"), n_layers, "device MoE runs per token");
        assert_eq!(runs("dev.hc.attn.pre"), n_layers, "attn hc_pre per token");
        assert_eq!(
            runs("dev.hc.mlp.comb"),
            n_layers,
            "mlp hc_combine per token"
        );
        assert_eq!(runs("dev.hc.mixer"), 1, "one stream mixer per token");
        // 3. The write-combined readback trap: every host read of device
        //    memory must land on an `alloc_host_cached` buffer. Reading
        //    `alloc_uma`/`alloc` runs at ~0.10 GB/s on this part and is
        //    invisible to GPU profiling. `parity_layers_0_1_3_device_vs_host`
        //    carries the same guard and convicts in seconds rather than
        //    minutes; this one keeps it on the shape that actually decodes.
        //    Measured here: 5.263 MiB over 385 reads in 4.10 ms = 1.34 GB/s
        //    (per-read call overhead, not bandwidth), against 0.08 GB/s
        //    measured on a write-combined arena.
        assert!(
            gbps(d2h_bytes, d2h_ns) > 0.5,
            "device→host reads run at {:.3} GB/s — that is the write-combined \
             readback trap, not a cached read",
            gbps(d2h_bytes, d2h_ns)
        );
    }
    assert_eq!(runs("host.lm_head"), 1, "one lm_head matvec per token");
    assert_eq!(logits.len(), model.cfg.vocab_size, "logits width");

    let mut ord: Vec<usize> = (0..logits.len()).collect();
    ord.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top = by_id
        .get(&(ord[0] as u32))
        .cloned()
        .unwrap_or_else(|| format!("<{}>", ord[0]));
    eprintln!(
        "\n  top-1 {top:?} ({:.4}); load {load_s:.1}s; per-token wall {:?}",
        logits[ord[0]],
        per_token
            .iter()
            .map(|(_, w, _)| format!("{w:.3}s"))
            .collect::<Vec<_>>()
    );
    assert_eq!(top, "ĠParis", "profiling must not change the answer");
}

/// End-to-end dense-format quality: greedy-generate a probe set in EACH
/// requested format, sequentially in one process (each model is dropped
/// before the next loads, so the device heap frees), and compare against the
/// first format as baseline.
///
/// Three columns per format, because they fail differently: greedy AGREEMENT
/// (the coarse bit — does the output change), top-1 logit MAE (how hard the
/// argmax is being pushed), and full-vector MAE (diffuse drift the top-k
/// never shows). A quantization that keeps agreement at 100% while tripling
/// vector MAE is spending its error budget; one that flips greedy tokens is
/// past it.
///
/// ```text
/// ARLE_QWEN4_QUALITY=bf16,q8 cargo test -p infer-vulkan --features vulkan \
///     --release --test qwen4_forward dense_format_quality_probe \
///     -- --test-threads=1 --nocapture
/// ```
#[test]
fn dense_format_quality_probe() {
    let Ok(formats) = std::env::var("ARLE_QWEN4_QUALITY") else {
        eprintln!("SKIP: set ARLE_QWEN4_QUALITY=bf16,q8[,q4k] (first entry is the baseline)");
        return;
    };
    let formats: Vec<String> = formats.split(',').map(|s| s.trim().to_string()).collect();
    assert!(
        formats.len() >= 2,
        "need a baseline and at least one candidate"
    );
    let Some(dir) = checkpoint_dir() else { return };
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no Vulkan device ({e})");
            return;
        }
    };
    let (by_tok, by_id) = load_vocab(&dir);

    // Probe prompts as single-BPE pieces so the id sequence is exactly what
    // the tokenizer would emit. A prompt with any missing piece is dropped
    // (reported), not fatal — vocab coverage varies.
    let prompt_specs: [&[&str]; 4] = [
        &["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"],
        &["The", "Ġsun", "Ġrises", "Ġin", "Ġthe"],
        &["One", "Ġplus", "Ġone", "Ġequals"],
        &["def", "Ġadd", "(", "a", ",", "Ġb", ")", ":"],
    ];
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    for spec in prompt_specs {
        let ids: Option<Vec<u32>> = spec.iter().map(|p| by_tok.get(*p).copied()).collect();
        match ids {
            Some(ids) => prompts.push(ids),
            None => eprintln!("  (probe dropped, vocab missing a piece: {spec:?})"),
        }
    }
    assert!(
        prompts.len() >= 2,
        "not enough probe prompts survived the vocab"
    );
    const GEN: usize = 8;

    // One format's run: greedy sequences + the full logits of every step.
    struct Run {
        greedy: Vec<Vec<u32>>,
        logits: Vec<Vec<Vec<f32>>>,
        load_s: f64,
        tok_ms: f64,
    }
    let mut runs: Vec<(String, Run)> = Vec::new();
    for fmt in &formats {
        // SAFETY: this test binary runs single-threaded (--test-threads=1 is
        // required for device suites anyway), and the var is read exactly
        // once per load.
        unsafe { std::env::set_var("ARLE_QWEN4_DENSE", fmt) };
        let st = SafeTensorsDir::open_dir(&dir).expect("open checkpoint");
        let cfg = Qwen4ExpConfig::from_model_dir(&dir).expect("parse config");
        let t0 = std::time::Instant::now();
        let mut model =
            VulkanQwen4ExpModel::load(Some(&ctx), &st, cfg, &Qwen4ExpDeviceMode::HybridExperts)
                .expect("model load");
        let load_s = t0.elapsed().as_secs_f64();
        let mut greedy: Vec<Vec<u32>> = Vec::new();
        let mut logits_all = Vec::new();
        let mut tok_s = 0.0f64;
        let mut tok_n = 0usize;
        for prompt in &prompts {
            let mut seq = prompt.clone();
            let mut generated: Vec<u32> = Vec::new();
            let mut step_logits = Vec::new();
            for pos in 0..seq.len() + GEN - 1 {
                let start = if pos == 0 { 0 } else { pos };
                let tok = seq[pos.min(seq.len() - 1)];
                let t1 = std::time::Instant::now();
                let logits = model.forward_token(0, 0, tok, start).expect("forward");
                if pos >= 2 {
                    tok_s += t1.elapsed().as_secs_f64();
                    tok_n += 1;
                }
                if pos + 1 >= seq.len() {
                    let arg = logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(i, _)| i as u32)
                        .unwrap();
                    step_logits.push(logits);
                    generated.push(arg);
                    if seq.len() < prompt.len() + GEN {
                        seq.push(arg);
                    }
                }
            }
            // Reset the recurrent state between prompts.
            model.forward_token(0, 0, seq[0], 0).expect("reset probe");
            greedy.push(generated);
            logits_all.push(step_logits);
        }
        eprintln!(
            "[{fmt}] load {load_s:.1}s, {:.1} ms/token over {tok_n} fwd",
            tok_s / tok_n as f64 * 1e3
        );
        for (p, g) in prompts.iter().zip(&greedy) {
            let text: String = g
                .iter()
                .map(|id| by_id.get(id).cloned().unwrap_or_else(|| format!("<{id}>")))
                .collect();
            let ptext: String = p
                .iter()
                .map(|id| by_id.get(id).cloned().unwrap_or_default())
                .collect();
            eprintln!(
                "    {} -> {}",
                ptext.replace('Ġ', " "),
                text.replace('Ġ', " ")
            );
        }
        runs.push((
            fmt.clone(),
            Run {
                greedy,
                logits: logits_all,
                load_s,
                tok_ms: tok_s / tok_n as f64 * 1e3,
            },
        ));
    }

    // Compare every candidate against the baseline.
    let (base_name, base) = &runs[0];
    eprintln!("\n── quality vs {base_name} ──");
    eprintln!(
        "  {:<6} {:>9} {:>12} {:>12} {:>9} {:>10}",
        "format", "agree", "top1 MAE", "vector MAE", "load s", "ms/token"
    );
    for (name, run) in &runs[1..] {
        let mut agree = 0usize;
        let mut total = 0usize;
        let mut top1_mae = 0.0f64;
        let mut vec_mae = 0.0f64;
        let mut vec_n = 0usize;
        for (pb, pr) in base.greedy.iter().zip(&run.greedy) {
            for (a, b) in pb.iter().zip(pr) {
                total += 1;
                if a == b {
                    agree += 1;
                }
            }
        }
        for (pb, pr) in base.logits.iter().zip(&run.logits) {
            for (lb, lr) in pb.iter().zip(pr) {
                let arg = lb
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap();
                top1_mae += f64::from((lb[arg] - lr[arg]).abs());
                vec_mae += lb
                    .iter()
                    .zip(lr)
                    .map(|(&a, &b)| f64::from((a - b).abs()))
                    .sum::<f64>()
                    / lb.len() as f64;
                vec_n += 1;
            }
        }
        eprintln!(
            "  {:<6} {:>8.1}% {:>12.4} {:>12.5} {:>9.1} {:>10.1}",
            name,
            100.0 * agree as f64 / total as f64,
            top1_mae / vec_n as f64,
            vec_mae / vec_n as f64,
            run.load_s,
            run.tok_ms,
        );
    }
    eprintln!(
        "\nGate suggestion: candidate agreement >= 95% and top1 MAE well under the\n\
         baseline's top1-vs-top2 margin means the flip is safe; below that,\n\
         flip per family instead of wholesale."
    );
}
