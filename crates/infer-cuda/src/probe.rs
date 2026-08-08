//! Inference-analysis probe: per-token entropy/NLL + decode logit lens, JSONL.
//!
//! Off (no `ARLE_PROBE_JSONL`) every hook is one `OnceLock` load + a compare —
//! no alloc, no D2H, no formatting. Only TP rank 0 (or an unset rank, which is
//! rank 0 by `tp::resolve_tp_config` semantics) resolves an active config; other
//! ranks are permanent no-ops. Probe I/O failures disable the probe, never
//! inference.

#[cfg(feature = "cuda")]
use std::cell::{Cell, RefCell};
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceContext, HiddenStates};

pub(crate) struct ProbeConfig {
    lens_layers: usize,
    token_entropy: bool,
    stages: bool,
    /// `ARLE_PROBE_STAGE_FULL`: append the whole row to each stage record.
    /// A summary cannot answer "how far apart are two runtimes' vectors" —
    /// that needs `||a-b|| / ||a||`, so the full row has to cross the wire.
    stage_full: bool,
    path: PathBuf,
}

/// Unlike `executor::dump_topk_positions` (literal `"0"` required), an UNSET
/// rank var counts as rank 0 — mirroring `tp::resolve_tp_config`.
fn is_rank0() -> bool {
    std::env::var("INFER_TP_RANK")
        .ok()
        .or_else(|| std::env::var("ARLE_TP_RANK").ok())
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0)
        == 0
}

fn init_config() -> Option<ProbeConfig> {
    let path = std::env::var("ARLE_PROBE_JSONL").ok()?;
    let path = path.trim();
    if path.is_empty() || !is_rank0() {
        return None;
    }
    let lens_layers = std::env::var("ARLE_PROBE_LENS_LAYERS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(10);
    let token_entropy = std::env::var("ARLE_PROBE_TOKEN_ENTROPY")
        .ok()
        .is_none_or(|value| value.trim() != "0");
    let stages = std::env::var("ARLE_PROBE_STAGES")
        .ok()
        .is_some_and(|value| value.trim() != "0" && !value.trim().is_empty());
    let stage_full = std::env::var("ARLE_PROBE_STAGE_FULL")
        .ok()
        .is_some_and(|value| value.trim() != "0" && !value.trim().is_empty());
    Some(ProbeConfig {
        lens_layers,
        token_entropy,
        stages,
        stage_full,
        path: PathBuf::from(path),
    })
}

fn config() -> Option<&'static ProbeConfig> {
    static CONFIG: OnceLock<Option<ProbeConfig>> = OnceLock::new();
    CONFIG.get_or_init(init_config).as_ref()
}

/// Logit-lens depth (last N layers); 0 when the probe or the lens is off.
#[inline]
pub(crate) fn lens_layers() -> usize {
    config().map_or(0, |cfg| cfg.lens_layers)
}

/// Whether per-token entropy/NLL records are on (false when the probe is off).
#[inline]
pub(crate) fn token_entropy() -> bool {
    config().is_some_and(|cfg| cfg.token_entropy)
}

// ── Host math ────────────────────────────────────────────────────────────────

/// One-pass `logsumexp` + entropy over raw (T=1) logits, max-shifted for
/// stability: `H = lse − Σ p·l` with `Σ p·l = (Σ e^(l−max)·l) / Σ e^(l−max)`.
fn lse_entropy(logits: &[f32]) -> (f64, f32) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let (mut sum_exp, mut sum_el) = (0f64, 0f64);
    for &l in logits {
        let e = (f64::from(l) - max).exp();
        sum_exp += e;
        sum_el += e * f64::from(l);
    }
    let lse = max + sum_exp.ln();
    (lse, (lse - sum_el / sum_exp) as f32)
}

/// Entropy of the next-token distribution plus the NLL of `target` under it
/// (`None` when the target is unknown, e.g. the last prefill row of a chunk).
pub(crate) fn entropy_nll(logits: &[f32], target: Option<u32>) -> (f32, Option<f32>) {
    let (lse, entropy) = lse_entropy(logits);
    let nll = target
        .and_then(|t| logits.get(t as usize))
        .map(|&l| (lse - f64::from(l)) as f32);
    (entropy, nll)
}

pub(crate) struct LensStats {
    pub(crate) entropy: f32,
    pub(crate) nll: f32,
    pub(crate) top1: u32,
    pub(crate) top1_logprob: f32,
}

/// Lens record math: entropy, NLL of the finally-emitted `target`, and the
/// layer distribution's top-1 (id, logprob). `None` if `target` is out of vocab.
pub(crate) fn lens_stats(logits: &[f32], target: u32) -> Option<LensStats> {
    let &target_logit = logits.get(target as usize)?;
    let (top1, &top1_logit) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?;
    let (lse, entropy) = lse_entropy(logits);
    Some(LensStats {
        entropy,
        nll: (lse - f64::from(target_logit)) as f32,
        top1: top1 as u32,
        top1_logprob: (f64::from(top1_logit) - lse) as f32,
    })
}

// ── JSONL writer (rank-0 only, lazy, flush per line) ─────────────────────────

static DISABLED: AtomicBool = AtomicBool::new(false);
/// Set by the first `lens_begin`; stays false on a forward that never captures.
static LENS_EVER_ARMED: AtomicBool = AtomicBool::new(false);

fn fail(err: &dyn fmt::Display) {
    if !DISABLED.swap(true, Ordering::Relaxed) {
        eprintln!("[probe] write failed, probe disabled: {err}");
    }
}

fn writer() -> Option<&'static Mutex<BufWriter<File>>> {
    static WRITER: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();
    WRITER
        .get_or_init(|| {
            let cfg = config()?;
            let file = match File::create(&cfg.path) {
                Ok(file) => file,
                Err(err) => {
                    fail(&format_args!("open {}: {err}", cfg.path.display()));
                    return None;
                }
            };
            let mut out = BufWriter::new(file);
            if let Err(err) = writeln!(
                out,
                "{{\"phase\":\"meta\",\"lens_layers\":{},\"token_entropy\":{}}}",
                cfg.lens_layers, cfg.token_entropy
            )
            .and_then(|()| out.flush())
            {
                fail(&err);
                return None;
            }
            Some(Mutex::new(out))
        })
        .as_ref()
}

fn emit(line: fmt::Arguments) {
    if DISABLED.load(Ordering::Relaxed) {
        return;
    }
    let Some(writer) = writer() else { return };
    let Ok(mut writer) = writer.lock() else {
        fail(&"poisoned probe writer mutex");
        return;
    };
    if let Err(err) = writeln!(writer, "{line}").and_then(|()| writer.flush()) {
        fail(&err);
    }
}

/// Warn once when the lens is configured on a model whose forward never arms it
/// (only the DSv4 executor calls `lens_begin`). Without this the `meta` line
/// reports `lens_layers: N` and no lens record ever follows, which reads as
/// "the lens found nothing".
fn warn_lens_unwired() {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if lens_layers() > 0
        && !LENS_EVER_ARMED.load(Ordering::Relaxed)
        && !WARNED.swap(true, Ordering::Relaxed)
    {
        eprintln!(
            "[probe] ARLE_PROBE_LENS_LAYERS is set but this model's forward does not \
             capture lens layers; no lens records will be emitted"
        );
    }
}

/// `prefill`/`decode` record. `token`/`nll` are `None` only for the last
/// prefill row of a chunk (the target crosses the chunk boundary).
pub(crate) fn emit_token(
    phase: &str,
    pos: u64,
    token: Option<u32>,
    nll: Option<f32>,
    entropy: f32,
) {
    warn_lens_unwired();
    match (token, nll) {
        (Some(token), Some(nll)) => emit(format_args!(
            "{{\"phase\":\"{phase}\",\"pos\":{pos},\"token\":{token},\"nll\":{nll:.6},\"entropy\":{entropy:.6}}}"
        )),
        (Some(token), None) => emit(format_args!(
            "{{\"phase\":\"{phase}\",\"pos\":{pos},\"token\":{token},\"entropy\":{entropy:.6}}}"
        )),
        _ => emit(format_args!(
            "{{\"phase\":\"{phase}\",\"pos\":{pos},\"entropy\":{entropy:.6}}}"
        )),
    }
}

fn emit_lens(pos: u64, layer: usize, stats: &LensStats, emitted: u32) {
    emit(format_args!(
        "{{\"phase\":\"lens\",\"pos\":{pos},\"layer\":{layer},\"entropy\":{:.6},\"nll\":{:.6},\"top1\":{},\"top1_logprob\":{:.6},\"agree\":{}}}",
        stats.entropy,
        stats.nll,
        stats.top1,
        stats.top1_logprob,
        stats.top1 == emitted
    ))
}

// ── Lens stash (cuda-gated: holds device buffers) ────────────────────────────
// The layer loop stashes per-layer DEVICE logits (owned buffers keep them alive
// past the loop); D2H + math + emit happen only at `lens_flush`, which sits at
// the existing post-sampling sync point — never mid-loop.

#[cfg(feature = "cuda")]
thread_local! {
    static LENS_STASH: RefCell<Vec<(usize, HiddenStates)>> = const { RefCell::new(Vec::new()) };
    static LENS_ARMED: Cell<bool> = const { Cell::new(false) };
}

/// Arm the lens for one decode step, clearing any stale stash a failed prior
/// step may have left behind.
#[cfg(feature = "cuda")]
pub(crate) fn lens_begin() {
    LENS_STASH.with(|stash| stash.borrow_mut().clear());
    LENS_ARMED.set(true);
    LENS_EVER_ARMED.store(true, Ordering::Relaxed);
}

/// Stash one layer's `[vocab, rows]` device logits. NO sync, NO D2H.
#[cfg(feature = "cuda")]
pub(crate) fn lens_push(layer_idx: usize, logits: HiddenStates) {
    if !LENS_ARMED.get() {
        return;
    }
    LENS_STASH.with(|stash| stash.borrow_mut().push((layer_idx, logits)));
}

/// Blocking bf16→f32 D2H of a `[hidden_dim, seq_len]` buffer (syncs the stream).
/// Probe-on paths only — never called when the probe is off.
#[cfg(feature = "cuda")]
pub(crate) fn stream_to_host_f32(
    ctx: &DeviceContext,
    hs: &HiddenStates,
) -> anyhow::Result<Vec<f32>> {
    let host: Vec<half::bf16> = ctx
        .stream
        .clone_dtoh(&hs.data)
        .map_err(|e| anyhow::anyhow!("probe lens D2H failed: {e}"))?;
    ctx.sync()?;
    Ok(host.iter().map(|x| x.to_f32()).collect())
}

/// Drain the stash: D2H each stashed layer's logits, compute lens stats against
/// the finally-emitted tokens, emit one record per (row, layer). D2H errors
/// disable the probe instead of failing the decode step.
#[cfg(feature = "cuda")]
pub(crate) fn lens_flush(ctx: &DeviceContext, positions: &[u64], final_tokens: &[u32]) {
    if lens_layers() == 0 || !LENS_ARMED.replace(false) {
        return;
    }
    let stashed = LENS_STASH.with(|stash| std::mem::take(&mut *stash.borrow_mut()));
    for (layer_idx, logits) in &stashed {
        debug_assert_eq!(logits.seq_len, positions.len());
        let host = match stream_to_host_f32(ctx, logits) {
            Ok(host) => host,
            Err(err) => {
                fail(&err);
                return;
            }
        };
        let vocab = logits.hidden_dim;
        for (row, (&pos, &token)) in positions.iter().zip(final_tokens).enumerate() {
            if let Some(stats) = lens_stats(&host[row * vocab..(row + 1) * vocab], token) {
                emit_lens(pos, *layer_idx, &stats, token);
            }
        }
    }
}

// ── Substage fingerprint (`ARLE_PROBE_STAGES`, diagnostic-only) ─────────────
// Per-row, per-substage checksum along the decode layer stack (attn-norm, DSA
// raw score, compressor output, attn output, hc-post x2, ffn input, MoE out),
// added for the 2026-07-07 concurrent-decode digit-corruption investigation
// (docs/experience/errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md).
// D2H's only the ONE row's slice (not the whole batched buffer), gated by an
// optional decode-position window so a full sweep isn't required — off (no
// `ARLE_PROBE_STAGES`) costs one bool compare per call site.

/// Substage capture on + `pos` inside the optional `ARLE_PROBE_STAGE_POS_{MIN,MAX}`
/// window (both default unbounded — set to bracket the known corruption tick
/// and skip D2H on the surrounding steps).
#[cfg(feature = "cuda")]
pub(crate) fn stage_want(pos: u64) -> bool {
    config().is_some_and(|cfg| cfg.stages) && {
        static WINDOW: OnceLock<(u64, u64)> = OnceLock::new();
        let (lo, hi) = *WINDOW.get_or_init(|| {
            let bound = |var: &str, default: u64| {
                std::env::var(var)
                    .ok()
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(default)
            };
            (
                bound("ARLE_PROBE_STAGE_POS_MIN", 0),
                bound("ARLE_PROBE_STAGE_POS_MAX", u64::MAX),
            )
        });
        pos >= lo && pos <= hi
    }
}

/// Sum (f64 accumulation) + first/last 3 elements + extremum (value, index) of
/// one row's slice — cheap (all from the ALREADY-fetched host `Vec`, no extra
/// D2H), order-sensitive enough to catch a single-ULP-class drift, and the
/// argmin/argmax pinpoint WHICH element an anomaly lives at (a whole-array sum
/// can hide a single wild outlier behind cancellation).
fn emit_stage(stage: &str, layer: usize, row: usize, pos: u64, values: &[f32]) {
    if values.is_empty() {
        return; // never crash inference over an empty (e.g. zero-key-count) slice.
    }
    let sum: f64 = values.iter().map(|&v| f64::from(v)).sum();
    let head: Vec<f32> = values.iter().take(3).copied().collect();
    let tail: Vec<f32> = values.iter().rev().take(3).copied().collect();
    let (argmin, argmax) = values
        .iter()
        .enumerate()
        .fold((0usize, 0usize), |(mn, mx), (i, &v)| {
            (
                if v < values[mn] { i } else { mn },
                if v > values[mx] { i } else { mx },
            )
        });
    // L2 in f64: the scale a cross-runtime delta must be judged against.
    let l2 = values
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt();
    let full = if config().is_some_and(|cfg| cfg.stage_full) {
        format!(",\"values\":{values:?}")
    } else {
        String::new()
    };
    emit(format_args!(
        "{{\"phase\":\"stage\",\"stage\":\"{stage}\",\"layer\":{layer},\"row\":{row},\"pos\":{pos},\"n\":{},\"sum\":{sum:.6},\"l2\":{l2:.6},\"head\":{head:?},\"tail\":{tail:?},\"min\":{:.6},\"argmin\":{argmin},\"max\":{:.6},\"argmax\":{argmax}{full}}}",
        values.len(),
        values.get(argmin).copied().unwrap_or(f32::NAN),
        values.get(argmax).copied().unwrap_or(f32::NAN),
    ));
}

/// D2H + emit one row's bf16 slice (e.g. a `HiddenStates` row-major column).
/// Blocking (syncs the stream) — caller must already have gated on `stage_want`.
#[cfg(feature = "cuda")]
pub(crate) fn stage_bf16(
    ctx: &DeviceContext,
    stage: &str,
    layer: usize,
    row: usize,
    pos: u64,
    view: cudarc::driver::CudaView<'_, half::bf16>,
) {
    let host: Vec<half::bf16> = match ctx.stream.clone_dtoh(&view) {
        Ok(host) => host,
        Err(err) => return fail(&format_args!("stage[{stage}] bf16 D2H failed: {err}")),
    };
    if let Err(err) = ctx.sync() {
        return fail(&format_args!("stage[{stage}] sync failed: {err}"));
    }
    let values: Vec<f32> = host.iter().map(|x| x.to_f32()).collect();
    emit_stage(stage, layer, row, pos, &values);
}

/// D2H + emit one row's f32 slice (e.g. the DSA raw-score `logits_batch`).
#[cfg(feature = "cuda")]
pub(crate) fn stage_f32(
    ctx: &DeviceContext,
    stage: &str,
    layer: usize,
    row: usize,
    pos: u64,
    view: cudarc::driver::CudaView<'_, f32>,
) {
    let host: Vec<f32> = match ctx.stream.clone_dtoh(&view) {
        Ok(host) => host,
        Err(err) => return fail(&format_args!("stage[{stage}] f32 D2H failed: {err}")),
    };
    if let Err(err) = ctx.sync() {
        return fail(&format_args!("stage[{stage}] sync failed: {err}"));
    }
    emit_stage(stage, layer, row, pos, &host);
}

#[cfg(test)]
mod tests {
    use super::{entropy_nll, lens_stats};

    #[test]
    fn uniform_logits_entropy_is_ln_vocab() {
        let vocab = 1024usize;
        let logits = vec![0.25f32; vocab];
        let (entropy, nll) = entropy_nll(&logits, Some(3));
        let ln_v = (vocab as f32).ln();
        assert!((entropy - ln_v).abs() < 1e-4, "H={entropy} ln(V)={ln_v}");
        assert!((nll.unwrap() - ln_v).abs() < 1e-4);
    }

    #[test]
    fn one_hot_entropy_is_zero() {
        let mut logits = vec![-30.0f32; 512];
        logits[7] = 30.0;
        let (entropy, nll) = entropy_nll(&logits, Some(7));
        assert!(entropy.abs() < 1e-4, "H={entropy}");
        assert!(nll.unwrap().abs() < 1e-4);
    }

    #[test]
    fn nll_of_argmax_equals_negated_top1_logprob() {
        let logits: Vec<f32> = (0..97)
            .map(|i| ((i * 37) % 19) as f32 * 0.37 - 3.0)
            .collect();
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap();
        let stats = lens_stats(&logits, argmax).unwrap();
        assert_eq!(stats.top1, argmax);
        assert!((stats.nll + stats.top1_logprob).abs() < 1e-5);
        let (entropy, nll) = entropy_nll(&logits, Some(argmax));
        assert!((stats.entropy - entropy).abs() < 1e-6);
        assert!((stats.nll - nll.unwrap()).abs() < 1e-6);
    }

    #[test]
    fn out_of_vocab_target_yields_no_nll() {
        let logits = vec![0.0f32; 8];
        let (_, nll) = entropy_nll(&logits, Some(99));
        assert!(nll.is_none());
        assert!(lens_stats(&logits, 99).is_none());
    }
}

// ── Component parity injection (`ARLE_PARITY_*`, diagnostic-only) ────────────
// Overwrite one forward buffer with a reference tensor so a single component can
// be compared on an EXACTLY shared input. Comparing accumulated states cannot
// localize a cross-runtime divergence below bf16 granularity (the residual
// stream is bf16 on device, so every layer's own rounding is the floor); an
// injected input removes the accumulation instead of trying to measure past it.
//
// `ARLE_PARITY_INJECT` = path to raw little-endian f32, row-major
// `[seq_len, hidden_dim]`, exactly matching the target buffer.
// `ARLE_PARITY_POINT`  = `attn_in` (post input_layernorm) or `mlp_in`
//                        (post post_attention_layernorm).
// `ARLE_PARITY_LAYER`  = layer index.

#[cfg(feature = "cuda")]
struct ParityInject {
    layer: usize,
    point: String,
    values: Vec<half::bf16>,
}

#[cfg(feature = "cuda")]
fn parity_config() -> Option<&'static ParityInject> {
    static CONFIG: OnceLock<Option<ParityInject>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let path = std::env::var("ARLE_PARITY_INJECT").ok()?;
            let point = std::env::var("ARLE_PARITY_POINT").unwrap_or_else(|_| "mlp_in".to_string());
            let layer = std::env::var("ARLE_PARITY_LAYER")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0);
            let bytes = match std::fs::read(&path) {
                Ok(bytes) => bytes,
                Err(err) => {
                    eprintln!("[parity] read {path} failed: {err}; injection disabled");
                    return None;
                }
            };
            if bytes.len() % 4 != 0 {
                eprintln!("[parity] {path} is {} bytes, not f32-aligned", bytes.len());
                return None;
            }
            let values: Vec<half::bf16> = bytes
                .chunks_exact(4)
                .map(|b| half::bf16::from_f32(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                .collect();
            eprintln!(
                "[parity] injecting {} values at {point} layer {layer} from {path}",
                values.len()
            );
            Some(ParityInject {
                layer,
                point,
                values,
            })
        })
        .as_ref()
}

/// Replace `buf` with the injected reference tensor when this is the configured
/// point and layer. Loud on every fire, on a bounds mismatch, and on a read
/// failure — a silent injection would make a parity run look like a clean forward.
///
/// Prefill is chunked and the split is not predictable from prompt length (130
/// tokens arrive as 128+2, 64 tokens as 48+16), so the file holds the WHOLE
/// prompt and each chunk takes the slice at its own `start_row`. `start_row` is
/// the absolute position, which is the file offset only for a fresh request —
/// the one-request diagnostic this exists for.
#[cfg(feature = "cuda")]
pub(crate) fn parity_inject(
    ctx: &DeviceContext,
    point: &str,
    layer: usize,
    start_row: usize,
    buf: &mut HiddenStates,
) {
    let Some(cfg) = parity_config() else { return };
    if cfg.layer != layer || cfg.point != point {
        return;
    }
    let width = buf.hidden_dim;
    let lo = start_row * width;
    let hi = lo + buf.seq_len * width;
    if hi > cfg.values.len() {
        eprintln!(
            "[parity] {point} layer {layer}: rows {start_row}..{} need {hi} values, file has {} \
             — NOT injecting",
            start_row + buf.seq_len,
            cfg.values.len()
        );
        return;
    }
    match ctx.stream.clone_htod(&cfg.values[lo..hi]) {
        Ok(data) => {
            buf.data = data;
            eprintln!(
                "[parity] injected {point} layer {layer} rows {start_row}..{} ({} values)",
                start_row + buf.seq_len,
                hi - lo
            );
        }
        Err(err) => eprintln!("[parity] {point} layer {layer} H2D failed: {err}"),
    }
}
