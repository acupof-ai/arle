//! Whole-drafter-step batch-invariance gate (dev-only, synthetic weights).
//!
//! The kernel-level `dspark_draft_attn_parity` gate already proves the single
//! and batched ring-varlen attention kernels agree. This gate covers everything
//! around that kernel in one real drafter step — every draft layer, the
//! per-slot KV rings / positions / window tables, the batched GEMM row layout
//! and the draft sampler — by running the SAME sequence and anchor through
//! `dspark_draft_block` (one slot) and `dspark_draft_blocks` (that slot at
//! batch index 0 and 3, with seven other LIVE slots holding different
//! contexts), then comparing the target slot's draft logits and greedy tokens.
//!
//! Invariance is independent of weight values, so the head is filled from one
//! deterministic RNG at the served geometry; no trunk or checkpoint is needed
//! (the draft fns read only `model.ctx` and `model.embed_tokens`; the head
//! carries its own layers/norm/rope and the tied lm output projection).

use super::Qwen35Model;
use super::dspark::{
    DsparkScratch, Qwen35DsparkHead, Qwen35DsparkSlotState, gate_cap, gate_fill_rings,
    gate_slot_logits_to_host, gate_swap_layer0_rings, synth_drafter_model, synth_dspark_head,
};
use anyhow::Result;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix};
use half::bf16;
use infer_plan::SamplingParams;
use qwen35_spec::DsparkConfig;

const SECONDARY: usize = 3;

/// The vocab only fixes the embedding/lm-head GEMM width and the argmax space;
/// batch invariance holds for any vocab. The real Qwen3.8-27B-DSpark vocab is
/// ~152k, which would make the synthetic embed and per-block logits ~6 GB and
/// ~1.7 GB respectively; cap to a small width for the synthetic gate. Geometry
/// (q/kv heads, head_dim, block, window) still comes from config.json.
const GATE_VOCAB: usize = 2048;

/// rel-L2 floor between the same math at GEMM rows `block` vs `8*block`. The
/// two paths run identical kernels; this covers batched-GEMM reduction
/// reordering only, not algorithmic drift.
const LOGITS_REL_L2_FLOOR: f64 = 2e-2;
/// Per-element bf16 abs-error cap on the draft logits.
const LOGITS_ABS_FLOOR: f64 = 5e-2;

/// Served DSpark geometry parsed from config.json.
#[derive(Clone, Copy)]
pub(crate) struct Geometry {
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block: usize,
    sliding_window: Option<usize>,
    max_seq_len: usize,
}

impl Geometry {
    fn from_config(path: &std::path::Path) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)
            .map_err(|e| anyhow::anyhow!("read dspark config {}: {e}", path.display()))?;
        let d = v
            .get("dflash_config")
            .filter(|d| d.is_object())
            .unwrap_or(&v);
        let f = |name: &str| d.get(name).or_else(|| v.get(name));
        let num = |name: &str| -> Result<usize> {
            f(name)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| anyhow::anyhow!("missing {name} in {}", path.display()))
        };
        Ok(Self {
            q_heads: num("num_attention_heads")?,
            kv_heads: num("num_key_value_heads")?,
            head_dim: num("head_dim")?,
            block: num("block_size")?,
            sliding_window: f("sliding_window")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize),
            max_seq_len: num("max_position_embeddings").unwrap_or(32768),
        })
    }
}

fn hidden_of(g: &Geometry) -> usize {
    g.q_heads * g.head_dim * 4
}

fn dspark_cfg(g: &Geometry) -> DsparkConfig {
    let hidden = hidden_of(g);
    DsparkConfig {
        hidden_size: hidden,
        intermediate_size: hidden * 2,
        num_hidden_layers: 1,
        num_attention_heads: g.q_heads,
        num_key_value_heads: g.kv_heads,
        head_dim: g.head_dim,
        rms_norm_eps: 1e-6,
        rope_theta: 1e7,
        rope_scaling: None,
        sliding_window: g.sliding_window,
        layer_types: vec![qwen35_spec::DsparkLayerType::Full],
        block_size: g.block,
        mask_token_id: 0,
        target_layer_ids: vec![-1],
        next_token_heads: true,
    }
}

#[derive(Clone)]
struct SlotCase {
    anchor: u32,
    start: usize,
    ctx_len: usize,
    seed: u64,
}

/// Eight distinct per-slot contexts. The target is index 0 (also re-run at
/// index 3). Distractor starts differ and include a block-boundary case and a
/// ring-wrap case so a cross-slot stride bug cannot hide behind equal lengths.
fn slot_cases(block: usize, cap: usize) -> Vec<SlotCase> {
    let target = SlotCase {
        anchor: 101,
        start: 200,
        ctx_len: 200,
        seed: 0xA1,
    };
    let mk = |anchor: u32, start: usize, seed: u64| SlotCase {
        anchor,
        start,
        ctx_len: start,
        seed,
    };
    // Place one window so its last draft rows reach the ring end: the kernel
    // walks kv up to start+block, bounded to <= cap by draft_attention_windows.
    let wrap_start = cap.saturating_sub(block + 3);
    vec![
        target,
        mk(202, 97, 0xB2), // short, odd (block-boundary straddle)
        mk(303, 320, 0xC3),
        mk(404, 512 + 5, 0xD4), // start not block-aligned
        mk(505, 640, 0xE5),
        mk(606, 1024, 0xF6),
        mk(707, wrap_start, 0x07), // window reaches ring end -> modulus wrap
        mk(808, 401, 0x18),
    ]
}

fn greedy() -> SamplingParams {
    // SamplingParams::default is greedy (temperature 0).
    SamplingParams::default()
}

fn rel_l2(got: &[bf16], want: &[bf16]) -> (f64, usize) {
    let (mut d2, mut r2, mut viol) = (0.0, 0.0, 0);
    for (a, b) in got.iter().zip(want) {
        let (x, y) = (f64::from(*a), f64::from(*b));
        let d = x - y;
        d2 += d * d;
        r2 += y * y;
        if d.abs() > LOGITS_ABS_FLOOR {
            viol += 1;
        }
    }
    ((d2 / r2.max(1e-12)).sqrt(), viol)
}

fn argmax_row(logits: &[bf16], vocab: usize, row: usize) -> u32 {
    let base = row * vocab;
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in logits[base..base + vocab].iter().enumerate() {
        let v = f32::from(x);
        if v > bv {
            bv = v;
            best = i as u32;
        }
    }
    best
}

/// One target slot alone (GEMM rows = block).
fn run_single(
    model: &Qwen35Model,
    head: &Qwen35DsparkHead,
    scratch: &mut DsparkScratch,
    target: &SlotCase,
    vocab: usize,
) -> Result<Vec<bf16>> {
    let ctx = &model.ctx;
    let mut df = Qwen35DsparkSlotState::new(ctx, head)?;
    df.ctx_base = 0;
    df.ctx_end = target.start;
    gate_fill_rings(ctx, &mut df, head, target.seed, target.ctx_len)?;
    model.dspark_draft_block(
        head,
        &mut df,
        scratch,
        target.anchor,
        target.start,
        &greedy(),
    )?;
    gate_slot_logits_to_host(ctx, &mut df, head, vocab)
}

#[derive(Clone, Copy, Default)]
struct Sabotage {
    /// Swap the two slots' (start position, ctx_end) for the batched launch.
    pos: Option<(usize, usize)>,
    /// Exchange the two slots' layer-0 K/V ring DeviceVecs before the launch.
    kv: Option<(usize, usize)>,
}

/// All slots in one batched step; return the `read_idx` slot's draft logits.
fn run_batched(
    model: &Qwen35Model,
    head: &Qwen35DsparkHead,
    scratch: &mut DsparkScratch,
    cases: &[SlotCase],
    read_idx: usize,
    vocab: usize,
    sab: Sabotage,
) -> Result<Vec<bf16>> {
    let ctx = &model.ctx;
    let mut slots: Vec<Qwen35DsparkSlotState> = (0..cases.len())
        .map(|_| Qwen35DsparkSlotState::new(ctx, head))
        .collect::<Result<Vec<_>>>()?;
    for (s, c) in cases.iter().enumerate() {
        slots[s].ctx_base = 0;
        slots[s].ctx_end = c.start;
        gate_fill_rings(ctx, &mut slots[s], head, c.seed, c.ctx_len)?;
    }

    let mut starts: Vec<usize> = cases.iter().map(|c| c.start).collect();
    if let Some((a, b)) = sab.pos {
        starts.swap(a, b);
        slots[a].ctx_end = starts[a];
        slots[b].ctx_end = starts[b];
    }
    if let Some((a, b)) = sab.kv {
        // Exchange the two slots' layer-0 K/V ring DeviceVecs before the
        // batched launch: the kv_bases table then points each slot at the
        // other's context.
        let (lo, hi) = (a.min(b), a.max(b));
        let (left, right) = slots.split_at_mut(hi);
        gate_swap_layer0_rings(&mut left[lo], &mut right[0]);
    }

    let anchors: Vec<u32> = cases.iter().map(|c| c.anchor).collect();
    let params = greedy();
    let param_refs: Vec<&SamplingParams> = (0..cases.len()).map(|_| &params).collect();
    let mut refs: Vec<&mut Qwen35DsparkSlotState> = slots.iter_mut().collect();
    model.dspark_draft_blocks(head, &mut refs, scratch, &anchors, &starts, &param_refs)?;
    gate_slot_logits_to_host(ctx, &mut slots[read_idx], head, vocab)
}

fn compare(
    label: &str,
    single: &[bf16],
    batched: &[bf16],
    vocab: usize,
    block: usize,
    lines: &mut Vec<String>,
) -> bool {
    let (rel, viol) = rel_l2(single, batched);
    let mut tokens_same = true;
    for r in 0..block {
        if argmax_row(single, vocab, r) != argmax_row(batched, vocab, r) {
            tokens_same = false;
            lines.push(format!(
                "  row {r}: single={} batched={}",
                argmax_row(single, vocab, r),
                argmax_row(batched, vocab, r)
            ));
        }
    }
    let ok = rel < LOGITS_REL_L2_FLOOR && viol == 0 && tokens_same;
    lines.push(format!(
        "{label}: rel_l2={rel:.3e} abs_violations={viol} tokens_equal={tokens_same} => {}",
        if ok { "PASS" } else { "FAIL" }
    ));
    ok
}

fn expect_moved(label: &str, single: &[bf16], moved: &[bf16], lines: &mut Vec<String>) -> bool {
    let (rel, _) = rel_l2(single, moved);
    let fired = rel >= LOGITS_REL_L2_FLOOR;
    lines.push(format!(
        "{label}: rel_l2={rel:.3e} (need >= {LOGITS_REL_L2_FLOOR}) => {}",
        if fired { "FIRED" } else { "DID NOT FIRE" }
    ));
    fired
}

/// Whole-drafter-step batch-invariance gate result (one line per family).
pub struct GateReport {
    pub lines: Vec<String>,
    pub ok: bool,
}

pub(crate) fn run(config_path: Option<&std::path::Path>, negative: bool) -> Result<GateReport> {
    // Default geometry names Qwen3.8-27B-DSpark: 40 q / 8 kv heads, head_dim
    // 128, drafter block 7, no sliding window.
    let g = match config_path {
        Some(p) => Geometry::from_config(p)?,
        None => Geometry {
            q_heads: 40,
            kv_heads: 8,
            head_dim: 128,
            block: 7,
            sliding_window: None,
            max_seq_len: 32768,
        },
    };
    let cfg = dspark_cfg(&g);
    let (block, vocab, hidden) = (g.block, GATE_VOCAB, hidden_of(&g));

    let ctx = DeviceContext::new()?;
    let head = synth_dspark_head(&ctx, cfg, g.max_seq_len, g.max_seq_len)?;
    let embed_host: Vec<bf16> = (0..vocab * hidden)
        .map(|i| bf16::from_f32(((i % 17) as f32 - 8.0) * 0.01))
        .collect();
    let embed = DeviceMatrix::from_host(&ctx, &embed_host, vocab, hidden)?;
    let model = synth_drafter_model(ctx, embed, vocab, hidden, 1e-6);
    let cap = gate_cap(&head);
    let cases = slot_cases(block, cap);

    let mut scratch = DsparkScratch::default();
    let mut lines = Vec::new();
    let mut ok = true;

    let single = run_single(&model, &head, &mut scratch, &cases[0], vocab)?;
    let b0 = run_batched(
        &model,
        &head,
        &mut scratch,
        &cases,
        0,
        vocab,
        Sabotage::default(),
    )?;
    ok &= compare(
        "target@b0 single-vs-batched",
        &single,
        &b0,
        vocab,
        block,
        &mut lines,
    );

    // Same target case placed at batch index 3.
    let mut reordered = cases.clone();
    reordered.swap(0, SECONDARY);
    let b3 = run_batched(
        &model,
        &head,
        &mut scratch,
        &reordered,
        SECONDARY,
        vocab,
        Sabotage::default(),
    )?;
    ok &= compare(
        "target@b3 single-vs-batched",
        &single,
        &b3,
        vocab,
        block,
        &mut lines,
    );

    if negative {
        let pos = run_batched(
            &model,
            &head,
            &mut scratch,
            &cases,
            0,
            vocab,
            Sabotage {
                pos: Some((0, 1)),
                kv: None,
            },
        )?;
        ok &= expect_moved("neg swap-pos target@b0", &single, &pos, &mut lines);
        let kv = run_batched(
            &model,
            &head,
            &mut scratch,
            &cases,
            0,
            vocab,
            Sabotage {
                pos: None,
                kv: Some((0, 1)),
            },
        )?;
        ok &= expect_moved("neg swap-kvbase target@b0", &single, &kv, &mut lines);
    }

    lines.push(if ok {
        "ALL PASS".to_string()
    } else {
        "FAIL".to_string()
    });
    Ok(GateReport { lines, ok })
}
