# SGLang parity — systematic analysis (infer vs sglang 0.5.10, L4)

## Scope

Rollup of 2026-04-17's measurements across four runs:
- `2026-04-17-sglang-p99-parity-qwen3-4b.md` — Qwen3-4B guidellm sweep
- `2026-04-17-sglang-p99-parity-qwen35-4b.md` — Qwen3.5-4B guidellm sweep
- `2026-04-17-ttft-scaling-infer-vs-sglang-l4.md` — TTFT scan at 128..4096 prompt tokens

The question the user asked: *p99 对齐 sglang，然后挖方案超越 — 我们是单独适配的模型.*
Translation: verify p99 parity, then systematically find surpass paths using
our single-model adaptation advantage.

## The numbers in one table (matched-rate cells; 4096+256 workload)

| axis | Qwen3-4B | Qwen3.5-4B |
|------|----------|------------|
| **ITL p99 (infer vs sglang)** | +3% infer slower | **-3% to -8% infer FASTER** |
| **TTFT p99 (infer vs sglang)** | +50% infer slower (+420ms) | +60% infer slower (+510ms) |
| **Peak throughput (infer vs sglang)** | -18% infer | **-47% infer** |
| **Fixed overhead @128-tok prompt (infer vs sglang)** | -37% infer faster (63 vs 100ms) | n/a (not probed) |
| **Per-token prefill cost @4096 tok** | +24% infer slower | +50%+ infer slower |

## Three surprises worth writing down

### 1. infer already beats sglang on Qwen3.5 ITL (3-8%)

This is the first measurement where our engine wins a p99 metric outright.
Root cause: we've invested in Qwen3.5-specific decode kernels (gated delta
net, interleaved SSM+attn, per-layer state snapshots for prefix reuse);
sglang's `qwen3_5.py` runs through a generic decoder framework.

**Narrative lever:** "On the hybrid model where architecture specialisation
matters most, agent-infer ITL p99 is 3-8% lower than sglang."

### 2. Short-prompt workloads already win (10-37%)

From the TTFT scan (`2026-04-17-ttft-scaling-infer-vs-sglang-l4.md`):
the crossover between infer and sglang is around n≈1500 tokens. Below that,
infer is uniformly faster because our Rust HTTP→scheduler→kernel-launch
path has ~37ms lower fixed overhead than sglang's Python-dispatch path.

**Narrative lever:** "Chat and agent workloads (<1024-token prompts) see
infer at 10-37% lower TTFT than sglang across every measurement."

### 3. TTFT gap is NOT chunking

Verified: setting `--decode-prefill-cap 4096` (disables sub-chunking when
decode is active) moved single-request TTFT from 820ms to 800ms. The
~20ms change is within noise. So the `decode_active_prefill_cap: 512`
policy is not the bottleneck.

The real TTFT gap is structural:
- ~120ms pure single-request prefill (sglang's FlashInfer wrapper is tighter)
- ~300ms step-interleaving overhead when prefill and decode coexist under
  load (exactly what `docs/plans/scheduler-gpu-cpu-overlap.md` targets)

## Where we are vs where we could be

### Dense model (Qwen3-4B)

| axis | status | next unlock |
|------|--------|-------------|
| ITL p99 | **parity (±3%)** | — saturated; L4 HBM-bound |
| short-prompt TTFT | **already winning** | — publish as a targeted benchmark |
| long-prompt TTFT (single req) | -100ms | FlashInfer `plan()` cache across calls |
| steady-state TTFT p99 (0.13-0.33 r/s) | -420ms | **GPU/CPU overlap (plan exists)** |
| peak throughput | -18% | Derived from TTFT gap; same fix |

### Hybrid model (Qwen3.5-4B)

| axis | status | next unlock |
|------|--------|-------------|
| ITL p99 | **WINNING (3-8%)** | Extend decode fusion to cover attn output gate |
| TTFT p99 | -510ms (+60%) | **Full-forward CUDA Graph for prefill** |
| peak throughput | **-47%** | Same (per-layer launch overhead compounds) |

## Prioritised surpass plan

Assume 1 engineer-week of focus, ROI-ranked:

### P0 — GPU/CPU overlap for mixed prefill+decode (~300ms of Qwen3-4B p99 gap)

- Already documented: `docs/plans/scheduler-gpu-cpu-overlap.md`
- Expected: infer Qwen3-4B TTFT p99 drops from 1250 → ~950ms, within 15% of sglang
- Ripple effect: peak throughput +10-15% (from ~98 → ~110 tok/s)

### P1 — Full-forward CUDA Graph prefill for Qwen3.5 (~46% of Qwen3.5 throughput gap)

- Net new work. sglang's `qwen3_5.py` single-graphs all 32 layers.
- Ours alternates DeltaNet chunk calls with FlashInfer prefill.
- Fix: capture the full 32-layer forward as one graph per (B, seq_len) shape
  including the handoff state buffers. Needs contiguous state allocation.
- Expected: Qwen3.5 peak throughput from 91 → ~120+ tok/s (close to sglang 134).
- **This is the single most leveraged item for the "surpass because we specialised"
  argument — it directly targets where specialisation should pay off.**

### P2 — FlashInfer `plan()` cache (single-request TTFT)

- 100ms off single-request TTFT on Qwen3-4B 4096-prompt.
- sglang caches the wrapper state; we rebuild per-call.
- Audit `crates/infer-cuda-kernels/csrc/attention/flashinfer_prefill*.cu` bindings.

### P3 — Extend decode fusion to Qwen3.5 attn output gate path

- Extends the existing ITL lead (3-8%) to a wider advantage.
- Work after P1 — same hot path.

### P4 — Publish "short-prompt advantage" bench

- No engineering — docs-only.
- Positions infer as the agent/chat-first engine.
- Template: repeat the 128/512/1024-token TTFT scan under realistic load
  (not just single-request). Include the crossover plot.

## Rules

1. **p99 ITL on Qwen3.5-4B is already below sglang's.** Any future work on
   this specific metric for this specific model is defending a lead, not
   chasing. Shift attention to TTFT.

2. **Short-prompt workloads are structurally ours.** The Rust HTTP path has
   a ~37ms fixed advantage that no kernel optimisation by sglang can close
   without them also rewriting in Rust.

3. **The per-layer CUDA Graph strategy is the lever we haven't pulled.**
   Every measurable throughput gap on both models traces back to sglang
   batching layer launches into graphs more aggressively than we do.
   Full-forward capture on prefill is the unlock for both.

4. **"Specialisation beats generalisation" is proved on ITL but not yet on
   TTFT for hybrid models.** The decode path is where we've done the work
   and it shows; the prefill path is where we haven't and it shows.

## Artefacts

All wins files from 2026-04-17 — cited individually above.
Raw bench outputs under `bench-output/2026-04-17-*`.
