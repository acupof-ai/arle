# DSv4 garbage output root-caused: incomplete GPU-native forward, NOT a FlashMLA / KV regression

## Context

Pod smoke (`dsv4_toolchain.sh smoke`, 8×H20, TP=8, DSv4-Flash, fp8 KV)
with `ARLE_DSV4_FLASHMLA_PREFILL=1 ARLE_DSV4_FLASHMLA_DECODE=1` returned a
200 but produced garbage: prompt "Compute 137 + 269. Answer with the
number only." → `"4062 0.0000 0.0000 0.0000 ..."` (expected `406`).

Initial suspicion fell on the freshly-landed FlashMLA decode + D-4 pack
hooks (this session's work).

## Root Cause

**Two independent issues, only one of which was mine:**

### 1. FlashMLA prefill panic (my regression — FIXED)

`8ebe3ff5` (D-4 FlashMLA decode plumbing) changed the prefill KV-pack in
`finish_attention_gpu` to source the bf16 SW window via a hard
`cache.as_deref_mut().expect("FlashMLA prefill requires cache")`. But
serving prefill runs the **stateless batched path**
(`compute_gpu_logits_after_prefill` → `compute_top_level_logits`,
`cache=None`) by design — the incremental cached path is decode-only
(regresses long-prefill TTFT >2× per the V2.4 entry). So FlashMLA prefill
panicked all 8 scheduler workers the moment it was exercised end-to-end.

V2.4 (`86d27946`) ran FlashMLA prefill cache-less fine because it sourced
the window via a binding that fell back to a freshly-zeroed scratch when
`cache=None`. Fix `439e66cb` restores exactly that (mirrors the legacy
hybrid path's `window_scratch_local`). At `start_pos=0` the SW ring is
empty, so a zeroed scratch is the correct prior.

### 2. Garbage output (pre-existing, NOT mine — the real blocker)

**Control experiment (CLAUDE.md §0 — isolate confounders):** ran the
same smoke with FlashMLA fully OFF (`ARLE_DSV4_FLASHMLA_PREFILL=0
ARLE_DSV4_FLASHMLA_DECODE=0`). Result: **identical garbage**
(`"4062 0.0000 0.0000..."`), no panic. → FlashMLA is innocent.

The garbage is the documented, expected state of the **incomplete
GPU-native DSv4 forward**:

- Serving tries `compute_reference_logits_after_*` first; it returns
  `None` because the reference model is gated off
  (`ARLE_DSV4_INFER_REAL_REFERENCE` unset → `self.reference = None`).
- Falls through to `compute_gpu_logits_after_decode`, whose own comment
  states: *"Phase 2A.1 uses the loaded top-level tensors for non-zero
  logits when available. **Real contextual attention and shared-expert
  compute land in later, separately gated tranches.**"*
- `dsv4_gpu_full_layer_limit()` defaults to **0** → the GPU full-layer
  attention path runs essentially no real contextual attention.

This matches `wins/2026-05-27-dsv4-native-deepep-pod-e2e.md` exactly:
both `=deepep` and `=native-deepep` backends produced the identical
`4262 / 0.0000 0.0000...` garbage shape there too, days before any of
this session's work. The "correct-ish first token (≈406) then 0.0000
cascade" is the signature of a forward that gets the embedding/top-level
projection roughly right but has no working contextual attention.

## Fix

- FlashMLA prefill panic: `439e66cb` (scratch-window fallback). Build
  clean, pod-validated TP=8 end-to-end, no panic.
- Garbage output: **not a code bug to patch** — it is the in-progress
  GPU-native forward buildout. Correct output requires either:
  (a) `ARLE_DSV4_INFER_REAL_REFERENCE=1` (+ `ARLE_DSV4_LOAD_LAYER_WEIGHTS=1`)
  to drive the CPU correctness reference (`model/deepseek/reference.rs`,
  "CPU smoke path for correctness work"), or
  (b) completing the phased GPU-native contextual attention + shared-expert
  tranches.

## Rule

**DSv4 perf benches run on garbage-shaped output by design — the
GPU-native forward is a phased buildout (Phase 2A.1), and correctness is
gated behind `ARLE_DSV4_INFER_REAL_REFERENCE` / unfinished tranches.**
Before treating DSv4 garbage as a regression: (1) run the control with
the new path OFF — if prod is *also* garbage, it is the pre-existing
incomplete forward, not your diff (this is the
`2026-05-27-dsv4-native-deepep-pod-e2e` config-suspect lesson); (2) a
"correct first token then 0.0000 cascade" specifically indicates absent
contextual attention, not a KV-format / FlashMLA bug. Perf (TTFT / TPOT /
throughput) is still measurable on the garbage-shaped output because the
forward runs the same FLOPs; correctness is a separate, larger axis.
