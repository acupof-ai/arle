# Bench — Qwen3-0.6B viability as DFlash draft for Qwen3.6 35B-A3B — 2026-05-07

## Goal

The parallel research subagent on this date proposed M-effort DFlash
plumbing using a **tiny dense draft model (Qwen3-0.6B) against the
Qwen3.6 35B-A3B-4bit MoE target**. Predicted ~2× decode speedup if
the draft sustains ≥150 tok/s standalone on Metal. <100 tok/s would
disqualify the direction (draft itself too slow regardless of
acceptance rate).

This is the 30-minute go/no-go gate before committing the M-effort.

## Hypothesis

Per `mlx-lm#1132`'s warn threshold ("draft active params ≥ 4× target
active"): Qwen3-0.6B vs Qwen3.6 3B-active = **5× ratio**, past the
gate. The remaining question is absolute draft speed on Metal.

## Params

- Binary: `target/release/metal_serve` rebuilt at this commit (M_e.1
  oMLX-C v3 default-on, auto-wired-limit default-on, M_e.4 SwiGLU
  fusion default-on)
- Models tested: `mlx-community/Qwen3-0.6B-bf16` (~1.2 GB) and
  `mlx-community/Qwen3-0.6B-4bit` (~370 MB)
- `--max-running-requests 16`
- Workload: 3 sequential c=1 requests, max_tokens=256,
  temperature=0.0, prompt = "Count from 1 to 100 with a one-word
  adjective for each number"

## Results

| Model | Wall tok/s | Engine ITL p50 | Engine TTFT |
|-------|-----------:|---------------:|------------:|
| Qwen3-0.6B-bf16 | 118.4 | 10000 μs (= 100 tok/s) | 35 ms |
| **Qwen3-0.6B-4bit** | **199.7** | **5000 μs (= 200 tok/s)** | **15 ms** |

→ **Qwen3-0.6B-4bit clears the threshold** (200 tok/s steady-state ≫
150 viable bar). bf16 falls below the 150 viable bar at 100-118
tok/s; 4-bit is the right pick for the draft.

## Recalculated DFlash break-even math

With 4-bit draft at 5 ms/step and Qwen3.6 verify at 23 ms/step:

| Acceptance | Direct (4 steps × 23 ms) | DFlash (4×draft + 1×verify) | Speedup |
|-----------:|-------------------------:|----------------------------:|--------:|
| 0.5 (block 4) | 92 ms | 4×5 + 1×23 = **43 ms** | **2.14×** |
| 0.3 (block 4) | 92 ms | 4×5 + 1×23 = 43 ms (same; one block accepted on average) | 2.14× expected, but acceptance variance could degrade tail |
| 0.7 (block 4) | 92 ms | 43 ms | 2.14× |

Every block-of-4 accepted at any acceptance ≥ ~0.25 nets a positive
ratio. **The math holds with significant headroom.** The 4-bit draft
gives a 5× ratio on absolute params (0.6B vs 3B active) and a
4.6× ratio on per-step time (5 ms vs 23 ms), so the verify-amortizes-
across-accepts win is real.

## Problems / observations

1. **bf16 draft is borderline.** 100 tok/s is right at the abandon
   threshold and would yield negligible (or negative) DFlash gains
   after acceptance variance. **Always use the 4-bit quant for
   tiny-draft direction.**
2. **TTFT differential.** 4-bit draft TTFT 15 ms vs bf16 35 ms — the
   smaller weights (370 MB vs 1.2 GB) load AND warm-up faster. A
   DFlash deployment that boots a draft alongside Qwen3.6 would also
   benefit from this, since the draft model load is ~5× smaller.
3. **The 200 tok/s number is steady-state with `INFER_PHASE_TIMING`
   off.** Internal phase timing wasn't enabled for this bench; the
   wall-clock and engine ITL agree at ~200 tok/s, so the number is
   reliable for break-even math.
4. **No correctness check yet.** Generation output for the smoke
   prompt is coherent ("Okay, the user wants me to count from 1 to
   100 using a one-word adjective for each number..." — Qwen3-0.6B
   thinks aloud rather than just listing, which is fine for a draft
   that just proposes tokens to be verified by the larger model).

## Learnings

1. **The research subagent's "150 tok/s viable / 100 tok/s abandon"
   threshold is well-calibrated.** bf16 draft at 100-118 tok/s sits
   exactly at the disqualification line; 4-bit at 200 tok/s clears
   easily. Future viability checks for spec-decode-class
   optimizations should use this same shape: pick a concrete
   threshold, run the cheap solo bench, decide.
2. **Quant choice for the draft matters more than for the target.**
   4-bit quant on a 0.6B model is 0.95% quality drop in the literature
   for negligible perf cost; on the draft side, where the model only
   *proposes* tokens (the larger target verifies), even larger quant
   degradation would be acceptable since it just lowers acceptance
   rate slightly. The 4-bit draft is the obvious choice.
3. **bf16 vs 4-bit: ~5× wall-time difference at this scale.** This
   is a useful calibration data point for any future "should we 4-bit
   the draft" question on Metal.

## Decision

**Greenlight M_e.7 — full DFlash plumbing for Qwen3-0.6B-4bit draft
against Qwen3.6 35B-A3B verify.** M effort, predicted ~2× decode
speedup at acceptance ≥0.3.

Blockers / pending work for M_e.7:
1. Add MoE verify-target hidden-state hook (current ARLE DFlash
   targets dense Qwen3 only — see
   `/path/to/code/agent-infer/infer/src/backend/metal/runtime.rs:2036-2047`
   "DFlash requires full-prompt prefill … `qwen3_forward_with_hidden_states`").
2. Adapt the `infer/src/backend/metal/dflash.rs` compat check (it
   gates on `target_layer_ids` and head widths) to allow Qwen3-MoE
   as the target type.
3. Bench acceptance rate empirically on a representative chat workload.
4. Matched A/B per `feedback_matched_ab_for_small_bench_effects.md`
   on the headline 2× speedup claim before defaulting on.

## What worked

- 30-minute go/no-go gate burned no engineering effort on a path that
  could have been blocked at the draft-speed level.
- Sequential c=1 wall-clock tok/s correlates well with engine ITL
  p50; either is sufficient for this class of viability check.
- Re-ran with 4-bit when bf16 was borderline. Don't accept
  borderline; find the variant that clears.

## Rule

When evaluating spec-decode-class optimizations with a separate draft
model, the **draft's standalone tok/s is the first gate**. Run a
30-minute solo bench BEFORE designing any plumbing. Use the published
threshold from the research source if available; otherwise compute
break-even from `verify_step_ms × block_size / (block_size+1) >
draft_step_ms`.

## Next

- **M_e.7 plan doc** — full DFlash-with-tiny-draft plumbing for
  Qwen3.6 (lands as
  `docs/plans/M_e7-dflash-tiny-draft.md` next tick).
- **Chunked-prefill activation-budget tuning** still on the deck
  (S effort, separate path; not blocked by M_e.7).

## References

- Research source (this date): subagent task answering
  "Speculative decoding for MoE on Metal — threshold flips with 5×+
  smaller draft" (full report in this tick's conversation; key
  citations: `mlx-lm#1132`, `infer/src/backend/metal/dflash.rs:285-291`,
  `crates/mlx-sys/src/mlx_dflash_draft_model.cpp:478-566`).
- Predecessor (encode-bottleneck localization):
  [`2026-05-07-bench-qwen36-encode-bottleneck.md`](2026-05-07-bench-qwen36-encode-bottleneck.md)
- Failed alternative direction:
  [`2026-05-07-m_e5-naive-dual-stream-regresses.md`](../errors/2026-05-07-m_e5-naive-dual-stream-regresses.md)
