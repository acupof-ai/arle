# OPD `fused_distill` default ran the lm_head on the HOST — 53× slower than dense, GPU idle

## Context

The 2026-06-18 change made `fused_linear_distill_loss` the **default** windowed-KL
path (`GkdLossConfig::fused_distill = true`), to avoid materializing the
`[window, vocab]` student-logits tensor. Its wins entry
([`2026-06-18-opd-fused-distill-default-pending-remote.md`](../wins/2026-06-18-opd-fused-distill-default-pending-remote.md))
shipped the default flip with the perf A/B explicitly **pending-remote** — the
"default fused vs `--no-fused-distill`" same-binary comparison was never run on
GPU. Every capability run since (math500 multiseed, agentic-OPD) quietly passed
`--no-fused-distill`, so the *good* numbers were all on the dense path; the fused
default was shipped but never actually exercised by a real run.

While verifying #19 (device-resident `sum_all`) on H20 with Qwen3.6-27B-FP8, the
27B OPD step measured **~205 s/step** — no better than the original "4.5 min,
CPU 99.9%, GPU idle" symptom that motivated #19. That forced the A/B that had
been owed since 2026-06-18.

## Root Cause

`fused_linear_distill_loss` computes the lm_head projection (hidden → full
vocab=248320 logits) **and** the KD loss in a host triple-loop on a single CPU
core. The op-level decode (`ARLE_OPD_BACKWARD_PROFILE=1` +
`ARLE_OPD_STEP_TRACE=1`, GPU-util sampler) pinned it exactly:

| window event | fused (default) | dense (`--no-fused-distill`) |
|---|---|---|
| `fused_linear_distill_start` → `kl_loss_done` | **201.7 s** | 3 ms |
| `windowed_backward` total | 205.6 s | **3.8 s** |
| **full step** | **205.6 s** | **3.86 s** (~53×) |
| GPU-0 util during window-forward | **0 %** (host-bound) | 98 % (device-bound) |
| `opd_backward_profile total_seconds` | 2.1 s | 2.1 s (same; incl. the #19 `sum_all`) |
| `loss_accum` | 5.1535 | 5.1510 (match, ~5e-4) |

The actual GPU backward — including #19's now-device-resident `sum_all` — is only
**2.1 s**. The entire 200 s is the host lm_head loop. The `[window, vocab]` f32
tensor the fused path "saves" is **31.8 MB** (window=32) — trivially resident on
a 96 GB H20; the default window never gets near a size where avoiding it matters.
So `fused`-as-default traded ~free device compute for catastrophic host compute
to save memory that windowing already saves.

**Two attributions corrected at once (§0 case-as-fact):**
1. The fused default's "memory win" was never worth its compute cost — the
   pending A/B, once run, refutes the 2026-06-18 default flip.
2. #19's `sum_all` wins-doc had mis-attributed the 27B "4.5 min/step, GPU idle"
   to `sum_all`. The decode shows `sum_all` lives in the fast 2.1 s backward; the
   dominant cost was always `fused_linear_distill`. `sum_all` device-residency is
   a correct optimization but was never the bottleneck in this config.

## Fix

- **Default flipped to dense.** `GkdLossConfig::fused_distill` defaults `false`
  (`crates/train/src/opd.rs`); CLI gains `--fused-distill` (opt-in, off by
  default) and the production mapping becomes
  `fused_distill = args.fused_distill && !args.no_fused_distill`
  (`crates/cli/src/{args,train_cli}.rs`). `--no-fused-distill` is kept as a
  now-redundant no-op so existing invocations / `examples/opd/*.sh` still run.
- The fused path is retained as opt-in for any future window too large to
  materialize `[window, vocab]` — measured equivalent to dense to ~5e-4, so it
  stays a valid (if rarely worth it) lever.

## Rule

A default flip that ships its perf A/B as "pending-remote" is **not landed** —
it is a hypothesis. Run the same-binary A/B before trusting the default; here the
owed A/B sat for 5 days while every real run bypassed the new default, so the
regression was invisible until a fresh task happened to exercise it. And when a
step is slow with "GPU idle / one core 100%", **decode the op trace + sample GPU
util before crediting any single op** — the host-bound op (`fused_linear_distill`)
was a different op than the one being optimized (`sum_all`). See
[[reference_opd_fused_distill_host_loop_pathological]] and
[[feedback_bench_delta_vs_baseline_not_raw]].
