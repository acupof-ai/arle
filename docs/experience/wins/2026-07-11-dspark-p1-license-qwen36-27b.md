# DSpark block-draft spec-decode LICENSED — 2.4–3.1× decode tok/s on Qwen3.6-27B

## Context

The OPD rollout is decode-bound and native NextN-MTP capped at ~1.03× (accept-
length-limited). Plan
[2026-07-09-dspark-dflash-spec-decode-qwen36](../../plans/2026-07-09-dspark-dflash-spec-decode-qwen36.md)
adopts the z-lab **DFlash** block drafter (backbone-only; our DSpark Markov +
confidence heads are P3-trained) as an alternate draft source for the existing
Qwen3.6 spec-decode substrate. P0 (contract) + load kill-gate already passed;
this is the P1 throughput license.

- Checkpoint `z-lab/Qwen3.6-27B-DFlash` (HF mirror `hf-mirror.com`, 3.46 GB,
  58 tensors, `fc [5120,25600]` = 5 taps × hidden). block_size 16, 5 layers
  (4 sliding-2048 + 1 full), taps `[1,16,31,46,61]`, reuses base embed/lm_head.
- H20 GPU 1, TP=1, base `Qwen3.6-27B-FP8`, `--spec-type dspark --mtp-draft-model
  <dflash>`. B=1, greedy (argmax verify), thinking-mode on, no shared prefix
  (radix/decode-reuse never fires — raw draft acceptance), `max_tokens=384`,
  decode-graph OFF uniformly. One-flag delta. Binary pod HEAD `93dcf4be`.

## Result — LICENSED (clears 1.15× by a wide margin)

| ctx | no-spec tok/s | dspark tok/s | Δ (×) | accept_rate | accepted/block |
|---|---|---|---|---|---|
| ~50 tok | 45.77 | 109.46 | **2.39×** | 0.199 | 2.99 |
| ~3K tok | 32.10 | 100.94 | **3.14×** | 0.228 | 3.41 |

Per-request dspark 90–152 tok/s (accept-length variance); no-spec steady ~45.8.
Correctness PASS: needle 7391 recalled both repeats, byte-identical across two
greedy runs, coherent — no degradation vs no-spec.

## Watch-list resolved

- **block-16 verify overhead does NOT eat the win.** ~15 spec tokens/chain
  widen every verify row, yet tok/s is 2.4–3.1× *up*. No KILL signal.
- **Accept-rate rises with ctx** (0.199→0.228, ~50→~3K tok; `partial_ctx_chains`
  = all chains) — opposite of MTP's accept-limited-at-long-ctx. The *relative*
  advantage grows because no-spec decode is attention-bound and slows faster
  (45.8→32.1) than dspark's amortized verify.
- **No competing native-MTP arm.** `--spec-type mtp` is DSv4-only on CUDA serve
  (`loaded.rs:1851` bails for Qwen35); `from_qwen35_safetensors` wires only the
  dspark drafter. dspark is the sole CUDA-Qwen3.6 serve spec path — prior
  Qwen3.6 MTP wins were Metal.

## Not a default flip yet — two gates for the OPD-decisive regime

1. **Memory clamp**: 544 MB draft-KV/slot forces slots 256→84 → per-request
   arena 4096 tokens; a 13K prompt doesn't fit one dspark slot. OPD ctx is
   20–45K → must shrink draft-KV footprint or page it before OPD use.
2. **Prefix-restore (P2.5)**: at OPD's ~91% hit rate dspark degrades to plain
   decode after restore (draft ctx gap). This A/B deliberately used no-prefix-hit
   prompts to isolate raw acceptance; the OPD rollout lane needs P2.5 first.

## Rule

A block drafter's verify-row width (K=16) is not a throughput tax to fear —
license it by the measured net tok/s, not the row count. DSpark's advantage
*grows* with context because the baseline it amortizes (attention-bound decode)
degrades faster than the verify does. But an isolated-acceptance A/B (no prefix
hit, short ctx) is a license for that regime only — the OPD rollout regime
(91% prefix hit, 20–45K ctx) is gated separately on P2.5 + the draft-KV memory
footprint.
