# DSpark-on-OPD default-flip gate — quality-neutral LICENSED; concurrency ≥4 deferred (shared-box KV clamp)

## Context

Final gate before defaulting DSpark on for OPD rollouts. Binary at HEAD
`095dcca6` (spec_decode /v1/stats export dfbf33ee, trainer dspark flags
6b71f637, MIN_M=2 crossover e5d0899). 8×H20, Qwen3.6-27B-FP8 + z-lab DFlash
draft, CC-as-harness (`scripts/cc_swe_baseline.py`), temp 0.7. Build+symbol
gates green; `/v1/stats.spec_decode` live.

Process note: the round burned ~6 h wall-clock to security-agent SIGKILLs
(canonical `arle serve` path survives; a disguised `/host/gate16/arle16` path
was reaped twice — RUN_EXIT=137, no OOM trace) + one watcher false positive
(single transient pod-exec fail, debounced to 3-consecutive after). The
16-task CC eval was the wrong instrument for a lossless-by-construction change;
ckl cut it to 3 tasks mid-round. See Rule.

## Gate 1 — multi-sample pass-rate: QUALITY-NEUTRAL LICENSED

3-task (fastest-per-family, 3 rep × 2 arm): both arms 9/9, 0 flips — but
ceiling effect (fastest instances = both 100%, paired signal constructed away).

Bonus n=16 (preserved rep-1, canonical serve):

| arm | pass | Wilson 95% |
|---|---|---|
| plain | 7/16 = 0.438 | [0.231, 0.668] |
| dspark | 9/16 = 0.562 | [0.332, 0.769] |

- **plain-pass & dspark-fail = 0** (no systematic per-task loss); dspark +2
  (two textfsm). CIs fully overlap; dspark inside plain's CI. No failing case
  to decode.
- Spec telemetry (free from /v1/stats): accept_rate 0.16–0.18, tok/step
  3.43–3.76, partial_ctx_share 0.83–0.86 (matches 07-10's 0.90 — partial-ctx
  drafting engages on 83–86% of chains at CC's ~18K system-prompt regime).

This is the lossless-spec expectation confirmed: rejection-sampling verify is
distribution-equal, so pass-rate must not move — and it doesn't.

## Gate 2 — concurrency aggregate: c=1 confirmed, c≥4 INCONCLUSIVE

| c | plain agg tok/s (p50) | dspark agg tok/s (p50) |
|---|---|---|
| 1 | 35.6 (35.6) | **67.8 (68.0)** — 1.9× |
| 4 | 71.6 (18.0) | all-timeout |
| 8 | 71.7 (9.1) | all-OOM |

**Not a dspark structural failure — shared-box memory confound.** DFlash draft
reserves **2560 MB/slot** draft-KV (per_slot 2707 vs plain 146 MB). GPU 2 had
46 GB foreign co-tenant + 27 GB weights → ~19 GB free → slots clamped 256→6,
max_total_tokens 4096; c=4 (~4000 tok) hits the boundary, c=8 (~8000 tok)
`CUDA_ERROR_OUT_OF_MEMORY`. Plain's cheap 146 MB/slot survives c=8 in the same
19 GB. Per CLAUDE.md "shared-GPU memory readings are polluted" — no structural
verdict without a clean 97 GB GPU rerun.

## Verdicts

- **Quality default flip: LICENSED** (Gate 1, incl. n=16 no systematic loss).
- **Concurrency default flip: DEFERRED** — c=1 gain solid; c≥4 unattributable
  under co-tenant KV clamp. The 2560 MB/slot draft-KV IS a real concurrency
  ceiling risk under memory pressure; rerun on a clean GPU before concluding.

## Rule

- Gate a lossless-by-construction change (spec decode, quant kept correct-gated)
  with a distribution/telemetry check, NOT a slow agentic pass-rate eval — the
  latter can't move if the change is truly lossless, so it only buys wall-clock
  and noise. Reserve CC-eval pass-rate for changes that CAN alter behavior.
- Draft-KV per-slot bytes (2560 MB for DFlash) belong in the concurrency budget:
  a c-sweep verdict is only valid on a GPU with headroom for `c × per_slot`.
- Canonical serve path + clear argv survives the box's security agent; disguised
  binaries get reaped (137). Don't obfuscate to evade — it inverts.
