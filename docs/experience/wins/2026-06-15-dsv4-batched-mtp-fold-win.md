# DSv4 batched MTP decode — +81% @c=12, breaks per-row MTP's throughput plateau (fold commit)

## Context
Production DSv4 serves `--spec-type mtp --mtp-draft-tokens 2` at steady high
concurrency, but MTP **disabled the batched lane** → the executor looped per-row
`spec_step` (one full draft+verify+commit pipeline per slot, sequentially). The
理想态 (ckl): **batched MTP** — batch the draft+verify across N slots, combining MTP's
~2× acceptance with batched amortization (what SGLang's `frozen_kv_mtp_worker` does).
[plan](../../plans/dsv4-batched-mtp-decode.md). Gated OFF `ARLE_DSV4_BATCHED_MTP`.

## Result (pod 8×H20 TP=8/EP=8, allreduce, same binary, `--num-slots 16`)

| arm @ c=12 offered | agg decode tok/s | sustained active | per-req tok/s |
|-----|---|---|---|
| per-row MTP (control) | 42.18 | ~7 | 6.03 |
| **batched fold MTP** | **76.50** | **12** | 6.38 |

**+81% aggregate.** Decode-read coherence 4/4 both arms (France→Paris/Eiffel,
Italy→Rome/Colosseum, Canada→Ottawa, Egypt→Cairo/Giza) — identical answers, MTP
acceptance preserved (~0.75 accept/step depth=2, same as per-row).

### Correctness verified — batched fold == per-row reference, ZERO cross-slot contam
Concurrent DISTINCT-needle (each slot a unique codeword buried in a long context,
barrier-synced to one batched wave, must retrieve ITS OWN) — fold vs per-row, c=4/8,
short + long (filler=120) context:
- **cross-contam = 0 in EVERY run** (both arms) — no slot read another's needle (the
  critical batched property: no cross-slot KV/attention leakage).
- **fold own-rate == per-row own-rate** (6–7/8, varying run-to-run by ±1, e.g. `quasar`
  vs `quamar` flips arm-to-arm = MoE non-determinism, NOT a systematic gap).
- The non-OWN cases are a SHARED model word-recall limit (both arms truncate the same
  uncommon words), not a batched bug — the correct-inference gate (needle +
  self-consistency + NON-byte-identity for MoE non-det) PASSES. The earlier 6-digit
  needle "FAIL" was a test artifact (model can't recall 6 digits; both arms failed
  identically) — fixed to word codewords.

### The mechanism — per-row MTP PLATEAUS, batched SCALES (the load-bearing finding)
Per-row MTP aggregate is FLAT across offered concurrency — it cannot use more slots:

| offered c | per-row MTP agg | batched fold agg |
|---|---|---|
| 4 | 42.37 | — |
| 8 | 41.71 (avg act 6) | — |
| 12 | 42.18 (avg act 7) | **76.50 (avg act 12)** |

Per-row loops `spec_step` per slot → the decode wave grows linearly with c → throughput
ceilings ~42 tok/s regardless of offered load. Batched fold runs ONE amortized wave →
aggregate scales with c. This is why batching wins: it breaks the sequential plateau.

## What it took — the iteration arc (every step measured, §0 wall-clock)
1. **Stage 1 (sub-mode 2 ring-replay attn): −44% @c=4** — un-batched the attention
   per-row MTP already tree-batches (12 FlashMLA/layer vs 4)
   ([errors](../errors/2026-06-15-dsv4-batched-mtp-stage1-submode2-regression.md)).
2. **tree-attn per slot (sub-mode 1): −44%→−32%** — attention now matches per-row, but
   still losing.
3. **re-forward commit can't win: 30.75 vs 41.71 @c=8** — batched saves c verify-forwards
   but ADDS c re-forward commits ≈ same cost. Per-row is fast *because* of fold.
4. **batched FOLD commit (the win): 76.50** — `forward_decode_batch_verify` persists each
   slot's per-layer attn-normed chain rows into the OWNING slot's `spec_normed` (codex P2
   per-slot scatter, no cross-slot aliasing); `spec_step_batched` commits via
   `commit_accepted_fold` (the cheap path, no re-forward). The 30.75→76.50 swing is fold.
5. **cap root-caused:** default `num_slots=4` (`infer-api/src/loaded.rs:69`) — the serve
   must pass `--num-slots` to reach c≥8; without it the c=8 SLO is unreachable.

## Design (gated OFF until a default-flip license)
`spec_step_batched` (executor/spec_decode.rs): per-slot capture+draft (looped) → ONE
`forward_decode_batch_verify` (dsv4.rs: tree-attn per slot + grouped MoE over all M rows
+ per-slot `spec_normed` persist) → per-slot `commit_accepted_fold`. Allreduce transport;
default OFF → main per-row path byte-identical.

## Why absolute throughput is still low — phase profile (`ARLE_DSV4_MTP_STEP_PROFILE`)
Per-wave host-ms (profiling syncs distort throughput → attribution only, not the headline):

| | draft+capture | verify (batched) | commit (fold) | total |
|---|---|---|---|---|
| **batched fold** n=4 | 15.5 | **76.2 (69%)** | 18.5 | 110 ms |
| **batched fold** n=6 | 23.1 | **99.7 (64%)** | 32.6 | 155 ms |
| per-row, **PER SLOT** | 3.8 (cap+draft) | 33.8 | 5–10 | ~43–47 ms |

Root cause, NAILED:
- **The verify forward DOMINATES (~70% of the batched wave) and is inherently
  expensive** — DSv4 is 60 layers of MoE + a per-layer TP all-reduce; one batched
  verify is 76 ms even for 4 slots. It amortizes well (batched 76 ms/12 rows ≈ 6.3
  ms/row vs per-row 34 ms/3 rows ≈ 11 ms/row = **1.8×**), but it's the floor.
- Per-row plateaus because its per-slot step (~45 ms, verify-dominated) runs c× serially
  (c=8 → ~360 ms wave → ~39 tok/s, matching the measured ~42).
- The un-batched residual (draft+capture ~15 ms, commit ~18 ms = ~30% of the wave) is
  the smaller lever; **the verify cost is the bigger one.**

Next levers, ranked by this profile:
1. **Attack the verify (~70%)** — DP-attention (removes the Q-allgather + the 4–9×
   lockstep all-reduce skew the c-sweep measured, inside the verify attention) + CUDA
   graph (launch overhead) + lighter TP collectives. These are the c-sweep's #2/#3
   levers, now confirmed to bind the verify.
2. **Batch the draft + commit (~30%)** — `mtp_forward_level` over N slots (depth-seq)
   + a batched fold commit; moderate work, partial unlock.

## Residual
The per-slot DRAFT (`mtp_forward_level` looped) + per-slot capture/restore + per-slot
fold commit stay un-batched (~30% of the wave). The verify (dominant) is batched.

## Rule
- **A batched lane must batch the axis that ceilings throughput, AND match the commit the
  reference uses.** Per-row MTP plateaus on sequential per-slot processing AND wins its
  per-step race via cheap fold; batched must batch the wave (verify+MoE) AND fold-commit
  — re-forward commit (chosen to dodge the spec_normed scatter) erased the entire win.
- **Plateau vs scale is the right frame for a concurrency lever** — compare aggregate
  ACROSS offered c, not at one point. Per-row's flat 42-tok/s across c=4/8/12 is the
  tell; a single-c number would have hidden it ([[feedback_measure_batching_before_ceiling]]).
- **decode-read coherence at c≥4 gated every iteration** — the fold commit is a NEW path;
  4/4 coherent confirmed it before the perf number was trusted (faster-nonsense guard).
- **Verify the SLO concurrency is reachable first** ([[feedback_verify_slo_lane_runs_before_optimizing]]):
  default num_slots=4 capped decode at c=4; the win only appears once `--num-slots` lifts it.
