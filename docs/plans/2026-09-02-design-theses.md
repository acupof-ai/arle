# Design theses: what the runtime demonstrates, and how to make it legible

Date: 2026-09-02 · Status: Plan, accepted · Owner: ckl

## Purpose

The repository holds 313 wins and 140 errors entries. The engineering
arguments inside them are real, and at that granularity nobody reads them. A
reader with twenty minutes (a reviewer, a hiring panel, a contributor) needs
five theses, each backed by one design note and one reproduction that runs on
a Mac in under ten minutes. This plan names the theses, the evidence each
already has, the missing piece, and the artifact that closes it.

The 1000-star track (README first screen, per-turn TTFT table, distribution)
is [`2026-08-24-roadmap.md` Goal 0](2026-08-24-roadmap.md). This plan is the
depth track behind it. The two share thesis 1.

## The five theses

| # | Thesis | Standard practice | What this runtime does | Evidence in tree | Missing |
|---|---|---|---|---|---|
| 1 | A prefix cache for hybrid models (linear attention + full attention) needs state snapshots, and the snapshots need the radix's page identity, not the slot's | Radix / paged KV treats the attention KV as the whole state; hybrid models get no prefix reuse or block-aligned reuse only | Recurrent state is snapshotted at page boundaries into a sidecar bound to the radix block; a match is capped at `prompt_len - 1` so the tail always re-prefills; the disk tier is content-keyed; restored pages keep their logical id on republish and snapshots are re-keyed onto the radix-canonical chain | [wins 2026-09-02](../experience/wins/2026-09-02-metal-prefix-restore-survives-turns.md), [wins 2026-08-26 content-keyed](../experience/wins/2026-08-26-metal-kv-disk-content-keyed-restart-cache.md), [wins 2026-08-26 DSpark + prefix](../experience/wins/2026-08-26-metal-dspark-prefix-reuse.md), [wins 2026-07-08 seed token](../experience/wins/2026-07-08-prefix-cache-wrong-seed-token-fix.md) | The note. The 2026-09-02 aliasing bug is its failure section as written |
| 2 | Speculative decoding is a correctness-preserving transform with a measurable regime, and the regime ends where batching starts paying | Accepted as an approximate speedup; the concurrency at which it stops paying is rarely published | Output equals greedy token for token; a block drafter rides the prefix cache (draft KV reset per block, target hidden re-seeded by the tail prefill); the c-sweep shows verify is free only while the GPU has idle compute, and batched DSpark over quantized KV was rejected twice on measurement | [wins 2026-06-21 Metal MTP](../experience/wins/2026-06-21-metal-qwen36-mtp-spec-decode.md), [wins 2026-07-11 27B license](../experience/wins/2026-07-11-dspark-p1-license-qwen36-27b.md), [errors 2026-07-26 serializes above c=1](../experience/errors/2026-07-26-dspark-spec-decode-serializes-and-loses-above-c1.md), [errors 2026-08-22 quant-KV verify](../experience/errors/2026-08-22-batched-dspark-quant-kv-verify-loses.md) | A script that compares greedy and speculative output token by token over N prompts |
| 3 | The backend seam is a cost contract, and two host-only traits are enough | A worker / model-runner hierarchy or a backend op registry; new features add a defaulted method the other backend silently lacks | `BackendExecutor` (submit/poll + `StepLimits` + capability accessors that default to `None`) and `KvPool`; capability traits carry zero default bodies; a family whose decode is not submit/poll shaped forks the loop (`diffusion_executor.rs`); the Metal host KV pool doubles as the CPU smoke path | [wins 2026-08-14 seam refactor 49 → 15](../experience/wins/2026-08-14-seam-cost-contract-refactor.md), [architecture.md](../architecture.md) | The note: why two traits suffice and what would justify a third |
| 4 | On a bandwidth-bound decoder, memory is the product: resident bytes and bytes read per token are the two numbers, and every feature is judged by both | Heuristic fractions; a repack that keeps its source | Marlin's repack freed its source (18.7 GB, the model had been resident twice); the DeepGEMM prefill operand is derived per call from Marlin's resident layout into scratch; the DSv4 slot solve subtracts an itemized prefill-transient working set before planning; INT8/FP8 paged KV runs on tensor cores | [wins 2026-08-20 stored twice](../experience/wins/2026-08-20-marlin-source-freed-18gb.md), [wins 2026-08-20 widen to E4M3](../experience/wins/2026-08-20-nvfp4-widen-to-e4m3-deepgemm-prefill.md), [wins 2026-08-24 prefill reserve](../experience/wins/2026-08-24-dsv4-budget-prefill-reserve.md), [plans 2026-08-22 quantized-KV unification](2026-08-22-quantized-kv-attention-unification.md) | The note. The "stored twice, violated again within hours" entry is the failure section |
| 5 | A result is licensed or killed by a matched measurement, and the gate itself needs a positive control | Run once, merge if faster | TTFT and ITL are separate SLOs and `out tok/s` is never reported; same-shell A/B, three trials minimum; correct inference is a needle-ladder envelope because MoE decode is non-deterministic; a gate arm that errors stops gating; three flag-deletion waves removed levers whose off arm had been measured | [agent-method.md](../agent-method.md), [bench-and-trace-spec.md](../bench-and-trace-spec.md), [wins 2026-08-22 flag waves 1–3](../experience/wins/2026-08-22-flag-deletion-wave.md), `docs/experience/errors/` (140 entries) | A ten-item selection from the errors corpus |

On-Policy Distillation (the teacher is the serving engine; training must see
token-exact what the behavior policy generated) is the sixth thesis and
belongs to a different field. It stays as one proof that runtime authority
matters, linked from thesis 3, and gets no note of its own in this plan.

## Note structure

Each note lives in `docs/design/`, at most 1500 words, with these sections in
this order:

1. Problem, in one paragraph, with the workload that exposes it.
2. Standard practice and the exact point where it fails on this workload.
3. The design here, to `file:line` for the two or three load-bearing decisions.
4. The failure hit on the way, linked to its errors or wins entry.
5. The number, with the matched A/B it came from.
6. What would be done differently, in one paragraph.

The README first screen gets one line under Documentation: `Design notes`.
`docs/index.md` gets a row.

## Reproductions

One per note, on `Qwen3.5-0.8B-MLX-4bit` unless stated, each under ten minutes
on an M-series Mac:

| # | Command | Passes when |
|---|---|---|
| 1 | `scripts/bench_multiturn_ttft.py --turns 12 --warmup` against `arle serve --backend metal` | Turns 2–12 median under a quarter of turn 1; `licensed_blocks` tracks the raw match on every turn |
| 2 | New: `scripts/spec_parity.py` runs N prompts greedy with and without `--draft-model`, diffs token ids | Zero mismatches; prints accept rate and decode tok/s for both arms |
| 3 | `cargo test -p infer-core --release --no-default-features --features cpu,no-cuda` (the CPU smoke path is the Metal host KV pool) | Engine tests pass with no accelerator present |
| 4 | `arle --doctor` on the 35B prints the resource-guard solve: weights, headroom, anti-swap reserve, KV budget, planned slots | The printed sum equals the measured resident bytes within the tolerance stated in the note |
| 5 | `scripts/needle_gate.py … --check` ×3 on any served model | 18/18 exact and every length deterministic; the note explains why byte identity is the wrong bar |

Items 2 and 4 are new work. Item 4 needs the doctor to print the solve;
today it prints the verdict only.

## The failure corpus

The errors directory is the least common asset in the tree. Ten entries,
chosen for a root cause that generalizes beyond this codebase, go into one
page, `docs/design/what-breaks.md`, one paragraph each: Symptom, Root cause,
Rule. Candidates, in the order they were found:

- [2026-07-08 wrong seed token on a full prefix match](../experience/wins/2026-07-08-prefix-cache-wrong-seed-token-fix.md)
- [2026-07-26 spec decode serializes above c=1](../experience/errors/2026-07-26-dspark-spec-decode-serializes-and-loses-above-c1.md)
- [2026-08-01 decode-graph flag is a no-op under paged KV](../experience/errors/2026-08-01-decode-graph-flag-is-a-noop-under-paged-kv.md)
- [2026-08-13 GDN zero-token chunk kills the engine](../experience/errors/2026-08-13-gdn-zero-token-chunk-kills-engine.md)
- [2026-08-20 Marlin stored the model twice](../experience/wins/2026-08-20-marlin-source-freed-18gb.md)
- [2026-08-22 Marlin FP4 parity against the wrong oracle](../experience/errors/2026-08-22-marlin-fp4-parity-wrong-oracle.md)
- [2026-08-22 batched DSpark over quantized KV loses, cause unknown](../experience/errors/2026-08-22-batched-dspark-quant-kv-verify-loses.md)
- [2026-08-23 NVFP4 tool calls corrupt](../experience/errors/2026-08-23-nvfp4-tool-calls-corrupt.md)
- [2026-08-24 slot budget without the prefill transient](../experience/wins/2026-08-24-dsv4-budget-prefill-reserve.md)
- [2026-09-02 restored pages minted new logical ids](../experience/wins/2026-09-02-metal-prefix-restore-survives-turns.md)

## Summary statements

Three lines, one number each, for the README Status section, talks, and
posts. Numbers are the ones already in `docs/baselines.md` and the linked
entries; the 35B row replaces the 0.8B row when Goal 0 lands it.

- Prefix cache for hybrid recurrent + attention models: page-boundary state
  snapshots bound to radix blocks, content-keyed disk tier. Per-turn TTFT
  2.0 s → 180 ms on an M4 Pro (Qwen3.5-0.8B, 12 turns), restored turns equal
  cold prefill token for token.
- Speculative decoding with output equal to greedy, block drafter riding the
  prefix cache: Qwen3.6-27B decode +44% past the 15.2 tok/s bandwidth ceiling
  on Metal; the concurrency at which speculation stops paying is measured and
  published (c ≥ 4 on H20).
- Pure-Rust runtime with a two-trait backend seam over CUDA (FlashMLA,
  DeepGEMM, DeepEP, TP=8) and Metal (MLX): DeepSeek-V4-Flash 4-bit (167 GB) on
  2×H20; Qwen3.6-27B decode 2.8% faster than SGLang on the same kernel.

## Out of the narrative

Frozen, labeled experimental in `support-matrix.md`, absent from README,
landing, and the five notes: Vulkan, HIP, GLM-5.2, DeepSeek-OCR, agent REPL
features beyond what `arle serve` needs. Code stays; each one dilutes a
thesis above.

## Order and exit

1. Note 1 (hybrid prefix cache). Newest material, the failure section is
   already written, and it is the same fact as the README's headline table.
   Its structure becomes the template for the rest.
2. Note 4 (memory), then 2 (speculation) with `spec_parity.py`.
3. Note 3 (seam), note 5 (method) with `what-breaks.md`.
4. README Documentation line, `docs/index.md` row, freeze list applied to
   `support-matrix.md`.

Exit: five notes, two new scripts, one failure page, README line. Each note
carries a CHANGELOG line on the day it lands (phase exit class).
