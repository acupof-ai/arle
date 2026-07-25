# Recompute preemption resumes at the committed position — no duplicate stream (#156)

> Status: Shipped. Bench **measured 2026-07-25** on 4×H20 GPUs 0-3 TP=4/EP=4 at
> `d0525cb06` — no throughput regression (see §Bench).

## Context

A `Decoding` request preempted onto the plain-recompute path cleared
`generated_tokens` and restarted from the prompt; the token observer re-forwarded
every regenerated token, so a streaming client saw the generation twice — and a
*different* second copy under temperature>0 (#156, pre-existing recompute
semantics).

## What Worked

vLLM-style resume-at-committed, aligning plain recompute with the whole-slot
swap path (one canonical resume semantics):

- `reset_for_recompute` preserves `generated_tokens`; prefill targets the
  committed stream (prompt + generated) via `committed_len/tokens/slice`
  helpers; the prefill→decode transition fires at committed length, so the only
  observed token after resume is the newly sampled one.
- Admission + both recompute fallbacks radix-match against the committed
  stream — the requeue-time `publish_prefix_blocks` becomes self-serving: the
  resumed request re-attaches through its own published generated blocks.
- max_tokens/EOS accounting unchanged (`generated_tokens` never cleared).
- **Codex-review P1 fixed in the same diff**: a DSv4 finish-write-through
  sidecar could restore the full committed stream and enter `Decoding` with the
  last token's KV already materialized (decode seed re-feeds it → silent KV
  duplication). The restore clamp now holds one token back
  (`restored.min(target-1)`), same contract as the full-match trim.

Gates: new e2e `recompute_preemption_never_reemits_committed_tokens` (observer
stream has zero duplicates and equals final `generated_tokens`) +
`frontier_tail_restore_still_prefills_the_last_token`; infer-core 110/110,
clippy -D warnings clean, cuda-lane typecheck clean.

## Bench (2026-07-25, `d0525cb06`)

`run_dsv4_bench.sh` / `bench_throughput.py`, 60 s/point, seed 20260416,
max_tokens 256, GPUs 0-3 TP=4/EP=4 eager, slot line `59 slots / per_slot 338MB
/ budget 20584MB / 84736 tok` (champion `59 / 338MB / 20582MB / 83968 tok` —
same fingerprint; the +2 MB / +768 tok is `5c2931cd3`'s measured-VRAM sizing).

| c | out tok/s | Δ% vs champion | TTFT p50/p99 ms | ITL p50/p99 ms |
|---|---:|---:|---|---|
| 1 | 38.66 | −8.6% (42.3) | 1085 / 1113 | 21.9 / 41.0 |
| 4 | 74.67 | +1.9% (73.3) | 1447 / 2985 | 43.8 / 89.2 |
| 8 | 152.82 | +11.3% (137.3) | 1069 / 1204 | 47.5 / 93.2 |
| 16 | 197.51 | +16.5% (169.6) | 2238 / 2265 | 71.4 / 119.0 |

0 errors / 0 incomplete / 0 correctness_failed at every point.

**Dataset caveat — the c1 row is not comparable.** The champion's
`bench-prompts.jsonl` (repeated-filler, prefix hit_rate 0.925) no longer exists
on the pod and has no in-repo generator; the run substituted the first 20 docs
of `bench-prompts-64.jsonl` (unique docs, hit_rate 0.048→0.767). c1 is
prefill-dominated at 3.4k tok, so losing the prefix hit explains −8.6% (TTFT
442 → 1085 ms) — a dataset delta, not a scheduler regression. c4/c8/c16 are all
above champion and outside the ±3% drift band. The recompute path itself was
heavily exercised during the #160 pressure runs: 316 `KV-overflow preempt →
requeued for recompute`, zero errored or incomplete requests.

Raw: pod `/host/arle-build/bench-output/2026-07-24-b156-d0525cb0/`. **These rows
are now the anchored champion** (#180, `04e769cbf`): the substitute dataset
regenerates byte-for-byte from the repo (sha256 `e095ddf1…`), the lost one never
will, so Rule 3 re-anchored on the reproducible one.

## Rule

- Any resume path (recompute, swap restore, sidecar) must end with KV
  materialized to exactly committed−1 and the decode seed sampled from real
  logits — a fully-materialized restore double-appends the seed token's KV.
