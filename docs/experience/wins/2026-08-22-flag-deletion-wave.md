# Flag deletion wave — 10 proven A/B flags deleted, comm-backend default reverted

> Status: Accepted. `1864ddac5`, audit + edits + verify via a 6-agent workflow.
> Serve flags 66 → 56; `ARLE_*` env names 38 → 33.

## Context

A full audit of every `arle serve` flag and `ARLE_*` env var (consumer,
default, dated evidence) found 10 flags whose off-arm already has a losing
verdict, 5 env aliases duplicating another name, and one unlicensed default
flip. Flags that select between a proven path and a dead one are dead A/B
wiring; they stay in --help, get swept into scripts, and make every future
bench carry arms nobody should run.

## What Worked

Deleted (hardcoded winner, flag + runtime-flags field + all reads):

| flag | hardcoded | verdict |
|---|---|---|
| `--qwen35-decode-graph` | on | `cb6b3389d`, −58.7 % TPOT off |
| `--qwen35-batched-decode` | on | sequential off-arm killed 2026-06-29 |
| `--qwen35-gpu-router` | on | off = host routing, not graph-capturable |
| `--qwen35-fa3-decode-splits` | derive | derive = every explicit arm, 2026-08-04 |
| `--max-num-batched-tokens` | 16384 | swept 2026-08-07, does not bind |
| `--dsv4-decode-reuse` | on | +25 % c=16, 2026-07-11 |
| `--metal-pipeline` | on | only the on-arm ever verified |
| `--metal-paged-kv-read` | on | same |
| `--pool-model`, `-- extra_args` | — | reject-only stubs |

`--qwen35-decode-graph` keeps its seam field: the OPD rollout engine has its
own off-by-default `--qwen35-decode-graph` in `OpdRuntimeArgs`.
`--dsv4-flashmla-decode` was skipped (its consumer file had peer edits);
`dsv4_decode_reuse_enabled()` stays as a `true` shim until the peer's
`executor/dsv4.rs` call site lands.

**comm-backend default reverted Auto → Nccl.** `84c60dee5` (a dead-code
commit) flipped it with no bench entry, contradicting two verdicts
(2026-06-10 wall-neutral; 2026-08-17 one-shot 51–53 vs NCCL 70–80 tok/s).
Auto also activates one-shot's different FP summation order in every TP≥2
serve — a correctness surface the digit-corruption investigation never
exercised. `--comm-backend auto` stays opt-in.

**Env merges:** `ARLE_DSV4_NVTX`→`ARLE_NVTX`; `ARLE_QWEN35_QUANT_PROFILE`
folded into `ARLE_QWEN35_PROFILE`; `ARLE_DSV4_MOE_BACKEND` alias deleted
(writers in 3 scripts renamed); `ARLE_ATTN_CP_SIZE` read deleted (only
`INFER_` is ever written); `ARLE_TP_SIZE`/`ARLE_TP_RANK` aliases deleted.

**Consistency fixes:** runtime-flags statics now equal
`CudaRuntimeFlags::default()` (decode-graph, gdr-chunked were `false`,
spec-max-batch was 1 vs shipped 16). DSv4's `spec_max_batch.min(1)` confirmed
intentional (`c0d302f52`, per-slot draft) — help text now says the flag is
pinned to 1 on DSv4.

Verified: cuda-lane clippy `-D warnings`, metal `cargo check`, cpu smoke
tests (5/5). No new bench: hardcodes reproduce the shipped defaults, and the
comm-backend revert restores the state every prior bench ran under.

## Rule

A flag is a promise that both arms are viable. Once one arm has a dated
losing verdict, the flag is dead wiring — delete it and hardcode the winner.
A default flip inside a refactor commit with no bench entry is an unlicensed
flip regardless of intent.
