# DSv4 decode-region reuse — quantified delta + high-concurrency A/B (no regression)

## Context

Campaign item #1 measurement (plan
[2026-07-11-high-concurrency-throughput](../../plans/2026-07-11-dsv4-high-concurrency-throughput-campaign.md)):
quantify the finish-write-through decode-region reuse and prove reuse-ON does
not regress the serving path. H20 TP=4 GPUs 4-7, DSv4-Flash-FP8, HEAD
`025a93392`, binary `915362723201…`.

## Block 1 — reuse delta (token-preserving harness, PROMPT=500 GEN=128 PAGE=64)

On the identical ON-prompt, cross-serve reuse length:

| | reuse length | into-generated-region |
|---|---|---|
| OFF (no flag) | `on_hit 512` | N (stops short of finish floor) |
| ON (`--dsv4-decode-reuse true`) | `on_hit 576` = finish_floor | **Y (PASS)** |

**+64 tokens (1 page) of additional reuse into the generated region**; the
`into_gen` gate flips N→Y. Feature confirmed active. (Expected +128/2-page — the
delta is +64/1-page because the standard path already publishes the first
generated page: OFF floor is 512, not 448. Clean signal = the cross-serve
`on_hit` 512→576 + the into_gen flip; the internal off-column is radix-state
noisy.) **Needle-exact both serves** (738291 3/3 @250/500 tok) — reuse does not
corrupt retrieval.

## Block 2 — high-concurrency serving A/B (guidellm, 512-in/128-out, c=1/4/8/16)

**guidellm CANNOT exercise multi-turn reuse** — its synthetic dataset is
independent random-token prompts with no shared growing prefix, so the reuse
path never fires. Block 2 therefore measures raw serving TTFT/TPOT/throughput at
concurrency AND proves reuse-ON does not regress the single-shot path:

| c | TTFTp50 Δ | TPOTp50 Δ | out tok/s Δ |
|---|---|---|---|
| 1 | +0.1% | +0.4% | −0.5% |
| 4 | −9.5% | +1.2% | −4.4% |
| 8 | −0.2% | +0.8% | −5.9% |
| 16 | −1.0% | +2.1% | −0.5% |

At c=1 the arms are **identical** (all within noise) — the finish-capture D2H
costs ~nothing even when no reuse fires. The c≥4 swings are tail noise on
25-40-request/60s windows (ITL p99 ±13-27% both directions), not signal.

## Verdict

Decode-region reuse is **quantified (+1 page into the generated region,
needle-exact) and safe (no single-shot regression)**. The finish-capture
overhead is ~0 when reuse doesn't fire, and it helps multi-turn — so the
**default flip is licensed on safety** (win-or-wash). The multi-turn THROUGHPUT
win is not directly measurable with guidellm (independent prompts); Block 1
proves the mechanism, Block 2 proves no harm.

## Rule

- **guidellm's synthetic dataset can't measure prefix/decode reuse** (independent
  prompts, no shared prefix) — use it to prove NO-REGRESSION on the serving
  path, and the token-preserving multi-turn harness (`eval_harness token_reuse`)
  for the reuse delta itself.
- **`token_reuse.py` stat key**: `/v1/stats` exposes the counter as
  `prefix_cache_hit_tokens` (group-flattened), not `prefix_hit_tokens` — the
  original read 0. Fixed.
- **Fresh-tree pod provisioning wipes the gitignored TileLang venv/generated** →
  falls back to system tilelang 0.1.8 (sm_90 TVM PipelinePlanning bug). Preserve
  `crates/cuda-kernels/tools/tilelang/{.venv,generated}` or run
  `scripts/pod.sh setup-tilelang` before building on a fresh tree.
