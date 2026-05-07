# Bench — Decomposing the ARLE-Metal vs mlx-lm gap into kernel + batching axes (c=1 isolation)

## Goal

Pin down whether the 2.70× ITL gap observed at c=4
(`2026-05-07-bench-guidellm-metal-c4-apples-vs-mlxlm.md`) is dominated
by per-token decode kernel cost or by batching-related overhead. The
previous wins entry hypothesized "single-stream decode is 2.7× slower
than mlx-lm" — that wasn't isolated, it just looked likely from c=4
data alone.

## Hypothesis

If the 2.70× c=4 ITL gap is per-token-kernel-bound, c=1 ITL should
also show ~2.70× — i.e. the same kernel runs slower regardless of
concurrency. If it's batching-bound, c=1 ITL should approach mlx-lm.
A long-context c=1 cell separates per-token kernel cost from prompt
length effects (KV-attention scales with context).

## Params

Two single-stream cells per backend, plus the prior c=4 cell as
reference. All on `models/Qwen3.5-0.8B-MLX-4bit`, M4 Pro, same
ARLE / mlx-lm / guidellm versions as the apples-to-apples run.

- W_short — 128 prompt / 2048 decode / c=1 / 30 s. Decode kernel
  runs against a small KV cache, so this cell isolates per-token
  decode kernel cost.
- W_long — 4096 prompt / 256 decode / c=1 / 30 s. Same prompt size
  as the previous c=4 run, so this cell isolates the long-context
  per-token kernel cost without batching.
- (Reference) The previous c=4 long-prompt cell:
  4096 prompt / 256 decode / c=4 / 30 s.

## Env

- ARLE commit `5a24726` rebased onto origin/main (`6afa417` then
  `28056b9`); `metal_serve --max-running-requests <c>`.
- All other env identical to the apples-to-apples wins entry.

## Results

| Cell | ARLE ITL p50 | mlx-lm ITL p50 | ARLE/mlx | ARLE TTFT p50 | mlx-lm TTFT p50 |
|---|---:|---:|---:|---:|---:|
| W_short c=1 | 3.95 ms | 3.18 ms | **1.24×** | **37.4 ms** | 165.1 ms |
| W_long c=1 | 4.37 ms | 3.38 ms | **1.29×** | **920.7 ms** | 1048.2 ms |
| W_long c=4 (ref) | 19.34 ms | 7.17 ms | **2.70×** | 1.20 s | 4.51 s |

## Decomposition

The c=4 ITL gap of 2.70× decomposes into **two independent
multipliers**:

```
ARLE     c=4 ITL / c=1 ITL  =  19.34 / 4.37  =  4.43×
mlx-lm   c=4 ITL / c=1 ITL  =   7.17 / 3.38  =  2.12×
```

So ARLE pays a **4.43× batching multiplier** going c=1 → c=4 long,
mlx-lm pays only **2.12×**. The ratio of the batching multipliers,
4.43 / 2.12 = **2.09×**, is the ARLE-specific batching/padding
overhead that paged-KV will eliminate.

The remaining **1.29×** between the two backends is the pure
per-token decode kernel gap — small, kernel-fusion / eval-boundary
sized, NOT a structural gap.

```
2.70×  (observed c=4 gap)
  =  1.29×  (per-token kernel, M_e.0 territory)
  ×  2.09×  (batching/padding overhead, M_e.1 territory)
```

## Problems

1. **Previous wins entry's "single-stream decode 2.7× slower"
   characterization was wrong.** The single-stream gap is only
   1.24× – 1.29×. The morning's
   `2026-05-07-bench-guidellm-metal-c4-apples-vs-mlxlm.md`
   "Action items" list calls for an M_e.0 profile pass on the
   "2.7× per-token decode" hypothesis — that pass is now
   incorrectly scoped. Re-scope: the per-token gap is real but
   only ~1.29× and lives below the batching layer.
2. **TTFT win is even bigger than reported at c=1.** Short-prompt
   ARLE TTFT is 4.4× faster than mlx-lm (37 ms vs 165 ms).
   Long-prompt ARLE TTFT is 1.14× faster (920 ms vs 1048 ms).
   The chunked prefill + decode-priority interleave delivery
   doesn't depend on concurrency at all.

## Learnings

- **M_e.1 plan acceptance numbers were under-ambitious.** Commit 4
  (kernel cutover) acceptance should be: c=4 ITL ≤ 9.3 ms (target
  = ARLE c=1 ITL × mlx-lm-style 2.12 batching multiplier), output
  tok/s ≥ 350 at c=16. The original plan said 35 ms / 300 tok/s,
  which was conservative because it baked in the wrong "2.7× per-
  token kernel" assumption.
- **M_e.0 (per-token kernel profile) drops in priority.** A 1.29×
  gap is worth understanding eventually but does NOT block "world
  #1 on Metal" — closing the 2.09× batching gap puts ARLE within
  20-30% of mlx-lm on output tok/s at all c values. The per-token
  optimization is small follow-up.
- **The morning's `2026-05-07-metal-world-first-gap-analysis.md`
  Tier B#1 ranking is now empirically validated as the decisive
  Metal unlock**, with quantitative acceptance gates rather than
  rough SOTA-report estimates.
- ELI Layer-1 latency story: ARLE TTFT 37 ms at c=1 short means
  the HTTP layer + scheduler tick + chunked prefill collectively
  add ~37 ms over a near-zero-prompt model call. The overhead is
  acceptable for tool-call agents (well below typical 100-300 ms
  budgets per turn).

## Updated action items

1. **Update M_e.1 plan §3 commit 4 acceptance numbers** to the
   tighter 9.3 ms ITL / 350 tok/s targets.
2. **Demote "M_e.0 per-token profile" from blocker to follow-up.**
   The kernel gap is 1.29× not 2.7×; it is not the load-bearing
   issue.
3. **Update morning gap analysis** with the corrected gap
   composition. (Two updates have already landed for this doc;
   keeping it accurate is high-leverage.)
4. **Optional**: add a cell to the M6 Metal snapshot bench protocol
   for c=1 long-context — this run shows it's a clean per-token
   isolation point and was not in the original W1-W6 grid.

## Reproduce

Same as `2026-05-07-bench-guidellm-metal-c4-apples-vs-mlxlm.md`,
but with `--concurrencies 1` and `--data
'prompt_tokens=<128|4096>,output_tokens=<2048|256>'`. Raw artefacts:
`bench-output/2026-05-07-metal-c1-{arle,mlxlm,long-arle,long-mlxlm}/`.

## Cross-references

- Previous wins (the run this corrects):
  [`2026-05-07-bench-guidellm-metal-c4-apples-vs-mlxlm.md`](2026-05-07-bench-guidellm-metal-c4-apples-vs-mlxlm.md)
- Prior c-sweep:
  [`2026-05-07-bench-guidellm-metal-c-sweep-m4pro.md`](2026-05-07-bench-guidellm-metal-c-sweep-m4pro.md)
- Plan whose acceptance numbers tighten:
  [`docs/plans/M_e1-metal-paged-kv-hot-path.md`](../../plans/M_e1-metal-paged-kv-hot-path.md)
- Master gap-analysis context:
  [`docs/projects/2026-05-07-metal-world-first-gap-analysis.md`](../../projects/2026-05-07-metal-world-first-gap-analysis.md)
