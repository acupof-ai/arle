# cuda-l4 TileLang prefill HD128 causal-bound KV loop — pending-remote

> **Pending-remote stub.** Patch lands locally (clean working tree on
> Linux dev box without nvcc); bench must run on the L4 box with
> `cargo build --release --features cuda,tilelang-attn`. Target shape
> matches the parent regression diagnosis — c=16 / 4096-in / 256-out /
> Qwen3-4B / FP8 paged KV / `--num-slots 16 --max-seq-len 4608
> --mem-fraction-static 0.94 --cuda-graph true`. Use
> `scripts/bench_guidellm.sh tilelang-causal-bound` against last
> 2026-04-28 baseline at
> [`2026-04-28-bench-guidellm-cuda-l4-tilelang-tc-decode-hd128.md`](2026-04-28-bench-guidellm-cuda-l4-tilelang-tc-decode-hd128.md).

## Goal

Close the prefill HD128 perf gap that drives TileLang's -17% tok/s
regression (despite +15% ITL). Goal type: **regression-recover**.

## Hypothesis

Bounding the KV pipelined loop by causal reach
(`min(kv_total_len, kv_offset + row0 + BLOCK_M)`) skips strictly
upper-triangle KV tiles that the unbounded loop currently visits, then
masks to -inf. For 4096-in cold prefill with `BLOCK_M=64`,
`BLOCK_N=64`, the unbounded loop visits ~64 KV tiles per Q-tile
regardless of position; the bounded loop visits only the diagonal +
below tiles, averaging ~32 — a ~50% per-Q-tile FLOP cut on the cold
prefill. Expected:

- TTFT p50 -10 to -20% on c=16 / 4096-in
- out tok/s **+10 to +25%** vs the 2026-04-28 TileLang ON baseline
  (132.63 median tok/s) — recovers some or all of the -17% regression
- ITL p50 unchanged or marginally improved (decode tile rarely benefits
  from the bound — qlen=1 makes `kv_offset + BLOCK_M` ≈ `kv_total_len`)

If the bench shows ≥10% out tok/s recovery without ITL regression,
`tilelang-attn` becomes promotable to default for Qwen3-4B at c=16
admission burst. If <10%, see deferred items below — that's the signal
that single-CTA-walks-all-KV (FlashInfer's split-KV cooperative grid)
is the dominant remaining gap.

## Patch summary

`crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py` only.
Two layered patches in one bench cycle:

**Patch A — causal-bound KV loop** (commit `242d766a`):

1. Hoist `kv_offset = kv_total_len - qlen` out of the inner KV loop.
2. Compute `kv_visible_end = min(kv_total_len, kv_offset + row0 + BLOCK_M)`.
3. Replace `T.ceildiv(kv_total_len, BLOCK_N)` with
   `T.ceildiv(kv_visible_end, BLOCK_N)` in the `T.Pipelined` count.

Mirrors FlashInfer's `mask_iteration` / `window_iteration` skip pattern
in `prefill.cuh:2256-2263` (vendored at
`/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/`).

**Patch C — page-lookup hoist** (this commit):

1. Allocate three 1D fragments outside the `T.Pipelined` loop:
   `page_idx_j[BLOCK_N]`, `in_page_j[BLOCK_N]`, `valid_j[BLOCK_N]`
   (all `index_dtype = int32`).
2. Inside the loop, fill them with a `T.Parallel(BLOCK_N)` block —
   one divmod + one `KV_indices[]` gather per `j`.
3. Replace the original `T.Parallel(BLOCK_N, HEAD_DIM)` per-element
   page lookup with a fragment-read using the precomputed values.

Eliminates ~128× duplicate divmod + `KV_indices` gather per outer
KV-tile iteration (only the `(j, d)` load actually needs `d`; pages
were never `d`-dependent). Mirrors FlashInfer's `thr_local_kv_offset[]`
cache in `prefill.cuh:2192-2287`. Codex audit + FlashInfer-comparison
agent both surfaced this independently. The fragment write-then-read
pattern matches the existing `scale_i[BLOCK_M]` precompute at lines
167-172 of the same file (LayoutInferencer-tested OK).

## Cross-validation

- **Codex perf audit** (background `bp6l6vmkx`, 2026-04-28): identified
  this as item #1 of 5 with ~35-50% FLOP cut, ~10-20% TTFT win.
- **FlashInfer-comparison agent** (background `a4af54fd`, 2026-04-28):
  identified the same pattern as item #4 of 5 deltas, citing
  FlashInfer prefill kernel iteration-skip mechanism by file:line.
- **Codex bottleneck-diagnosis** (background `blxo0ydin`, 2026-04-28):
  confirmed prefill-bound at c=16/4096-in via `StepPlan::Mixed`
  scheduler arm (`infer/src/scheduler/cuda/execution.rs:33,335`,
  `decode.rs:498`) and observed-ceiling math (TileLang ON achieves
  only ~59% of pure-decode ceiling, vs ~79% for OFF — meaning ≥41%
  of wall-time is mixed-prefill-decode penalty in TileLang ON).

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4 / sm_89 / Driver 580.82.07 / CUDA 13.0
- **Commit:** `<sha after this entry lands>`
- **Feature set:**
  `cargo build --release --features cuda,tilelang-attn`
- **Non-default flags / env vars:** none beyond canonical
- **Server launch:** `scripts/start_infer.sh Qwen/Qwen3-4B 8000
  --num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94
  --cuda-graph true`

## Canonical params

- `--profile concurrent --rate 16` (matches the parent A/B; the
  4096-in/256-out shape saturates at c=16 on L4)
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- 6 runs per side (server kept up across runs to amortize cuda-graph
  capture; warm vs cold prefix-cache is the dominant variance source).

## Results — sweep headline table

_pending-remote: fill from `bench-output/2026-04-28-tilelang-causal-bound-on/`
after L4 run._

| run | OFF tok/s | ON tok/s | OFF ITL p50 | ON ITL p50 | OFF TTFT p50 | ON TTFT p50 |
|-----|----------:|---------:|------------:|-----------:|-------------:|------------:|
| r1  | …         | …        | …           | …          | …            | …           |
| r2  | …         | …        | …           | …          | …            | …           |
| r3  | …         | …        | …           | …          | …            | …           |
| r4  | …         | …        | …           | …          | …            | …           |
| r5  | …         | …        | …           | …          | …            | …           |
| r6  | …         | …        | …           | …          | …            | …           |

`OFF` = HEAD before this commit (TileLang ON, unbounded KV loop —
i.e. matches the parent baseline at
`2026-04-28-bench-guidellm-cuda-l4-tilelang-tc-decode-hd128.md` ON
column). `ON` = TileLang with the causal-bounded loop.

## Decision rule

- ≥+10% median out tok/s **and** no ITL regression worse than 5% →
  ship; promote `tilelang-attn` to default for Qwen3-4B and update
  the parent wins entry's "Decision" section.
- 0% to +10% → keep the patch (FLOP cut is real and helps long-context),
  document it but do not promote default; layer Patch C (page-lookup
  hoist) and Patch D (sm_89 BLOCK_M=128 variant) before re-evaluating.
- Negative regression → revert. Most likely cause would be
  `T.if_then_else` codegen on a scalar TIR Int producing worse
  pipelining than the static loop bound.

## Deferred follow-up patches

Same operator-roadmap thread as this one:

1. **Patch B** — split mask: full-valid path vs diagonal-tile path.
   Codex audit estimated ~3-8% on top of the bound. Held back today
   because TileLang 0.1.9 runtime `if`-statement codegen risk is
   unverified — needs an AOT smoke before stacking.
2. **Patch D** — sm_89 AOT variant with `BLOCK_M=128` for
   `avg_packed_qo_len * gqa_group ≫ 64`. FlashInfer evidence at
   `scheduler.cuh:548-549` -> `utils.cuh:384-402` shows it dispatches
   to 128 on this exact shape. Estimated ~6-12% on top.
3. **Patch E** — split-KV cooperative grid + merge-states (FlashInfer
   `kv_chunk_size`). Architectural; opens a new plan ticket. Big at
   long-ctx where `~58 SM` L4 leaves CTAs idle. Out of scope for this
   patch.

## Cross-references

- Code: this commit
  (`crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py`).
- Parent diagnosis bench:
  [`2026-04-28-bench-guidellm-cuda-l4-tilelang-tc-decode-hd128.md`](2026-04-28-bench-guidellm-cuda-l4-tilelang-tc-decode-hd128.md)
- TileLang plan:
  [`docs/plans/tilelang-integration.md`](../../plans/tilelang-integration.md)
- FlashInfer reference path: `crates/cuda-kernels/csrc/attention/flashinfer_prefill_paged.cu`,
  vendored kernel at `/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/attention/prefill.cuh`.

## Rule

**When two reviewers (codex + cross-impl agent) independently surface
the same delta, ship the patch with cross-citation.** Both reviewers
named causal-loop-bound as the dominant gap on this workload, with
matching mechanism (FlashInfer's iteration-skip pattern). When that
happens, the patch's risk is in the codegen, not the algorithm — the
algorithm is settled.
