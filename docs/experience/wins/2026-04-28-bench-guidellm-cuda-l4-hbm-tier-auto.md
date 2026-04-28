# cuda-l4 HBM-tier auto chunked_prefill — c=16 / 4096-in / 256-out: 120 → 161 tok/s (+33.9%)

## Context

Codebase had hardcoded CLI defaults `chunked_prefill_size = 4096` and
`max_prefill_tokens = 16384` regardless of GPU tier. On L4 (~22 GiB),
the prefill activation buffer is sized by `max_prefill_tokens` (not
chunk; see `qwen3/forward.rs::scheduler_runtime_workspace_bytes`), so
the activation cost is `4 * h + 2 * q + 2 * kv + 3 * im) * tokens * 2`
≈ **1.22 GB** on Qwen3-4B at 16 384 tokens. That's ~1.7 GB the KV pool
never sees.

vLLM v1 and SGLang both pick `chunked_prefill_size` from a HBM-tier
table by default; both clamp `max_num_batched_tokens` (vLLM) /
`max_prefill_tokens` (SGLang) to the same chunk size unless explicitly
overridden. We weren't.

## What Worked

`commit 9aba02ea` adds:

- `RuntimeEnvelopeOverrides { chunked_prefill_size: Option<usize>,
  max_prefill_tokens: Option<usize> }` in `scheduler/types.rs`. `None`
  means "auto from HBM"; `Some(v)` pins.
- SGLang HBM table:
  ```
  <35 GiB → 2048   (L4)
  <60 GiB → 4096   (L40S)
  <90 GiB → 8192   (A100-80)
  ≥90 GiB → 16384  (H100, H200)
  ```
- `max_prefill_tokens` defaults to the resolved chunk so the activation
  is sized for one chunk, not the whole-step token budget.
- Resolution lives in `backend/cuda/bootstrap.rs`, called right after
  `DeviceContext::gpu_memory_info()`.
- Resolved chunk is also clamped to `max_num_batched_tokens` so a
  tightened step budget never strands long prefill rows (codex review
  P2).
- CLI args become `Option<usize>` — explicit values pin, missing →
  auto-pick.

## Bench

`scripts/bench_guidellm.sh cuda-l4-hbm-tier-auto --fast` and
`... cuda-l4-legacy-4096-16384 --fast` (profile=concurrent rate=16,
data=4096-in/256-out, max-seconds=30, random-seed=20260416). Same
HEAD (`9aba02ea`), same box, same 30s window — A/B is server-restart
controlled, not historical.

L4 / Qwen3-4B BF16 / Driver 580.82.07 / CUDA 13.0 / guidellm 0.6.0.
Server: `--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94
--cuda-graph true`, with/without explicit `--chunked-prefill-size 4096
--max-prefill-tokens 16384`.

| metric                | legacy (chunk=4096 / max-prefill=16384) | auto (chunk=2048 / max-prefill=2048) | Δ        |
|-----------------------|----------------------------------------:|-------------------------------------:|---------:|
| TokenKVPool budget    | 5.2 GB                                  | **7.0 GB**                           | **+1.8 GB** |
| runtime_workspace     | 3.6 GB                                  | 1.9 GB                               | -1.7 GB  |
| paged-pool max tokens | (similar)                               | 47 216                               |          |
| TTFT p50 (ms)         | 6092                                    | 7002                                 | +14.9%   |
| TTFT p99 (ms)         | 26 137                                  | **7481**                             | **-71%** |
| ITL p50 (ms)          | 54.8                                    | 61.3                                 | +11.9%   |
| out tok/s (headline)  | 120.16                                  | **160.88**                           | **+33.9%** |
| req/s actual          | 0.267                                   | 0.333                                | +24.7%   |
| peak_active           | 8                                       | **11**                               | **+38%** |
| peak_kv_util          | 98.1%                                   | 96.4%                                |          |

The TTFT p99 swing (26.1 s → 7.5 s) is the clearest tell that admission
was previously serialized behind a memory-bound prefill activation.
With the smaller chunk, more prefill rows clear per step; the c=16
admission burst is no longer queued behind the prior fixed activation
cost.

## Cross-references

- vs prior baseline at `af042d1c` (122.1 tok/s, peak_active=8): the
  legacy A in this run reproduced 120 tok/s, ~within noise, confirming
  the +34% delta is the change and not env drift.
  [`2026-04-27-bench-guidellm-cuda-l4-budget-fix.md`](2026-04-27-bench-guidellm-cuda-l4-budget-fix.md)
- vs SGLang head-to-head (139 tok/s @ c=16, 2026-04-26): we're now
  ahead by ~16% on this workload. Closing with operator-level work
  (TileLang prefill HD128, GEMV) is the next item.
  [`2026-04-26-bench-guidellm-cuda-l4-vs-sglang-c1-c16.md`](2026-04-26-bench-guidellm-cuda-l4-vs-sglang-c1-c16.md)
- Code: commit `9aba02ea` (4 files, +176/-8).
- Raw artefacts:
  `bench-output/2026-04-28-cuda-l4-hbm-tier-auto/` (auto)
  `bench-output/2026-04-28-cuda-l4-legacy-4096-16384/` (legacy A)

## Rule

**Bind the prefill activation to the chunk, not the step budget.** A
2× larger `max_prefill_tokens` doesn't make the planner faster — it
just inflates the activation buffer and steals from KV. SGLang and
vLLM v1 both default these together for a reason; mirror their HBM
tiering and clamp both knobs in lockstep. When `max_prefill_tokens`
is exposed independently, treat it as a tunable cap on top of the
chunk, not a placeholder set to "max admission step budget".
