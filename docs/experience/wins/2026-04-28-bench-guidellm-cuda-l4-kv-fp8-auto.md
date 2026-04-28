# cuda-l4 auto KV pool defaults to FP8 — c=16: 161 → 197 tok/s (+22%)

## Context

After the HBM-tier auto chunked_prefill_size landed (`9aba02ea`,
161 tok/s @ c=16, peak_active=11), `peak_kv_util` was still at 96.4%
— the KV pool was again the binding constraint. `auto` mode kept
the paged pool in BF16 by default; FP8 was only selected when an
explicit `--max-seq-len` was set AND BF16 failed the envelope check.

Both vLLM v1 and SGLang default `auto` to FP8 paged pool on L4-class
GPUs because the per-token KV bytes drop ~50% with negligible quality
regression on Qwen3 / Qwen3.5 family models. We weren't.

## What Worked

`commit <hash>` flips two knobs:

- `kv_mode_candidates(Auto)` now puts FP8 paged pool first (BF16 stays
  as fallback if FP8 dispatch is unavailable for the model arch).
  Single candidate list — no `has_explicit_max_seq_len` branch.
- `RequestedKvCacheMode::slot_sizing_format` returns `FP8E4M3` for
  `Auto` so `auto_num_slots` sizes the slot count against the format
  the runtime will actually pick (codex review P2: previously slot
  count was sized for BF16 even when FP8 was chosen, silently capping
  the slot count when `--num-slots` was omitted).

The contiguous single-request KV cache stays BF16; quantization
happens on migration into the paged pool. This is the same
"auto-fp8" path that has been used for explicit
`--kv-cache-dtype fp8` requests for weeks; the change only flips
when it's selected by default.

## Bench

`scripts/bench_guidellm.sh cuda-l4-hbm-tier-fp8-auto --fast`
(profile=concurrent rate=16, data=4096-in/256-out, max-seconds=30,
random-seed=20260416). Same HEAD, same box, server-restart A/B
against the BF16 auto-mode run from earlier today.

L4 / Qwen3-4B BF16 weights / Driver 580.82.07 / CUDA 13.0 / guidellm 0.6.0.
Server: `--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94
--cuda-graph true` (same for both runs, dtype change only).

| metric                | legacy (chunk=4096, BF16 KV) | auto-tuned chunk + BF16 KV | auto-tuned chunk + **FP8 KV** | Δ vs prior step |
|-----------------------|------------------------------:|----------------------------:|------------------------------:|----------------:|
| TokenKVPool budget    | 5.2 GB                        | 7.0 GB                      | **8.0 GB**                    | +1.0 GB         |
| runtime_workspace     | 3.6 GB                        | 1.9 GB                      | 0.9 GB                        | -1.0 GB         |
| out tok/s (headline)  | 120.16                        | 160.88                      | **197.06**                    | **+22.5%**      |
| peak_active           | 8                             | 11                          | **16** (full saturation)      | +5 slots        |
| peak_kv_util          | 98.1%                         | 96.4%                       | **69.1%**                     | KV no longer bound |
| TTFT p50 (ms)         | 6092                          | 7002                        | 9927                          | +42%            |
| ITL p50 (ms)          | 54.8                          | 61.3                        | 77.9                          | +27%            |

**peak_active = 16 saturates the slot count** for the first time at
c=16 / 4096-in: every concurrent request can fit. peak_kv_util drops
to 69.1% — the KV pool now has headroom; the bottleneck has migrated
to compute (decode rows per step). That's the right kind of
bottleneck to chase next via TileLang / GEMV kernel work.

## Compounded result vs the original baseline

- 120.16 (legacy CLI defaults, c=16) → **197.06 tok/s** = **+64%**
  total from chunked_prefill auto + FP8 auto, two commits, three
  files changed total.
- vs SGLang reference (139 tok/s @ c=16, 2026-04-26): we're now
  **+42%** ahead on this workload.

## Cross-references

- Code: this commit (1 file, +13/-7 in `main.rs`).
- Prior step (auto chunked_prefill):
  [`2026-04-28-bench-guidellm-cuda-l4-hbm-tier-auto.md`](2026-04-28-bench-guidellm-cuda-l4-hbm-tier-auto.md)
- vs SGLang head-to-head:
  [`2026-04-26-bench-guidellm-cuda-l4-vs-sglang-c1-c16.md`](2026-04-26-bench-guidellm-cuda-l4-vs-sglang-c1-c16.md)
- Raw artefacts: `bench-output/2026-04-28-cuda-l4-hbm-tier-fp8-auto/`

## Rule

**`auto` should mean "what an L4 user would pick for production,"
not "the safest legacy choice."** FP8 paged KV has been the
production default for SGLang and vLLM v1 on L4-class GPUs for
months. Defaulting to BF16 wastes 50% of the per-token KV budget
without quality benefit Qwen3 users actually consume. The flip is
one-line; the slot-sizing alignment is the real subtlety.
