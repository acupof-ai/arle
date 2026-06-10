# Qwen3.5/3.6 CUDA admission capacity: slot-arena truth (num_slots × per-slot pages)

**Date:** 2026-06-10. **Backend:** CUDA, Qwen3.5/3.6 MoE (single GPU).
**Scope:** `infer-api/src/loaded.rs` only.
**Status: pending-remote** — needs a CUDA GPU c>1 run (no local GPU).

## Context

The Qwen3.5/3.6 CUDA executor is a slot-arena model: `Qwen35SlotState::new`
eagerly allocates a contiguous full-attn K/V cache of
`total_pages × page_size` tokens per layer PER SLOT
(`infer-cuda/src/executor.rs:1484`, `qwen35.rs:117`), so true device KV
capacity is `num_slots ×` that. Admission nevertheless gated on the SHARED
`config.total_pages` — the exact "fictional pages" under-admission the same
file already documents and fixes for DSv4: at c>1 a second long request
waits for pages while a real slot arena sits free. The doc comment claiming
"total_pages IS the real GPU KV budget" for Qwen3Moe was false by a factor
of `num_slots`.

## What Worked

`cuda_admission_total_pages` gives `Qwen3Moe` its own arm
(`total_pages × page_size × num_slots`), and `build_cuda_engine` now passes
executors the CONFIGURED `total_pages` (per-slot semantics) while only the
host `CudaKvPool` receives the derived admission capacity. Dense stays
exactly `config.total_pages` (host page ids mirror the device pool 1:1 —
load-bearing for the host-mirror prefix-attach change landing alongside).
DSv4 flow byte-identical. Comment rewritten to the three truthful regimes.

Known deferred: Qwen3Moe still has no DSv4-style `kv_budget_num_slots` mem
clamp (load OOM risk at large `total_pages × num_slots`), flagged in-code.

## Verification

- Mac typecheck (`cuda,no-cuda --lib`) PASS; `cargo test -p infer-core` 33/33.
- **pending-remote:** c=4 two-long-prompt admission test (pre-change the
  second queues on fictional pages; post-change both admit) + guidellm
  sweep vs latest CUDA Qwen3.5 baseline with Δ%.

## Rule

- For slot-arena executors the admission page pool must encode
  `num_slots × per-slot tokens` so the page gate never binds before the slot
  gate; only a genuinely shared paged pool may gate on its own total.
