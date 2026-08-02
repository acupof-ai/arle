# FA3 for batch==1 prefill (−4%) and the driver-context thread lottery — 2026-08-02

> Status: Shipped, default path (`b0368426a`). 33K cold prefill now
> **19.5 s vs the 28.9 s 2026-08-01 baseline (−32%)** combined with the
> chunked GDR.

## Context

After the chunked GDR landed, the prefill #1 was TileLang full attention
(3.99 s / 25% of the step, 54 TFLOPS = 36% of BF16 peak). FA3 prefill was
killed 2026-07-28 — but that kill was the one-launch-per-request cost on
ragged c=8 batches (TTFT p50 12.07 → 18.23 s). A single request is a single
launch either way, so `FA3_MAX_QLEN` now admits `batch == 1` too, with
split-KV forced to 1 for long q.

Measured (H20 GPU 6, same binary, chunked GDR default-on in both arms, two
distinct 33K prompts, cold): TileLang attn **20.51/20.27 s** → FA3
**19.72/19.47 s (−4%)**; greedy-64 identical; needle 1k/4k/8k ×3 = 9/9 exact
deterministic on the combined default. The −4% is smaller than the 25% share
suggested — the wall has non-attention terms (DtoH round-trips, host gaps)
and FA3 itself is not free; the residual decomposition is future work.

## What Worked — and the real find

The A/B's first run produced arm T at 28.6 s (= chunked-GDR-off speed) with
"FlashQLA chunked GDR unavailable (stub build)" in the log — **from a binary
whose other arm served chunked fine**. Root cause: the TileLang AOT dispatch
wrapper resolves SM + module through the calling thread's **driver context**;
the engine forward thread is not guaranteed to have one bound (runtime-API
kernels don't need it), so the fq path was a per-thread lottery returning
`CUDA_ERROR_NOT_SUPPORTED`. This is also the true mechanism behind the
original 2026-08-02 serve failures that were mis-attributed to stale-binary
corruption. Fix: `bind_to_thread` in the availability probe (whose OnceLock
would otherwise cache a losing lottery ticket forever) and in the branch.

## Rule

**A `__thread`-cached, driver-context-dependent dispatch makes kernel
availability a property of which thread called first.** Any AOT wrapper that
does `cuCtxGetDevice`/`cuModuleLoadData` must be entered with the context
bound, and a cached availability probe must bind before probing — otherwise
the same binary can permanently disable a feature in one process and run it
in the next.
