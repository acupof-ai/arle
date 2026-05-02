---
name: Bench must run serially on single-machine Apple Silicon
description: Never run two guidellm benches in parallel on the same Mac — GPU/memory contention invalidates both results
type: feedback
---

Benchmarks on Apple Silicon (unified memory, single GPU) MUST run serially — one bench at a time. Running two `guidellm` sweeps in parallel on the same machine contends for Metal compute and memory bandwidth, making both results invalid.

**Why:** The user caught Claude launching a DFlash bench AND a baseline bench simultaneously on the same M4 Pro. Both would have reported artificially degraded numbers.

**How to apply:** When benchmarking multiple configurations (e.g. DFlash vs baseline, Qwen3 vs Qwen3.5), always: (1) run config A, (2) wait for completion, (3) kill config A's server, (4) start config B's server, (5) run config B. Never parallelize bench runs on a single host.
