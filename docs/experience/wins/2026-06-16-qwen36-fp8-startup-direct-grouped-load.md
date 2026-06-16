# Qwen3.6 FP8 startup hang: direct grouped load fixes cold-load wall

## Context

Qwen3.6 FP8 serve stalled for minutes before binding: the visible log stopped
near tokenizer init, CPU was pegged, and host RSS climbed to model-size. Kernel
microbenches had already fixed FP8 prefill JIT warmup and fused decode, so this
was a cold-load/serve-startup blocker.

## Root Cause

Per-phase startup logging isolated two loader bugs:

| Phase | Before | Evidence |
|---|---:|---|
| quant header discovery | 15.5 s, `cached_shards=42` | `tensor_headers()` read all 42 shard data payloads just to inspect headers |
| routed expert load | ~13.6 s/layer before header fix; ~2.52 s/layer after header fix | 256 experts x gate/up/down built as transient `DeviceMatrix` values, then D2D-concatted into grouped FP8 caches |
| grouped cache build | 8-49 ms/layer before direct load | not the wall |

The tokenizer line was just the last user-visible log before model loading. The
real hang was metadata/data loading, not tokenization and not an FP8 kernel.

## Fix

- Parse safetensors metadata header-only for quant view detection: read the
  8-byte header length + JSON header, not shard payload bytes.
- Cache tensor headers behind `Rc<BTreeMap<...>>` so `quant_view_for()` does
  not clone a 64k-entry map per routed expert.
- Bound `shard_cache` to the current shard; outstanding `SharedTensor` borrows
  keep their own `Rc`, but completed layer shards no longer accumulate in host
  RSS.
- For Qwen per-expert `Fp8BlockScaled` routed MoE with native DeepGEMM ready,
  load one layer shard once and pack gate/up/down directly into resident
  grouped FP8 caches. This skips the transient 768 `DeviceMatrix` path.

## Results

Startup profile on H20, TP=1, `num_slots=1`, 6144 max sequence:

| Metric | Before | After |
|---|---:|---:|
| header discovery | 15.5 s, 42 cached shards | 0.123 s, 1 cached shard |
| routed FP8 MoE load | ~2.52 s/layer after header fix | ~0.47-0.60 s/layer |
| model load | did not bind in the failing run | 46.3 s |
| JIT warmup | already required by FP8 path | 26.0 s |
| server bind | minutes / hung | ~72 s from Qwen load start |
| host RSS after bind | ~37 GB symptom | ~0.5 GB |

4K/256 c=1 streaming e2e, stable rerun after both servers were already bound:

| Backend | Prompt / output usage | TTFT ms | ITL ms | Latency s | Output tok/s |
|---|---:|---:|---:|---:|---:|
| FP8 startup-fixed | 4095 / 256 | 1774.5 | 27.93 | 8.90 | 28.78 |
| BF16 baseline | 4095 / 256 | 1715.6 | 24.41 | 7.94 | 32.24 |
| FP8 delta | - | +3.4% | +14.4% | +12.0% | -10.7% |

The old FP8 decode e2e symptom was 71.4 ms ITL vs 23.7 ms BF16. Post fused
decode + startup fix, FP8 is no longer 3x slower in decode, but it is still not
faster than BF16 on this c=1 SLO shape. Do not claim a throughput/default win
from this run; the memory/slot license remains the valid FP8 value, while raw
decode throughput still needs a separate >=BF16 license.

## Artifacts

- Startup log: `/tmp/arle-fp8-startup-profile/fp8-8135.log`
- FP8 e2e JSON: `/tmp/arle-fp8-startup-profile/e2e-fp8-startupfixed-rerun.json`
- BF16 e2e JSON: `/tmp/arle-fp8-startup-profile/e2e-bf16-startupfixed-rerun.json`

## Rule

When a serve startup appears stuck at the last high-level log line, add
phase-level loader timing before touching kernels. For quant MoE, grouped
resident caches must be loaded directly; constructing per-expert device
matrices first is a startup and memory bug, even if forward kernels are correct.
