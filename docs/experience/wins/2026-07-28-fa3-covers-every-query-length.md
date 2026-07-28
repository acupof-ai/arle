# FA3 paged was gated on qlen 1 — the spec verify never got it

## Context

The 2026-07-27 champion row measured DSpark net-negative at serving
concurrency (−6.3% at c=8, −7.1% at c=16) and repriced it as "a speculation win
that was really paying for a kernel defect"
([repricing](../../research/2026-07-27-dspark-repriced-after-fa3.md)).

That comparison was rigged, by me. `full_attention_paged` selected FA3 with:

```rust
let decode = meta.seq_len == 1;
if decode && pool.format == KVFormat::BF16 && ... { /* FA3 */ }
```

A DSpark verify carries `block + 1 = 17` query rows, so `meta.seq_len == 17`
and the whole spec arm fell through to the TileLang paged **prefill** kernel.
The 2.76× reached the no-spec arm only. I benchmarked a fixed path against an
unfixed one and published the ratio.

The arithmetic said so before the code did: DSpark measured 19.31 ms/token at
E[k] ≈ 2.19, so a verify step cost ≈ 61.6 ms against a 28.64 ms decode step —
**2.15×**. Verifying 17 tokens reads exactly the same KV bytes as verifying 1;
only a few 17-row GEMMs are added. The honest ceiling is ~1.05×.

## What Worked

One predicate, not two paths. Per-request `seqlen_q` from `meta.q_offsets`,
causal (the shim demotes to non-causal at qlen 1), split-KV only below 64 query
rows — a real prefill chunk already fills the SMs. The vendored units needed
nothing: `arle_fa3_shim.cu` sizes `out_accum`/`softmax_lse` by `seqlen_q`
already, and both paged dispatches (Split true/false) are compiled.

Prefill chunks now take FA3 too, which is what makes it one predicate instead
of a widened special case.

Needle gate on the new binary (`needle_gate.py 512,4096,16384,32768 3 0.0`,
`qwen3_nonthink`, RAW): **exact=3 miss=0 DET at every length**. No-spec decode
unchanged — ITL p50 26.34 vs 26.1 ms — so the widened predicate costs the path
that already had FA3 nothing.

Bench: pending-remote.

## Problems

- **The single-request probe used for the 2.76× is unsound and is dropped.** It
  timed `for line in urllib_response`, which stamps buffer fills, not SSE
  arrivals. On one binary it reported mean 72.6 ms against its own p50 of
  26.3, where the harness measured mean 28.64 / p50 28.71 / max 32.3 over
  14208 gaps — no tail at all. The p50 survived the buffering and the mean did
  not. Only `bench_throughput.py`'s `itl_s` is quoted now.
- **`DsparkConfig.layer_types` is parsed and never honored** — every draft
  layer runs the 2048 sliding window. DFlash declares 1 of 5 layers full;
  the DSpark checkpoints declare all 5. Honoring them needs a ctx ring the
  length of the request (671 MB/slot at 32k vs 42), so windowing stays — but it
  sits directly on acceptance and is now a startup warning instead of silence.
- **No `ncu`.** The needle gate and the harness deltas are the evidence.

## Rule

**A/B arms must differ in exactly the thing under test.** A capability
predicate written for one shape (`seq_len == 1`) silently partitions the arms
when another arm has a different shape, and the result reads as a property of
the feature instead of an artifact of the gate. Before comparing two arms,
check that the code path you just changed is reached by both — the per-step
cost ratio will tell you before the profiler does.
