# Qwen3.5 prefill timing breakdown — where the 820ms at seq_len=4096 actually goes

## Context

Follow-up to the parity diagnosis in
`2026-04-17-sglang-p99-parity-qwen35-4b.md` and
`2026-04-17-ttft-scaling-infer-vs-sglang-l4.md`. Those showed a
~100-510ms TTFT gap vs sglang without explaining **where** our 820ms
goes. This note instruments `Qwen35Model::prefill_forward` with
per-phase `std::time::Instant` timers + `self.ctx.stream.synchronize()`
to attribute wall-clock to kernel groups.

- Hardware: NVIDIA L4 24GB, CUDA 12.8, driver 580.82.07
- Model: Qwen/Qwen3.5-4B bf16
- Server: `./target/release/infer --num-slots 10 --max-seq-len 5120
  --mem-fraction-static 0.88`
- Probe: 4096-token single-request streaming completion; scheduler chunks
  into 8 × seq_len=512 via `idle_chunk=512` policy.
- Instrumentation was a temporary local patch — reverted after
  measurement because per-layer sync caused a
  `CUDA_ERROR_ILLEGAL_ADDRESS` race on chunks 3+. Per-phase syncs were
  safe; per-layer syncs were not.

## Data — per seq_len=512 chunk (2 successful chunks observed)

| phase                | chunk 1 (µs) | chunk 2 (µs) | share (chunk 2) |
|---------------------|-------------:|-------------:|----------------:|
| embedding           |           67 |         1324 | 1.2%            |
| `GdrChunkwiseScratch35::new` (alloc) |        218 |         1122 | 1.0%            |
| 32 layers (forward loop) |     86894 |        99449 | **92.9%**       |
| `compute_logits_batch` (final norm + LM head) |  5422 |         5107 | 4.8%            |
| **total**           |        93238 |       107005 | 100%            |

The 107ms per chunk × 8 chunks = 856ms — matches the 820ms TTFT floor
measured externally (within 4%).

## Findings

### 1. Per-call allocation is NOT the bottleneck (disproves P1 allocation-hoisting hypothesis)

`GdrChunkwiseScratch35::new` = **~1ms per chunk**. Audit counted 100+
allocations per forward but their aggregate is ~1ms. Even eliminating
the full allocation surface via `PrefillForwardScratch35` preallocation
would save **under 10ms across the entire 4096-token prefill** — less
than 2% of the 820ms total. Not where the time goes.

### 2. Kernel compute inside the layer loop dominates — 93% of per-chunk wall-clock

99ms of 107ms is spent inside the 32 `prefill_layer` calls. That is
where the optimization focus must be.

### 3. Chunking overhead is ~50-80ms across 8 chunks

- embed: 8 × ~1ms = 8ms
- logits: 8 × ~5ms = 40ms
- KV cache migration / scheduler cycle: ~5-10ms per chunk × 7 boundaries
  = 35-70ms

Total chunking overhead: **~80ms** across 8 chunks for a 4096-token
prefill. Roughly matches the 100ms single-request gap vs sglang from
`2026-04-17-ttft-scaling-infer-vs-sglang-l4.md`.

**Implication:** doing the 4096-token prefill **in one forward call**
(no scheduler chunking at `idle_chunk=512`) would save ~80ms. That's
~50% of the 160ms per-request TTFT gap against sglang at single-request
4096-token probe.

### 4. Per-chunk compute itself (99ms/512-tok) is reasonable

99ms / 512 tokens = 0.193 ms/tok. sglang measures 0.150 ms/tok at the
same prompt size. Ratio 1.28 — consistent with the per-token ratio
(1.24-1.26) we observed externally. The kernel work is slower than
sglang but not by a huge factor; the compounding factor is that we
re-pay setup cost (embed + logits + scheduler cycle) 8 times.

## Plan adjustment

- **Drop "allocation hoisting" as a priority.** It's ~10ms of 820ms; not
  worth the refactor risk. The earlier plan
  (`docs/plans/qwen35-single-graph-prefill.md`) listed allocation
  hoisting as the prerequisite for CUDA Graph capture — still valid,
  but as a graph-capture enabler, not as a perf win on its own.
- **Promote "prefill without chunking" as next test.** Change the
  scheduler so single-request prefill uses `idle_chunk = prompt_len`
  rather than 512. Safe if seq_len ≤ max_seq_len ≤ 5120. Prefix-cache
  partial-hit logic needs to tolerate this. Expected saving: 50-80ms on
  4096-token prefill; scales linearly with prompt length.
- **Piecewise CUDA Graph capture of linear layers remains the bigger
  lever** for peak-throughput (46% gap). The chunking fix hits TTFT;
  graph capture hits per-token cost. Distinct hypotheses.

## Reproducing the measurement

1. Temporarily add `std::time::Instant::now()` timing + a single
   `self.ctx.stream.synchronize()` after each phase — NOT per-layer
   (per-layer sync crashes with illegal memory access on chunks 3+).
2. Build `--release -p infer`, start server, run `/tmp/ttft_probe.py`
   at seq_len=4096 against `Qwen3.5-4B`.
3. `grep 'qwen35 prefill' /tmp/infer.log` → per-chunk timings.

## Artefacts

- Instrumented patch & output in `/tmp/bench-run/infer-q35-diag.log`,
  `infer-q35-diag2.log`.
- Revert: all instrumentation removed; working tree clean at commit
  3cd9d93 or later.

## Rule

Before building a large fix around a hypothesis ("per-call allocations
are the bottleneck"), add a 5-minute timer around the actual phase and
measure. The allocation-hoist hypothesis survived two docs + one Codex
brief until the first timer disproved it. **Timers before refactors.**
