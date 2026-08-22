# DSv4 FP32 prefill compressor grid-parallelized — same-shell A/B +3.7%..+18.9%, needle zero-miss — CUDA, 2026-07-16

> Status: Shipped

## Goal

Recover the prefill throughput lost when the #146/#150 FP32 compressor was
extended to all compression boundaries. Root cause (source-level): the FP32
probe kernel launched `<<<1, 256>>>` — one thread block serially sweeping every
compressed block of the chunk, replacing the grid-parallel bf16 path
(`dsv4_compressor_block_kernel<<<completed, 256>>>` + finalize).

## Change (`2e635eda3`, one file, net −24 lines)

`dsv4_compressor.cu`: templated `dsv4_compressor_block_kernel` +
`dsv4_compressor_finalize_kernel` on input/carry type `T` (uint16_t | float);
the FP32 wrapper now launches the float instantiation with the same grid
decomposition as the bf16 path; finalize gains nullable bf16 mirror params
(FP32 path mirrors the prev_overlap page for decode-lane readers); serial probe
kernel deleted. FP32 carry is exact, so the parallel re-read of source tokens
is bit-identical to the serial sweep. FFI signature unchanged, no Rust changes.

## Correctness (needle gate, new binary, ship gate PASS)

Depth 0.0: 27/27 exact across 9 lengths (115–8000). Depth 0.5: 20 exact +
7 partial + **0 miss** (partials at 115/180/1000 = known mid-depth output
behavior, same pattern as the all-boundaries entry). Logs:
`/host/arle-build/needle-armB-d05.log`, `needle-armB-d00.log` (pod).

## A/B — same shell, same GPUs (0–3), same eager config, TP=4/EP=4

Arm A = serial probe (HEAD~, backup binary `arle-armA-serialprobe`); arm B =
grid-parallel (`2e635eda3`). Rates: 20-prompt `bench-prompts.jsonl` (~3352 tok,
60 s); var-c1/c32: 64-doc varied `bench-prompts-64.jsonl` (~3350 tok, 120 s,
unique prefixes). guidellm concurrent profile, seed 20260416.

| run | A TTFT p50 | B TTFT p50 |
| --- | ---: | ---: |
| rate 1 | 552 ms | 521 ms |
| rate 4 | 1545 ms | 1467 ms |
| rate 8 | 4141 ms | 3921 ms |
| rate 16 | 7145 ms | 6698 ms |
| var-c1 | 3267 ms | 3019 ms |
| var-c32 | 60985 ms | 51842 ms |

Arm A reproduces the morning fp32all table within noise (different GPU set
1–4 vs 0–3), so the baseline is stable. Raw artifacts:
`/host/arle-build/bench-output/2026-07-16-{fp32par,fp32serialA}-*/result.{json,csv}`.
Column semantics as extracted by the driver; result.json is authoritative.

## Environment

8×H20 (97.9 GB), driver 535.161.08, CUDA 12.9, DeepSeek-V4-Flash-FP8,
TP=4 EP=4 on GPUs 0–3, `arle serve --backend cuda --port 8000` (eager, no
spec-type). Build sha256 `fd568375fc06…` (`/v1/stats build_identity`); binary
gate: `dsv4_compressor_block_kernelIfE/ItE` present, `fp32_prefill_probe_kernel`
absent.

## Learnings

- **LICENSED**: stable TTFT gain at every point, needle
  zero-miss at both depths. The serial-probe cost measures ~30–40 ms per
  ~3352-tok prefill (TTFT c1 delta), i.e. ~0.4 ms per compressor call × 86
  calls — the earlier "hundreds of ms" estimate was 5–10× high, and the
  all-boundaries entry's regression was TP=8→TP=4 + workload confounded, not
  the probe's isolated cost. The gain compounds under queue pressure
  (−15% TTFT at c32) because every queued request repays the
  prefill saving.
- **num_slots is the real high-concurrency wall**: both arms log `requested
  256 slots clamped to 2 (per_slot 9618MB, slot-state 9596MB, budget
  20840MB)` — c32 saturates at 2 concurrent decodes. The slot-state is
  dominated by the per-(layer,slot) FP32 probe scratch
  (2 × width × max_seq_len × 4 B per compressor state); hoisted to a
  model-wide shared scratch in `672b8ac08` (pending its own A/B, expect
  slots 2 → ~4).
- Arm B var-c32 ITL p50 160 ms vs arm A 70 ms while completing +7 requests:
  faster prefills admit more chunked-prefill interleave per decode tick.
  Completions are the wall-clock verdict; per-token ITL alone would mislead
  here.
