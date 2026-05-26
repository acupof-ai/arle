# DSv4 A3 Phase 1 — device-side local-route offsets

## Goal

Execute Phase 1 from
[`docs/plans/2026-05-26-dsv4-a3-in-graph-metadata.md`](../../plans/2026-05-26-dsv4-a3-in-graph-metadata.md):
move site 1 local expert offsets from host scan + H2D upload to a device-side
scan, keeping the old path behind `DSV4_A3_PHASE1=0`.

This is not the full A3 target. Phase 1 is expected to remove Class A offsets
H2D traffic, while `cuMemcpyDtoHAsync_v2` count stays pinned until Phase 2
persistent dispatch and Phase 3 DeepEP device-count mode.

## Hypothesis

For `experts_per_rank <= 64`, a single-block i32 exclusive-scan kernel can
produce offsets on device cheaply enough to remove per-layer H2D metadata
without changing greedy output or hurting per-request wall-clock.

## Params

| Item | Value |
|---|---|
| GPU | 8x H20 |
| CUDA | 12.9 toolchain, sm_90 cubins |
| Runtime | ARLE DSv4 CUDA, 8 workers, `--num-slots 1` |
| KV | FP8 |
| Toggle | `DSV4_A3_PHASE1=0` vs `1` |
| Build | `cargo build --release -p infer --features cuda,nccl --bin infer` |
| Local check | `CUDARC_CUDA_VERSION=12090 cargo check -p infer --no-default-features --features cuda,no-cuda` |

Model path and host identifiers are intentionally omitted.

## Results

### Correctness

Greedy output was byte-identical in all A/B runs.

| Request | Phase 1 off | Phase 1 on |
|---|---|---|
| short decode, `max_tokens=2` | `4062` | `4062` |
| longseq, `max_tokens=1` | `The` | `The` |
| longseq, `max_tokens=64` | `The2 2 2 ...` | `The2 2 2 ...` |

### Wall-clock

| Request | Phase 1 off | Phase 1 on | Delta |
|---|---:|---:|---:|
| short profile HTTP, `max_tokens=2` | 0.6363 s | 0.6167 s | -3.1% |
| longseq TTFT, `max_tokens=1` | 31.3438 s | 31.3414 s | -0.01% |
| longseq total, `max_tokens=64` | 36.4854 s | 36.4439 s | -0.11% |
| nsys decode wave wall | 247.066 ms | 247.430 ms | +0.15% |

Ground truth is wall-clock/per-request. The nsys decode wave is effectively
flat; the non-nsys longseq run shows no regression.

### nsys CUDA activity

Single profile request, filtered to one `step_decode_kernel_launch` wave:

| Metric | Phase 1 off | Phase 1 on | Delta |
|---|---:|---:|---:|
| `cuMemcpyHtoDAsync_v2` runtime calls | 546 | 352 | -35.5% |
| H2D payload | 26,240 B | 1,408 B | -94.6% |
| H2D activity time | 0.432 ms | 0.278 ms | -35.7% |
| `cuMemcpyDtoHAsync_v2` runtime calls | 344 | 344 | 0% |
| D2H payload | 44,032 B | 44,032 B | 0% |

The new `dsv4_exclusive_scan_i32_kernel` is tiny in this trace:
538 launches across the capture, 0.809 ms aggregate GPU time, 0.0015 ms average.

## Problems

- Phase 1 cannot remove the 344 D2H calls. The existing expert dispatch still
  needs host counts to decide per-expert launches. Removing that requires
  Phase 2's persistent/grouped dispatch change.
- DeepEP Class B still reads host collective sizes. That remains Phase 3.
- The A3 backlog's final target (`DtoH <= 50`, wall-clock -5%+) is therefore
  not claimed here. This entry only licenses continuing past Phase 1.

## Artefacts

- HTTP short A/B:
  `dsv4-a3-phase1-http-20260526-030520`
- nsys off:
  `dsv4-a3-phase1-nsys-mode0-20260526-030633`
- nsys on:
  `dsv4-a3-phase1-nsys-mode1-20260526-030732`
- longseq A/B:
  `dsv4-a3-phase1-longseq-20260526-031102`

## Learnings

Phase 1 is a safe metadata cleanup, not a throughput unlock. It removes the
offsets H2D leg and preserves output, but the wall-clock frame says the decode
wave remains bound by allocator/API sync, D2H, NCCL, and expert GEMV. Continue
to Phase 2 only if the next variable is persistent Class A dispatch; do not
keep polishing offsets.

## Rule

Metadata moves that only shrink payload must still be judged by per-request
wall-clock. Payload reduction without D2H count reduction is not enough to
claim the A3 objective.
