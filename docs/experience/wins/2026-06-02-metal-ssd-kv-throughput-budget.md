# Metal SSD KV Throughput Budget

## Goal

Diagnosis / ceiling: measure the local SSD and decide whether Qwen3.6 Metal KV
can live entirely on SSD during inference.

## Hypothesis

SSD bandwidth is enough for appending new KV and for one-shot prefix snapshot
load/store, but not enough to read the full active KV history from SSD on every
decode step at long context.

## Command

The benchmark used one-off Python probes on the same APFS Data volume as
`~/.cache/arle/metal_kv`. File descriptors were opened with macOS
`F_NOCACHE=48`, data buffers were pre-generated random bytes, and test files
were deleted after each pass.

Raw output:
`docs/experience/wins/assets/2026-06-02-metal-ssd-kv-throughput.json`

## Environment

- Commit: `1847e910`.
- Dirty tree note: unrelated `crates/cuda-kernels/build.rs` was already dirty
  and was not touched by this run.
- Host: macOS 26.3.1, arm64, 48 GiB unified memory.
- Volume: `/System/Volumes/Data` on `/dev/disk3s1`, 460 GiB total, 58 GiB
  available before the run.
- Test path: `~/.cache/arle/ssd_bench`.
- Model shape for math: `mlx-community/Qwen3.6-35B-A3B-4bit`.

KV bytes per token uses the Metal config formula:

```text
2 * full_attention_layers * num_key_value_heads * head_dim * dtype_bytes
= 2 * 10 * 2 * 256 * 2
= 20,480 bytes/token
```

## Results

Measured SSD:

| probe | read | write |
|---|---:|---:|
| 2 GiB x 3, median | 5.27 GiB/s | 2.77 GiB/s |
| 16 GiB, conservative | 4.05 GiB/s | 1.09 GiB/s |

The active-KV calculation below uses the conservative 16 GiB sequence read
number: 4.05 GiB/s.

| context | ARLE TPOT | tok/s | active KV per output token | read BW if full history comes from SSD each step | SSD ratio | append new-KV write | one-shot snapshot read | one-shot snapshot write |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | 12.19 ms | 82.03 | 20 MiB | 1.60 GiB/s | 0.40x | 1.60 MiB/s | 4.8 ms | 17.9 ms |
| 2k | 12.38 ms | 80.78 | 40 MiB | 3.16 GiB/s | 0.78x | 1.58 MiB/s | 9.6 ms | 35.8 ms |
| 4k | 12.40 ms | 80.65 | 80 MiB | 6.30 GiB/s | 1.55x | 1.58 MiB/s | 19.3 ms | 71.6 ms |
| 8k | 13.38 ms | 74.74 | 160 MiB | 11.68 GiB/s | 2.88x | 1.46 MiB/s | 38.6 ms | 143.2 ms |
| 12k | 14.89 ms | 67.16 | 240 MiB | 15.74 GiB/s | 3.88x | 1.31 MiB/s | 57.8 ms | 214.8 ms |

Verdict:

- Appending only the newly generated token KV to SSD is trivial: roughly
  1.3-1.6 MiB/s at the measured decode rate.
- Loading or storing a completed prefix snapshot once is viable. At 12k, pure
  KV is 240 MiB, so the conservative one-shot read is about 58 ms. Full ARLE
  snapshots can be larger than pure KV, but the bandwidth class is still useful
  for bounded prefix cache and background spill.
- Using SSD as the active KV backing store for every decode step is not viable
  past about 2k context. At 4k it already needs 6.30 GiB/s, above the measured
  conservative read ceiling; at 12k it needs 15.74 GiB/s before counting Metal
  upload, dispatch, fragmentation, or recurrent-state snapshot overhead.

## Problems

- The 2 GiB 4 MiB random-read probe reported about 10 GiB/s, which is likely
  inflated by caching effects. It was not used for the active-KV verdict.
- This was a local component ceiling probe, not a full HTTP serving benchmark.

## Learnings

SSD is a good bounded prefix-snapshot tier. It is not a substitute for keeping
active decode KV resident in unified memory unless a future design proves
chunked prefetch, overlap, and long-context serving A/B under the real model.

## Delta vs Baseline

First measured SSD KV bandwidth budget on this host.
