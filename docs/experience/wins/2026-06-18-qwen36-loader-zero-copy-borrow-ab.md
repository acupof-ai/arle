# Qwen3.6 loader zero-copy borrow path A/B

## Goal

Remove extra host `Vec<u8>` copies in the CUDA safetensors loader without
changing device bytes. The original #101 profile showed Qwen3.6-35B-A3B BF16
startup dominated by cold shard I/O plus large host copies in the stacked MoE
loader.

## Hypothesis

Borrow tensor byte ranges from the mmap-backed shard cache for full tensors, and
allocate only when TP sharding has to materialize sliced bytes. Expected benefit:
lower transient host-copy pressure and possibly lower startup wall-clock.

## Params

- Pod: `.62` (`iv-ye8is8fbi8s6iplibbg7`), H20 GPU1.
- Build route: `scripts/dsv4_fast_build.sh`,
  `CUDARC_CUDA_VERSION=12090`, `FEATURES=cuda`, `PROFILE=release-fast`,
  prebuilt CUDA kernels.
- BF16 model: `/data01/models/Qwen3.6-35B-A3B`.
- FP8 model: `/data01/models/Qwen3.6-35B-A3B-FP8`.
- Serve flags: `--backend cuda --num-slots 8 --total-pages 16384 --page-size 16
  --kv-cache-dtype auto`.
- Cold runs: `sync; echo 3 > /proc/sys/vm/drop_caches` before each arm.
- Before commit: `92d9c21e` (parent of `bb069c52` loader zero-copy commit).
- After commit: `00bb8970` (current main, includes `bb069c52` plus train
  follow-ups).

## Results

Local gates:

```text
rustfmt --edition 2024 --check crates/infer-cuda/src/loader.rs
PASS

CUDARC_CUDA_VERSION=12090 cargo test -p infer-cuda --release \
  --no-default-features --features cuda,no-cuda shards_ --lib -- --nocapture
PASS (3 tests)

CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib
PASS
```

BF16 startup A/B:

| Arm | Cache | Ready wall | `qwen35.total` | MoE routed sum | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| before `92d9c21e` | cold | 40.342s | 39.078s | 35.228s | baseline |
| after `00bb8970` | cold | 40.336s | 39.272s | 35.293s | wash |
| before `92d9c21e` | warm | 13.094s | 11.992s | 9.497s | baseline |
| after `00bb8970` | warm | 14.095s | 13.036s | 10.443s | wash / noise |

FP8 startup observations:

| Arm | Native DeepGEMM | Ready wall | Model load | Warmup | Dispatch |
| --- | --- | ---: | ---: | ---: | --- |
| `00bb8970` | not compiled | 23.231s | 22.026s | 0.000s | fallback, `direct_fp8_grouped=false` |
| `00bb8970` | compiled | 58.476s | 30.871s | 26.011s | direct, `direct_fp8_grouped=true` |

The BF16 zero-copy borrow patch is correct and keeps the ownership boundary
clean, but it is **not** licensed as a startup-speed win. Cold startup remains
dominated by per-layer stacked MoE load (~35.3s of the ~40.3s wall). Removing the
host `Vec` copy does not reduce wall-clock because the path still performs 256 x
3 small H2D uploads per MoE layer and faults/reads the same shard bytes on the
upload path.

I also tried a Linux `madvise(MADV_WILLNEED)` read-ahead patch on the mmap shard
path. It was not landed: BF16 cold stayed 40.345s, essentially identical to
40.336s after baseline.

## Problems

- Do not claim the loader change fixed #101 startup latency. It removes a full
  tensor ownership copy but does not move wall-clock on the measured 35B BF16
  serve load.
- FP8 native startup includes the intended DeepGEMM warmup/JIT cost. The deploy
  binary must compile the native bridge for throughput, but startup reporting
  should separate model load from warmup.

## Learnings

The next real #101 lever is not safetensors parse or mmap metadata. It is the
stacked expert upload shape: reduce the 768 small uploads per BF16 MoE layer, or
add a grouped expert owner/offset pointer table so the stacked tensors can be
uploaded as large contiguous buffers without per-expert `DeviceMatrix` owners.
