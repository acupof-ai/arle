# DSv4 DeepGEMM device-count metadata default

## Goal

Finish the remaining A3 recv-side local-count D2H removal for the default
DeepEP + DeepGEMM decode path. The target is the A3 in-graph metadata gate:
single-token decode D2H <= 50, per-request wall-clock win >= 5%, and greedy
output byte-identical with `max_tokens >= 32`.

This entry covers the padded B=1 DeepGEMM local-expert path. It does not claim
the final 32K / 1.5K, c=8, qps=8 SLO is closed.

## Hypothesis

For padded decode, `total_recv_routes` is a fixed upper bound. The DeepGEMM
kernels already accept device `active_counts` and skip rows where
`row >= count`, so the runtime can pass dense all-local-expert metadata on
device and avoid cloning `local_counts` back to host. Unused compact rows are
made safe by initializing `expert_route_slot` to `-1` and skipping negative
slots in scatter.

## Params

| Item | Value |
|---|---|
| GPU | 8x H20 |
| CUDA | 12.9 toolchain, sm_90 cubins |
| Runtime | ARLE DSv4 CUDA, 8 workers, `--num-slots 1` |
| KV | FP8 |
| Request | short prompt, `max_tokens=32` |
| Backend | DeepEP-style dispatch/combine + required DeepGEMM validation mode |
| Control | `ARLE_DSV4_DEEPGEMM_DEVICE_COUNTS=0` |
| Experiment | `ARLE_DSV4_DEEPGEMM_DEVICE_COUNTS=1` |
| Build | `cargo build --release -p infer --features cuda,nccl --bin infer` |
| Local check | `CUDARC_CUDA_VERSION=12090 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda` |

Model path and host identifiers are intentionally omitted.

## Results

### Build and smoke

| Gate | Result |
|---|---|
| Local CUDA/NCCL/no-cuda typecheck | pass with existing warnings |
| Remote env-check | pass: DeepGEMM root resolved, NCCL found |
| Remote release build | pass in 6m51s |
| Default-on rebuild | pass in 6m48s |
| Default unset smoke | pass, `elapsed_s=7.4489`, 32 completion tokens |

Smoke output stayed byte-identical across off, on, and default-unset runs:

```text
4062 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
```

### Short non-nsys smoke

| Mode | Wall-clock | Delta vs off |
|---|---:|---:|
| device counts off | 7.6534 s | baseline |
| device counts on | 7.5590 s | -1.23% |
| default unset after flip | 7.4489 s | -2.67% |

The single non-nsys smoke is directionally positive but not enough by itself
for the A3 wall-clock gate. The license decision uses the warmed nsys profile
request below.

### nsys decode trace

Single profile request after warmup, filtered to `step_decode_kernel_launch`
NVTX ranges:

| Metric | Off | On | Delta |
|---|---:|---:|---:|
| Profile request wall-clock | 3.3974 s | 3.1908 s | -6.08% |
| Decode waves | 31 | 31 | 0 |
| Decode range p50 | 58.177 ms | 33.296 ms | -42.8% |
| Decode wave max | 255.517 ms | 263.843 ms | +3.3% |
| D2H memcpy activity | 10,711 calls / 1,365,180 B | 11 calls / 44 B | -99.9% calls |
| H2D memcpy activity | 47,376 calls / 1,320,956 B | 15,003 calls / 266,092 B | -68.3% calls |
| `cuMemcpyDtoHAsync_v2` runtime API | 10,706 calls, 25.129 ms per-rank-range | absent from top APIs | removed from hot frame |
| `cudaLaunchKernel_v7000` runtime API | 11.476 ms per-rank-range | 12.583 ms per-rank-range | +9.6% |

## What Worked

- Added a dense all-local-expert metadata kernel that writes active expert ids,
  offsets, and counts on device for DeepGEMM.
- The existing DeepGEMM pack, SwiGLU, and unpad kernels safely skip zero-count
  experts.
- `masked_m` now copies from device counts via D2D in the device-count path,
  removing the host masked-count construction and H2D upload.
- `dsv4_scatter_all_route_slots_cuda` now ignores negative route slots, which
  makes `total_recv_routes` safe as an upper-bound compact capacity.
- `ARLE_DSV4_DEEPGEMM_DEVICE_COUNTS` is default-on after the PASS; set it to
  `0` for the older host `local_counts` D2H path.

## Problems

- This only covers the padded B=1 DeepGEMM path. The non-small/non-padded path
  still needs a device offset scan or native DeepEP device-count mode before it
  can avoid host metadata.
- Launch/API churn remains high. `cudaLaunchKernel_v7000` increases because the
  device-count path adds small helper kernels; the D2H removal still wins the
  warmed request.
- The first decode wave cold tail remains large. Final SLO evidence still needs
  the 32K / 1.5K, c=8, qps=8 framing.

## Artifacts

- Build:
  `/sgl-workspace/bench-artifacts/dsv4-device-count-build-20260526`
- Control smoke:
  `/sgl-workspace/bench-artifacts/dsv4-device-count-off-smoke-max32-20260526`
- Device-count smoke:
  `/sgl-workspace/bench-artifacts/dsv4-device-count-on-smoke-max32-20260526`
- Control nsys:
  `/sgl-workspace/bench-artifacts/dsv4-device-count-off-nsys-max32-20260526`
- Device-count nsys:
  `/sgl-workspace/bench-artifacts/dsv4-device-count-on-nsys-max32-20260526`
- Default-on rebuild:
  `/sgl-workspace/bench-artifacts/dsv4-device-count-default-build-20260526`
- Default-unset smoke:
  `/sgl-workspace/bench-artifacts/dsv4-device-count-default-smoke-max32-20260526`

## Rule

For padded DSv4 DeepGEMM decode, do not clone recv-side `local_counts` to host.
Keep local expert metadata on device, treat fixed padded route count as
capacity, and guard invalid compact rows with a negative route-slot sentinel.
