# Qwen3.6 CUDA Resident Quant P2/P3 ABI Scaffold

## SLO-shape probed?

N. This tranche adds host quant-format detection plus resident `DeviceMatrix`
ABI sidecars only. No Qwen3.6 quant loader path, decode kernel dispatch, serve
path, or guidellm run is wired yet, so it makes no throughput or support claim.

## Roofline check

Deferred. No new kernel is callable in this tranche; P4/P5 will add decode
kernel dispatch, and P6/P7 will make Qwen3.6 quant checkpoints reachable. The
A100 roofline/guidellm gate remains pending for that later tranche.

## Goal

Land the generic resident weight-quant metadata and device-buffer ABI needed by
the CUDA Qwen3.6 FP8/NVFP4 plan without materializing dense BF16 weights.

## What Worked

- Added a host-only `infer-cuda::quant_format` module for E4M3/E2M1 decode LUTs,
  quant manifest parsing, sibling-tensor detection, scale-shape validation, and
  explicit DSv4 E8M0 ABI rejection.
- Extended `cuda-kernels::WeightFormat` with ABI-named
  `Fp8BlockScaled`, `Fp8PerShard`, and `Fp4E2M1Group` variants.
- Extended `DeviceMatrix` with unsigned resident quant bytes plus FP8/f32 scale
  sidecars. Offload/reload snapshots now include these buffers, so resident
  quant matrices do not collapse to the dummy BF16 `data` placeholder.

## Verification

```bash
cargo test -p infer-cuda --release --no-default-features --features no-cuda --lib
CUDARC_CUDA_VERSION=12060 cargo check -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib
```

Results:

- `infer-cuda` no-cuda lib tests: 85 passed, including 9 new quant-format tests.
- `infer-api` CUDA/no-cuda typecheck: passed.
- `cargo check -p cuda-kernels --release --no-default-features --features no-cuda`: passed.

Limitation:

- `cargo test -p cuda-kernels --features cuda,no-cuda ...` cannot link on this
  Mac because the existing CUDA test binary references native CUDA symbols while
  `no-cuda` skips nvcc. The new `WeightFormat` shape logic typechecks through
  the `infer-api` CUDA/no-cuda gate; A100 runtime tests remain pending.

## Problems

- Qwen3.6 FP8/NVFP4 checkpoints still fail closed at runtime; P6/P7 loader wiring
  and P4/P5 kernel dispatch are not implemented in this tranche.
- No A100 serve, needle, or guidellm evidence yet. Remote gate remains:
  resident bytes log with `dense_materialized_weight_bytes=0`, serve+curl,
  `scripts/needle_gate.py` for FP8 and NVFP4, then guidellm.

## Rule

Resident quant support must first preserve the storage ABI and offload buffers.
Do not route a quant checkpoint to any fallback that expands full weights to
BF16; unsupported shapes stay fail-closed until the matching kernel dispatch
lands.
