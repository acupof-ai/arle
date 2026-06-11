# CUDA Clippy Gate Clean — 2026-06-11

## SLO-shape probed? N

No runtime benchmark was run. This is a lint-only cleanup for the local
`cuda,no-cuda` gate.

## Roofline check

Deferred. No runtime kernel behavior was changed.

| Op | Achieved | Peak (this HW) | % | Verdict |
|---|---:|---:|---:|---|
| CUDA lint gate | n/a | n/a | n/a | pass |

## Goal

- Make the CUDA front-door clippy gate pass with `-D warnings`.

## Command

```bash
CUDARC_CUDA_VERSION=12080 cargo clippy -p infer-api --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
cargo test -p infer-cuda --release --no-default-features --features no-cuda
CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
```

## Results

| Gate | Result |
|---|---:|
| `infer-api` clippy `cuda,no-cuda -D warnings` | pass |
| `infer-api` check `cuda,no-cuda` | pass |
| `infer-cuda` no-cuda tests | 65 pass |
| `cargo fmt --check` | pass |
| `git diff --check` | pass |

## Problems

- `cargo test -p infer-api --release --no-default-features --features cuda,no-cuda --lib`
  tried to build cudarc's test target through `nvcc`; local macOS has no nvcc.

## Learnings

- Keep the Mac CUDA gate on `cargo check` / `cargo clippy` with
  `CUDARC_CUDA_VERSION=12080`; runtime and full CUDA tests remain remote-only.
