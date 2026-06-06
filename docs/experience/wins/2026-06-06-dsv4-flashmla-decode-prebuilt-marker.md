# DSv4 FlashMLA decode prebuilt marker

## Context

Rebuilt DSv4 parity binaries sometimes failed FlashMLA decode init with
`CUDA_ERROR_NOT_SUPPORTED`, while an older production binary could run the same
FlashMLA decode path. The failure blocked validation of the official DSA indexer
on the real decode configuration.

## What Worked

The fast-build prebuilt archive validator now requires a real-shim marker symbol:
`arle_flashmla_sm90_sparse_decode_real_kernel_marker_cuda`.

The real FlashMLA decode shim exports the marker. Fallback stubs do not, so stale
or stubbed archives cannot satisfy a DSv4 FlashMLA-decode build.

Verification:

- Local `cargo fmt`.
- Local `CUDARC_CUDA_VERSION=12080 cargo check -p infer-cuda --features cuda,no-cuda`.
- Pod fast-build rejected the stale prebuilt archive because the marker was absent.
- A source rebuild exported the marker from `libkernels_cuda.a`.
- FlashMLA-decode default smoke on the pod produced `clean_tokens=[344, 34837]`.

## Rule

Pod-side DSv4 FlashMLA validation must reject stale prebuilt CUDA archives with a
symbol-level real-shim check before runtime correctness or perf claims are trusted.
