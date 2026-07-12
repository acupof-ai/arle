# Isolated nvcc+gnu-ld link test false-passed an ODR bug that rust-lld caught

> Verify a CUDA kernel change "links clean" only via the real
> `cargo build --release --features cuda` to the final `arle` bin (rust-lld) —
> isolated `nvcc -c` + `ar` + gnu-ld `--whole-archive` is NOT a faithful proxy.

## Context

Split the 2104-line `csrc/attention/dsv4_attention.cu` god-file into 5 `.cu`
(swa/compressor/prep/oproj/hybrid) + a shared header `dsv4_attention_common.cuh`.
Shared `__device__` helpers `dsv4_attn_block_{sum,max}` moved into the header.
Verified "compile + link clean" with an isolated per-file `nvcc -c` plus an
`ar` + gnu-ld `--whole-archive` link probe → judged clean → pushed `92ec562d9`.

## Root Cause

The header defined the two helpers as **bare `__device__` (external linkage)**.
All 5 split TUs `#include` the header → **5 strong definitions** of each symbol.
- gnu-ld `--whole-archive` (the isolated probe) **tolerates/dedups** the duplicate
  strong `__device__` symbol → RC=0, false PASS.
- The real ARLE build links the final bin with **`rust-lld`**, which **rejects**
  `duplicate symbol: dsv4_attn_block_max(float)` → build fails at the very last step.

The proxy linker disagreed with the production linker; the isolated test was
**inference dressed as evidence** (§0 SOLID). Compiling each TU proves single-TU
correctness, never whole-program link.

## Fix

`eeafdf390` — add `__forceinline__` to both header helpers (internal linkage, no
external symbol emitted per TU). Full `cargo build --release --features cuda`
(preserve venv) at `f099a17a2` then reaches the `arle`-bin rust-lld stage clean:
`NO_DUPLICATE_SYMBOL`, BUILD_EXIT=0, Finished in 1m08s.

## Rule

1. Any `__device__`/helper defined **in a header** included by >1 TU must be
   `__forceinline__` (or `static`) — bare `__device__` in a header is an ODR
   duplicate-symbol bug waiting for the real linker.
2. "Links clean" is verified **only** by the real `cargo build --release
   --features cuda` reaching the final bin's rust-lld link. Isolated
   `nvcc -c` / `ar` / gnu-ld `--whole-archive` proves compilation, not linkage —
   their duplicate-symbol tolerance differs from rust-lld's.
