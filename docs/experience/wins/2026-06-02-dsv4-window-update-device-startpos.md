# DSv4 SW ring update reads start_pos from device

## Context

Goal: continue DSv4 CUDA Graph readiness by removing host scalar launch
parameters from the FlashMLA decode sliding-window cache update. The SW ring
update previously received `start_pos` as a host `int`, so a captured graph
would freeze that scalar unless the launch node was patched every decode step.

## What Worked

- Kept the existing `dsv4_update_window_cache_cuda` scalar ABI for legacy and
  non-FlashMLA paths.
- Added `dsv4_update_window_cache_start_pos_ptr_cuda`, a sibling CUDA C ABI
  that reads the one-i32 device `start_pos` slot and derives the ring position
  on stream.
- Reused the existing FlashMLA decode `fm_decode_start_pos` cache slot instead
  of adding a second start-position buffer.
- Changed the DSv4 FlashMLA decode branch to call the device-pointer ABI for
  the unfused SW ring update. SWA-only and legacy paths still use the scalar
  ABI.

This is a replay-safety tranche, not a final CUDA Graph enablement.

## Verification

- Local `cargo fmt --check`
- Local `cargo check -p infer --no-default-features --features no-cuda`
- Local `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
- Local `git diff --check`
- Remote pod worktree `/tmp/arle-dsv4-window-update-ce9a2faa`, HEAD
  `ce9a2faab00a03174d1ed7411861372cb95ab1ac`.
- Remote `cargo +stable fmt --check`
- Remote `cargo +stable check -p infer --no-default-features --features no-cuda --offline`
- Remote `CUDARC_CUDA_VERSION=12080 cargo +stable check -p infer --no-default-features --features cuda,no-cuda --offline`
- Remote targeted CUDA C compile:
  `/usr/local/cuda/bin/nvcc -c csrc/misc/dsv4_attention.cu -o /tmp/dsv4_attention_ce9a2faa.o -O3 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 --compiler-options -fPIC -Icsrc -std c++17 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math`
- Remote `nm -g /tmp/dsv4_attention_ce9a2faa.o` confirmed:
  `dsv4_update_window_cache_cuda` and
  `dsv4_update_window_cache_start_pos_ptr_cuda`.

No runtime benchmark, decode correctness result, CUDA Graph replay result, or
TPOT claim is made from this buildability tranche.

## Pending Graph Enablement

The DSv4 CUDA Graph gate remains closed. This removes another scalar launch
parameter from the FlashMLA decode body, but the runtime still needs a
graph-safe `start_pos` producer contract, replay-safe top-k metadata fill,
compressor FP8 pack metadata cleanup, output inverse-RoPE/hybrid scalar audit,
and TP/EP collective capture semantics before a real graph replay A/B is
licensed.

## Rule

When a decode kernel needs the current position, prefer a stable device scalar
over a host launch parameter. Keep the scalar ABI until all non-graph paths have
an equivalent device metadata source.
