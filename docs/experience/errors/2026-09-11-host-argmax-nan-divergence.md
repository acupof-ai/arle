# Host argmax selected NaN, diverging from the CUDA greedy selector

## Context

S12 added a host-written numeric parity gate for the CUDA argmax kernels
(`crates/infer-cuda/examples/argmax_parity.rs`), pinning the kernel contract:
ties resolve to the lowest index, NaN is never selected, and an all-NaN or
all-`-inf` row yields index 0. The gate review cross-checked the host sampler
`infer_plan::sample::argmax_logit` (`crates/infer-plan/src/sample.rs`) against
that contract.

## Root Cause

The host selector used `f32::total_cmp` to reduce the logits row. `total_cmp`
follows IEEE total order, under which positive NaN sorts above `+inf` and
negative NaN below `-inf`. A logits row containing a NaN therefore selected a
NaN token id on the host, while the CUDA kernel in
`crates/cuda-kernels/csrc/sampling/sampling.cu` scans with strict `>`
(`if (v > local_max) …`), so every NaN comparison is false and NaN can never
win; an all-NaN row keeps the initial `(−INFINITY, 0)`. The host and device
greedy paths could return different token ids for the same logits row. NaN
rows are reachable in production: `-inf * 0` from a grammar bitmask under a
penalty (a regression test already guards the arithmetic), and any malformed
model output.

## Fix

Replace the `total_cmp` reduction with a strict-`>` scan from index 0.
Scanning in index order gives lowest-index ties for free and leaves NaN
unselected, matching the CUDA kernel and the S12 gate.
`crates/infer-plan/tests/raw_argmax_gate.rs::argmax_logit_matches_cuda_contract`
pins the contract with tie, NaN-at-zero, all-NaN, `-inf`, all-`-inf`, and
empty-row cases; it runs on the Mac under `cargo test -p infer-plan`.

## Rule

A host reference and a device kernel that claim bit-identical greedy
decoding must share the NaN comparison contract, not just the tie-break.
IEEE total order is not the same order as a strict-`>` CUDA scan: it orders
NaN above all finite values. Pin the comparison semantics with a host unit
test whenever a device kernel contract is encoded on the host.
